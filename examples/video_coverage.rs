//! Label-free video sampling benchmark: dev.sh run --release --example video_coverage -- <dir> [limit]
//!
//! Each video is decoded at up to one keyframe per second as a reference and embedded with the
//! default image model. Candidate sampling policies are then simulated against that reference:
//! a policy's coverage is the mean, over reference frames, of the best cosine similarity to any
//! frame the policy keeps. Decode time is measured for the old and new persisted defaults.
//! Only aggregates are printed; file paths never appear in the output.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use nicegal_core::embedding::{ImageEmbedder, ImageEmbedderOptions};
use nicegal_core::runtime::{self, ExecutionProvider};
use nicegal_core::video::{self, SamplePlacement, VideoSamplingOptions};

const EXTENSIONS: &[&str] = &[
    "mp4", "m4v", "mov", "mkv", "webm", "avi", "mpg", "mpeg", "ts", "m2ts",
];
/// Reference density; long videos are capped so one file cannot dominate the run.
const REFERENCE: SamplePlacement = SamplePlacement::Spread {
    interval_ms: 1_000,
    min: 1,
    max: 180,
};
/// Measured on a real library at JPEG quality 85: 128/256/512/1024 buckets.
const BUCKET_KB: [f64; 4] = [3.4, 9.3, 26.5, 79.2];

struct Reference {
    duration_ms: u64,
    frames: Vec<(i64, Vec<f32>)>,
}

struct Decoded {
    duration_ms: u64,
    frames: Vec<(i64, image::RgbImage)>,
    old_decode: Duration,
    new_decode: Duration,
    old_frames: usize,
    new_frames: usize,
}

struct Policy {
    name: String,
    placement: SamplePlacement,
    threshold: Option<f32>,
    small_extras: bool,
}

#[derive(Default)]
struct Totals {
    videos: usize,
    coverage: Vec<f32>,
    worst_frame: Vec<f32>,
    kept: usize,
    storage_kb: f64,
}

fn main() -> Result<()> {
    let root = std::env::args()
        .nth(1)
        .context("usage: video_coverage <video-directory> [limit]")?;
    let limit: usize = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(usize::MAX);
    if let Some(path) = std::env::var_os("NICEGAL_VALIDATION_ORT") {
        runtime::initialize_from_dylib(std::path::Path::new(&path))?;
    } else {
        runtime::initialize_bundled_runtime(ExecutionProvider::Cpu)?;
    }
    let mut files = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| Utf8PathBuf::try_from(entry.into_path()).ok())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        })
        .collect::<Vec<_>>();
    files.sort();
    // Deterministic, evenly spread subset when a limit is given.
    if files.len() > limit {
        let step = files.len() as f64 / limit as f64;
        files = (0..limit)
            .map(|index| files[(index as f64 * step) as usize].clone())
            .collect();
    }
    eprintln!("decoding {} videos", files.len());

    let embedder = ImageEmbedder::load(&ImageEmbedderOptions::default())?;
    eprintln!("model: {}", embedder.model());
    let cursor = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let (send, receive) = crossbeam_channel::bounded::<Decoded>(8);
    let mut references = Vec::new();
    let (mut old_decode, mut new_decode) = (Duration::ZERO, Duration::ZERO);
    let (mut old_frames, mut new_frames) = (0, 0);
    let started = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        for _ in 0..4 {
            let send = send.clone();
            let (files, cursor, failed) = (&files, &cursor, &failed);
            scope.spawn(move || {
                while let Some(path) = files.get(cursor.fetch_add(1, Ordering::Relaxed)) {
                    match decode(path) {
                        Ok(decoded) => {
                            if send.send(decoded).is_err() {
                                break;
                            }
                        }
                        Err(_) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
        drop(send);
        for decoded in receive {
            old_decode += decoded.old_decode;
            new_decode += decoded.new_decode;
            old_frames += decoded.old_frames;
            new_frames += decoded.new_frames;
            let mut vectors = Vec::with_capacity(decoded.frames.len());
            let pixels = decoded
                .frames
                .iter()
                .map(|(_, image)| embedder.preprocess_image(image.clone()))
                .collect::<Result<Vec<_>>>()?;
            let mut pixels = pixels.into_iter();
            loop {
                let chunk = pixels
                    .by_ref()
                    .take(embedder.max_batch_size())
                    .collect::<Vec<_>>();
                if chunk.is_empty() {
                    break;
                }
                vectors.extend(embedder.embed_preprocessed_images(chunk)?);
            }
            references.push(Reference {
                duration_ms: decoded.duration_ms,
                frames: decoded
                    .frames
                    .iter()
                    .map(|(timestamp, _)| *timestamp)
                    .zip(vectors)
                    .collect(),
            });
            if references.len() % 50 == 0 {
                eprintln!("  {} embedded", references.len());
            }
        }
        Ok(())
    })?;
    let videos = references.len().max(1) as f64;
    println!(
        "videos {} (failed {}), wall {:.0}s, mean reference frames {:.1}",
        references.len(),
        failed.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64(),
        references.iter().map(|r| r.frames.len()).sum::<usize>() as f64 / videos
    );
    println!(
        "real decode per video at {}px: old 10/50/90 {:.0} ms ({:.2} frames), new default {:.0} ms ({:.2} frames before dedup)",
        video::SAMPLE_MAX_EDGE,
        old_decode.as_secs_f64() * 1000.0 / videos,
        old_frames as f64 / videos,
        new_decode.as_secs_f64() * 1000.0 / videos,
        new_frames as f64 / videos,
    );
    report(&references);
    Ok(())
}

fn decode(path: &camino::Utf8Path) -> Result<Decoded> {
    let metadata = video::probe(path)?;
    let duration_ms = metadata.duration_ms.unwrap_or_default();
    let timed = |placement: SamplePlacement| -> Result<(Duration, usize)> {
        let start = Instant::now();
        let samples = video::samples_with_options(
            path,
            VideoSamplingOptions {
                placement,
                ..VideoSamplingOptions::default()
            },
            || false,
        )?;
        Ok((start.elapsed(), samples.len()))
    };
    let (old_decode, old_frames) = timed(SamplePlacement::Percentages(vec![10, 50, 90]))?;
    let (new_decode, new_frames) = timed(VideoSamplingOptions::default().placement)?;
    let frames = video::samples_with_options(
        path,
        VideoSamplingOptions {
            max_edge: 256,
            placement: REFERENCE,
            ..VideoSamplingOptions::default()
        },
        || false,
    )?
    .into_iter()
    .map(|sample| {
        let (width, height) = (sample.raster.width(), sample.raster.height());
        image::RgbImage::from_raw(width, height, sample.raster.pixels().to_vec())
            .map(|image| (sample.timestamp_ms, image))
            .context("reference frame shape")
    })
    .collect::<Result<Vec<_>>>()?;
    Ok(Decoded {
        duration_ms,
        frames,
        old_decode,
        new_decode,
        old_frames,
        new_frames,
    })
}

fn policies() -> Vec<Policy> {
    let mut policies = vec![Policy {
        name: "old 10/50/90, all sizes".into(),
        placement: SamplePlacement::Percentages(vec![10, 50, 90]),
        threshold: None,
        small_extras: false,
    }];
    for (interval_ms, min, max) in [
        (10_000, 3, 8),
        (5_000, 3, 8),
        (5_000, 3, 12),
        (5_000, 3, 20),
        (3_000, 3, 12),
        (2_500, 4, 20),
    ] {
        for threshold in [None, Some(0.97), Some(0.95), Some(0.92), Some(0.90)] {
            policies.push(Policy {
                name: format!(
                    "{:>4.1}s {min}..{max:<2} dedup {}",
                    interval_ms as f64 / 1000.0,
                    threshold.map_or("none".into(), |t| format!("{t:.2}"))
                ),
                placement: SamplePlacement::Spread {
                    interval_ms,
                    min,
                    max,
                },
                threshold,
                small_extras: true,
            });
        }
    }
    policies
}

fn report(references: &[Reference]) {
    let classes: [(&str, u64, u64); 4] = [
        ("all", 0, u64::MAX),
        ("<15s", 0, 15_000),
        ("15-90s", 15_000, 90_000),
        (">90s", 90_000, u64::MAX),
    ];
    for (label, low, high) in classes {
        let selected = references
            .iter()
            .filter(|r| (low..high).contains(&r.duration_ms))
            .collect::<Vec<_>>();
        if selected.is_empty() {
            continue;
        }
        println!("\n== {label}: {} videos", selected.len());
        println!(
            "{:<32} {:>7} {:>7} {:>7} {:>7} {:>8}",
            "policy", "frames", "cover", "p10", "worst", "KB/video"
        );
        for policy in policies() {
            let mut totals = Totals::default();
            for reference in &selected {
                evaluate(&policy, reference, &mut totals);
            }
            let n = totals.videos.max(1) as f64;
            println!(
                "{:<32} {:>7.2} {:>7.4} {:>7.4} {:>7.4} {:>8.0}",
                policy.name,
                totals.kept as f64 / n,
                mean(&totals.coverage),
                percentile(&mut totals.coverage.clone(), 0.10),
                mean(&totals.worst_frame),
                totals.storage_kb / n,
            );
        }
    }
}

fn evaluate(policy: &Policy, reference: &Reference, totals: &mut Totals) {
    let frames = &reference.frames;
    // Seeks land on the last keyframe at or before a target; the reference holds those frames.
    let mut chosen: Vec<usize> = policy
        .placement
        .targets(Some(reference.duration_ms))
        .into_iter()
        .map(|target| {
            frames
                .iter()
                .rposition(|(timestamp, _)| *timestamp <= target)
                .unwrap_or(0)
        })
        .collect();
    if chosen.is_empty() {
        chosen.push(0);
    }
    chosen.dedup();
    let mut kept: Vec<usize> = Vec::new();
    for index in chosen {
        let distinct = policy.threshold.is_none_or(|threshold| {
            kept.iter()
                .all(|old| cosine(&frames[*old].1, &frames[index].1) < threshold)
        });
        if distinct {
            kept.push(index);
        }
    }
    let best = frames
        .iter()
        .map(|(_, vector)| {
            kept.iter()
                .map(|index| cosine(&frames[*index].1, vector))
                .fold(f32::MIN, f32::max)
        })
        .collect::<Vec<_>>();
    totals.videos += 1;
    totals.coverage.push(mean(&best));
    totals
        .worst_frame
        .push(best.iter().copied().fold(f32::MAX, f32::min));
    totals.kept += kept.len();
    let full: f64 = BUCKET_KB.iter().sum();
    let small: f64 = BUCKET_KB[..3].iter().sum();
    let extra = if policy.small_extras { small } else { full };
    totals.storage_kb += full + extra * (kept.len() - 1) as f64;
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut aa, mut bb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    let norm = (aa * bb).sqrt();
    if norm > 0.0 { dot / norm } else { 0.0 }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn percentile(values: &mut [f32], fraction: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}
