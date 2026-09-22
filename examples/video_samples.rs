//! Read-only corpus check: dev.cmd run --example video_samples -- testdata/video
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use nicegal_core::video;

fn main() -> Result<()> {
    let root = std::env::args()
        .nth(1)
        .context("usage: video_samples <video-directory>")?;
    let files = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| Utf8PathBuf::try_from(entry.into_path()))
        .collect::<Result<Vec<_>, _>>()?;
    let cursor = AtomicUsize::new(0);
    let (send, receive) = crossbeam_channel::unbounded();
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let send = send.clone();
            let files = &files;
            let cursor = &cursor;
            scope.spawn(move || {
                while let Some(path) = files.get(cursor.fetch_add(1, Ordering::Relaxed)) {
                    let start = Instant::now();
                    let result = video::samples(path, video::SAMPLE_MAX_EDGE, || false);
                    let value = match result {
                        Ok(samples) => serde_json::json!({"path":path.as_str(), "ms": start.elapsed().as_millis(),
                            "frames":samples.iter().map(|frame| serde_json::json!({"timestampMs":frame.timestamp_ms,
                            "width":frame.raster.width(),"height":frame.raster.height()})).collect::<Vec<_>>() }),
                        Err(error) => serde_json::json!({"path":path.as_str(),"ms":start.elapsed().as_millis(),"error":format!("{error:#}")}),
                    };
                    let _ = send.send(value);
                }
            });
        }
    });
    drop(send);
    let results = receive.into_iter().collect::<Vec<_>>();
    let failed = results
        .iter()
        .filter(|result| result.get("error").is_some())
        .count();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"files":files.len(), "failed":failed,
        "elapsedMs":started.elapsed().as_millis(),"results":results}))?
    );
    Ok(())
}
