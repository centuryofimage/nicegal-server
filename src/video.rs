//! Bounded, in-process video work. FFmpeg contexts never leave their worker thread.
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf};
use crossbeam_channel::{RecvTimeoutError, Sender, bounded};

use crate::imaging::Raster;

pub const SAMPLING_VERSION: u32 = 3;
pub const SAMPLE_MAX_EDGE: u32 = 1024;

/// Frame selection and decoded output size. Persisted callers use the default
/// positions; changing them requires a new `SAMPLING_VERSION` for cache currency.
#[derive(Debug, Clone)]
pub struct VideoSamplingOptions {
    pub max_edge: u32,
    pub seek_percentages: Vec<u8>,
    /// Include the first decoded frame as the gallery poster and a search sample.
    pub first_frame_poster: bool,
}

impl Default for VideoSamplingOptions {
    fn default() -> Self {
        Self {
            max_edge: SAMPLE_MAX_EDGE,
            seek_percentages: vec![10, 50, 90],
            first_frame_poster: false,
        }
    }
}

/// Encoding for video-derived thumbnails. Persisted callers use the default;
/// changing it requires new sampling and thumbnail generator versions.
#[derive(Debug, Clone, Copy)]
pub struct VideoOutputOptions {
    pub encoding: crate::thumbs::ThumbnailEncoding,
    pub jpeg_quality: f32,
}

impl Default for VideoOutputOptions {
    fn default() -> Self {
        Self {
            encoding: crate::thumbs::ThumbnailEncoding::Jpeg,
            jpeg_quality: 85.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub duration_ms: Option<u64>,
    pub frame_count: Option<u32>,
    pub codec: String,
    pub frame_rate: Option<f64>,
    pub bit_rate: Option<u64>,
    pub creation_time: Option<String>,
    pub title: Option<String>,
    pub is_hdr: bool,
}

#[derive(Debug, Clone)]
pub struct VideoSample {
    /// Actual presentation timestamp relative to the video's start.
    pub timestamp_ms: i64,
    pub raster: Raster,
}

enum Work {
    Probe,
    Samples { options: VideoSamplingOptions },
}
enum Output {
    Metadata(VideoMetadata),
    Samples(Vec<VideoSample>),
}
struct Request {
    path: Utf8PathBuf,
    work: Work,
    cancelled: Arc<AtomicBool>,
    reply: Sender<Result<Output>>,
}

fn pool() -> Result<&'static Sender<Request>> {
    static POOL: OnceLock<Result<Sender<Request>, String>> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .saturating_sub(1)
            .clamp(1, 4);
        let (tx, rx) = bounded::<Request>(workers);
        for index in 0..workers {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("video-decode-{index}"))
                .spawn(move || {
                    while let Ok(request) = rx.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let mut reader = crate::imaging::video::VideoReader::open(
                                &request.path,
                                request.cancelled,
                            )?;
                            match request.work {
                                Work::Probe => Ok(Output::Metadata(reader.metadata().clone())),
                                Work::Samples { options } => {
                                    reader.samples(&options).map(Output::Samples)
                                }
                            }
                        }))
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("video worker panicked")));
                        let _ = request.reply.send(result);
                    }
                })
                .map_err(|error| error.to_string())?;
        }
        Ok(tx)
    })
    .as_ref()
    .map_err(|error| anyhow::anyhow!("starting video workers: {error}"))
}

fn execute(path: &Path, work: Work, is_cancelled: impl Fn() -> bool) -> Result<Output> {
    let cancelled = Arc::new(AtomicBool::new(false));
    struct CancelOnDrop(Arc<AtomicBool>);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    let _guard = CancelOnDrop(Arc::clone(&cancelled));
    let (reply, receive) = bounded(1);
    let mut request = Request {
        path: path.to_owned(),
        work,
        cancelled,
        reply,
    };
    loop {
        if is_cancelled() {
            bail!("video work cancelled");
        }
        match pool()?.send_timeout(request, Duration::from_millis(50)) {
            Ok(()) => break,
            Err(crossbeam_channel::SendTimeoutError::Timeout(value)) => request = value,
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                bail!("video workers stopped")
            }
        }
    }
    loop {
        if is_cancelled() {
            bail!("video work cancelled");
        }
        match receive.recv_timeout(Duration::from_millis(50)) {
            Ok(result) => return result.with_context(|| format!("processing video: {path}")),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => bail!("video worker stopped"),
        }
    }
}

pub fn probe(path: &Path) -> Result<VideoMetadata> {
    match execute(path, Work::Probe, || false)? {
        Output::Metadata(metadata) => Ok(metadata),
        _ => unreachable!(),
    }
}

pub fn samples(
    path: &Path,
    max_edge: u32,
    is_cancelled: impl Fn() -> bool,
) -> Result<Vec<VideoSample>> {
    samples_with_options(
        path,
        VideoSamplingOptions {
            max_edge,
            ..VideoSamplingOptions::default()
        },
        is_cancelled,
    )
}

pub fn samples_with_options(
    path: &Path,
    options: VideoSamplingOptions,
    is_cancelled: impl Fn() -> bool,
) -> Result<Vec<VideoSample>> {
    if options.max_edge == 0 || options.max_edge > SAMPLE_MAX_EDGE {
        bail!("video sample edge must be between 1 and {SAMPLE_MAX_EDGE}");
    }
    if options
        .seek_percentages
        .iter()
        .any(|percent| *percent > 100)
    {
        bail!("video seek percentages must be between 0 and 100");
    }
    match execute(path, Work::Samples { options }, is_cancelled)? {
        Output::Samples(samples) => Ok(samples),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Utf8PathBuf {
        Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/video")
            .join(name)
    }

    #[test]
    fn samples_real_h264_keyframes_and_container_metadata() -> Result<()> {
        let path = fixture("keyframes.mp4");
        let metadata = probe(&path)?;
        assert_eq!((metadata.width, metadata.height), (128, 96));
        assert_eq!(metadata.duration_ms, Some(6000));
        assert!(
            metadata
                .creation_time
                .as_deref()
                .unwrap_or_default()
                .starts_with("2020-01-02T03:04:05")
        );
        let frames = samples(&path, 64, || false)?;
        assert_eq!(
            frames
                .iter()
                .map(|frame| frame.timestamp_ms)
                .collect::<Vec<_>>(),
            vec![0, 2000, 4000]
        );
        for frame in &frames {
            assert_eq!((frame.raster.width(), frame.raster.height()), (64, 48));
            assert_eq!(frame.raster.pixels().len(), 64 * 48 * 3);
        }
        assert_ne!(frames[0].raster.pixels(), frames[1].raster.pixels());
        Ok(())
    }

    #[test]
    fn options_can_use_first_frame_without_seeking_or_select_a_poster() -> Result<()> {
        let path = fixture("keyframes.mp4");
        let first = samples_with_options(
            &path,
            VideoSamplingOptions {
                max_edge: 64,
                seek_percentages: Vec::new(),
                first_frame_poster: false,
            },
            || false,
        )?;
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].timestamp_ms, 0);
        let selected = samples_with_options(
            &path,
            VideoSamplingOptions {
                max_edge: 64,
                seek_percentages: vec![50],
                first_frame_poster: true,
            },
            || false,
        )?;
        assert_eq!(
            selected
                .iter()
                .map(|sample| sample.timestamp_ms)
                .collect::<Vec<_>>(),
            vec![0, 2000]
        );
        assert_eq!(
            (selected[0].raster.width(), selected[0].raster.height()),
            (64, 48)
        );
        Ok(())
    }

    #[test]
    fn rotation_and_software_hevc10_work_without_filters() -> Result<()> {
        let metadata = probe(&fixture("rotated.mp4"))?;
        assert_eq!((metadata.width, metadata.height), (96, 128));
        let rotated = samples(&fixture("rotated.mp4"), 64, || false)?;
        assert_eq!(
            (rotated[0].raster.width(), rotated[0].raster.height()),
            (48, 64)
        );
        let base = samples(&fixture("keyframes.mp4"), 64, || false)?;
        assert_eq!(
            rotated[0].raster,
            base[0]
                .raster
                .clone()
                .orient(crate::imaging::ExifOrientation::Rotate270Clockwise)
        );
        assert!(probe(&fixture("hevc10.mp4"))?.is_hdr);
        let hdr = samples(&fixture("hevc10.mp4"), 64, || false)?;
        assert!(!hdr.is_empty());
        let pixels = hdr[0].raster.pixels();
        assert!(pixels.iter().min() < pixels.iter().max());
        assert!(pixels.iter().copied().max().unwrap_or_default() < 255);
        Ok(())
    }

    #[test]
    fn keeps_decoded_frames_when_container_outlasts_video_track() -> Result<()> {
        let path = fixture("short-video-long-container.mkv");
        let metadata = probe(&path)?;
        assert!(metadata.duration_ms.unwrap_or_default() > 7000);
        let frames = samples(&path, 64, || false)?;
        assert_eq!(
            frames
                .iter()
                .map(|frame| frame.timestamp_ms)
                .collect::<Vec<_>>(),
            vec![0, 2000]
        );
        assert_eq!(
            (frames[0].raster.width(), frames[0].raster.height()),
            (64, 48)
        );
        Ok(())
    }

    #[test]
    fn cancellation_bad_input_and_unicode_paths_do_not_poison_workers() -> Result<()> {
        let temp = tempfile::TempDir::new()?;
        let path = Utf8PathBuf::try_from(temp.path().join("视频 🎞.mp4"))?;
        std::fs::copy(fixture("keyframes.mp4"), &path)?;
        assert!(samples(&path, 64, || true).is_err());
        assert_eq!(samples(&path, 64, || false)?.len(), 3);
        std::fs::write(&path, b"not a movie")?;
        assert!(samples(&path, 64, || false).is_err());
        assert_eq!(samples(&fixture("keyframes.mp4"), 64, || false)?.len(), 3);
        Ok(())
    }
}
