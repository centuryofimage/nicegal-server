use std::fs::File;
use std::io::BufReader;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use camino::Utf8Path as Path;
use nom_exif::{ExifTag, read_exif};

use super::{Asset, MediaKind};
use crate::imaging::{ExifOrientation, orientation_from_exif};

// Creation time is filesystem metadata, not media-probe output. Keeping this version unchanged
// lets version-3 catalogs backfill it with a narrow update instead of re-reading every image.
pub(super) const METADATA_VERSION: i32 = 2;
// Older video rows were cataloged before the video probe supplied display dimensions. Refresh
// only videos on the next scan; reprobing the entire image library would be unnecessarily costly.
pub(super) const VIDEO_METADATA_VERSION: i32 = 3;

pub(super) fn metadata_version_for(kind: MediaKind) -> i32 {
    match kind {
        MediaKind::Image => METADATA_VERSION,
        MediaKind::Video => VIDEO_METADATA_VERSION,
    }
}

#[derive(Debug)]
pub(super) struct MediaProbe {
    pub(super) kind: MediaKind,
    pub(super) format: String,
    pub(super) exif_taken_ns: Option<i64>,
    pub(super) width: Option<u32>,
    pub(super) height: Option<u32>,
    pub(super) is_animated: bool,
    pub(super) frame_count: Option<u32>,
    pub(super) duration_ms: Option<u64>,
}

#[derive(Debug, Default)]
pub(super) struct MediaProbeTimings {
    pub(super) dimensions: Duration,
    pub(super) exif: Duration,
    pub(super) animation: Duration,
}

pub(super) fn probe_media(
    path: &Path,
    video_metadata: Option<Option<crate::video::VideoMetadata>>,
) -> (MediaProbe, MediaProbeTimings) {
    let mut timings = MediaProbeTimings::default();
    let extension = path.extension().unwrap_or_default().to_ascii_lowercase();

    if is_video_extension(&extension) {
        let span = tracing::debug_span!("video_catalog_probe", path = %path);
        let _entered = span.enter();
        let started = Instant::now();
        // The catalog keeps the source even when its container or codec cannot be read.
        let metadata = video_metadata.expect("video metadata must be probed before classification");
        timings.dimensions = started.elapsed();
        return (
            MediaProbe {
                kind: MediaKind::Video,
                format: extension,
                // The capture timeline uses this shared timestamp column. For videos it is the
                // container creation time; the inspector reports it separately from EXIF.
                exif_taken_ns: metadata.as_ref().and_then(|value| {
                    value
                        .creation_time
                        .as_deref()
                        .and_then(parse_video_creation_ns)
                }),
                width: metadata.as_ref().map(|value| value.width),
                height: metadata.as_ref().map(|value| value.height),
                is_animated: false,
                frame_count: metadata.as_ref().and_then(|value| value.frame_count),
                duration_ms: metadata.as_ref().and_then(|value| value.duration_ms),
            },
            timings,
        );
    }

    let started = Instant::now();
    let (sniffed_format, dimensions) = probe_image_header(path);
    timings.dimensions = started.elapsed();

    // The extension is only a fallback: a still image's real format comes from its header, so a
    // misnamed file (a JPEG saved with a `.png` extension, say) is catalogued as what it actually
    // is rather than what its name claims.
    let format = sniffed_format.unwrap_or_else(|| match extension.as_str() {
        "jpg" => "jpeg".to_owned(),
        "" => "unknown".to_owned(),
        value => value.to_owned(),
    });

    let started = Instant::now();
    let gif_metadata = if format == "gif" {
        probe_gif(path).ok()
    } else {
        None
    };
    timings.animation = started.elapsed();

    let started = Instant::now();
    let (exif_taken_ns, orientation) =
        probe_exif_metadata(path).unwrap_or((None, ExifOrientation::Identity));
    timings.exif = started.elapsed();

    (
        MediaProbe {
            kind: MediaKind::Image,
            format,
            exif_taken_ns,
            width: dimensions.map(|(width, height)| {
                if orientation.swaps_dimensions() {
                    height
                } else {
                    width
                }
            }),
            height: dimensions.map(|(width, height)| {
                if orientation.swaps_dimensions() {
                    width
                } else {
                    height
                }
            }),
            is_animated: gif_metadata.is_some_and(|value| value.0 > 1),
            frame_count: gif_metadata.map(|value| value.0),
            duration_ms: gif_metadata.map(|value| value.1),
        },
        timings,
    )
}

/// Detect the real still-image format from its magic bytes, along with its pixel dimensions read
/// from the same header. `None` for either half if the file can't be opened, or its header isn't
/// a still-image format nicegal-server decodes; callers fall back to the file extension.
fn probe_image_header(path: &Path) -> (Option<String>, Option<(u32, u32)>) {
    let Ok(file) = File::open(path) else {
        return (None, None);
    };
    let mut reader = BufReader::new(file);
    let Ok(kind) = imagesize::reader_type(&mut reader) else {
        return (None, None);
    };
    let dimensions = kind.reader_size(&mut reader).ok().and_then(|size| {
        Some((
            u32::try_from(size.width).ok()?,
            u32::try_from(size.height).ok()?,
        ))
    });
    (image_type_name(kind), dimensions)
}

fn image_type_name(kind: imagesize::ImageType) -> Option<String> {
    Some(
        match kind {
            imagesize::ImageType::Jpeg => "jpeg",
            imagesize::ImageType::Png => "png",
            imagesize::ImageType::Gif => "gif",
            imagesize::ImageType::Webp => "webp",
            imagesize::ImageType::Bmp => "bmp",
            _ => return None,
        }
        .to_owned(),
    )
}

fn probe_exif_metadata(path: &Path) -> Result<(Option<i64>, ExifOrientation)> {
    let exif =
        read_exif(path.as_std_path()).with_context(|| format!("reading EXIF data: {path}"))?;
    let orientation = orientation_from_exif(&exif);
    let Some(datetime) = exif
        .get(ExifTag::DateTimeOriginal)
        .and_then(|value| value.as_datetime())
        .or_else(|| {
            exif.get(ExifTag::ModifyDate)
                .and_then(|value| value.as_datetime())
        })
    else {
        return Ok((None, orientation));
    };
    let timestamp = match datetime.aware() {
        Some(datetime) => datetime.timestamp_nanos_opt(),
        // EXIF timestamps without an offset are interpreted as UTC for stable, host-independent
        // catalog values.
        None => datetime.into_naive().and_utc().timestamp_nanos_opt(),
    }
    .context("EXIF capture time exceeds the nanosecond timestamp range")?;
    Ok((Some(timestamp), orientation))
}

fn probe_gif(path: &Path) -> Result<(u32, u64)> {
    let file = File::open(path).with_context(|| format!("opening GIF: {path}"))?;
    let mut options = gif::DecodeOptions::new();
    // Cataloging needs frame control data, not pixels. Skipping LZW decompression avoids allocating
    // a frame-sized buffer and decoding every image block merely to count frames and add delays.
    options.skip_frame_decoding(true);
    let mut decoder = options
        .read_info(file)
        .with_context(|| format!("decoding GIF header: {path}"))?;
    let mut frame_count = 0_u32;
    let mut duration_ms = 0_u64;
    while let Some(frame) = decoder
        .next_frame_info()
        .with_context(|| format!("reading GIF frame metadata: {path}"))?
    {
        frame_count = frame_count
            .checked_add(1)
            .context("GIF frame count overflow")?;
        duration_ms = duration_ms.saturating_add(u64::from(frame.delay) * 10);
    }
    if frame_count == 0 {
        bail!("GIF contains no decodable frames: {path}");
    }
    Ok((frame_count, duration_ms))
}

pub fn is_catalog_media(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        let extension = extension.to_ascii_lowercase();
        matches!(
            extension.as_str(),
            "png" | "jpeg" | "jpg" | "gif" | "webp" | "bmp"
        ) || is_video_extension(&extension)
    })
}

pub fn is_catalog_video(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| is_video_extension(&extension.to_ascii_lowercase()))
}

fn is_video_extension(extension: &str) -> bool {
    matches!(
        extension,
        "mp4" | "m4v" | "mov" | "mkv" | "webm" | "avi" | "mpg" | "mpeg" | "m2ts"
    )
}

fn parse_video_creation_ns(value: &str) -> Option<i64> {
    let datetime =
        time::OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339).ok()?;
    i64::try_from(datetime.unix_timestamp_nanos()).ok()
}

pub fn is_ocr_image(asset: &Asset) -> bool {
    asset.media_kind == MediaKind::Image
        && matches!(asset.media_format.as_str(), "png" | "jpeg" | "gif" | "webp")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::*;
    use std::borrow::Cow;
    use std::fs::{self, File};
    use tempfile::TempDir;

    #[test]
    fn video_container_creation_time_parses_as_capture_time() {
        assert_eq!(
            parse_video_creation_ns("1970-01-01T00:00:01.123Z"),
            Some(1_123_000_000)
        );
        assert_eq!(
            parse_video_creation_ns("1970-01-01T01:00:00+01:00"),
            Some(0)
        );
        assert_eq!(parse_video_creation_ns("not a timestamp"), None);
    }

    #[test]
    fn gif_probe_counts_frames_and_delays_without_decoding_pixels() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("animated.gif"))?;
        {
            let mut file = File::create(&source)?;
            let mut encoder = gif::Encoder::new(&mut file, 1, 1, &[0, 0, 0])?;
            for delay in [3, 7, 11] {
                encoder.write_frame(&gif::Frame {
                    delay,
                    width: 1,
                    height: 1,
                    buffer: Cow::Borrowed(&[0]),
                    ..gif::Frame::default()
                })?;
            }
        }

        assert_eq!(probe_gif(&source)?, (3, 210));

        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;
        assert!(asset.is_animated);
        assert_eq!(asset.frame_count, Some(3));
        assert_eq!(asset.duration_ms, Some(210));
        Ok(())
    }

    #[test]
    fn media_format_comes_from_the_header_not_the_extension() -> Result<()> {
        let temp = TempDir::new()?;
        // A real JPEG wearing a `.png` extension, the exact mislabeling this catalog guards
        // against: the format it records must match the bytes, not the filename.
        let source = PathBuf::try_from(temp.path().join("mislabeled.png"))?;
        let jpeg = crate::imaging::encode_jpeg(4, 2, &[0u8; 4 * 2 * 3], 85.0)?;
        fs::write(&source, jpeg)?;

        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;

        assert_eq!(asset.media_format, "jpeg");
        assert_eq!((asset.width, asset.height), (Some(4), Some(2)));
        Ok(())
    }

    #[test]
    fn jpeg_orientation_sets_visual_catalog_dimensions() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("oriented.jpg"))?;
        fs::write(
            &source,
            crate::imaging::test_support::jpeg_with_orientation(6)?,
        )?;

        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;

        assert_eq!((asset.width, asset.height), (Some(2), Some(3)));
        Ok(())
    }
}
