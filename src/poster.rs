use anyhow::{Context, Result, bail};
use camino::Utf8Path as Path;
use tracing::{debug_span, field, trace_span};

use crate::imaging::{self, Raster};
use crate::thumbs::ThumbnailEncoding;

#[derive(Debug, PartialEq, Eq)]
pub struct StaticPoster {
    pub width: u32,
    pub height: u32,
    pub encoding: ThumbnailEncoding,
    pub data: Vec<u8>,
}

/// Decode a supported still image and encode a bounded static thumbnail.
///
/// Alpha-bearing sources remain PNGs. Opaque sources use JPEG to keep the hot thumbnail store
/// compact. An animated GIF source uses its first composited frame.
pub fn still_image(path: &Path, maximum_edge: u32) -> Result<StaticPoster> {
    if maximum_edge == 0 {
        bail!("poster maximum edge must be greater than zero");
    }
    let span = debug_span!("poster", path = %path, maximum_edge);
    let _entered = span.enter();
    let source = still_image_source(path)?;
    let poster = resize_for_poster(path, &source, maximum_edge, maximum_edge)?;
    encode_poster(path, poster)
}

/// Decode once and produce all requested maximum-edge variants.
pub fn image_buckets(path: &Path, buckets: &[u16]) -> Result<Vec<StaticPoster>> {
    if buckets.contains(&0) {
        bail!("poster maximum edge must be greater than zero");
    }
    let span = debug_span!("poster", path = %path, buckets = buckets.len());
    let _entered = span.enter();
    let source = still_image_source(path)?;
    buckets
        .iter()
        .map(|bucket| {
            let maximum_edge = u32::from(*bucket);
            let poster = resize_for_poster(path, &source, maximum_edge, maximum_edge)?;
            encode_poster(path, poster)
        })
        .collect()
}

fn still_image_source(path: &Path) -> Result<Raster> {
    let data = {
        let span = trace_span!(target: "nicegal_core::thumbs", "poster_open", path = %path);
        let _entered = span.enter();
        std::fs::read(path).with_context(|| format!("opening image: {path}"))?
    };
    let decode = trace_span!(target: "nicegal_core::thumbs", "poster_decode", path = %path);
    let _entered = decode.enter();
    imaging::decode(&data).with_context(|| format!("decoding image: {path}"))
}

fn resize_for_poster(
    path: &Path,
    source: &Raster,
    maximum_width: u32,
    maximum_height: u32,
) -> Result<Raster> {
    let resize = trace_span!(
        target: "nicegal_core::thumbs",
        "poster_resize",
        maximum_width,
        maximum_height,
        width = field::Empty,
        height = field::Empty
    );
    let _entered = resize.enter();
    let (width, height) = imaging::fit_within(
        source.width(),
        source.height(),
        maximum_width,
        maximum_height,
    );
    let poster = imaging::resize(source, width, height)
        .with_context(|| format!("resizing image: {path}"))?;
    resize.record("width", width);
    resize.record("height", height);
    Ok(poster)
}

fn encode_poster(path: &Path, poster: Raster) -> Result<StaticPoster> {
    let width = poster.width();
    let height = poster.height();
    let span = trace_span!(
        target: "nicegal_core::thumbs",
        "poster_encode",
        path = %path,
        width,
        height,
        encoding = field::Empty,
        data_bytes = field::Empty,
    );
    let _entered = span.enter();
    let (data, encoding) = if poster.has_alpha() {
        let data = imaging::encode_png(width, height, &poster.into_rgba_bytes())
            .with_context(|| format!("encoding PNG thumbnail: {path}"))?;
        (data, ThumbnailEncoding::Png)
    } else {
        let data = imaging::encode_jpeg(width, height, &poster.flatten_rgb(WHITE), 85.0)
            .with_context(|| format!("encoding JPEG thumbnail: {path}"))?;
        (data, ThumbnailEncoding::Jpeg)
    };
    span.record("encoding", encoding.content_type());
    span.record("data_bytes", data.len());
    Ok(StaticPoster {
        width,
        height,
        encoding,
        data,
    })
}

const WHITE: [u8; 3] = [255, 255, 255];

/// Decode the GIF canvas's first composited frame and encode that static poster.
/// The animated source is never resized, re-encoded, or returned as thumbnail bytes.
pub fn gif_first_frame(path: &Path, maximum_edge: u32) -> Result<StaticPoster> {
    if maximum_edge == 0 {
        bail!("poster maximum edge must be greater than zero");
    }
    let span = debug_span!("gif_poster", path = %path, maximum_edge);
    let _entered = span.enter();
    let source = still_image_source(path)?;
    let poster = resize_for_poster(path, &source, maximum_edge, maximum_edge)?;
    encode_poster(path, poster)
}

#[cfg(test)]
mod tests {
    use super::*;
    use camino::Utf8PathBuf as PathBuf;
    use std::borrow::Cow;
    use std::fs::File;
    use tempfile::TempDir;

    fn write_gif(path: &std::path::Path) -> Result<()> {
        let mut file = File::create(path)?;
        // Palette: index 0 = red, index 1 = blue.
        let mut encoder = gif::Encoder::new(&mut file, 4, 2, &[255, 0, 0, 0, 0, 255])?;
        encoder.set_repeat(gif::Repeat::Infinite)?;
        for index in [0u8, 1u8] {
            encoder.write_frame(&gif::Frame {
                delay: 10,
                width: 4,
                height: 2,
                buffer: Cow::Owned(vec![index; 8]),
                ..gif::Frame::default()
            })?;
        }
        Ok(())
    }

    #[test]
    fn animated_gif_produces_static_first_frame() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("animated.gif"))?;
        write_gif(path.as_std_path())?;

        let poster = gif_first_frame(&path, 2)?;
        assert_eq!((poster.width, poster.height), (2, 1));
        assert_eq!(poster.encoding, ThumbnailEncoding::Jpeg);
        let decoded = imaging::decode(&poster.data)?;
        let pixel = &decoded.pixels()[0..3];
        assert!(
            pixel[0] > 200 && pixel[1] < 55 && pixel[2] < 55,
            "expected a red pixel after lossy JPEG round-trip, got {pixel:?}"
        );
        Ok(())
    }

    #[test]
    fn jpeg_thumbnail_uses_visual_orientation() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("oriented.jpg"))?;
        std::fs::write(&path, imaging::test_support::jpeg_with_orientation(6)?)?;

        let poster = still_image(&path, 3)?;

        assert_eq!((poster.width, poster.height), (2, 3));
        Ok(())
    }

    #[test]
    fn broken_gif_returns_an_error() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("broken.gif"))?;
        File::create(&path)?;
        assert!(gif_first_frame(&path, 128).is_err());
        Ok(())
    }
}
