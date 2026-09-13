mod gif;
mod jpeg;
mod legacy;
mod png;
mod raster;
mod resize;
mod webp;

pub(crate) use raster::ExifOrientation;
pub use raster::Raster;
pub use resize::{fit_within, resize};

use anyhow::{Context, Result, bail};
use nom_exif::{Exif, ExifTag, MediaParser, MediaSource};
use std::io::Cursor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Jpeg,
    Png,
    Gif,
    Webp,
    Bmp,
}

impl Format {
    pub fn detect(data: &[u8]) -> Result<Self> {
        let kind = imagesize::image_type(data).context("detecting image format")?;
        Ok(match kind {
            imagesize::ImageType::Jpeg => Self::Jpeg,
            imagesize::ImageType::Png => Self::Png,
            imagesize::ImageType::Gif => Self::Gif,
            imagesize::ImageType::Webp => Self::Webp,
            imagesize::ImageType::Bmp => Self::Bmp,
            other => bail!("unsupported image format: {other:?}"),
        })
    }
}

/// Decode a still image to raw pixels. For an animated GIF this is its first composited frame.
pub fn decode(data: &[u8]) -> Result<Raster> {
    let format = Format::detect(data)?;
    let raster = match format {
        Format::Jpeg => jpeg::decode(data),
        Format::Png => png::decode(data),
        Format::Gif => gif::decode_first_frame(data),
        Format::Webp => webp::decode(data),
        Format::Bmp => legacy::decode(data, Format::Bmp),
    }?;
    let orientation = match format {
        Format::Jpeg => orientation_from_jpeg(data),
        _ => ExifOrientation::Identity,
    };
    Ok(raster.orient(orientation))
}

pub fn encode_jpeg(width: u32, height: u32, rgb: &[u8], quality: f32) -> Result<Vec<u8>> {
    jpeg::encode(width, height, rgb, quality)
}

pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>> {
    png::encode(width, height, rgba)
}

pub(crate) fn orientation_from_exif(exif: &Exif) -> ExifOrientation {
    exif.get(ExifTag::Orientation)
        .and_then(|value| value.as_u16())
        .map(ExifOrientation::from_value)
        .unwrap_or(ExifOrientation::Identity)
}

fn orientation_from_jpeg(data: &[u8]) -> ExifOrientation {
    let source = match MediaSource::seekable(Cursor::new(data)) {
        Ok(source) => source,
        Err(_) => return ExifOrientation::Identity,
    };
    let mut parser = MediaParser::new();
    let exif: Exif = match parser.parse_exif(source) {
        Ok(exif) => exif.into(),
        Err(_) => return ExifOrientation::Identity,
    };
    orientation_from_exif(&exif)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn jpeg_with_orientation(orientation: u16) -> Result<Vec<u8>> {
        let pixels = [
            255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 255,
        ];
        let jpeg = encode_jpeg(3, 2, &pixels, 100.0)?;
        let mut oriented = Vec::with_capacity(jpeg.len() + 36);
        oriented.extend_from_slice(&jpeg[..2]);
        oriented.extend_from_slice(&[
            0xff, 0xe1, 0x00, 0x22, b'E', b'x', b'i', b'f', 0, 0, b'M', b'M', 0, 42, 0, 0, 0, 8, 0,
            1, 0x01, 0x12, 0, 3, 0, 0, 0, 1,
        ]);
        oriented.extend_from_slice(&orientation.to_be_bytes());
        oriented.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        oriented.extend_from_slice(&jpeg[2..]);
        Ok(oriented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_decode_applies_exif_orientation() -> Result<()> {
        let jpeg = test_support::jpeg_with_orientation(6)?;

        let decoded = decode(&jpeg)?;

        assert_eq!((decoded.width(), decoded.height()), (2, 3));
        Ok(())
    }
}
