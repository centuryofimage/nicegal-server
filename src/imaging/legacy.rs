use anyhow::{Context, Result};
use image::{GenericImageView, ImageFormat};

use super::{Format, Raster};

pub fn decode(data: &[u8], format: Format) -> Result<Raster> {
    let format = match format {
        Format::Bmp => ImageFormat::Bmp,
        _ => unreachable!("legacy decode only handles bmp"),
    };
    let decoded = image::load_from_memory_with_format(data, format).context("decoding image")?;
    let (width, height) = decoded.dimensions();
    Ok(Raster::Rgba {
        width,
        height,
        pixels: decoded.into_rgba8().into_raw(),
    })
}
