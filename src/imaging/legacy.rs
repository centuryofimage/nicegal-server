use anyhow::{Context, Result};
use image::{GenericImageView, ImageFormat};

use super::Raster;

pub fn decode_bmp(data: &[u8]) -> Result<Raster> {
    let decoded = image::load_from_memory_with_format(data, ImageFormat::Bmp)
        .context("decoding BMP image")?;
    let (width, height) = decoded.dimensions();
    Ok(Raster::Rgba {
        width,
        height,
        pixels: decoded.into_rgba8().into_raw(),
    })
}
