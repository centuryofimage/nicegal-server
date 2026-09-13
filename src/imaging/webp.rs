use std::io::Cursor;

use anyhow::{Context, Result};
use image_webp::WebPDecoder;

use super::Raster;

pub fn decode(data: &[u8]) -> Result<Raster> {
    let mut decoder = WebPDecoder::new(Cursor::new(data)).context("reading WebP header")?;
    let (width, height) = decoder.dimensions();
    let has_alpha = decoder.has_alpha();
    let mut pixels = vec![
        0u8;
        decoder
            .output_buffer_size()
            .context("WebP image too large")?
    ];
    decoder
        .read_image(&mut pixels)
        .context("decoding WebP image data")?;
    Ok(if has_alpha {
        Raster::Rgba {
            width,
            height,
            pixels,
        }
    } else {
        Raster::Rgb {
            width,
            height,
            pixels,
        }
    })
}
