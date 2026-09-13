use std::io::Cursor;

use anyhow::{Context, Result};
use png::{BitDepth, ColorType, Decoder, Encoder, Transformations};

use super::Raster;

pub fn decode(data: &[u8]) -> Result<Raster> {
    let mut decoder = Decoder::new(Cursor::new(data));
    decoder.set_transformations(Transformations::EXPAND | Transformations::STRIP_16);
    let mut reader = decoder.read_info().context("reading PNG header")?;
    let mut buffer = vec![0u8; reader.output_buffer_size().context("empty PNG image")?];
    let info = reader
        .next_frame(&mut buffer)
        .context("decoding PNG image data")?;
    let width = info.width;
    let height = info.height;
    buffer.truncate(info.buffer_size());
    Ok(match info.color_type {
        ColorType::Rgba => Raster::Rgba {
            width,
            height,
            pixels: buffer,
        },
        ColorType::Rgb => Raster::Rgb {
            width,
            height,
            pixels: buffer,
        },
        ColorType::GrayscaleAlpha => {
            let mut rgba = Vec::with_capacity(buffer.len() * 2);
            for pixel in buffer.chunks_exact(2) {
                rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]);
            }
            Raster::Rgba {
                width,
                height,
                pixels: rgba,
            }
        }
        ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity(buffer.len() * 3);
            for &gray in &buffer {
                rgb.extend_from_slice(&[gray, gray, gray]);
            }
            Raster::Rgb {
                width,
                height,
                pixels: rgb,
            }
        }
        ColorType::Indexed => unreachable!("EXPAND transformation removes palette output"),
    })
}

pub fn encode(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    let mut encoder = Encoder::new(&mut data, width, height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    // The crate's own default (`Balanced`) is tuned for smaller files over speed; `Fast` uses its
    // PNG-tuned deflate implementation instead, matching what a disposable thumbnail cache wants.
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header().context("writing PNG header")?;
    writer
        .write_image_data(rgba)
        .context("writing PNG image data")?;
    writer.finish().context("finishing PNG encode")?;
    Ok(data)
}
