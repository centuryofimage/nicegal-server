use std::io::Cursor;

use anyhow::{Context, Result};
use gif::ColorOutput;

use super::Raster;

pub fn decode_first_frame(data: &[u8]) -> Result<Raster> {
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(ColorOutput::RGBA);
    let mut decoder = options
        .read_info(Cursor::new(data))
        .context("decoding GIF header")?;
    let screen_width = u32::from(decoder.width());
    let screen_height = u32::from(decoder.height());
    let frame = decoder
        .read_next_frame()
        .context("decoding first GIF frame")?
        .context("GIF contains no decodable frames")?;

    let left = u32::from(frame.left);
    let top = u32::from(frame.top);
    let frame_width = u32::from(frame.width);
    let frame_height = u32::from(frame.height);
    if left == 0 && top == 0 && frame_width == screen_width && frame_height == screen_height {
        return Ok(Raster::Rgba {
            width: screen_width,
            height: screen_height,
            pixels: frame.buffer.to_vec(),
        });
    }

    // The first frame doesn't have to cover the whole logical screen; place it on an otherwise
    // transparent canvas at its own offset.
    let mut canvas = vec![0u8; screen_width as usize * screen_height as usize * 4];
    let copy_width = frame_width.min(screen_width.saturating_sub(left)) as usize;
    for row in 0..frame_height as usize {
        let dst_y = top as usize + row;
        if dst_y >= screen_height as usize {
            break;
        }
        let src_start = row * frame_width as usize * 4;
        let dst_start = (dst_y * screen_width as usize + left as usize) * 4;
        canvas[dst_start..dst_start + copy_width * 4]
            .copy_from_slice(&frame.buffer[src_start..src_start + copy_width * 4]);
    }
    Ok(Raster::Rgba {
        width: screen_width,
        height: screen_height,
        pixels: canvas,
    })
}
