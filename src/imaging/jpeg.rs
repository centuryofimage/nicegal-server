use std::panic::{AssertUnwindSafe, catch_unwind};

use anyhow::{Context, Result, anyhow};
use mozjpeg::{ColorSpace, Compress, DctMethod, Decompress};

use super::Raster;

/// mozjpeg surfaces malformed-input errors by unwinding across the C library's longjmp-based
/// error path; catch that here so a corrupt JPEG on disk returns an error like every other
/// decoder instead of taking down the caller's thread.
pub fn decode(data: &[u8]) -> Result<Raster> {
    catch_unwind(AssertUnwindSafe(|| decode_inner(data)))
        .unwrap_or_else(|_| Err(anyhow!("decoding JPEG panicked")))
}

fn decode_inner(data: &[u8]) -> Result<Raster> {
    let mut decompress = Decompress::new_mem(data).context("reading JPEG header")?;
    // The thumbnail gets downscaled right after this, so the accuracy this trades away
    // (slow-but-precise IDCT, careful chroma upsampling) would be invisible anyway.
    decompress.dct_method(DctMethod::IntegerFast);
    decompress.do_fancy_upsampling(false);
    let mut decompress = decompress.rgb().context("starting JPEG decompression")?;
    let width = decompress.width() as u32;
    let height = decompress.height() as u32;
    let pixels: Vec<u8> = decompress
        .read_scanlines()
        .context("decoding JPEG scanlines")?;
    decompress
        .finish()
        .context("finishing JPEG decompression")?;
    Ok(Raster::Rgb {
        width,
        height,
        pixels,
    })
}

pub fn encode(width: u32, height: u32, rgb: &[u8], quality: f32) -> Result<Vec<u8>> {
    let mut compress = Compress::new(ColorSpace::JCS_RGB);
    // mozjpeg's real defaults spend extra passes (trellis quantization, scan optimization) on a
    // smaller file, which is the wrong tradeoff for a disposable thumbnail cache; fall back to
    // plain libjpeg-turbo-speed encoding instead.
    compress.set_fastest_defaults();
    compress.set_size(width as usize, height as usize);
    compress.set_quality(quality);
    let mut compress = compress
        .start_compress(Vec::new())
        .context("starting JPEG compression")?;
    compress
        .write_scanlines(rgb)
        .context("writing JPEG scanlines")?;
    compress.finish().context("finishing JPEG compression")
}
