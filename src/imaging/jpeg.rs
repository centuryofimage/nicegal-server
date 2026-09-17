use std::panic::{AssertUnwindSafe, catch_unwind};

use anyhow::{Context, Result, anyhow};
use mozjpeg::{ColorSpace, Compress, DctMethod, Decompress};

use super::Raster;

/// mozjpeg surfaces malformed-input errors by unwinding across the C library's longjmp-based
/// error path; catch that here so a corrupt JPEG on disk returns an error like every other
/// decoder instead of taking down the caller's thread.
pub fn decode(data: &[u8]) -> Result<Raster> {
    catch_unwind(AssertUnwindSafe(|| {
        decode_inner(data, false, None).map(|(image, _)| image)
    }))
    .unwrap_or_else(|_| Err(anyhow!("decoding JPEG panicked")))
}

pub fn decode_accurate(data: &[u8]) -> Result<Raster> {
    catch_unwind(AssertUnwindSafe(|| {
        decode_inner(data, true, None).map(|(image, _)| image)
    }))
    .unwrap_or_else(|_| Err(anyhow!("decoding JPEG panicked")))
}

pub fn decode_for_thumbnail(data: &[u8], maximum_edge: u32) -> Result<(Raster, (u32, u32))> {
    catch_unwind(AssertUnwindSafe(|| {
        decode_inner(data, false, Some(maximum_edge))
    }))
    .unwrap_or_else(|_| Err(anyhow!("decoding JPEG panicked")))
}

fn thumbnail_scale_numerator(longest_edge: u32, maximum_edge: u32) -> u8 {
    // Keep at least two source pixels per output pixel for the final filtered resize.
    // libjpeg's IDCT scaling supports 1/8, 1/4, and 1/2 without a full-size RGB buffer.
    let minimum_edge = maximum_edge.saturating_mul(2);
    [1, 2, 4]
        .into_iter()
        .find(|numerator| {
            u64::from(longest_edge) * u64::from(*numerator) >= u64::from(minimum_edge) * 8
        })
        .unwrap_or(8)
}

fn decode_inner(
    data: &[u8],
    accurate: bool,
    maximum_edge: Option<u32>,
) -> Result<(Raster, (u32, u32))> {
    let mut decompress = Decompress::new_mem(data).context("reading JPEG header")?;
    let original_dimensions = (decompress.width() as u32, decompress.height() as u32);
    if let Some(maximum_edge) = maximum_edge {
        let longest_edge = original_dimensions.0.max(original_dimensions.1);
        decompress.scale(thumbnail_scale_numerator(longest_edge, maximum_edge));
    }
    // The thumbnail gets downscaled right after this, so the accuracy this trades away
    // (slow-but-precise IDCT, careful chroma upsampling) would be invisible anyway.
    decompress.dct_method(if accurate {
        DctMethod::IntegerSlow
    } else {
        DctMethod::IntegerFast
    });
    decompress.do_fancy_upsampling(accurate);
    let mut decompress = decompress.rgb().context("starting JPEG decompression")?;
    let width = decompress.width() as u32;
    let height = decompress.height() as u32;
    let pixels: Vec<u8> = decompress
        .read_scanlines()
        .context("decoding JPEG scanlines")?;
    decompress
        .finish()
        .context("finishing JPEG decompression")?;
    Ok((
        Raster::Rgb {
            width,
            height,
            pixels,
        },
        original_dimensions,
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thumbnail_scaling_keeps_oversampling() {
        assert_eq!(thumbnail_scale_numerator(512, 512), 8);
        assert_eq!(thumbnail_scale_numerator(2048, 512), 4);
        assert_eq!(thumbnail_scale_numerator(4096, 512), 2);
        assert_eq!(thumbnail_scale_numerator(8192, 512), 1);
        assert_eq!(thumbnail_scale_numerator(4096, 1024), 4);
    }

    #[test]
    fn thumbnail_decode_scales_without_changing_general_decode() -> Result<()> {
        let pixels = vec![128; 2048 * 1024 * 3];
        let jpeg = encode(2048, 1024, &pixels, 85.0)?;
        assert_eq!(
            (decode(&jpeg)?.width(), decode(&jpeg)?.height()),
            (2048, 1024)
        );
        let (thumbnail, original_dimensions) = decode_for_thumbnail(&jpeg, 512)?;
        assert_eq!(original_dimensions, (2048, 1024));
        assert_eq!((thumbnail.width(), thumbnail.height()), (1024, 512));
        Ok(())
    }
}
