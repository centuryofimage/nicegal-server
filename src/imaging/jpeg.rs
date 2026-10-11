use anyhow::{Context, Result, bail};
use turbojpeg::{Compressor, Decompressor, Image, PixelFormat, ScalingFactor, Subsamp};

use super::Raster;

pub fn decode(data: &[u8]) -> Result<Raster> {
    decode_inner(data, false, None).map(|(image, _)| image)
}

pub fn decode_accurate(data: &[u8]) -> Result<Raster> {
    decode_inner(data, true, None).map(|(image, _)| image)
}

pub fn decode_for_thumbnail(data: &[u8], maximum_edge: u32) -> Result<(Raster, (u32, u32))> {
    decode_inner(data, false, Some(maximum_edge))
}

fn thumbnail_scale_numerator(longest_edge: u32, maximum_edge: u32) -> u32 {
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
    let mut decompressor = Decompressor::new().context("creating JPEG decompressor")?;
    let header = decompressor
        .read_header(data)
        .context("reading JPEG header")?;
    let original_dimensions = (
        u32::try_from(header.width).context("JPEG width")?,
        u32::try_from(header.height).context("JPEG height")?,
    );
    let scaled = match maximum_edge {
        Some(maximum_edge) => {
            let longest_edge = original_dimensions.0.max(original_dimensions.1);
            let factor = ScalingFactor::new(
                thumbnail_scale_numerator(longest_edge, maximum_edge) as usize,
                8,
            );
            decompressor
                .set_scaling_factor(factor)
                .context("setting JPEG scaling factor")?;
            header.scaled(factor)
        }
        None => header,
    };
    // The thumbnail gets downscaled right after this, so cheaper chroma upsampling would be
    // invisible there; model preprocessing keeps libjpeg's smooth upsampling for parity.
    decompressor
        .set_fast_upsample(!accurate)
        .context("setting JPEG chroma upsampling")?;

    let pitch = scaled
        .width
        .checked_mul(PixelFormat::RGB.size())
        .context("JPEG row size overflows")?;
    let len = pitch
        .checked_mul(scaled.height)
        .context("JPEG image size overflows")?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(len)
        .context("allocating JPEG pixels")?;
    pixels.resize(len, 0);
    let output = Image {
        pixels: pixels.as_mut_slice(),
        width: scaled.width,
        pitch,
        height: scaled.height,
        format: PixelFormat::RGB,
    };
    decompressor
        .decompress(data, output)
        .context("decoding JPEG")?;
    Ok((
        Raster::Rgb {
            width: scaled.width as u32,
            height: scaled.height as u32,
            pixels,
        },
        original_dimensions,
    ))
}

pub fn encode(width: u32, height: u32, rgb: &[u8], quality: f32) -> Result<Vec<u8>> {
    let (width, height) = (width as usize, height as usize);
    // Plain libjpeg-turbo speed: a disposable thumbnail cache doesn't want extra optimization
    // passes for a slightly smaller file.
    let pitch = width.checked_mul(3).context("JPEG row size overflows")?;
    if width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        != Some(rgb.len())
    {
        bail!(
            "RGB buffer holds {} bytes, not {width}x{height}x3",
            rgb.len()
        );
    }
    let mut compressor = Compressor::new().context("creating JPEG compressor")?;
    compressor
        .set_quality(quality.round().clamp(1.0, 100.0) as i32)
        .context("setting JPEG quality")?;
    compressor
        .set_subsamp(Subsamp::Sub2x2)
        .context("setting JPEG chroma subsampling")?;
    compressor
        .compress_to_vec(Image {
            pixels: rgb,
            width,
            pitch,
            height,
            format: PixelFormat::RGB,
        })
        .context("encoding JPEG")
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
