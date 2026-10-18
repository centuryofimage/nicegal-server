use anyhow::{Context, Result, bail};
use turbojpeg::{Compressor, Decompressor, Image, PixelFormat, ScalingFactor, Subsamp};

use super::Raster;

/// libjpeg's decoder warnings. Like browsers and Pillow with truncated images allowed, we keep
/// whatever image libjpeg recovers: corrupt data becomes damaged blocks and a truncated file gets a
/// filled-in bottom. TurboJPEG also reports a fatal error that follows a warning as a warning, but
/// with the fatal message, so only these texts mean a usable image.
const LIBJPEG_WARNINGS: &[&str] = &[
    "Corrupt JPEG data:",
    "Premature end of JPEG file",
    "Inconsistent progression sequence",
    "Invalid SOS parameters for sequential JPEG",
    "Unknown Adobe color transform code",
    "Warning: unknown JFIF revision number",
];

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
    decompressor.set_accept_warnings(true);
    let header = decompressor
        .read_header(data)
        .context("reading JPEG header")?;
    reject_fatal(&decompressor)?;
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
    reject_fatal(&decompressor)?;
    Ok((
        Raster::Rgb {
            width: scaled.width as u32,
            height: scaled.height as u32,
            pixels,
        },
        original_dimensions,
    ))
}

fn reject_fatal(decompressor: &Decompressor) -> Result<()> {
    match decompressor.warning() {
        Some(warning)
            if LIBJPEG_WARNINGS
                .iter()
                .any(|known| warning.starts_with(known)) =>
        {
            tracing::debug!(warning, "decoding a JPEG past a libjpeg warning");
            Ok(())
        }
        Some(error) => bail!("invalid JPEG: {error}"),
        None => Ok(()),
    }
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

    /// A JPEG with enough entropy-coded data that cutting it short loses visible rows.
    fn noisy_jpeg() -> Result<Vec<u8>> {
        let mut state = 0x2545_f491_u32;
        let pixels = (0..256 * 192 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect::<Vec<_>>();
        encode(256, 192, &pixels, 90.0)
    }

    /// An APP1 (EXIF) segment whose length stops short of its contents, right after SOI.
    fn with_short_metadata(jpeg: &[u8]) -> Vec<u8> {
        [
            &jpeg[..2],
            &[0xff, 0xe1, 0, 4, b'E', b'x', 1, 2, 3],
            &jpeg[2..],
        ]
        .concat()
    }

    #[test]
    fn truncated_jpeg_keeps_its_decoded_rows() -> Result<()> {
        let jpeg = noisy_jpeg()?;
        let Raster::Rgb {
            pixels: expected, ..
        } = decode(&jpeg)?
        else {
            bail!("expected an RGB raster");
        };
        let Raster::Rgb {
            width,
            height,
            pixels,
        } = decode(&jpeg[..jpeg.len() / 2])?
        else {
            bail!("expected an RGB raster");
        };
        assert_eq!((width, height), (256, 192));
        let top = 16 * 256 * 3;
        assert_eq!(pixels[..top], expected[..top]);
        Ok(())
    }

    #[test]
    fn stray_bytes_before_a_marker_are_ignored() -> Result<()> {
        let jpeg = noisy_jpeg()?;
        let expected = decode(&jpeg)?;
        // Junk before the end-of-image marker.
        let end = jpeg.len() - 2;
        let trailing = [&jpeg[..end], &[1, 2, 3], &jpeg[end..]].concat();
        for damaged in [trailing, with_short_metadata(&jpeg)] {
            assert!(decode(&damaged)? == expected);
        }
        Ok(())
    }

    #[test]
    fn fatal_error_after_a_warning_is_rejected() -> Result<()> {
        // The metadata warning comes first, then the file ends before any frame header.
        let jpeg = with_short_metadata(&noisy_jpeg()?);
        let error = decode(&jpeg[..16]).unwrap_err();
        assert!(
            !LIBJPEG_WARNINGS
                .iter()
                .any(|known| format!("{error:#}").contains(known)),
            "{error:#}"
        );

        // Now the fatal error comes during decompression: a progressive file whose second
        // Huffman table counts more codes than exist.
        let mut compressor = Compressor::new()?;
        compressor.set_progressive(true)?;
        compressor.set_optimize(true)?;
        let pixels = vec![90; 64 * 64 * 3];
        let mut jpeg = with_short_metadata(&compressor.compress_to_vec(Image {
            pixels: pixels.as_slice(),
            width: 64,
            pitch: 64 * 3,
            height: 64,
            format: PixelFormat::RGB,
        })?);
        let first_scan = jpeg
            .windows(2)
            .position(|marker| marker == [0xff, 0xda])
            .unwrap();
        let table = first_scan
            + jpeg[first_scan..]
                .windows(2)
                .position(|marker| marker == [0xff, 0xc4])
                .unwrap();
        jpeg[table + 5..table + 21].fill(0xff);
        let error = decode(&jpeg).unwrap_err();
        assert!(
            format!("{error:#}").contains("invalid JPEG: Bogus Huffman table"),
            "{error:#}"
        );
        Ok(())
    }

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
