use anyhow::{Context, Result, anyhow};
use fast_image_resize::images::{Image, ImageRef, TypedImage, TypedImageRef};
use fast_image_resize::pixels::F32x3;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use super::Raster;

const MAX_FLOAT_SOURCE_BYTES: usize = 64 * 1024 * 1024;

fn resize_algorithm(filter: image::imageops::FilterType) -> Option<ResizeAlg> {
    Some(match filter {
        image::imageops::FilterType::Nearest => ResizeAlg::Nearest,
        image::imageops::FilterType::Triangle => ResizeAlg::Convolution(FilterType::Bilinear),
        image::imageops::FilterType::CatmullRom => ResizeAlg::Convolution(FilterType::CatmullRom),
        image::imageops::FilterType::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
        image::imageops::FilterType::Gaussian => return None,
    })
}

/// Byte convolution for shrinking model inputs; enlargement keeps float intermediates.
pub fn resize_rgb(
    source: image::RgbImage,
    width: u32,
    height: u32,
    filter: image::imageops::FilterType,
) -> Result<image::RgbImage> {
    if width > source.width() || height > source.height() || source.dimensions() == (width, height)
    {
        return resize_rgb_float(source, width, height, filter);
    }
    let Some(algorithm) = resize_algorithm(filter) else {
        return resize_rgb_float(source, width, height, filter);
    };
    resize_rgb_bytes(&source, width, height, algorithm)
}

/// Resize borrowed RGB pixels with byte intermediates and CPU-specific kernels.
pub fn resize_rgb_bytes(
    source: &image::RgbImage,
    width: u32,
    height: u32,
    algorithm: ResizeAlg,
) -> Result<image::RgbImage> {
    if source.dimensions() == (width, height) {
        return Ok(source.clone());
    }
    let input = ImageRef::new(
        source.width(),
        source.height(),
        source.as_raw(),
        PixelType::U8x3,
    )
    .context("building RGB byte resize source")?;
    let mut output = Image::new(width, height, PixelType::U8x3);
    Resizer::new()
        .resize(
            &input,
            &mut output,
            &ResizeOptions::new().resize_alg(algorithm),
        )
        .context("resizing RGB image with byte intermediates")?;
    image::RgbImage::from_raw(width, height, output.into_vec())
        .context("building byte-resized RGB image")
}

/// Float convolution for enlargement, preserving model-input precision.
fn resize_rgb_float(
    source: image::RgbImage,
    width: u32,
    height: u32,
    filter: image::imageops::FilterType,
) -> Result<image::RgbImage> {
    resize_rgb_with_budget(source, width, height, filter, MAX_FLOAT_SOURCE_BYTES)
}

fn resize_rgb_with_budget(
    source: image::RgbImage,
    width: u32,
    height: u32,
    filter: image::imageops::FilterType,
    max_float_bytes: usize,
) -> Result<image::RgbImage> {
    if source.dimensions() == (width, height) {
        return Ok(source);
    }
    // Four preprocessing workers may overlap; bound the extra full-resolution float copy.
    if source
        .as_raw()
        .len()
        .checked_mul(size_of::<f32>())
        .is_none_or(|bytes| bytes > max_float_bytes)
    {
        return Ok(image::imageops::resize(&source, width, height, filter));
    }
    let Some(algorithm) = resize_algorithm(filter) else {
        return Ok(image::imageops::resize(&source, width, height, filter));
    };
    let dimensions = source.dimensions();
    let pixels: Vec<_> = source
        .pixels()
        .map(|pixel| F32x3::new(pixel.0.map(f32::from)))
        .collect();
    drop(source);
    let source = TypedImageRef::new(dimensions.0, dimensions.1, &pixels)
        .context("building float resize source")?;
    let mut destination = TypedImage::<F32x3>::new(width, height);
    Resizer::new()
        .resize_typed(
            &source,
            &mut destination,
            &ResizeOptions::new().resize_alg(algorithm),
        )
        .context("resizing model input")?;
    let pixels = destination
        .pixels()
        .iter()
        .flat_map(|pixel| pixel.0.map(|value| value.round().clamp(0.0, 255.0) as u8))
        .collect();
    image::RgbImage::from_raw(width, height, pixels).context("building resized model input")
}

/// The size a `src_width`x`src_height` box scales to so it fits entirely within
/// `max_width`x`max_height`, preserving aspect ratio. Matches `image::DynamicImage::thumbnail`'s
/// semantics: an undersized source is scaled up to fill the box, not left at its original size.
pub fn fit_within(src_width: u32, src_height: u32, max_width: u32, max_height: u32) -> (u32, u32) {
    let ratio = f64::min(
        f64::from(max_width) / f64::from(src_width),
        f64::from(max_height) / f64::from(src_height),
    );
    let width = ((f64::from(src_width) * ratio).round() as u32).max(1);
    let height = ((f64::from(src_height) * ratio).round() as u32).max(1);
    (width, height)
}

pub fn resize(source: &Raster, width: u32, height: u32) -> Result<Raster> {
    if source.width() == width && source.height() == height {
        return Ok(source.clone());
    }
    let pixel_type = match source {
        Raster::Rgb { .. } => PixelType::U8x3,
        Raster::Rgba { .. } => PixelType::U8x4,
    };
    let src_image = ImageRef::new(source.width(), source.height(), source.pixels(), pixel_type)
        .map_err(|error| anyhow!("building resize source image: {error}"))?;
    let mut dst_image = Image::new(width, height, pixel_type);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    Resizer::new()
        .resize(&src_image, &mut dst_image, &options)
        .context("resizing image")?;
    let pixels = dst_image.into_vec();
    Ok(match source {
        Raster::Rgb { .. } => Raster::Rgb {
            width,
            height,
            pixels,
        },
        Raster::Rgba { .. } => Raster::Rgba {
            width,
            height,
            pixels,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_resize_only_changes_downscaling() -> Result<()> {
        let source = image::RgbImage::from_fn(317, 193, |x, y| {
            image::Rgb([((x * 17 + y * 23) % 256) as u8; 3])
        });
        for (width, height) in [(634, 386), (317, 193), (224, 224)] {
            let expected = resize_rgb_float(
                source.clone(),
                width,
                height,
                image::imageops::FilterType::CatmullRom,
            )?;
            assert_eq!(
                resize_rgb(
                    source.clone(),
                    width,
                    height,
                    image::imageops::FilterType::CatmullRom
                )?,
                expected
            );
        }
        let downscaled = resize_rgb(
            source.clone(),
            100,
            61,
            image::imageops::FilterType::CatmullRom,
        )?;
        assert_eq!(downscaled.dimensions(), (100, 61));
        assert_ne!(
            downscaled,
            resize_rgb_float(source, 100, 61, image::imageops::FilterType::CatmullRom)?
        );
        Ok(())
    }

    #[test]
    fn exceeded_float_budget_uses_the_precise_reference_resizer() -> Result<()> {
        let source = image::RgbImage::from_fn(317, 193, |x, y| {
            image::Rgb([
                ((x * 17 + y * 23) % 256) as u8,
                ((x * 53 + y * 7) % 256) as u8,
                ((x * 3 + y * 97) % 256) as u8,
            ])
        });
        let expected =
            image::imageops::resize(&source, 224, 224, image::imageops::FilterType::CatmullRom);
        let actual =
            resize_rgb_with_budget(source, 224, 224, image::imageops::FilterType::CatmullRom, 0)?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn model_resize_preserves_filtering_with_float_intermediates() -> Result<()> {
        for (width, height, target_width, target_height) in [
            (513, 317, 224, 224),
            (19, 37, 224, 224),
            (1, 37, 1, 224),
            (317, 13, 224, 13),
        ] {
            let source = image::RgbImage::from_fn(width, height, |x, y| {
                image::Rgb([
                    ((x * 17 + y * 23) % 256) as u8,
                    ((x * 53 + y * 7) % 256) as u8,
                    ((x * 3 + y * 97) % 256) as u8,
                ])
            });
            for filter in [
                image::imageops::FilterType::Nearest,
                image::imageops::FilterType::Triangle,
                image::imageops::FilterType::CatmullRom,
                image::imageops::FilterType::Lanczos3,
            ] {
                let expected =
                    image::imageops::resize(&source, target_width, target_height, filter);
                let actual = resize_rgb_float(source.clone(), target_width, target_height, filter)?;
                assert_eq!(actual.dimensions(), expected.dimensions());
                assert!(
                    actual
                        .as_raw()
                        .iter()
                        .zip(expected.as_raw())
                        .all(|(a, b)| a.abs_diff(*b) <= 1),
                    "filter {filter:?}, {width}x{height}"
                );
            }
        }
        Ok(())
    }
}
