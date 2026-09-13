use anyhow::{Context, Result, anyhow};
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use super::Raster;

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
