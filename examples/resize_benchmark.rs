//! Compare release builds with CARGO_PROFILE_RELEASE_LTO=thin and off.
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use anyhow::Result;
use fast_image_resize::{FilterType, ResizeAlg};
use nicegal_core::imaging::{self, Raster};

fn measure(name: &str, mut operation: impl FnMut() -> Result<Vec<u8>>) -> Result<()> {
    for _ in 0..10 {
        black_box(operation()?);
    }
    let mut samples = Vec::new();
    let checksum = operation()?.iter().fold(0_u64, |hash, byte| {
        hash.wrapping_mul(31).wrapping_add(u64::from(*byte))
    });
    for _ in 0..9 {
        let start = Instant::now();
        let mut iterations = 0_u64;
        while start.elapsed() < Duration::from_millis(400) {
            let output = black_box(operation()?);
            black_box(output);
            iterations += 1;
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{name},{:.6},{:.6},{:.6},{}",
        samples[4],
        samples[0],
        samples[8],
        black_box(checksum)
    );
    Ok(())
}

fn pixels(width: u32, height: u32, channels: usize) -> Vec<u8> {
    let mut state = 123456789_u32;
    (0..width as usize * height as usize * channels)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect()
}

fn main() -> Result<()> {
    println!("case,median_ms,min_ms,max_ms,checksum");
    for (width, height, target_width, target_height) in
        [(4000, 3000, 512, 384), (1920, 1080, 256, 144)]
    {
        for channels in [3, 4] {
            let data = pixels(width, height, channels);
            let raster = if channels == 3 {
                Raster::Rgb {
                    width,
                    height,
                    pixels: data,
                }
            } else {
                Raster::Rgba {
                    width,
                    height,
                    pixels: data,
                }
            };
            measure(
                &format!(
                    "thumbnail_{channels}ch_{width}x{height}_to_{target_width}x{target_height}"
                ),
                || {
                    let resized = imaging::resize(black_box(&raster), target_width, target_height)?;
                    Ok(match resized {
                        Raster::Rgb { pixels, .. } | Raster::Rgba { pixels, .. } => pixels,
                    })
                },
            )?;
        }
    }
    let source = image::RgbImage::from_raw(4000, 3000, pixels(4000, 3000, 3)).unwrap();
    for (name, filter) in [
        ("bilinear", FilterType::Bilinear),
        ("bicubic", FilterType::CatmullRom),
    ] {
        measure(&format!("model_{name}_4000x3000_to_224x224"), || {
            Ok(imaging::resize_rgb_bytes(
                black_box(&source),
                224,
                224,
                ResizeAlg::Convolution(filter),
            )?
            .into_raw())
        })?;
    }
    let source = image::RgbImage::from_raw(128, 96, pixels(128, 96, 3)).unwrap();
    measure("model_float_bicubic_128x96_to_224x224", || {
        Ok(imaging::resize_rgb(
            black_box(source.clone()),
            224,
            224,
            image::imageops::FilterType::CatmullRom,
        )?
        .into_raw())
    })?;
    Ok(())
}
