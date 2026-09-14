//! Compare native preprocessing and inference with public-corpus Python ORT fixtures.
use anyhow::{Context, Result, ensure};
use nicegal_core::{
    embedding::{ImageEmbedder, ImageEmbedderOptions, ImageEmbeddingModel},
    runtime::{self, ExecutionProvider, RuntimeOptions},
};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("pass image-tests.json")?;
    let provider = std::env::args()
        .nth(2)
        .unwrap_or("cpu".into())
        .parse::<ExecutionProvider>()?;
    if let Some(path) = std::env::var_os("NICEGAL_VALIDATION_ORT") {
        runtime::initialize_from_dylib(std::path::Path::new(&path))?;
    } else {
        runtime::initialize_bundled_runtime(provider)?;
    }
    let model = ImageEmbedder::load_cached(&ImageEmbedderOptions {
        model: ImageEmbeddingModel::DinoV3B16,
        max_batch_size: std::env::var("DINO_VALIDATION_BATCH")
            .unwrap_or("8".into())
            .parse()?,
        runtime: RuntimeOptions {
            execution_provider: provider,
            ..Default::default()
        },
    })?
    .context("DINOv3 files must already be downloaded")?;
    let cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut minimum_cosine = 1.0_f64;
    let mut tensors = Vec::new();
    let mut singles = Vec::new();
    for case in &cases {
        let raster = model.decode_image(&std::fs::read(
            case["path"].as_str().context("image path")?,
        )?)?;
        let rgb = image::RgbImage::from_raw(
            raster.width(),
            raster.height(),
            raster.clone().into_rgb_bytes(),
        )
        .context("image pixels")?;
        tensors.push(model.preprocess_image(rgb)?);
        let actual = model.embed_raster(raster)?;
        let expected: Vec<f32> = serde_json::from_value(case["embedding"].clone())?;
        ensure!(
            actual.len() == 768 && expected.len() == 768,
            "wrong dimensions"
        );
        ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite vector");
        let cosine: f64 = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        ensure!(cosine > 0.9999, "native/Python cosine {cosine}");
        minimum_cosine = minimum_cosine.min(cosine);
        singles.push(actual);
    }
    let batch = tensors
        .chunks(model.max_batch_size())
        .map(|chunk| model.embed_preprocessed_images(chunk.to_vec()))
        .collect::<Result<Vec<_>>>()?
        .concat();
    for (actual, expected) in batch.iter().zip(&singles) {
        ensure!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 2e-4),
            "batch/single mismatch"
        );
    }
    println!(
        "{} images, 768 dimensions, minimum cosine {minimum_cosine}, provider {}",
        cases.len(),
        model.execution_provider()
    );
    Ok(())
}
