//! Rank text descriptions against one image using locally cached image/text models.
//!
//! Usage: cargo run --example rank_image_text -- IMAGE TEXT_FILE
//! TEXT_FILE contains one candidate description per line. No model is downloaded.
//! On Windows, set NICEGAL_VALIDATION_ORT to a bundled onnxruntime.dll and add its
//! directory to PATH, as for the other validation examples.
//! Set NICEGAL_RANK_MODEL to one model ID to run only that model.

use anyhow::{Context, Result};
use nicegal_core::embedding::{
    ImageEmbedder, ImageEmbedderOptions, ImageEmbeddingModel, ImageQueryEmbedder,
    ImageQueryEmbedderOptions,
};
use nicegal_core::runtime::{self, ExecutionProvider};
use std::{fs, path::Path};

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let norm = |values: &[f32]| {
        values
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    dot / (norm(left) * norm(right))
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let image_path = args.next().context("pass an image path")?;
    let text_path = args.next().context("pass a newline-delimited text file")?;
    anyhow::ensure!(args.next().is_none(), "expected exactly two arguments");
    let image_bytes = fs::read(&image_path)?;
    let descriptions = fs::read_to_string(&text_path)?;
    let descriptions: Vec<&str> = descriptions
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    anyhow::ensure!(!descriptions.is_empty(), "text file has no descriptions");

    if let Some(path) = std::env::var_os("NICEGAL_VALIDATION_ORT") {
        runtime::initialize_from_dylib(Path::new(&path))?;
    } else {
        runtime::initialize_bundled_runtime(ExecutionProvider::Cpu)?;
    }

    let selected_model = std::env::var("NICEGAL_RANK_MODEL").ok();
    if let Some(id) = selected_model.as_deref() {
        let _: ImageEmbeddingModel = id.parse()?;
    }
    for model in ImageEmbeddingModel::ALL {
        if selected_model.as_deref().is_some_and(|id| id != model.id()) {
            continue;
        }
        if !model.supports_text_queries() {
            println!("{}: image-only model; no text encoder", model.name());
            continue;
        }
        let Some(text_encoder) = ImageQueryEmbedder::load_cached(&ImageQueryEmbedderOptions {
            model,
            ..Default::default()
        })?
        else {
            println!("{}: text encoder is not cached", model.name());
            continue;
        };
        let Some(image_encoder) = ImageEmbedder::load_cached(&ImageEmbedderOptions {
            model,
            ..Default::default()
        })?
        else {
            println!("{}: image encoder is not cached", model.name());
            continue;
        };
        let image = image_encoder.decode_image(&image_bytes)?;
        let image_vector = image_encoder.embed_raster(image)?;
        let mut scored = descriptions
            .iter()
            .map(|description| {
                let text_vector = text_encoder.embed_query(description)?;
                Ok((cosine(&image_vector, &text_vector), *description))
            })
            .collect::<Result<Vec<_>>>()?;
        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
        println!("\n{} ({} candidates):", model.name(), scored.len());
        for (score, description) in scored {
            println!("{score:.5}\t{description}");
        }
    }
    Ok(())
}
