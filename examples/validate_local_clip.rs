//! Validate exported reference fixtures on the non-private corpus through native inference.
use anyhow::{Context, Result, ensure};
use nicegal_core::embedding::{ImageEmbedder, ImageEmbedderOptions, ImageEmbeddingModel};
use nicegal_core::runtime::{self, ExecutionProvider};
mod support;
use std::{fs, path::Path};
use support::{ImageCase, TextCase, compare, read_cases};

fn main() -> Result<()> {
    let id = std::env::args()
        .nth(1)
        .context("pass model ID; NICEGAL_LOCAL_MODELS_DIR must be set")?;
    let model: ImageEmbeddingModel = id.parse()?;
    let root = model.local_directory().context("model must be local")?;
    if let Some(path) = std::env::var_os("NICEGAL_VALIDATION_ORT") {
        runtime::initialize_from_dylib(Path::new(&path))?;
    } else {
        runtime::initialize_bundled_runtime(ExecutionProvider::Cpu)?;
    }
    let text_cases = read_cases::<TextCase>(root.join("tokenizer-tests.json"))?;
    let files = fastembed::TokenizerFiles {
        tokenizer_file: fs::read(root.join("tokenizer.json"))?,
        tokenizer_config_file: fs::read(root.join("tokenizer_config.json"))?,
        special_tokens_map_file: fs::read(root.join("special_tokens_map.json"))?,
        config_file: fs::read(root.join("config.json"))?,
    };
    let mut text = fastembed::TextEmbedding::try_new_from_path(
        root.join("text.onnx"),
        files,
        fastembed::InitOptionsUserDefined::new()
            .with_max_length(model.context_length())
            .with_intra_threads(4),
    )?;
    let mut text_max_error = 0.0_f32;
    let mut text_count = 0;
    for case in &text_cases {
        let prompt = case.text.as_str();
        let encoding = text
            .tokenizer
            .encode(prompt, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        ensure!(
            encoding.get_ids() == case.ids,
            "token IDs differ in case {text_count}"
        );
        let actual = text.embed([prompt], Some(1))?.remove(0);
        let (error, similarity) = compare(&actual, &case.embedding)?;
        ensure!(
            error < 0.0005 && similarity > 0.9999,
            "text vector differs in case {text_count}: {error}, {similarity}"
        );
        text_max_error = text_max_error.max(error);
        text_count += 1;
    }
    drop(text);
    let image = ImageEmbedder::load(&ImageEmbedderOptions {
        model,
        ..Default::default()
    })?;
    let image_cases = read_cases::<ImageCase>(root.join("image-tests.json"))?;
    let mut minimum_cosine = 1.0_f32;
    let mut image_max_error = 0.0_f32;
    let mut image_count = 0;
    for case in &image_cases {
        let path = case.path.as_path();
        // Fixtures are deliberately limited to the non-private test corpus.
        ensure!(
            path.components().any(|part| part.as_os_str() == "catcopy"),
            "fixture is outside catcopy"
        );
        let raster = image.decode_image(&fs::read(path)?)?;
        let rgb =
            image::RgbImage::from_raw(raster.width(), raster.height(), raster.into_rgb_bytes())
                .context("RGB shape")?;
        let pixels = image.preprocess_image(rgb)?;
        if let Some(directory) = std::env::var_os("NICEGAL_VALIDATION_PIXELS_DIR") {
            fs::create_dir_all(&directory)?;
            let bytes: Vec<u8> = pixels
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            fs::write(
                Path::new(&directory).join(format!("{image_count}.bin")),
                bytes,
            )?;
        }
        let actual = image.embed_preprocessed_images(vec![pixels])?.remove(0);
        let (error, similarity) = compare(&actual, &case.embedding)?;
        // Pillow and Rust bicubic filters are not bit-identical; compare semantic vectors.
        ensure!(
            similarity > 0.9999,
            "image vector differs in case {image_count}: {error}, {similarity}"
        );
        minimum_cosine = minimum_cosine.min(similarity);
        image_max_error = image_max_error.max(error);
        image_count += 1;
    }
    println!(
        "{}",
        serde_json::json!({"modelId": id, "textCases": text_count,
        "textMaxAbsoluteError": text_max_error, "imageCases": image_count,
        "imageMaxAbsoluteError": image_max_error, "imageMinimumCosine": minimum_cosine})
    );
    Ok(())
}
