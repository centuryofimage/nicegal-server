//! Compare native DeepGHS loading with Python ONNX Runtime references on testdata/catcopy.
use anyhow::{Context, Result, ensure};
use nicegal_core::embedding::{ImageEmbedder, ImageEmbedderOptions, ImageEmbeddingModel};
use nicegal_core::hub::ModelSource;
use nicegal_core::runtime::{self, ExecutionProvider};
mod support;
use std::{fs, path::Path};
use support::{ImageCase, TextCase, compare, read_cases};

const REVISION: &str = "03aa79c8a4a6c41e06ca87aa6e44fee563b2491d";

fn cached_file(model: ImageEmbeddingModel, file: &str) -> Result<std::path::PathBuf> {
    let subdirectory = model
        .id()
        .strip_prefix("deepghs/siglip_beta/")
        .context("model is not a DeepGHS checkpoint")?;
    ModelSource {
        model_id: "deepghs/siglip_beta".into(),
        revision: Some(REVISION.into()),
        filename: format!("{subdirectory}/{file}"),
    }
    .cached()
    .with_context(|| format!("{file} has not been cached"))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let id = args.next().context("pass DeepGHS model ID")?;
    let model: ImageEmbeddingModel = id.parse()?;
    ensure!(model.is_deepghs(), "expected a DeepGHS checkpoint");
    let fixture_dir = args.next().context("pass fixture directory")?;
    let fixture_dir = Path::new(&fixture_dir);
    if let Some(path) = std::env::var_os("NICEGAL_VALIDATION_ORT") {
        runtime::initialize_from_dylib(Path::new(&path))?;
    } else {
        runtime::initialize_bundled_runtime(ExecutionProvider::Cpu)?;
    }

    let mut text = fastembed::TextEmbedding::try_new_from_deepghs_path(
        cached_file(model, "text_encode.onnx")?,
        cached_file(model, "tokenizer.json")?,
        fastembed::InitOptionsUserDefined::new()
            .with_max_length(model.context_length())
            .with_intra_threads(4),
    )?;
    let mut text_error = 0.0_f32;
    let mut text_cosine = 1.0_f32;
    let text_cases = read_cases::<TextCase>(fixture_dir.join("text-tests.json"))?;
    for case in &text_cases {
        let prompt = case.text.as_str();
        let encoded = text
            .tokenizer
            .encode(prompt, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        ensure!(encoded.get_ids() == case.ids, "tokenizer IDs differ");
        let actual = text.embed([prompt], Some(1))?.remove(0);
        let (error, cosine) = compare(&actual, &case.embedding)?;
        ensure!(cosine > 0.9999, "text vector differs: {error}, {cosine}");
        text_error = text_error.max(error);
        text_cosine = text_cosine.min(cosine);
    }
    drop(text);

    let image = ImageEmbedder::load_cached(&ImageEmbedderOptions {
        model,
        ..Default::default()
    })?
    .context("DeepGHS image checkpoint is not cached")?;
    let mut image_error = 0.0_f32;
    let mut image_cosine = 1.0_f32;
    let image_cases = read_cases::<ImageCase>(fixture_dir.join("image-tests.json"))?;
    for case in &image_cases {
        let path = case.path.as_path();
        ensure!(
            path.components().any(|part| part.as_os_str() == "catcopy"),
            "fixture is outside catcopy"
        );
        let raster = image.decode_image(&fs::read(path)?)?;
        let actual = image.embed_raster(raster)?;
        let (error, cosine) = compare(&actual, &case.embedding)?;
        ensure!(cosine > 0.999, "image vector differs: {error}, {cosine}");
        image_error = image_error.max(error);
        image_cosine = image_cosine.min(cosine);
    }
    println!(
        "{}",
        serde_json::json!({"modelId": id, "textCases": text_cases.len(),
        "textMaxAbsoluteError": text_error, "textMinimumCosine": text_cosine,
        "imageCases": image_cases.len(), "imageMaxAbsoluteError": image_error,
        "imageMinimumCosine": image_cosine})
    );
    Ok(())
}
