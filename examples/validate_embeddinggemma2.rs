//! Validate an EmbeddingGemma 2 export against Python reference fixtures on the non-private
//! corpus, through the application's decoder, preprocessing, batching and query prefix.
//!
//! Pass the model ID and the fixture directory from model-exports'
//! `export_embeddinggemma2_fixtures.py`. The published export is used unless
//! `NICEGAL_LOCAL_MODELS` lists a local one.
use anyhow::{Context, Result, ensure};
use nicegal_core::embedding::{
    ImageEmbedder, ImageEmbedderOptions, ImageEmbeddingModel, ImageQueryEmbedder,
    ImageQueryEmbedderOptions,
};
use nicegal_core::runtime::{self, ExecutionProvider, RuntimeOptions};
mod support;
use std::{fs, path::PathBuf};
use support::{ImageCase, TextCase, compare, read_cases};

const PROMPT: &str = "task: search result | query: ";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let id = args
        .next()
        .context("pass the model ID and the fixture directory")?;
    let fixtures = PathBuf::from(args.next().context("pass the fixture directory")?);
    let model: ImageEmbeddingModel = id.parse()?;
    ensure!(
        model.is_embedding_gemma2(),
        "expected an EmbeddingGemma 2 model"
    );
    ensure!(
        model.available(),
        "the local export in NICEGAL_LOCAL_MODELS is missing files"
    );
    let provider = std::env::var("NICEGAL_VALIDATION_PROVIDER")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(ExecutionProvider::Cpu);
    match std::env::var_os("NICEGAL_VALIDATION_ORT") {
        Some(path) => runtime::initialize_from_dylib(std::path::Path::new(&path))?,
        None => runtime::initialize_bundled_runtime(provider)?,
    }
    let runtime = RuntimeOptions {
        execution_provider: provider,
        allow_cpu_fallback: false,
        ..Default::default()
    };

    let text_cases = read_cases::<TextCase>(fixtures.join("tokenizer-tests.json"))?;
    let text = ImageQueryEmbedder::load(&ImageQueryEmbedderOptions {
        model,
        runtime,
        ..Default::default()
    })?;
    let mut text_minimum = 1.0_f32;
    let mut queries = Vec::new();
    for (index, case) in text_cases.iter().enumerate() {
        let query = case
            .text
            .strip_prefix(PROMPT)
            .context("fixture query lacks the search prefix")?;
        // Token IDs are compared through the embedding; the fixture must still be a whole sequence.
        ensure!(
            case.ids.first() == Some(&2) && case.ids.last() == Some(&1),
            "fixture {index} is not wrapped in <bos> and <eos>"
        );
        let actual = text.embed_query(query)?;
        let (_, similarity) = compare(&actual, &case.embedding)?;
        ensure!(similarity > 0.9999, "query {index} differs: {similarity}");
        text_minimum = text_minimum.min(similarity);
        queries.push(actual);
    }
    drop(text);

    let image = ImageEmbedder::load(&ImageEmbedderOptions {
        model,
        runtime,
        ..Default::default()
    })?;
    let image_cases = read_cases::<ImageCase>(fixtures.join("image-tests.json"))?;
    let mut similarities = Vec::new();
    let mut cat_first = 0;
    // Full batches and a short final batch, through the application's padding path.
    for cases in image_cases.chunks(image.max_batch_size()) {
        let mut sizes = Vec::new();
        let prepared = cases
            .iter()
            .map(|case| {
                ensure!(
                    case.path
                        .components()
                        .any(|part| part.as_os_str() == "catcopy"),
                    "fixture is outside catcopy"
                );
                let raster = image.decode_image(&fs::read(&case.path)?)?;
                sizes.push((raster.width(), raster.height()));
                let rgb = image::RgbImage::from_raw(
                    raster.width(),
                    raster.height(),
                    raster.into_rgb_bytes(),
                )
                .context("RGB shape")?;
                image.preprocess_image(rgb)
            })
            .collect::<Result<Vec<_>>>()?;
        let vectors = image.embed_preprocessed_images(prepared)?;
        ensure!(
            vectors.len() == cases.len(),
            "padded vectors leaked into output"
        );
        for ((vector, case), size) in vectors.iter().zip(cases).zip(sizes) {
            let (_, similarity) = compare(vector, &case.embedding)?;
            similarities.push((similarity, size));
            // The first five fixtures are the trial benchmark's queries, with "a cat" first.
            let best = queries[..5]
                .iter()
                .enumerate()
                .map(|(index, query)| (index, compare(vector, query).map(|(_, s)| s)))
                .try_fold((0, f32::MIN), |best, (index, score)| {
                    score.map(|score| if score > best.1 { (index, score) } else { best })
                })?;
            cat_first += usize::from(best.0 == 0);
        }
    }
    similarities.sort_by(|a, b| a.0.total_cmp(&b.0));
    let image_minimum = similarities.first().map_or(1.0, |worst| worst.0);
    println!(
        "worst image cosines, with source sizes: {:?}",
        &similarities[..similarities.len().min(5)]
    );
    // Native decoding and resizing can differ from Pillow by one RGB level.
    ensure!(
        image_minimum > 0.9995,
        "image vectors differ: {image_minimum}"
    );
    println!(
        "{}",
        serde_json::json!({"modelId": id, "provider": provider.to_string(),
        "executionProvider": image.execution_provider().to_string(),
        "textCases": text_cases.len(), "textMinimumCosine": text_minimum,
        "imageCases": image_cases.len(), "imageMinimumCosine": image_minimum,
        "catRankedFirst": cat_first})
    );
    Ok(())
}
