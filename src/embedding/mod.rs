//! Text embedding for vector search.

mod fastembed;
mod image;
mod model;

use anyhow::{Context, Result, bail};
use tracing::{info, instrument};

use crate::runtime::{ExecutionProvider, RuntimeOptions};
use fastembed::{FastEmbedBackend, ocr_model_to_fastembed};
pub use image::{
    ImageEmbedder, ImageEmbedderOptions, ImageEmbeddingModel, ImageQueryEmbedder,
    ImageQueryEmbedderOptions,
};
pub use model::{ParseTextEmbeddingModelError, TextEmbeddingModel};

/// How a [`TextEmbedder`] is built.
#[derive(Debug, Clone)]
pub struct TextEmbedderOptions {
    /// The supported model to load.
    pub model: TextEmbeddingModel,
    /// Longest input the backend accepts, in bytes of UTF-8. Longer text is truncated on a
    /// character boundary before embedding.
    pub max_input_bytes: usize,
    /// Largest batch [`TextEmbedder::embed_documents`] should be given. A caller may ask for less;
    /// asking for more is clamped, because the backend sizes its buffers from this.
    ///
    /// 64 rather than something larger: attention memory is `batch * heads * seq^2`, so cost grows
    /// with batch size *and* with the longest row in the batch (ONNX pads every row to it). Live
    /// benchmarking (`benches/text_embed_index.rs`) against a corpus mixing short OCR lines with
    /// occasional near-cap rows plateaued throughput by batch 32-64 on both CPU and DirectML, then
    /// DirectML failed outright at batch 256 with a device-level error, and CPU failed at batch
    /// 1024 trying to allocate a single ~12 GiB buffer. 64 sits with real margin below both cliffs
    /// while giving up none of the measured throughput, and keeps peak memory modest for the 16 GB
    /// machines this backend also has to run on.
    pub max_batch_size: usize,
    /// The execution provider to compile the ONNX session for, with the same
    /// request/fallback/thread-budget contract [`crate::runtime::load_sessions`] gives OCR.
    /// `replicas` is ignored: the backend is a single mutex-guarded session regardless of provider.
    pub runtime: RuntimeOptions,
}

impl Default for TextEmbedderOptions {
    fn default() -> Self {
        Self {
            model: TextEmbeddingModel::default(),
            max_input_bytes: 8192,
            max_batch_size: 64,
            runtime: RuntimeOptions::default(),
        }
    }
}

/// A loaded text embedding model.
///
/// The inference backend requires mutable access to its ONNX session. The backend's mutex keeps
/// one process-wide model safe to share between the search and backfill workers; ONNX Runtime
/// supplies the parallelism inside each inference call.
pub struct TextEmbedder {
    backend: FastEmbedBackend,
    model: TextEmbeddingModel,
    max_input_bytes: usize,
    max_batch_size: usize,
    execution_provider: ExecutionProvider,
}

impl TextEmbedder {
    /// Load the model described by `options`.
    ///
    /// The model is downloaded on first use into the standard Hugging Face cache and reused from
    /// there on subsequent starts. ONNX Runtime binaries are provided by the backend's download
    /// feature for now.
    #[instrument(name = "embedder_load", skip_all, fields(model = %options.model))]
    pub fn load(options: &TextEmbedderOptions) -> Result<Self> {
        Self::load_with_cache_policy(options, false)?.context("model loader returned no model")
    }

    /// Load only existing local model files. Never downloads; None means setup is needed.
    pub fn load_cached(options: &TextEmbedderOptions) -> Result<Option<Self>> {
        Self::load_with_cache_policy(options, true)
    }

    fn load_with_cache_policy(
        options: &TextEmbedderOptions,
        cached_only: bool,
    ) -> Result<Option<Self>> {
        if options.max_input_bytes == 0 {
            bail!("embedding input limit must be greater than zero");
        }
        if options.max_batch_size == 0 {
            bail!("embedding batch size must be greater than zero");
        }

        let Some((backend, cache_dir, execution_provider)) =
            FastEmbedBackend::load_with_cache_policy(
                ocr_model_to_fastembed(options.model),
                &options.model.to_string(),
                options.runtime,
                cached_only,
            )?
        else {
            return Ok(None);
        };
        let embedder = Self {
            backend,
            model: options.model,
            max_input_bytes: options.max_input_bytes,
            max_batch_size: options.max_batch_size,
            execution_provider,
        };
        // Announced here rather than at each call site: loading is the one moment every consumer
        // shares, and the shape of the model explains every embedding timing that follows.
        // `model` is not repeated here: the enclosing `embedder_load` span already carries it, and
        // the compact log writer appends every ambient span field to each event inside it.
        info!(
            dimensions = embedder.dimensions(),
            max_batch = embedder.max_batch_size,
            execution_provider = %embedder.execution_provider,
            cache = %cache_dir.display(),
            "loaded the embedding model"
        );
        Ok(Some(embedder))
    }

    /// The typed model that produced this embedder's vectors.
    pub fn model(&self) -> TextEmbeddingModel {
        self.model
    }

    /// The execution provider the backend actually compiled onto, which is not the requested one
    /// when `runtime`'s fallback chain ran.
    pub fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// The width of every vector this embedder produces.
    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    /// Largest batch [`Self::embed_documents`] should be given. Callers clamp to this rather than
    /// choosing a batch size of their own, so swapping the backend re-tunes every caller at once.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Embed one search query.
    ///
    /// Kept separate from [`Self::embed_documents`] because asymmetric models prefix the two
    /// differently; a backend that does not care can forward one to the other.
    #[instrument(name = "embed_query", level = "debug", skip_all, fields(bytes = text.len()))]
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            bail!("cannot embed empty text");
        }
        let text = truncate_on_boundary(text, self.max_input_bytes);
        let mut vectors = self.embed(&[text])?;
        vectors
            .pop()
            .context("embedding backend returned no vector for the query")
    }

    /// Embed a batch of OCR texts, returning one vector per input in the same order.
    ///
    /// Batching is the shape the real backend wants — a single padded forward pass over the whole
    /// batch, parallelised inside ONNX Runtime — so the backfill job is written against it from
    /// the start rather than being retrofitted later. A batch longer than
    /// [`Self::max_batch_size`] is refused rather than silently split, because a caller that has
    /// stopped honouring the limit will otherwise quietly lose the batching it thinks it has.
    #[instrument(
        name = "embed_documents",
        level = "debug",
        skip_all,
        fields(batch = texts.len(), bytes = texts.iter().map(|text| text.len()).sum::<usize>())
    )]
    pub fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.len() > self.max_batch_size {
            bail!(
                "batch of {} exceeds the {} the {} backend accepts",
                texts.len(),
                self.max_batch_size,
                self.model
            );
        }
        let texts: Vec<&str> = texts
            .iter()
            .map(|text| {
                if text.trim().is_empty() {
                    bail!("cannot embed empty text");
                }
                Ok(truncate_on_boundary(text, self.max_input_bytes))
            })
            .collect::<Result<_>>()?;
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.embed(&texts)
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let vectors = self
            .backend
            .embed(texts, self.max_batch_size)
            .with_context(|| format!("running the {} embedding model", self.model))?;
        if vectors.len() != texts.len() {
            bail!(
                "{} returned {} vectors for {} inputs",
                self.model,
                vectors.len(),
                texts.len()
            );
        }
        if let Some(vector) = vectors
            .iter()
            .find(|vector| vector.len() != self.dimensions())
        {
            bail!(
                "{} returned a {}-element vector, expected {}",
                self.model,
                vector.len(),
                self.dimensions()
            );
        }
        Ok(vectors)
    }
}

/// Trim to at most `max_bytes`, never splitting a character.
fn truncate_on_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    fn embedder() -> &'static TextEmbedder {
        static EMBEDDER: LazyLock<TextEmbedder> = LazyLock::new(|| {
            // Tests do not run main(), which selects the bundled runtime in applications.
            #[cfg(windows)]
            {
                let executable = std::env::current_exe().unwrap();
                crate::runtime::initialize_from_dylib(
                    &executable
                        .parent()
                        .unwrap()
                        .join("onnxruntime/directml/onnxruntime.dll"),
                )
                .expect("the bundled CPU runtime initializes");
            }
            TextEmbedder::load(&TextEmbedderOptions::default()).expect("the default embedder loads")
        });
        &EMBEDDER
    }

    fn assert_vectors_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        let greatest_difference = left
            .iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(greatest_difference < 1e-5, "{greatest_difference}");
    }

    #[test]
    fn default_embedder_reports_its_model_metadata() {
        let embedder = embedder();
        assert_eq!(embedder.model(), TextEmbeddingModel::default());
        assert_eq!(embedder.dimensions(), embedder.model().dimensions());
        assert_eq!(
            embedder.embed_query("receipt").unwrap().len(),
            embedder.model().dimensions()
        );
    }

    #[test]
    fn embeddings_depend_on_the_input_and_preserve_batch_order() {
        let embedder = embedder();
        let first = embedder.embed_query("receipt").unwrap();
        let second = embedder.embed_query("something else entirely").unwrap();
        assert_ne!(first, second);

        let batch = embedder
            .embed_documents(&["receipt", "something else entirely"])
            .unwrap();
        assert_vectors_close(&batch[0], &first);
        assert_vectors_close(&batch[1], &second);
    }

    #[test]
    fn bge_vectors_are_unit_length_for_cosine_distance() {
        let norm: f32 = embedder()
            .embed_query("receipt")
            .unwrap()
            .iter()
            .map(|value| value * value)
            .sum();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");
    }

    #[test]
    fn empty_text_is_rejected_rather_than_embedded() {
        let embedder = embedder();
        assert!(embedder.embed_query("   ").is_err());
        assert!(embedder.embed_documents(&["ok", ""]).is_err());
    }

    #[test]
    fn truncation_does_not_split_a_character() {
        // "é" is two bytes, so a naive cut at 3 would split the second one.
        assert_eq!(truncate_on_boundary("éé", 3), "é");
    }
}
