use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use super::image::ImageEmbeddingModel;
use super::model::TextEmbeddingModel;
use crate::runtime::{self, ExecutionProvider, RuntimeOptions};

/// FastEmbed's mutable ONNX session, protected for sharing by the facade.
pub(super) struct FastEmbedBackend {
    inner: Mutex<TextEmbedding>,
}

impl FastEmbedBackend {
    /// Load onto `runtime_options.execution_provider`, falling back through
    /// [`runtime::with_fallback`] exactly like the OCR model families so a GPU that cannot
    /// actually compile a session doesn't take embedding down with it.
    pub(super) fn load_with_cache_policy(
        model: EmbeddingModel,
        model_label: &str,
        runtime_options: RuntimeOptions,
        cached_only: bool,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let cache_dir = crate::hub::cache_dir();
        let intra_threads = runtime_options.intra_threads;
        let (backend, execution_provider) = runtime::with_fallback(runtime_options, |provider| {
            let configured = runtime::configure_provider(provider, intra_threads)?;
            // `EmbeddingModel` is not `Copy` and the fallback chain may build the session more
            // than once, so each attempt takes its own clone.
            let init = TextInitOptions::new(model.clone())
                .with_cache_dir(cache_dir.clone())
                .with_show_download_progress(true)
                .with_execution_providers(vec![configured.dispatch])
                .with_intra_threads(configured.intra_threads.get());
            let loaded = if cached_only {
                TextEmbedding::try_new_cached(init)
            } else {
                TextEmbedding::try_new(init).map(Some)
            };
            loaded.with_context(|| {
                format!(
                    "loading {model_label} from {} on the {provider} execution provider",
                    cache_dir.display()
                )
            })
        })?;
        let Some(backend) = backend else {
            return Ok(None);
        };
        Ok(Some((
            Self {
                inner: Mutex::new(backend),
            },
            cache_dir,
            execution_provider,
        )))
    }

    pub(super) fn embed(&self, texts: &[&str], max_batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut backend = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("embedding model lock was poisoned"))?;
        backend
            .embed(texts, Some(max_batch_size))
            .context("running FastEmbed inference")
    }
}

// Keep the backend's model enum out of the public model catalogs. Every catalog variant must be
// deliberately mapped here before it can be loaded by FastEmbed.
/// The OCR engine's catalog: models whose vectors are stored in the OCR database's spaces.
pub(super) fn ocr_model_to_fastembed(model: TextEmbeddingModel) -> EmbeddingModel {
    match model {
        TextEmbeddingModel::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
    }
}

/// The image engine's catalog: the paired text encoder of each image embedding model, whose
/// vectors must land in that model's coordinate space to be comparable with its image vectors.
pub(super) fn image_query_model_to_fastembed(model: ImageEmbeddingModel) -> EmbeddingModel {
    match model {
        ImageEmbeddingModel::ClipVitB32 => EmbeddingModel::ClipVitB32,
    }
}
