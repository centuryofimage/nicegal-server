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
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let cache_dir = crate::hub::cache_dir();
        if !cached_only {
            let info = TextEmbedding::get_model_info(&model)?;
            for filename in [
                info.model_file.as_str(),
                "tokenizer.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ]
            .into_iter()
            .chain(info.additional_files.iter().map(String::as_str))
            {
                crate::hub::ModelSource {
                    model_id: info.model_code.clone(),
                    revision: None,
                    filename: filename.to_owned(),
                }
                .get_sync_with_progress(progress)?;
            }
        }
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
            let loaded = TextEmbedding::try_new_cached(init);
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

    pub(super) fn load_local_image_query(
        model: ImageEmbeddingModel,
        options: RuntimeOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let Some(path) = model.validated_model_directory(cached_only, progress)? else {
            return Ok(None);
        };
        let (backend, provider) = runtime::with_fallback(options, |provider| {
            let configured = runtime::configure_provider(provider, options.intra_threads)?;
            let files = fastembed::TokenizerFiles {
                tokenizer_file: std::fs::read(path.join("tokenizer.json"))?,
                tokenizer_config_file: std::fs::read(path.join("tokenizer_config.json"))?,
                special_tokens_map_file: std::fs::read(path.join("special_tokens_map.json"))?,
                config_file: std::fs::read(path.join("config.json"))?,
            };
            TextEmbedding::try_new_from_path(
                path.join("text.onnx"),
                files,
                fastembed::InitOptionsUserDefined::new()
                    .with_max_length(model.context_length())
                    .with_execution_providers(vec![configured.dispatch])
                    .with_intra_threads(configured.intra_threads.get()),
            )
            .context("loading local paired text ONNX encoder")
        })?;
        Ok(Some((
            Self {
                inner: Mutex::new(backend),
            },
            path,
            provider,
        )))
    }

    pub(super) fn load_deepghs_image_query(
        model: ImageEmbeddingModel,
        options: RuntimeOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let text = model.deepghs_file("text_encode.onnx", cached_only, progress)?;
        let tokenizer = model.deepghs_file("tokenizer.json", cached_only, progress)?;
        let meta = model.deepghs_file("meta.json", cached_only, progress)?;
        let (Some(text), Some(tokenizer), Some(meta)) = (text, tokenizer, meta) else {
            return Ok(None);
        };
        model.validate_deepghs_meta(&meta)?;
        let (backend, provider) = runtime::with_fallback(options, |provider| {
            let configured = runtime::configure_provider(provider, options.intra_threads)?;
            TextEmbedding::try_new_from_deepghs_path(
                &text,
                &tokenizer,
                fastembed::InitOptionsUserDefined::new()
                    .with_max_length(model.context_length())
                    .with_execution_providers(vec![configured.dispatch])
                    .with_intra_threads(configured.intra_threads.get()),
            )
            .context("loading DeepGHS paired text ONNX encoder")
        })?;
        Ok(Some((
            Self {
                inner: Mutex::new(backend),
            },
            crate::hub::cache_dir(),
            provider,
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
