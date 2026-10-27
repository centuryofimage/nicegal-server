use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use fastembed::{
    EmbeddingGemma2, EmbeddingGemma2InitOptionsUserDefined, EmbeddingGemma2Sessions,
    EmbeddingModel, TextEmbedding, TextInitOptions,
};

use super::image::ImageEmbeddingModel;
use crate::runtime::{self, ExecutionProvider, RuntimeOptions};

/// FastEmbed's mutable ONNX session, protected for sharing by the facade.
pub(super) struct FastEmbedBackend {
    inner: Inner,
}

enum Inner {
    Text(Box<Mutex<TextEmbedding>>),
    /// EmbeddingGemma 2's text backbone alone, without its encoders.
    Gemma(Box<Mutex<EmbeddingGemma2>>),
}

impl FastEmbedBackend {
    fn text(backend: TextEmbedding) -> Self {
        Self {
            inner: Inner::Text(Box::new(Mutex::new(backend))),
        }
    }
}

impl FastEmbedBackend {
    /// Load onto `runtime_options.execution_provider`, falling back through
    /// [`runtime::with_fallback`] exactly like the OCR model families so a GPU that cannot
    /// actually compile a session doesn't take embedding down with it.
    pub(super) fn load_bge(
        runtime_options: RuntimeOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let cache_dir = crate::hub::cache_dir();
        if !cached_only {
            fetch_bge_files(false, progress)?;
        }
        let intra_threads = runtime_options.intra_threads;
        let (backend, execution_provider) = runtime::with_fallback(runtime_options, |provider| {
            let configured = runtime::configure_provider(provider, intra_threads)?;
            let init = TextInitOptions::new(BGE)
                .with_cache_dir(cache_dir.clone())
                .with_show_download_progress(true)
                .with_environment(runtime::environment()?)
                .with_execution_providers(vec![configured.dispatch])
                .with_intra_threads(configured.intra_threads.get());
            let loaded = TextEmbedding::try_new_cached(init);
            loaded.with_context(|| {
                format!(
                    "loading {} from {} on the {provider} execution provider",
                    super::TextEmbedder::MODEL_ID,
                    cache_dir.display()
                )
            })
        })?;
        let Some(backend) = backend else {
            return Ok(None);
        };
        Ok(Some((Self::text(backend), cache_dir, execution_provider)))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(model = %model, cached_only))]
    pub(super) fn load_local_image_query(
        model: ImageEmbeddingModel,
        options: RuntimeOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let directory = {
            let _span = tracing::debug_span!("image_query_cache_validation").entered();
            model.validated_model_directory(cached_only, progress)?
        };
        let Some(super::image::ModelDirectory { path, .. }) = directory else {
            return Ok(None);
        };
        let (backend, provider) = runtime::with_fallback(options, |provider| {
            let _span = tracing::debug_span!("image_query_provider_attempt", %provider).entered();
            let configured = runtime::configure_provider(provider, options.intra_threads)?;
            let files = {
                let _span = tracing::debug_span!("image_query_tokenizer_files_read").entered();
                fastembed::TokenizerFiles {
                    tokenizer_file: std::fs::read(path.join("tokenizer.json"))?,
                    tokenizer_config_file: std::fs::read(path.join("tokenizer_config.json"))?,
                    special_tokens_map_file: std::fs::read(path.join("special_tokens_map.json"))?,
                    config_file: std::fs::read(path.join("config.json"))?,
                }
            };
            let mut init = fastembed::InitOptionsUserDefined::new()
                .with_max_length(model.context_length())
                .with_environment(runtime::environment()?)
                .with_execution_providers(vec![configured.dispatch])
                .with_intra_threads(configured.intra_threads.get());
            if model.fixed_batch().is_some() {
                init = init.with_dimension_override("batch", 1);
            }
            TextEmbedding::try_new_from_path(path.join("text.onnx"), files, init)
                .context("loading local paired text ONNX encoder")
        })?;
        Ok(Some((Self::text(backend), path, provider)))
    }

    pub(super) fn load_deepghs_image_query(
        model: ImageEmbeddingModel,
        options: RuntimeOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let Some((text, tokenizer)) = model.deepghs_query_files(cached_only, progress)? else {
            return Ok(None);
        };
        let (backend, provider) = runtime::with_fallback(options, |provider| {
            let configured = runtime::configure_provider(provider, options.intra_threads)?;
            TextEmbedding::try_new_from_deepghs_path(
                &text,
                &tokenizer,
                fastembed::InitOptionsUserDefined::new()
                    .with_max_length(model.context_length())
                    .with_environment(runtime::environment()?)
                    .with_execution_providers(vec![configured.dispatch])
                    .with_intra_threads(configured.intra_threads.get()),
            )
            .context("loading DeepGHS paired text ONNX encoder")
        })?;
        Ok(Some((
            Self::text(backend),
            crate::hub::cache_dir(),
            provider,
        )))
    }

    /// The EmbeddingGemma 2 text backbone, for queries in its image space.
    pub(super) fn load_embedding_gemma2_image_query(
        model: ImageEmbeddingModel,
        options: RuntimeOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(Self, PathBuf, ExecutionProvider)>> {
        let Some(directory) = model.embedding_gemma2_directory(cached_only, progress)? else {
            return Ok(None);
        };
        let tokenizer = std::fs::read(directory.join("tokenizer.json"))
            .context("reading the EmbeddingGemma 2 tokenizer")?;
        let (backend, provider) = runtime::with_fallback(options, |provider| {
            let configured = runtime::configure_provider(provider, options.intra_threads)?;
            let init = EmbeddingGemma2InitOptionsUserDefined::new()
                .with_environment(runtime::environment()?)
                .with_execution_providers(vec![configured.dispatch])
                .with_intra_threads(configured.intra_threads.get());
            let text = EmbeddingGemma2::session_builder(&init)?
                .commit_from_file(super::image::embedding_gemma2_graph(
                    &directory, "model", provider,
                ))
                .context("loading the EmbeddingGemma 2 text backbone")?;
            EmbeddingGemma2::try_new_from_sessions(
                EmbeddingGemma2Sessions::new(text),
                &tokenizer,
                init.settings,
            )
            .context("preparing EmbeddingGemma 2")
        })?;
        Ok(Some((
            Self {
                inner: Inner::Gemma(Box::new(Mutex::new(backend))),
            },
            directory,
            provider,
        )))
    }

    pub(super) fn embed(&self, texts: &[&str], max_batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let poisoned = || anyhow::anyhow!("embedding model lock was poisoned");
        match &self.inner {
            Inner::Text(backend) => backend
                .lock()
                .map_err(|_| poisoned())?
                .embed(texts, Some(max_batch_size))
                .context("running FastEmbed inference"),
            Inner::Gemma(backend) => backend
                .lock()
                .map_err(|_| poisoned())?
                .embed_text(texts, Some(max_batch_size))
                .context("running EmbeddingGemma 2 inference"),
        }
    }
}

/// The text model behind [`super::TextEmbedder`].
const BGE: EmbeddingModel = EmbeddingModel::BGESmallENV15;

/// Resolve every file FastEmbed reads for BGE without building a session: downloading the
/// missing ones, or with `cached_only` reporting whether all are already cached.
pub(super) fn fetch_bge_files(
    cached_only: bool,
    progress: &dyn crate::hub::DownloadObserver,
) -> Result<bool> {
    let info = TextEmbedding::get_model_info(&BGE)?;
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
        let source = crate::hub::ModelSource {
            model_id: info.model_code.clone(),
            revision: None,
            filename: filename.to_owned(),
        };
        if source.resolve(cached_only, progress)?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}
