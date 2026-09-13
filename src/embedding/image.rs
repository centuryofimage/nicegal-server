use std::fmt;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use fastembed::{
    ImageEmbedding, ImageEmbeddingModel as FastEmbedImageModel, ImageInitOptions, ImagePreprocessor,
};
use image::{DynamicImage, RgbImage};
use ndarray::Array3;
use tracing::{info, instrument};

use crate::runtime::{self, ExecutionProvider, RuntimeOptions};

use super::fastembed::{FastEmbedBackend, image_query_model_to_fastembed};

/// An image embedding model supported by this build.
///
/// The identifier names the shared image/text coordinate space rather than FastEmbed's
/// encoder-specific repositories. A future text-query encoder must use this same identifier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ImageEmbeddingModel {
    /// OpenAI CLIP ViT-B/32, served by FastEmbed's Qdrant ONNX exports.
    #[default]
    ClipVitB32,
}

impl ImageEmbeddingModel {
    /// Stable identifier for the compatible image/text embedding pair.
    pub const fn id(self) -> &'static str {
        match self {
            Self::ClipVitB32 => "Qdrant/clip-ViT-B-32",
        }
    }

    /// Width of every vector produced by this model.
    pub const fn dimensions(self) -> usize {
        match self {
            Self::ClipVitB32 => 512,
        }
    }

    /// Model-specific database filename derived from the complete publisher/model slug.
    pub fn database_file_name(self) -> String {
        let mut stem = String::with_capacity(self.id().len() + 3);
        let mut separator = false;
        for character in self.id().chars() {
            if character.is_ascii_alphanumeric() {
                if separator && !stem.is_empty() {
                    stem.push('-');
                }
                stem.push(character.to_ascii_lowercase());
                separator = false;
            } else {
                separator = true;
            }
        }
        stem.push_str(".db");
        stem
    }
}

impl fmt::Display for ImageEmbeddingModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.id())
    }
}

/// How an [`ImageEmbedder`] is built.
#[derive(Debug, Clone)]
pub struct ImageEmbedderOptions {
    pub model: ImageEmbeddingModel,
    /// Largest set of normalized image tensors submitted in one forward pass.
    pub max_batch_size: usize,
    pub runtime: RuntimeOptions,
}

impl Default for ImageEmbedderOptions {
    fn default() -> Self {
        Self {
            model: ImageEmbeddingModel::default(),
            // Tensor dimensions come from the selected model's preprocessor config, so this can
            // be tuned for accelerator throughput without retaining full-resolution source images.
            max_batch_size: 8,
            runtime: RuntimeOptions::default(),
        }
    }
}

/// A loaded image embedding model.
pub struct ImageEmbedder {
    backend: Mutex<ImageEmbedding>,
    preprocessor: ImagePreprocessor,
    model: ImageEmbeddingModel,
    max_batch_size: usize,
    execution_provider: ExecutionProvider,
}

impl ImageEmbedder {
    #[instrument(name = "image_embedder_load", skip_all, fields(model = %options.model))]
    pub fn load(options: &ImageEmbedderOptions) -> Result<Self> {
        Self::load_with_cache_policy(options, false)?
            .context("image model loader returned no model")
    }

    pub fn load_cached(options: &ImageEmbedderOptions) -> Result<Option<Self>> {
        Self::load_with_cache_policy(options, true)
    }

    fn load_with_cache_policy(
        options: &ImageEmbedderOptions,
        cached_only: bool,
    ) -> Result<Option<Self>> {
        if options.max_batch_size == 0 {
            bail!("image embedding batch size must be greater than zero");
        }
        let cache_dir = crate::hub::cache_dir();
        let intra_threads = options.runtime.intra_threads;
        let (backend, execution_provider) = runtime::with_fallback(options.runtime, |provider| {
            let configured = runtime::configure_provider(provider, intra_threads)?;
            let init = ImageInitOptions::new(model_to_fastembed(options.model))
                .with_cache_dir(cache_dir.clone())
                .with_show_download_progress(true)
                .with_execution_providers(vec![configured.dispatch])
                .with_intra_threads(configured.intra_threads.get());
            let loaded = if cached_only {
                ImageEmbedding::try_new_cached(init)
            } else {
                ImageEmbedding::try_new(init).map(Some)
            };
            loaded.with_context(|| {
                format!(
                    "loading {} from {} on the {provider} execution provider",
                    options.model,
                    cache_dir.display()
                )
            })
        })?;
        let Some(backend) = backend else {
            return Ok(None);
        };
        let preprocessor = backend.preprocessor();
        let embedder = Self {
            backend: Mutex::new(backend),
            preprocessor,
            model: options.model,
            max_batch_size: options.max_batch_size,
            execution_provider,
        };
        info!(
            dimensions = embedder.dimensions(),
            max_batch = embedder.max_batch_size,
            execution_provider = %embedder.execution_provider,
            cache = %cache_dir.display(),
            "loaded the image embedding model"
        );
        Ok(Some(embedder))
    }

    pub fn model(&self) -> ImageEmbeddingModel {
        self.model
    }

    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    #[instrument(
        name = "preprocess_image",
        level = "debug",
        skip_all,
        fields(width = image.width(), height = image.height())
    )]
    pub fn preprocess_image(&self, image: RgbImage) -> Result<Array3<f32>> {
        self.preprocessor
            .preprocess(DynamicImage::ImageRgb8(image))
            .context("preprocessing image for FastEmbed")
    }

    /// Encode a decoded external image without adding it to the catalog.
    pub fn embed_raster(&self, raster: crate::imaging::Raster) -> Result<Vec<f32>> {
        let image = RgbImage::from_raw(raster.width(), raster.height(), raster.into_rgb_bytes())
            .context("invalid decoded image dimensions")?;
        let tensor = self.preprocess_image(image)?;
        self.embed_preprocessed_images(vec![tensor])?
            .pop()
            .context("missing image vector")
    }

    #[instrument(
        name = "embed_preprocessed_images",
        level = "debug",
        skip_all,
        fields(batch = images.len())
    )]
    pub fn embed_preprocessed_images(&self, images: Vec<Array3<f32>>) -> Result<Vec<Vec<f32>>> {
        if images.len() > self.max_batch_size {
            bail!(
                "image batch of {} exceeds the {} the {} backend accepts",
                images.len(),
                self.max_batch_size,
                self.model
            );
        }
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let expected = images.len();
        let mut backend = self
            .backend
            .lock()
            .map_err(|_| anyhow::anyhow!("image embedding model lock was poisoned"))?;
        let vectors = backend
            .embed_preprocessed(images)
            .context("running FastEmbed image inference")?;
        if vectors.len() != expected {
            bail!(
                "{} returned {} vectors for {expected} images",
                self.model,
                vectors.len()
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

fn model_to_fastembed(model: ImageEmbeddingModel) -> FastEmbedImageModel {
    match model {
        ImageEmbeddingModel::ClipVitB32 => FastEmbedImageModel::ClipVitB32,
    }
}

/// How an [`ImageQueryEmbedder`] is built.
#[derive(Debug, Clone)]
pub struct ImageQueryEmbedderOptions {
    /// The image embedding model whose coordinate space query vectors must land in. The text
    /// encoder is the paired half of that model, so it is selected with it and only changes
    /// through a restart, exactly like the image encoder.
    pub model: ImageEmbeddingModel,
    /// Longest input embedded, in bytes of UTF-8. Longer text is truncated on a character
    /// boundary first; the backend's tokenizer then truncates to the model's own context window.
    pub max_input_bytes: usize,
    /// The execution provider to compile the ONNX session for, with the same
    /// request/fallback/thread-budget contract [`crate::runtime::load_sessions`] gives OCR.
    ///
    /// CPU by default and in practice: one short forward pass per search, where a GPU upload
    /// costs more than the pass itself, and the accelerator is wanted by indexing and OCR.
    pub runtime: RuntimeOptions,
}

impl Default for ImageQueryEmbedderOptions {
    fn default() -> Self {
        Self {
            model: ImageEmbeddingModel::default(),
            // Matches the search API's query cap; the tokenizer is the tighter limit that
            // actually applies.
            max_input_bytes: 4096,
            runtime: RuntimeOptions::default(),
        }
    }
}

/// The text half of an image embedding pair: embeds search queries into the image model's
/// coordinate space.
///
/// CLIP-style models are two encoders trained to share one space, so this carries the same
/// [`ImageEmbeddingModel`] as the [`ImageEmbedder`] it answers. It is deliberately not a
/// [`super::TextEmbedder`]: the OCR engine's model catalog and the image engine's model catalog
/// are separate selections with separate databases, and only the image engine's queries may land
/// in its space.
pub struct ImageQueryEmbedder {
    backend: FastEmbedBackend,
    model: ImageEmbeddingModel,
    max_input_bytes: usize,
    execution_provider: ExecutionProvider,
}

impl ImageQueryEmbedder {
    #[instrument(name = "image_query_embedder_load", skip_all, fields(model = %options.model))]
    pub fn load(options: &ImageQueryEmbedderOptions) -> Result<Self> {
        Self::load_with_cache_policy(options, false)?.context("model loader returned no model")
    }

    /// Load the paired text encoder strictly from cached files, without a network client.
    pub fn load_cached(options: &ImageQueryEmbedderOptions) -> Result<Option<Self>> {
        Self::load_with_cache_policy(options, true)
    }

    fn load_with_cache_policy(
        options: &ImageQueryEmbedderOptions,
        cached_only: bool,
    ) -> Result<Option<Self>> {
        if options.max_input_bytes == 0 {
            bail!("image query embedding input limit must be greater than zero");
        }
        let Some((backend, cache_dir, execution_provider)) =
            FastEmbedBackend::load_with_cache_policy(
                image_query_model_to_fastembed(options.model),
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
            execution_provider,
        };
        info!(
            dimensions = embedder.dimensions(),
            execution_provider = %embedder.execution_provider,
            cache = %cache_dir.display(),
            "loaded the image query embedding model"
        );
        Ok(Some(embedder))
    }

    /// The image embedding model whose space this embedder's vectors belong to, matching the
    /// [`ImageEmbedder`] that produced the stored image vectors they are compared against.
    pub fn model(&self) -> ImageEmbeddingModel {
        self.model
    }

    /// The width of every vector this embedder produces, equal to the image model's width.
    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    /// The execution provider the backend actually compiled onto, which is not the requested one
    /// when `runtime`'s fallback chain ran.
    pub fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// Embed one search query. Queries are the only input this encoder ever sees, so this is its
    /// whole interface; document embedding belongs to the OCR engine's [`super::TextEmbedder`].
    #[instrument(name = "embed_image_query", level = "debug", skip_all, fields(bytes = text.len()))]
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            bail!("cannot embed an empty query");
        }
        let text = super::truncate_on_boundary(text, self.max_input_bytes);
        let mut vectors = self
            .backend
            .embed(&[text], 1)
            .with_context(|| format!("running the {} image query model", self.model))?;
        if vectors.len() != 1 {
            bail!(
                "{} returned {} vectors for one query",
                self.model,
                vectors.len()
            );
        }
        let vector = vectors
            .pop()
            .context("the embedding backend returned no vector for the query")?;
        if vector.len() != self.dimensions() {
            bail!(
                "{} returned a {}-element vector, expected {}",
                self.model,
                vector.len(),
                self.dimensions()
            );
        }
        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_model_metadata_is_stable() {
        let model = ImageEmbeddingModel::default();
        assert_eq!(model, ImageEmbeddingModel::ClipVitB32);
        assert_eq!(model.id(), "Qdrant/clip-ViT-B-32");
        assert_eq!(model.dimensions(), 512);
        assert_eq!(model.database_file_name(), "qdrant-clip-vit-b-32.db");
    }

    #[test]
    fn image_query_defaults_stay_with_the_image_model_on_cpu() {
        let options = ImageQueryEmbedderOptions::default();
        assert_eq!(options.model, ImageEmbeddingModel::ClipVitB32);
        assert_eq!(options.max_input_bytes, 4096);
        assert_eq!(options.runtime.execution_provider, ExecutionProvider::Cpu);
    }
}
