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

use super::fastembed::FastEmbedBackend;

/// An image embedding model supported by this build.
///
/// The identifier names the shared image/text coordinate space rather than FastEmbed's
/// encoder-specific repositories. A future text-query encoder must use this same identifier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ImageEmbeddingModel {
    /// OpenAI CLIP ViT-B/32, served by FastEmbed's Qdrant ONNX exports.
    ClipVitB32,
    #[default]
    MetaClip2B32,
    MetaClip2B16,
    SigLip2Base256,
    LaionClipB32,
    SigLipBetaSwinV2Frozen,
    SigLipBetaSwinV2,
    SigLipBetaEva02,
    DinoV3B16,
}

impl ImageEmbeddingModel {
    /// Stable identifier for the compatible image/text embedding pair.
    pub const fn id(self) -> &'static str {
        match self {
            Self::ClipVitB32 => "Qdrant/clip-ViT-B-32",
            Self::MetaClip2B32 => "facebook/metaclip-2-worldwide-b32",
            Self::MetaClip2B16 => "facebook/metaclip-2-worldwide-b16",
            Self::SigLip2Base256 => "google/siglip2-base-patch16-256",
            Self::LaionClipB32 => "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            Self::DinoV3B16 => "facebook/dinov3-vitb16-pretrain-lvd1689m",
            Self::SigLipBetaSwinV2Frozen => {
                "deepghs/siglip_beta/smilingwolf/siglip_swinv2_base_2025_02_22_18h56m54s"
            }
            Self::SigLipBetaSwinV2 => {
                "deepghs/siglip_beta/smilingwolf/siglip_swinv2_base_2025_05_02_22h02m36s"
            }
            Self::SigLipBetaEva02 => {
                "deepghs/siglip_beta/smilingwolf/siglip_eva02_base_2025_05_02_21h53m54s"
            }
        }
    }

    /// Width of every vector produced by this model.
    pub const fn dimensions(self) -> usize {
        match self {
            Self::SigLip2Base256 => 768,
            Self::SigLipBetaSwinV2Frozen | Self::SigLipBetaSwinV2 => 1024,
            Self::SigLipBetaEva02 | Self::DinoV3B16 => 768,
            _ => 512,
        }
    }

    pub const ALL: [Self; 9] = [
        Self::ClipVitB32,
        Self::MetaClip2B32,
        Self::MetaClip2B16,
        Self::SigLip2Base256,
        Self::LaionClipB32,
        Self::SigLipBetaSwinV2Frozen,
        Self::SigLipBetaSwinV2,
        Self::SigLipBetaEva02,
        Self::DinoV3B16,
    ];

    /// Search-model choices offered to users. Retired spaces remain parseable and their
    /// databases stay readable for previously saved selections and existing indexes.
    pub const SELECTABLE: [Self; 5] = [
        Self::MetaClip2B32,
        Self::MetaClip2B16,
        Self::SigLip2Base256,
        Self::SigLipBetaSwinV2Frozen,
        Self::DinoV3B16,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::ClipVitB32 => "OpenAI CLIP B/32 224",
            Self::MetaClip2B32 => "MetaCLIP2 B/32 224",
            Self::MetaClip2B16 => "MetaCLIP2 B/16 224",
            Self::SigLip2Base256 => "SigLIP2 Base B/16 256",
            Self::LaionClipB32 => "LAION ViT-B/32 (laion2b_s34b_b79k)",
            Self::SigLipBetaSwinV2Frozen => "SigLIP beta SwinV2 Base (experimental)",
            Self::SigLipBetaSwinV2 => "SigLIP beta SwinV2 Base (unfrozen image encoder)",
            Self::SigLipBetaEva02 => "SigLIP beta EVA02 Base",
            Self::DinoV3B16 => "DINOv3 B/16 224 (experimental)",
        }
    }

    pub const fn license(self) -> &'static str {
        match self {
            Self::MetaClip2B32 | Self::MetaClip2B16 => "CC-BY-NC-4.0",
            Self::SigLip2Base256 => "Apache-2.0",
            Self::DinoV3B16 => "DINOv3 License",
            Self::SigLipBetaSwinV2Frozen | Self::SigLipBetaSwinV2 | Self::SigLipBetaEva02 => {
                "Apache-2.0"
            }
            _ => "MIT",
        }
    }

    pub const fn context_length(self) -> usize {
        match self {
            Self::DinoV3B16 => 0,
            Self::SigLip2Base256 => 64,
            Self::SigLipBetaSwinV2Frozen | Self::SigLipBetaSwinV2 | Self::SigLipBetaEva02 => 128,
            _ => 77,
        }
    }

    /// Image-only encoders cannot resolve descriptions in their vector space.
    pub const fn supports_text_queries(self) -> bool {
        !matches!(self, Self::DinoV3B16)
    }

    pub const fn is_deepghs(self) -> bool {
        matches!(
            self,
            Self::SigLipBetaSwinV2Frozen | Self::SigLipBetaSwinV2 | Self::SigLipBetaEva02
        )
    }

    pub fn source_url(self) -> String {
        if self.is_deepghs() {
            format!(
                "https://huggingface.co/deepghs/siglip_beta/tree/main/{}",
                self.deepghs_subdirectory().expect("DeepGHS model")
            )
        } else {
            format!("https://huggingface.co/{}", self.id())
        }
    }

    fn deepghs_subdirectory(self) -> Option<&'static str> {
        self.id().strip_prefix("deepghs/siglip_beta/")
    }

    fn deepghs_source(self, filename: &str) -> Result<crate::hub::ModelSource> {
        let subdirectory = self
            .deepghs_subdirectory()
            .context("model is not a DeepGHS checkpoint")?;
        Ok(crate::hub::ModelSource {
            model_id: "deepghs/siglip_beta".to_owned(),
            revision: Some("03aa79c8a4a6c41e06ca87aa6e44fee563b2491d".to_owned()),
            filename: format!("{subdirectory}/{filename}"),
        })
    }

    pub(super) fn deepghs_file(
        self,
        filename: &str,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<std::path::PathBuf>> {
        let source = self.deepghs_source(filename)?;
        if cached_only {
            Ok(source.cached())
        } else {
            source.get_sync_with_progress(progress).map(Some)
        }
    }

    pub(super) fn validate_deepghs_meta(self, path: &std::path::Path) -> Result<()> {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if meta["image_embedding_width"].as_u64() != Some(self.dimensions() as u64)
            || meta["text_embedding_width"].as_u64() != Some(self.dimensions() as u64)
            || meta["image_size"].as_u64()
                != Some(if self == Self::SigLipBetaEva02 {
                    420
                } else {
                    448
                })
        {
            bail!("DeepGHS model metadata does not match {}", self);
        }
        Ok(())
    }

    /// Local export override for development and benchmark comparisons.
    pub fn local_directory(self) -> Option<std::path::PathBuf> {
        if self == Self::ClipVitB32 || self.is_deepghs() {
            return None;
        }
        std::env::var_os("NICEGAL_LOCAL_MODELS_DIR").map(|root| {
            std::path::PathBuf::from(root).join(self.database_file_name().trim_end_matches(".db"))
        })
    }

    fn published_source(self, filename: &str) -> Option<crate::hub::ModelSource> {
        let (model_id, revision) = match self {
            Self::MetaClip2B32 => (
                "bep256/metaclip-2-worldwide-b32-ONNX",
                "a70ddb6e8ac8a2be823ce11f2d296684e150802d",
            ),
            Self::MetaClip2B16 => (
                "bep256/metaclip-2-worldwide-b16-ONNX",
                "33cd628449b4e87dfc3f348d9843b32d32713860",
            ),
            Self::SigLip2Base256 => (
                "bep256/siglip2-base-patch16-256-ONNX",
                "9b5bf05e40e88b58d8076b2508cc7495e75b3224",
            ),
            Self::DinoV3B16 => (
                "bep256/dinov3-vitb16-pretrain-lvd1689m-ONNX",
                "05f9d720e2169b6b7ffa499db098d08dd374bd9d",
            ),
            _ => return None,
        };
        Some(crate::hub::ModelSource {
            model_id: model_id.to_owned(),
            revision: Some(revision.to_owned()),
            filename: filename.to_owned(),
        })
    }

    fn required_files(self) -> &'static [&'static str] {
        if self == Self::DinoV3B16 {
            &[
                "image.onnx",
                "model.onnx_data",
                "manifest.json",
                "preprocessor_config.json",
            ]
        } else {
            &[
                "image.onnx",
                "text.onnx",
                "manifest.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config.json",
            ]
        }
    }

    pub fn available(self) -> bool {
        if self == Self::ClipVitB32 || self.is_deepghs() {
            return true;
        }
        if let Some(path) = self.local_directory() {
            return self
                .required_files()
                .iter()
                .all(|file| path.join(file).is_file());
        }
        self.published_source("manifest.json").is_some()
    }

    /// Resolve one complete pinned snapshot. Cached-only loading never accesses the network.
    pub(super) fn validated_model_directory(
        self,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<std::path::PathBuf>> {
        let path = if let Some(path) = self.local_directory() {
            if cached_only && !self.available() {
                return Ok(None);
            }
            path
        } else {
            let mut directory = None;
            for filename in self.required_files() {
                let source = self.published_source(filename).with_context(|| {
                    format!("No published ONNX export is configured for {self}")
                })?;
                let file = if cached_only {
                    source.cached()
                } else {
                    Some(source.get_sync_with_progress(progress)?)
                };
                let Some(file) = file else { return Ok(None) };
                let parent = file.parent().context("model cache file has no directory")?;
                if let Some(current) = &directory {
                    if current != parent {
                        bail!("model files for {self} were cached in different directories");
                    }
                } else {
                    directory = Some(parent.to_path_buf());
                }
            }
            directory.context("model has no required files")?
        };
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
        if manifest["modelId"].as_str() != Some(self.id())
            || manifest["dimensions"].as_u64() != Some(self.dimensions() as u64)
            || manifest["contextLength"].as_u64() != Some(self.context_length() as u64)
        {
            bail!("local export manifest does not match {}", self);
        }
        if self == Self::DinoV3B16
            && (manifest["imageSize"].as_u64() != Some(224)
                || manifest["compatibility"].as_str() != Some("reshape-nonzero-v1"))
        {
            bail!("DINOv3 needs the validated 224px ONNX export");
        }
        Ok(Some(path))
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

impl std::str::FromStr for ImageEmbeddingModel {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|model| model.id() == value)
            .with_context(|| format!("unsupported image model: {value}"))
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
        Self::load_with_cache_policy(options, false, &())?
            .context("image model loader returned no model")
    }

    pub fn load_with_progress(
        options: &ImageEmbedderOptions,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Self> {
        Self::load_with_cache_policy(options, false, progress)?
            .context("model loader returned no model")
    }

    pub fn load_cached(options: &ImageEmbedderOptions) -> Result<Option<Self>> {
        Self::load_with_cache_policy(options, true, &())
    }

    fn load_with_cache_policy(
        options: &ImageEmbedderOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<Self>> {
        if options.max_batch_size == 0 {
            bail!("image embedding batch size must be greater than zero");
        }
        let local =
            if options.model != ImageEmbeddingModel::ClipVitB32 && !options.model.is_deepghs() {
                let Some(path) = options
                    .model
                    .validated_model_directory(cached_only, progress)?
                else {
                    return Ok(None);
                };
                Some(path)
            } else {
                None
            };
        let cache_dir = crate::hub::cache_dir();
        let intra_threads = options.runtime.intra_threads;
        let (backend, execution_provider) = runtime::with_fallback(options.runtime, |provider| {
            let configured = runtime::configure_provider(provider, intra_threads)?;
            if options.model.is_deepghs() {
                let image =
                    options
                        .model
                        .deepghs_file("image_encode.onnx", cached_only, progress)?;
                let preprocessor =
                    options
                        .model
                        .deepghs_file("preprocessor.json", cached_only, progress)?;
                let meta = options
                    .model
                    .deepghs_file("meta.json", cached_only, progress)?;
                let (Some(image), Some(preprocessor), Some(meta)) = (image, preprocessor, meta)
                else {
                    return Ok(None);
                };
                options.model.validate_deepghs_meta(&meta)?;
                let init = fastembed::ImageInitOptionsUserDefined::new()
                    .with_execution_providers(vec![configured.dispatch])
                    .with_intra_threads(configured.intra_threads.get());
                return ImageEmbedding::try_new_from_deepghs_path(
                    image,
                    &std::fs::read(preprocessor)?,
                    init,
                )
                .map(Some)
                .context("loading DeepGHS image ONNX encoder");
            }
            if let Some(path) = &local {
                let init = fastembed::ImageInitOptionsUserDefined::new()
                    .with_execution_providers(vec![configured.dispatch])
                    .with_intra_threads(configured.intra_threads.get());
                let mut config: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(path.join("preprocessor_config.json"))?)?;
                config["nicegal_center_crop_round"] =
                    (options.model == ImageEmbeddingModel::LaionClipB32).into();
                return ImageEmbedding::try_new_from_path(
                    path.join("image.onnx"),
                    &serde_json::to_vec(&config)?,
                    init,
                )
                .map(|backend| {
                    Some(if options.model == ImageEmbeddingModel::DinoV3B16 {
                        backend.with_output_key("pooler_output")
                    } else {
                        backend
                    })
                })
                .context("loading local image ONNX encoder");
            }
            if !cached_only {
                let info = ImageEmbedding::get_model_info(&FastEmbedImageModel::ClipVitB32);
                for filename in [info.model_file.as_str(), "preprocessor_config.json"] {
                    crate::hub::ModelSource {
                        model_id: info.model_code.clone(),
                        revision: None,
                        filename: filename.to_owned(),
                    }
                    .get_sync_with_progress(progress)?;
                }
            }
            let init = ImageInitOptions::new(FastEmbedImageModel::ClipVitB32)
                .with_cache_dir(cache_dir.clone())
                .with_show_download_progress(true)
                .with_execution_providers(vec![configured.dispatch])
                .with_intra_threads(configured.intra_threads.get());
            let loaded = ImageEmbedding::try_new_cached(init);
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

    /// Keep the original model's decoding compatible with its persisted vectors.
    pub fn decode_image(&self, data: &[u8]) -> Result<crate::imaging::Raster> {
        if self.model == ImageEmbeddingModel::ClipVitB32 {
            crate::imaging::decode(data)
        } else {
            crate::imaging::decode_accurate(data)
        }
    }

    /// Encode a decoded external image without adding it to the catalog.
    pub fn embed_raster(&self, raster: crate::imaging::Raster) -> Result<Vec<f32>> {
        let (width, height) = (raster.width(), raster.height());
        let pixels = if self.model.is_deepghs() {
            raster.flatten_rgb([255, 255, 255])
        } else {
            raster.into_rgb_bytes()
        };
        let image = RgbImage::from_raw(width, height, pixels)
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
        Self::load_with_cache_policy(options, false, &())?.context("model loader returned no model")
    }

    /// Load the paired text encoder, reporting any model-file downloads.
    pub fn load_with_progress(
        options: &ImageQueryEmbedderOptions,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Self> {
        Self::load_with_cache_policy(options, false, progress)?
            .context("model loader returned no model")
    }

    /// Load the paired text encoder strictly from cached files, without a network client.
    pub fn load_cached(options: &ImageQueryEmbedderOptions) -> Result<Option<Self>> {
        Self::load_with_cache_policy(options, true, &())
    }

    fn load_with_cache_policy(
        options: &ImageQueryEmbedderOptions,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<Self>> {
        if options.max_input_bytes == 0 {
            bail!("image query embedding input limit must be greater than zero");
        }
        if !options.model.supports_text_queries() {
            bail!(
                "{} supports image examples only, not text descriptions",
                options.model
            );
        }
        let loaded = if options.model.is_deepghs() {
            FastEmbedBackend::load_deepghs_image_query(
                options.model,
                options.runtime,
                cached_only,
                progress,
            )?
        } else if options.model != ImageEmbeddingModel::ClipVitB32 {
            FastEmbedBackend::load_local_image_query(
                options.model,
                options.runtime,
                cached_only,
                progress,
            )?
        } else {
            FastEmbedBackend::load_with_cache_policy(
                fastembed::EmbeddingModel::ClipVitB32,
                &options.model.to_string(),
                options.runtime,
                cached_only,
                progress,
            )?
        };
        let Some((backend, cache_dir, execution_provider)) = loaded else {
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
        let model = ImageEmbeddingModel::ClipVitB32;
        assert_eq!(
            ImageEmbeddingModel::default(),
            ImageEmbeddingModel::MetaClip2B32
        );
        assert_eq!(model.id(), "Qdrant/clip-ViT-B-32");
        assert_eq!(model.dimensions(), 512);
        assert_eq!(model.database_file_name(), "qdrant-clip-vit-b-32.db");
    }

    #[test]
    fn model_spaces_have_distinct_databases_and_matching_parsers() {
        let mut files = std::collections::HashSet::new();
        for model in ImageEmbeddingModel::ALL {
            assert_eq!(model.id().parse::<ImageEmbeddingModel>().unwrap(), model);
            assert!(files.insert(model.database_file_name()));
            assert_eq!(
                model.dimensions(),
                match model {
                    ImageEmbeddingModel::SigLip2Base256
                    | ImageEmbeddingModel::SigLipBetaEva02
                    | ImageEmbeddingModel::DinoV3B16 => 768,
                    ImageEmbeddingModel::SigLipBetaSwinV2Frozen
                    | ImageEmbeddingModel::SigLipBetaSwinV2 => 1024,
                    _ => 512,
                }
            );
        }
        assert!("../unknown".parse::<ImageEmbeddingModel>().is_err());
        assert_eq!(ImageEmbeddingModel::SELECTABLE.len(), 5);
        assert!(!ImageEmbeddingModel::SELECTABLE.contains(&ImageEmbeddingModel::SigLipBetaSwinV2));
        assert!(!ImageEmbeddingModel::SELECTABLE.contains(&ImageEmbeddingModel::SigLipBetaEva02));
    }

    #[test]
    fn deepghs_sources_point_inside_the_shared_pinned_repository() {
        for model in [
            ImageEmbeddingModel::SigLipBetaSwinV2Frozen,
            ImageEmbeddingModel::SigLipBetaSwinV2,
            ImageEmbeddingModel::SigLipBetaEva02,
        ] {
            assert!(model.is_deepghs());
            let source = model.deepghs_source("image_encode.onnx").unwrap();
            assert_eq!(source.model_id, "deepghs/siglip_beta");
            assert_eq!(
                source.revision.as_deref(),
                Some("03aa79c8a4a6c41e06ca87aa6e44fee563b2491d")
            );
            assert_eq!(
                source.filename,
                format!(
                    "{}/image_encode.onnx",
                    model.id().strip_prefix("deepghs/siglip_beta/").unwrap()
                )
            );
            assert!(
                model
                    .source_url()
                    .ends_with(model.deepghs_subdirectory().unwrap())
            );
        }
    }

    #[test]
    #[ignore = "requires access to the published Hugging Face model repositories"]
    fn published_onnx_metadata_resolves_into_one_cached_snapshot() -> Result<()> {
        for model in ImageEmbeddingModel::SELECTABLE
            .into_iter()
            .filter(|model| !model.is_deepghs())
        {
            let manifest_source = model.published_source("manifest.json").unwrap();
            let preprocessor_source = model.published_source("preprocessor_config.json").unwrap();
            let manifest_path = manifest_source.get_sync()?;
            let preprocessor_path = preprocessor_source.get_sync()?;
            assert_eq!(manifest_path.parent(), preprocessor_path.parent());
            assert_eq!(
                manifest_source.cached().as_deref(),
                Some(manifest_path.as_path())
            );
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(manifest_path)?)?;
            assert_eq!(manifest["modelId"].as_str(), Some(model.id()));
        }
        Ok(())
    }

    #[test]
    fn image_query_defaults_stay_with_the_image_model_on_cpu() {
        let options = ImageQueryEmbedderOptions::default();
        assert_eq!(options.model, ImageEmbeddingModel::MetaClip2B32);
        assert_eq!(options.max_input_bytes, 4096);
        assert_eq!(options.runtime.execution_provider, ExecutionProvider::Cpu);
    }
}
