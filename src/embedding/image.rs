use std::fmt;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use fastembed::{ImageEmbedding, ImagePreprocessor};
use image::{DynamicImage, RgbImage};
use ndarray::Array3;
use rayon::prelude::*;
use tracing::{info, instrument, warn};

use crate::runtime::{self, ExecutionProvider, RuntimeOptions};

use super::fastembed::FastEmbedBackend;
use super::patches::{self, PatchFeatures, PatchMethod};

/// An image embedding model supported by this build.
///
/// The identifier names the shared image/text coordinate space used by both the image encoder and
/// its paired text-query encoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ImageEmbeddingModel {
    #[default]
    MetaClip2B32,
    MetaClip2B16,
    MetaClip2L14,
    SigLip2Base256,
    SigLipBetaSwinV2Frozen,
    DinoV3B16,
}

struct ModelSpec {
    id: &'static str,
    name: &'static str,
    dimensions: usize,
    license: &'static str,
    context_length: usize,
    /// Whether query text is lowercased before tokenization, matching how the model was trained.
    lowercase_text: bool,
    published_export: Option<(&'static str, &'static str)>,
    required_files: &'static [&'static str],
}

const PAIRED_MODEL_FILES: &[&str] = &[
    "image.onnx",
    "text.onnx",
    "manifest.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
];
const PAIRED_MODEL_FILES_WITH_TEXT_DATA: &[&str] = &[
    "image.onnx",
    "text.onnx",
    "text.onnx_data",
    "manifest.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
];
const IMAGE_ONLY_MODEL_FILES: &[&str] = &[
    "image.onnx",
    "model.onnx_data",
    "manifest.json",
    "preprocessor_config.json",
];

impl ImageEmbeddingModel {
    const fn spec(self) -> ModelSpec {
        match self {
            Self::MetaClip2B32 => ModelSpec {
                id: "facebook/metaclip-2-worldwide-b32",
                name: "MetaCLIP2 B/32 224",
                dimensions: 512,
                license: "CC-BY-NC-4.0",
                context_length: 77,
                lowercase_text: false,
                published_export: Some((
                    "bep256/metaclip-2-worldwide-b32-ONNX",
                    "b4ee5fd6043c2b33df398eb0288a6282706c1687",
                )),
                required_files: PAIRED_MODEL_FILES,
            },
            Self::MetaClip2B16 => ModelSpec {
                id: "facebook/metaclip-2-worldwide-b16",
                name: "MetaCLIP2 B/16 224",
                dimensions: 512,
                license: "CC-BY-NC-4.0",
                context_length: 77,
                lowercase_text: false,
                published_export: Some((
                    "bep256/metaclip-2-worldwide-b16-ONNX",
                    "d96138fa24aa9cc3f46abf34a06f45f35e71bbba",
                )),
                required_files: PAIRED_MODEL_FILES,
            },
            Self::MetaClip2L14 => ModelSpec {
                id: "facebook/metaclip-2-worldwide-l14",
                name: "MetaCLIP2 L/14 224",
                dimensions: 768,
                license: "CC-BY-NC-4.0",
                context_length: 77,
                lowercase_text: false,
                published_export: Some((
                    "bep256/metaclip-2-worldwide-l14-ONNX",
                    "77e0837a2b1d7134c6d5678133a5abadaa933bde",
                )),
                required_files: PAIRED_MODEL_FILES_WITH_TEXT_DATA,
            },
            Self::SigLip2Base256 => ModelSpec {
                id: "google/siglip2-base-patch16-256",
                name: "SigLIP2 Base B/16 256",
                dimensions: 768,
                license: "Apache-2.0",
                context_length: 64,
                lowercase_text: true,
                published_export: Some((
                    "bep256/siglip2-base-patch16-256-ONNX",
                    "1fa886058822dbe657d57cfa4e5686c6b886f910",
                )),
                required_files: PAIRED_MODEL_FILES,
            },
            Self::SigLipBetaSwinV2Frozen => ModelSpec {
                id: "deepghs/siglip_beta/smilingwolf/siglip_swinv2_base_2025_02_22_18h56m54s",
                name: "SigLIP beta SwinV2 Base (experimental)",
                dimensions: 1024,
                license: "Apache-2.0",
                context_length: 128,
                lowercase_text: false,
                published_export: None,
                required_files: PAIRED_MODEL_FILES,
            },
            Self::DinoV3B16 => ModelSpec {
                id: "facebook/dinov3-vitb16-pretrain-lvd1689m",
                name: "DINOv3 B/16 224 (experimental)",
                dimensions: 768,
                license: "DINOv3 License",
                context_length: 0,
                lowercase_text: false,
                published_export: Some((
                    "bep256/dinov3-vitb16-pretrain-lvd1689m-ONNX",
                    "05f9d720e2169b6b7ffa499db098d08dd374bd9d",
                )),
                required_files: IMAGE_ONLY_MODEL_FILES,
            },
        }
    }

    /// Stable identifier for the compatible image/text embedding pair.
    pub const fn id(self) -> &'static str {
        self.spec().id
    }

    /// Width of every vector produced by this model.
    pub const fn dimensions(self) -> usize {
        self.spec().dimensions
    }

    pub const ALL: [Self; 6] = [
        Self::MetaClip2B32,
        Self::MetaClip2B16,
        Self::SigLip2Base256,
        Self::MetaClip2L14,
        Self::SigLipBetaSwinV2Frozen,
        Self::DinoV3B16,
    ];

    pub const fn name(self) -> &'static str {
        self.spec().name
    }
    pub const fn license(self) -> &'static str {
        self.spec().license
    }
    pub const fn context_length(self) -> usize {
        self.spec().context_length
    }

    /// Image-only encoders cannot resolve descriptions in their vector space.
    pub const fn supports_text_queries(self) -> bool {
        self.context_length() > 0
    }

    /// The text a paired query encoder tokenizes for `text`. SigLIP2 was trained on lowercased
    /// text and its published `tokenizer.json` does not lowercase, so its queries are lowercased
    /// here. Other models see the text unchanged.
    pub(super) fn prepare_query_text(self, text: &str) -> std::borrow::Cow<'_, str> {
        if self.spec().lowercase_text {
            std::borrow::Cow::Owned(text.to_lowercase())
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    pub const fn is_deepghs(self) -> bool {
        matches!(self, Self::SigLipBetaSwinV2Frozen)
    }

    /// Whether [`ImageEmbedder::patch_features`] can expose spatial image features.
    pub const fn supports_patch_features(self) -> bool {
        self.patch_method().is_some()
    }

    /// How the image graph gains patch features. SigLIP2's attention-pooling head gives maps too
    /// weak to be worth showing. DINOv3 exposes its normalized tokens before class pooling.
    pub(super) const fn patch_method(self) -> Option<PatchMethod> {
        match self {
            Self::MetaClip2B32 | Self::MetaClip2B16 | Self::MetaClip2L14 => {
                Some(PatchMethod::ClearClip)
            }
            // `encodings` is the spatial mean of this map and `embeddings` its L2 normalization.
            Self::SigLipBetaSwinV2Frozen => Some(PatchMethod::FeatureMap(
                "/siglip_model/norm/LayerNormalization_output_0",
            )),
            Self::DinoV3B16 => Some(PatchMethod::DinoTokens("/norm/LayerNormalization_output_0")),
            Self::SigLip2Base256 => None,
        }
    }

    /// The pooled output the image index stores.
    const fn embedding_output(self) -> &'static str {
        if self.is_deepghs() {
            "embeddings"
        } else if matches!(self, Self::DinoV3B16) {
            "pooler_output"
        } else {
            "image_embeds"
        }
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
            || meta["image_size"].as_u64() != Some(448)
        {
            bail!("DeepGHS model metadata does not match {}", self);
        }
        Ok(())
    }

    fn deepghs_image_files(
        self,
        cached_only: bool,
        progress: &dyn crate::hub::DownloadObserver,
    ) -> Result<Option<(std::path::PathBuf, Vec<u8>)>> {
        let image = self.deepghs_file("image_encode.onnx", cached_only, progress)?;
        let preprocessor = self.deepghs_file("preprocessor.json", cached_only, progress)?;
        let meta = self.deepghs_file("meta.json", cached_only, progress)?;
        let (Some(image), Some(preprocessor), Some(meta)) = (image, preprocessor, meta) else {
            return Ok(None);
        };
        self.validate_deepghs_meta(&meta)?;
        Ok(Some((image, std::fs::read(preprocessor)?)))
    }

    /// Local export override for development and validation.
    pub fn local_directory(self) -> Option<std::path::PathBuf> {
        if self.is_deepghs() {
            return None;
        }
        std::env::var_os("NICEGAL_LOCAL_MODELS_DIR").map(|root| {
            std::path::PathBuf::from(root).join(self.database_file_name().trim_end_matches(".db"))
        })
    }

    fn published_source(self, filename: &str) -> Option<crate::hub::ModelSource> {
        let (model_id, revision) = self.spec().published_export?;
        Some(crate::hub::ModelSource {
            model_id: model_id.to_owned(),
            revision: Some(revision.to_owned()),
            filename: filename.to_owned(),
        })
    }

    fn required_files(self) -> &'static [&'static str] {
        self.spec().required_files
    }

    /// Called only after the replacement encoder has loaded successfully. Local overrides
    /// never remove shared downloads. Old metadata stays available for provenance.
    pub(super) fn retire_previous_encoder(self, text: bool) {
        if self.local_directory().is_some() {
            return;
        }
        let previous = match self {
            Self::MetaClip2B32 => "a70ddb6e8ac8a2be823ce11f2d296684e150802d",
            Self::MetaClip2B16 => "33cd628449b4e87dfc3f348d9843b32d32713860",
            Self::MetaClip2L14 => "c7193980d96e63812a6f70f8ef3feb934549326f",
            Self::SigLip2Base256 => "9b5bf05e40e88b58d8076b2508cc7495e75b3224",
            _ => return,
        };
        let Some((repo, current)) = self.spec().published_export else {
            return;
        };
        if current == previous {
            return;
        }
        let files: &[&str] = if text {
            &["text.onnx", "text.onnx_data"]
        } else {
            &["image.onnx"]
        };
        if let Err(error) =
            crate::hub::retire_files(&crate::hub::cache_dir(), repo, previous, files)
        {
            warn!(%error, model = %self, "could not retire old encoder files; will retry on next load");
        }
    }

    pub fn available(self) -> bool {
        if self.is_deepghs() {
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
        for filename in self.external_export_files(&manifest)? {
            if self.local_directory().is_some() {
                if !path.join(filename).is_file() {
                    if cached_only {
                        return Ok(None);
                    }
                    bail!("local export is missing {filename}");
                }
            } else {
                let source = self
                    .published_source(filename)
                    .context("missing export source")?;
                let file = if cached_only {
                    source.cached()
                } else {
                    Some(source.get_sync_with_progress(progress)?)
                };
                let Some(file) = file else { return Ok(None) };
                if file.parent() != Some(path.as_path()) {
                    bail!("external model files were cached in different directories");
                }
            }
        }
        let files: Vec<&str> = self
            .required_files()
            .iter()
            .copied()
            .chain(self.external_export_files(&manifest)?.iter().copied())
            .collect();
        Ok(Some(local_external_weights_directory(&path, &files)?))
    }

    /// Fixed filenames only: a manifest cannot request arbitrary paths or repositories.
    fn external_export_files(
        self,
        manifest: &serde_json::Value,
    ) -> Result<&'static [&'static str]> {
        match manifest
            .get("storageFormat")
            .and_then(serde_json::Value::as_str)
        {
            None => Ok(&[]),
            Some("external-weights-v1") => match self {
                Self::MetaClip2B32 | Self::MetaClip2B16 | Self::MetaClip2L14 => {
                    if manifest["patchGraph"].as_str() != Some("clearclip-v1") {
                        bail!("external MetaCLIP export requires the clearclip-v1 graph");
                    }
                    Ok(&["image.onnx_data", "text.onnx_data", "image_patched.onnx"])
                }
                Self::SigLip2Base256 => Ok(&["image.onnx_data", "text.onnx_data"]),
                _ => bail!("external-weights-v1 is not supported for {self}"),
            },
            Some(format) => bail!("unsupported export storage format: {format}"),
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

/// ONNX Runtime rejects external tensor files that resolve outside the graph directory.
/// Python's HF cache uses symlinks into `blobs`; keep that shared cache intact and expose
/// the validated export through hard links instead. This also covers the text encoder.
fn local_external_weights_directory(
    source: &std::path::Path,
    files: &[&str],
) -> Result<std::path::PathBuf> {
    use std::hash::{Hash, Hasher};
    let linked_weights = files.iter().any(|name| {
        name.ends_with(".onnx_data")
            && std::fs::symlink_metadata(source.join(name))
                .is_ok_and(|metadata| metadata.file_type().is_symlink())
    });
    if !linked_weights {
        return Ok(source.to_path_buf());
    }
    // File identity changes produce a fresh view, including for mutable local overrides.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    let mut resolved = Vec::with_capacity(files.len());
    for name in files {
        let path = source
            .join(name)
            .canonicalize()
            .with_context(|| format!("resolving model file {name}"))?;
        let metadata = std::fs::metadata(&path)?;
        name.hash(&mut hash);
        path.hash(&mut hash);
        metadata.len().hash(&mut hash);
        metadata.modified()?.hash(&mut hash);
        resolved.push((name, path));
    }
    let directory = source.join(format!("nicegal-onnx-v1-{:016x}", hash.finish()));
    std::fs::create_dir_all(&directory).context("creating local ONNX model view")?;
    for (name, path) in resolved {
        let target = directory.join(name);
        match std::fs::hard_link(&path, &target) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && target.is_file() => {
            }
            Err(error) => {
                return Err(error).with_context(|| format!("linking local ONNX file {name}"));
            }
        }
    }
    Ok(directory)
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
    /// Present when the session was built with [`patches::PATCH_OUTPUT`].
    patch_setup: Option<PatchSetup>,
}

enum PatchedGraph {
    Memory(Vec<u8>),
    /// ONNX external tensor data resolves relative to the graph file, not a memory buffer.
    File(std::path::PathBuf),
}

#[derive(Debug, Clone, Copy)]
struct PatchSetup {
    method: PatchMethod,
    geometry: InputGeometry,
}

/// Where the model's square input comes from in a source image, following its preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputGeometry {
    /// An exact resize: the whole image, stretched.
    Stretch,
    /// FastEmbed's shortest-edge resize, which truncates the long side, then a center crop.
    ShortestEdge { edge: u32, crop: (u32, u32) },
    /// DeepGHS: the short side to `size` with the long side capped at `max_size`, then a square
    /// center crop.
    DeepGhs { size: u32, max_size: u32, crop: u32 },
}

impl InputGeometry {
    /// From a Hugging Face `preprocessor_config.json`.
    fn from_config(config: &[u8]) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(config).context("parsing image preprocessor configuration")?;
        let crop = config["do_center_crop"]
            .as_bool()
            .unwrap_or(false)
            .then(|| match &config["crop_size"] {
                size if size.is_object() => {
                    dimension(&size["width"]).zip(dimension(&size["height"]))
                }
                size => dimension(size).map(|edge| (edge, edge)),
            })
            .flatten();
        Ok(match (dimension(&config["size"]["shortest_edge"]), crop) {
            (Some(edge), Some(crop)) => Self::ShortestEdge { edge, crop },
            _ => Self::Stretch,
        })
    }

    /// From a DeepGHS `preprocessor.json` transform description.
    fn from_deepghs_config(config: &[u8]) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(config).context("parsing DeepGHS preprocessor configuration")?;
        let stage = |kind: &str| {
            config["stages"]
                .as_array()
                .and_then(|stages| stages.iter().find(|stage| stage["type"] == kind))
                .with_context(|| format!("DeepGHS preprocessor has no {kind} stage"))
        };
        let (resize, crop) = (stage("resize")?, stage("center_crop")?);
        let field = |stage: &serde_json::Value, name: &str| {
            dimension(&stage[name])
                .with_context(|| format!("DeepGHS preprocessor lacks a numeric {name}"))
        };
        Ok(Self::DeepGhs {
            size: field(resize, "size")?,
            max_size: field(resize, "max_size")?,
            crop: field(crop, "size")?,
        })
    }

    /// `[x, y, width, height]` of the source, as fractions, that the model input covers. Values
    /// fall outside 0 to 1 where the center crop pads a short axis.
    fn region(self, width: u32, height: u32) -> [f64; 4] {
        let (width, height) = (width.max(1), height.max(1));
        let (resized, (crop_width, crop_height)) = match self {
            Self::Stretch => return [0.0, 0.0, 1.0, 1.0],
            Self::ShortestEdge { edge, crop } => {
                let (short, long) = (width.min(height), width.max(height));
                let long = ((f64::from(edge) * f64::from(long)) / f64::from(short)) as u32;
                let resized = if width <= height {
                    (edge, long)
                } else {
                    (long, edge)
                };
                (resized, crop)
            }
            Self::DeepGhs {
                size,
                max_size,
                crop,
            } => {
                let scale = |value: u32, numerator: u32, denominator: u32| {
                    (u64::from(value) * u64::from(numerator) / u64::from(denominator)) as u32
                };
                let (mut resized_width, mut resized_height) = if width < height {
                    (size, scale(size, height, width))
                } else {
                    (scale(size, width, height), size)
                };
                if resized_width.max(resized_height) > max_size {
                    if resized_height > resized_width {
                        resized_width = scale(max_size, resized_width, resized_height);
                        resized_height = max_size;
                    } else {
                        resized_height = scale(max_size, resized_height, resized_width);
                        resized_width = max_size;
                    }
                }
                ((resized_width, resized_height), (crop, crop))
            }
        };
        let (x, w) = crop_axis(resized.0.max(1), crop_width);
        let (y, h) = crop_axis(resized.1.max(1), crop_height);
        [x, y, w, h]
    }
}

fn dimension(value: &serde_json::Value) -> Option<u32> {
    value.as_u64().and_then(|v| u32::try_from(v).ok())
}

/// The start and length, as fractions of `resized`, of a center crop to `crop` pixels. A longer
/// axis keeps its middle; a shorter one is padded evenly, so the crop starts before the image.
fn crop_axis(resized: u32, crop: u32) -> (f64, f64) {
    let start = if resized >= crop {
        f64::from((resized - crop) / 2)
    } else {
        -f64::from((crop - resized) / 2)
    };
    (
        start / f64::from(resized),
        f64::from(crop) / f64::from(resized),
    )
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
        let cache_dir = crate::hub::cache_dir();
        // Resolve and validate files once; only session construction belongs in provider retries.
        let (image, preprocessor_config, deepghs) = if options.model.is_deepghs() {
            let Some(files) = options.model.deepghs_image_files(cached_only, progress)? else {
                return Ok(None);
            };
            (files.0, files.1, true)
        } else {
            let Some(path) = options
                .model
                .validated_model_directory(cached_only, progress)?
            else {
                return Ok(None);
            };
            (
                path.join("image.onnx"),
                std::fs::read(path.join("preprocessor_config.json"))
                    .context("reading image preprocessor configuration")?,
                false,
            )
        };
        // Built once, outside provider retries: the graph with its patch output appended.
        let static_patch = if !deepghs {
            let directory = image
                .parent()
                .context("image graph has no parent directory")?;
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
            (manifest["patchGraph"].as_str() == Some("clearclip-v1")
                && manifest["storageFormat"].as_str() == Some("external-weights-v1"))
            .then(|| PatchedGraph::File(directory.join("image_patched.onnx")))
        } else {
            None
        };
        let patched_graph = static_patch.or_else(|| {
            options.model.patch_method().and_then(|method| {
                Self::patched_graph(
                    &image,
                    method,
                    options.model == ImageEmbeddingModel::DinoV3B16,
                )
            })
        });
        let (backend, execution_provider) = runtime::with_fallback(options.runtime, |provider| {
            let configured = runtime::configure_provider(provider, options.runtime.intra_threads)?;
            let init = fastembed::ImageInitOptionsUserDefined::new()
                .with_execution_providers(vec![configured.dispatch])
                .with_intra_threads(configured.intra_threads.get());
            if let Some(graph) = &patched_graph {
                let preprocessor = if deepghs {
                    ImagePreprocessor::from_deepghs_config(&preprocessor_config)
                } else {
                    ImagePreprocessor::from_local_config(&preprocessor_config)
                }
                .context("preparing image preprocessing")?;
                let mut builder = ImageEmbedding::session_builder(init)
                    .context("configuring image ONNX session")?;
                let session = match graph {
                    PatchedGraph::Memory(bytes) => builder.commit_from_memory(bytes),
                    PatchedGraph::File(path) => builder.commit_from_file(path),
                }
                .context("loading image ONNX encoder with patch features")?;
                return Ok(Some(
                    ImageEmbedding::try_new_from_session(session, preprocessor)
                        .with_output_key(options.model.embedding_output()),
                ));
            }
            let backend = if deepghs {
                ImageEmbedding::try_new_from_deepghs_path(&image, &preprocessor_config, init)
                    .context("loading DeepGHS image ONNX encoder")?
            } else {
                ImageEmbedding::try_new_from_path(&image, &preprocessor_config, init)
                    .context("loading local image ONNX encoder")?
            };
            Ok(Some(if options.model == ImageEmbeddingModel::DinoV3B16 {
                backend.with_output_key("pooler_output")
            } else {
                backend
            }))
        })?;
        let Some(backend) = backend else {
            return Ok(None);
        };
        let patch_setup = match (patched_graph.is_some(), options.model.patch_method()) {
            (true, Some(method)) => Some(PatchSetup {
                method,
                geometry: if deepghs {
                    InputGeometry::from_deepghs_config(&preprocessor_config)?
                } else {
                    InputGeometry::from_config(&preprocessor_config)?
                },
            }),
            _ => None,
        };
        drop(patched_graph);
        let preprocessor = backend
            .preprocessor()
            .with_resize(|image, width, height, filter| {
                crate::imaging::resize_rgb(image.into_rgb8(), width, height, filter)
                    .map(DynamicImage::ImageRgb8)
                    .map_err(|error| fastembed::Error::ImageTransform(format!("{error:#}")))
            });
        let embedder = Self {
            backend: Mutex::new(backend),
            preprocessor,
            model: options.model,
            max_batch_size: options.max_batch_size,
            execution_provider,
            patch_setup,
        };
        options.model.retire_previous_encoder(false);
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

    pub fn decode_image(&self, data: &[u8]) -> Result<crate::imaging::Raster> {
        crate::imaging::decode_accurate(data)
    }

    /// Encode a decoded external image without adding it to the catalog.
    pub fn embed_raster(&self, raster: crate::imaging::Raster) -> Result<Vec<f32>> {
        let tensor = self.preprocess_raster(raster)?;
        self.embed_preprocessed_images(vec![tensor])?
            .pop()
            .context("missing image vector")
    }

    fn preprocess_raster(&self, raster: crate::imaging::Raster) -> Result<Array3<f32>> {
        let (width, height) = (raster.width(), raster.height());
        let pixels = if self.model.is_deepghs() {
            raster.flatten_rgb([255, 255, 255])
        } else {
            raster.into_rgb_bytes()
        };
        let image = RgbImage::from_raw(width, height, pixels)
            .context("invalid decoded image dimensions")?;
        self.preprocess_image(image)
    }

    /// Read the image graph and append its patch output. A graph without the expected layout
    /// still loads, without patch features.
    fn patched_graph(
        path: &std::path::Path,
        method: PatchMethod,
        file_backed: bool,
    ) -> Option<PatchedGraph> {
        let result = std::fs::read(path)
            .context("reading image ONNX encoder")
            .and_then(|mut graph| {
                let fragment = method.fragment(&graph)?;
                graph.extend_from_slice(&fragment);
                if file_backed {
                    Self::write_patched_graph(path, &graph).map(PatchedGraph::File)
                } else {
                    Ok(PatchedGraph::Memory(graph))
                }
            });
        match result {
            Ok(graph) => Some(graph),
            Err(error) => {
                warn!(
                    error = format!("{error:#}"),
                    "image patch features are unavailable"
                );
                None
            }
        }
    }

    /// Keep the patched graph and external data in one derived cache directory. Hugging Face's
    /// snapshot uses a symlink to a blob outside that directory, which ONNX Runtime rejects for a
    /// newly written graph; a hard link to the same blob keeps paths local without copying weights.
    fn write_patched_graph(source: &std::path::Path, bytes: &[u8]) -> Result<std::path::PathBuf> {
        static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        use std::hash::{Hash, Hasher};
        use std::sync::atomic::Ordering;
        let snapshot = source
            .parent()
            .context("image graph has no parent directory")?;
        let directory = snapshot.join("nicegal-patches-v1");
        std::fs::create_dir_all(&directory).context("creating patched image graph cache")?;
        let external = directory.join("model.onnx_data");
        if !external.is_file() {
            std::fs::hard_link(snapshot.join("model.onnx_data").canonicalize()?, &external)
                .context("linking DINOv3 external tensor data")?;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        let target = directory.join(format!("image-{:016x}.onnx", hasher.finish()));
        if target.is_file() {
            return Ok(target);
        }
        let temp = directory.join(format!(
            "image.{}.{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&temp, bytes).context("writing patched image graph")?;
        match std::fs::rename(&temp, &target) {
            Ok(()) => Ok(target),
            Err(error) => {
                let _ = std::fs::remove_file(&temp);
                if target.is_file() {
                    Ok(target)
                } else {
                    Err(error).context("installing patched image graph")
                }
            }
        }
    }

    /// Whether [`Self::patch_features`] is available for the loaded session.
    pub fn supports_patch_features(&self) -> bool {
        self.patch_setup.is_some()
    }

    /// Per-patch vectors in the joint text–image space for one decoded, oriented image, for
    /// visualization. Their similarity to a query vector shows where the model sees it.
    #[instrument(name = "image_patch_features", level = "debug", skip_all)]
    pub fn patch_features(&self, raster: crate::imaging::Raster) -> Result<PatchFeatures> {
        self.patch_features_batch(vec![raster])?
            .pop()
            .context("missing image patch features")
    }

    /// Run one patch inference for up to `max_batch_size` decoded images.
    #[instrument(name = "image_patch_features_batch", level = "debug", skip_all, fields(batch = rasters.len()))]
    pub fn patch_features_batch(
        &self,
        rasters: Vec<crate::imaging::Raster>,
    ) -> Result<Vec<PatchFeatures>> {
        if rasters.len() > self.max_batch_size {
            bail!(
                "image batch of {} exceeds the {} images {} accepts",
                rasters.len(),
                self.max_batch_size,
                self.model
            );
        }
        if rasters.is_empty() {
            return Ok(Vec::new());
        }
        let setup = self
            .patch_setup
            .with_context(|| format!("{} does not provide patch features", self.model))?;
        let regions: Vec<_> = rasters
            .iter()
            .map(|raster| setup.geometry.region(raster.width(), raster.height()))
            .collect();
        let embedding_output = self.model.embedding_output();
        let pixels: Vec<_> = rasters
            .into_par_iter()
            .map(|raster| self.preprocess_raster(raster))
            .collect::<Result<_>>()?;
        let pixels = ndarray::stack(
            ndarray::Axis(0),
            &pixels
                .iter()
                .map(|pixels| pixels.view())
                .collect::<Vec<_>>(),
        )?;
        let mut backend = self
            .backend
            .lock()
            .map_err(|_| anyhow::anyhow!("image embedding model lock was poisoned"))?;
        let session = backend.session_mut();
        let input = session.inputs()[0].name().to_owned();
        let options = ort::session::RunOptions::new()?.with_outputs(
            ort::session::OutputSelector::no_default()
                .with(embedding_output)
                .with(patches::PATCH_OUTPUT),
        );
        let outputs = session
            .run_with_options(
                ort::inputs![input => ort::value::Tensor::from_array(pixels)?],
                &options,
            )
            .context("running image patch inference")?;
        let (embedding_shape, embedding) = outputs[embedding_output].try_extract_tensor::<f32>()?;
        let (shape, data) = outputs[patches::PATCH_OUTPUT].try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let grids = patches::patch_grids(&shape, data, setup.method.prefix_tokens())?;
        if grids.len() != regions.len()
            || embedding_shape.len() != 2
            || embedding_shape[0] != regions.len() as i64
            || embedding_shape[1] != self.dimensions() as i64
        {
            bail!(
                "{} returned inconsistent patch or embedding batch shapes",
                self.model
            );
        }
        Ok(grids
            .into_iter()
            .zip(regions)
            .zip(embedding.chunks_exact(self.dimensions()))
            .map(
                |(((rows, columns, dimensions, patches), region), embedding)| {
                    let norm = embedding
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>()
                        .sqrt()
                        .max(f32::EPSILON);
                    PatchFeatures {
                        rows,
                        columns,
                        dimensions,
                        patches,
                        embedding: embedding.iter().map(|v| v / norm).collect(),
                        region,
                        method: setup.method.name(),
                    }
                },
            )
            .collect())
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
    /// Image model whose coordinate space query vectors must use.
    pub model: ImageEmbeddingModel,
    /// Longest input embedded, in bytes of UTF-8. Longer text is truncated on a character
    /// boundary first; the backend's tokenizer then truncates to the model's own context window.
    pub max_input_bytes: usize,
    /// Runtime settings for the paired text encoder.
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
/// It uses the same [`ImageEmbeddingModel`] coordinate space as its paired [`ImageEmbedder`].
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
        } else {
            FastEmbedBackend::load_local_image_query(
                options.model,
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

    /// Embed one search query in the paired image model's coordinate space.
    #[instrument(name = "embed_image_query", level = "debug", skip_all, fields(bytes = text.len()))]
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            bail!("cannot embed an empty query");
        }
        let text = super::truncate_on_boundary(text, self.max_input_bytes);
        let text = self.model.prepare_query_text(text);
        let mut vectors = self
            .backend
            .embed(&[text.as_ref()], 1)
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
    #[test]
    fn external_weight_symlinks_get_a_reusable_local_view() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let snapshot = temp.path().join("snapshot");
        let blobs = temp.path().join("blobs");
        std::fs::create_dir_all(&snapshot)?;
        std::fs::create_dir_all(&blobs)?;
        let files = [
            "image.onnx",
            "image.onnx_data",
            "text.onnx",
            "text.onnx_data",
        ];
        for name in files {
            let blob = blobs.join(name);
            std::fs::write(&blob, name)?;
            #[cfg(windows)]
            std::os::windows::fs::symlink_file(&blob, snapshot.join(name))?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&blob, snapshot.join(name))?;
        }
        let view = local_external_weights_directory(&snapshot, &files)?;
        assert_ne!(view, snapshot);
        for name in files {
            assert_eq!(std::fs::read(view.join(name))?, name.as_bytes());
            assert_eq!(
                view.join(name).canonicalize()?.parent(),
                Some(view.canonicalize()?.as_path())
            );
            assert!(
                std::fs::symlink_metadata(snapshot.join(name))?
                    .file_type()
                    .is_symlink()
            );
        }
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let snapshot = snapshot.clone();
                std::thread::spawn(move || local_external_weights_directory(&snapshot, &files))
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap()?, view);
        }
        std::fs::remove_file(blobs.join("image.onnx_data"))?;
        std::fs::write(blobs.join("image.onnx_data"), "replacement weights")?;
        let updated = local_external_weights_directory(&snapshot, &files)?;
        assert_ne!(updated, view);
        assert_eq!(
            std::fs::read(updated.join("image.onnx_data"))?,
            b"replacement weights"
        );
        assert_eq!(local_external_weights_directory(&updated, &files)?, updated);
        Ok(())
    }

    #[test]
    #[ignore = "requires cached L/14 weights and ONNX Runtime"]
    fn cached_l14_external_weights_load_and_infer() -> Result<()> {
        initialize_test_runtime();
        let model = ImageEmbeddingModel::MetaClip2L14;
        let image = ImageEmbedder::load_cached(&ImageEmbedderOptions {
            model,
            ..ImageEmbedderOptions::default()
        })?
        .context("L/14 image weights are not cached")?;
        let vector = image.embed_raster(crate::imaging::Raster::Rgb {
            width: 224,
            height: 224,
            pixels: vec![127; 224 * 224 * 3],
        })?;
        assert_eq!(vector.len(), model.dimensions());
        assert!(vector.iter().all(|value| value.is_finite()));
        assert!(image.supports_patch_features());
        drop(image);
        let text = ImageQueryEmbedder::load_cached(&ImageQueryEmbedderOptions {
            model,
            ..ImageQueryEmbedderOptions::default()
        })?
        .context("L/14 text weights are not cached")?;
        let vector = text.embed_query("a photo of a cat")?;
        assert_eq!(vector.len(), model.dimensions());
        assert!(vector.iter().all(|value| value.is_finite()));
        Ok(())
    }

    use super::*;

    #[test]
    fn default_model_metadata_is_stable() {
        let model = ImageEmbeddingModel::MetaClip2B32;
        assert_eq!(
            ImageEmbeddingModel::default(),
            ImageEmbeddingModel::MetaClip2B32
        );
        assert_eq!(model.id(), "facebook/metaclip-2-worldwide-b32");
        assert_eq!(model.dimensions(), 512);
        assert_eq!(
            model.database_file_name(),
            "facebook-metaclip-2-worldwide-b32.db"
        );
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
                    ImageEmbeddingModel::MetaClip2L14
                    | ImageEmbeddingModel::SigLip2Base256
                    | ImageEmbeddingModel::DinoV3B16 => 768,
                    ImageEmbeddingModel::SigLipBetaSwinV2Frozen => 1024,
                    _ => 512,
                }
            );
        }
        assert!("../unknown".parse::<ImageEmbeddingModel>().is_err());
        assert_eq!(ImageEmbeddingModel::ALL.len(), 6);
    }

    #[test]
    fn deepghs_sources_point_inside_the_shared_pinned_repository() {
        let model = ImageEmbeddingModel::SigLipBetaSwinV2Frozen;
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

    #[test]
    #[ignore = "requires access to the published Hugging Face model repositories"]
    fn published_onnx_metadata_resolves_into_one_cached_snapshot() -> Result<()> {
        for model in ImageEmbeddingModel::ALL
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
    fn only_siglip2_lowercases_query_text() {
        assert_eq!(
            ImageEmbeddingModel::SigLip2Base256.prepare_query_text("A Dog On The BEACH"),
            "a dog on the beach"
        );
        for model in [
            ImageEmbeddingModel::MetaClip2B32,
            ImageEmbeddingModel::MetaClip2B16,
            ImageEmbeddingModel::MetaClip2L14,
        ] {
            assert_eq!(
                model.prepare_query_text("A Dog On The BEACH"),
                "A Dog On The BEACH"
            );
        }
    }

    #[test]
    fn image_query_defaults_stay_with_the_image_model_on_cpu() {
        let options = ImageQueryEmbedderOptions::default();
        assert_eq!(options.model, ImageEmbeddingModel::MetaClip2B32);
        assert_eq!(options.max_input_bytes, 4096);
        assert_eq!(options.runtime.execution_provider, ExecutionProvider::Cpu);
    }

    #[test]
    fn input_region_follows_resize_and_crop() {
        let stretched = InputGeometry::from_config(
            br#"{"size": {"height": 224, "width": 224}, "do_center_crop": true, "crop_size": 224}"#,
        )
        .unwrap();
        assert_eq!(stretched.region(400, 200), [0.0, 0.0, 1.0, 1.0]);
        let cropped = InputGeometry::from_config(
            br#"{"size": {"shortest_edge": 224}, "do_center_crop": true,
                 "crop_size": {"height": 224, "width": 224}}"#,
        )
        .unwrap();
        assert_eq!(cropped.region(400, 200), [0.25, 0.0, 0.5, 1.0]);
        assert_eq!(cropped.region(200, 800), [0.0, 0.375, 1.0, 0.25]);
    }

    #[test]
    fn deepghs_region_marks_the_padded_axis() {
        let geometry = InputGeometry::from_deepghs_config(
            br#"{"stages": [
                {"type": "convert_rgb", "force_background": "white"},
                {"type": "resize", "size": 448, "max_size": 448, "interpolation": "bicubic"},
                {"type": "center_crop", "size": 448},
                {"type": "maybe_to_tensor"},
                {"type": "normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
            ]}"#,
        )
        .unwrap();
        assert_eq!(geometry.region(500, 500), [0.0, 0.0, 1.0, 1.0]);
        // 800x400 fits as 448x224, centered in a 448 square with 112 rows of padding each side.
        assert_eq!(geometry.region(800, 400), [0.0, -0.5, 1.0, 2.0]);
        assert_eq!(geometry.region(400, 800), [-0.5, 0.0, 2.0, 1.0]);
        assert!(InputGeometry::from_deepghs_config(br#"{"stages": []}"#).is_err());
    }

    fn initialize_test_runtime() {
        static RUNTIME: std::sync::Once = std::sync::Once::new();
        RUNTIME.call_once(|| {
            #[cfg(windows)]
            crate::runtime::initialize_from_dylib(
                &std::env::current_exe()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("onnxruntime/directml/onnxruntime.dll"),
            )
            .unwrap();
            #[cfg(not(windows))]
            crate::runtime::initialize_bundled_runtime(ExecutionProvider::Cpu).unwrap();
        });
    }

    /// Patch features load for `model`, produce the expected grid and region for a 320x200
    /// image, and leave the pooled embedding equal to `plain`'s.
    fn assert_patch_features(
        model: ImageEmbeddingModel,
        mut plain: ImageEmbedding,
        grid: (usize, usize, usize),
        region: [f64; 4],
    ) {
        let embedder = ImageEmbedder::load_cached(&ImageEmbedderOptions {
            model,
            ..ImageEmbedderOptions::default()
        })
        .unwrap()
        .expect("cached image model");
        assert!(embedder.supports_patch_features());
        let (width, height) = (320, 200);
        let rgb: Vec<u8> = (0..width * height * 3)
            .map(|i| (i * 7 % 251) as u8)
            .collect();
        let raster = crate::imaging::Raster::Rgb {
            width,
            height,
            pixels: rgb,
        };
        let features = embedder.patch_features(raster.clone()).unwrap();
        assert_eq!((features.rows, features.columns, features.dimensions), grid);
        assert_eq!(features.patches.len(), grid.0 * grid.1 * grid.2);
        assert_eq!(features.region, region);
        let batch = embedder
            .patch_features_batch(vec![raster.clone(), raster.clone()])
            .unwrap();
        assert_eq!(batch.len(), 2);
        for item in &batch {
            assert_eq!((item.rows, item.columns, item.dimensions), grid);
            assert_eq!(item.region, region);
            let patch_difference = item
                .patches
                .iter()
                .zip(&features.patches)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(patch_difference < 1e-5, "{patch_difference}");
        }

        let expected = plain
            .embed_preprocessed(vec![embedder.preprocess_raster(raster.clone()).unwrap()])
            .unwrap()
            .remove(0);
        for actual in [&features.embedding, &embedder.embed_raster(raster).unwrap()] {
            let difference = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(difference < 1e-5, "{difference}");
        }
    }

    #[test]
    #[ignore = "requires the MetaCLIP2 B/32 export cached by explicit setup"]
    fn patch_branch_leaves_the_indexed_embedding_unchanged() {
        initialize_test_runtime();
        let directory = ImageEmbeddingModel::MetaClip2B32
            .validated_model_directory(true, &())
            .unwrap()
            .unwrap();
        let plain = ImageEmbedding::try_new_from_path(
            directory.join("image.onnx"),
            &std::fs::read(directory.join("preprocessor_config.json")).unwrap(),
            fastembed::ImageInitOptionsUserDefined::new(),
        )
        .unwrap();
        assert_patch_features(
            ImageEmbeddingModel::MetaClip2B32,
            plain,
            (7, 7, 512),
            [0.0, 0.0, 1.0, 1.0],
        );
    }

    #[test]
    #[ignore = "downloads the four paired exports and runs both encoders"]
    fn published_external_exports_load_and_infer() {
        initialize_test_runtime();
        assert!(std::env::var_os("NICEGAL_LOCAL_MODELS_DIR").is_none());
        for model in [
            ImageEmbeddingModel::MetaClip2B32,
            ImageEmbeddingModel::MetaClip2B16,
            ImageEmbeddingModel::MetaClip2L14,
            ImageEmbeddingModel::SigLip2Base256,
        ] {
            eprintln!("Validating published export: {model}");
            let image = ImageEmbedder::load(&ImageEmbedderOptions {
                model,
                ..ImageEmbedderOptions::default()
            })
            .unwrap();
            let raster = crate::imaging::Raster::Rgb {
                width: 224,
                height: 224,
                pixels: vec![127; 224 * 224 * 3],
            };
            let embedding = image.embed_raster(raster.clone()).unwrap();
            assert_eq!(embedding.len(), model.dimensions());
            assert!(embedding.iter().all(|x| x.is_finite()));
            if model.supports_patch_features() {
                assert!(image.supports_patch_features());
                let patches = image.patch_features(raster).unwrap();
                assert_eq!(patches.dimensions, model.dimensions());
                assert!(patches.patches.iter().all(|x| x.is_finite()));
            }
            drop(image);
            let text = ImageQueryEmbedder::load(&ImageQueryEmbedderOptions {
                model,
                ..ImageQueryEmbedderOptions::default()
            })
            .unwrap();
            let query = text.embed_query("a photo of a cat").unwrap();
            assert_eq!(query.len(), model.dimensions());
            assert!(query.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    #[ignore = "requires the SigLIP SwinV2 checkpoint cached by explicit setup"]
    fn swinv2_feature_map_leaves_the_indexed_embedding_unchanged() {
        initialize_test_runtime();
        let model = ImageEmbeddingModel::SigLipBetaSwinV2Frozen;
        let (image, preprocessor) = model.deepghs_image_files(true, &()).unwrap().unwrap();
        let plain = ImageEmbedding::try_new_from_deepghs_path(
            image,
            &preprocessor,
            fastembed::ImageInitOptionsUserDefined::new(),
        )
        .unwrap();
        // 320x200 fits as 448x280: 84 rows of padding above and below.
        assert_patch_features(model, plain, (14, 14, 1024), [0.0, -0.3, 1.0, 1.6]);
    }

    #[test]
    #[ignore = "requires the DINOv3 ONNX export cached by explicit setup"]
    fn dinov3_patch_tokens_leave_the_indexed_embedding_unchanged() {
        initialize_test_runtime();
        let model = ImageEmbeddingModel::DinoV3B16;
        let directory = model.validated_model_directory(true, &()).unwrap().unwrap();
        let plain = ImageEmbedding::try_new_from_path(
            directory.join("image.onnx"),
            &std::fs::read(directory.join("preprocessor_config.json")).unwrap(),
            fastembed::ImageInitOptionsUserDefined::new(),
        )
        .unwrap()
        .with_output_key("pooler_output");
        assert_patch_features(model, plain, (14, 14, 768), [0.0, 0.0, 1.0, 1.0]);
    }
}
