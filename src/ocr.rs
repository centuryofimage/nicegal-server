//! PaddleOCR text detection and recognition.
//!
//! Detector and recognizer models are loaded and run as a pair. There is one pair, PP-OCRv6 small.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ort::session::Session;
use tracing::{Span, field, info, instrument};

use crate::hub::{DownloadObserver, ModelSource};
use crate::runtime::{
    ExecutionProvider, LoadedSessions, RuntimeOptions, SessionSpec, load_sessions,
};

mod inference;

use inference::{PaddleOcrConfig, PaddleOcrEngine, PaddleOcrScratch};
pub use inference::{PaddleOcrOptions, PaddleOcrOutput};

/// PP-OCRv6 small: the detector and recognizer every OCR result is recognized with.
const DETECTION_REPOSITORY: &str = "PaddlePaddle/PP-OCRv6_small_det_onnx";
const RECOGNITION_REPOSITORY: &str = "PaddlePaddle/PP-OCRv6_small_rec_onnx";

/// The ONNX graph and preprocessing configuration for one OCR model.
#[derive(Clone, Copy)]
pub struct OcrModelFiles<'a> {
    pub model_path: &'a Path,
    pub config_path: &'a Path,
}

/// [`OcrModelFiles`] with owned paths, as resolved from the Hugging Face cache.
struct ResolvedFiles {
    model: PathBuf,
    config: PathBuf,
}

impl ResolvedFiles {
    fn files(&self) -> OcrModelFiles<'_> {
        OcrModelFiles {
            model_path: &self.model,
            config_path: &self.config,
        }
    }
}

/// `repository`'s graph and configuration, downloading missing ones unless `cached_only`; `None`
/// means one is not cached.
fn resolve(
    repository: &str,
    cached_only: bool,
    progress: &dyn DownloadObserver,
) -> Result<Option<ResolvedFiles>> {
    let source = |filename: &str| ModelSource {
        model_id: repository.to_owned(),
        revision: Some("main".to_owned()),
        filename: filename.to_owned(),
    };
    let Some(model) = source("inference.onnx").resolve(cached_only, progress)? else {
        return Ok(None);
    };
    let Some(config) = source("inference.yml").resolve(cached_only, progress)? else {
        return Ok(None);
    };
    Ok(Some(ResolvedFiles { model, config }))
}

/// Both models' files, see [`resolve`].
fn resolve_pair(
    cached_only: bool,
    progress: &dyn DownloadObserver,
) -> Result<Option<[ResolvedFiles; 2]>> {
    let Some(detection) = resolve(DETECTION_REPOSITORY, cached_only, progress)? else {
        return Ok(None);
    };
    let Some(recognition) = resolve(RECOGNITION_REPOSITORY, cached_only, progress)? else {
        return Ok(None);
    };
    Ok(Some([detection, recognition]))
}

/// The PP-OCRv6 detector and recognizer compiled for one ONNX Runtime execution provider.
pub struct PaddleOcrModels {
    detection: Session,
    recognition: Session,
    config: PaddleOcrConfig,
    scratch: PaddleOcrScratch,
    execution_provider: ExecutionProvider,
}

impl PaddleOcrModels {
    #[instrument(
        name = "paddle_ocr_load",
        skip_all,
        fields(
            requested_execution_provider = %options.execution_provider,
            execution_provider = field::Empty
        )
    )]
    pub fn load_files(
        detection_files: OcrModelFiles<'_>,
        recognition_files: OcrModelFiles<'_>,
        options: RuntimeOptions,
    ) -> Result<Self> {
        let config =
            PaddleOcrConfig::load(detection_files.config_path, recognition_files.config_path)
                .context("loading PaddleOCR preprocessing and decoding configuration")?;

        let LoadedSessions {
            sessions: [detection, recognition],
            execution_provider,
        } = load_sessions(
            [
                SessionSpec::new("PaddleOCR detection", detection_files.model_path),
                SessionSpec::new("PaddleOCR recognition", recognition_files.model_path),
            ],
            options,
        )
        .context("compiling the PaddleOCR models")?;
        Span::current().record("execution_provider", field::display(execution_provider));
        info!(execution_provider = %execution_provider, "PaddleOCR models compiled");

        Ok(Self {
            detection,
            recognition,
            config,
            scratch: PaddleOcrScratch::default(),
            execution_provider,
        })
    }

    /// The execution provider successfully configured for both OCR sessions.
    pub fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }

    /// Detect and recognize the text in one already-decoded RGB image.
    ///
    /// The caller owns decode scheduling. `ort` requires mutable session access.
    pub fn scan(
        &mut self,
        image: &image::RgbImage,
        options: PaddleOcrOptions,
    ) -> Result<PaddleOcrOutput> {
        let Self {
            detection,
            recognition,
            config,
            scratch,
            execution_provider,
            ..
        } = self;
        PaddleOcrEngine::new(detection, recognition, config, scratch, *execution_provider)
            .scan(image, options)
    }
}

/// One replica avoids multiplying ONNX Runtime's intra-op thread pools by default. Callers can
/// opt into concurrent sessions through [`RuntimeOptions::replicas`].
const DEFAULT_REPLICAS: usize = 1;

/// Independent detector/recognizer pairs for concurrent inference. The count comes from
/// [`RuntimeOptions::replicas`] and defaults to one.
pub struct PaddleOcrPool {
    replicas: Vec<PaddleOcrModels>,
}

impl PaddleOcrPool {
    /// The pair's Hugging Face repositories, for display.
    pub const MODEL_ID: &'static str =
        "PaddlePaddle/PP-OCRv6_small_det_onnx + PaddlePaddle/PP-OCRv6_small_rec_onnx";

    /// Compile the pair, downloading missing files first.
    pub fn load(options: RuntimeOptions, progress: &dyn DownloadObserver) -> Result<Self> {
        let [detection, recognition] =
            resolve_pair(false, progress)?.context("a download resolves every file")?;
        Self::load_files(detection.files(), recognition.files(), options)
    }

    /// Compile the pair from cached files only; `None` means a file is missing.
    pub fn load_cached(options: RuntimeOptions) -> Result<Option<Self>> {
        resolve_pair(true, &())?
            .map(|[detection, recognition]| {
                Self::load_files(detection.files(), recognition.files(), options)
            })
            .transpose()
    }

    /// Download any missing file without compiling, or with `cached_only` report whether every
    /// file is cached.
    pub fn fetch(cached_only: bool, progress: &dyn DownloadObserver) -> Result<bool> {
        Ok(resolve_pair(cached_only, progress)?.is_some())
    }

    /// Compile the requested number of pairs, reusing their file descriptors.
    /// Every replica after the first is pinned to the provider the first pair obtained,
    /// so one pool never mixes providers.
    pub fn load_files(
        detection: OcrModelFiles<'_>,
        recognition: OcrModelFiles<'_>,
        options: RuntimeOptions,
    ) -> Result<Self> {
        let first = PaddleOcrModels::load_files(detection, recognition, options)?;
        let configured = first.execution_provider();
        let target = replica_target(options);
        let mut replicas = Vec::with_capacity(target.get());
        replicas.push(first);
        // The provider is already settled, so a later replica must not silently land somewhere
        // else: pin it and refuse the fallback rather than mix providers inside one pool.
        let replica_options = RuntimeOptions {
            execution_provider: configured,
            allow_cpu_fallback: false,
            ..options
        };
        while replicas.len() < target.get() {
            replicas.push(PaddleOcrModels::load_files(
                detection,
                recognition,
                replica_options,
            )?);
        }
        info!(
            replicas = replicas.len(),
            execution_provider = %configured,
            "PaddleOCR pool ready"
        );
        Ok(Self { replicas })
    }

    pub fn execution_provider(&self) -> ExecutionProvider {
        self.first().execution_provider()
    }

    /// How many images this pool can hold in inference at once.
    pub fn replicas(&self) -> usize {
        self.replicas.len()
    }

    /// The replicas themselves, for a caller that wants to drive one per worker thread.
    pub fn replicas_mut(&mut self) -> &mut [PaddleOcrModels] {
        &mut self.replicas
    }

    /// Scan one image on a single replica, for callers with nothing to parallelize.
    pub fn scan(
        &mut self,
        image: &image::RgbImage,
        options: PaddleOcrOptions,
    ) -> Result<PaddleOcrOutput> {
        self.replicas[0].scan(image, options)
    }

    fn first(&self) -> &PaddleOcrModels {
        &self.replicas[0]
    }
}

fn replica_target(options: RuntimeOptions) -> NonZeroUsize {
    options
        .replicas
        .unwrap_or(NonZeroUsize::new(DEFAULT_REPLICAS).expect("the default is non-zero"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(provider: ExecutionProvider, intra_threads: usize) -> RuntimeOptions {
        RuntimeOptions {
            execution_provider: provider,
            intra_threads: NonZeroUsize::new(intra_threads).unwrap(),
            ..RuntimeOptions::default()
        }
    }

    #[test]
    fn every_provider_loads_one_replica_by_default() {
        for provider in [ExecutionProvider::Cpu, ExecutionProvider::Directml] {
            assert_eq!(replica_target(options(provider, 4)).get(), 1);
        }
    }

    #[test]
    fn a_benchmark_can_still_ask_for_more_replicas() {
        let forced = RuntimeOptions {
            replicas: NonZeroUsize::new(7),
            ..options(ExecutionProvider::Cpu, 4)
        };
        assert_eq!(replica_target(forced).get(), 7);
    }
}
