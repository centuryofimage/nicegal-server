//! PaddleOCR text detection and recognition.
//!
//! Detector and recognizer models are loaded and run as a pair.

use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use tracing::{Span, field, info, instrument};

use crate::hub::ModelSource;
use crate::runtime::{
    ExecutionProvider, LoadedSessions, RuntimeOptions, SessionSpec, load_sessions,
};

mod inference;

use inference::{PaddleOcrConfig, PaddleOcrEngine, PaddleOcrScratch};
pub use inference::{PaddleOcrOptions, PaddleOcrOutput};

/// The source identity, ONNX graph, and preprocessing configuration for one OCR model.
#[derive(Clone, Copy)]
pub struct OcrModelFiles<'a> {
    pub source: &'a ModelSource,
    pub model_path: &'a Path,
    pub config_path: &'a Path,
}

/// The PP-OCRv6 detector and recognizer compiled for one ONNX Runtime execution provider.
pub struct PaddleOcrModels {
    detection_source: ModelSource,
    recognition_source: ModelSource,
    detection: Session,
    recognition: Session,
    config: PaddleOcrConfig,
    scratch: PaddleOcrScratch,
    execution_provider: ExecutionProvider,
}

// Keep the old positional API compatible while both implementations use file descriptors.
macro_rules! legacy_loaders {
    () => {
        #[deprecated(note = "use load_files with OcrModelFiles")]
        pub fn load(
            detection_source: ModelSource,
            detection_path: &Path,
            detection_config_path: &Path,
            recognition_source: ModelSource,
            recognition_path: &Path,
            recognition_config_path: &Path,
        ) -> Result<Self> {
            #[allow(deprecated)]
            Self::load_with_options(
                detection_source,
                detection_path,
                detection_config_path,
                recognition_source,
                recognition_path,
                recognition_config_path,
                RuntimeOptions::default(),
            )
        }

        #[deprecated(note = "use load_files with OcrModelFiles")]
        pub fn load_with_options(
            detection_source: ModelSource,
            detection_path: &Path,
            detection_config_path: &Path,
            recognition_source: ModelSource,
            recognition_path: &Path,
            recognition_config_path: &Path,
            options: RuntimeOptions,
        ) -> Result<Self> {
            Self::load_files(
                OcrModelFiles {
                    source: &detection_source,
                    model_path: detection_path,
                    config_path: detection_config_path,
                },
                OcrModelFiles {
                    source: &recognition_source,
                    model_path: recognition_path,
                    config_path: recognition_config_path,
                },
                options,
            )
        }
    };
}

impl PaddleOcrModels {
    legacy_loaders!();

    #[instrument(
        name = "paddle_ocr_load",
        skip_all,
        fields(
            detection_model = %detection_files.source.model_id,
            recognition_model = %recognition_files.source.model_id,
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
            detection_source: detection_files.source.clone(),
            recognition_source: recognition_files.source.clone(),
            detection,
            recognition,
            config,
            scratch: PaddleOcrScratch::default(),
            execution_provider,
        })
    }

    pub fn detection_source(&self) -> &ModelSource {
        &self.detection_source
    }

    pub fn recognition_source(&self) -> &ModelSource {
        &self.recognition_source
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
            ..
        } = self;
        PaddleOcrEngine::new(detection, recognition, config, scratch).scan(image, options)
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
    legacy_loaders!();

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

    pub fn detection_source(&self) -> &ModelSource {
        self.first().detection_source()
    }

    pub fn recognition_source(&self) -> &ModelSource {
        self.first().recognition_source()
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
