//! PaddleOCR text detection and recognition.
//!
//! Model files are resolved by [`crate::hub`] and compiled by [`crate::runtime`]. What is left
//! here is the pairing: the detector and the recognizer load, configure, and run together, and
//! neither is useful without the other.

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

impl PaddleOcrModels {
    pub fn load(
        detection_source: ModelSource,
        detection_path: &Path,
        detection_config_path: &Path,
        recognition_source: ModelSource,
        recognition_path: &Path,
        recognition_config_path: &Path,
    ) -> Result<Self> {
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

    #[instrument(
        name = "paddle_ocr_load",
        skip_all,
        fields(
            detection_model = %detection_source.model_id,
            recognition_model = %recognition_source.model_id,
            requested_execution_provider = %options.execution_provider,
            execution_provider = field::Empty
        )
    )]
    pub fn load_with_options(
        detection_source: ModelSource,
        detection_path: &Path,
        detection_config_path: &Path,
        recognition_source: ModelSource,
        recognition_path: &Path,
        recognition_config_path: &Path,
        options: RuntimeOptions,
    ) -> Result<Self> {
        let config = PaddleOcrConfig::load(detection_config_path, recognition_config_path)
            .context("loading PaddleOCR preprocessing and decoding configuration")?;

        let LoadedSessions {
            sessions: [detection, recognition],
            execution_provider,
        } = load_sessions(
            [
                SessionSpec::new("PaddleOCR detection", detection_path),
                SessionSpec::new("PaddleOCR recognition", recognition_path),
            ],
            options,
        )
        .context("compiling the PaddleOCR models")?;
        Span::current().record("execution_provider", field::display(execution_provider));
        info!(execution_provider = %execution_provider, "PaddleOCR models compiled");

        Ok(Self {
            detection_source,
            recognition_source,
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
    /// The caller owns decode scheduling so it can keep a bounded queue in front of ONNX Runtime
    /// without creating one session per worker. `ort` deliberately requires mutable sessions:
    /// some execution-provider allocators and statistics are not safe for concurrent runs.
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

/// Replicas are opt-in, and this is why.
///
/// Measured over 490 images of a real photo corpus (CPU, 4 intra-op threads): four replicas ran
/// 1.49x faster in wall time but burned 834s of CPU against a single replica's 311s, because
/// per-image detection inference went from 362ms to 928ms. Four replicas times four intra-op
/// threads is sixteen ORT threads, all spinning — ORT only stops spinning when a session is given
/// a single thread — so most of the gain goes back into contending for cores.
///
/// Extra sessions are the wrong knob for that. ONNX Runtime already parallelizes one inference
/// across its intra-op pool, and the default of four threads is precisely why a single replica
/// used only 1.7 of 20 cores; raise [`RuntimeOptions::intra_threads`] to use a bigger machine.
/// [`RuntimeOptions::replicas`] stays so the trade can be re-measured on other hardware.
const DEFAULT_REPLICAS: usize = 1;

/// Several independent copies of the detector/recognizer pair, so more than one image can be in
/// inference at a time.
///
/// `ort` deliberately requires `&mut Session`, so one pair can only ever hold one image. On CPU
/// that pinned the whole OCR phase to a single consumer thread — measured at 1.7 of 20 cores on a
/// 1,770-image run — and let one slow image stall every decoded image queued behind it. The two
/// models are about 10 MB each, so a few copies cost far less memory than the throughput they buy.
///
/// Accelerated providers get exactly one replica: the device is already the parallel unit there,
/// and extra sessions would only duplicate its memory and contend for it.
pub struct PaddleOcrPool {
    replicas: Vec<PaddleOcrModels>,
}

impl PaddleOcrPool {
    pub fn load(
        detection_source: ModelSource,
        detection_path: &Path,
        detection_config_path: &Path,
        recognition_source: ModelSource,
        recognition_path: &Path,
        recognition_config_path: &Path,
    ) -> Result<Self> {
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

    /// Compile the pair, then compile any further replicas against the provider the first one
    /// actually got. Reading the count from the configured provider rather than the requested one
    /// keeps a DirectML request that quietly fell back to CPU from being left with a single
    /// replica.
    pub fn load_with_options(
        detection_source: ModelSource,
        detection_path: &Path,
        detection_config_path: &Path,
        recognition_source: ModelSource,
        recognition_path: &Path,
        recognition_config_path: &Path,
        options: RuntimeOptions,
    ) -> Result<Self> {
        let first = PaddleOcrModels::load_with_options(
            detection_source.clone(),
            detection_path,
            detection_config_path,
            recognition_source.clone(),
            recognition_path,
            recognition_config_path,
            options,
        )?;
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
            replicas.push(PaddleOcrModels::load_with_options(
                detection_source.clone(),
                detection_path,
                detection_config_path,
                recognition_source.clone(),
                recognition_path,
                recognition_config_path,
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
