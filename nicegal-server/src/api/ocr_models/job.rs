use crate::api::jobs::cancel_if;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hf_hub::api::tokio::Progress;
use nicegal_core::hub::{self, DownloadObserver, ModelSource};
use nicegal_core::index::IndexObserver;
use nicegal_core::ocr::{OcrModelFiles, PaddleOcrPool};
use nicegal_core::runtime::RuntimeOptions;
use serde::Deserialize;

use super::ModelStore;
use crate::api::RuntimeSettings;
use crate::api::error::ApiError;
use crate::api::jobs::Job;

const DEFAULT_MODEL_FILENAME: &str = "inference.onnx";
const DEFAULT_CONFIG_FILENAME: &str = "inference.yml";

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    detection: ModelRequest,
    recognition: ModelRequest,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ModelRequest {
    model_id: String,
    revision: Option<String>,
    #[serde(default = "default_model_filename")]
    filename: String,
    #[serde(default = "default_config_filename")]
    config_filename: String,
}

#[derive(Clone)]
pub(crate) struct Spec {
    detection: ModelSource,
    detection_config: ModelSource,
    recognition: ModelSource,
    recognition_config: ModelSource,
}

impl Spec {
    pub(crate) fn request(&self) -> Request {
        fn model(source: &ModelSource, config: &ModelSource) -> ModelRequest {
            ModelRequest {
                model_id: source.model_id.clone(),
                revision: source.revision.clone(),
                filename: source.filename.clone(),
                config_filename: config.filename.clone(),
            }
        }
        Request {
            detection: model(&self.detection, &self.detection_config),
            recognition: model(&self.recognition, &self.recognition_config),
        }
    }
    /// Whether this exact pair is the one already loaded, so a scan can skip reloading it.
    pub(crate) fn is_loaded_in(&self, store: &ModelStore) -> bool {
        store.is_loaded_with(&self.detection, &self.recognition)
    }
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    let (detection, detection_config) = prepare_source("detection", request.detection)?;
    let (recognition, recognition_config) = prepare_source("recognition", request.recognition)?;
    Ok(Spec {
        detection,
        detection_config,
        recognition,
        recognition_config,
    })
}

fn prepare_source(
    kind: &str,
    request: ModelRequest,
) -> Result<(ModelSource, ModelSource), ApiError> {
    let model_id = request.model_id.trim();
    if model_id.is_empty() || !model_id.contains('/') {
        return Err(ApiError::bad_request(format!(
            "{kind} modelId must be a Hugging Face owner/repository ID"
        )));
    }
    if request
        .revision
        .as_ref()
        .is_some_and(|revision| revision.trim().is_empty())
    {
        return Err(ApiError::bad_request(format!(
            "{kind} model revision must not be empty"
        )));
    }
    validate_repository_path(kind, "model filename", &request.filename)?;
    validate_repository_path(kind, "configFilename", &request.config_filename)?;

    let source = ModelSource {
        model_id: model_id.to_owned(),
        revision: request.revision,
        filename: request.filename,
    };
    let config = ModelSource {
        model_id: source.model_id.clone(),
        revision: source.revision.clone(),
        filename: request.config_filename,
    };
    Ok((source, config))
}

fn validate_repository_path(kind: &str, field: &str, value: &str) -> Result<(), ApiError> {
    if value.is_empty()
        || value.starts_with(['/', '\\'])
        || value.split(['/', '\\']).any(|part| part == "..")
    {
        return Err(ApiError::bad_request(format!(
            "{kind} {field} must be a relative repository path"
        )));
    }
    Ok(())
}

pub(crate) async fn run(
    spec: Spec,
    store: &ModelStore,
    runtime: &Arc<RuntimeSettings>,
    job: Arc<Job>,
) -> anyhow::Result<()> {
    let cache = hub::cache();
    let api = hub::api()?;

    job.downloading_models();
    let detection_path = spec
        .detection
        .get_with_progress(
            &api,
            &cache,
            ModelDownloadProgress::new(Arc::clone(&job), spec.detection.clone()),
        )
        .await?;
    job.model_download_complete();
    cancel_if(job.is_cancelled())?;
    let detection_config_path = spec
        .detection_config
        .get_with_progress(
            &api,
            &cache,
            ModelDownloadProgress::new(Arc::clone(&job), spec.detection_config.clone()),
        )
        .await?;
    cancel_if(job.is_cancelled())?;

    let recognition_path = spec
        .recognition
        .get_with_progress(
            &api,
            &cache,
            ModelDownloadProgress::new(Arc::clone(&job), spec.recognition.clone()),
        )
        .await?;
    job.model_download_complete();
    cancel_if(job.is_cancelled())?;
    let recognition_config_path = spec
        .recognition_config
        .get_with_progress(
            &api,
            &cache,
            ModelDownloadProgress::new(Arc::clone(&job), spec.recognition_config.clone()),
        )
        .await?;
    cancel_if(job.is_cancelled())?;

    job.loading_models();
    let detection = spec.detection;
    let recognition = spec.recognition;
    let runtime_options = runtime_options(store, runtime);
    let models = tokio::task::spawn_blocking(move || {
        PaddleOcrPool::load_files(
            OcrModelFiles {
                source: &detection,
                model_path: &detection_path,
                config_path: &detection_config_path,
            },
            OcrModelFiles {
                source: &recognition,
                model_path: &recognition_path,
                config_path: &recognition_config_path,
            },
            runtime_options,
        )
    })
    .await
    .map_err(|error| anyhow::anyhow!("model loader worker failed: {error}"))??;
    job.models_loaded(2);
    let actual = models.execution_provider();
    runtime.accept_loaded_provider(actual)?;
    store.replace(models);
    Ok(())
}

fn runtime_options(store: &ModelStore, runtime: &RuntimeSettings) -> RuntimeOptions {
    runtime.model_runtime_options(RuntimeOptions {
        execution_provider: store.requested_execution_provider(),
        ..RuntimeOptions::default()
    })
}

#[derive(Clone)]
struct ModelDownloadProgress {
    job: Arc<Job>,
    source: ModelSource,
    // hf-hub clones progress for concurrent chunks; all clones share counters and throttling.
    state: Arc<Mutex<DownloadState>>,
}

struct DownloadState {
    downloaded: usize,
    total: usize,
    reported: usize,
    reported_total: usize,
    last_update: Instant,
}

impl ModelDownloadProgress {
    fn new(job: Arc<Job>, source: ModelSource) -> Self {
        Self {
            job,
            source,
            state: Arc::new(Mutex::new(DownloadState {
                downloaded: 0,
                total: 0,
                reported: 0,
                reported_total: 0,
                last_update: Instant::now(),
            })),
        }
    }

    fn report(&self, state: &mut DownloadState) {
        self.job.ocr_download_progress(
            &self.source,
            state.downloaded,
            state.total,
            state.reported,
            state.reported_total,
        );
        state.reported = state.downloaded;
        state.reported_total = state.total;
        state.last_update = Instant::now();
    }
}

impl Progress for ModelDownloadProgress {
    fn is_cancelled(&self) -> bool {
        DownloadObserver::download_cancelled(self.job.as_ref())
    }

    async fn init(&mut self, size: usize, _filename: &str) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.downloaded = 0;
        state.total = size;
        self.report(&mut state);
    }

    async fn update(&mut self, size: usize) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.downloaded = state.downloaded.saturating_add(size);
        if state.last_update.elapsed() >= Duration::from_millis(100) {
            self.report(&mut state);
        }
    }

    async fn finish(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.report(&mut state);
        DownloadObserver::finish(self.job.as_ref());
    }
}

fn default_model_filename() -> String {
    DEFAULT_MODEL_FILENAME.to_owned()
}

fn default_config_filename() -> String {
    DEFAULT_CONFIG_FILENAME.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nicegal_core::runtime::ExecutionProvider;
    use tempfile::TempDir;

    #[test]
    fn a_provider_that_worked_before_cannot_fall_back_after_restart() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let path = camino::Utf8PathBuf::try_from(temp.path().join("runtime.json"))?;
        let store = ModelStore::new(ExecutionProvider::Directml);
        let runtime = RuntimeSettings::load(path.clone(), None)?;
        assert!(runtime_options(&store, &runtime).allow_cpu_fallback);
        runtime.record_working_provider(ExecutionProvider::Directml)?;

        let restarted = RuntimeSettings::load(path, None)?;
        assert!(!runtime_options(&store, &restarted).allow_cpu_fallback);
        Ok(())
    }

    #[test]
    fn model_ids_are_required_and_not_defaulted() {
        let request: Request = serde_json::from_str(
            r#"{
                "detection": { "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx" },
                "recognition": { "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx" }
            }"#,
        )
        .unwrap();
        let spec = prepare(request).unwrap();
        assert_eq!(spec.detection.filename, DEFAULT_MODEL_FILENAME);
        assert_eq!(spec.detection_config.filename, DEFAULT_CONFIG_FILENAME);
        assert_eq!(spec.recognition.filename, DEFAULT_MODEL_FILENAME);
        assert_eq!(spec.recognition_config.filename, DEFAULT_CONFIG_FILENAME);
    }

    #[test]
    fn repository_paths_cannot_escape_the_cache() {
        let request = ModelRequest {
            model_id: "owner/model".to_owned(),
            revision: None,
            filename: "../model.onnx".to_owned(),
            config_filename: DEFAULT_CONFIG_FILENAME.to_owned(),
        };
        assert!(prepare_source("detection", request).is_err());
    }
}
