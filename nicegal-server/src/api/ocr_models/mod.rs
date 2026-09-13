pub(super) mod job;

use std::sync::{Arc, Mutex, MutexGuard};

use axum::Json;
use axum::extract::State;
use axum::routing::{MethodRouter, get};
use nicegal_core::hub::ModelSource;
use nicegal_core::ocr::PaddleOcrPool;
use nicegal_core::runtime::ExecutionProvider;
use serde::Serialize;

use super::AppState;

pub(crate) struct ModelStore {
    inner: Mutex<Option<LoadedModels>>,
    /// The provider every `ocrModelLoad` job asks for; `PaddleOcrPool` falls back to CPU on its
    /// own if this one is unavailable or fails to compile.
    requested_execution_provider: ExecutionProvider,
}

struct LoadedModels {
    detection: ModelSource,
    recognition: ModelSource,
    /// The provider actually compiled for, which is CPU whenever the load fell back. Kept here so
    /// reading it does not take the session lock away from whatever is inferring.
    execution_provider: ExecutionProvider,
    sessions: Arc<Mutex<PaddleOcrPool>>,
}

impl ModelStore {
    pub(crate) fn new(requested_execution_provider: ExecutionProvider) -> Self {
        Self {
            inner: Mutex::new(None),
            requested_execution_provider,
        }
    }

    pub(super) fn requested_execution_provider(&self) -> ExecutionProvider {
        self.requested_execution_provider
    }

    pub(super) fn replace(&self, models: PaddleOcrPool) {
        let detection = models.detection_source().clone();
        let recognition = models.recognition_source().clone();
        let execution_provider = models.execution_provider();
        *self.models() = Some(LoadedModels {
            detection,
            recognition,
            execution_provider,
            sessions: Arc::new(Mutex::new(models)),
        });
    }

    fn models(&self) -> MutexGuard<'_, Option<LoadedModels>> {
        self.inner.lock().unwrap_or_else(|error| error.into_inner())
    }

    pub(super) fn is_loaded(&self) -> bool {
        self.models().is_some()
    }

    pub(super) fn snapshot(&self) -> Option<Arc<Mutex<PaddleOcrPool>>> {
        self.models()
            .as_ref()
            .map(|models| Arc::clone(&models.sessions))
    }

    fn status(&self) -> StatusResponse {
        let models = self.models();
        let Some(models) = models.as_ref() else {
            return StatusResponse { loaded: None };
        };
        StatusResponse {
            loaded: Some(LoadedModelsResponse {
                detection: ModelSourceResponse::from(&models.detection),
                recognition: ModelSourceResponse::from(&models.recognition),
                execution_provider: models.execution_provider.into(),
            }),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded: Option<LoadedModelsResponse>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoadedModelsResponse {
    detection: ModelSourceResponse,
    recognition: ModelSourceResponse,
    execution_provider: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelSourceResponse {
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    revision: Option<String>,
    filename: String,
}

impl From<&ModelSource> for ModelSourceResponse {
    fn from(source: &ModelSource) -> Self {
        Self {
            model_id: source.model_id.clone(),
            revision: source.revision.clone(),
            filename: source.filename.clone(),
        }
    }
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(status)
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    Json(state.ocr_models.status())
}
