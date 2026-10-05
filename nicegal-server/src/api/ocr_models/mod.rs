pub(super) mod job;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::model_slot::ModelSlot;

use axum::Json;
use axum::extract::State;
use axum::routing::{MethodRouter, get};
use nicegal_core::hub::ModelSource;
use nicegal_core::ocr::PaddleOcrPool;
use nicegal_core::runtime::ExecutionProvider;
use serde::Serialize;

use super::AppState;

pub(crate) struct ModelStore {
    slot: ModelSlot<LoadedModels, job::Spec>,
    /// The provider every `ocrModelLoad` job asks for; `PaddleOcrPool` falls back to CPU on its
    /// own if this one is unavailable or fails to compile.
    requested_execution_provider: ExecutionProvider,
}

pub(super) struct LoadedModels {
    detection: ModelSource,
    recognition: ModelSource,
    /// The provider actually compiled for, which is CPU whenever the load fell back. Kept here so
    /// reading it does not take the session lock away from whatever is inferring.
    execution_provider: ExecutionProvider,
    pub(super) sessions: Mutex<PaddleOcrPool>,
}

impl LoadedModels {
    pub(super) fn new(models: PaddleOcrPool) -> Self {
        Self {
            detection: models.detection_source().clone(),
            recognition: models.recognition_source().clone(),
            execution_provider: models.execution_provider(),
            sessions: Mutex::new(models),
        }
    }
}

impl ModelStore {
    pub(crate) fn new(requested_execution_provider: ExecutionProvider) -> Self {
        Self {
            slot: ModelSlot::new("OCR models"),
            requested_execution_provider,
        }
    }

    pub(super) fn requested_execution_provider(&self) -> ExecutionProvider {
        self.requested_execution_provider
    }

    pub(super) fn is_loaded(&self) -> bool {
        self.slot.inspect(|models| models.is_some())
    }

    pub(super) fn snapshot(&self) -> Option<Arc<LoadedModels>> {
        self.slot.snapshot()
    }

    pub(super) fn unload_idle(&self, now: Instant) {
        self.slot.unload_idle(now);
    }

    fn status(&self) -> StatusResponse {
        self.slot.inspect(|models| StatusResponse {
            loaded: models.map(|models| LoadedModelsResponse {
                detection: ModelSourceResponse::from(&models.detection),
                recognition: ModelSourceResponse::from(&models.recognition),
                execution_provider: models.execution_provider.into(),
            }),
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn ocr_reuse_requires_both_models_and_configs_and_protects_the_acquired_pair() {
        let request = serde_json::json!({
            "detection": { "modelId": "test/det", "revision": "main" },
            "recognition": { "modelId": "test/rec", "revision": "main" }
        });
        let prepare = |value| job::prepare(serde_json::from_value(value).unwrap()).unwrap();
        let spec = prepare(request.clone());
        let slot = ModelSlot::new("fake OCR models");
        drop(slot.prepare(spec.clone(), &(), || Ok(Some(42))).unwrap());
        let now = Instant::now();
        slot.unload_idle(now);
        for role in ["detection", "recognition"] {
            for (field, value) in [
                ("modelId", "other/model"),
                ("revision", "other"),
                ("filename", "other.onnx"),
                ("configFilename", "other.yml"),
            ] {
                let mut changed = request.clone();
                changed[role][field] = value.into();
                assert!(
                    slot.acquire(&prepare(changed)).is_none(),
                    "must reload for {role}.{field}"
                );
            }
        }
        // Failed matches do not keep the old pair awake.
        slot.unload_idle(now + Duration::from_secs(300));
        assert!(slot.acquire(&spec).is_none());
        let active = slot
            .prepare(spec.clone(), &(), || Ok(Some(43)))
            .unwrap()
            .unwrap();
        slot.unload_idle(now + Duration::from_secs(900));
        assert!(Arc::ptr_eq(&active, &slot.acquire(&spec).unwrap()));
        drop(active);
        slot.unload_idle(now + Duration::from_secs(900));
        slot.unload_idle(now + Duration::from_secs(1200));
        assert!(slot.acquire(&spec).is_none());
    }
}
