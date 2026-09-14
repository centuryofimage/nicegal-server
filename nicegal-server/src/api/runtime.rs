use std::fs;
use std::io::Write as _;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{MethodRouter, get};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::runtime::ExecutionProvider;
use serde::{Deserialize, Serialize};

use super::AppState;
use super::error::ApiError;
use super::extract::ApiJson;

/// The runtime selection read at server startup and the persisted selection for its next launch.
/// ONNX Runtime binds to its loaded DLL process-wide, so changing this setting deliberately does
/// not try to replace sessions in a live process.
pub(crate) struct RuntimeSettings {
    path: PathBuf,
    active_execution_provider: ExecutionProvider,
    onnx_runtime_build_info: Mutex<String>,
    configured_execution_provider: Mutex<ExecutionProvider>,
}

impl RuntimeSettings {
    pub(crate) fn load(
        path: PathBuf,
        command_line_provider: Option<ExecutionProvider>,
    ) -> Result<Self> {
        let configured_execution_provider = read_provider(&path)?;
        Ok(Self {
            path,
            active_execution_provider: command_line_provider
                .unwrap_or(configured_execution_provider),
            onnx_runtime_build_info: Mutex::new(String::new()),
            configured_execution_provider: Mutex::new(configured_execution_provider),
        })
    }

    pub(crate) fn active_execution_provider(&self) -> ExecutionProvider {
        self.active_execution_provider
    }

    pub(crate) fn set_onnx_runtime_build_info(&self, build_info: String) {
        *self
            .onnx_runtime_build_info
            .lock()
            .expect("runtime settings mutex poisoned") = build_info;
    }

    fn status(&self) -> RuntimeStatusResponse {
        let configured_execution_provider = *self
            .configured_execution_provider
            .lock()
            .expect("runtime settings mutex poisoned");
        RuntimeStatusResponse {
            active_execution_provider: self.active_execution_provider.to_string(),
            active_runtime_distribution: runtime_distribution(self.active_execution_provider)
                .to_owned(),
            onnx_runtime_build_info: self
                .onnx_runtime_build_info
                .lock()
                .expect("runtime settings mutex poisoned")
                .clone(),
            configured_execution_provider: configured_execution_provider.to_string(),
            restart_required: self.active_execution_provider != configured_execution_provider,
            image_model: None,
        }
    }

    pub(super) fn set(
        &self,
        execution_provider: ExecutionProvider,
    ) -> Result<RuntimeStatusResponse> {
        write_provider(&self.path, execution_provider)?;
        *self
            .configured_execution_provider
            .lock()
            .expect("runtime settings mutex poisoned") = execution_provider;
        Ok(self.status())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RuntimeStatusResponse {
    active_execution_provider: String,
    active_runtime_distribution: String,
    onnx_runtime_build_info: String,
    configured_execution_provider: String,
    restart_required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_model: Option<super::image_model::ModelStatus>,
}

/// CPU models can execute against either accelerated distribution. The Windows server uses the
/// DirectML distribution for CPU selection because it is the default general-purpose bundle.
fn runtime_distribution(execution_provider: ExecutionProvider) -> &'static str {
    if cfg!(target_os = "linux") {
        return "openvino";
    }
    match execution_provider {
        ExecutionProvider::OpenVino => "openvino",
        ExecutionProvider::Cpu | ExecutionProvider::Directml => "directml",
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeUpdateRequest {
    execution_provider: Option<String>,
    image_model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeSettingsFile {
    execution_provider: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeSettingsFileRef {
    execution_provider: String,
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(status).put(update)
}

pub(super) fn status_response(state: &AppState) -> RuntimeStatusResponse {
    let mut status = state.runtime.status();
    let model = state.image_model_settings.status();
    status.restart_required |= model.restart_required;
    status.image_model = Some(model);
    status
}

async fn status(State(state): State<AppState>) -> Json<RuntimeStatusResponse> {
    Json(status_response(&state))
}

async fn update(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<RuntimeUpdateRequest>,
) -> Result<(StatusCode, Json<RuntimeStatusResponse>), ApiError> {
    if request.execution_provider.is_none() && request.image_model.is_none() {
        return Err(ApiError::bad_request(
            "Provide executionProvider or imageModel",
        ));
    }
    let execution_provider = request
        .execution_provider
        .as_deref()
        .map(ExecutionProvider::from_str)
        .transpose()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let model = request
        .image_model
        .as_deref()
        .map(str::parse::<nicegal_core::embedding::ImageEmbeddingModel>)
        .transpose()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    if let Some(model) = model {
        if !nicegal_core::embedding::ImageEmbeddingModel::SELECTABLE.contains(&model) {
            return Err(ApiError::bad_request(
                "This model is retired from the model selector",
            ));
        }
        if !model.available() {
            return Err(ApiError::bad_request("Image model is unavailable"));
        }
        if state.jobs.has_active_job() {
            return Err(ApiError::job_busy());
        }
    }
    let runtime = Arc::clone(&state.runtime);
    let settings = Arc::clone(&state.image_model_settings);
    tokio::task::spawn_blocking(move || -> Result<()> {
        if let Some(provider) = execution_provider {
            runtime.set(provider)?;
        }
        if let Some(model) = model {
            settings.set(model)?;
        }
        Ok(())
    })
    .await
    .map_err(|error| ApiError::internal(anyhow!("runtime settings task failed: {error}")))?
    .map_err(ApiError::internal)?;
    Ok((StatusCode::OK, Json(status_response(&state))))
}

fn read_provider(path: &PathBuf) -> Result<ExecutionProvider> {
    let contents = match fs::read(path) {
        Ok(contents) => contents,
        // Prefer each platform's bundled accelerator, with CPU fallback when unavailable.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(if cfg!(windows) {
                ExecutionProvider::Directml
            } else if cfg!(target_os = "linux") {
                ExecutionProvider::OpenVino
            } else {
                ExecutionProvider::Cpu
            });
        }
        Err(error) => {
            return Err(error).with_context(|| format!("reading runtime settings: {path}"));
        }
    };
    let settings: RuntimeSettingsFile = serde_json::from_slice(&contents)
        .with_context(|| format!("parsing runtime settings: {path}"))?;
    ExecutionProvider::from_str(&settings.execution_provider)
        .map_err(|error| anyhow!("invalid execution provider in runtime settings {path}: {error}"))
}

fn write_provider(path: &PathBuf, execution_provider: ExecutionProvider) -> Result<()> {
    let parent = path
        .parent()
        .context("runtime settings path has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating runtime settings directory: {parent}"))?;

    let contents = serde_json::to_vec_pretty(&RuntimeSettingsFileRef {
        execution_provider: execution_provider.to_string(),
    })
    .context("serializing runtime settings")?;
    let temporary = path.with_extension("json.tmp");
    let mut file = fs::File::create(&temporary)
        .with_context(|| format!("creating temporary runtime settings: {temporary}"))?;
    file.write_all(&contents)
        .with_context(|| format!("writing temporary runtime settings: {temporary}"))?;
    file.sync_all()
        .with_context(|| format!("syncing temporary runtime settings: {temporary}"))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("replacing runtime settings {path} with {temporary}"))
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn missing_settings_use_platform_default_and_persist_updates() {
        let temp = TempDir::new().unwrap();
        let path = PathBuf::try_from(temp.path().join("runtime.json")).unwrap();
        let settings = RuntimeSettings::load(path.clone(), None).unwrap();
        settings.set_onnx_runtime_build_info("test build".to_owned());
        let expected = if cfg!(windows) {
            "directml"
        } else if cfg!(target_os = "linux") {
            "openvino"
        } else {
            "cpu"
        };
        assert_eq!(settings.status().active_execution_provider, expected);
        assert_eq!(settings.status().configured_execution_provider, expected);
        if cfg!(target_os = "linux") {
            assert_eq!(settings.status().active_runtime_distribution, "openvino");
        }
        assert!(!settings.status().restart_required);

        let status = settings.set(ExecutionProvider::Cpu).unwrap();
        assert_eq!(status.active_execution_provider, expected);
        assert_eq!(status.configured_execution_provider, "cpu");
        assert_eq!(status.restart_required, expected != "cpu");

        let status = settings.set(ExecutionProvider::Directml).unwrap();
        assert_eq!(status.configured_execution_provider, "directml");

        let restarted = RuntimeSettings::load(path, None).unwrap();
        assert_eq!(restarted.status().active_execution_provider, "directml");
        assert!(!restarted.status().restart_required);
    }

    #[test]
    fn command_line_provider_overrides_saved_setting_for_this_launch() {
        let temp = TempDir::new().unwrap();
        let path = PathBuf::try_from(temp.path().join("runtime.json")).unwrap();
        write_provider(&path, ExecutionProvider::OpenVino).unwrap();

        let settings = RuntimeSettings::load(path, Some(ExecutionProvider::Directml)).unwrap();
        let status = settings.status();
        assert_eq!(status.active_execution_provider, "directml");
        assert_eq!(status.configured_execution_provider, "openvino");
        assert!(status.restart_required);
    }
}
