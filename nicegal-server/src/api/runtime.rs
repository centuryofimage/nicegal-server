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
use nicegal_core::embedding::ImageEmbeddingModel;
use nicegal_core::runtime::{ExecutionProvider, RuntimeOptions};
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::extract::ApiJson;
use super::ocr_models;
use super::{AppState, RESTART_EXIT_CODE};

/// The runtime selection read at server startup and the persisted selection for its next launch.
/// ONNX Runtime binds to its loaded DLL process-wide, so changing this setting deliberately does
/// not try to replace sessions in a live process.
pub(crate) struct RuntimeSettings {
    path: PathBuf,
    active_execution_provider: ExecutionProvider,
    onnx_runtime_build_info: Mutex<String>,
    configured: Mutex<ConfiguredSettings>,
    loaded_execution_provider: Mutex<Option<ExecutionProvider>>,
}

impl RuntimeSettings {
    pub(crate) fn load(
        path: PathBuf,
        command_line_provider: Option<ExecutionProvider>,
    ) -> Result<Self> {
        if let Some(provider) = command_line_provider {
            anyhow::ensure!(
                provider_available(provider),
                "the {provider} execution provider is not compiled for this platform"
            );
        }
        let configured = read_settings(&path)?;
        Ok(Self {
            path,
            active_execution_provider: command_line_provider
                .unwrap_or(configured.execution_provider),
            onnx_runtime_build_info: Mutex::new(String::new()),
            configured: Mutex::new(configured),
            loaded_execution_provider: Mutex::new(None),
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
        let loaded_execution_provider = self
            .loaded_execution_provider
            .lock()
            .expect("loaded provider mutex poisoned")
            .map(|provider| provider.to_string());
        let configured = self
            .configured
            .lock()
            .expect("runtime settings mutex poisoned");
        RuntimeStatusResponse {
            active_execution_provider: self.active_execution_provider.to_string(),
            loaded_execution_provider,
            active_runtime_distribution: runtime_distribution(self.active_execution_provider)
                .to_owned(),
            onnx_runtime_build_info: self
                .onnx_runtime_build_info
                .lock()
                .expect("runtime settings mutex poisoned")
                .clone(),
            configured_execution_provider: configured.execution_provider.to_string(),
            index_videos: configured.index_videos,
            ocr_models: configured.ocr_models.clone(),
            restart_required: self.active_execution_provider != configured.execution_provider,
            image_model: None,
            available_execution_providers: available_execution_providers()
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
    }

    pub(super) fn set(
        &self,
        execution_provider: ExecutionProvider,
    ) -> Result<RuntimeStatusResponse> {
        self.update(Some(execution_provider), None)?;
        Ok(self.status())
    }

    /// A provider that has compiled an indexing model successfully must not be treated as unsupported
    /// after a later process crash or model reload failure.
    pub(super) fn provider_has_worked(&self, provider: ExecutionProvider) -> bool {
        self.configured
            .lock()
            .expect("runtime settings mutex poisoned")
            .working_execution_providers
            .contains(&provider.to_string())
    }

    pub(super) fn record_working_provider(&self, provider: ExecutionProvider) -> Result<()> {
        let mut configured = self
            .configured
            .lock()
            .expect("runtime settings mutex poisoned");
        let name = provider.to_string();
        if configured.working_execution_providers.contains(&name) {
            return Ok(());
        }
        let mut next = configured.clone();
        next.working_execution_providers.push(name);
        write_settings(&self.path, &RuntimeSettingsFile::from(&next))?;
        *configured = next;
        Ok(())
    }

    /// All indexing models share the provider chosen by the first successful model load.
    pub(super) fn model_runtime_options(&self, mut options: RuntimeOptions) -> RuntimeOptions {
        let loaded = *self
            .loaded_execution_provider
            .lock()
            .expect("loaded provider mutex poisoned");
        options.execution_provider = loaded.unwrap_or(self.active_execution_provider);
        options.allow_cpu_fallback =
            loaded.is_none() && !self.provider_has_worked(self.active_execution_provider);
        options
    }

    pub(super) fn accept_loaded_provider(&self, provider: ExecutionProvider) -> Result<()> {
        let mut loaded = self
            .loaded_execution_provider
            .lock()
            .expect("loaded provider mutex poisoned");
        if let Some(previous) = *loaded {
            anyhow::ensure!(
                previous == provider,
                "models in this server process cannot mix {previous} and {provider} execution providers"
            );
            return Ok(());
        }
        if provider == self.active_execution_provider {
            if let Err(error) = self.record_working_provider(provider) {
                tracing::warn!(%error, %provider, "failed to save a working execution provider");
            }
        } else if self.active_execution_provider == ExecutionProvider::Directml
            && provider == ExecutionProvider::Cpu
        {
            // The loaded DirectML distribution cannot run OpenVINO sessions. A new process can.
            self.set(ExecutionProvider::OpenVino)?;
            tracing::warn!("DirectML failed its first model load; restarting with OpenVINO");
            std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                std::process::exit(RESTART_EXIT_CODE);
            });
        }
        *loaded = Some(provider);
        Ok(())
    }

    pub(super) fn image_model(&self) -> ImageEmbeddingModel {
        self.configured
            .lock()
            .expect("runtime settings mutex poisoned")
            .image_model
    }

    pub(super) fn index_videos(&self) -> bool {
        self.configured
            .lock()
            .expect("runtime settings mutex poisoned")
            .index_videos
    }

    pub(super) fn ocr_models(&self) -> Result<ocr_models::job::Spec> {
        let request = self
            .configured
            .lock()
            .expect("runtime settings mutex poisoned")
            .ocr_models
            .clone();
        ocr_models::job::prepare(request)
            .map_err(|error| anyhow!("invalid saved OCR models: {error:?}"))
    }

    pub(super) fn save_scan_options(
        &self,
        videos: Option<bool>,
        ocr: Option<&ocr_models::job::Spec>,
    ) -> Result<()> {
        self.update_all(None, None, videos, ocr)
    }

    /// Publish both selections only after their shared record has been atomically replaced.
    pub(super) fn update(
        &self,
        provider: Option<ExecutionProvider>,
        model: Option<ImageEmbeddingModel>,
    ) -> Result<()> {
        self.update_all(provider, model, None, None)
    }

    fn update_all(
        &self,
        provider: Option<ExecutionProvider>,
        model: Option<ImageEmbeddingModel>,
        videos: Option<bool>,
        ocr: Option<&ocr_models::job::Spec>,
    ) -> Result<()> {
        if let Some(provider) = provider {
            anyhow::ensure!(
                provider_available(provider),
                "the {provider} execution provider is not compiled for this platform"
            );
        }
        if let Some(model) = model {
            anyhow::ensure!(model.available(), "image model is unavailable");
        }
        let mut configured = self
            .configured
            .lock()
            .expect("runtime settings mutex poisoned");
        let mut next = configured.clone();
        if let Some(provider) = provider {
            next.execution_provider = provider;
        }
        if let Some(model) = model {
            next.image_model = model;
        }
        if let Some(videos) = videos {
            next.index_videos = videos;
        }
        if let Some(ocr) = ocr {
            next.ocr_models = ocr.request();
        }
        write_settings(&self.path, &RuntimeSettingsFile::from(&next))?;
        *configured = next;
        Ok(())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RuntimeStatusResponse {
    active_execution_provider: String,
    loaded_execution_provider: Option<String>,
    active_runtime_distribution: String,
    onnx_runtime_build_info: String,
    configured_execution_provider: String,
    index_videos: bool,
    ocr_models: ocr_models::job::Request,
    restart_required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_model: Option<super::image_model::ModelStatus>,
    available_execution_providers: Vec<String>,
}

/// CPU models can execute against either accelerated distribution. The Windows server uses the
/// DirectML distribution for CPU selection because it is the default general-purpose bundle.
fn runtime_distribution(execution_provider: ExecutionProvider) -> &'static str {
    match execution_provider {
        ExecutionProvider::OpenVino => "openvino",
        ExecutionProvider::Webgpu => "webgpu",
        ExecutionProvider::CoreML => "coreml",
        ExecutionProvider::Cpu | ExecutionProvider::Directml => {
            if cfg!(target_os = "linux") {
                if cfg!(feature = "ort-webgpu") {
                    "webgpu"
                } else {
                    "openvino"
                }
            } else if cfg!(windows) {
                "directml"
            } else if cfg!(target_os = "macos") {
                "coreml"
            } else {
                "cpu"
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn available_execution_providers() -> &'static [ExecutionProvider] {
    &[
        #[cfg(feature = "ort-webgpu")]
        ExecutionProvider::Webgpu,
        #[cfg(feature = "ort-openvino")]
        ExecutionProvider::OpenVino,
        ExecutionProvider::Cpu,
    ]
}

#[cfg(windows)]
fn available_execution_providers() -> &'static [ExecutionProvider] {
    &[
        #[cfg(feature = "ort-directml")]
        ExecutionProvider::Directml,
        #[cfg(feature = "ort-openvino")]
        ExecutionProvider::OpenVino,
        ExecutionProvider::Cpu,
    ]
}

#[cfg(not(any(target_os = "linux", windows)))]
fn available_execution_providers() -> &'static [ExecutionProvider] {
    &[
        #[cfg(all(target_os = "macos", feature = "ort-coreml"))]
        ExecutionProvider::CoreML,
        ExecutionProvider::Cpu,
    ]
}

fn provider_available(provider: ExecutionProvider) -> bool {
    available_execution_providers().contains(&provider)
}

fn default_provider() -> ExecutionProvider {
    available_execution_providers()[0]
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeUpdateRequest {
    execution_provider: Option<String>,
    image_model: Option<String>,
    index_videos: Option<bool>,
    ocr_models: Option<ocr_models::job::Request>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeSettingsFile {
    execution_provider: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    working_execution_providers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    image_model: Option<String>,
    #[serde(default = "default_true")]
    index_videos: bool,
    #[serde(default = "default_ocr_models")]
    ocr_models: ocr_models::job::Request,
}

fn default_true() -> bool {
    true
}

fn default_ocr_models() -> ocr_models::job::Request {
    serde_json::from_value(serde_json::json!({
        "detection": { "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx", "revision": "main" },
        "recognition": { "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx", "revision": "main" }
    }))
    .expect("built-in OCR model pair is valid")
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
    if request.execution_provider.is_none()
        && request.image_model.is_none()
        && request.index_videos.is_none()
        && request.ocr_models.is_none()
    {
        return Err(ApiError::bad_request("Provide a runtime setting"));
    }
    let ocr_models = request
        .ocr_models
        .map(ocr_models::job::prepare)
        .transpose()?;
    let execution_provider = request
        .execution_provider
        .as_deref()
        .map(ExecutionProvider::from_str)
        .transpose()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    if let Some(provider) = execution_provider
        && !provider_available(provider)
    {
        return Err(ApiError::bad_request(format!(
            "The {provider} execution provider is not available on this platform"
        )));
    }
    let model = request
        .image_model
        .as_deref()
        .map(str::parse::<nicegal_core::embedding::ImageEmbeddingModel>)
        .transpose()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    if state.jobs.has_active_job()
        && (request.execution_provider.is_some() || request.image_model.is_some())
    {
        return Err(ApiError::job_busy());
    }
    if let Some(model) = model
        && !model.available()
    {
        return Err(ApiError::bad_request("Image model is unavailable"));
    }
    let runtime = Arc::clone(&state.runtime);
    tokio::task::spawn_blocking(move || -> Result<()> {
        runtime.update_all(
            execution_provider,
            model,
            request.index_videos,
            ocr_models.as_ref(),
        )
    })
    .await
    .map_err(|error| ApiError::internal(anyhow!("runtime settings task failed: {error}")))?
    .map_err(ApiError::internal)?;
    Ok((StatusCode::OK, Json(status_response(&state))))
}

#[derive(Clone)]
struct ConfiguredSettings {
    execution_provider: ExecutionProvider,
    working_execution_providers: Vec<String>,
    image_model: ImageEmbeddingModel,
    index_videos: bool,
    ocr_models: ocr_models::job::Request,
}

impl From<&ConfiguredSettings> for RuntimeSettingsFile {
    fn from(settings: &ConfiguredSettings) -> Self {
        Self {
            execution_provider: settings.execution_provider.to_string(),
            working_execution_providers: settings.working_execution_providers.clone(),
            image_model: Some(settings.image_model.id().to_owned()),
            index_videos: settings.index_videos,
            ocr_models: settings.ocr_models.clone(),
        }
    }
}

fn read_settings(path: &PathBuf) -> Result<ConfiguredSettings> {
    let file: Option<RuntimeSettingsFile> = match fs::read(path) {
        Ok(contents) => Some(
            serde_json::from_slice(&contents)
                .with_context(|| format!("parsing runtime settings: {path}"))?,
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| format!("reading runtime settings: {path}"));
        }
    };
    let saved = file.as_ref().map(|file| file.execution_provider.as_str());
    let provider =
        match saved {
            None | Some("cuda" | "migraphx") => None,
            Some(value) => Some(value.parse::<ExecutionProvider>().with_context(|| {
                format!("invalid execution provider in runtime settings {path}")
            })?),
        }
        .filter(|provider| provider_available(*provider));
    let settings = ConfiguredSettings {
        execution_provider: provider.unwrap_or_else(default_provider),
        working_execution_providers: file
            .as_ref()
            .map_or_else(Vec::new, |file| file.working_execution_providers.clone()),
        image_model: match file.as_ref().and_then(|file| file.image_model.as_deref()) {
            Some(model) => model.parse()?,
            None => read_legacy_image_model(path)?,
        },
        index_videos: file.as_ref().is_none_or(|file| file.index_videos),
        ocr_models: file
            .as_ref()
            .map_or_else(default_ocr_models, |file| file.ocr_models.clone()),
    };
    if saved.is_some() && provider.is_none() {
        write_settings(path, &RuntimeSettingsFile::from(&settings))?;
        tracing::warn!(previous = saved, provider = %settings.execution_provider, "saved execution provider is absent from this build; selected bundled default");
    }
    Ok(settings)
}

fn read_legacy_image_model(path: &PathBuf) -> Result<ImageEmbeddingModel> {
    // Read legacy selection until the first successful write migrates it into runtime.json.
    #[derive(Deserialize)]
    struct LegacySelection {
        model: String,
    }
    let legacy_path = path.with_file_name("image-model.json");
    match fs::read(&legacy_path) {
        Ok(contents) => serde_json::from_slice::<LegacySelection>(&contents)?
            .model
            .parse(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(ImageEmbeddingModel::default())
        }
        Err(error) => {
            Err(error).with_context(|| format!("reading image model settings: {legacy_path}"))
        }
    }
}

fn write_settings(path: &PathBuf, settings: &RuntimeSettingsFile) -> Result<()> {
    let parent = path
        .parent()
        .context("runtime settings path has no parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating runtime settings directory: {parent}"))?;

    let contents = serde_json::to_vec_pretty(settings).context("serializing runtime settings")?;
    let temporary = path.with_extension("json.tmp");
    let mut file = fs::File::create(&temporary)
        .with_context(|| format!("creating temporary runtime settings: {temporary}"))?;
    file.write_all(&contents)
        .with_context(|| format!("writing temporary runtime settings: {temporary}"))?;
    file.sync_all()
        .with_context(|| format!("syncing temporary runtime settings: {temporary}"))?;
    drop(file);
    fs::rename(&temporary, path)
        .with_context(|| format!("replacing runtime settings {path} with {temporary}"))
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn working_provider_survives_restart_and_provider_change() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        let provider = default_provider();
        let settings = RuntimeSettings::load(path.clone(), None)?;
        assert!(!settings.provider_has_worked(provider));
        settings.record_working_provider(provider)?;
        settings.record_working_provider(provider)?;
        settings.set(ExecutionProvider::Cpu)?;

        let restarted = RuntimeSettings::load(path.clone(), None)?;
        assert!(restarted.provider_has_worked(provider));
        let saved: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        assert_eq!(
            saved["workingExecutionProviders"],
            serde_json::json!([provider.to_string()])
        );
        Ok(())
    }

    #[test]
    fn first_loaded_model_sets_the_provider_for_later_models() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        let settings = RuntimeSettings::load(path, Some(ExecutionProvider::Cpu))?;
        assert!(
            settings
                .model_runtime_options(RuntimeOptions::default())
                .allow_cpu_fallback
        );
        settings.accept_loaded_provider(ExecutionProvider::Cpu)?;
        let later = settings.model_runtime_options(RuntimeOptions {
            execution_provider: ExecutionProvider::Directml,
            ..RuntimeOptions::default()
        });
        assert_eq!(later.execution_provider, ExecutionProvider::Cpu);
        assert!(!later.allow_cpu_fallback);
        assert!(
            settings
                .accept_loaded_provider(ExecutionProvider::Directml)
                .is_err()
        );
        assert_eq!(
            settings.status().loaded_execution_provider.as_deref(),
            Some("cpu")
        );
        Ok(())
    }

    #[test]
    fn scan_options_survive_restart() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        let settings = RuntimeSettings::load(path.clone(), None)?;
        let request: ocr_models::job::Request = serde_json::from_value(serde_json::json!({
            "detection": {"modelId": "example/detector"},
            "recognition": {"modelId": "example/recognizer"}
        }))?;
        let models = ocr_models::job::prepare(request).map_err(|error| anyhow!("{error:?}"))?;
        settings.save_scan_options(Some(false), Some(&models))?;
        let restarted = RuntimeSettings::load(path, None)?;
        assert!(!restarted.index_videos());
        let saved = serde_json::to_value(restarted.ocr_models()?.request())?;
        assert_eq!(saved["detection"]["modelId"], "example/detector");
        Ok(())
    }

    #[test]
    fn legacy_image_selection_migrates_and_combined_updates_survive_restart() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        fs::write(
            path.with_file_name("image-model.json"),
            serde_json::to_vec(
                &serde_json::json!({"model": ImageEmbeddingModel::SigLip2Base256.id()}),
            )?,
        )?;
        let settings = RuntimeSettings::load(path.clone(), None)?;
        assert_eq!(settings.image_model(), ImageEmbeddingModel::SigLip2Base256);
        settings.update(
            Some(ExecutionProvider::Cpu),
            Some(ImageEmbeddingModel::MetaClip2B16),
        )?;
        let restarted = RuntimeSettings::load(path, None)?;
        assert_eq!(
            restarted.active_execution_provider(),
            ExecutionProvider::Cpu
        );
        assert_eq!(restarted.image_model(), ImageEmbeddingModel::MetaClip2B16);
        Ok(())
    }

    #[test]
    fn provider_migration_preserves_model_and_rejects_invalid_models_before_writing() -> Result<()>
    {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        let selected = ImageEmbeddingModel::SigLip2Base256;
        fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "executionProvider": "cuda", "imageModel": selected.id()
            }))?,
        )?;
        let settings = RuntimeSettings::load(path.clone(), None)?;
        assert_eq!(settings.image_model(), selected);
        assert_eq!(
            RuntimeSettings::load(path.clone(), None)?.image_model(),
            selected
        );

        let invalid = br#"{"executionProvider":"cuda","imageModel":"invalid-model"}"#;
        fs::write(&path, invalid)?;
        assert!(RuntimeSettings::load(path.clone(), None).is_err());
        assert_eq!(fs::read(path)?, invalid);
        Ok(())
    }

    #[test]
    fn failed_combined_write_preserves_both_selections() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        let settings = RuntimeSettings::load(path.clone(), None)?;
        settings.set(default_provider())?;
        let before = fs::read(&path)?;
        let selected = settings.image_model();
        fs::create_dir(path.with_extension("json.tmp"))?;
        assert!(
            settings
                .update(
                    Some(ExecutionProvider::Cpu),
                    Some(ImageEmbeddingModel::SigLip2Base256)
                )
                .is_err()
        );
        assert_eq!(fs::read(&path)?, before);
        assert_eq!(settings.image_model(), selected);
        assert_eq!(
            settings.status().configured_execution_provider,
            default_provider().to_string()
        );
        Ok(())
    }

    #[test]
    fn concurrent_updates_preserve_independent_selections() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        let settings = Arc::new(RuntimeSettings::load(path.clone(), None)?);
        std::thread::scope(|scope| {
            for index in 0..16 {
                let settings = Arc::clone(&settings);
                scope.spawn(move || {
                    if index % 2 == 0 {
                        settings.update(Some(ExecutionProvider::Cpu), None).unwrap();
                    } else {
                        settings
                            .update(None, Some(ImageEmbeddingModel::SigLip2Base256))
                            .unwrap();
                    }
                });
            }
        });
        let restarted = RuntimeSettings::load(path, None)?;
        assert_eq!(
            restarted.active_execution_provider(),
            ExecutionProvider::Cpu
        );
        assert_eq!(restarted.image_model(), ImageEmbeddingModel::SigLip2Base256);
        Ok(())
    }

    #[test]
    fn missing_settings_use_platform_default_and_persist_updates() {
        let temp = TempDir::new().unwrap();
        let path = PathBuf::try_from(temp.path().join("runtime.json")).unwrap();
        let settings = RuntimeSettings::load(path.clone(), None).unwrap();
        settings.set_onnx_runtime_build_info("test build".to_owned());
        let expected = default_provider().to_string();
        assert_eq!(settings.status().active_execution_provider, expected);
        assert_eq!(settings.status().configured_execution_provider, expected);
        if cfg!(target_os = "linux") {
            assert_eq!(
                settings.status().active_runtime_distribution,
                runtime_distribution(default_provider())
            );
        }
        assert!(!settings.status().restart_required);

        let status = settings.set(ExecutionProvider::Cpu).unwrap();
        assert_eq!(status.active_execution_provider, expected);
        assert_eq!(status.configured_execution_provider, "cpu");
        assert_eq!(status.restart_required, expected != "cpu");

        let status = settings.set(default_provider()).unwrap();
        assert_eq!(status.configured_execution_provider, expected);

        let restarted = RuntimeSettings::load(path, None).unwrap();
        assert_eq!(restarted.status().active_execution_provider, expected);
        assert!(!restarted.status().restart_required);
    }

    #[test]
    fn command_line_provider_overrides_saved_setting_for_this_launch() {
        let temp = TempDir::new().unwrap();
        let path = PathBuf::try_from(temp.path().join("runtime.json")).unwrap();
        RuntimeSettings::load(path.clone(), None)
            .unwrap()
            .set(default_provider())
            .unwrap();

        let settings = RuntimeSettings::load(path, Some(ExecutionProvider::Cpu)).unwrap();
        let status = settings.status();
        assert_eq!(status.active_execution_provider, "cpu");
        assert_eq!(
            status.configured_execution_provider,
            default_provider().to_string()
        );
        assert_eq!(
            status.restart_required,
            default_provider() != ExecutionProvider::Cpu
        );
    }

    #[test]
    fn saved_unavailable_providers_migrate_but_explicit_overrides_fail() {
        let temp = TempDir::new().unwrap();
        let path = PathBuf::try_from(temp.path().join("runtime.json")).unwrap();
        for saved in [
            "cuda", "migraphx", "webgpu", "openvino", "directml", "coreml",
        ] {
            if saved.parse().is_ok_and(provider_available) {
                continue;
            }
            fs::write(&path, format!(r#"{{"executionProvider":"{saved}"}}"#)).unwrap();
            let settings = RuntimeSettings::load(path.clone(), None).unwrap();
            assert_eq!(settings.active_execution_provider(), default_provider());
            let persisted: RuntimeSettingsFile =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            assert_eq!(persisted.execution_provider, default_provider().to_string());
            if let Ok(provider) = saved.parse() {
                assert!(RuntimeSettings::load(path.clone(), Some(provider)).is_err());
                assert!(settings.set(provider).is_err());
            }
        }
    }
}
