//! Every model the server loads, and when each one is built.
//!
//! - Query encoders are [`Residency::Interactive`] and run on CPU: a search builds one from
//!   cached files on first use, never downloads, and it stays through a working session. CPU
//!   also keeps searches off the DirectML queue, which runs one session at a time and is held by
//!   indexing.
//! - Indexing models are [`Residency::Batch`]: a job builds one, downloading if needed, only
//!   when it has work for it.
//! - Scans check that query encoders' files are present ([`LazyModel::ensure_available`]);
//!   building one belongs to the first search that needs it.
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use super::model_idle::Residency;
use super::model_slot::{ModelSlot, ModelState, ModelStatus, PrepareObserver};

use anyhow::{Context, Result};
use axum::{
    Json,
    extract::State,
    routing::{MethodRouter, get, post},
};
use nicegal_core::embedding::{
    ImageEmbedder, ImageEmbedderOptions, ImageEmbeddingModel, ImageQueryEmbedder,
    ImageQueryEmbedderOptions, TextEmbedder, TextEmbedderOptions,
};
use nicegal_core::hub::DownloadObserver;
use nicegal_core::ocr::PaddleOcrPool;
use nicegal_core::runtime::{ExecutionProvider, RuntimeOptions};
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};

use super::{AppState, RuntimeSettings, error::ApiError};

/// A model session type and how to build it.
pub(crate) trait Loader: Sized + Send + Sync + 'static {
    type Options: Clone + Send + Sync;
    /// Build a session, downloading missing files.
    fn load(options: &Self::Options, progress: &dyn DownloadObserver) -> Result<Self>;
    /// Build a session from cached files only; `None` means the files are missing.
    fn load_cached(options: &Self::Options) -> Result<Option<Self>>;
    /// Download the files a session needs without building one, or with `cached_only` report
    /// whether they are all present.
    fn fetch(
        options: &Self::Options,
        cached_only: bool,
        progress: &dyn DownloadObserver,
    ) -> Result<bool>;
    fn execution_provider(&self) -> ExecutionProvider;
    fn runtime(options: &mut Self::Options) -> &mut RuntimeOptions;
    /// What `options` loads, for progress reports.
    fn model_id(options: &Self::Options) -> &'static str;
}

macro_rules! loader {
    ($session:ty, $options:ty, $model_id:expr) => {
        impl Loader for $session {
            type Options = $options;
            fn model_id(options: &$options) -> &'static str {
                let model_id: fn(&$options) -> &'static str = $model_id;
                model_id(options)
            }
            fn load(options: &$options, progress: &dyn DownloadObserver) -> Result<Self> {
                <$session>::load_with_progress(options, progress)
            }
            fn load_cached(options: &$options) -> Result<Option<Self>> {
                <$session>::load_cached(options)
            }
            fn fetch(
                options: &$options,
                cached_only: bool,
                progress: &dyn DownloadObserver,
            ) -> Result<bool> {
                <$session>::fetch(options, cached_only, progress)
            }
            fn execution_provider(&self) -> ExecutionProvider {
                <$session>::execution_provider(self)
            }
            fn runtime(options: &mut $options) -> &mut RuntimeOptions {
                &mut options.runtime
            }
        }
    };
}
loader!(TextEmbedder, TextEmbedderOptions, |_| {
    TextEmbedder::MODEL_ID
});
loader!(ImageEmbedder, ImageEmbedderOptions, |options| options
    .model
    .id());
loader!(ImageQueryEmbedder, ImageQueryEmbedderOptions, |options| {
    options.model.id()
});

/// The PaddleOCR pair. Recognition needs exclusive access to the sessions; jobs run one at a
/// time, so the lock is never contended.
pub(crate) struct OcrSessions {
    /// Kept outside the lock, so reading it never waits for recognition.
    execution_provider: ExecutionProvider,
    pool: Mutex<PaddleOcrPool>,
}

impl OcrSessions {
    fn new(pool: PaddleOcrPool) -> Self {
        Self {
            execution_provider: pool.execution_provider(),
            pool: Mutex::new(pool),
        }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, PaddleOcrPool> {
        self.pool.lock()
    }
}

impl Loader for OcrSessions {
    type Options = RuntimeOptions;
    fn load(options: &RuntimeOptions, progress: &dyn DownloadObserver) -> Result<Self> {
        PaddleOcrPool::load(*options, progress).map(Self::new)
    }
    fn load_cached(options: &RuntimeOptions) -> Result<Option<Self>> {
        Ok(PaddleOcrPool::load_cached(*options)?.map(Self::new))
    }
    fn fetch(
        _options: &RuntimeOptions,
        cached_only: bool,
        progress: &dyn DownloadObserver,
    ) -> Result<bool> {
        PaddleOcrPool::fetch(cached_only, progress)
    }
    fn execution_provider(&self) -> ExecutionProvider {
        self.execution_provider
    }
    fn runtime(options: &mut RuntimeOptions) -> &mut RuntimeOptions {
        options
    }
    fn model_id(_options: &RuntimeOptions) -> &'static str {
        PaddleOcrPool::MODEL_ID
    }
}

pub(crate) struct LazyModel<T: Loader> {
    options: T::Options,
    slot: ModelSlot<T>,
    /// Supplies the persisted provider choice and records the one a load actually used. Only
    /// models on the process's selected provider take part; CPU-pinned ones do not.
    runtime: Option<Arc<RuntimeSettings>>,
    #[cfg(test)]
    cached_loading: bool,
}

pub(crate) type TextModel = LazyModel<TextEmbedder>;
pub(crate) type ImageModel = LazyModel<ImageEmbedder>;
pub(crate) type ImageQueryModel = LazyModel<ImageQueryEmbedder>;
pub(crate) type OcrModel = LazyModel<OcrSessions>;

impl<T: Loader> LazyModel<T> {
    pub(super) fn new(options: T::Options, name: &'static str, residency: Residency) -> Self {
        Self {
            slot: ModelSlot::new(name, residency).identified_by(T::model_id(&options)),
            options,
            runtime: None,
            #[cfg(test)]
            cached_loading: true,
        }
    }

    fn with_provider_tracking(mut self, runtime: Arc<RuntimeSettings>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub(super) fn status(&self) -> ModelStatus {
        self.slot.status()
    }

    pub(super) fn unload_idle(&self, now: Instant) {
        self.slot.unload_idle(now);
    }

    fn configured_options(&self) -> Cow<'_, T::Options> {
        match &self.runtime {
            Some(runtime) => {
                let mut options = self.options.clone();
                let configured = T::runtime(&mut options);
                *configured = runtime.model_runtime_options(*configured);
                Cow::Owned(options)
            }
            None => Cow::Borrowed(&self.options),
        }
    }

    fn build(
        &self,
        observer: &dyn PrepareObserver,
        load: impl FnOnce(&T::Options) -> Result<Option<T>>,
    ) -> Result<Option<Arc<T>>> {
        self.slot.prepare(observer, || {
            let loaded = load(&self.configured_options())?;
            if let (Some(model), Some(runtime)) = (&loaded, &self.runtime) {
                runtime.accept_loaded_provider(model.execution_provider())?;
            }
            Ok(loaded)
        })
    }

    /// For indexing: the session, built and downloaded as needed. Call from a blocking worker.
    pub(super) fn prepare(&self, observer: &dyn PrepareObserver) -> Result<Arc<T>> {
        self.build(observer, |options| T::load(options, observer).map(Some))?
            .context("download-capable loader returned no model")
    }

    /// For searches: the session, built from cached files if it is not in memory. Never
    /// downloads; missing files are a `models_not_ready` error. Call from a blocking worker.
    pub(super) fn session(&self) -> Result<Arc<T>, ApiError> {
        self.cached()?.ok_or_else(|| self.not_ready())
    }

    /// The session if it is in memory or its files are cached.
    fn cached(&self) -> Result<Option<Arc<T>>, ApiError> {
        if let Some(session) = self.slot.acquire() {
            return Ok(Some(session));
        }
        #[cfg(test)]
        if !self.cached_loading {
            return Ok(None);
        }
        self.build(&(), T::load_cached)
            .map_err(|error| ApiError::models_not_ready(format!("{error:#}")))
    }

    /// For scans: make sure a search will find this model's files, downloading missing ones,
    /// without building a session. A failed model is built instead, which is how rescanning
    /// retries it.
    pub(super) fn ensure_available(&self, observer: &dyn PrepareObserver) -> Result<()> {
        let status = self.status();
        if status.resident {
            return Ok(());
        }
        if status.state == ModelState::Failed {
            self.prepare(observer)?;
        } else if T::fetch(&self.options, false, observer)? {
            self.slot.mark_available();
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn ready(&self) -> Result<Arc<T>, ApiError> {
        self.slot.acquire().ok_or_else(|| self.not_ready())
    }

    fn not_ready(&self) -> ApiError {
        ApiError::models_not_ready(format!(
            "{} is not ready. Search preparation runs automatically for enabled libraries; reopen or rescan the library to retry.",
            self.slot.name()
        ))
    }

    #[cfg(test)]
    pub(super) fn without_cached_loading(mut self) -> Self {
        self.cached_loading = false;
        self
    }
}

macro_rules! model_kind {
    ($session:ty, $kind:ty) => {
        impl LazyModel<$session> {
            pub(crate) fn model(&self) -> $kind {
                self.options.model
            }
            pub(crate) fn dimensions(&self) -> usize {
                self.model().dimensions()
            }
        }
    };
}
model_kind!(ImageEmbedder, ImageEmbeddingModel);
model_kind!(ImageQueryEmbedder, ImageEmbeddingModel);

/// Every model session the server keeps, shared by searches and jobs.
pub(crate) struct Models {
    /// BGE for OCR-text queries.
    pub(crate) text_query: TextModel,
    /// BGE for OCR-text indexing, on the selected provider.
    pub(crate) text_index: TextModel,
    pub(crate) image: ImageModel,
    /// The image model's paired text encoder, for picture queries.
    pub(crate) image_query: ImageQueryModel,
    pub(crate) ocr: OcrModel,
}

impl Models {
    /// `image` describes the image model; every provider is chosen here.
    pub(crate) fn new(image: ImageEmbedderOptions, runtime: &Arc<RuntimeSettings>) -> Self {
        let provider = runtime.active_execution_provider();
        let cpu = RuntimeOptions {
            execution_provider: ExecutionProvider::Cpu,
            ..RuntimeOptions::default()
        };
        let selected = RuntimeOptions {
            execution_provider: provider,
            ..RuntimeOptions::default()
        };
        Self {
            text_query: TextModel::new(
                TextEmbedderOptions {
                    runtime: cpu,
                    ..TextEmbedderOptions::default()
                },
                "Text search model",
                Residency::Interactive,
            ),
            text_index: TextModel::new(
                TextEmbedderOptions {
                    runtime: selected,
                    ..TextEmbedderOptions::default()
                },
                "Text indexing model",
                Residency::Batch,
            )
            .with_provider_tracking(Arc::clone(runtime)),
            image_query: ImageQueryModel::new(
                ImageQueryEmbedderOptions {
                    model: image.model,
                    runtime: cpu,
                    ..ImageQueryEmbedderOptions::default()
                },
                "Image description model",
                Residency::Interactive,
            ),
            image: ImageModel::new(
                ImageEmbedderOptions {
                    runtime: selected,
                    ..image
                },
                "Image model",
                Residency::Batch,
            )
            .with_provider_tracking(Arc::clone(runtime)),
            ocr: OcrModel::new(selected, "OCR models", Residency::Batch)
                .with_provider_tracking(Arc::clone(runtime)),
        }
    }

    pub(super) fn unload_idle(&self, now: Instant) {
        self.text_query.unload_idle(now);
        self.text_index.unload_idle(now);
        self.image.unload_idle(now);
        self.image_query.unload_idle(now);
        self.ocr.unload_idle(now);
    }

    fn supports_image_text(&self) -> bool {
        self.image.model().supports_text_queries()
    }

    /// The paired text encoder's files, when the image model has one.
    pub(super) fn ensure_image_query_available(
        &self,
        observer: &dyn PrepareObserver,
    ) -> Result<()> {
        if self.supports_image_text() {
            self.image_query.ensure_available(observer)?;
        }
        Ok(())
    }

    fn status(&self) -> StatusResponse {
        StatusResponse {
            text: self.text_query.status(),
            text_index: self.text_index.status(),
            clip_image: self.image.status(),
            clip_text: if self.supports_image_text() {
                self.image_query.status()
            } else {
                ModelStatus {
                    state: ModelState::Unsupported,
                    error: None,
                    resident: false,
                }
            },
            ocr: self.ocr.status(),
        }
    }

    /// Build query encoders from cached files ahead of a search.
    fn warm(&self, model: WarmModel) -> Result<(), ApiError> {
        match model {
            WarmModel::Text => self.text_query.cached().map(drop),
            WarmModel::ClipText if self.supports_image_text() => {
                self.image_query.cached().map(drop)
            }
            WarmModel::ClipText => Ok(()),
            WarmModel::ClipImage => self.image.cached().map(drop),
        }
    }
}

#[cfg(test)]
impl Models {
    /// Every model on CPU with no provider tracking. The picture query encoder never loads; the
    /// text one loads from the cache only when `cached_text`.
    pub(crate) fn for_tests(cached_text: bool) -> Self {
        let text =
            |name, residency| TextModel::new(TextEmbedderOptions::default(), name, residency);
        let text_query = text("Text search model", Residency::Interactive);
        Self {
            text_query: if cached_text {
                text_query
            } else {
                text_query.without_cached_loading()
            },
            text_index: text("Text indexing model", Residency::Batch),
            image: ImageModel::new(
                ImageEmbedderOptions::default(),
                "Image model",
                Residency::Batch,
            ),
            image_query: ImageQueryModel::new(
                ImageQueryEmbedderOptions::default(),
                "Image description model",
                Residency::Interactive,
            )
            .without_cached_loading(),
            ocr: OcrModel::new(RuntimeOptions::default(), "OCR models", Residency::Batch),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusResponse {
    text: ModelStatus,
    text_index: ModelStatus,
    clip_image: ModelStatus,
    clip_text: ModelStatus,
    ocr: ModelStatus,
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(status)
}

pub(super) fn warm_route() -> MethodRouter<AppState> {
    post(warm)
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
enum WarmModel {
    Text,
    ClipText,
    ClipImage,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WarmRequest {
    models: Vec<WarmModel>,
}

/// Load the named models from cached files, then report every model's status. Missing files
/// are not an error here: the status says what is not ready.
async fn warm(
    State(state): State<AppState>,
    super::extract::ApiJson(request): super::extract::ApiJson<WarmRequest>,
) -> Result<Json<StatusResponse>, ApiError> {
    let models = Arc::clone(&state.models);
    super::run_blocking(move || {
        for model in request.models {
            if let Err(error) = models.warm(model)
                && error.code.as_str() != "models_not_ready"
            {
                return Err(error);
            }
        }
        Ok(models.status())
    })
    .await
    .map(Json)
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    Json(state.models.status())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tempfile::TempDir;

    type Load = Arc<dyn Fn() -> Result<i32> + Send + Sync>;
    type LoadCached = Arc<dyn Fn() -> Result<Option<i32>> + Send + Sync>;

    /// A fake session whose options carry its loaders.
    #[derive(Clone)]
    pub(crate) struct Fake {
        load: Load,
        cached: LoadCached,
        fetched: Arc<AtomicUsize>,
        runtime: RuntimeOptions,
    }

    impl Fake {
        fn new(load: impl Fn() -> Result<i32> + Send + Sync + 'static) -> Self {
            Self {
                load: Arc::new(load),
                cached: Arc::new(|| Ok(None)),
                fetched: Arc::default(),
                runtime: RuntimeOptions::default(),
            }
        }

        fn cached(
            mut self,
            cached: impl Fn() -> Result<Option<i32>> + Send + Sync + 'static,
        ) -> Self {
            self.cached = Arc::new(cached);
            self
        }
    }

    impl Loader for i32 {
        type Options = Fake;
        fn load(options: &Fake, _: &dyn DownloadObserver) -> Result<Self> {
            (options.load)()
        }
        fn load_cached(options: &Fake) -> Result<Option<Self>> {
            (options.cached)()
        }
        fn fetch(options: &Fake, _: bool, _: &dyn DownloadObserver) -> Result<bool> {
            options.fetched.fetch_add(1, Ordering::SeqCst);
            Ok(true)
        }
        fn execution_provider(&self) -> ExecutionProvider {
            ExecutionProvider::Cpu
        }
        fn runtime(options: &mut Fake) -> &mut RuntimeOptions {
            &mut options.runtime
        }
        fn model_id(_options: &Fake) -> &'static str {
            "test/model"
        }
    }

    fn model(options: Fake) -> LazyModel<i32> {
        LazyModel::new(options, "test model", Residency::Interactive)
    }

    #[test]
    fn idle_eviction_preserves_active_handles_and_reloads_from_cache() {
        let model = model(Fake::new(|| panic!("search must not download")).cached(|| Ok(Some(42))));
        let active = model.session().unwrap();
        let weak = Arc::downgrade(&active);
        let now = Instant::now();
        let timeout = Residency::Interactive.idle_timeout();
        model.unload_idle(now);
        model.unload_idle(now + timeout);
        assert!(model.status().resident);
        drop(active);
        model.unload_idle(now + timeout);
        // Status polling must not reset the countdown.
        assert!(model.status().resident);
        model.unload_idle(now + timeout * 2);
        assert!(!model.status().resident);
        assert_eq!(model.status().state, ModelState::Ready);
        assert!(
            weak.upgrade().is_none(),
            "eviction must destroy the session"
        );
        assert_eq!(*model.session().unwrap(), 42);
        assert!(model.status().resident);
    }

    #[test]
    fn cancellation_during_successful_compilation_retains_the_session() {
        struct Observer(AtomicBool);
        impl DownloadObserver for Observer {
            fn progress(&self, _: &nicegal_core::hub::ModelSource, _: usize, _: usize) {}
            fn download_cancelled(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
        }
        impl PrepareObserver for Observer {}
        let observer = Arc::new(Observer(AtomicBool::new(false)));
        let cancelling = Arc::clone(&observer);
        let model = model(Fake::new(move || {
            cancelling.0.store(true, Ordering::Release);
            Ok(42)
        }));
        let error = model.prepare(observer.as_ref()).unwrap_err();
        assert!(super::super::jobs::is_cancelled(&error));
        assert_eq!(model.status().state, ModelState::Ready);
        assert_eq!(*model.ready().unwrap(), 42);
        observer.0.store(false, Ordering::Release);
        assert_eq!(*model.prepare(observer.as_ref()).unwrap(), 42);
    }

    #[test]
    fn cancelled_preparation_returns_to_not_loaded_and_can_retry() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&attempts);
        let model = model(Fake::new(move || {
            if counted.fetch_add(1, Ordering::SeqCst) > 0 {
                return Ok(42);
            }
            Err(hf_hub::api::sync::ApiError::Cancelled.into())
        }));
        let error = model.prepare(&()).unwrap_err();
        assert!(super::super::jobs::is_cancelled(&error));
        assert_eq!(model.status().state, ModelState::NotLoaded);
        assert!(model.status().error.is_none());
        assert!(model.ready().is_err());
        assert_eq!(*model.prepare(&()).unwrap(), 42);
        assert_eq!(model.status().state, ModelState::Ready);
    }

    #[test]
    fn successful_lazy_load_records_its_provider_across_restarts() -> Result<()> {
        let temp = TempDir::new()?;
        let path = camino::Utf8PathBuf::try_from(temp.path().join("runtime.json"))?;
        let runtime = Arc::new(RuntimeSettings::load(
            path.clone(),
            Some(ExecutionProvider::Cpu),
        )?);
        let model = model(Fake::new(|| Ok(42))).with_provider_tracking(Arc::clone(&runtime));
        assert!(!runtime.provider_has_worked(ExecutionProvider::Cpu));
        model.prepare(&())?;
        assert!(RuntimeSettings::load(path, None)?.provider_has_worked(ExecutionProvider::Cpu));
        Ok(())
    }

    #[test]
    #[ignore = "requires the search models to have been downloaded by explicit setup"]
    fn cached_query_models_reload_and_embed_without_setup() {
        super::super::tests::initialize_test_runtime();
        let text = TextModel::new(
            TextEmbedderOptions::default(),
            "text",
            Residency::Interactive,
        );
        let clip = ImageQueryModel::new(
            ImageQueryEmbedderOptions::default(),
            "clip",
            Residency::Interactive,
        );
        assert_eq!(text.status().state, ModelState::NotLoaded);
        assert_eq!(clip.status().state, ModelState::NotLoaded);
        let text_vector = text
            .session()
            .expect("text model cached by setup")
            .embed_query("a red car")
            .unwrap();
        let clip_vector = clip
            .session()
            .expect("CLIP text model cached by setup")
            .embed_query("a red car")
            .unwrap();
        assert_eq!(text_vector.len(), TextEmbedder::DIMENSIONS);
        assert_eq!(clip_vector.len(), clip.dimensions());
        assert!(
            text_vector
                .iter()
                .chain(&clip_vector)
                .all(|value| value.is_finite())
        );
        assert_eq!(text.status().state, ModelState::Ready);
        assert_eq!(clip.status().state, ModelState::Ready);
    }

    #[test]
    fn deferred_models_do_not_load_until_prepared_and_failures_can_retry() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&attempts);
        let model = model(Fake::new(move || {
            if counted.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("simulated download failure");
            }
            Ok(42)
        }));
        assert_eq!(model.status().state, ModelState::NotLoaded);
        assert_eq!(model.ready().unwrap_err().code.as_str(), "models_not_ready");
        assert_eq!(attempts.load(Ordering::SeqCst), 0);
        assert!(model.prepare(&()).is_err());
        assert_eq!(model.status().state, ModelState::Failed);
        assert!(
            model
                .status()
                .error
                .unwrap()
                .contains("simulated download failure")
        );
        let session = model.prepare(&()).unwrap();
        assert_eq!(*session, 42);
        assert_eq!(model.status().state, ModelState::Ready);
        assert!(model.status().error.is_none());
        assert!(Arc::ptr_eq(&session, &model.prepare(&()).unwrap()));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn preparation_is_observable_without_waiting_for_the_loader() {
        let entered = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let barriers = (Arc::clone(&entered), Arc::clone(&release));
        let model = Arc::new(model(Fake::new(move || {
            barriers.0.wait();
            barriers.1.wait();
            Ok(42)
        })));
        let worker_model = Arc::clone(&model);
        let worker = std::thread::spawn(move || worker_model.prepare(&()).unwrap());
        entered.wait();
        assert_eq!(model.status().state, ModelState::Preparing);
        assert!(model.ready().is_err());
        release.wait();
        assert_eq!(*worker.join().unwrap(), 42);
        assert_eq!(model.status().state, ModelState::Ready);
    }

    #[test]
    fn a_loader_panic_is_a_visible_retryable_failure() {
        let attempts = AtomicUsize::new(0);
        let attempts = Arc::new(attempts);
        let counted = Arc::clone(&attempts);
        let model = model(Fake::new(move || {
            if counted.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("simulated loader panic");
            }
            Ok(42)
        }));
        assert!(model.prepare(&()).is_err());
        assert_eq!(model.status().state, ModelState::Failed);
        assert!(
            model
                .status()
                .error
                .unwrap()
                .contains("simulated loader panic")
        );
        assert!(model.ready().is_err());
        assert_eq!(*model.prepare(&()).unwrap(), 42);
        assert_eq!(model.status().state, ModelState::Ready);
    }

    #[test]
    fn searches_use_only_the_cached_loader_and_reuse_its_session() {
        let downloads = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&downloads);
        let missing = model(Fake::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        }));
        assert_eq!(
            missing.session().unwrap_err().code.as_str(),
            "models_not_ready"
        );
        assert_eq!(missing.status().state, ModelState::NotLoaded);
        assert_eq!(downloads.load(Ordering::SeqCst), 0);

        let cached = model(Fake::new(|| panic!("must not download")).cached(|| Ok(Some(42))));
        let session = cached.session().unwrap();
        assert_eq!(*session, 42);
        assert_eq!(cached.status().state, ModelState::Ready);
        assert!(Arc::ptr_eq(&session, &cached.session().unwrap()));
    }

    #[test]
    fn a_corrupt_cached_model_does_not_trigger_a_download() {
        let model = model(
            Fake::new(|| panic!("must not download"))
                .cached(|| anyhow::bail!("invalid cached ONNX model")),
        );
        assert!(model.session().is_err());
        assert_eq!(model.status().state, ModelState::Failed);
        assert!(
            model
                .status()
                .error
                .unwrap()
                .contains("invalid cached ONNX model")
        );
    }

    #[test]
    fn scans_check_files_without_building_and_rebuild_only_a_failed_model() {
        let builds = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&builds);
        let options = Fake::new(move || {
            if counted.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("bad graph");
            }
            Ok(42)
        });
        let fetched = Arc::clone(&options.fetched);
        let model = model(options);
        model.ensure_available(&()).unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        assert_eq!(fetched.load(Ordering::SeqCst), 1);
        let status = model.status();
        assert_eq!(status.state, ModelState::Ready);
        assert!(!status.resident);

        assert!(model.prepare(&()).is_err());
        assert_eq!(model.status().state, ModelState::Failed);
        model.ensure_available(&()).unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert!(model.status().resident);
        // A model in memory needs no file check.
        model.ensure_available(&()).unwrap();
        assert_eq!(fetched.load(Ordering::SeqCst), 1);
    }
}
