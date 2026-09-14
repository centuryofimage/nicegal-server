//! Opening a library never loads sessions. Indexing may download; search loads cached files only.
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use axum::{
    Json,
    extract::State,
    routing::{MethodRouter, get},
};
use nicegal_core::embedding::{
    ImageEmbedder, ImageEmbedderOptions, ImageQueryEmbedder, ImageQueryEmbedderOptions,
    TextEmbedder, TextEmbedderOptions,
};
use serde::Serialize;

use super::{AppState, error::ApiError};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelStatus {
    state: ModelState,
    error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
enum ModelState {
    NotLoaded,
    Preparing,
    Ready,
    Failed,
    Unsupported,
}

type CachedLoader<T, O> = fn(&O) -> Result<Option<T>>;

pub(crate) struct LazyModel<T, O> {
    pub(crate) options: O,
    session: OnceLock<Arc<T>>,
    preparation: Mutex<()>,
    status: Mutex<ModelStatus>,
    loader: fn(&O) -> Result<T>,
    cached_loader: Option<CachedLoader<T, O>>,
    name: &'static str,
}

impl<T, O> LazyModel<T, O> {
    fn new(options: O, name: &'static str, loader: fn(&O) -> Result<T>) -> Self {
        Self {
            options,
            session: OnceLock::new(),
            preparation: Mutex::new(()),
            status: Mutex::new(ModelStatus {
                state: ModelState::NotLoaded,
                error: None,
            }),
            loader,
            cached_loader: None,
            name,
        }
    }

    pub(super) fn status(&self) -> ModelStatus {
        self.status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub(super) fn prepare(&self) -> Result<Arc<T>> {
        self.prepare_with(|| (self.loader)(&self.options).map(Some))?
            .context("download-capable loader returned no model")
    }

    fn prepare_with(&self, loader: impl FnOnce() -> Result<Option<T>>) -> Result<Option<Arc<T>>> {
        let _guard = self.preparation.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(session) = self.session.get() {
            return Ok(Some(Arc::clone(session)));
        }
        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = ModelStatus {
            state: ModelState::Preparing,
            error: None,
        };
        // No session is published until the loader has returned successfully. Catching a
        // loader panic here keeps status truthful and lets the user retry with fresh sessions.
        // AssertUnwindSafe applies only to the loader boundary; options are read-only.
        let loaded = catch_unwind(AssertUnwindSafe(loader))
            .unwrap_or_else(|panic| {
                let message = panic.downcast_ref::<String>().map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown model loader failure");
                Err(anyhow::anyhow!("Model loader panicked: {message}"))
            }).with_context(|| {
            format!(
                "Preparing {} failed. Check your connection and available disk space, then retry model preparation in Settings",
                self.name
            )
        });
        match loaded {
            Ok(Some(session)) => {
                let session = Arc::new(session);
                let _ = self.session.set(Arc::clone(&session));
                *self.status.lock().unwrap_or_else(|e| e.into_inner()) = ModelStatus {
                    state: ModelState::Ready,
                    error: None,
                };
                Ok(Some(session))
            }
            Ok(None) => {
                *self.status.lock().unwrap_or_else(|e| e.into_inner()) = ModelStatus {
                    state: ModelState::NotLoaded,
                    error: None,
                };
                Ok(None)
            }
            Err(error) => {
                *self.status.lock().unwrap_or_else(|e| e.into_inner()) = ModelStatus {
                    state: ModelState::Failed,
                    error: Some(format!("{error:#}")),
                };
                Err(error)
            }
        }
    }

    /// Searches may compile cached sessions but never invoke the download-capable loader.
    pub(super) fn ready_or_cached(&self) -> Result<Arc<T>, ApiError> {
        if let Some(session) = self.session.get() {
            return Ok(Arc::clone(session));
        }
        if let Some(loader) = self.cached_loader {
            match self.prepare_with(|| loader(&self.options)) {
                Ok(Some(session)) => return Ok(session),
                Ok(None) => {}
                Err(error) => return Err(ApiError::models_not_ready(format!("{error:#}"))),
            }
        }
        self.ready()
    }

    #[cfg(test)]
    pub(super) fn without_cached_loading(mut self) -> Self {
        self.cached_loader = None;
        self
    }

    pub(super) fn ready(&self) -> Result<Arc<T>, ApiError> {
        self.session.get().map(Arc::clone).ok_or_else(|| {
            ApiError::models_not_ready(format!(
                "{} is not ready. Start indexing or prepare search models in Settings.",
                self.name
            ))
        })
    }
}

pub(crate) type TextModel = LazyModel<TextEmbedder, TextEmbedderOptions>;
pub(crate) type ImageModel = LazyModel<ImageEmbedder, ImageEmbedderOptions>;
pub(crate) type ImageQueryModel = LazyModel<ImageQueryEmbedder, ImageQueryEmbedderOptions>;

macro_rules! model {
    ($alias:ident, $session:ident, $options:ident, $kind:ty, $name:literal, $cached:expr) => {
        impl $alias {
            pub(crate) fn deferred(options: $options) -> Self {
                let mut model = Self::new(options, $name, $session::load);
                model.cached_loader = $cached;
                model
            }
            pub(super) fn model(&self) -> $kind {
                self.options.model
            }
            pub(crate) fn dimensions(&self) -> usize {
                self.model().dimensions()
            }
        }
    };
}
model!(
    TextModel,
    TextEmbedder,
    TextEmbedderOptions,
    nicegal_core::embedding::TextEmbeddingModel,
    "Text search model",
    Some(TextEmbedder::load_cached)
);
model!(
    ImageModel,
    ImageEmbedder,
    ImageEmbedderOptions,
    nicegal_core::embedding::ImageEmbeddingModel,
    "Image model",
    Some(ImageEmbedder::load_cached)
);
model!(
    ImageQueryModel,
    ImageQueryEmbedder,
    ImageQueryEmbedderOptions,
    nicegal_core::embedding::ImageEmbeddingModel,
    "Image description model",
    Some(ImageQueryEmbedder::load_cached)
);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusResponse {
    text: ModelStatus,
    clip_image: ModelStatus,
    clip_text: ModelStatus,
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(status)
}
async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    Json(StatusResponse {
        text: state.embedder.status(),
        clip_image: state.image_embedder.status(),
        clip_text: if state.image_embedder.model().supports_text_queries() {
            state.image_query_embedder.status()
        } else {
            ModelStatus {
                state: ModelState::Unsupported,
                error: None,
            }
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    #[ignore = "requires the search models to have been downloaded by explicit setup"]
    fn cached_query_models_reload_and_embed_without_setup() {
        super::super::tests::initialize_test_runtime();
        let text = TextModel::deferred(TextEmbedderOptions::default());
        let clip = ImageQueryModel::deferred(ImageQueryEmbedderOptions::default());
        assert_eq!(text.status().state, ModelState::NotLoaded);
        assert_eq!(clip.status().state, ModelState::NotLoaded);
        let text_vector = text
            .ready_or_cached()
            .expect("text model cached by setup")
            .embed_query("a red car")
            .unwrap();
        let clip_vector = clip
            .ready_or_cached()
            .expect("CLIP text model cached by setup")
            .embed_query("a red car")
            .unwrap();
        assert_eq!(text_vector.len(), text.dimensions());
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
        let model = LazyModel::new(Arc::clone(&attempts), "test model", |attempts| {
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("simulated download failure");
            }
            Ok(42)
        });
        assert_eq!(model.status().state, ModelState::NotLoaded);
        assert_eq!(model.ready().unwrap_err().code.as_str(), "models_not_ready");
        assert_eq!(attempts.load(Ordering::SeqCst), 0);
        assert!(model.prepare().is_err());
        assert_eq!(model.status().state, ModelState::Failed);
        assert!(
            model
                .status()
                .error
                .unwrap()
                .contains("simulated download failure")
        );
        let session = model.prepare().unwrap();
        assert_eq!(*session, 42);
        assert_eq!(model.status().state, ModelState::Ready);
        assert!(model.status().error.is_none());
        assert!(Arc::ptr_eq(&session, &model.prepare().unwrap()));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn preparation_is_observable_without_waiting_for_the_loader() {
        let entered = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let model = Arc::new(LazyModel::new(
            (Arc::clone(&entered), Arc::clone(&release)),
            "test model",
            |barriers| {
                barriers.0.wait();
                barriers.1.wait();
                Ok(42)
            },
        ));
        let worker_model = Arc::clone(&model);
        let worker = std::thread::spawn(move || worker_model.prepare().unwrap());
        entered.wait();
        assert_eq!(model.status().state, ModelState::Preparing);
        assert!(model.ready().is_err());
        release.wait();
        assert_eq!(*worker.join().unwrap(), 42);
        assert_eq!(model.status().state, ModelState::Ready);
    }

    #[test]
    fn a_loader_panic_is_a_visible_retryable_failure() {
        let model = LazyModel::new(AtomicUsize::new(0), "test model", |attempts| {
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("simulated loader panic");
            }
            Ok(42)
        });
        assert!(model.prepare().is_err());
        assert_eq!(model.status().state, ModelState::Failed);
        assert!(
            model
                .status()
                .error
                .unwrap()
                .contains("simulated loader panic")
        );
        assert!(model.ready().is_err());
        assert_eq!(*model.prepare().unwrap(), 42);
        assert_eq!(model.status().state, ModelState::Ready);
    }

    #[test]
    fn searches_use_only_the_cached_loader_and_reuse_its_session() {
        let downloads = Arc::new(AtomicUsize::new(0));
        let mut model = LazyModel::new(Arc::clone(&downloads), "test model", |downloads| {
            downloads.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        });
        model.cached_loader = Some(|_| Ok(None));
        assert_eq!(
            model.ready_or_cached().unwrap_err().code.as_str(),
            "models_not_ready"
        );
        assert_eq!(model.status().state, ModelState::NotLoaded);
        assert_eq!(downloads.load(Ordering::SeqCst), 0);
        model.cached_loader = Some(|_| Ok(Some(42)));
        let session = model.ready_or_cached().unwrap();
        assert_eq!(*session, 42);
        assert_eq!(model.status().state, ModelState::Ready);
        assert!(Arc::ptr_eq(&session, &model.ready_or_cached().unwrap()));
        assert_eq!(downloads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_corrupt_cached_model_does_not_trigger_a_download() {
        let mut model: LazyModel<i32, ()> =
            LazyModel::new((), "test model", |_| panic!("must not download"));
        model.cached_loader = Some(|_| anyhow::bail!("invalid cached ONNX model"));
        assert!(model.ready_or_cached().is_err());
        assert_eq!(model.status().state, ModelState::Failed);
        assert!(
            model
                .status()
                .error
                .unwrap()
                .contains("invalid cached ONNX model")
        );
    }
}
