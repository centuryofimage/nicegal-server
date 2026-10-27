//! Shared model lifecycle. Loaders supply models; this module owns reuse,
//! serialization, cancellation, publication, status, and idle eviction.
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use nicegal_core::hub::DownloadObserver;
use parking_lot::Mutex;
use serde::Serialize;

use super::jobs;
use super::model_idle::{CachedSession, Residency};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelStatus {
    /// Whether the model can be used. `ready` survives idle eviction: an evicted model reloads
    /// from its cached files on its next use.
    pub(super) state: ModelState,
    pub(super) error: Option<String>,
    /// Whether a session is in memory right now.
    pub(super) resident: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum ModelState {
    NotLoaded,
    Preparing,
    Ready,
    Failed,
    Unsupported,
}

/// Receives a preparation's downloads and, only when a session is actually built, its load.
/// Acquiring a model that is already in memory reports nothing.
pub(super) trait PrepareObserver: DownloadObserver {
    /// `name` is the slot's role, such as "Image model"; `model_id` what it loads.
    fn loading_model(&self, _name: &'static str, _model_id: &'static str) {}
    fn model_loaded(&self) {}
}

impl PrepareObserver for () {}

struct SlotState<T> {
    session: Option<CachedSession<T>>,
    state: ModelState,
    error: Option<String>,
}

impl<T> SlotState<T> {
    fn set(&mut self, state: ModelState, error: Option<String>) {
        self.state = state;
        self.error = error;
    }

    /// Where a preparation that published nothing leaves the slot: as usable as before it, but
    /// not failed, since nothing failed.
    fn rest(&mut self, previous: ModelState) {
        let usable = self.session.is_some() || previous == ModelState::Ready;
        self.set(
            if usable {
                ModelState::Ready
            } else {
                ModelState::NotLoaded
            },
            None,
        );
    }
}

pub(super) struct ModelSlot<T> {
    state: Mutex<SlotState<T>>,
    /// Held while a session is built, so concurrent users wait for that build instead of
    /// starting their own.
    preparation: Mutex<()>,
    name: &'static str,
    /// The loaded model's identifier, such as its Hugging Face repository.
    model_id: &'static str,
    residency: Residency,
}

impl<T> ModelSlot<T> {
    pub(super) fn new(name: &'static str, residency: Residency) -> Self {
        Self {
            state: Mutex::new(SlotState {
                session: None,
                state: ModelState::NotLoaded,
                error: None,
            }),
            preparation: Mutex::new(()),
            name,
            model_id: name,
            residency,
        }
    }

    pub(super) fn identified_by(mut self, model_id: &'static str) -> Self {
        self.model_id = model_id;
        self
    }

    pub(super) fn name(&self) -> &'static str {
        self.name
    }

    pub(super) fn status(&self) -> ModelStatus {
        let state = self.state.lock();
        ModelStatus {
            state: state.state,
            error: state.error.clone(),
            resident: state.session.is_some(),
        }
    }

    /// Record that the model's files are in place without building a session. A failed model
    /// stays failed: only a successful build clears that.
    pub(super) fn mark_available(&self) {
        let mut state = self.state.lock();
        if state.state == ModelState::NotLoaded {
            state.set(ModelState::Ready, None);
        }
    }

    pub(super) fn acquire(&self) -> Option<Arc<T>> {
        self.state
            .lock()
            .session
            .as_mut()
            .map(CachedSession::acquire)
    }

    /// The session, building it with `loader` unless one is in memory. Downloads are cancellable
    /// through `observer`. A build that completes is published even if the job was cancelled
    /// meanwhile, so the work is kept for the next user. Call from a blocking worker, as
    /// compilation and contention can block.
    pub(super) fn prepare(
        &self,
        observer: &dyn PrepareObserver,
        loader: impl FnOnce() -> Result<Option<T>>,
    ) -> Result<Option<Arc<T>>> {
        let check = || jobs::cancel_if(observer.download_cancelled());
        check()?;
        if let Some(session) = self.acquire() {
            return Ok(Some(session));
        }
        let _gate = self.preparation.lock();
        check()?;
        if let Some(session) = self.acquire() {
            return Ok(Some(session));
        }
        let result = self.build(observer, loader)?;
        check()?;
        Ok(result)
    }

    fn build(
        &self,
        observer: &dyn PrepareObserver,
        loader: impl FnOnce() -> Result<Option<T>>,
    ) -> Result<Option<Arc<T>>> {
        let previous = {
            let mut state = self.state.lock();
            let previous = state.state;
            state.set(ModelState::Preparing, None);
            previous
        };
        tracing::info!(model = self.name, "model preparation started");
        observer.loading_model(self.name, self.model_id);
        let loaded = catch_loader(loader).with_context(|| format!("Preparing {} failed. Check your connection and available disk space, then reopen or rescan the library to retry", self.name));
        let mut state = self.state.lock();
        let mut retired = None;
        let result = match loaded {
            Ok(Some(model)) => {
                let model = Arc::new(model);
                retired = state
                    .session
                    .replace(CachedSession::new(Arc::clone(&model)));
                state.set(ModelState::Ready, None);
                tracing::info!(model = self.name, "model preparation completed");
                Ok(Some(model))
            }
            Ok(None) => {
                state.rest(previous);
                Ok(None)
            }
            Err(error) => {
                if jobs::is_cancelled(&error) {
                    state.rest(previous);
                } else {
                    tracing::error!(model = self.name, error = %format_args!("{error:#}"), "model preparation failed");
                    state.set(ModelState::Failed, Some(format!("{error:#}")));
                }
                Err(error)
            }
        };
        drop(state);
        drop(retired);
        if matches!(result, Ok(Some(_))) {
            observer.model_loaded();
        }
        result
    }

    pub(super) fn unload_idle(&self, now: Instant) {
        let Some(gate) = self.preparation.try_lock() else {
            return;
        };
        let timeout = self.residency.idle_timeout();
        let retired = {
            let mut state = self.state.lock();
            if !state
                .session
                .as_mut()
                .is_some_and(|session| session.expired(now, timeout))
            {
                return;
            }
            state.session.take()
        };
        drop(gate);
        drop(retired);
        tracing::info!(model = self.name, "unloaded idle model");
    }
}

fn catch_loader<T>(loader: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(loader)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("unknown model loader failure");
        Err(anyhow::anyhow!("Model loader panicked: {message}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    #[derive(Default)]
    struct Observer(AtomicBool);
    impl DownloadObserver for Observer {
        fn progress(&self, _: &nicegal_core::hub::ModelSource, _: usize, _: usize) {}
        fn download_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }
    impl PrepareObserver for Observer {}

    #[derive(Default)]
    struct LoadCounter {
        started: AtomicUsize,
        loaded: AtomicUsize,
    }
    impl DownloadObserver for LoadCounter {
        fn progress(&self, _: &nicegal_core::hub::ModelSource, _: usize, _: usize) {}
    }
    impl PrepareObserver for LoadCounter {
        fn loading_model(&self, _: &'static str, _: &'static str) {
            self.started.fetch_add(1, Ordering::Relaxed);
        }
        fn model_loaded(&self) {
            self.loaded.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn slot<T>(name: &'static str) -> ModelSlot<T> {
        ModelSlot::new(name, Residency::Batch)
    }

    #[test]
    fn only_an_actual_build_is_reported_as_loading() {
        let slot = slot("reported model");
        let observer = LoadCounter::default();
        let first = slot.prepare(&observer, || Ok(Some(42))).unwrap();
        let again = slot.prepare(&observer, || panic!("must reuse")).unwrap();
        assert!(Arc::ptr_eq(&first.unwrap(), &again.unwrap()));
        assert_eq!(observer.started.load(Ordering::Relaxed), 1);
        assert_eq!(observer.loaded.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn available_files_make_a_model_ready_without_a_session_but_never_mask_failure() {
        let available = slot::<i32>("available model");
        available.mark_available();
        let status = available.status();
        assert_eq!(status.state, ModelState::Ready);
        assert!(!status.resident);

        let failed = slot::<i32>("failed model");
        assert!(failed.prepare(&(), || anyhow::bail!("bad graph")).is_err());
        failed.mark_available();
        assert_eq!(failed.status().state, ModelState::Failed);
    }

    #[test]
    fn cancellation_during_compilation_publishes_but_does_not_return_the_model() {
        let slot = slot("cancelled model");
        let observer = Observer::default();
        let error = slot
            .prepare(&observer, || {
                observer.0.store(true, Ordering::Release);
                Ok(Some(42))
            })
            .unwrap_err();
        assert!(jobs::is_cancelled(&error));
        assert_eq!(slot.status().state, ModelState::Ready);
        let retained = slot.acquire().unwrap();
        observer.0.store(false, Ordering::Release);
        let reused = slot
            .prepare(&observer, || panic!("must reuse"))
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&retained, &reused));
    }

    #[test]
    fn cancellation_while_waiting_for_preparation_never_invokes_loader() {
        let slot = slot::<i32>("queued model");
        let observer = Observer::default();
        let gate = slot.preparation.lock();
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| slot.prepare(&observer, || panic!("must not load")));
            observer.0.store(true, Ordering::Release);
            drop(gate);
            assert!(jobs::is_cancelled(&waiting.join().unwrap().unwrap_err()));
        });
        assert_eq!(slot.status().state, ModelState::NotLoaded);
    }

    #[test]
    fn concurrent_preparations_build_once_and_share_the_session() {
        let slot = slot("shared model");
        let loads = AtomicUsize::new(0);
        let (entered, building) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let (slot, loads) = (&slot, &loads);
        std::thread::scope(|scope| {
            let first = scope.spawn(move || {
                slot.prepare(&(), || {
                    loads.fetch_add(1, Ordering::Relaxed);
                    entered.send(()).unwrap();
                    released.recv().unwrap();
                    Ok(Some(42))
                })
            });
            building.recv().unwrap();
            let second = scope.spawn(|| {
                slot.prepare(&(), || {
                    loads.fetch_add(1, Ordering::Relaxed);
                    Ok(Some(43))
                })
            });
            release.send(()).unwrap();
            let first = first.join().unwrap().unwrap().unwrap();
            let second = second.join().unwrap().unwrap().unwrap();
            assert!(Arc::ptr_eq(&first, &second));
        });
        assert_eq!(loads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_panicking_build_fails_the_model_and_can_be_retried() {
        let slot = slot("flaky model");
        let error = slot
            .prepare(&(), || panic!("fake compiler panic"))
            .unwrap_err();
        assert!(format!("{error:#}").contains("fake compiler panic"));
        assert_eq!(slot.status().state, ModelState::Failed);
        let retried = slot.prepare(&(), || Ok(Some(43))).unwrap().unwrap();
        assert_eq!(*retried, 43);
        assert_eq!(slot.status().state, ModelState::Ready);
    }

    #[test]
    fn eviction_frees_memory_but_the_model_stays_usable() {
        let slot = slot("evicted model");
        drop(slot.prepare(&(), || Ok(Some(42))).unwrap());
        let now = Instant::now();
        slot.unload_idle(now);
        assert!(slot.status().resident);
        slot.unload_idle(now + Residency::Batch.idle_timeout());
        let status = slot.status();
        assert_eq!(status.state, ModelState::Ready);
        assert!(!status.resident);
    }

    #[test]
    fn a_session_in_use_is_never_evicted() {
        let slot = slot("busy model");
        let held = slot.prepare(&(), || Ok(Some(42))).unwrap().unwrap();
        let now = Instant::now();
        slot.unload_idle(now);
        slot.unload_idle(now + Residency::Batch.idle_timeout() * 2);
        assert!(Arc::ptr_eq(&held, &slot.acquire().unwrap()));
    }

    #[test]
    fn residency_sets_how_long_an_idle_model_stays() {
        let interactive = ModelSlot::new("query model", Residency::Interactive);
        drop(interactive.prepare(&(), || Ok(Some(42))).unwrap());
        let now = Instant::now();
        interactive.unload_idle(now);
        interactive.unload_idle(now + Residency::Batch.idle_timeout() + Duration::from_secs(1));
        assert!(interactive.status().resident);
        interactive.unload_idle(now + Residency::Interactive.idle_timeout());
        assert!(!interactive.status().resident);
    }
}
