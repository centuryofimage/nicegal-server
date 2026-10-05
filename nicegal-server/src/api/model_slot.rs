//! Shared model lifecycle. Loaders supply models; this module owns reuse,
//! serialization, cancellation, publication, status, and idle eviction.
use std::future::{Future, poll_fn};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Instant;

use anyhow::{Context, Result};
use nicegal_core::hub::DownloadObserver;
use parking_lot::Mutex;
use serde::Serialize;

use super::{jobs, model_idle::CachedSession};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelStatus {
    pub(super) state: ModelState,
    pub(super) error: Option<String>,
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

struct SlotState<T, K> {
    session: Option<(K, CachedSession<T>)>,
    status: ModelStatus,
}

pub(super) struct ModelSlot<T, K = ()> {
    state: Mutex<SlotState<T, K>>,
    preparation: tokio::sync::Mutex<()>,
    name: &'static str,
}

/// Download/file-resolution stages are cancellable. Compilation deliberately has
/// no post-check: a successful model must be published before cancellation returns.
#[derive(Clone, Copy)]
pub(super) struct Preparation<'a> {
    observer: &'a dyn DownloadObserver,
}

impl Preparation<'_> {
    fn check(&self) -> Result<()> {
        jobs::cancel_if(self.observer.download_cancelled())
    }

    pub(super) async fn resolve<T>(&self, work: impl Future<Output = Result<T>>) -> Result<T> {
        self.check()?;
        let result = work.await?;
        self.check()?;
        Ok(result)
    }

    /// Await preparation to completion; signal cancellation through the observer.
    /// Dropping/aborting this future detaches the non-interruptible worker and
    /// discards its result, allowing another preparation to start before it exits.
    pub(super) async fn compile_blocking<T: Send + 'static>(
        &self,
        compile: impl FnOnce() -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.check()?;
        tokio::task::spawn_blocking(move || catch_loader(compile))
            .await
            .context("model loader worker failed")?
    }
}

impl<T, K: PartialEq> ModelSlot<T, K> {
    pub(super) fn new(name: &'static str) -> Self {
        Self {
            state: Mutex::new(SlotState {
                session: None,
                status: ModelStatus {
                    state: ModelState::NotLoaded,
                    error: None,
                },
            }),
            preparation: tokio::sync::Mutex::new(()),
            name,
        }
    }

    pub(super) fn status(&self) -> ModelStatus {
        self.state.lock().status.clone()
    }

    pub(super) fn acquire(&self, key: &K) -> Option<Arc<T>> {
        self.state
            .lock()
            .session
            .as_mut()
            .and_then(|(loaded_key, session)| (loaded_key == key).then(|| session.acquire()))
    }

    pub(super) fn snapshot(&self) -> Option<Arc<T>> {
        self.state
            .lock()
            .session
            .as_mut()
            .map(|(_, session)| session.acquire())
    }

    /// Inspect without marking the model used or waiting for inference.
    pub(super) fn inspect<R>(&self, inspect: impl FnOnce(Option<&T>) -> R) -> R {
        let state = self.state.lock();
        inspect(state.session.as_ref().map(|(_, session)| session.get()))
    }

    /// Call from a blocking worker, as compilation and contention can block.
    pub(super) fn prepare(
        &self,
        key: K,
        observer: &dyn DownloadObserver,
        loader: impl FnOnce() -> Result<Option<T>>,
    ) -> Result<Option<Arc<T>>> {
        let context = Preparation { observer };
        context.check()?;
        if let Some(session) = self.acquire(&key) {
            return Ok(Some(session));
        }
        let _gate = self.preparation.blocking_lock();
        context.check()?;
        if let Some(session) = self.acquire(&key) {
            return Ok(Some(session));
        }
        let loading = self.begin();
        let loaded = catch_loader(loader);
        let result = loading.finish(key, loaded)?;
        context.check()?;
        Ok(result)
    }

    pub(super) async fn prepare_async<'a, F: Future<Output = Result<Option<T>>>>(
        &self,
        key: K,
        observer: &'a dyn DownloadObserver,
        loader: impl FnOnce(Preparation<'a>) -> F,
    ) -> Result<Option<Arc<T>>> {
        let context = Preparation { observer };
        context.check()?;
        if let Some(session) = self.acquire(&key) {
            return Ok(Some(session));
        }
        let _gate = self.preparation.lock().await;
        context.check()?;
        if let Some(session) = self.acquire(&key) {
            return Ok(Some(session));
        }
        let loading = self.begin();
        // Catch construction as well as polling; failures must leave the slot retryable.
        let loaded = match catch_loader(|| Ok(loader(context))) {
            Ok(future) => {
                let mut future = pin!(future);
                poll_fn(|cx| match catch_loader(|| Ok(future.as_mut().poll(cx))) {
                    Ok(poll) => poll,
                    Err(error) => Poll::Ready(Err(error)),
                })
                .await
            }
            Err(error) => Err(error),
        };
        let result = loading.finish(key, loaded)?;
        context.check()?;
        Ok(result)
    }

    fn begin(&self) -> Loading<'_, T, K> {
        self.state.lock().status = ModelStatus {
            state: ModelState::Preparing,
            error: None,
        };
        tracing::info!(model = self.name, "model preparation started");
        Loading {
            slot: self,
            finished: false,
        }
    }

    pub(super) fn unload_idle(&self, now: Instant) {
        let Ok(gate) = self.preparation.try_lock() else {
            return;
        };
        let retired = {
            let mut state = self.state.lock();
            if !state
                .session
                .as_mut()
                .is_some_and(|(_, session)| session.expired(now))
            {
                return;
            }
            let retired = state.session.take();
            state.status = ModelStatus {
                state: ModelState::NotLoaded,
                error: None,
            };
            retired
        };
        drop(gate);
        drop(retired);
        tracing::info!(model = self.name, "unloaded idle model");
    }
}

struct Loading<'a, T, K> {
    slot: &'a ModelSlot<T, K>,
    finished: bool,
}

impl<T, K> Loading<'_, T, K> {
    fn finish(mut self, key: K, loaded: Result<Option<T>>) -> Result<Option<Arc<T>>> {
        let loaded = loaded.with_context(|| format!("Preparing {} failed. Check your connection and available disk space, then reopen or rescan the library to retry", self.slot.name));
        let mut state = self.slot.state.lock();
        let mut retired = None;
        let result = match loaded {
            Ok(Some(model)) => {
                let model = Arc::new(model);
                retired = state
                    .session
                    .replace((key, CachedSession::new(Arc::clone(&model))));
                state.status = ModelStatus {
                    state: ModelState::Ready,
                    error: None,
                };
                tracing::info!(model = self.slot.name, "model preparation completed");
                Ok(Some(model))
            }
            Ok(None) => {
                state.status = resting_status(state.session.is_some());
                Ok(None)
            }
            Err(error) => {
                state.status = if jobs::is_cancelled(&error) {
                    resting_status(state.session.is_some())
                } else {
                    tracing::error!(model = self.slot.name, error = %format_args!("{error:#}"), "model preparation failed");
                    ModelStatus {
                        state: ModelState::Failed,
                        error: Some(format!("{error:#}")),
                    }
                };
                Err(error)
            }
        };
        self.finished = true;
        drop(state);
        drop(retired);
        result
    }
}

impl<T, K> Drop for Loading<'_, T, K> {
    fn drop(&mut self) {
        if !self.finished {
            // Dropping an async preparation must not strand the slot in Preparing.
            let mut state = self.slot.state.lock();
            state.status = resting_status(state.session.is_some());
        }
    }
}

fn resting_status(loaded: bool) -> ModelStatus {
    ModelStatus {
        state: if loaded {
            ModelState::Ready
        } else {
            ModelState::NotLoaded
        },
        error: None,
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
    use std::time::Duration;

    #[derive(Default)]
    struct Observer(AtomicBool);
    impl DownloadObserver for Observer {
        fn progress(&self, _: &nicegal_core::hub::ModelSource, _: usize, _: usize) {}
        fn download_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[tokio::test]
    async fn cancellation_during_compilation_publishes_but_does_not_return_the_model() {
        let slot = ModelSlot::new("fake OCR");
        let observer = Arc::new(Observer::default());
        let compile_observer = Arc::clone(&observer);
        let error = slot
            .prepare_async((), observer.as_ref(), |preparation| async move {
                preparation
                    .compile_blocking(move || {
                        compile_observer.0.store(true, Ordering::Release);
                        Ok(Some(42))
                    })
                    .await
            })
            .await
            .unwrap_err();
        assert!(jobs::is_cancelled(&error));
        assert_eq!(slot.status().state, ModelState::Ready);
        let retained = slot.acquire(&()).unwrap();
        observer.0.store(false, Ordering::Release);
        let reused = slot
            .prepare_async((), observer.as_ref(), |_| async { panic!("must reuse") })
            .await
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&retained, &reused));
    }

    #[tokio::test]
    async fn cancelled_download_stage_cannot_proceed_to_compilation() {
        let slot = ModelSlot::<i32>::new("fake OCR");
        let observer = Observer::default();
        let observer = &observer;
        let error = slot
            .prepare_async((), observer, |preparation| async move {
                preparation
                    .resolve(async {
                        observer.0.store(true, Ordering::Release);
                        Ok(())
                    })
                    .await?;
                panic!("cancelled download must not reach compilation");
            })
            .await
            .unwrap_err();
        assert!(jobs::is_cancelled(&error));
        assert_eq!(slot.status().state, ModelState::NotLoaded);
        assert!(slot.snapshot().is_none());
    }

    #[tokio::test]
    async fn cancellation_while_waiting_for_preparation_never_invokes_loader() {
        let slot = ModelSlot::<i32>::new("queued model");
        let observer = Observer::default();
        let gate = slot.preparation.lock().await;
        let mut waiting =
            pin!(slot.prepare_async((), &observer, |_| async { panic!("must not load") }));
        tokio::select! {
            biased;
            _ = &mut waiting => panic!("must wait for preparation lock"),
            _ = tokio::task::yield_now() => (),
        }
        observer.0.store(true, Ordering::Release);
        drop(gate);
        assert!(jobs::is_cancelled(&waiting.await.unwrap_err()));
        assert_eq!(slot.status().state, ModelState::NotLoaded);
    }

    #[tokio::test]
    async fn concurrent_preparations_compile_once_and_share_the_session() {
        let slot = ModelSlot::new("shared model");
        let loads = AtomicUsize::new(0);
        let entered = tokio::sync::Notify::new();
        let release = tokio::sync::Notify::new();
        let first = slot.prepare_async((), &(), |_| async {
            loads.fetch_add(1, Ordering::Relaxed);
            entered.notify_one();
            release.notified().await;
            Ok(Some(42))
        });
        let second = async {
            entered.notified().await;
            release.notify_one();
            slot.prepare_async((), &(), |_| async {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok(Some(43))
            })
            .await
        };
        let (first, second) = tokio::join!(first, second);
        assert!(Arc::ptr_eq(
            &first.unwrap().unwrap(),
            &second.unwrap().unwrap()
        ));
        assert_eq!(loads.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn failed_replacement_preserves_previous_model_and_can_retry() {
        let slot = ModelSlot::new("keyed model");
        let original = slot
            .prepare_async("old", &(), |_| async { Ok(Some(42)) })
            .await
            .unwrap()
            .unwrap();
        let error = slot
            .prepare_async("new", &(), |_| async { panic!("fake compiler panic") })
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("fake compiler panic"));
        assert_eq!(slot.status().state, ModelState::Failed);
        assert!(Arc::ptr_eq(&original, &slot.acquire(&"old").unwrap()));
        assert!(slot.acquire(&"new").is_none());
        let replacement = slot
            .prepare_async("new", &(), |_| async { Ok(Some(43)) })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(*replacement, 43);
        assert_eq!(*original, 42);
        assert!(slot.acquire(&"old").is_none());
        assert_eq!(slot.status().state, ModelState::Ready);
    }

    #[tokio::test]
    async fn dropped_preparation_restores_state_and_releases_gate() {
        let slot = ModelSlot::<i32>::new("abandoned model");
        {
            let mut loading = pin!(slot.prepare_async((), &(), |_| std::future::pending()));
            tokio::select! {
                biased;
                _ = &mut loading => panic!("must stay pending"),
                _ = tokio::task::yield_now() => (),
            }
            assert_eq!(slot.status().state, ModelState::Preparing);
            // Cleanup must not block on preparation or change its status.
            slot.unload_idle(Instant::now());
            assert_eq!(slot.status().state, ModelState::Preparing);
        }
        assert_eq!(slot.status().state, ModelState::NotLoaded);
        assert_eq!(
            *slot
                .prepare_async((), &(), |_| async { Ok(Some(42)) })
                .await
                .unwrap()
                .unwrap(),
            42
        );
    }

    #[test]
    fn inspection_does_not_keep_an_idle_model_alive() {
        let slot = ModelSlot::new("inspected model");
        drop(slot.prepare((), &(), || Ok(Some(42))).unwrap());
        let now = Instant::now();
        slot.unload_idle(now);
        assert_eq!(slot.inspect(|model| model.copied()), Some(42));
        slot.unload_idle(now + Duration::from_secs(300));
        assert_eq!(slot.inspect(|model| model.copied()), None);
        assert_eq!(slot.status().state, ModelState::NotLoaded);
    }
}
