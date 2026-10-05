use std::sync::Arc;
use std::time::{Duration, Instant};

pub(super) const CHECK_INTERVAL: Duration = Duration::from_secs(15);
const IDLE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Access only under the owning cache's mutex. Callers hold strong references for
/// their entire operation; no weak session references may escape the cache.
pub(super) struct CachedSession<T> {
    model: Arc<T>,
    idle_since: Option<Instant>,
}

impl<T> CachedSession<T> {
    pub(super) fn new(model: Arc<T>) -> Self {
        Self {
            model,
            idle_since: None,
        }
    }

    pub(super) fn acquire(&mut self) -> Arc<T> {
        self.idle_since = None;
        Arc::clone(&self.model)
    }

    pub(super) fn get(&self) -> &T {
        &self.model
    }

    pub(super) fn expired(&mut self, now: Instant) -> bool {
        if Arc::strong_count(&self.model) > 1 {
            self.idle_since = None;
            return false;
        }
        let since = self.idle_since.get_or_insert(now);
        now.saturating_duration_since(*since) >= IDLE_TIMEOUT
    }
}

impl super::AppState {
    pub(crate) fn start_model_cleanup(&self) -> tokio::task::JoinHandle<()> {
        let state = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(CHECK_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let state = state.clone();
                // Session destruction can block; keep it off the async workers.
                if let Err(error) = tokio::task::spawn_blocking(move || {
                    let now = Instant::now();
                    state.embedder.unload_idle(now);
                    state.image_embedder.unload_idle(now);
                    state.image_query_embedder.unload_idle(now);
                    state.ocr_models.unload_idle(now);
                })
                .await
                {
                    tracing::error!(%error, "idle model cleanup failed");
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_handles_protect_models_and_release_starts_a_full_idle_period() {
        let active = Arc::new(42);
        let mut session = CachedSession::new(Arc::clone(&active));
        let now = Instant::now();
        assert!(!session.expired(now));
        assert!(!session.expired(now + IDLE_TIMEOUT * 2));
        drop(active);
        let released = now + IDLE_TIMEOUT * 3;
        assert!(!session.expired(released));
        assert!(!session.expired(released + IDLE_TIMEOUT - Duration::from_nanos(1)));
        assert!(session.expired(released + IDLE_TIMEOUT));
    }

    #[test]
    fn brief_use_between_checks_restarts_idle_period() {
        let mut session = CachedSession::new(Arc::new(42));
        let now = Instant::now();
        assert!(!session.expired(now));
        drop(session.acquire());
        assert!(!session.expired(now + IDLE_TIMEOUT));
        assert!(session.expired(now + IDLE_TIMEOUT * 2));
    }
}
