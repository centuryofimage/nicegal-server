use std::sync::Arc;
use std::time::{Duration, Instant};

pub(super) const CHECK_INTERVAL: Duration = Duration::from_secs(15);

/// How long a model stays in memory once nothing holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Residency {
    /// Query encoders: a search pays the whole load while the user types, and their weights are
    /// file-backed mappings the OS reclaims on its own, so they stay through a working session.
    Interactive,
    /// Indexing sessions: they hold device memory and are needed again only by the next scan, so
    /// they stay just long enough for back-to-back scans to share them.
    Batch,
}

impl Residency {
    pub(super) const fn idle_timeout(self) -> Duration {
        match self {
            Self::Interactive => Duration::from_secs(30 * 60),
            Self::Batch => Duration::from_secs(2 * 60),
        }
    }
}

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

    pub(super) fn expired(&mut self, now: Instant, timeout: Duration) -> bool {
        if Arc::strong_count(&self.model) > 1 {
            self.idle_since = None;
            return false;
        }
        let since = self.idle_since.get_or_insert(now);
        now.saturating_duration_since(*since) >= timeout
    }
}

impl super::AppState {
    pub(crate) fn start_model_cleanup(&self) -> tokio::task::JoinHandle<()> {
        let models = Arc::clone(&self.models);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(CHECK_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let models = Arc::clone(&models);
                // Session destruction can block; keep it off the async workers.
                if let Err(error) =
                    tokio::task::spawn_blocking(move || models.unload_idle(Instant::now())).await
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

    const TIMEOUT: Duration = Residency::Batch.idle_timeout();

    #[test]
    fn held_handles_protect_models_and_release_starts_a_full_idle_period() {
        let active = Arc::new(42);
        let mut session = CachedSession::new(Arc::clone(&active));
        let now = Instant::now();
        assert!(!session.expired(now, TIMEOUT));
        assert!(!session.expired(now + TIMEOUT * 2, TIMEOUT));
        drop(active);
        let released = now + TIMEOUT * 3;
        assert!(!session.expired(released, TIMEOUT));
        assert!(!session.expired(released + TIMEOUT - Duration::from_nanos(1), TIMEOUT));
        assert!(session.expired(released + TIMEOUT, TIMEOUT));
    }

    #[test]
    fn brief_use_between_checks_restarts_idle_period() {
        let mut session = CachedSession::new(Arc::new(42));
        let now = Instant::now();
        assert!(!session.expired(now, TIMEOUT));
        drop(session.acquire());
        assert!(!session.expired(now + TIMEOUT, TIMEOUT));
        assert!(session.expired(now + TIMEOUT * 2, TIMEOUT));
    }

    #[test]
    fn query_encoders_outlive_indexing_sessions() {
        assert!(Residency::Interactive.idle_timeout() > Residency::Batch.idle_timeout());
    }
}
