//! Timing history for one inference session, so a failed run is logged with what led up to it.
//!
//! A GPU fault surfaces only as an error from the run that hit it. The history kept here says
//! whether runs were slowing down beforehand, how long the failing run waited before it errored
//! (a run held near the Windows GPU timeout points to a hang rather than a rejected command), and
//! how long the session had been working.

use std::collections::VecDeque;
use std::fmt::{self, Write as _};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{error, info, warn};

/// Runs kept for the failure report.
const RECENT_RUNS: usize = 32;
/// How often steady work logs a summary line.
const SUMMARY_INTERVAL: Duration = Duration::from_secs(60);
/// Half the default Windows GPU timeout: a run this long is close to being reset.
const SLOW_RUN: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RunKind {
    Embedding,
    Patches,
}

impl fmt::Display for RunKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Embedding => "embedding",
            Self::Patches => "patches",
        })
    }
}

/// The inputs of one run: real images and the padded count actually submitted.
#[derive(Debug, Clone, Copy)]
pub(super) struct RunShape {
    pub kind: RunKind,
    pub images: usize,
    pub submitted: usize,
}

#[derive(Debug, Clone, Copy)]
struct Run {
    sequence: u64,
    elapsed: Duration,
}

struct State {
    runs: u64,
    images: u64,
    recent: VecDeque<Run>,
    slowest: Option<Run>,
    last_success: Option<Instant>,
    window_start: Instant,
    window_runs: u64,
    window_images: u64,
    window_elapsed: Duration,
    window_max: Duration,
}

pub(super) struct RunLog {
    /// Identifies the session in every line: model and execution provider.
    session: String,
    loaded: Instant,
    state: Mutex<State>,
}

impl RunLog {
    pub fn new(session: String) -> Self {
        let now = Instant::now();
        Self {
            session,
            loaded: now,
            state: Mutex::new(State {
                runs: 0,
                images: 0,
                recent: VecDeque::with_capacity(RECENT_RUNS),
                slowest: None,
                last_success: None,
                window_start: now,
                window_runs: 0,
                window_images: 0,
                window_elapsed: Duration::ZERO,
                window_max: Duration::ZERO,
            }),
        }
    }

    /// Time `run`, record it, and log a failure with the session's recent history.
    pub fn record<T>(&self, shape: RunShape, run: impl FnOnce() -> Result<T>) -> Result<T> {
        let started = Instant::now();
        let result = run();
        let elapsed = started.elapsed();
        // History is diagnostic; a poisoned lock must not turn a good run into a failure.
        let Ok(mut state) = self.state.lock() else {
            return result;
        };
        let sequence = state.runs + 1;
        match &result {
            Ok(_) => self.succeeded(&mut state, shape, Run { sequence, elapsed }),
            Err(failure) => self.failed(&state, shape, sequence, elapsed, failure),
        }
        result
    }

    fn succeeded(&self, state: &mut State, shape: RunShape, run: Run) {
        let now = Instant::now();
        state.runs = run.sequence;
        state.images += shape.images as u64;
        state.last_success = Some(now);
        if state.recent.len() == RECENT_RUNS {
            state.recent.pop_front();
        }
        state.recent.push_back(run);
        if state.slowest.is_none_or(|slowest| run.elapsed > slowest.elapsed) {
            state.slowest = Some(run);
        }
        state.window_runs += 1;
        state.window_images += shape.images as u64;
        state.window_elapsed += run.elapsed;
        state.window_max = state.window_max.max(run.elapsed);

        if run.elapsed >= SLOW_RUN {
            warn!(
                session = %self.session,
                kind = %shape.kind,
                run = run.sequence,
                images = shape.images,
                submitted = shape.submitted,
                elapsed_ms = millis(run.elapsed),
                recent_ms = %recent(&state.recent),
                "slow inference run"
            );
        }
        if now.duration_since(state.window_start) >= SUMMARY_INTERVAL {
            info!(
                session = %self.session,
                runs = state.window_runs,
                images = state.window_images,
                mean_ms = millis(state.window_elapsed) / state.window_runs as f64,
                max_ms = millis(state.window_max),
                total_runs = state.runs,
                slowest_ms = state.slowest.map_or(0.0, |slowest| millis(slowest.elapsed)),
                session_age_s = self.loaded.elapsed().as_secs(),
                "inference summary"
            );
            state.window_start = now;
            state.window_runs = 0;
            state.window_images = 0;
            state.window_elapsed = Duration::ZERO;
            state.window_max = Duration::ZERO;
        }
    }

    fn failed(
        &self,
        state: &State,
        shape: RunShape,
        sequence: u64,
        elapsed: Duration,
        failure: &anyhow::Error,
    ) {
        let message = format!("{failure:#}");
        error!(
            session = %self.session,
            kind = %shape.kind,
            run = sequence,
            images = shape.images,
            submitted = shape.submitted,
            failed_after_ms = millis(elapsed),
            gpu_status = gpu_status(&message).unwrap_or("none"),
            completed_runs = state.runs,
            completed_images = state.images,
            session_age_s = self.loaded.elapsed().as_secs(),
            since_last_success_ms = state.last_success.map(|at| millis(at.elapsed())),
            slowest_ms = state.slowest.map(|slowest| millis(slowest.elapsed)),
            slowest_run = state.slowest.map(|slowest| slowest.sequence),
            recent_ms = %recent(&state.recent),
            error = %message,
            "inference run failed"
        );
    }
}

/// The DXGI status a DirectML failure carries, by name. Each means the device is lost: the
/// session cannot run again.
fn gpu_status(message: &str) -> Option<&'static str> {
    let message = message.to_ascii_uppercase();
    [
        ("887A0005", "DXGI_ERROR_DEVICE_REMOVED"),
        ("887A0006", "DXGI_ERROR_DEVICE_HUNG"),
        ("887A0007", "DXGI_ERROR_DEVICE_RESET"),
        ("887A0020", "DXGI_ERROR_DRIVER_INTERNAL_ERROR"),
    ]
    .into_iter()
    .find(|(code, _)| message.contains(code))
    .map(|(_, name)| name)
}

/// `sequence:milliseconds` for each recent run, oldest first.
fn recent(runs: &VecDeque<Run>) -> String {
    let mut text = String::with_capacity(runs.len() * 10);
    for run in runs {
        if !text.is_empty() {
            text.push(' ');
        }
        let _ = write!(text, "{}:{:.0}", run.sequence, millis(run.elapsed));
    }
    text
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_dxgi_status_in_a_directml_failure() {
        let message = "Non-zero status code returned while running DmlFusedNode_0_0 node. \
                       Exception(2) tid(2f3c) 887A0006 The GPU will not respond to more commands";
        assert_eq!(gpu_status(message), Some("DXGI_ERROR_DEVICE_HUNG"));
        assert_eq!(gpu_status("shape mismatch"), None);
    }

    #[test]
    fn a_failed_run_keeps_its_error_and_does_not_count_as_completed() {
        let log = RunLog::new("test".to_owned());
        let shape = RunShape {
            kind: RunKind::Embedding,
            images: 3,
            submitted: 4,
        };
        assert_eq!(log.record(shape, || Ok(7)).unwrap(), 7);
        let failure = log
            .record(shape, || -> Result<()> { anyhow::bail!("lost") })
            .unwrap_err();
        assert_eq!(failure.to_string(), "lost");
        let state = log.state.lock().unwrap();
        assert_eq!((state.runs, state.images, state.recent.len()), (1, 3, 1));
    }
}
