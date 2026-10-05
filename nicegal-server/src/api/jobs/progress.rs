//! Indexing events folded into a job's phase progress and throughput.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use nicegal_core::index::{IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta};

use super::job::{FolderProgress, Job, JobData, JobItemError, JobPhase, JobProgress};

impl IndexObserver for Job {
    fn is_cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    fn cancellation_token(&self) -> Option<Arc<AtomicBool>> {
        Some(Arc::clone(&self.cancel_requested))
    }

    fn on_event(&self, event: IndexEvent) {
        // Record every failure before the UI's bounded error list, including failures from
        // thumbnail workers that have no separate logger or attached event-stream client.
        if let IndexEvent::Error { path, message } = &event {
            tracing::warn!(job_id = self.id, library_id = self.library_id, ?path,
                error = %message, "job item failed");
        }
        let mut data = self.data();
        match event {
            IndexEvent::PhaseChanged(phase) => {
                let phase = match phase {
                    IndexPhase::Scanning => JobPhase::Scanning,
                    IndexPhase::Cataloging => JobPhase::Cataloging,
                    IndexPhase::Thumbnails => JobPhase::Thumbnails,
                    IndexPhase::Ocr => JobPhase::Ocr,
                    IndexPhase::ImageEmbedding => JobPhase::ImageEmbedding,
                    IndexPhase::TextEmbedding => JobPhase::TextEmbedding,
                    IndexPhase::Pruning => JobPhase::Pruning,
                    IndexPhase::Cleanup => JobPhase::Cleanup,
                };
                set_phase(&mut data, phase);
            }
            IndexEvent::Discovered { count } => {
                data.progress.discovered = count;
                if let Some(folder) = current_folder(&mut data) {
                    folder.discovered = count;
                }
            }
            IndexEvent::DiscoveryComplete { total } => {
                data.progress.total = Some(total as u64);
                if data.phase == JobPhase::Scanning {
                    data.progress.phase_completed = total as u64;
                    data.phase_work = total as u64;
                }
            }
            IndexEvent::Progress(delta) => {
                data.phase_work += phase_work(data.phase, &delta);
                apply_delta(&mut data.progress, delta);
                if let Some(folder) = current_folder(&mut data) {
                    folder.cataloged += delta.cataloged;
                    folder.failed += delta.failed;
                }
            }
            IndexEvent::ActiveAsset { path, active } => {
                if active {
                    data.active_asset_paths.insert(path);
                } else {
                    data.active_asset_paths.remove(&path);
                }
            }
            IndexEvent::Error { path, message } => {
                if data.errors.len() < 100 {
                    data.errors.push(JobItemError { path, message });
                }
            }
        }
        self.publish(&mut data);
    }
}

pub(super) fn current_folder(data: &mut JobData) -> Option<&mut FolderProgress> {
    let index = data.current_folder?;
    data.folders.as_mut()?.get_mut(index)
}

pub(super) fn apply_delta(progress: &mut JobProgress, delta: IndexProgressDelta) {
    progress.phase_completed += delta.phase_completed as u64;
    progress.processed += delta.processed;
    progress.cataloged += delta.cataloged;
    progress.thumbnails_generated += delta.thumbnails_generated;
    progress.thumbnail_failures += delta.thumbnail_failures;
    progress.prune_candidates += delta.prune_candidates;
    progress.embedded += delta.embedded;
    progress.indexed += delta.indexed;
    progress.skipped += delta.skipped;
    progress.failed += delta.failed;
    progress.deleted += delta.deleted;
}

pub(super) fn set_phase(data: &mut JobData, next: JobPhase) {
    if data.phase != next {
        tracing::info!(previous = ?data.phase, phase = ?next,
            completed = data.progress.phase_completed,
            active_assets = data.active_asset_paths.len(), "job phase changed");
        data.phase = next;
        data.progress.discovered = 0;
        data.progress.total = None;
        data.progress.phase_completed = 0;
        data.progress.items_per_second = None;
        data.phase_started_at = Some(Instant::now());
        data.phase_work = 0;
        data.throughput_samples.clear();
        data.active_asset_paths.clear();
    }
}

/// The throughput work a delta reports for `phase`: completed items, except that OCR leaves out
/// skipped assets so a resumed index does not count earlier work, and image embedding counts
/// model inputs, so each sampled video frame counts as one image.
pub(super) fn phase_work(phase: JobPhase, delta: &IndexProgressDelta) -> u64 {
    let work = match phase {
        JobPhase::ImageEmbedding => delta.images_inferred,
        JobPhase::Ocr => delta.phase_completed.saturating_sub(delta.skipped),
        _ => delta.phase_completed,
    };
    work as u64
}

pub(super) fn refresh_phase_throughput(data: &mut JobData) {
    refresh_phase_throughput_at(data, Instant::now());
}

/// Keep a five-second rolling window. Sampling at most ten times per second bounds storage
/// even when indexing emits thousands of events per second. Interpolate the oldest interval
/// at the window boundary so irregular updates do not change the averaging duration.
pub(super) fn refresh_phase_throughput_at(data: &mut JobData, now: Instant) {
    const WINDOW: Duration = Duration::from_secs(5);
    const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

    if matches!(data.phase, JobPhase::Queued | JobPhase::Finished) {
        return;
    }
    if data.phase_work == 0 {
        data.progress.items_per_second = None;
        return;
    }
    let Some(started_at) = data.phase_started_at else {
        return;
    };
    let samples = &mut data.throughput_samples;
    if samples.is_empty() {
        samples.push_back((started_at, 0));
    }
    let cutoff = now - WINDOW;
    // Retain one sample before the cutoff for interpolation.
    while samples.len() > 1 && samples[1].0 <= cutoff {
        samples.pop_front();
    }
    let (oldest_at, oldest_work) = samples[0];
    let (window_start, baseline) = if oldest_at < cutoff {
        let (next_at, next_work) = samples.get(1).copied().unwrap_or((now, data.phase_work));
        let fraction = cutoff.duration_since(oldest_at).as_secs_f64()
            / next_at.duration_since(oldest_at).as_secs_f64();
        (
            cutoff,
            oldest_work as f64 + (next_work - oldest_work) as f64 * fraction,
        )
    } else {
        (oldest_at, oldest_work as f64)
    };
    let elapsed = now.duration_since(window_start).as_secs_f64();
    if elapsed > 0.0 {
        data.progress.items_per_second = Some((data.phase_work as f64 - baseline) / elapsed);
    }
    if now.duration_since(samples.back().unwrap().0) >= SAMPLE_INTERVAL {
        samples.push_back((now, data.phase_work));
    }
}
