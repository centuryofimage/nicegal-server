//! Indexing events folded into a job's phase progress and throughput.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

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
        data.phase = next;
        data.progress.discovered = 0;
        data.progress.total = None;
        data.progress.phase_completed = 0;
        data.progress.items_per_second = None;
        data.phase_started_at = Some(Instant::now());
        data.phase_work = 0;
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
    let elapsed = started_at.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        data.progress.items_per_second = Some(data.phase_work as f64 / elapsed);
    }
}
