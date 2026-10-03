//! One job's state, progress, and published snapshots.
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use camino::Utf8PathBuf as PathBuf;
use nicegal_core::hub::{DownloadObserver, ModelSource};
use parking_lot::{Mutex, MutexGuard};
use serde::Serialize;
use tokio::sync::watch;
use tracing::{error, info};

use super::cancel_if;
use super::progress::{refresh_phase_throughput, set_phase};
use super::routes::wire_job_id;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) enum JobKind {
    OcrModelLoad,
    ModelPrepare,
    LibraryScan,
    ThumbnailGenerate,
    TextEmbed,
    ImageEmbed,
    PruneMissing,
    LibraryPurge,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) enum JobStatus {
    Queued,
    Running,
    Cancelling,
    Cancelled,
    Completed,
    Failed,
}

impl JobStatus {
    pub(super) fn is_terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::Completed | Self::Failed)
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) enum JobPhase {
    Queued,
    DownloadingModels,
    LoadingModels,
    Scanning,
    Cataloging,
    Thumbnails,
    Ocr,
    ImageEmbedding,
    TextEmbedding,
    Pruning,
    Cleanup,
    Finished,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(super) struct JobProgress {
    pub(super) discovered: usize,
    pub(super) total: Option<u64>,
    pub(super) phase_completed: u64,
    pub(super) processed: usize,
    pub(super) cataloged: usize,
    pub(super) thumbnails_generated: usize,
    pub(super) thumbnail_failures: usize,
    pub(super) prune_candidates: usize,
    pub(super) embedded: usize,
    pub(super) indexed: usize,
    pub(super) skipped: usize,
    pub(super) failed: usize,
    pub(super) deleted: usize,
    pub(super) downloaded_bytes: usize,
    pub(super) download_total_bytes: usize,
    /// Current file only; absent while loading sessions or using cached files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) download: Option<ModelDownload>,
    pub(super) models_loaded: usize,
    /// Work per second since the current phase began, in the unit [`phase_work`] counts.
    pub(super) items_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelDownload {
    pub(super) model_id: String,
    pub(super) filename: String,
    pub(super) downloaded_bytes: usize,
    pub(super) total_bytes: usize,
}

impl DownloadObserver for Job {
    fn progress(&self, source: &ModelSource, downloaded: usize, total: usize) {
        let mut data = self.data();
        data.progress.download = Some(ModelDownload {
            model_id: source.model_id.clone(),
            filename: source.filename.clone(),
            downloaded_bytes: downloaded,
            total_bytes: total,
        });
        self.publish(&mut data);
    }

    fn finish(&self) {
        let mut data = self.data();
        data.progress.download = None;
        self.publish(&mut data);
    }

    fn download_cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub(in crate::api) struct IndexStages {
    pub(in crate::api) ocr: bool,
    pub(in crate::api) image: bool,
    pub(in crate::api) text: bool,
}

/// Where one folder of a `libraryScan` stands.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(in crate::api) enum FolderState {
    Queued,
    Scanning,
    /// Walked and cataloged completely; the library's indexes are still catching up.
    Scanned,
    /// Scanned, and every enabled index has caught up with it.
    Completed,
    /// Walked, but some entries could not be read or a debug limit stopped it. Not cleaned up.
    Incomplete,
    /// Offline or unreadable. Its cached entries are kept and it stays pending.
    Unavailable,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(super) struct FolderProgress {
    pub(super) path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) scan_mode: Option<&'static str>,
    pub(super) state: FolderState,
    /// Files found by this folder's walk.
    pub(super) discovered: usize,
    /// Files confirmed in the catalog by this walk, whether new, changed, or already current.
    pub(super) cataloged: usize,
    pub(super) failed: usize,
    pub(super) error: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct JobData {
    pub(super) index_stages: Option<IndexStages>,
    /// Per-folder progress of a `libraryScan`, in scan order.
    pub(super) folders: Option<Vec<FolderProgress>>,
    /// The folder whose walk the current catalog events belong to.
    pub(super) current_folder: Option<usize>,
    pub(super) status: JobStatus,
    pub(super) phase: JobPhase,
    pub(super) progress: JobProgress,
    pub(super) error: Option<String>,
    pub(super) errors: Vec<JobItemError>,
    pub(super) active_asset_paths: BTreeSet<PathBuf>,
    pub(super) phase_started_at: Option<Instant>,
    /// Work done in the current phase, as [`phase_work`] counts it.
    pub(super) phase_work: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(super) struct JobItemError {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) path: Option<PathBuf>,
    pub(super) message: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(in crate::api) struct JobResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) index_stages: Option<IndexStages>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) folders: Option<Vec<FolderProgress>>,
    pub(super) job_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) library_id: Option<i64>,
    #[serde(rename = "type")]
    pub(super) kind: JobKind,
    pub(super) status: JobStatus,
    pub(super) phase: JobPhase,
    pub(super) progress: JobProgress,
    /// All source assets currently being worked. This is a set because OCR and CLIP decode
    /// concurrently; serialized ordering is stable for inexpensive UI updates.
    pub(super) active_asset_paths: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<String>,
    pub(super) errors: Vec<JobItemError>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct JobListResponse {
    pub(super) active_job_id: Option<String>,
    pub(super) jobs: Vec<JobResponse>,
}

pub(in crate::api) struct Job {
    pub(super) id: u64,
    pub(super) kind: JobKind,
    pub(super) library_id: Option<i64>,
    pub(super) scan_requests: Mutex<Vec<(PathBuf, i64)>>,
    pub(super) cancel_requested: Arc<AtomicBool>,
    pub(super) data: Mutex<JobData>,
    pub(super) updates: watch::Sender<JobResponse>,
}

impl Job {
    #[cfg(test)]
    pub(super) fn new(id: u64, kind: JobKind) -> Self {
        Self::new_for(id, kind, None)
    }

    pub(super) fn new_for(id: u64, kind: JobKind, library_id: Option<i64>) -> Self {
        let data = JobData {
            index_stages: None,
            folders: None,
            current_folder: None,
            status: JobStatus::Queued,
            phase: JobPhase::Queued,
            progress: JobProgress::default(),
            error: None,
            errors: Vec::new(),
            active_asset_paths: BTreeSet::new(),
            phase_started_at: None,
            phase_work: 0,
        };
        let (updates, _) = watch::channel(Self::response_from_data(id, kind, library_id, &data));
        Self {
            id,
            kind,
            library_id,
            scan_requests: Mutex::new(Vec::new()),
            cancel_requested: Arc::new(AtomicBool::new(false)),
            data: Mutex::new(data),
            updates,
        }
    }

    pub(super) fn data(&self) -> MutexGuard<'_, JobData> {
        self.data.lock()
    }

    pub(in crate::api) fn response(&self) -> JobResponse {
        self.updates.borrow().clone()
    }

    pub(super) fn response_from_data(
        id: u64,
        kind: JobKind,
        library_id: Option<i64>,
        data: &JobData,
    ) -> JobResponse {
        JobResponse {
            index_stages: data.index_stages,
            folders: data.folders.clone(),
            job_id: wire_job_id(id),
            library_id,
            kind,
            status: data.status,
            phase: data.phase,
            progress: data.progress.clone(),
            active_asset_paths: data.active_asset_paths.iter().cloned().collect(),
            error: data.error.clone(),
            errors: data.errors.clone(),
        }
    }

    pub(super) fn publish(&self, data: &mut JobData) {
        refresh_phase_throughput(data);
        self.updates.send_replace(Self::response_from_data(
            self.id,
            self.kind,
            self.library_id,
            data,
        ));
    }

    pub(super) fn subscribe(&self) -> watch::Receiver<JobResponse> {
        self.updates.subscribe()
    }

    // `begin`, `complete`, and `fail` skip `job_id`/`kind` on their own log lines: all three run
    // only from inside the `job` span `start()` opens below, which already carries both, and the
    // compact writer appends ambient span fields to every event inside it. `request_cancel` is
    // different — it runs on the HTTP handler's or the shutdown path's task, outside that span —
    // so it states its own identity.

    pub(super) fn begin(&self) -> bool {
        let mut data = self.data();
        if self.cancel_requested.load(Ordering::Acquire) {
            data.status = JobStatus::Cancelled;
            data.phase = JobPhase::Finished;
            data.progress.download = None;
            data.active_asset_paths.clear();
            self.publish(&mut data);
            info!("job cancelled before it started");
            return false;
        }
        data.status = JobStatus::Running;
        self.publish(&mut data);
        info!("job started");
        true
    }

    pub(super) fn request_cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
        let mut data = self.data();
        if !data.status.is_terminal() {
            data.status = if data.status == JobStatus::Queued {
                JobStatus::Cancelled
            } else {
                JobStatus::Cancelling
            };
            if data.status == JobStatus::Cancelled {
                data.phase = JobPhase::Finished;
            }
            self.publish(&mut data);
            info!(job_id = self.id, kind = ?self.kind, "job cancellation requested");
        }
    }

    pub(in crate::api) fn check_cancelled(&self) -> anyhow::Result<()> {
        cancel_if(self.cancel_requested.load(Ordering::Acquire))
    }

    pub(super) fn complete(&self, cancelled: bool) {
        let mut data = self.data();
        refresh_phase_throughput(&mut data);
        data.status = if cancelled {
            JobStatus::Cancelled
        } else {
            JobStatus::Completed
        };
        data.active_asset_paths.clear();
        data.phase = JobPhase::Finished;
        data.progress.download = None;
        self.publish(&mut data);
        info!(progress = ?data.progress, cancelled, "job finished");
    }

    pub(in crate::api) fn ocr_download_progress(
        &self,
        source: &ModelSource,
        downloaded: usize,
        total: usize,
        previous: usize,
        previous_total: usize,
    ) {
        let mut data = self.data();
        data.progress.downloaded_bytes = data
            .progress
            .downloaded_bytes
            .saturating_sub(previous)
            .saturating_add(downloaded);
        data.progress.download_total_bytes = data
            .progress
            .download_total_bytes
            .saturating_sub(previous_total)
            .saturating_add(total);
        data.progress.download = Some(ModelDownload {
            model_id: source.model_id.clone(),
            filename: source.filename.clone(),
            downloaded_bytes: downloaded,
            total_bytes: total,
        });
        self.publish(&mut data);
    }

    pub(in crate::api) fn downloading_models(&self) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::DownloadingModels);
        self.publish(&mut data);
    }

    pub(in crate::api) fn model_download_complete(&self) {
        let mut data = self.data();
        data.progress.processed += 1;
        self.publish(&mut data);
    }

    pub(in crate::api) fn set_index_stages(&self, stages: IndexStages) {
        let mut data = self.data();
        data.index_stages = Some(stages);
        self.publish(&mut data);
    }

    pub(in crate::api) fn set_folders(&self, paths: Vec<PathBuf>) {
        let mut data = self.data();
        data.folders = Some(
            paths
                .into_iter()
                .map(|path| FolderProgress {
                    path,
                    scan_mode: None,
                    state: FolderState::Queued,
                    discovered: 0,
                    cataloged: 0,
                    failed: 0,
                    error: None,
                })
                .collect(),
        );
        self.publish(&mut data);
    }

    pub(in crate::api) fn set_folder_scan_mode(&self, index: usize, mode: &'static str) {
        let mut data = self.data();
        if let Some(folder) = data
            .folders
            .as_mut()
            .and_then(|folders| folders.get_mut(index))
        {
            folder.scan_mode = Some(mode);
        }
        self.publish(&mut data);
    }

    pub(in crate::api) fn set_scan_requests(&self, requests: Vec<(PathBuf, i64)>) {
        *self.scan_requests.lock() = requests;
    }

    /// Attribute the following catalog events to folder `index` and mark it scanning.
    pub(in crate::api) fn enter_folder(&self, index: usize) {
        let mut data = self.data();
        data.current_folder = Some(index);
        if let Some(folder) = data
            .folders
            .as_mut()
            .and_then(|folders| folders.get_mut(index))
        {
            folder.state = FolderState::Scanning;
        }
        self.publish(&mut data);
    }

    pub(in crate::api) fn leave_folder(&self) {
        self.data().current_folder = None;
    }

    pub(in crate::api) fn finish_folder(
        &self,
        index: usize,
        state: FolderState,
        error: Option<String>,
    ) {
        let mut data = self.data();
        if let Some(folder) = data
            .folders
            .as_mut()
            .and_then(|folders| folders.get_mut(index))
        {
            folder.state = state;
            folder.error = error;
        }
        self.publish(&mut data);
    }

    pub(in crate::api) fn preparing_models(&self, count: u64) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::LoadingModels);
        data.progress.total = Some(count);
        self.publish(&mut data);
    }

    pub(in crate::api) fn loading_models(&self) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::LoadingModels);
        data.progress.total = Some(2);
        self.publish(&mut data);
    }

    pub(in crate::api) fn models_loaded(&self, count: usize) {
        let mut data = self.data();
        data.progress.models_loaded += count;
        data.progress.phase_completed += count as u64;
        data.phase_work += count as u64;
        self.publish(&mut data);
    }

    pub(super) fn fail(&self, message: impl Into<String>) {
        let mut data = self.data();
        refresh_phase_throughput(&mut data);
        data.status = JobStatus::Failed;
        data.progress.download = None;
        data.active_asset_paths.clear();
        data.phase = JobPhase::Finished;
        data.progress.download = None;
        let message = message.into();
        data.error = Some(message.clone());
        self.publish(&mut data);
        error!(error = %message, "job failed");
    }
}
