use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::hub::{DownloadObserver, ModelSource};
use nicegal_core::index::{IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta};
use nicegal_core::libraries::ScanOutcome;
use nicegal_core::thumbs::ThumbnailService;
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio_stream::Stream;
use tokio_stream::wrappers::WatchStream;
use tracing::{Instrument, Span, error, info, info_span};

use super::error::ApiError;
use super::extract::ApiJson;
use super::models::{ImageModel as ImageEmbedder, ImageQueryModel, TextModel as TextEmbedder};
use super::{
    AppState, Databases, image_embeddings, library_scan, ocr_models, prune_jobs, text_embeddings,
    thumbnails,
};

const MAX_RETAINED_JOBS: usize = 32;

#[derive(Debug)]
pub(crate) struct JobCancelled;

impl std::fmt::Display for JobCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("job cancelled")
    }
}

impl std::error::Error for JobCancelled {}

pub(crate) fn cancel_if(cancelled: bool) -> anyhow::Result<()> {
    if cancelled {
        Err(JobCancelled.into())
    } else {
        Ok(())
    }
}

pub(crate) fn is_cancelled(error: &anyhow::Error) -> bool {
    error.downcast_ref::<JobCancelled>().is_some() || nicegal_core::hub::is_cancellation(error)
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "type",
    content = "params",
    rename_all = "camelCase",
    deny_unknown_fields
)]
enum JobRequest {
    ModelPrepare(EmptyParams),
    OcrModelLoad(ocr_models::job::Request),
    LibraryScan(library_scan::Request),
    ThumbnailGenerate(thumbnails::job::Request),
    TextEmbed(text_embeddings::job::Request),
    ImageEmbed(image_embeddings::Request),
    PruneMissing(prune_jobs::Request),
    LibraryPurge(prune_jobs::LibraryPurgeRequest),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyParams {}

pub(super) enum JobSpec {
    ModelPrepare,
    OcrModelLoad(ocr_models::job::Spec),
    LibraryScan(library_scan::Spec),
    ThumbnailGenerate(thumbnails::job::Spec),
    TextEmbed(text_embeddings::job::Spec),
    ImageEmbed(image_embeddings::Spec),
    PruneMissing(prune_jobs::Spec),
    LibraryPurge(prune_jobs::LibraryPurgeSpec),
}

impl JobRequest {
    fn prepare(self) -> Result<JobSpec, ApiError> {
        match self {
            Self::ModelPrepare(_) => Ok(JobSpec::ModelPrepare),
            Self::OcrModelLoad(request) => {
                Ok(JobSpec::OcrModelLoad(ocr_models::job::prepare(request)?))
            }
            Self::LibraryScan(request) => Ok(JobSpec::LibraryScan(library_scan::prepare(request)?)),
            Self::ThumbnailGenerate(request) => Ok(JobSpec::ThumbnailGenerate(
                thumbnails::job::prepare(request)?,
            )),
            Self::TextEmbed(request) => {
                Ok(JobSpec::TextEmbed(text_embeddings::job::prepare(request)?))
            }
            Self::ImageEmbed(request) => {
                Ok(JobSpec::ImageEmbed(image_embeddings::prepare(request)?))
            }
            Self::PruneMissing(request) => Ok(JobSpec::PruneMissing(prune_jobs::prepare(request)?)),
            Self::LibraryPurge(request) => Ok(JobSpec::LibraryPurge(
                prune_jobs::prepare_library_purge(request)?,
            )),
        }
    }
}

impl JobSpec {
    /// The library the job works on, which must exist when the job is accepted.
    fn library_id(&self) -> Option<i64> {
        match self {
            Self::LibraryScan(spec) => Some(spec.library_id()),
            Self::ThumbnailGenerate(spec) => Some(spec.library_id()),
            Self::TextEmbed(spec) => spec.library_id(),
            Self::ImageEmbed(spec) => spec.library_id(),
            Self::PruneMissing(spec) => Some(spec.library_id()),
            Self::LibraryPurge(spec) => Some(spec.library_id()),
            Self::ModelPrepare | Self::OcrModelLoad(_) => None,
        }
    }

    fn kind(&self) -> JobKind {
        match self {
            Self::ModelPrepare => JobKind::ModelPrepare,
            Self::OcrModelLoad(_) => JobKind::OcrModelLoad,
            Self::LibraryScan(_) => JobKind::LibraryScan,
            Self::ThumbnailGenerate(_) => JobKind::ThumbnailGenerate,
            Self::TextEmbed(_) => JobKind::TextEmbed,
            Self::ImageEmbed(_) => JobKind::ImageEmbed,
            Self::PruneMissing(_) => JobKind::PruneMissing,
            Self::LibraryPurge(_) => JobKind::LibraryPurge,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum JobKind {
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
enum JobStatus {
    Queued,
    Running,
    Cancelling,
    Cancelled,
    Completed,
    Failed,
}

impl JobStatus {
    fn is_terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::Completed | Self::Failed)
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum JobPhase {
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
struct JobProgress {
    discovered: usize,
    total: Option<u64>,
    phase_completed: u64,
    processed: usize,
    cataloged: usize,
    thumbnails_generated: usize,
    thumbnail_failures: usize,
    prune_candidates: usize,
    embedded: usize,
    indexed: usize,
    skipped: usize,
    failed: usize,
    deleted: usize,
    downloaded_bytes: usize,
    download_total_bytes: usize,
    /// Current file only; absent while loading sessions or using cached files.
    #[serde(skip_serializing_if = "Option::is_none")]
    download: Option<ModelDownload>,
    models_loaded: usize,
    /// The rate of completed work since the current phase began. OCR excludes skipped assets
    /// so resuming an index does not count previously indexed images as new OCR work.
    items_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct ModelDownload {
    model_id: String,
    filename: String,
    downloaded_bytes: usize,
    total_bytes: usize,
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
pub(super) struct IndexStages {
    pub(super) ocr: bool,
    pub(super) image: bool,
    pub(super) text: bool,
}

/// Where one folder of a `libraryScan` stands.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) enum FolderState {
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
struct FolderProgress {
    path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    scan_mode: Option<&'static str>,
    state: FolderState,
    /// Files found by this folder's walk.
    discovered: usize,
    /// Files confirmed in the catalog by this walk, whether new, changed, or already current.
    cataloged: usize,
    failed: usize,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct JobData {
    index_stages: Option<IndexStages>,
    /// Per-folder progress of a `libraryScan`, in scan order.
    folders: Option<Vec<FolderProgress>>,
    /// The folder whose walk the current catalog events belong to.
    current_folder: Option<usize>,
    status: JobStatus,
    phase: JobPhase,
    progress: JobProgress,
    error: Option<String>,
    errors: Vec<JobItemError>,
    active_asset_paths: BTreeSet<PathBuf>,
    phase_started_at: Option<Instant>,
    phase_started_skipped: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct JobItemError {
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<PathBuf>,
    message: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(super) struct JobResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    index_stages: Option<IndexStages>,
    #[serde(skip_serializing_if = "Option::is_none")]
    folders: Option<Vec<FolderProgress>>,
    job_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    library_id: Option<i64>,
    #[serde(rename = "type")]
    kind: JobKind,
    status: JobStatus,
    phase: JobPhase,
    progress: JobProgress,
    /// All source assets currently being worked. This is a set because OCR and CLIP decode
    /// concurrently; serialized ordering is stable for inexpensive UI updates.
    active_asset_paths: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    errors: Vec<JobItemError>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobListResponse {
    active_job_id: Option<String>,
    jobs: Vec<JobResponse>,
}

pub(super) struct Job {
    id: u64,
    kind: JobKind,
    library_id: Option<i64>,
    scan_requests: Mutex<Vec<(PathBuf, i64)>>,
    cancel_requested: Arc<AtomicBool>,
    data: Mutex<JobData>,
    updates: watch::Sender<JobResponse>,
}

impl Job {
    #[cfg(test)]
    fn new(id: u64, kind: JobKind) -> Self {
        Self::new_for(id, kind, None)
    }

    fn new_for(id: u64, kind: JobKind, library_id: Option<i64>) -> Self {
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
            phase_started_skipped: 0,
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

    fn data(&self) -> MutexGuard<'_, JobData> {
        self.data.lock()
    }

    pub(super) fn response(&self) -> JobResponse {
        self.updates.borrow().clone()
    }

    fn response_from_data(
        id: u64,
        kind: JobKind,
        library_id: Option<i64>,
        data: &JobData,
    ) -> JobResponse {
        JobResponse {
            index_stages: data.index_stages,
            folders: data.folders.clone(),
            job_id: id.to_string(),
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

    fn publish(&self, data: &mut JobData) {
        refresh_phase_throughput(data);
        self.updates.send_replace(Self::response_from_data(
            self.id,
            self.kind,
            self.library_id,
            data,
        ));
    }

    fn subscribe(&self) -> watch::Receiver<JobResponse> {
        self.updates.subscribe()
    }

    // `begin`, `complete`, and `fail` skip `job_id`/`kind` on their own log lines: all three run
    // only from inside the `job` span `start()` opens below, which already carries both, and the
    // compact writer appends ambient span fields to every event inside it. `request_cancel` is
    // different — it runs on the HTTP handler's or the shutdown path's task, outside that span —
    // so it states its own identity.

    fn begin(&self) -> bool {
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

    fn request_cancel(&self) {
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

    pub(super) fn check_cancelled(&self) -> anyhow::Result<()> {
        cancel_if(self.cancel_requested.load(Ordering::Acquire))
    }

    fn complete(&self, cancelled: bool) {
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

    pub(super) fn ocr_download_progress(
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

    pub(super) fn downloading_models(&self) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::DownloadingModels);
        self.publish(&mut data);
    }

    pub(super) fn model_download_complete(&self) {
        let mut data = self.data();
        data.progress.processed += 1;
        self.publish(&mut data);
    }

    pub(super) fn set_index_stages(&self, stages: IndexStages) {
        let mut data = self.data();
        data.index_stages = Some(stages);
        self.publish(&mut data);
    }

    pub(super) fn set_folders(&self, paths: Vec<PathBuf>) {
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

    pub(super) fn set_folder_scan_mode(&self, index: usize, mode: &'static str) {
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

    pub(super) fn set_scan_requests(&self, requests: Vec<(PathBuf, i64)>) {
        *self.scan_requests.lock() = requests;
    }

    /// Attribute the following catalog events to folder `index` and mark it scanning.
    pub(super) fn enter_folder(&self, index: usize) {
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

    pub(super) fn leave_folder(&self) {
        self.data().current_folder = None;
    }

    pub(super) fn finish_folder(&self, index: usize, state: FolderState, error: Option<String>) {
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

    pub(super) fn preparing_models(&self, count: u64) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::LoadingModels);
        data.progress.total = Some(count);
        self.publish(&mut data);
    }

    pub(super) fn loading_models(&self) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::LoadingModels);
        data.progress.total = Some(2);
        self.publish(&mut data);
    }

    pub(super) fn models_loaded(&self, count: usize) {
        let mut data = self.data();
        data.progress.models_loaded += count;
        data.progress.phase_completed += count as u64;
        self.publish(&mut data);
    }

    fn fail(&self, message: impl Into<String>) {
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
                }
            }
            IndexEvent::Progress(delta) => {
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

fn current_folder(data: &mut JobData) -> Option<&mut FolderProgress> {
    let index = data.current_folder?;
    data.folders.as_mut()?.get_mut(index)
}

fn apply_delta(progress: &mut JobProgress, delta: IndexProgressDelta) {
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

fn set_phase(data: &mut JobData, next: JobPhase) {
    if data.phase != next {
        data.phase = next;
        data.progress.discovered = 0;
        data.progress.total = None;
        data.progress.phase_completed = 0;
        data.progress.items_per_second = None;
        data.phase_started_at = Some(Instant::now());
        data.phase_started_skipped = data.progress.skipped;
        data.active_asset_paths.clear();
    }
}

fn refresh_phase_throughput(data: &mut JobData) {
    if matches!(data.phase, JobPhase::Queued | JobPhase::Finished) {
        return;
    }
    let completed = if data.phase == JobPhase::Ocr {
        data.progress.phase_completed.saturating_sub(
            data.progress
                .skipped
                .saturating_sub(data.phase_started_skipped) as u64,
        )
    } else {
        data.progress.phase_completed
    };
    if completed == 0 {
        data.progress.items_per_second = None;
        return;
    }
    let Some(started_at) = data.phase_started_at else {
        return;
    };
    let elapsed = started_at.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        data.progress.items_per_second = Some(completed as f64 / elapsed);
    }
}

#[derive(Default)]
struct JobRegistry {
    active: Option<u64>,
    jobs: BTreeMap<u64, Arc<Job>>,
    queued: VecDeque<(u64, JobSpec)>,
}

pub(crate) struct JobManager {
    next_id: AtomicU64,
    shutting_down: AtomicBool,
    registry: Mutex<JobRegistry>,
    databases: Arc<Databases>,
    thumbnails: Arc<ThumbnailService>,
    embedder: Arc<TextEmbedder>,
    image_embedder: Arc<ImageEmbedder>,
    image_query_embedder: Arc<ImageQueryModel>,
    ocr_models: Arc<ocr_models::ModelStore>,
    runtime: Arc<super::RuntimeSettings>,
}

impl JobManager {
    pub(crate) fn new(
        databases: Arc<Databases>,
        thumbnails: Arc<ThumbnailService>,
        embedder: Arc<TextEmbedder>,
        image_embedder: Arc<ImageEmbedder>,
        image_query_embedder: Arc<ImageQueryModel>,
        ocr_models: Arc<ocr_models::ModelStore>,
        runtime: Arc<super::RuntimeSettings>,
    ) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            shutting_down: AtomicBool::new(false),
            registry: Mutex::new(JobRegistry::default()),
            databases,
            thumbnails,
            embedder,
            image_embedder,
            image_query_embedder,
            ocr_models,
            runtime,
        }
    }

    fn registry(&self) -> MutexGuard<'_, JobRegistry> {
        self.registry.lock()
    }

    pub(super) fn start(self: &Arc<Self>, spec: JobSpec) -> Result<Arc<Job>, ApiError> {
        let mut spec = Some(spec);
        if let Some(JobSpec::LibraryScan(scan)) = spec.as_mut() {
            scan.apply_settings(&self.runtime)
                .map_err(ApiError::internal)?;
        }
        let mut replaced = Vec::new();
        let (job, run_now) = {
            let mut registry = self.registry();
            if self.shutting_down.load(Ordering::Acquire) {
                return Err(ApiError::shutting_down());
            }
            if let Some(active_id) = registry.active {
                let active = registry.jobs.get(&active_id);
                if active.is_some() && !matches!(spec, Some(JobSpec::LibraryScan(_))) {
                    return Err(ApiError::job_busy());
                }
            }
            // At most one scan waits for the worker, and the newest request wins: a request for
            // the queued library merges into it, and one for another library replaces it.
            if let Some(JobSpec::LibraryScan(ref incoming)) = spec
                && let Some((id, JobSpec::LibraryScan(queued))) = registry.queued.front_mut()
                && queued.library_id() == incoming.library_id()
            {
                queued.merge(incoming);
                let id = *id;
                return Ok(Arc::clone(
                    registry.jobs.get(&id).expect("queued job exists"),
                ));
            }
            if registry.active.is_some() {
                replaced.extend(registry.queued.drain(..).map(|(id, _)| id));
            }
            evict_retained_jobs(&mut registry);
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let job = Arc::new(Job::new_for(
                id,
                spec.as_ref().unwrap().kind(),
                spec.as_ref().unwrap().library_id(),
            ));
            let run_now = registry.active.is_none();
            if run_now {
                registry.active = Some(id);
            } else {
                registry.queued.push_back((id, spec.take().unwrap()));
            }
            registry.jobs.insert(id, Arc::clone(&job));
            (job, run_now)
        };
        // Outside the registry lock: cancelling records the replaced scan's folders as stopped.
        for id in replaced {
            self.cancel(id);
        }
        if run_now {
            self.spawn_job(spec.take().unwrap(), Arc::clone(&job));
        }
        Ok(job)
    }

    fn spawn_job(self: &Arc<Self>, spec: JobSpec, job: Arc<Job>) {
        let manager = Arc::clone(self);
        let worker_job = Arc::clone(&job);
        let job_span = info_span!("job", job_id = job.id, kind = ?job.kind);
        tokio::spawn(
            async move {
                match manager.run_job(spec, Arc::clone(&worker_job)).await {
                    Ok(()) => worker_job.complete(false),
                    Err(error) if is_cancelled(&error) => {
                        manager.record_stopped_scan(
                            &worker_job,
                            ScanOutcome::Cancelled,
                            "cancelled",
                        );
                        worker_job.complete(true);
                    }
                    Err(error) => {
                        let message = format!("{error:#}");
                        manager.record_stopped_scan(&worker_job, ScanOutcome::Failed, &message);
                        worker_job.fail(message);
                    }
                }
                manager.finish(worker_job.id);
            }
            .instrument(job_span),
        );
    }

    async fn run_job(self: &Arc<Self>, spec: JobSpec, job: Arc<Job>) -> anyhow::Result<()> {
        if !job.begin() {
            return Err(JobCancelled.into());
        }
        let spec = match spec {
            JobSpec::OcrModelLoad(spec) => {
                return ocr_models::job::run(spec, &self.ocr_models, &self.runtime, job).await;
            }
            spec => spec,
        };

        let databases = Arc::clone(&self.databases);
        let thumbnails = Arc::clone(&self.thumbnails);
        let embedder = Arc::clone(&self.embedder);
        let image_embedder = Arc::clone(&self.image_embedder);
        let manager = Arc::clone(self);
        let handle = tokio::runtime::Handle::current();
        let span = Span::current();
        tokio::task::spawn_blocking(move || {
            let _entered = span.enter();
            let result = (|| match spec {
                JobSpec::ModelPrepare => manager.prepare_job_models(&job),
                JobSpec::LibraryScan(spec) => library_scan::run(
                    spec,
                    &library_scan::Services {
                        databases: &databases,
                        thumbnails: &thumbnails,
                        text_embedder: &embedder,
                        image_embedder: &image_embedder,
                        ocr_store: &manager.ocr_models,
                        runtime: &manager.runtime,
                        handle,
                    },
                    &job,
                ),
                JobSpec::ThumbnailGenerate(spec) => {
                    thumbnails::job::run(spec, &databases.assets, &thumbnails, job.as_ref())
                }
                JobSpec::TextEmbed(spec) => {
                    let spec = spec.resolve(&databases)?;
                    if !text_embeddings::job::has_pending(
                        &spec,
                        &databases.ocr,
                        embedder.model().id(),
                        embedder.dimensions(),
                    )? {
                        return Ok(());
                    }
                    job.preparing_models(1);
                    let model = embedder.prepare_with_progress(job.as_ref())?;
                    job.models_loaded(1);
                    text_embeddings::job::run(spec, &databases.ocr, model.as_ref(), job.as_ref())
                }
                JobSpec::ImageEmbed(spec) => {
                    let spec = spec.resolve(&databases)?;
                    if !image_embeddings::has_pending(
                        &spec,
                        &databases,
                        image_embedder.dimensions(),
                        None,
                    )? {
                        return Ok(());
                    }
                    job.preparing_models(1);
                    let model = image_embedder.prepare_with_progress(job.as_ref())?;
                    job.models_loaded(1);
                    image_embeddings::run(
                        spec,
                        &databases,
                        &thumbnails,
                        model.as_ref(),
                        job.as_ref(),
                        None,
                    )
                }
                JobSpec::PruneMissing(spec) => prune_jobs::run(
                    spec,
                    &databases.assets,
                    &databases.ocr,
                    &databases.images,
                    image_embedder.dimensions(),
                    &thumbnails,
                    job.as_ref(),
                ),
                JobSpec::LibraryPurge(spec) => prune_jobs::run_library_purge(
                    spec,
                    &databases.assets,
                    &databases.ocr,
                    &databases.images,
                    image_embedder.dimensions(),
                    &thumbnails,
                    job.as_ref(),
                ),
                JobSpec::OcrModelLoad(_) => {
                    unreachable!("handled before the blocking job boundary")
                }
            })();
            manager.maintain_databases();
            result
        })
        .await
        .map_err(|error| anyhow::anyhow!("job worker failed: {error}"))?
    }

    /// Jobs are already globally serialized, making their common epilogue the safe place to
    /// reconcile independently stored derived rows and perform bounded SQLite maintenance.
    #[tracing::instrument(level = "debug", skip(self))]
    fn maintain_databases(&self) {
        let result = (|| -> anyhow::Result<(usize, usize, usize)> {
            let assets = nicegal_core::assets::AssetCatalog::new(&self.databases.assets)?;
            let mut ocr = nicegal_core::db::DB::new(&self.databases.ocr)?;
            let mut images = nicegal_core::image_index::ImageIndexDb::new(
                &self.databases.images,
                self.image_embedder.dimensions(),
            )?;
            let ocr_deleted = ocr.prune_orphans(&self.databases.assets)?;
            let image_deleted = images.prune_orphans(&self.databases.assets)?;
            let thumbnail_deleted = self
                .thumbnails
                .prune_orphans_and_maintain(self.databases.assets.clone())?;
            ocr.maintain()?;
            images.maintain()?;
            assets.maintain()?;
            Ok((ocr_deleted, image_deleted, thumbnail_deleted))
        })();
        match result {
            Ok((ocr, images, thumbnails)) => tracing::debug!(
                orphaned_ocr = ocr,
                orphaned_images = images,
                orphaned_thumbnails = thumbnails,
                "job database maintenance completed"
            ),
            Err(error) => tracing::warn!(
                error = %format_args!("{error:#}"),
                "job database maintenance failed"
            ),
        }
    }

    fn prepare_job_models(&self, job: &Job) -> anyhow::Result<()> {
        let needs_image_text = self.image_embedder.model().supports_text_queries();
        job.preparing_models(2 + u64::from(needs_image_text));
        self.embedder.prepare_with_progress(job)?;
        job.models_loaded(1);
        job.check_cancelled()?;
        self.image_embedder.prepare_with_progress(job)?;
        job.models_loaded(1);
        job.check_cancelled()?;
        if needs_image_text {
            self.image_query_embedder.prepare_with_progress(job)?;
            job.models_loaded(1);
        }
        job.check_cancelled()
    }

    fn get(&self, id: u64) -> Option<Arc<Job>> {
        self.registry().jobs.get(&id).cloned()
    }

    fn record_stopped_scan(&self, job: &Job, outcome: ScanOutcome, message: &str) {
        let Some(library_id) = job.library_id.filter(|_| job.kind == JobKind::LibraryScan) else {
            return;
        };
        let result = (|| -> anyhow::Result<()> {
            let catalog = nicegal_core::assets::AssetCatalog::new(&self.databases.assets)?;
            let requests = job.scan_requests.lock().clone();
            if requests.is_empty() {
                // A queued scan was cancelled before it could read the definition.
                if let Some(library) = catalog.library(library_id)? {
                    for folder in library.include.iter().filter(|folder| folder.scan_pending) {
                        catalog.fail_folder_scan_request(
                            library_id,
                            &folder.path,
                            folder.scan_request,
                            outcome,
                            message,
                        )?;
                    }
                }
            } else {
                for (path, request) in requests {
                    catalog
                        .fail_folder_scan_request(library_id, &path, request, outcome, message)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            tracing::warn!(?error, library_id, "could not record stopped scan");
        }
    }

    fn cancel(&self, id: u64) -> Option<Arc<Job>> {
        let job = self.get(id)?;
        let queued = job.response().status == JobStatus::Queued;
        job.request_cancel();
        if queued {
            self.record_stopped_scan(&job, ScanOutcome::Cancelled, "cancelled");
        }
        Some(job)
    }

    fn list(&self) -> JobListResponse {
        let registry = self.registry();
        JobListResponse {
            active_job_id: registry.active.and_then(|id| {
                registry
                    .jobs
                    .get(&id)
                    .filter(|job| !job.response().status.is_terminal())
                    .map(|_| id.to_string())
            }),
            jobs: registry
                .jobs
                .values()
                .rev()
                .map(|job| job.response())
                .collect(),
        }
    }

    pub(super) fn has_active_job(&self) -> bool {
        let registry = self.registry();
        registry
            .active
            .and_then(|id| registry.jobs.get(&id))
            .is_some_and(|job| !job.data().status.is_terminal())
    }

    pub(crate) fn cancel_all(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let jobs: Vec<_> = self.registry().jobs.values().cloned().collect();
        for job in jobs {
            job.request_cancel();
        }
    }

    fn finish(self: &Arc<Self>, id: u64) {
        let mut registry = self.registry();
        if registry.active == Some(id) {
            registry.active = None;
        }
        while let Some((next_id, spec)) = registry.queued.pop_front() {
            let job = Arc::clone(registry.jobs.get(&next_id).expect("queued job exists"));
            if job.response().status.is_terminal() {
                continue;
            }
            registry.active = Some(next_id);
            drop(registry);
            self.spawn_job(spec, job);
            return;
        }
    }
}

/// Drop the oldest finished jobs so the retained history stays bounded.
fn evict_retained_jobs(registry: &mut JobRegistry) {
    while registry.jobs.len() >= MAX_RETAINED_JOBS {
        let removable = registry
            .jobs
            .iter()
            .find(|(id, job)| {
                Some(**id) != registry.active
                    && !registry.queued.iter().any(|(queued, _)| queued == *id)
                    && job.response().status.is_terminal()
            })
            .map(|(id, _)| *id);
        let Some(id) = removable else {
            break;
        };
        registry.jobs.remove(&id);
    }
}

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/jobs", post(create_job).get(list_jobs))
        .route("/v1/jobs/{job_id}/events", get(job_events))
        .route("/v1/jobs/{job_id}", get(get_job).delete(cancel_job))
}

struct JobEventStream {
    inner: WatchStream<JobResponse>,
    finished: bool,
}

impl Stream for JobEventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.inner).poll_next(context) {
            Poll::Ready(Some(response)) => {
                self.finished = response.status.is_terminal();
                let data = serde_json::to_string(&response)
                    .expect("serializing a job response should not fail");
                Poll::Ready(Some(Ok(Event::default().event("snapshot").data(data))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

async fn job_events(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let job = state
        .jobs
        .get(parse_job_id(&job_id)?)
        .ok_or_else(ApiError::job_not_found)?;
    let stream = JobEventStream {
        inner: WatchStream::new(job.subscribe()),
        finished: false,
    };
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

async fn list_jobs(State(state): State<AppState>) -> Json<JobListResponse> {
    Json(state.jobs.list())
}

async fn create_job(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<JobRequest>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    let job = start_job(&state, request.prepare()?).await?;
    Ok((StatusCode::ACCEPTED, Json(job.response())))
}

/// Start a job after checking that the library it names exists, so an unknown library is a
/// `404` rather than a job that fails.
pub(super) async fn start_job(state: &AppState, spec: JobSpec) -> Result<Arc<Job>, ApiError> {
    if let Some(library_id) = spec.library_id() {
        let databases = Arc::clone(&state.databases);
        super::run_blocking(move || {
            match databases.open_assets_read_only()?.library(library_id)? {
                Some(_) => Ok(()),
                None => Err(ApiError::library_not_found(library_id)),
            }
        })
        .await?;
    }
    state.jobs.start(spec)
}

async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobResponse>, ApiError> {
    let job = state
        .jobs
        .get(parse_job_id(&job_id)?)
        .ok_or_else(ApiError::job_not_found)?;
    Ok(Json(job.response()))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    let job = state
        .jobs
        .cancel(parse_job_id(&job_id)?)
        .ok_or_else(ApiError::job_not_found)?;
    let response = job.response();
    let status = if response.status.is_terminal() {
        StatusCode::OK
    } else {
        StatusCode::ACCEPTED
    };
    Ok((status, Json(response)))
}

fn parse_job_id(value: &str) -> Result<u64, ApiError> {
    let id = value
        .parse::<u64>()
        .map_err(|_| ApiError::bad_request("job identifier must be a positive integer"))?;
    if id == 0 {
        return Err(ApiError::bad_request(
            "job identifier must be a positive integer",
        ));
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt as _;

    #[test]
    fn library_job_snapshots_identify_their_library() {
        let job = Job::new_for(42, JobKind::LibraryScan, Some(7));
        let snapshot = serde_json::to_value(job.response()).unwrap();
        assert_eq!(snapshot["libraryId"], 7);
        assert_eq!(snapshot["status"], "queued");
        assert!(
            serde_json::to_value(Job::new(43, JobKind::ModelPrepare).response())
                .unwrap()
                .get("libraryId")
                .is_none()
        );
    }

    #[test]
    fn model_preparation_wire_contract_and_progress() {
        let request: JobRequest = serde_json::from_value(serde_json::json!({
            "type": "modelPrepare", "params": {}
        }))
        .unwrap();
        let spec = request.prepare().unwrap();
        assert!(matches!(spec, JobSpec::ModelPrepare));
        let job = Job::new(1, spec.kind());
        assert!(job.begin());
        job.preparing_models(3);
        job.models_loaded(1);
        let response = serde_json::to_value(job.response()).unwrap();
        assert_eq!(response["type"], "modelPrepare");
        assert_eq!(response["phase"], "loadingModels");
        assert_eq!(response["progress"]["total"], 3);
        assert_eq!(response["progress"]["phaseCompleted"], 1);
        assert_eq!(response["progress"]["modelsLoaded"], 1);
        assert!(
            serde_json::from_value::<JobRequest>(serde_json::json!({
                "type": "modelPrepare", "params": {"root": "unexpected"}
            }))
            .is_err()
        );
    }

    #[test]
    fn job_observer_tracks_progress_and_cancellation() {
        let job = Job::new(7, JobKind::OcrModelLoad);
        assert!(job.begin());
        job.on_event(IndexEvent::PhaseChanged(IndexPhase::Scanning));
        job.on_event(IndexEvent::Discovered { count: 12 });
        job.on_event(IndexEvent::DiscoveryComplete { total: 12 });
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            processed: 4,
            cataloged: 4,
            indexed: 2,
            skipped: 1,
            failed: 1,
            deleted: 0,
            ..IndexProgressDelta::default()
        }));
        job.on_event(IndexEvent::Error {
            path: Some(PathBuf::from("C:/gallery/broken.png")),
            message: "OCR failed: invalid image".to_owned(),
        });
        job.request_cancel();

        let response = job.response();
        assert_eq!(response.job_id, "7");
        assert_eq!(response.kind, JobKind::OcrModelLoad);
        assert_eq!(response.status, JobStatus::Cancelling);
        assert_eq!(response.phase, JobPhase::Scanning);
        assert_eq!(response.progress.discovered, 12);
        assert_eq!(response.progress.total, Some(12));
        assert_eq!(response.progress.processed, 4);
        assert_eq!(response.errors.len(), 1);
        assert_eq!(
            response.errors[0].path.as_deref(),
            Some(camino::Utf8Path::new("C:/gallery/broken.png"))
        );
        assert_eq!(response.errors[0].message, "OCR failed: invalid image");
        assert!(job.is_cancelled());
    }

    #[test]
    fn file_downloads_preserve_model_steps_and_retry_byte_totals() {
        let job = Job::new(88, JobKind::ModelPrepare);
        job.begin();
        job.preparing_models(3);
        job.models_loaded(1);
        let source = ModelSource::local(std::path::Path::new("image.onnx")).unwrap();
        job.progress(&source, 25, 100);
        let response = serde_json::to_value(job.response()).unwrap();
        assert_eq!(response["phase"], "loadingModels");
        assert_eq!(response["progress"]["phaseCompleted"], 1);
        assert_eq!(response["progress"]["total"], 3);
        assert_eq!(response["progress"]["download"]["filename"], "image.onnx");
        assert_eq!(response["progress"]["download"]["downloadedBytes"], 25);
        DownloadObserver::finish(&job);
        assert!(job.response().progress.download.is_none());
        assert_eq!(job.response().progress.phase_completed, 1);

        let ocr = Job::new(89, JobKind::OcrModelLoad);
        ocr.ocr_download_progress(&source, 50, 100, 0, 0);
        // Retrying this file replaces its contribution instead of double-counting it.
        ocr.ocr_download_progress(&source, 0, 100, 50, 100);
        ocr.ocr_download_progress(&source, 100, 100, 0, 100);
        // A second file adds to the legacy job totals but has its own current-file counters.
        ocr.ocr_download_progress(&source, 20, 40, 0, 0);
        let progress = ocr.response().progress;
        assert_eq!(progress.downloaded_bytes, 120);
        assert_eq!(progress.download_total_bytes, 140);
        assert_eq!(progress.download.unwrap().downloaded_bytes, 20);
        ocr.fail("network error");
        assert!(ocr.response().progress.download.is_none());
    }

    #[test]
    fn phase_progress_is_scoped_to_the_current_phase() {
        let job = Job::new(8, JobKind::LibraryScan);
        assert!(job.begin());
        job.on_event(IndexEvent::PhaseChanged(IndexPhase::Scanning));
        job.on_event(IndexEvent::Discovered { count: 5 });
        job.on_event(IndexEvent::DiscoveryComplete { total: 5 });
        assert_eq!(job.response().progress.total, Some(5));
        assert_eq!(job.response().progress.phase_completed, 5);

        job.on_event(IndexEvent::PhaseChanged(IndexPhase::Cataloging));
        let after_change = job.response();
        assert_eq!(after_change.progress.discovered, 0);
        assert_eq!(after_change.progress.total, None);
        assert_eq!(after_change.progress.phase_completed, 0);

        job.on_event(IndexEvent::DiscoveryComplete { total: 5 });
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: 2,
            cataloged: 2,
            ..IndexProgressDelta::default()
        }));
        let response = job.response();
        assert_eq!(response.progress.total, Some(5));
        assert_eq!(response.progress.phase_completed, 2);
        assert_eq!(response.progress.cataloged, 2);
    }

    #[test]
    fn throughput_resets_when_indexing_phase_changes() {
        let job = Job::new(8, JobKind::LibraryScan);
        assert!(job.begin());

        for phase in [
            IndexPhase::Cataloging,
            IndexPhase::Thumbnails,
            IndexPhase::Ocr,
            IndexPhase::Cleanup,
            IndexPhase::ImageEmbedding,
            IndexPhase::TextEmbedding,
        ] {
            job.on_event(IndexEvent::PhaseChanged(phase));
            let response = job.response();
            assert_eq!(response.progress.items_per_second, None);
            assert_eq!(response.progress.phase_completed, 0);
            assert!(job.data().phase_started_at.unwrap().elapsed() < Duration::from_secs(1));

            // A controlled elapsed time makes each phase's expected rate independent of
            // machine speed and proves that earlier phases' work is not counted again.
            job.data().phase_started_at = Some(Instant::now() - Duration::from_secs(10));
            job.on_event(IndexEvent::Progress(IndexProgressDelta {
                phase_completed: 20,
                ..IndexProgressDelta::default()
            }));
            let rate = job.response().progress.items_per_second.unwrap();
            assert!((1.9..=2.0).contains(&rate), "unexpected rate: {rate}");
        }
    }

    #[test]
    fn resumed_ocr_throughput_excludes_skips_but_preserves_progress() {
        let job = Job::new(10, JobKind::LibraryScan);
        assert!(job.begin());
        job.on_event(IndexEvent::PhaseChanged(IndexPhase::Cataloging));
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: 100,
            skipped: 100,
            ..IndexProgressDelta::default()
        }));
        job.on_event(IndexEvent::PhaseChanged(IndexPhase::Ocr));
        job.on_event(IndexEvent::DiscoveryComplete { total: 10_020 });
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: 10_000,
            processed: 10_000,
            skipped: 10_000,
            ..IndexProgressDelta::default()
        }));
        assert_eq!(job.response().progress.items_per_second, None);

        job.data().phase_started_at = Some(Instant::now() - Duration::from_secs(10));
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: 20,
            processed: 20,
            ..IndexProgressDelta::default()
        }));
        let progress = job.response().progress;
        assert_eq!(progress.phase_completed, 10_020);
        assert_eq!(progress.total, Some(10_020));
        assert_eq!(progress.skipped, 10_100);
        assert!((1.9..=2.0).contains(&progress.items_per_second.unwrap()));

        // Saving the completed OCR batch must not count the same work twice.
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            indexed: 20,
            ..IndexProgressDelta::default()
        }));
        assert!((1.9..=2.0).contains(&job.response().progress.items_per_second.unwrap()));
        job.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
        assert_eq!(job.response().progress.items_per_second, None);
    }

    #[test]
    fn job_identifiers_must_be_positive_integers() {
        assert_eq!(parse_job_id("42").unwrap(), 42);
        for invalid in ["0", "-1", "not-a-job"] {
            assert!(parse_job_id(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn typed_job_envelope_rejects_unknown_fields() {
        let valid = serde_json::json!({
            "type": "ocrModelLoad",
            "params": {
                "detection": {
                    "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx"
                },
                "recognition": {
                    "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx"
                }
            }
        });
        assert!(serde_json::from_value::<JobRequest>(valid).is_ok());

        let library_scan = serde_json::json!({
            "type": "libraryScan",
            "params": { "libraryId": 1, "pendingOnly": true }
        });
        assert!(serde_json::from_value::<JobRequest>(library_scan).is_ok());
        for retired in ["libraryIndex", "catalogSync"] {
            let request =
                serde_json::json!({ "type": retired, "params": { "root": "C:/gallery" } });
            assert!(
                serde_json::from_value::<JobRequest>(request).is_err(),
                "{retired}"
            );
        }

        let thumbnail_generate = serde_json::json!({
            "type": "thumbnailGenerate",
            "params": {
                "libraryId": 1
            }
        });
        assert!(serde_json::from_value::<JobRequest>(thumbnail_generate).is_ok());

        let library_purge = serde_json::json!({
            "type": "libraryPurge",
            "params": {
                "libraryId": 1
            }
        });
        assert!(serde_json::from_value::<JobRequest>(library_purge).is_ok());

        let unknown = serde_json::json!({
            "type": "ocrModelLoad",
            "params": {
                "detection": {
                    "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx"
                },
                "recognition": {
                    "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx"
                }
            },
            "unexpected": true
        });
        assert!(serde_json::from_value::<JobRequest>(unknown).is_err());
    }

    #[test]
    fn folder_progress_attributes_catalog_events_to_the_current_folder() {
        let job = Job::new(11, JobKind::LibraryScan);
        assert!(job.begin());
        job.set_folders(vec!["/a".into(), "/b".into()]);
        job.enter_folder(0);
        job.on_event(IndexEvent::PhaseChanged(IndexPhase::Scanning));
        job.on_event(IndexEvent::Discovered { count: 3 });
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            cataloged: 2,
            failed: 1,
            ..IndexProgressDelta::default()
        }));
        job.finish_folder(0, FolderState::Scanned, None);
        job.enter_folder(1);
        job.on_event(IndexEvent::Discovered { count: 5 });
        job.finish_folder(1, FolderState::Unavailable, Some("offline".to_owned()));
        job.leave_folder();
        // Later phases are library-wide and leave folder counts alone.
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            cataloged: 9,
            ..IndexProgressDelta::default()
        }));

        let folders = serde_json::to_value(job.response()).unwrap()["folders"].clone();
        assert_eq!(
            folders,
            serde_json::json!([
                {"path": "/a", "state": "scanned", "discovered": 3, "cataloged": 2, "failed": 1, "error": null},
                {"path": "/b", "state": "unavailable", "discovered": 5, "cataloged": 0, "failed": 0, "error": "offline"}
            ])
        );
    }

    #[test]
    fn library_purge_job_kind_serializes_as_the_typed_request_name() {
        assert_eq!(
            serde_json::to_value(JobKind::LibraryPurge).unwrap(),
            serde_json::json!("libraryPurge")
        );
    }

    #[tokio::test]
    async fn event_stream_emits_initial_and_terminal_snapshots_then_closes() {
        let job = Job::new(9, JobKind::ThumbnailGenerate);
        let mut stream = JobEventStream {
            inner: WatchStream::new(job.subscribe()),
            finished: false,
        };
        assert!(stream.next().await.is_some());

        assert!(job.begin());
        assert!(stream.next().await.is_some());
        job.complete(false);
        assert!(stream.next().await.is_some());
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn manager_rejects_concurrent_jobs_and_cancels_queued_work() {
        crate::api::tests::initialize_test_runtime();
        let runtime_config_dir = tempfile::TempDir::new().unwrap();
        let current_dir = PathBuf::try_from(runtime_config_dir.path().to_path_buf()).unwrap();
        let request: JobRequest = serde_json::from_value(serde_json::json!({
            "type": "ocrModelLoad",
            "params": {
                "detection": { "modelId": "owner/detection" },
                "recognition": { "modelId": "owner/recognition" }
            }
        }))
        .unwrap();
        let thumbnail_path = current_dir.join("unused-thumbnails.db");
        let runtime = Arc::new(
            crate::api::RuntimeSettings::load(
                PathBuf::try_from(runtime_config_dir.path().join("runtime.json")).unwrap(),
                None,
            )
            .unwrap(),
        );
        let manager = Arc::new(JobManager::new(
            Arc::new(Databases {
                assets: current_dir.join("unused-assets.db"),
                images: current_dir.join("unused-images.db"),
                ocr: current_dir.join("unused-ocr.db"),
                thumbnails: thumbnail_path.clone(),
            }),
            Arc::new(ThumbnailService::new(&thumbnail_path).unwrap()),
            Arc::new(TextEmbedder::deferred(
                nicegal_core::embedding::TextEmbedderOptions::default(),
            )),
            Arc::new(ImageEmbedder::deferred(
                nicegal_core::embedding::ImageEmbedderOptions::default(),
            )),
            Arc::new(ImageQueryModel::deferred(
                nicegal_core::embedding::ImageQueryEmbedderOptions::default(),
            )),
            Arc::new(ocr_models::ModelStore::new(
                nicegal_core::runtime::ExecutionProvider::Cpu,
            )),
            runtime,
        ));
        let job = manager.start(request.prepare().unwrap()).unwrap();

        let second: JobRequest = serde_json::from_value(serde_json::json!({
            "type": "ocrModelLoad",
            "params": {
                "detection": { "modelId": "owner/detection" },
                "recognition": { "modelId": "owner/recognition" }
            }
        }))
        .unwrap();
        assert!(manager.start(second.prepare().unwrap()).is_err());

        let scan = |params: serde_json::Value| {
            let request: JobRequest = serde_json::from_value(
                serde_json::json!({ "type": "libraryScan", "params": params }),
            )
            .unwrap();
            manager.start(request.prepare().unwrap()).unwrap()
        };
        let queued = scan(serde_json::json!({ "libraryId": 7, "pendingOnly": true }));
        assert_eq!(queued.response().status, JobStatus::Queued);
        assert_eq!(queued.response().library_id, Some(7));
        let merged = scan(serde_json::json!({ "libraryId": 7, "retryFailed": true }));
        assert_eq!(
            merged.id, queued.id,
            "a request for the queued library merges"
        );
        assert_eq!(manager.registry().queued.len(), 1);
        let replacement = scan(serde_json::json!({ "libraryId": 8 }));
        assert_ne!(replacement.id, queued.id);
        assert_eq!(
            queued.response().status,
            JobStatus::Cancelled,
            "another library replaces it"
        );
        assert_eq!(replacement.response().status, JobStatus::Queued);
        assert_eq!(manager.registry().queued.len(), 1);

        let mut updates = job.subscribe();
        manager.cancel_all();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !updates.borrow().status.is_terminal() {
                updates.changed().await.unwrap();
            }
        })
        .await
        .expect("cancelled job should reach a terminal state");
        assert_eq!(job.response().status, JobStatus::Cancelled);
    }
}
