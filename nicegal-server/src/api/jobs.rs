use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
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
use nicegal_core::index::{IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta};
use nicegal_core::thumbs::ThumbnailService;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio_stream::Stream;
use tokio_stream::wrappers::WatchStream;
use tracing::{Instrument, Span, error, info, info_span};

use super::error::ApiError;
use super::extract::ApiJson;
use super::models::{ImageModel as ImageEmbedder, ImageQueryModel, TextModel as TextEmbedder};
use super::{
    AppState, Databases, image_embeddings, indexing, ocr_models, prune_jobs, text_embeddings,
    thumbnails,
};

const MAX_RETAINED_JOBS: usize = 32;

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
    OcrIndex(indexing::Request),
    CatalogSync(indexing::CatalogSyncRequest),
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
    OcrIndex(indexing::Spec),
    CatalogSync(indexing::CatalogSyncSpec),
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
            Self::OcrIndex(request) => Ok(JobSpec::OcrIndex(indexing::prepare(request)?)),
            Self::CatalogSync(request) => Ok(JobSpec::CatalogSync(indexing::prepare_catalog_sync(
                request,
            )?)),
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
    fn kind(&self) -> JobKind {
        match self {
            Self::ModelPrepare => JobKind::ModelPrepare,
            Self::OcrModelLoad(_) => JobKind::OcrModelLoad,
            Self::OcrIndex(_) => JobKind::OcrIndex,
            Self::CatalogSync(_) => JobKind::CatalogSync,
            Self::ThumbnailGenerate(_) => JobKind::ThumbnailGenerate,
            Self::TextEmbed(_) => JobKind::TextEmbed,
            Self::ImageEmbed(_) => JobKind::ImageEmbed,
            Self::PruneMissing(_) => JobKind::PruneMissing,
            Self::LibraryPurge(_) => JobKind::LibraryPurge,
        }
    }

    fn requires_loaded_ocr_models(&self) -> bool {
        matches!(self, Self::OcrIndex(_))
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum JobKind {
    OcrModelLoad,
    ModelPrepare,
    OcrIndex,
    CatalogSync,
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
    models_loaded: usize,
    /// The rate of `phaseCompleted` work since the current phase began. It deliberately does not
    /// combine unlike units from different phases.
    items_per_second: Option<f64>,
}

#[derive(Debug, Clone)]
struct JobData {
    status: JobStatus,
    phase: JobPhase,
    progress: JobProgress,
    error: Option<String>,
    errors: Vec<JobItemError>,
    active_asset_paths: BTreeSet<PathBuf>,
    phase_started_at: Option<Instant>,
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
    job_id: String,
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
    cancel_requested: AtomicBool,
    data: Mutex<JobData>,
    updates: watch::Sender<JobResponse>,
}

impl Job {
    fn new(id: u64, kind: JobKind) -> Self {
        let data = JobData {
            status: JobStatus::Queued,
            phase: JobPhase::Queued,
            progress: JobProgress::default(),
            error: None,
            errors: Vec::new(),
            active_asset_paths: BTreeSet::new(),
            phase_started_at: None,
        };
        let (updates, _) = watch::channel(Self::response_from_data(id, kind, &data));
        Self {
            id,
            kind,
            cancel_requested: AtomicBool::new(false),
            data: Mutex::new(data),
            updates,
        }
    }

    fn data(&self) -> MutexGuard<'_, JobData> {
        self.data.lock().unwrap_or_else(|error| error.into_inner())
    }

    pub(super) fn response(&self) -> JobResponse {
        self.updates.borrow().clone()
    }

    fn response_from_data(id: u64, kind: JobKind, data: &JobData) -> JobResponse {
        JobResponse {
            job_id: id.to_string(),
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
        self.updates
            .send_replace(Self::response_from_data(self.id, self.kind, data));
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
            data.status = JobStatus::Cancelling;
            self.publish(&mut data);
            info!(job_id = self.id, kind = ?self.kind, "job cancellation requested");
        }
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
        self.publish(&mut data);
        info!(progress = ?data.progress, cancelled, "job finished");
    }

    pub(super) fn downloading_model(&self, total_bytes: usize) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::DownloadingModels);
        data.progress.download_total_bytes += total_bytes;
        self.publish(&mut data);
    }

    pub(super) fn downloading_models(&self) {
        let mut data = self.data();
        set_phase(&mut data, JobPhase::DownloadingModels);
        self.publish(&mut data);
    }

    pub(super) fn downloaded_bytes(&self, bytes: usize) {
        let mut data = self.data();
        data.progress.downloaded_bytes += bytes;
        self.publish(&mut data);
    }

    pub(super) fn model_download_complete(&self) {
        let mut data = self.data();
        data.progress.processed += 1;
        self.publish(&mut data);
    }

    fn preparing_models(&self, count: u64) {
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
        data.active_asset_paths.clear();
        data.phase = JobPhase::Finished;
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
            IndexEvent::Discovered { count } => data.progress.discovered = count,
            IndexEvent::DiscoveryComplete { total } => {
                data.progress.total = Some(total as u64);
                if data.phase == JobPhase::Scanning {
                    data.progress.phase_completed = total as u64;
                }
            }
            IndexEvent::Progress(delta) => apply_delta(&mut data.progress, delta),
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
        data.active_asset_paths.clear();
    }
}

fn refresh_phase_throughput(data: &mut JobData) {
    if matches!(data.phase, JobPhase::Queued | JobPhase::Finished)
        || data.progress.phase_completed == 0
    {
        return;
    }
    let Some(started_at) = data.phase_started_at else {
        return;
    };
    let elapsed = started_at.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        data.progress.items_per_second = Some(data.progress.phase_completed as f64 / elapsed);
    }
}

#[derive(Default)]
struct JobRegistry {
    active: Option<u64>,
    jobs: BTreeMap<u64, Arc<Job>>,
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
        self.registry
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    pub(super) fn start(self: &Arc<Self>, spec: JobSpec) -> Result<Arc<Job>, ApiError> {
        let job = {
            let mut registry = self.registry();
            if self.shutting_down.load(Ordering::Acquire) {
                return Err(ApiError::shutting_down());
            }
            if let Some(active_id) = registry.active {
                let active = registry.jobs.get(&active_id);
                if active.is_some_and(|job| !job.response().status.is_terminal()) {
                    return Err(ApiError::job_busy());
                }
                registry.active = None;
            }
            evict_retained_jobs(&mut registry);
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let job = Arc::new(Job::new(id, spec.kind()));
            registry.active = Some(id);
            registry.jobs.insert(id, Arc::clone(&job));
            job
        };

        let manager = Arc::clone(self);
        let worker_job = Arc::clone(&job);
        let job_span = info_span!("job", job_id = job.id, kind = ?job.kind);
        tokio::spawn(
            async move {
                match manager.run_job(spec, Arc::clone(&worker_job)).await {
                    Ok(cancelled) => worker_job.complete(cancelled),
                    Err(error) => worker_job.fail(format!("{error:#}")),
                }
                manager.finish(worker_job.id);
            }
            .instrument(job_span),
        );
        Ok(job)
    }

    async fn run_job(&self, spec: JobSpec, job: Arc<Job>) -> anyhow::Result<bool> {
        if !job.begin() {
            return Ok(true);
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
        let image_query_embedder = Arc::clone(&self.image_query_embedder);
        let ocr_models = matches!(&spec, JobSpec::OcrIndex(_))
            .then(|| self.ocr_models.snapshot())
            .flatten();
        let span = Span::current();
        tokio::task::spawn_blocking(move || {
            let _entered = span.enter();
            let needs_text = matches!(&spec, JobSpec::TextEmbed(_) | JobSpec::ModelPrepare)
                || matches!(&spec, JobSpec::OcrIndex(spec) if spec.embeds());
            let needs_image = matches!(&spec, JobSpec::ImageEmbed(_) | JobSpec::ModelPrepare)
                || matches!(&spec, JobSpec::OcrIndex(spec) if spec.embeds());
            if needs_text || needs_image {
                job.preparing_models(u64::from(needs_text) + 2 * u64::from(needs_image));
                if needs_text {
                    embedder.prepare()?;
                    job.models_loaded(1);
                }
                if job.is_cancelled() {
                    return Ok(true);
                }
                if needs_image {
                    image_embedder.prepare()?;
                    job.models_loaded(1);
                    if job.is_cancelled() {
                        return Ok(true);
                    }
                    image_query_embedder.prepare()?;
                    job.models_loaded(1);
                }
                if job.is_cancelled() {
                    return Ok(true);
                }
            }
            match spec {
                JobSpec::ModelPrepare => Ok(false),
                JobSpec::OcrIndex(spec) => {
                    let models = ocr_models.ok_or_else(|| {
                        anyhow::anyhow!(
                            "PaddleOCR models were unloaded after the index job was accepted"
                        )
                    })?;
                    let embed = spec.embeds();
                    let root = spec.root().clone();
                    let retry_failed = spec.retry_failed();
                    let reconciliation = spec.reconciliation();
                    let summary = indexing::run(
                        spec,
                        &databases.assets,
                        &databases.ocr,
                        &models,
                        job.as_ref(),
                    )?;
                    let cancelled = summary.cancelled
                        || (summary.scan_complete
                            && prune_jobs::reconcile(
                                reconciliation,
                                &databases,
                                image_embedder.dimensions(),
                                &thumbnails,
                                job.as_ref(),
                            )?);
                    if cancelled || !embed {
                        return Ok(cancelled);
                    }
                    let cancelled = image_embeddings::run(
                        image_embeddings::Spec::pending_for(root.clone(), retry_failed),
                        &databases.assets,
                        &databases.images,
                        image_embedder.prepare()?.as_ref(),
                        job.as_ref(),
                    )?;
                    if cancelled {
                        return Ok(true);
                    }
                    text_embeddings::job::run(
                        text_embeddings::job::Spec::pending_for(root),
                        &databases.ocr,
                        embedder.prepare()?.as_ref(),
                        job.as_ref(),
                    )
                }
                JobSpec::CatalogSync(spec) => {
                    let reconciliation = spec.reconciliation();
                    let summary =
                        indexing::run_catalog_sync(spec, &databases.assets, job.as_ref())?;
                    Ok(summary.cancelled
                        || (summary.scan_complete
                            && prune_jobs::reconcile(
                                reconciliation,
                                &databases,
                                image_embedder.dimensions(),
                                &thumbnails,
                                job.as_ref(),
                            )?))
                }
                JobSpec::ThumbnailGenerate(spec) => {
                    thumbnails::job::run(spec, &databases.assets, &thumbnails, job.as_ref())
                }
                JobSpec::TextEmbed(spec) => text_embeddings::job::run(
                    spec,
                    &databases.ocr,
                    embedder.prepare()?.as_ref(),
                    job.as_ref(),
                ),
                JobSpec::ImageEmbed(spec) => image_embeddings::run(
                    spec,
                    &databases.assets,
                    &databases.images,
                    image_embedder.prepare()?.as_ref(),
                    job.as_ref(),
                ),
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
            }
        })
        .await
        .map_err(|error| anyhow::anyhow!("job worker failed: {error}"))?
    }

    fn get(&self, id: u64) -> Option<Arc<Job>> {
        self.registry().jobs.get(&id).cloned()
    }

    fn cancel(&self, id: u64) -> Option<Arc<Job>> {
        let job = self.get(id)?;
        job.request_cancel();
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

    pub(crate) fn cancel_all(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let jobs: Vec<_> = self.registry().jobs.values().cloned().collect();
        for job in jobs {
            job.request_cancel();
        }
    }

    fn finish(&self, id: u64) {
        let mut registry = self.registry();
        if registry.active == Some(id) {
            registry.active = None;
        }
    }
}

/// Drop the oldest finished jobs so the retained history stays bounded.
fn evict_retained_jobs(registry: &mut JobRegistry) {
    while registry.jobs.len() >= MAX_RETAINED_JOBS {
        let removable = registry
            .jobs
            .iter()
            .find(|(id, job)| Some(**id) != registry.active && job.response().status.is_terminal())
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
    let spec = request.prepare()?;
    if spec.requires_loaded_ocr_models() && !state.ocr_models.is_loaded() {
        return Err(ApiError::ocr_models_not_loaded());
    }
    let job = state.jobs.start(spec)?;
    Ok((StatusCode::ACCEPTED, Json(job.response())))
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
    fn phase_progress_is_scoped_to_the_current_phase() {
        let job = Job::new(8, JobKind::OcrIndex);
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

        let catalog_sync = serde_json::json!({
            "type": "catalogSync",
            "params": {
                "root": "C:/gallery",
                "scan": {
                    "recursive": false,
                    "exclude": ["*/.cache"]
                }
            }
        });
        assert!(serde_json::from_value::<JobRequest>(catalog_sync).is_ok());

        let thumbnail_generate = serde_json::json!({
            "type": "thumbnailGenerate",
            "params": {
                "root": "C:/gallery"
            }
        });
        assert!(serde_json::from_value::<JobRequest>(thumbnail_generate).is_ok());

        let library_purge = serde_json::json!({
            "type": "libraryPurge",
            "params": {
                "root": "C:/gallery"
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
        let current_dir = PathBuf::try_from(std::env::current_dir().unwrap()).unwrap();
        let request: JobRequest = serde_json::from_value(serde_json::json!({
            "type": "ocrModelLoad",
            "params": {
                "detection": { "modelId": "owner/detection" },
                "recognition": { "modelId": "owner/recognition" }
            }
        }))
        .unwrap();
        let thumbnail_path = current_dir.join("unused-thumbnails.db");
        let runtime_config_dir = tempfile::TempDir::new().unwrap();
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

        let mut updates = job.subscribe();
        job.request_cancel();
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
