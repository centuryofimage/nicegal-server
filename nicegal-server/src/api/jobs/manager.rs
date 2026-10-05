//! Job admission, the serialized worker, and retained history.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use nicegal_core::libraries::ScanOutcome;
use nicegal_core::thumbs::ThumbnailService;
use parking_lot::{Mutex, MutexGuard};
use tracing::{Instrument, Span, info_span};

use super::job::{Job, JobKind, JobListResponse, JobStatus};
use super::request::JobSpec;
use super::routes::{RetainedRequest, wire_job_id};
use super::views::LibraryView;
use super::{JobCancelled, is_cancelled};
use crate::api::error::ApiError;
use crate::api::models::{ImageModel as ImageEmbedder, ImageQueryModel, TextModel as TextEmbedder};
use crate::api::ttl_map::TtlMap;
use crate::api::{
    Databases, image_embeddings, library_scan, ocr_models, prune_jobs, text_embeddings, thumbnails,
};

pub(super) const MAX_RETAINED_JOBS: usize = 32;

#[derive(Default)]
pub(super) struct JobRegistry {
    pub(super) active: Option<u64>,
    pub(super) jobs: BTreeMap<u64, Arc<Job>>,
    pub(super) queued: VecDeque<(u64, JobSpec)>,
    pub(super) views: TtlMap<LibraryView>,
    pub(super) visited: BTreeSet<i64>,
}

pub(crate) struct JobManager {
    pub(super) next_id: AtomicU64,
    pub(super) view_changes: Mutex<()>,
    pub(super) requests: tokio::sync::Mutex<TtlMap<RetainedRequest>>,
    pub(super) shutting_down: AtomicBool,
    pub(super) registry: Mutex<JobRegistry>,
    pub(super) databases: Arc<Databases>,
    pub(super) thumbnails: Arc<ThumbnailService>,
    pub(super) embedder: Arc<TextEmbedder>,
    pub(super) image_embedder: Arc<ImageEmbedder>,
    pub(super) image_query_embedder: Arc<ImageQueryModel>,
    pub(super) ocr_models: Arc<ocr_models::ModelStore>,
    pub(super) runtime: Arc<crate::api::RuntimeSettings>,
}

impl JobManager {
    pub(crate) fn new(
        databases: Arc<Databases>,
        thumbnails: Arc<ThumbnailService>,
        embedder: Arc<TextEmbedder>,
        image_embedder: Arc<ImageEmbedder>,
        image_query_embedder: Arc<ImageQueryModel>,
        ocr_models: Arc<ocr_models::ModelStore>,
        runtime: Arc<crate::api::RuntimeSettings>,
    ) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            view_changes: Mutex::new(()),
            requests: tokio::sync::Mutex::new(TtlMap::default()),
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
}

impl JobManager {
    pub(in crate::api) fn schedule_scan(
        self: &Arc<Self>,
        library_id: i64,
        pending_only: bool,
    ) -> Result<(), ApiError> {
        self.start(JobSpec::LibraryScan(library_scan::Spec::fast(
            library_id,
            pending_only,
        )))?;
        Ok(())
    }

    pub(super) fn registry(&self) -> MutexGuard<'_, JobRegistry> {
        self.registry.lock()
    }

    pub(in crate::api) fn start(self: &Arc<Self>, mut spec: JobSpec) -> Result<Arc<Job>, ApiError> {
        if let JobSpec::LibraryScan(scan) = &mut spec {
            scan.apply_settings(&self.runtime)
                .map_err(ApiError::internal)?;
        }
        let (job, run_now) = {
            let mut registry = self.registry();
            if self.shutting_down.load(Ordering::Acquire) {
                return Err(ApiError::shutting_down());
            }
            let active_exists = registry
                .active
                .is_some_and(|id| registry.jobs.contains_key(&id));
            if active_exists && !spec.queues_behind_active() {
                return Err(ApiError::job_busy());
            }
            // Each library keeps one queued scan. Clients cannot replace another library's work.
            if let JobSpec::LibraryScan(incoming) = &spec {
                let merged = registry
                    .queued
                    .iter_mut()
                    .find_map(|(id, queued)| match queued {
                        JobSpec::LibraryScan(queued)
                            if queued.library_id() == incoming.library_id() =>
                        {
                            queued.merge(incoming);
                            Some(*id)
                        }
                        _ => None,
                    });
                if let Some(id) = merged {
                    return Ok(Arc::clone(
                        registry.jobs.get(&id).expect("queued job exists"),
                    ));
                }
            }
            if let JobSpec::ThumbnailGenerate(thumbnails) = &spec {
                let path = self.resume_path();
                let temporary = path.with_extension("json.tmp");
                std::fs::write(
                    &temporary,
                    serde_json::to_vec(&thumbnails.resume_request())
                        .map_err(|error| ApiError::internal(error.into()))?,
                )
                .map_err(|error| ApiError::internal(error.into()))?;
                std::fs::rename(&temporary, &path)
                    .map_err(|error| ApiError::internal(error.into()))?;
            }
            evict_retained_jobs(&mut registry);
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let job = Arc::new(Job::new_for(id, spec.kind(), spec.library_id()));
            let run_now = if registry.active.is_none() {
                registry.active = Some(id);
                Some(spec)
            } else {
                registry.queued.push_back((id, spec));
                None
            };
            registry.jobs.insert(id, Arc::clone(&job));
            (job, run_now)
        };
        if let Some(spec) = run_now {
            self.spawn_job(spec, Arc::clone(&job));
        }
        Ok(job)
    }

    pub(super) fn spawn_job(self: &Arc<Self>, spec: JobSpec, job: Arc<Job>) {
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
                if worker_job.kind == JobKind::ThumbnailGenerate
                    && !manager.shutting_down.load(Ordering::Acquire)
                {
                    manager.clear_resume();
                }
                manager.finish(worker_job.id);
            }
            .instrument(job_span),
        );
    }

    pub(super) async fn run_job(
        self: &Arc<Self>,
        spec: JobSpec,
        job: Arc<Job>,
    ) -> anyhow::Result<()> {
        if !job.begin() {
            return Err(JobCancelled.into());
        }
        let spec = match spec {
            JobSpec::OcrModelLoad(spec) => {
                return ocr_models::job::run(spec, &self.ocr_models, &self.runtime, job)
                    .await
                    .map(|_| ());
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
                        image_query_embedder: &manager.image_query_embedder,
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
                    if !image_embeddings::has_sources(&spec, &databases)? {
                        return Ok(());
                    }
                    manager.image_query_embedder.prepare_text_queries(&job)?;
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
                    job.check_cancelled()?;
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
    #[tracing::instrument(level = "info", skip(self))]
    pub(super) fn maintain_databases(&self) {
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

    pub(super) fn prepare_job_models(&self, job: &Job) -> anyhow::Result<()> {
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

    pub(super) fn get(&self, id: u64) -> Option<Arc<Job>> {
        self.registry().jobs.get(&id).cloned()
    }

    pub(super) fn record_stopped_scan(&self, job: &Job, outcome: ScanOutcome, message: &str) {
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

    pub(super) fn cancel(&self, id: u64) -> Option<Arc<Job>> {
        let job = self.get(id)?;
        let queued = job.response().status == JobStatus::Queued;
        let active = !job.response().status.is_terminal();
        job.request_cancel();
        if active
            && job.kind == JobKind::ThumbnailGenerate
            && !self.shutting_down.load(Ordering::Acquire)
        {
            self.clear_resume();
        }
        if queued {
            self.registry()
                .queued
                .retain(|(queued_id, _)| *queued_id != id);
            self.record_stopped_scan(&job, ScanOutcome::Cancelled, "cancelled");
        }
        Some(job)
    }

    pub(super) fn list(&self) -> JobListResponse {
        let registry = self.registry();
        JobListResponse {
            active_job_id: registry.active.and_then(|id| {
                registry
                    .jobs
                    .get(&id)
                    .filter(|job| !job.response().status.is_terminal())
                    .map(|_| wire_job_id(id))
            }),
            jobs: registry
                .jobs
                .values()
                .rev()
                .map(|job| job.response())
                .collect(),
        }
    }

    /// Stop is one registry transaction: the worker cannot advance a queued job
    /// between cancelling the active job and cancelling the rest of the queue.
    pub(super) fn stop(&self) -> JobListResponse {
        let (queued, clear_resume) = {
            let mut registry = self.registry();
            let queued: Vec<_> = registry
                .queued
                .iter()
                .filter_map(|(id, _)| registry.jobs.get(id).cloned())
                .collect();
            let mut clear_resume = false;
            for job in registry
                .jobs
                .values()
                .filter(|job| !job.response().status.is_terminal())
            {
                clear_resume |= job.kind == JobKind::ThumbnailGenerate;
                job.request_cancel();
            }
            registry.queued.clear();
            (queued, clear_resume)
        };
        if clear_resume {
            self.clear_resume();
        }
        for job in queued {
            self.record_stopped_scan(&job, ScanOutcome::Cancelled, "cancelled");
        }
        self.list()
    }

    /// Hold job admission while a runtime configuration write checks and updates its state.
    pub(in crate::api) fn update_runtime(
        &self,
        requires_idle: bool,
        update: impl FnOnce() -> anyhow::Result<()>,
    ) -> Result<(), ApiError> {
        let registry = self.registry();
        if requires_idle
            && registry
                .active
                .and_then(|id| registry.jobs.get(&id))
                .is_some_and(|job| !job.data().status.is_terminal())
        {
            return Err(ApiError::job_busy());
        }
        update().map_err(ApiError::internal)
    }

    pub(crate) fn cancel_all(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let jobs: Vec<_> = self.registry().jobs.values().cloned().collect();
        for job in jobs {
            job.request_cancel();
        }
    }

    pub(super) fn finish(self: &Arc<Self>, id: u64) {
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
pub(super) fn evict_retained_jobs(registry: &mut JobRegistry) {
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
