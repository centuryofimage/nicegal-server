use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, select};
use glob::Pattern;
use image::RgbImage;
use tracing::{Span, debug, error, field, info, instrument};
use walkdir::WalkDir;

use crate::assets::{
    Asset, AssetCatalog, CatalogUpsertTimings, canonicalize_path, is_catalog_media, is_ocr_image,
};
use crate::db::{DB, OcrResult};
use crate::ocr::{PaddleOcrModels, PaddleOcrOptions, PaddleOcrPool};

const DEFAULT_COMMIT_CHUNK_SIZE: usize = 32;
const CATALOG_COMMIT_CHUNK_SIZE: usize = 128;
const MAX_DECODE_WORKERS: usize = 4;

pub struct IndexOptions {
    pub limit: Option<usize>,
    pub exclude: Vec<Pattern>,
    pub rescan: bool,
    /// Retry source revisions whose prior decode failed, while retaining current OCR results.
    pub retry_failed: bool,
    pub subdirs: bool,
    pub commit_chunk_size: usize,
    pub cleanup: bool,
    pub max_dimensions: Option<(usize, usize)>,
    pub ocr: PaddleOcrOptions,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            limit: None,
            exclude: Vec::new(),
            rescan: false,
            retry_failed: false,
            subdirs: true,
            commit_chunk_size: DEFAULT_COMMIT_CHUNK_SIZE,
            cleanup: false,
            max_dimensions: None,
            ocr: PaddleOcrOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexSummary {
    pub indexed: usize,
    pub deleted: usize,
    pub cancelled: bool,
    /// Discovery and cataloging exhausted their scope without access errors or a debug limit.
    pub scan_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogSummary {
    pub cataloged: usize,
    pub cancelled: bool,
    pub scan_complete: bool,
}

/// Shared phases emitted by the independent catalog, thumbnail, OCR, embedding, and prune jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexPhase {
    Scanning,
    Cataloging,
    Thumbnails,
    Ocr,
    ImageEmbedding,
    TextEmbedding,
    Pruning,
    Cleanup,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexProgressDelta {
    /// Completed units in the current phase. Unlike the cumulative counters below, this resets
    /// when the observer receives [`IndexEvent::PhaseChanged`].
    pub phase_completed: usize,
    pub processed: usize,
    pub cataloged: usize,
    pub thumbnails_generated: usize,
    pub thumbnail_failures: usize,
    pub prune_candidates: usize,
    pub embedded: usize,
    pub indexed: usize,
    pub skipped: usize,
    pub failed: usize,
    pub deleted: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexEvent {
    PhaseChanged(IndexPhase),
    /// An asset has entered or left the currently executing work set. Parallel pipelines may
    /// report more than one active path at once.
    ActiveAsset {
        path: PathBuf,
        active: bool,
    },
    Discovered {
        count: usize,
    },
    DiscoveryComplete {
        total: usize,
    },
    Progress(IndexProgressDelta),
    Error {
        path: Option<PathBuf>,
        message: String,
    },
}

/// Receives structured background-job progress and supplies cooperative cancellation.
pub trait IndexObserver: Send + Sync {
    fn is_cancelled(&self) -> bool {
        false
    }

    fn on_event(&self, _event: IndexEvent) {}
}

/// Scan and catalog a root without starting any derived work.
///
/// Only `IndexOptions::exclude`, `IndexOptions::limit`, and `IndexOptions::subdirs` apply. Changed
/// assets are committed in bounded batches; progress is published only after each batch is durable
/// and visible to concurrent SQLite readers.
pub fn catalog_dir_observed(
    assets: &AssetCatalog,
    path: &Path,
    options: IndexOptions,
    observer: &dyn IndexObserver,
) -> Result<CatalogSummary> {
    catalog_dir_with_step(assets, path, options, observer, |_| Ok(false))
}

/// Catalog a directory and pass its committed assets, in scan order, to derived work.
/// Returning true from the step cancels the job without discarding committed results.
pub fn catalog_dir_with_step(
    assets: &AssetCatalog,
    path: &Path,
    options: IndexOptions,
    observer: &dyn IndexObserver,
    catalog_step: impl FnOnce(&[Asset]) -> Result<bool>,
) -> Result<CatalogSummary> {
    let (catalog, mut summary) = catalog_snapshot(assets, path, options, observer)?;
    summary.cancelled = summary.cancelled || catalog_step(&catalog)? || observer.is_cancelled();
    Ok(summary)
}

#[instrument(
    name = "catalog_sync",
    skip_all,
    fields(root = %path, cataloged = field::Empty, cancelled = field::Empty)
)]
fn catalog_snapshot(
    assets: &AssetCatalog,
    path: &Path,
    options: IndexOptions,
    observer: &dyn IndexObserver,
) -> Result<(Vec<Asset>, CatalogSummary)> {
    let root = canonicalize_path(path).context("canonicalizing catalog path")?;
    let pipeline = CatalogPipeline {
        assets,
        options: &options,
        observer,
    };
    let files = pipeline.collect_files(&root)?;
    let (catalog, catalog_complete) = if pipeline.cancelled() {
        (Vec::new(), false)
    } else {
        pipeline.catalog_files(&files.paths)?
    };
    let scan_complete = files.complete && catalog_complete && !pipeline.cancelled();
    let cancelled = pipeline.cancelled();
    let summary = CatalogSummary {
        cataloged: catalog.len(),
        cancelled,
        scan_complete,
    };
    let span = Span::current();
    span.record("cataloged", summary.cataloged);
    span.record("cancelled", summary.cancelled);
    Ok((catalog, summary))
}

/// Scan and catalog a root, then OCR its new or changed images with the loaded PaddleOCR pair.
///
/// Cataloging finishes before derived work. OCR decode and completion order may differ from scan
/// order; see [`BoundedOcrPipeline`].
pub fn index_dir_observed(
    assets: &mut AssetCatalog,
    db: &mut DB,
    models: &mut PaddleOcrPool,
    path: &Path,
    options: IndexOptions,
    observer: &dyn IndexObserver,
) -> Result<IndexSummary> {
    index_dir_with_catalog_step(assets, db, models, path, options, observer, |_| Ok(false))
}

/// Run an additional derived-index step with committed assets in scan order, before OCR.
/// Returning true cancels the remaining work while retaining committed results.
pub fn index_dir_with_catalog_step(
    assets: &mut AssetCatalog,
    db: &mut DB,
    models: &mut PaddleOcrPool,
    path: &Path,
    options: IndexOptions,
    observer: &dyn IndexObserver,
    catalog_step: impl FnOnce(&[Asset]) -> Result<bool>,
) -> Result<IndexSummary> {
    if options.commit_chunk_size == 0 {
        bail!("OCR commit chunk size must be greater than zero");
    }
    options.ocr.validate()?;
    IndexPipeline {
        assets,
        db,
        models,
        options,
        observer,
    }
    .run(path, catalog_step)
}

struct IndexPipeline<'a> {
    assets: &'a mut AssetCatalog,
    db: &'a mut DB,
    models: &'a mut PaddleOcrPool,
    options: IndexOptions,
    observer: &'a dyn IndexObserver,
}

struct OcrSelection {
    sources: Vec<Asset>,
    total: usize,
    skipped: usize,
}

fn announce_ocr_selection(observer: &dyn IndexObserver, total: usize, skipped: usize) {
    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::Ocr));
    observer.on_event(IndexEvent::DiscoveryComplete { total });
    if skipped > 0 {
        observer.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: skipped,
            processed: skipped,
            skipped,
            ..IndexProgressDelta::default()
        }));
    }
}

impl IndexPipeline<'_> {
    #[instrument(
        name = "index",
        skip_all,
        fields(root = %path, indexed = field::Empty, deleted = field::Empty, cancelled = field::Empty)
    )]
    fn run(
        &mut self,
        path: &Path,
        catalog_step: impl FnOnce(&[Asset]) -> Result<bool>,
    ) -> Result<IndexSummary> {
        let root = canonicalize_path(path).context("canonicalizing index path")?;
        let (selection, scan_complete) = {
            let Some((catalog, scan_complete)) = self.prepare_catalog(&root)? else {
                return Ok(record_summary(cancelled_summary(0)));
            };
            let selection = self.select_ocr_sources(&catalog)?;
            if catalog_step(&catalog)? || self.cancelled() {
                return Ok(record_summary(cancelled_summary(0)));
            }
            (selection, scan_complete)
        };
        // The derived catalog step may have switched the shared observer to image embedding.
        // Announce OCR only after that step returns so its completions cannot be accumulated under
        // the image phase. Selection still happens first so a CLIP decode failure recorded by the
        // derived step cannot suppress an independent OCR attempt in this same run.
        announce_ocr_selection(self.observer, selection.total, selection.skipped);
        let indexed = self.run_ocr(selection.sources)?;
        if self.cancelled() {
            return Ok(record_summary(cancelled_summary(indexed)));
        }

        let mut summary = self.finish_cleanup(indexed)?;
        summary.scan_complete = scan_complete;
        Ok(record_summary(summary))
    }

    fn prepare_catalog(&mut self, root: &Path) -> Result<Option<(Vec<Asset>, bool)>> {
        let files = CatalogPipeline {
            assets: &*self.assets,
            options: &self.options,
            observer: self.observer,
        }
        .collect_files(root)?;
        if self.cancelled() {
            return Ok(None);
        }
        // Preserve the catalog-first barrier introduced in e19f372. OCR is a derived index; it
        // must not delay otherwise usable image paths from becoming visible to the gallery.
        let (catalog, catalog_complete) = CatalogPipeline {
            assets: &*self.assets,
            options: &self.options,
            observer: self.observer,
        }
        .catalog_files(&files.paths)?;
        if self.cancelled() {
            return Ok(None);
        }
        let scan_complete = files.complete && catalog_complete;
        // Legacy OCR-only mark/sweep cannot represent excluded or unvisited paths.
        self.options.cleanup &=
            scan_complete && self.options.subdirs && self.options.exclude.is_empty();
        if self.options.cleanup {
            self.db.mark_for_deletion(root)?;
        }
        Ok(Some((catalog, scan_complete)))
    }

    #[instrument(
        name = "select",
        skip_all,
        fields(catalog = catalog.len(), sources = field::Empty)
    )]
    fn select_ocr_sources(&mut self, catalog: &[Asset]) -> Result<OcrSelection> {
        let image_fingerprints = catalog
            .iter()
            .filter(|asset| is_ocr_image(asset))
            .map(|asset| (asset.asset_id, asset.fingerprint))
            .collect::<Vec<_>>();
        if self.options.cleanup {
            // A present image is not stale even when unchanged, over the caller's dimension limit,
            // or a new OCR attempt fails. Cleanup removes disappeared sources, not good old results.
            let asset_ids = image_fingerprints
                .iter()
                .map(|(asset_id, _)| *asset_id)
                .collect::<Vec<_>>();
            self.db.unmark_assets(&asset_ids)?;
        }
        let current = if self.options.rescan {
            Default::default()
        } else {
            self.db.current_asset_ids(&image_fingerprints)?
        };
        let decode_failed = if self.options.rescan || self.options.retry_failed {
            Default::default()
        } else {
            self.assets
                .current_decode_failure_asset_ids(&image_fingerprints)?
        };
        let mut sources = Vec::new();
        let mut skipped = 0;
        for asset in catalog {
            if self.cancelled() {
                break;
            }
            if !is_ocr_image(asset) {
                skipped += 1;
                continue;
            }
            if current.contains(&asset.asset_id) {
                skipped += 1;
                continue;
            }
            if decode_failed.contains(&asset.asset_id) {
                skipped += 1;
                continue;
            }
            if self.exceeds_dimension_limit(asset) {
                debug!(
                    asset_id = asset.asset_id,
                    path = %asset.path,
                    width = asset.width.unwrap_or_default(),
                    height = asset.height.unwrap_or_default(),
                    "skipping image over the OCR dimension limit"
                );
                skipped += 1;
                continue;
            }
            sources.push(asset.clone());
        }
        Span::current().record("sources", sources.len());
        Ok(OcrSelection {
            sources,
            total: catalog.len(),
            skipped,
        })
    }

    #[instrument(
        name = "ocr",
        skip_all,
        fields(
            sources = sources.len(),
            decode_workers = field::Empty,
            inference_workers = field::Empty,
            decoded_capacity = field::Empty,
            indexed = field::Empty
        )
    )]
    fn run_ocr(&mut self, sources: Vec<Asset>) -> Result<usize> {
        BoundedOcrPipeline {
            assets: self.assets,
            db: self.db,
            models: self.models,
            options: self.options.ocr,
            commit_chunk_size: self.options.commit_chunk_size,
            observer: self.observer,
        }
        .run(sources)
    }

    #[instrument(name = "cleanup", skip_all, fields(deleted = field::Empty))]
    fn finish_cleanup(&mut self, indexed: usize) -> Result<IndexSummary> {
        self.phase(IndexPhase::Cleanup);
        self.observer
            .on_event(IndexEvent::DiscoveryComplete { total: 1 });
        let deleted = if self.options.cleanup {
            self.db.sweep_deletions()?
        } else {
            0
        };
        self.progress(IndexProgressDelta {
            phase_completed: 1,
            deleted,
            ..IndexProgressDelta::default()
        });
        Span::current().record("deleted", deleted);
        debug!("swept stale OCR entries");
        Ok(IndexSummary {
            indexed,
            deleted,
            cancelled: false,
            scan_complete: false,
        })
    }

    fn exceeds_dimension_limit(&self, asset: &Asset) -> bool {
        matches!(
            (self.options.max_dimensions, asset.width, asset.height),
            (Some((max_width, max_height)), Some(width), Some(height))
                if width as usize > max_width || height as usize > max_height
        )
    }

    fn cancelled(&self) -> bool {
        self.observer.is_cancelled()
    }

    fn phase(&self, phase: IndexPhase) {
        self.observer.on_event(IndexEvent::PhaseChanged(phase));
    }

    fn progress(&self, delta: IndexProgressDelta) {
        self.observer.on_event(IndexEvent::Progress(delta));
    }
}

struct CatalogPipeline<'a> {
    assets: &'a AssetCatalog,
    options: &'a IndexOptions,
    observer: &'a dyn IndexObserver,
}

struct DiscoveredFiles {
    paths: Vec<PathBuf>,
    complete: bool,
}

impl CatalogPipeline<'_> {
    #[instrument(name = "scan", skip_all, fields(files = field::Empty))]
    fn collect_files(&self, root: &Path) -> Result<DiscoveredFiles> {
        self.phase(IndexPhase::Scanning);
        let mut walker = WalkDir::new(root).follow_links(true);
        if !self.options.subdirs {
            walker = walker.max_depth(1);
        }

        let mut files = Vec::new();
        let mut complete = self.options.limit.is_none();
        for result in walker.into_iter().filter_entry(|entry| {
            !self
                .options
                .exclude
                .iter()
                .any(|pattern| pattern.matches_path(entry.path()))
        }) {
            if self.options.limit.is_some_and(|limit| files.len() >= limit) || self.cancelled() {
                complete = false;
                break;
            }
            let entry = match result {
                Ok(entry) => entry,
                Err(error) => {
                    complete = false;
                    let path = error
                        .path()
                        .and_then(|path| PathBuf::from_path_buf(path.to_owned()).ok());
                    self.report_item_error(path, format!("collecting files failed: {error}"));
                    self.progress(IndexProgressDelta {
                        failed: 1,
                        ..IndexProgressDelta::default()
                    });
                    continue;
                }
            };
            if entry.file_type().is_dir() {
                continue;
            }
            let source_path = match PathBuf::try_from(entry.into_path()) {
                Ok(path) => path,
                Err(error) => {
                    complete = false;
                    let path = error.into_path_buf();
                    self.report_item_error(
                        None,
                        format!("media path is not valid UTF-8: {}", path.display()),
                    );
                    self.progress(IndexProgressDelta {
                        failed: 1,
                        ..IndexProgressDelta::default()
                    });
                    continue;
                }
            };
            if is_catalog_media(&source_path) {
                files.push(source_path);
                self.observer
                    .on_event(IndexEvent::Discovered { count: files.len() });
            }
        }
        self.observer
            .on_event(IndexEvent::DiscoveryComplete { total: files.len() });
        Span::current().record("files", files.len());
        Ok(DiscoveredFiles {
            paths: files,
            complete: complete && !self.cancelled(),
        })
    }

    #[instrument(
        name = "catalog",
        skip_all,
        fields(
            files = files.len(),
            cataloged = field::Empty,
            unchanged = field::Empty,
            batch_size = CATALOG_COMMIT_CHUNK_SIZE,
            write_batches = field::Empty,
            metadata_us = field::Empty,
            canonicalize_us = field::Empty,
            fingerprint_us = field::Empty,
            lookup_us = field::Empty,
            probe_us = field::Empty,
            dimensions_us = field::Empty,
            exif_us = field::Empty,
            animation_us = field::Empty,
            store_us = field::Empty,
            transaction_begin_us = field::Empty,
            row_upsert_us = field::Empty,
            revision_update_us = field::Empty,
            commit_us = field::Empty
        )
    )]
    fn catalog_files(&self, files: &[PathBuf]) -> Result<(Vec<Asset>, bool)> {
        self.phase(IndexPhase::Cataloging);
        self.observer
            .on_event(IndexEvent::DiscoveryComplete { total: files.len() });
        let mut catalog = Vec::with_capacity(files.len());
        let mut complete = true;
        let mut metadata_time = Duration::ZERO;
        let mut upsert_timings = CatalogUpsertTimings::default();
        let mut unchanged = 0usize;
        let mut write_batches = 0usize;
        for source_paths in files.chunks(CATALOG_COMMIT_CHUNK_SIZE) {
            let mut prepared = Vec::with_capacity(source_paths.len());
            for source_path in source_paths {
                if self.cancelled() {
                    break;
                }
                let started = Instant::now();
                let metadata = match source_path.metadata() {
                    Ok(metadata) => {
                        metadata_time += started.elapsed();
                        metadata
                    }
                    Err(error) => {
                        metadata_time += started.elapsed();
                        if error.kind() == std::io::ErrorKind::NotFound {
                            self.progress(IndexProgressDelta {
                                phase_completed: 1,
                                processed: 1,
                                skipped: 1,
                                ..IndexProgressDelta::default()
                            });
                            continue;
                        }
                        complete = false;
                        self.report_item_error(
                            Some(source_path.clone()),
                            format!("reading source metadata failed: {error}"),
                        );
                        self.progress(IndexProgressDelta {
                            phase_completed: 1,
                            processed: 1,
                            failed: 1,
                            ..IndexProgressDelta::default()
                        });
                        continue;
                    }
                };
                match self.assets.prepare_upsert_timed(source_path, &metadata) {
                    Ok((asset, timings)) => {
                        unchanged += usize::from(timings.unchanged);
                        upsert_timings.accumulate(timings);
                        prepared.push(asset);
                    }
                    Err(error) => {
                        if is_missing_source_error(&error) {
                            report_disappeared_source(self.observer, source_path);
                            continue;
                        }
                        complete = false;
                        self.report_item_error(
                            Some(source_path.clone()),
                            format!("cataloging media failed: {error:#}"),
                        );
                        self.progress(IndexProgressDelta {
                            phase_completed: 1,
                            processed: 1,
                            failed: 1,
                            ..IndexProgressDelta::default()
                        });
                    }
                }
            }
            if prepared.is_empty() {
                if self.cancelled() {
                    break;
                }
                continue;
            }
            let (assets, timings) = self
                .assets
                .store_prepared_batch(prepared)
                .context("storing catalog batch")?;
            write_batches += usize::from(timings.store > Duration::ZERO);
            upsert_timings.accumulate(timings);
            self.progress(IndexProgressDelta {
                phase_completed: assets.len(),
                cataloged: assets.len(),
                ..IndexProgressDelta::default()
            });
            catalog.extend(assets);
            if self.cancelled() {
                break;
            }
        }
        let span = Span::current();
        span.record("cataloged", catalog.len());
        span.record("unchanged", unchanged);
        span.record("write_batches", write_batches);
        span.record("metadata_us", duration_micros(metadata_time));
        upsert_timings.record(&span);
        Ok((catalog, complete && !self.cancelled()))
    }

    fn cancelled(&self) -> bool {
        self.observer.is_cancelled()
    }

    fn phase(&self, phase: IndexPhase) {
        self.observer.on_event(IndexEvent::PhaseChanged(phase));
    }

    fn progress(&self, delta: IndexProgressDelta) {
        self.observer.on_event(IndexEvent::Progress(delta));
    }

    fn report_item_error(&self, path: Option<PathBuf>, message: String) {
        match &path {
            Some(path) => error!(path = %path, "{message}"),
            None => error!("{message}"),
        }
        self.observer.on_event(IndexEvent::Error { path, message });
    }
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// What a worker reports back to the coordinator about one asset.
///
/// Completion callbacks stay on the coordinator. Workers do announce when a source enters their
/// in-flight set, which lets job observers identify slow images while decode or inference runs.
enum OcrOutcome {
    /// One asset's recognized text, ready to be written.
    Success(Box<OcrResult>),
    /// This exact source revision could not be decoded and should be skipped by every image job.
    DecodeFailure { asset: Asset, message: String },
    /// This asset could not be decoded or inferred. Every other asset still gets indexed.
    ItemFailure { path: PathBuf, message: String },
    /// A worker died. The run can no longer be complete, so it stops and reports.
    Fatal {
        stage: &'static str,
        message: String,
    },
}

/// Decode, inference, and durable writes each run on their own threads, joined by two bounded
/// channels:
///
/// ```text
/// sources -> [decode workers] -> decoded -> [inference replicas] -> outcomes -> coordinator -> DB
/// ```
///
/// The coordinator is the only thread that touches `DB`, keeping SQLite writes single-threaded.
/// Workers may emit observer callbacks concurrently, while committed-result progress is emitted
/// by the coordinator. Results arrive in completion order rather than scan order.
///
/// Every blocking send races the coordinator's abort channel so teardown wakes workers parked on
/// full queues.
struct BoundedOcrPipeline<'a> {
    assets: &'a AssetCatalog,
    db: &'a mut DB,
    models: &'a mut PaddleOcrPool,
    options: PaddleOcrOptions,
    commit_chunk_size: usize,
    observer: &'a dyn IndexObserver,
}

impl BoundedOcrPipeline<'_> {
    fn run(self, sources: Vec<Asset>) -> Result<usize> {
        let Self {
            assets,
            db,
            models,
            options,
            commit_chunk_size,
            observer,
        } = self;
        if sources.is_empty() {
            Span::current().record("decode_workers", 0);
            Span::current().record("inference_workers", 0);
            Span::current().record("decoded_capacity", 0);
            Span::current().record("indexed", 0);
            return Ok(0);
        }

        let replicas = models.replicas_mut();
        assert!(
            !replicas.is_empty(),
            "a PaddleOcrPool always holds at least one replica, or nothing would drain the queue"
        );
        let inference_workers = replicas.len().min(sources.len());
        let decode_workers = decode_worker_count(inference_workers).min(sources.len());
        // Decoded images are full-resolution RGB and dominate this stage's memory — a 16 MP photo
        // is roughly 50 MB — so the decoded queue is sized to keep the replicas fed and no larger.
        // Outcomes are text and cost almost nothing to hold, so that queue is sized to absorb
        // bursts instead, which keeps workers off the coordinator's commit latency.
        let decoded_capacity = inference_workers * 2;
        let outcome_capacity = (commit_chunk_size * 2).max(decode_workers + inference_workers);
        Span::current().record("decode_workers", decode_workers);
        Span::current().record("inference_workers", inference_workers);
        Span::current().record("decoded_capacity", decoded_capacity);

        let next = AtomicUsize::new(0);
        let (decoded_sender, decoded_receiver) = bounded::<DecodedAsset>(decoded_capacity);
        let (outcome_sender, outcome_receiver) = bounded::<OcrOutcome>(outcome_capacity);
        // Nothing is ever sent here. Workers only ever observe the sender being dropped, which is
        // how they are released from a blocking send. See this type's teardown note.
        let (abort_sender, abort_receiver) = bounded::<()>(1);

        let mut pending = Vec::with_capacity(commit_chunk_size);
        let mut indexed = 0_usize;
        let mut write_error = None;
        let mut fatal = None;

        std::thread::scope(|scope| {
            let mut abort_sender = Some(abort_sender);
            let mut workers = Vec::with_capacity(decode_workers + inference_workers);

            for _ in 0..decode_workers {
                let decoded = decoded_sender.clone();
                let outcomes = outcome_sender.clone();
                let abort = abort_receiver.clone();
                let sources = &sources;
                let next = &next;
                workers.push(scope.spawn(move || {
                    guard_worker("decode", &outcomes, &abort, || {
                        decode_assets(sources, next, observer, &decoded, &outcomes, &abort);
                    });
                }));
            }
            drop(decoded_sender);

            for replica in replicas.iter_mut().take(inference_workers) {
                let decoded = decoded_receiver.clone();
                let outcomes = outcome_sender.clone();
                let abort = abort_receiver.clone();
                workers.push(scope.spawn(move || {
                    guard_worker("inference", &outcomes, &abort, || {
                        recognize_assets(replica, options, observer, &decoded, &outcomes, &abort);
                    });
                }));
            }
            drop(decoded_receiver);
            drop(outcome_sender);

            while let Ok(outcome) = outcome_receiver.recv() {
                let stopping = fatal.is_some() || write_error.is_some();
                match outcome {
                    OcrOutcome::Success(result) if !stopping => {
                        observer.on_event(IndexEvent::ActiveAsset {
                            path: result.path.clone(),
                            active: false,
                        });
                        observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                            phase_completed: 1,
                            processed: 1,
                            ..IndexProgressDelta::default()
                        }));
                        pending.push(*result);
                        if pending.len() >= commit_chunk_size {
                            match flush(db, observer, &mut pending) {
                                Ok(count) => indexed += count,
                                Err(error) => {
                                    write_error = Some(error);
                                    abort_sender.take();
                                }
                            }
                        }
                    }
                    OcrOutcome::DecodeFailure { asset, message } if !stopping => {
                        observer.on_event(IndexEvent::ActiveAsset {
                            path: asset.path.clone(),
                            active: false,
                        });
                        match assets.record_decode_failure(&asset) {
                            Ok(()) => report_item_failure(observer, &asset.path, message),
                            Err(error) => {
                                write_error = Some(error);
                                abort_sender.take();
                            }
                        }
                    }
                    OcrOutcome::ItemFailure { path, message } if !stopping => {
                        observer.on_event(IndexEvent::ActiveAsset {
                            path: path.clone(),
                            active: false,
                        });
                        report_item_failure(observer, &path, message);
                    }
                    OcrOutcome::Fatal { stage, message } => {
                        fatal.get_or_insert_with(|| format!("the {stage} stage failed: {message}"));
                        abort_sender.take();
                    }
                    // Already stopping: drain the rest so nothing is left parked in a send.
                    OcrOutcome::Success(_)
                    | OcrOutcome::DecodeFailure { .. }
                    | OcrOutcome::ItemFailure { .. } => {}
                }
            }

            // Release anything still parked, then confirm each worker actually finished instead of
            // reading a closed channel as success.
            drop(abort_sender);
            for worker in workers {
                if worker.join().is_err() {
                    fatal.get_or_insert_with(|| "an OCR worker thread panicked".to_owned());
                }
            }
        });

        if let Some(error) = write_error {
            return Err(error);
        }
        if let Some(message) = fatal {
            bail!("{message}");
        }

        indexed += flush(db, observer, &mut pending)?;
        Span::current().record("indexed", indexed);
        info!(
            indexed,
            decode_workers,
            inference_workers,
            cancelled = observer.is_cancelled(),
            "PaddleOCR phase complete"
        );
        Ok(indexed)
    }
}

pub(crate) fn worker_panic(body: impl FnOnce()) -> Option<String> {
    let panic = catch_unwind(AssertUnwindSafe(body)).err()?;
    Some(
        panic
            .downcast_ref::<&str>()
            .map(|text| (*text).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panicked".to_owned()),
    )
}

/// Run a worker body, turning a panic into a [`OcrOutcome::Fatal`] rather than unwinding out of
/// the thread.
///
/// The body is asserted unwind-safe because a panic ends the run: the replica it borrows is never
/// touched again, whatever state its scratch buffers were left in.
fn guard_worker(
    stage: &'static str,
    outcomes: &Sender<OcrOutcome>,
    abort: &Receiver<()>,
    body: impl FnOnce(),
) {
    let Some(message) = crate::index::worker_panic(body) else {
        return;
    };
    error!(stage, "{message}");
    send_unless_aborted(outcomes, OcrOutcome::Fatal { stage, message }, abort);
}

/// Send, unless the coordinator has torn the pipeline down. Returns whether the item was sent.
///
/// A plain blocking send on a full channel is only released by every receiver dropping, which
/// would leave this worker parked while `thread::scope` waits to join it. Racing the abort channel
/// makes every blocking send cancellable instead.
pub(crate) fn send_unless_aborted<T>(sender: &Sender<T>, item: T, abort: &Receiver<()>) -> bool {
    select! {
        send(sender, item) -> sent => sent.is_ok(),
        recv(abort) -> _ => false,
    }
}

/// Whether the coordinator has dropped its abort sender.
pub(crate) fn aborted(abort: &Receiver<()>) -> bool {
    matches!(abort.try_recv(), Err(TryRecvError::Disconnected))
}

/// Take decoded images and run one replica's sessions over them.
///
fn recognize_assets(
    models: &mut PaddleOcrModels,
    options: PaddleOcrOptions,
    observer: &dyn IndexObserver,
    decoded: &Receiver<DecodedAsset>,
    outcomes: &Sender<OcrOutcome>,
    abort: &Receiver<()>,
) {
    loop {
        let work = select! {
            recv(decoded) -> work => match work {
                Ok(work) => work,
                Err(_) => break,
            },
            recv(abort) -> _ => break,
        };
        // Keep draining after cancellation so decoders are never left parked on a full queue,
        // but start no further inference.
        if observer.is_cancelled() {
            continue;
        }
        if !send_unless_aborted(outcomes, scan_asset(models, options, work), abort) {
            break;
        }
    }
}

#[instrument(name = "ocr_asset", skip_all, fields(path = %work.asset.path))]
fn scan_asset(
    models: &mut PaddleOcrModels,
    options: PaddleOcrOptions,
    work: DecodedAsset,
) -> OcrOutcome {
    let asset = work.asset;
    match models.scan(&work.image, options) {
        Ok(output) => OcrOutcome::Success(Box::new(OcrResult {
            asset_id: asset.asset_id,
            path: asset.path,
            fingerprint: asset.fingerprint,
            exif_taken_ns: asset.exif_taken_ns,
            width: output.width,
            height: output.height,
            contents: output.contents,
        })),
        Err(error) => OcrOutcome::ItemFailure {
            path: asset.path,
            message: format!("PaddleOCR inference failed: {error:#}"),
        },
    }
}

/// Progress for indexed rows is published only once the transaction is durable, so a reader that
/// acts on the count always finds the rows behind it.
fn flush(db: &mut DB, observer: &dyn IndexObserver, pending: &mut Vec<OcrResult>) -> Result<usize> {
    if pending.is_empty() {
        return Ok(0);
    }
    let count = db.save_results(std::mem::take(pending))?;
    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
        indexed: count,
        ..IndexProgressDelta::default()
    }));
    debug!(rows = count, "saved a PaddleOCR chunk");
    Ok(count)
}

pub(crate) fn report_item_failure(observer: &dyn IndexObserver, path: &Path, message: String) {
    error!(path = %path, "{message}");
    observer.on_event(IndexEvent::Error {
        path: Some(path.to_owned()),
        message,
    });
    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: 1,
        processed: 1,
        failed: 1,
        ..IndexProgressDelta::default()
    }));
}

/// A source removed between discovery and opening is ordinary library churn. Only an actual
/// filesystem NotFound is skipped; permission, malformed media, and inference errors stay visible.
pub(crate) fn is_missing_source_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
    })
}

pub(crate) fn report_disappeared_source(observer: &dyn IndexObserver, path: &Path) {
    observer.on_event(IndexEvent::ActiveAsset {
        path: path.to_owned(),
        active: false,
    });
    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: 1,
        processed: 1,
        skipped: 1,
        ..IndexProgressDelta::default()
    }));
}

struct DecodedAsset {
    asset: Asset,
    image: RgbImage,
}

/// Claim sources from the shared cursor and decode them. A file that cannot be decoded is reported
/// as one failed item and the worker moves on.
fn decode_assets(
    sources: &[Asset],
    next: &AtomicUsize,
    observer: &dyn IndexObserver,
    decoded: &Sender<DecodedAsset>,
    outcomes: &Sender<OcrOutcome>,
    abort: &Receiver<()>,
) {
    loop {
        if aborted(abort) || observer.is_cancelled() {
            break;
        }
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(asset) = sources.get(index) else {
            break;
        };
        observer.on_event(IndexEvent::ActiveAsset {
            path: asset.path.clone(),
            active: true,
        });
        debug!(path = %asset.path, "decoding image for OCR");
        let sent = match decode_image(&asset.path) {
            Ok(image) => send_unless_aborted(
                decoded,
                DecodedAsset {
                    asset: asset.clone(),
                    image,
                },
                abort,
            ),
            Err(error) if is_missing_source_error(&error) => {
                report_disappeared_source(observer, &asset.path);
                continue;
            }
            Err(error) => send_unless_aborted(
                outcomes,
                OcrOutcome::DecodeFailure {
                    asset: asset.clone(),
                    message: format!("decoding image for OCR failed: {error:#}"),
                },
                abort,
            ),
        };
        if !sent {
            break;
        }
    }
}

fn decode_worker_count(inference_workers: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    cores
        .div_ceil(4)
        .max(inference_workers)
        .clamp(1, MAX_DECODE_WORKERS)
}

fn decode_image(path: &Path) -> Result<RgbImage> {
    let data = std::fs::read(path).with_context(|| format!("opening image: {path}"))?;
    let raster =
        crate::imaging::decode(&data).with_context(|| format!("decoding image: {path}"))?;
    let (width, height) = (raster.width(), raster.height());
    RgbImage::from_raw(width, height, raster.into_rgb_bytes())
        .context("assembling decoded image buffer")
}

fn record_summary(summary: IndexSummary) -> IndexSummary {
    let span = Span::current();
    span.record("indexed", summary.indexed);
    span.record("deleted", summary.deleted);
    span.record("cancelled", summary.cancelled);
    summary
}

fn cancelled_summary(indexed: usize) -> IndexSummary {
    IndexSummary {
        indexed,
        deleted: 0,
        cancelled: true,
        scan_complete: false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    use super::*;
    use camino::Utf8PathBuf as PathBuf;
    use tempfile::TempDir;

    struct TestObserver {
        cancelled: bool,
    }
    impl IndexObserver for TestObserver {
        fn is_cancelled(&self) -> bool {
            self.cancelled
        }
    }

    #[derive(Default)]
    struct EventObserver(Mutex<Vec<IndexEvent>>);

    impl IndexObserver for EventObserver {
        fn on_event(&self, event: IndexEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    #[test]
    fn ocr_progress_is_reannounced_after_a_derived_catalog_step() {
        let observer = EventObserver::default();
        observer.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
        observer.on_event(IndexEvent::DiscoveryComplete { total: 71_882 });
        observer.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: 71_882,
            ..IndexProgressDelta::default()
        }));

        announce_ocr_selection(&observer, 72_228, 71_882);

        let events = observer.0.lock().unwrap();
        assert!(matches!(
            events[3],
            IndexEvent::PhaseChanged(IndexPhase::Ocr)
        ));
        assert!(matches!(
            events[4],
            IndexEvent::DiscoveryComplete { total: 72_228 }
        ));
        assert!(matches!(
            events[5],
            IndexEvent::Progress(IndexProgressDelta {
                phase_completed: 71_882,
                skipped: 71_882,
                ..
            })
        ));
    }

    #[test]
    fn derived_step_receives_scan_order_instead_of_insertion_order() -> Result<()> {
        let temp = TempDir::new()?;
        let root = canonicalize_path(&PathBuf::try_from(temp.path().to_owned())?)?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        for name in ["first.png", "second.png", "third.png"] {
            std::fs::write(root.join(name), b"unreadable image")?;
        }
        let observer = TestObserver { cancelled: false };
        let options = IndexOptions::default();
        let expected = CatalogPipeline {
            assets: &assets,
            options: &options,
            observer: &observer,
        }
        .collect_files(&root)?
        .paths;
        for path in expected.iter().rev() {
            assets.upsert(path, &path.metadata()?)?;
        }
        let mut called = false;
        let summary = catalog_dir_with_step(&assets, &root, options, &observer, |catalog| {
            called = true;
            assert_eq!(
                catalog.iter().map(|asset| &asset.path).collect::<Vec<_>>(),
                expected.iter().collect::<Vec<_>>()
            );
            assert!(
                catalog
                    .windows(2)
                    .all(|pair| pair[0].asset_id > pair[1].asset_id)
            );
            Ok(false)
        })?;
        assert!(called);
        assert!(summary.scan_complete);
        assert_eq!(summary.cataloged, 3);
        Ok(())
    }

    #[test]
    fn scan_completeness_rejects_missing_roots_limits_and_cancellation() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_owned())?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        let observer = TestObserver { cancelled: false };
        let options = IndexOptions::default();
        let scan = CatalogPipeline {
            assets: &assets,
            options: &options,
            observer: &observer,
        };
        assert!(scan.collect_files(&root)?.complete);
        assert!(!scan.collect_files(&root.join("unavailable"))?.complete);
        let limited = IndexOptions {
            limit: Some(100),
            ..IndexOptions::default()
        };
        assert!(
            !CatalogPipeline {
                options: &limited,
                ..scan
            }
            .collect_files(&root)?
            .complete
        );
        let cancelled = TestObserver { cancelled: true };
        let summary = catalog_dir_observed(&assets, &root, options, &cancelled)?;
        assert!(summary.cancelled);
        assert!(!summary.scan_complete);
        Ok(())
    }

    #[test]
    fn disappeared_sources_are_not_confused_with_permission_or_decode_failures() -> Result<()> {
        let temp = TempDir::new()?;
        let missing = PathBuf::try_from(temp.path().join("missing.png"))?;
        assert!(is_missing_source_error(
            &decode_image(&missing).unwrap_err()
        ));
        std::fs::write(&missing, b"broken image")?;
        assert!(!is_missing_source_error(
            &decode_image(&missing).unwrap_err()
        ));
        let denied = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
            .context("opening source");
        assert!(!is_missing_source_error(&denied));
        Ok(())
    }

    #[test]
    fn catalog_access_errors_prevent_reconciliation_but_disappeared_files_do_not() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_owned())?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        let options = IndexOptions::default();
        let observer = TestObserver { cancelled: false };
        let catalog = CatalogPipeline {
            assets: &assets,
            options: &options,
            observer: &observer,
        };
        assert!(catalog.catalog_files(&[root.join("missing.png")])?.1);
        assert!(!catalog.catalog_files(&[root.join("invalid\0.png")])?.1);
        Ok(())
    }

    /// Runs on a helper thread so a blocked-producer regression times out instead of hanging.
    #[test]
    fn a_panicking_consumer_cannot_strand_a_blocked_producer() {
        let (finished, outcome) = channel();
        std::thread::spawn(move || {
            let (items_sender, items_receiver) = bounded::<usize>(2);
            let (abort_sender, abort_receiver) = bounded::<()>(1);
            let panicked = catch_unwind(AssertUnwindSafe(|| {
                std::thread::scope(|scope| {
                    // Moved into the scope closure on purpose: dropping it here during the unwind
                    // below, before the scope joins, is the whole mechanism under test.
                    let _abort_sender = abort_sender;
                    scope.spawn(move || {
                        for item in 0..1_000 {
                            if !send_unless_aborted(&items_sender, item, &abort_receiver) {
                                break;
                            }
                        }
                    });
                    items_receiver.recv().expect("the producer sends first");
                    panic!("inference failed");
                });
            }))
            .is_err();
            let _ = finished.send(panicked);
        });

        let panicked = outcome
            .recv_timeout(Duration::from_secs(10))
            .expect("the scope must finish rather than deadlock on a blocked producer");
        assert!(panicked, "the panic must still reach the caller");
    }

    /// A worker panic becomes a reported outcome instead of unwinding past the coordinator.
    #[test]
    fn a_worker_panic_is_reported_as_a_fatal_outcome() {
        let (outcomes, reported) = bounded::<OcrOutcome>(1);
        let (abort_sender, abort_receiver) = bounded::<()>(1);
        guard_worker("inference", &outcomes, &abort_receiver, || {
            panic!("detector exploded");
        });
        drop(abort_sender);
        match reported.recv() {
            Ok(OcrOutcome::Fatal { stage, message }) => {
                assert_eq!(stage, "inference");
                assert_eq!(message, "detector exploded");
            }
            other => panic!("expected a fatal outcome, got {:?}", other.is_ok()),
        }
    }

    #[test]
    fn decode_worker_count_stays_bounded_away_from_ort() {
        for inference_workers in 1..=8 {
            assert!((1..=MAX_DECODE_WORKERS).contains(&decode_worker_count(inference_workers)));
        }
    }

    #[test]
    fn ocr_decode_uses_visual_jpeg_orientation() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("oriented.jpg"))?;
        std::fs::write(
            &path,
            crate::imaging::test_support::jpeg_with_orientation(6)?,
        )?;

        let image = decode_image(&path)?;

        assert_eq!((image.width(), image.height()), (2, 3));
        Ok(())
    }

    #[test]
    fn default_index_options_keep_durable_chunks_small() {
        let options = IndexOptions::default();
        assert_eq!(options.commit_chunk_size, 32);
        assert!(options.subdirs);
        assert!(!options.rescan);
        assert!(!options.cleanup);
    }
}
