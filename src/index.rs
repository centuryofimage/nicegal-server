use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, select};
use glob::Pattern;
use image::RgbImage;
use tracing::{Span, debug, error, field, info, instrument};
use walkdir::WalkDir;

use crate::assets::{
    Asset, AssetCatalog, CatalogScanEntry, CatalogUpsertTimings, SourceFingerprint,
    canonicalize_path, is_catalog_media, is_catalog_video, is_ocr_image,
};
use crate::db::{DB, OcrResult};
use crate::ocr::{PaddleOcrModels, PaddleOcrOptions, PaddleOcrPool};
use crate::scope::PathScope;

const DEFAULT_COMMIT_CHUNK_SIZE: usize = 32;
const CATALOG_COMMIT_CHUNK_SIZE: usize = 128;
const CATALOG_VIDEO_PROBE_WORKERS: usize = 6;
const CATALOG_VIDEO_PROBE_BACKLOG: usize = 512;
const MAX_DECODE_WORKERS: usize = 4;

type CatalogVideoProbeResult = (PathBuf, Option<crate::video::VideoMetadata>, Duration);

pub struct IndexOptions {
    pub limit: Option<usize>,
    pub exclude: Vec<Pattern>,
    /// Literal directories the walk skips with their whole subtree, such as a library's excluded
    /// folders. Unlike `exclude`, these are paths, so glob characters in folder names are safe.
    pub exclude_dirs: Vec<PathBuf>,
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
            exclude_dirs: Vec::new(),
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
    /// Discovery and cataloging exhausted their scope without access errors or reaching a debug
    /// limit.
    pub scan_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogSummary {
    pub cataloged: usize,
    pub cancelled: bool,
    pub scan_complete: bool,
}

/// Work found by one directory walk. Only new or changed rows enter derived indexing;
/// unseen paths are candidates for verified deletion after a complete walk.
pub struct CatalogDelta {
    pub cataloged: Vec<Asset>,
    pub unseen_paths: Vec<PathBuf>,
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

    /// Shared cancellation flag for blocking decoders that must stop inside native calls.
    fn cancellation_token(&self) -> Option<Arc<AtomicBool>> {
        None
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

/// Discover new, changed, and missing catalog paths without probing unchanged media.
pub fn catalog_delta_dir_observed(
    assets: &AssetCatalog,
    path: &Path,
    options: IndexOptions,
    observer: &dyn IndexObserver,
) -> Result<CatalogDelta> {
    let root = canonicalize_path(path).context("canonicalizing catalog path")?;
    let pipeline = CatalogPipeline {
        assets,
        options: &options,
        observer,
    };
    let files = pipeline.collect_files(&root, ScanSelection::Delta)?;
    let (cataloged, catalog_complete) = if pipeline.cancelled() {
        (Vec::new(), false)
    } else {
        pipeline.catalog_files(&files.paths)?
    };
    Ok(CatalogDelta {
        cataloged,
        unseen_paths: files.unseen_paths,
        cancelled: pipeline.cancelled(),
        scan_complete: files.complete && catalog_complete && !pipeline.cancelled(),
    })
}

/// Scan and catalog a directory, returning every asset the walk visited in scan order. Unchanged
/// assets keep their rows; only new or changed files are probed.
#[instrument(
    name = "catalog_sync",
    skip_all,
    fields(root = %path, cataloged = field::Empty, cancelled = field::Empty)
)]
pub fn catalog_snapshot(
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
    let files = pipeline.collect_files(&root, ScanSelection::All)?;
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

/// The images in a catalog pass that still need text recognition.
pub struct OcrSelection {
    sources: Vec<Asset>,
    total: usize,
    skipped: usize,
}

impl OcrSelection {
    /// Whether recognition has anything to do, so its models need not load when it does not.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }
}

/// Choose which assets of `catalog` need OCR: current images without a result for their present
/// fingerprint, skipping recorded decode failures and images over `options.max_dimensions`.
/// `options.rescan` selects every image and `options.retry_failed` retries decode failures.
pub fn select_ocr_sources(
    assets: &AssetCatalog,
    db: &DB,
    catalog: &[Asset],
    options: &IndexOptions,
    observer: &dyn IndexObserver,
) -> Result<OcrSelection> {
    let image_fingerprints = catalog
        .iter()
        .filter(|asset| is_ocr_image(asset))
        .map(|asset| (asset.asset_id, asset.fingerprint))
        .collect::<Vec<_>>();
    let current = if options.rescan {
        Default::default()
    } else {
        db.current_asset_ids(&image_fingerprints)?
    };
    let decode_failed = if options.rescan || options.retry_failed {
        Default::default()
    } else {
        assets.current_decode_failure_asset_ids(&image_fingerprints)?
    };
    let exceeds_dimension_limit = |asset: &Asset| {
        matches!(
            (options.max_dimensions, asset.width, asset.height),
            (Some((max_width, max_height)), Some(width), Some(height))
                if width as usize > max_width || height as usize > max_height
        )
    };
    let mut sources = Vec::new();
    let mut skipped = 0;
    for asset in catalog {
        if observer.is_cancelled() {
            break;
        }
        if !is_ocr_image(asset)
            || current.contains(&asset.asset_id)
            || decode_failed.contains(&asset.asset_id)
        {
            skipped += 1;
            continue;
        }
        if exceeds_dimension_limit(asset) {
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
    Ok(OcrSelection {
        sources,
        total: catalog.len(),
        skipped,
    })
}

/// Recognize a selection with loaded models, announcing the OCR phase first. Returns how many
/// assets were indexed; cancellation stops early with committed results retained.
pub fn recognize_selection(
    assets: &AssetCatalog,
    db: &mut DB,
    models: &mut PaddleOcrPool,
    selection: OcrSelection,
    options: &IndexOptions,
    observer: &dyn IndexObserver,
) -> Result<usize> {
    if options.commit_chunk_size == 0 {
        bail!("OCR commit chunk size must be greater than zero");
    }
    options.ocr.validate()?;
    announce_ocr_selection(observer, selection.total, selection.skipped);
    let mut sources = selection.sources;
    sources.retain(is_ocr_image);
    BoundedOcrPipeline {
        assets,
        db,
        models,
        options: options.ocr,
        commit_chunk_size: options.commit_chunk_size,
        observer,
    }
    .run(sources)
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
        .collect_files(root, ScanSelection::All)?;
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
        self.options.cleanup &= scan_complete
            && self.options.subdirs
            && self.options.exclude.is_empty()
            && self.options.exclude_dirs.is_empty();
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
        if self.options.cleanup {
            // A present image is not stale even when unchanged, over the caller's dimension limit,
            // or a new OCR attempt fails. Cleanup removes disappeared sources, not good old results.
            let asset_ids = catalog
                .iter()
                .filter(|asset| is_ocr_image(asset))
                .map(|asset| asset.asset_id)
                .collect::<Vec<_>>();
            self.db.unmark_assets(&asset_ids)?;
        }
        let selection =
            select_ocr_sources(self.assets, self.db, catalog, &self.options, self.observer)?;
        Span::current().record("sources", selection.sources.len());
        Ok(selection)
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
    fn run_ocr(&mut self, mut sources: Vec<Asset>) -> Result<usize> {
        // Keep the OCR boundary explicit even if a caller bypasses normal selection during a
        // forced rescan or a retry of a failed source.
        sources.retain(is_ocr_image);
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

fn fill_catalog_video_probes<'scope>(
    scope: &rayon::Scope<'scope>,
    observer: &'scope dyn IndexObserver,
    sender: &Sender<CatalogVideoProbeResult>,
    candidates: &[PathBuf],
    next: &mut usize,
    in_flight: &mut usize,
    abort: &Arc<AtomicBool>,
) {
    while *in_flight < CATALOG_VIDEO_PROBE_BACKLOG && *next < candidates.len() {
        let path = candidates[*next].clone();
        *next += 1;
        *in_flight += 1;
        let sender = sender.clone();
        let abort = Arc::clone(abort);
        scope.spawn(move |_| {
            let started = Instant::now();
            let metadata = if observer.is_cancelled() || abort.load(Ordering::Relaxed) {
                None
            } else {
                observer.on_event(IndexEvent::ActiveAsset {
                    path: path.clone(),
                    active: true,
                });
                let metadata = crate::video::probe_catalog(&path).ok();
                observer.on_event(IndexEvent::ActiveAsset {
                    path: path.clone(),
                    active: false,
                });
                metadata
            };
            let _ = sender.send((path, metadata, started.elapsed()));
        });
    }
}

fn run_catalog_video_probes(
    observer: &dyn IndexObserver,
    candidates: Vec<PathBuf>,
    output: Sender<CatalogVideoProbeResult>,
    abort: Arc<AtomicBool>,
) -> Result<()> {
    if candidates.is_empty() {
        return Ok(());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        // The scope coordinator waits on Crossbeam from one Rayon thread.
        .num_threads(CATALOG_VIDEO_PROBE_WORKERS + 1)
        .thread_name(|index| format!("catalog-video-probe-{index}"))
        .build()?;
    let (sender, receiver) = crossbeam_channel::unbounded();
    pool.scope(|scope| -> Result<()> {
        let mut next = 0;
        let mut in_flight = 0;
        fill_catalog_video_probes(
            scope,
            observer,
            &sender,
            &candidates,
            &mut next,
            &mut in_flight,
            &abort,
        );
        for _ in 0..candidates.len() {
            let result = receiver.recv()?;
            in_flight -= 1;
            output.send(result)?;
            fill_catalog_video_probes(
                scope,
                observer,
                &sender,
                &candidates,
                &mut next,
                &mut in_flight,
                &abort,
            );
        }
        Ok(())
    })
}

struct DiscoveredFiles {
    paths: Vec<PathBuf>,
    unseen_paths: Vec<PathBuf>,
    complete: bool,
}

#[derive(Clone, Copy)]
enum ScanSelection {
    All,
    Delta,
}

impl CatalogPipeline<'_> {
    #[instrument(name = "scan", skip_all, fields(files = field::Empty))]
    fn collect_files(&self, root: &Path, selection: ScanSelection) -> Result<DiscoveredFiles> {
        self.phase(IndexPhase::Scanning);
        let mut existing: Option<HashMap<PathBuf, CatalogScanEntry>> = match selection {
            ScanSelection::All => None,
            ScanSelection::Delta => Some(self.assets.scan_entries_under_root(root)?),
        };
        let mut walker = WalkDir::new(root).follow_links(true);
        if !self.options.subdirs {
            walker = walker.max_depth(1);
        }

        let mut files = Vec::new();
        let mut complete = true;
        for result in walker.into_iter().filter_entry(|entry| {
            !self
                .options
                .exclude
                .iter()
                .any(|pattern| pattern.matches_path(entry.path()))
                && !is_excluded_dir(&self.options.exclude_dirs, entry)
        }) {
            if self.cancelled() {
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
            let source_path = match PathBuf::try_from(entry.path().to_owned()) {
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
            if !is_catalog_media(&source_path) {
                continue;
            }
            // Removing every visited path leaves only possible deletions at the end of the walk.
            if let Some(existing) = &mut existing
                && let Some(known) = existing.remove(&source_path)
            {
                // WalkDir reuses directory-entry metadata on Windows for ordinary files.
                let current = (|| -> Result<bool> { known.is_current(&entry.metadata()?) })();
                match current {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(error) => {
                        complete = false;
                        self.report_item_error(
                            Some(source_path),
                            format!("checking catalog source metadata failed: {error}"),
                        );
                        self.progress(IndexProgressDelta {
                            failed: 1,
                            ..IndexProgressDelta::default()
                        });
                        continue;
                    }
                }
            }
            // A limit only leaves the walk incomplete once another file is actually left out.
            if self.options.limit.is_some_and(|limit| files.len() >= limit) {
                complete = false;
                break;
            }
            files.push(source_path);
            self.observer
                .on_event(IndexEvent::Discovered { count: files.len() });
        }
        self.observer
            .on_event(IndexEvent::DiscoveryComplete { total: files.len() });
        Span::current().record("files", files.len());
        Ok(DiscoveredFiles {
            paths: files,
            // An incomplete walk cannot establish absence, so never expose deletion candidates.
            unseen_paths: if complete {
                existing
                    .map(|entries| entries.into_keys().collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            },
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
        // Run slow FFmpeg probes on a small dedicated pool. Catalog writes stay on this thread;
        // photos can be committed while video metadata is still being read.
        let (probe_sender, probe_receiver) = crossbeam_channel::unbounded();
        let abort = Arc::new(AtomicBool::new(false));
        let mut video_candidates = Vec::new();
        let mut video_fingerprints = HashMap::new();
        for source_path in files.iter().filter(|path| is_catalog_video(path)) {
            if self.cancelled() {
                break;
            }
            let Ok(metadata) = source_path.metadata() else {
                continue;
            };
            if self.assets.video_needs_probe(source_path, &metadata)? {
                video_fingerprints.insert(
                    source_path.clone(),
                    SourceFingerprint::from_metadata(&metadata)?,
                );
                video_candidates.push(source_path.clone());
            }
        }
        let candidate_set: HashSet<_> = video_candidates.iter().cloned().collect();
        let ordered_files: Vec<_> = files
            .iter()
            .enumerate()
            .filter(|(_, path)| !is_catalog_video(path))
            .chain(
                files
                    .iter()
                    .enumerate()
                    .filter(|(_, path)| is_catalog_video(path)),
            )
            .collect();
        let observer = self.observer;
        std::thread::scope(|scope| -> Result<()> {
            let worker_abort = Arc::clone(&abort);
            let probe_thread = scope.spawn(move || {
                run_catalog_video_probes(observer, video_candidates, probe_sender, worker_abort)
            });
            let catalog_result = (|| -> Result<()> {
                let mut video_probes = HashMap::new();
                for source_paths in ordered_files.chunks(CATALOG_COMMIT_CHUNK_SIZE) {
                    let mut prepared = Vec::with_capacity(source_paths.len());
                    for &(position, source_path) in source_paths {
                        if candidate_set.contains(source_path) {
                            while !video_probes.contains_key(source_path) {
                                let (path, metadata, elapsed) = probe_receiver.recv()?;
                                video_probes.insert(path, (metadata, elapsed));
                            }
                        }
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
                                video_probes.remove(source_path);
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
                        let preprobed = video_probes.remove(source_path).filter(|_| {
                            SourceFingerprint::from_metadata(&metadata).ok().as_ref()
                                == video_fingerprints.get(source_path)
                        });
                        let probe_duration = preprobed.as_ref().map(|(_, elapsed)| *elapsed);
                        let result = self.assets.prepare_upsert_timed_with_probe(
                            source_path,
                            &metadata,
                            |path| match preprobed {
                                Some((metadata, _)) => metadata,
                                None => {
                                    self.observer.on_event(IndexEvent::ActiveAsset {
                                        path: source_path.clone(),
                                        active: true,
                                    });
                                    let metadata = crate::video::probe_catalog(path).ok();
                                    self.observer.on_event(IndexEvent::ActiveAsset {
                                        path: source_path.clone(),
                                        active: false,
                                    });
                                    metadata
                                }
                            },
                        );
                        match result {
                            Ok((asset, mut timings)) => {
                                if let Some(duration) = probe_duration {
                                    timings.probe += duration;
                                    timings.dimensions += duration;
                                }
                                unchanged += usize::from(timings.unchanged);
                                upsert_timings.accumulate(timings);
                                prepared.push((position, asset));
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
                    let positions: Vec<_> =
                        prepared.iter().map(|(position, _)| *position).collect();
                    let (assets, timings) = self
                        .assets
                        .store_prepared_batch(
                            prepared.into_iter().map(|(_, asset)| asset).collect(),
                        )
                        .context("storing catalog batch")?;
                    write_batches += usize::from(timings.store > Duration::ZERO);
                    upsert_timings.accumulate(timings);
                    self.progress(IndexProgressDelta {
                        phase_completed: assets.len(),
                        cataloged: assets.len(),
                        ..IndexProgressDelta::default()
                    });
                    catalog.extend(positions.into_iter().zip(assets));
                    if self.cancelled() {
                        break;
                    }
                }
                Ok(())
            })();
            if catalog_result.is_err() || self.cancelled() {
                abort.store(true, Ordering::Relaxed);
            }
            let probe_result = probe_thread
                .join()
                .map_err(|_| anyhow::anyhow!("catalog video probe coordinator panicked"))?;
            catalog_result?;
            probe_result
        })?;
        catalog.sort_unstable_by_key(|(position, _)| *position);
        let catalog: Vec<_> = catalog.into_iter().map(|(_, asset)| asset).collect();
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

/// Whether a walked entry is one of the literal excluded directories, or a directory link that
/// resolves into one. The walk prunes an excluded directory's subtree, so matching the directory
/// itself is enough for ordinary entries; a link, including a Windows junction, can land anywhere
/// inside one.
fn is_excluded_dir(exclude_dirs: &[PathBuf], entry: &walkdir::DirEntry) -> bool {
    if exclude_dirs.is_empty() {
        return false;
    }
    let Some(path) = entry.path().to_str().map(Path::new) else {
        return false;
    };
    let trim = |path: &Path| path.as_str().trim_end_matches(['/', '\\']).to_owned();
    let is_or_inside = |path: &Path| {
        exclude_dirs
            .iter()
            .any(|dir| trim(dir) == trim(path) || PathScope::root(dir).contains(path))
    };
    if is_or_inside(path) {
        return true;
    }
    entry.path_is_symlink()
        && entry.file_type().is_dir()
        && canonicalize_path(path).is_ok_and(|target| is_or_inside(&target))
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
    fn catalog_reports_only_videos_while_their_metadata_is_probed() -> Result<()> {
        let temp = TempDir::new()?;
        let root = canonicalize_path(&PathBuf::try_from(temp.path().to_owned())?)?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        let image = root.join("image.png");
        let video = root.join("video.mp4");
        std::fs::write(&image, b"image")?;
        std::fs::write(&video, b"video")?;
        let observer = EventObserver::default();

        catalog_snapshot(&assets, &root, IndexOptions::default(), &observer)?;
        let active = observer
            .0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                IndexEvent::ActiveAsset { path, active } => Some((path.clone(), *active)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(active, [(video.clone(), true), (video, false)]);

        observer.0.lock().unwrap().clear();
        catalog_snapshot(&assets, &root, IndexOptions::default(), &observer)?;
        assert!(
            observer
                .0
                .lock()
                .unwrap()
                .iter()
                .all(|event| { !matches!(event, IndexEvent::ActiveAsset { .. }) })
        );
        Ok(())
    }

    #[test]
    fn deferred_video_catalog_preserves_scan_order() -> Result<()> {
        let temp = TempDir::new()?;
        let root = canonicalize_path(&PathBuf::try_from(temp.path().to_owned())?)?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        let video = root.join("a-video.mp4");
        let photo = root.join("z-photo.png");
        std::fs::write(&video, b"video")?;
        std::fs::write(&photo, b"photo")?;

        let (cataloged, summary) = catalog_snapshot(
            &assets,
            &root,
            IndexOptions::default(),
            &EventObserver::default(),
        )?;
        assert!(summary.scan_complete);
        assert_eq!(
            cataloged
                .iter()
                .map(|asset| &asset.path)
                .collect::<Vec<_>>(),
            [&video, &photo]
        );
        Ok(())
    }

    #[test]
    fn video_probe_backlog_refills_past_512_candidates() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_owned())?;
        let candidates: Vec<_> = (0..=CATALOG_VIDEO_PROBE_BACKLOG)
            .map(|index| root.join(format!("missing-{index}.mp4")))
            .collect();
        let (sender, receiver) = crossbeam_channel::unbounded();

        run_catalog_video_probes(
            &EventObserver::default(),
            candidates.clone(),
            sender,
            Arc::new(AtomicBool::new(false)),
        )?;
        let completed: HashSet<_> = receiver.iter().map(|(path, _, _)| path).collect();
        assert_eq!(completed, candidates.into_iter().collect());
        Ok(())
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

    /// A directory link that needs no privileges: a junction on Windows, a symlink elsewhere.
    fn link_dir(target: &Path, link: &Path) -> Result<()> {
        #[cfg(windows)]
        {
            let status = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J", link.as_str(), target.as_str()])
                .stdout(std::process::Stdio::null())
                .status()?;
            anyhow::ensure!(status.success(), "mklink /J failed");
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, link)?;
        Ok(())
    }

    #[test]
    fn a_link_into_an_excluded_folder_is_not_walked() -> Result<()> {
        let temp = TempDir::new()?;
        let root = canonicalize_path(&PathBuf::try_from(temp.path().to_owned())?)?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        let library = root.join("library");
        let private = library.join("private");
        std::fs::create_dir_all(private.join("nested"))?;
        std::fs::write(library.join("kept.png"), b"image")?;
        std::fs::write(private.join("hidden.png"), b"image")?;
        std::fs::write(private.join("nested").join("deeper.png"), b"image")?;
        link_dir(&private, &library.join("alias"))?;
        link_dir(&private.join("nested"), &library.join("nested-alias"))?;

        let observer = TestObserver { cancelled: false };
        let options = IndexOptions {
            exclude_dirs: vec![private],
            ..IndexOptions::default()
        };
        let files = CatalogPipeline {
            assets: &assets,
            options: &options,
            observer: &observer,
        }
        .collect_files(&library, ScanSelection::All)?
        .paths;
        assert_eq!(files, [library.join("kept.png")]);
        Ok(())
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
        .collect_files(&root, ScanSelection::All)?
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
    fn delta_catalog_classifies_new_changed_unchanged_and_unseen_paths() -> Result<()> {
        let temp = TempDir::new()?;
        let root = canonicalize_path(&PathBuf::try_from(temp.path().to_owned())?)?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        let existing = root.join("existing.png");
        std::fs::write(&existing, b"original")?;
        let original = assets.upsert(&existing, &existing.metadata()?)?;
        std::fs::write(&existing, b"changed content")?;
        let unchanged = root.join("unchanged.png");
        std::fs::write(&unchanged, b"unchanged")?;
        assets.upsert(&unchanged, &unchanged.metadata()?)?;
        let missing = root.join("missing.png");
        std::fs::write(&missing, b"removed")?;
        assets.upsert(&missing, &missing.metadata()?)?;
        std::fs::remove_file(&missing)?;
        let new_path = root.join("new.png");
        std::fs::write(&new_path, b"new")?;

        let observer = TestObserver { cancelled: false };
        let delta = catalog_delta_dir_observed(&assets, &root, IndexOptions::default(), &observer)?;
        let mut passed = delta
            .cataloged
            .iter()
            .map(|asset| asset.path.clone())
            .collect::<Vec<_>>();
        passed.sort();
        assert_eq!(passed, vec![existing.clone(), new_path]);
        assert_eq!(delta.unseen_paths, vec![missing]);
        assert!(delta.scan_complete);
        assert_ne!(
            assets.get_by_path(&existing)?.unwrap().fingerprint,
            original.fingerprint
        );
        assert!(assets.get_by_path(&unchanged)?.is_some());
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
        assert!(scan.collect_files(&root, ScanSelection::All)?.complete);
        assert!(
            !scan
                .collect_files(&root.join("unavailable"), ScanSelection::All)?
                .complete
        );
        std::fs::write(root.join("a.png"), b"a")?;
        std::fs::write(root.join("b.png"), b"b")?;
        // A limit the walk never reaches leaves it complete; reaching it with a file left does not.
        for (limit, complete) in [(100, true), (2, true), (1, false)] {
            let limited = IndexOptions {
                limit: Some(limit),
                ..IndexOptions::default()
            };
            let files = CatalogPipeline {
                options: &limited,
                ..scan
            }
            .collect_files(&root, ScanSelection::All)?;
            assert_eq!(files.complete, complete, "limit {limit}");
            assert_eq!(files.paths.len(), limit.min(2), "limit {limit}");
        }
        let cancelled = TestObserver { cancelled: true };
        let summary = catalog_dir_observed(&assets, &root, options, &cancelled)?;
        assert!(summary.cancelled);
        assert!(!summary.scan_complete);
        Ok(())
    }

    #[test]
    fn incomplete_delta_does_not_offer_paths_for_deletion() -> Result<()> {
        let temp = TempDir::new()?;
        let root = canonicalize_path(&PathBuf::try_from(temp.path().to_owned())?)?;
        let assets = AssetCatalog::new(&root.join("assets.db"))?;
        let missing = root.join("missing.png");
        std::fs::write(&missing, b"removed")?;
        assets.upsert(&missing, &missing.metadata()?)?;
        std::fs::remove_file(&missing)?;
        std::fs::write(root.join("a.png"), b"a")?;
        std::fs::write(root.join("b.png"), b"b")?;
        let observer = TestObserver { cancelled: false };
        let options = IndexOptions {
            limit: Some(1),
            ..IndexOptions::default()
        };
        let delta = catalog_delta_dir_observed(&assets, &root, options, &observer)?;
        assert!(!delta.scan_complete);
        assert!(delta.unseen_paths.is_empty());
        assert!(assets.get_by_path(&missing)?.is_some());
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
