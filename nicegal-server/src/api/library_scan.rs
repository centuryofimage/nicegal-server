//! `libraryScan`: catalog a library's folders, then bring its enabled search indexes up to date.
//!
//! The library's stored options decide which indexes run, so a client saves OCR and image choices
//! with `PUT /v1/libraries/<id>` before scanning. Each model is prepared only when the scan finds
//! work for it, including the OCR pair, which the request names and the job loads itself.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use camino::Utf8PathBuf as PathBuf;
use glob::Pattern;
use nicegal_core::assets::{Asset, AssetCatalog, canonicalize_path};
use nicegal_core::db::DB;
use nicegal_core::index::{self, IndexObserver, IndexOptions};
use nicegal_core::libraries::{DirectorySnapshot, IncludedFolder, Library, ScanOutcome};
use nicegal_core::scope::PathScope;
use nicegal_core::thumbs::ThumbnailService;
use serde::Deserialize;
use tokio::runtime::Handle;
use walkdir::WalkDir;

use super::error::ApiError;
use super::jobs::{FolderState, IndexStages, Job, JobCancelled};
use super::models::{ImageModel, TextModel};
use super::prune_jobs::{self, ReconcileInput, ReconcileScope};
use super::{Databases, RuntimeSettings, image_embeddings, ocr_models, text_embeddings};

const OCR_COMMIT_CHUNK_SIZE: usize = 32;
const FULL_SCAN_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// Cache folders that never hold library media.
const DEFAULT_EXCLUDES: [&str; 2] = ["*/.cache", "*/.thumb*"];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ScanMode {
    #[default]
    Full,
    Fast,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    library_id: i64,
    /// The OCR pair to recognize with. Loaded only when the scan finds images that need text
    /// recognition and a different pair (or none) is loaded.
    ocr_models: Option<ocr_models::job::Request>,
    /// Scan only the folders a library edit marked pending.
    #[serde(default)]
    pending_only: bool,
    #[serde(default)]
    scan_mode: ScanMode,
    /// Re-run OCR for images that already have current text.
    #[serde(default)]
    force: bool,
    /// Retry sources whose previous decode failed.
    #[serde(default)]
    retry_failed: bool,
    /// Skip OCR for images larger than this.
    max_dimensions: Option<MaxDimensions>,
    /// Debug guardrail: stop each folder's walk after this many files.
    debug_limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MaxDimensions {
    width: usize,
    height: usize,
}

pub(crate) struct Spec {
    library_id: i64,
    ocr_models: Option<ocr_models::job::Spec>,
    pending_only: bool,
    scan_mode: ScanMode,
    force: bool,
    retry_failed: bool,
    max_dimensions: Option<(usize, usize)>,
    debug_limit: Option<usize>,
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    if request.debug_limit == Some(0) {
        return Err(ApiError::bad_request(
            "debugLimit must be greater than zero",
        ));
    }
    let max_dimensions = request
        .max_dimensions
        .map(|dimensions| {
            if dimensions.width == 0 || dimensions.height == 0 {
                return Err(ApiError::bad_request(
                    "maximum width and height must be greater than zero",
                ));
            }
            Ok((dimensions.width, dimensions.height))
        })
        .transpose()?;
    Ok(Spec {
        library_id: request.library_id,
        ocr_models: request
            .ocr_models
            .map(ocr_models::job::prepare)
            .transpose()?,
        pending_only: request.pending_only,
        scan_mode: request.scan_mode,
        force: request.force,
        retry_failed: request.retry_failed,
        max_dimensions,
        debug_limit: request.debug_limit,
    })
}

impl Spec {
    pub(crate) fn library_id(&self) -> i64 {
        self.library_id
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.pending_only &= other.pending_only;
        if other.scan_mode == ScanMode::Full {
            self.scan_mode = ScanMode::Full;
        }
        self.force |= other.force;
        self.retry_failed |= other.retry_failed;
        if other.ocr_models.is_some() {
            self.ocr_models = other.ocr_models.clone();
        }
        if other.max_dimensions.is_some() {
            self.max_dimensions = other.max_dimensions;
        }
        if other.debug_limit.is_some() {
            self.debug_limit = other.debug_limit;
        }
    }

    pub(crate) fn apply_settings(&mut self, runtime: &RuntimeSettings) -> anyhow::Result<()> {
        match &self.ocr_models {
            Some(models) => runtime.save_ocr_models(models)?,
            None => self.ocr_models = Some(runtime.ocr_models()?),
        }
        Ok(())
    }

    /// Everything this scan would walk, recognize with, and index, fixed when the job is accepted.
    fn index_options(&self, exclude_dirs: &[PathBuf]) -> IndexOptions {
        IndexOptions {
            limit: self.debug_limit,
            exclude: DEFAULT_EXCLUDES
                .iter()
                .map(|pattern| Pattern::new(pattern).expect("default exclusions are valid"))
                .collect(),
            exclude_dirs: exclude_dirs.to_vec(),
            rescan: self.force,
            retry_failed: self.retry_failed,
            subdirs: true,
            commit_chunk_size: OCR_COMMIT_CHUNK_SIZE,
            max_dimensions: self.max_dimensions,
            ..IndexOptions::default()
        }
    }
}

/// Shared services a scan draws on; owned by the job manager.
pub(super) struct Services<'a> {
    pub(super) databases: &'a Databases,
    pub(super) thumbnails: &'a ThumbnailService,
    pub(super) text_embedder: &'a TextModel,
    pub(super) image_embedder: &'a ImageModel,
    pub(super) ocr_store: &'a ocr_models::ModelStore,
    pub(super) runtime: &'a Arc<RuntimeSettings>,
    /// Drives the async OCR model download from this blocking worker.
    pub(super) handle: Handle,
}

/// One walk of the scan: an included folder, plus any included folders nested inside it that
/// the walk also covers.
struct ScanFolder<'a> {
    folder: &'a IncludedFolder,
    covered: Vec<&'a IncludedFolder>,
}

#[tracing::instrument(
    level = "debug",
    skip(spec, services, job),
    fields(library_id = spec.library_id, scan_mode = ?spec.scan_mode, pending_only = spec.pending_only)
)]
pub(super) fn run(mut spec: Spec, services: &Services<'_>, job: &Arc<Job>) -> anyhow::Result<()> {
    let mut ocr_request = spec.ocr_models.take();
    let mut catalog = AssetCatalog::new(&services.databases.assets)?;
    let library = respell_reachable(&mut catalog, spec.library_id)?;
    job.set_index_stages(IndexStages {
        ocr: library.options.ocr,
        image: library.options.image,
        text: library.options.ocr,
    });
    let folders = scan_folders(&library, spec.pending_only);
    anyhow::ensure!(
        spec.pending_only || library.include.is_empty() || !folders.is_empty(),
        "scan selected no folders from a library with included folders"
    );
    tracing::debug!(walks = folders.len(), "selected scan folders");
    job.set_scan_requests(
        folders
            .iter()
            .flat_map(|walk| {
                walk.folders()
                    .map(|folder| (folder.path.clone(), folder.scan_request))
            })
            .collect(),
    );
    job.set_folders(
        folders
            .iter()
            .map(|walk| walk.folder.path.clone())
            .collect(),
    );

    let options = || spec.index_options(&library.exclude);
    let mut scanned = Vec::new();
    let mut complete = Vec::new();
    for (index, walk) in folders.iter().enumerate() {
        let path = &walk.folder.path;
        job.enter_folder(index);
        if let Err(error) = folder_available(path) {
            tracing::warn!(folder = %path, error = %format_args!("{error:#}"), "scan folder unavailable");
            let message = format!("{error:#}");
            job.finish_folder(index, FolderState::Unavailable, Some(message.clone()));
            for folder in walk.folders() {
                catalog.fail_folder_scan(
                    library.id,
                    &folder.path,
                    ScanOutcome::Unavailable,
                    &message,
                )?;
            }
            continue;
        }
        let scope_key = serde_json::to_string(&library.exclude)?;
        let previous = catalog.directory_snapshot(library.id, path, &scope_key)?;
        let now_ns = now_ns();
        for folder in walk.folders() {
            tracing::trace!(
                folder = %folder.path,
                scan_pending = folder.scan_pending,
                scan_outcome = ?folder.scan_outcome,
                last_scan_completed_ns = ?folder.last_scan_completed_ns,
                now_ns,
                "full scan clock"
            );
        }
        let full_due = full_scan_due(walk.folders(), now_ns);
        let snapshot_valid = previous.first().is_some_and(|entry| entry.path == *path);
        let full = spec.scan_mode == ScanMode::Full
            || full_due
            || spec.debug_limit.is_some()
            || !snapshot_valid;
        tracing::debug!(
            folder = %path,
            mode = if full { "full" } else { "fast" },
            requested_full = spec.scan_mode == ScanMode::Full,
            full_due,
            debug_limit = ?spec.debug_limit,
            snapshot_valid,
            snapshot_directories = previous.len(),
            "selected folder scan mode"
        );
        job.set_folder_scan_mode(index, if full { "full" } else { "fast" });
        let before = if full {
            match snapshot_directories(path, &options(), job.as_ref()) {
                Ok(snapshot) => {
                    tracing::debug!(folder = %path, directories = snapshot.len(), "captured directory snapshot");
                    Some(snapshot)
                }
                Err(error) => {
                    tracing::debug!(folder = %path, error = %format_args!("{error:#}"), "directory snapshot unavailable");
                    None
                }
            }
        } else {
            None
        };
        let (assets, summary, next_snapshot) = if full {
            let (assets, summary) =
                index::catalog_snapshot(&catalog, path, options(), job.as_ref())?;
            (assets, summary, before)
        } else {
            let checked = quick_catalog(
                &catalog,
                path,
                &previous,
                &options(),
                job.as_ref(),
                services,
            );
            match checked {
                Ok((assets, summary, snapshot)) => (assets, summary, Some(snapshot)),
                Err(error) if super::jobs::is_cancelled(&error) => {
                    job.finish_folder(index, FolderState::Cancelled, None);
                    return Err(error);
                }
                Err(error) => {
                    tracing::warn!(folder = %path, error = %format_args!("{error:#}"), "fast directory check failed");
                    let message = format!("directory check failed: {error:#}");
                    job.finish_folder(index, FolderState::Failed, Some(message.clone()));
                    for folder in walk.folders() {
                        catalog.fail_folder_scan(
                            library.id,
                            &folder.path,
                            ScanOutcome::Failed,
                            &message,
                        )?;
                    }
                    continue;
                }
            }
        };
        if summary.cancelled {
            job.finish_folder(index, FolderState::Cancelled, None);
            return Err(JobCancelled.into());
        }
        tracing::debug!(folder = %path, mode = if full { "full" } else { "fast" }, assets = assets.len(), scan_complete = summary.scan_complete, "folder catalog finished");
        if summary.scan_complete {
            // Missing-file cleanup only follows a complete walk, and only inside this folder.
            let reconciled = if full {
                prune_jobs::reconcile(
                    ReconcileScope::new(path.clone(), &options()),
                    services.databases,
                    services.image_embedder.dimensions(),
                    services.thumbnails,
                    ReconcileInput::Scanned(&assets),
                    job.as_ref(),
                )
            } else {
                Ok(())
            };
            match reconciled {
                Ok(()) => {
                    job.finish_folder(index, FolderState::Scanned, None);
                    complete.push((index, full, next_snapshot, scope_key));
                }
                Err(error) if super::jobs::is_cancelled(&error) => {
                    job.finish_folder(index, FolderState::Cancelled, None);
                    return Err(error);
                }
                Err(error) => {
                    let message = format!("cleanup failed: {error:#}");
                    job.finish_folder(index, FolderState::Failed, Some(message.clone()));
                    for folder in walk.folders() {
                        catalog.fail_folder_scan(
                            library.id,
                            &folder.path,
                            ScanOutcome::Failed,
                            &message,
                        )?;
                    }
                }
            }
        } else {
            let message = if spec.debug_limit.is_some() {
                "stopped at the debug limit, or some files or folders could not be read"
            } else {
                "some files or folders could not be read; see the job errors"
            };
            job.finish_folder(index, FolderState::Incomplete, Some(message.to_owned()));
            for folder in walk.folders() {
                catalog.fail_folder_scan(
                    library.id,
                    &folder.path,
                    ScanOutcome::Incomplete,
                    message,
                )?;
            }
        }
        scanned.extend(assets);
    }
    job.leave_folder();

    let scope = library.scope();
    if library.options.image {
        embed_images(
            &spec,
            &scope,
            library.options.videos,
            &scanned,
            services,
            job,
        )?;
    }
    if library.options.ocr {
        recognize_text(
            &mut ocr_request,
            &catalog,
            &scanned,
            &options(),
            services,
            job,
        )?;
        embed_text(&spec, &scope, services, job)?;
    }

    // Folders count as scanned only once every enabled index has caught up with them.
    let completed_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX)
        });
    for (index, full, snapshot, scope_key) in complete {
        // An unsuccessful capture must erase any old snapshot; otherwise this new full-scan
        // clock could make a stale snapshot look current on the next automatic request.
        if full || snapshot.is_some() {
            catalog.replace_directory_snapshot(
                library.id,
                &folders[index].folder.path,
                &scope_key,
                snapshot.as_deref().unwrap_or(&[]),
            )?;
        }
        for folder in folders[index].folders() {
            if full {
                catalog.complete_folder_scan(
                    library.id,
                    &folder.path,
                    folder.scan_request,
                    completed_ns,
                )?;
            } else {
                catalog.complete_folder_check(library.id, &folder.path, folder.scan_request)?;
            }
        }
        job.finish_folder(index, FolderState::Completed, None);
    }
    Ok(())
}

impl<'a> ScanFolder<'a> {
    fn folders(&self) -> impl Iterator<Item = &'a IncludedFolder> + '_ {
        std::iter::once(self.folder).chain(self.covered.iter().copied())
    }
}

/// The walks a scan performs: the selected included folders, minus any nested inside another
/// selected folder, which that folder's walk already covers.
fn scan_folders(library: &Library, pending_only: bool) -> Vec<ScanFolder<'_>> {
    let selected = library
        .include
        .iter()
        .filter(|folder| !pending_only || folder.scan_pending)
        .collect::<Vec<_>>();
    let outer = selected
        .iter()
        .filter(|folder| {
            !selected.iter().any(|other| {
                other.path != folder.path && PathScope::root(&other.path).contains(&folder.path)
            })
        })
        .copied()
        .collect::<Vec<_>>();
    outer
        .iter()
        .map(|folder| ScanFolder {
            folder,
            covered: selected
                .iter()
                .filter(|other| {
                    other.path != folder.path && PathScope::root(&folder.path).contains(&other.path)
                })
                .copied()
                .collect(),
        })
        .collect()
}

/// The library with every reachable folder under its canonical spelling. A folder saved while
/// offline keeps the spelling it was given, which can differ in case or separators from the
/// canonical paths the catalog records, and then the library would show none of its files.
fn respell_reachable(catalog: &mut AssetCatalog, library_id: i64) -> anyhow::Result<Library> {
    let deleted = || format!("library {library_id} was deleted");
    let library = catalog.library(library_id)?.with_context(deleted)?;
    let definition = library.definition();
    let spellings = definition
        .include
        .iter()
        .chain(&definition.exclude)
        .filter(|path| fs::metadata(path).is_ok_and(|metadata| metadata.is_dir()))
        .filter_map(|path| {
            let canonical = canonicalize_path(path).ok()?;
            // As strings: `Path` equality ignores `.` components and separator style.
            (canonical.as_str() != path.as_str()).then(|| (path.clone(), canonical))
        })
        .collect::<Vec<_>>();
    if spellings.is_empty() {
        return Ok(library);
    }
    catalog
        .respell_folders(library_id, &spellings)?
        .with_context(deleted)
}

/// A folder must be a readable directory right now; an unplugged drive is never mistaken for an
/// empty folder.
fn folder_available(path: &camino::Utf8Path) -> anyhow::Result<()> {
    let metadata = fs::metadata(path).with_context(|| format!("cannot read folder {path}"))?;
    if !metadata.is_dir() {
        anyhow::bail!("{path} is not a folder");
    }
    fs::read_dir(path).with_context(|| format!("cannot list folder {path}"))?;
    Ok(())
}

fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX)
        })
}

fn full_scan_due<'a>(folders: impl Iterator<Item = &'a IncludedFolder>, now_ns: i64) -> bool {
    folders.into_iter().any(|folder| {
        folder.scan_pending
            || folder.scan_outcome.is_some()
            || folder.last_scan_completed_ns.is_none_or(|last| {
                last <= 0 || now_ns.saturating_sub(last) >= FULL_SCAN_INTERVAL.as_nanos() as i64
            })
    })
}

fn directory_mtime(path: &camino::Utf8Path) -> anyhow::Result<i64> {
    let modified = fs::metadata(path)?.modified()?.duration_since(UNIX_EPOCH)?;
    Ok(i64::try_from(modified.as_nanos())?)
}

fn excluded_directory(path: &camino::Utf8Path, options: &IndexOptions) -> bool {
    options
        .exclude_dirs
        .iter()
        .any(|excluded| path == excluded || PathScope::root(excluded).contains(path))
        || path.ancestors().any(|ancestor| {
            options
                .exclude
                .iter()
                .any(|pattern| pattern.matches_path(ancestor.as_std_path()))
        })
}

/// Capture directory mtimes before cataloging. A failed or unsupported walk leaves no trusted
/// snapshot; the next automatic request then takes the full path again.
fn snapshot_directories(
    root: &camino::Utf8Path,
    options: &IndexOptions,
    job: &Job,
) -> anyhow::Result<Vec<DirectorySnapshot>> {
    let mut directories = Vec::new();
    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|entry| {
            entry
                .path()
                .to_str()
                .is_some_and(|path| !excluded_directory(camino::Utf8Path::new(path), options))
        })
    {
        job.check_cancelled()?;
        let entry = entry?;
        if entry.file_type().is_dir() {
            let path = PathBuf::try_from(entry.path().to_owned())?;
            // A linked directory can be retargeted without changing its parent mtime. Full
            // scanning it remains safe until a link-aware snapshot is available.
            if entry.path_is_symlink() {
                anyhow::bail!("linked directory has no stable snapshot");
            }
            directories.push(DirectorySnapshot {
                modified_ns: directory_mtime(&path)?,
                path,
            });
        }
    }
    directories.sort_by(|a, b| a.path.cmp(&b.path));
    anyhow::ensure!(
        directories.first().is_some_and(|entry| entry.path == root),
        "missing root snapshot"
    );
    Ok(directories)
}

fn shallow_options(options: &IndexOptions) -> IndexOptions {
    IndexOptions {
        limit: options.limit,
        exclude: options.exclude.clone(),
        exclude_dirs: options.exclude_dirs.clone(),
        rescan: options.rescan,
        retry_failed: options.retry_failed,
        subdirs: false,
        ..IndexOptions::default()
    }
}

/// Stat known directories, list only changed ones, and scan new child subtrees. Deletions are
/// reconciled only after every required listing and catalog operation completed successfully.
#[tracing::instrument(level = "trace", skip(catalog, previous, options, job, services), fields(folder = %root, known_directories = previous.len()))]
fn quick_catalog(
    catalog: &AssetCatalog,
    root: &camino::Utf8Path,
    previous: &[DirectorySnapshot],
    options: &IndexOptions,
    job: &Job,
    services: &Services<'_>,
) -> anyhow::Result<(Vec<Asset>, index::CatalogSummary, Vec<DirectorySnapshot>)> {
    let known: HashMap<_, _> = previous
        .iter()
        .map(|entry| (entry.path.clone(), entry.modified_ns))
        .collect();
    let mut next = previous.to_vec();
    let mut scanned = Vec::new();
    let mut candidates = Vec::new();
    let mut complete = true;
    let mut missing = Vec::<PathBuf>::new();
    let mut new_subtrees = HashSet::<PathBuf>::new();
    let mut unchanged_count = 0_usize;
    let mut changed_count = 0_usize;

    for entry in previous {
        job.check_cancelled()?;
        if missing
            .iter()
            .any(|parent| PathScope::root(parent).contains(&entry.path))
        {
            tracing::trace!(directory = %entry.path, "skipping directory under missing parent");
            continue;
        }
        let current = match directory_mtime(&entry.path) {
            Ok(value) => value,
            Err(_)
                if fs::metadata(&entry.path)
                    .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
            {
                tracing::trace!(directory = %entry.path, previous_mtime_ns = entry.modified_ns, "directory missing");
                missing.push(entry.path.clone());
                candidates.extend(catalog.under_root(&entry.path)?);
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("checking directory {}", entry.path));
            }
        };
        tracing::trace!(
            directory = %entry.path,
            previous_mtime_ns = entry.modified_ns,
            current_mtime_ns = current,
            changed = current != entry.modified_ns,
            "compared directory mtime"
        );
        if current == entry.modified_ns {
            unchanged_count += 1;
            continue;
        }
        changed_count += 1;
        let children = fs::read_dir(&entry.path)
            .with_context(|| format!("listing changed directory {}", entry.path))?;
        for child in children {
            job.check_cancelled()?;
            let child = child?;
            let path = PathBuf::try_from(child.path())?;
            if excluded_directory(&path, options) {
                continue;
            }
            let metadata = child.metadata()?;
            if metadata.is_dir() && !known.contains_key(&path) {
                tracing::trace!(directory = %path, parent = %entry.path, "found new directory subtree");
                new_subtrees.insert(path);
            }
        }
        let (found, summary) =
            index::catalog_snapshot(catalog, &entry.path, shallow_options(options), job)?;
        tracing::trace!(directory = %entry.path, assets = found.len(), scan_complete = summary.scan_complete, "cataloged changed directory");
        complete &= summary.scan_complete;
        let seen: HashSet<_> = found.iter().map(|asset| &asset.path).collect();
        candidates.extend(
            catalog
                .under_root(&entry.path)?
                .into_iter()
                .filter(|asset| {
                    asset.path.parent() == Some(entry.path.as_path()) && !seen.contains(&asset.path)
                }),
        );
        scanned.extend(found);
        if let Some(stored) = next.iter_mut().find(|stored| stored.path == entry.path) {
            stored.modified_ns = current;
        }
    }
    next.retain(|entry| {
        !missing
            .iter()
            .any(|path| entry.path == *path || PathScope::root(path).contains(&entry.path))
    });
    let mut new_subtrees: Vec<_> = new_subtrees.into_iter().collect();
    new_subtrees.sort();
    for path in new_subtrees {
        job.check_cancelled()?;
        if next.iter().any(|entry| entry.path == path)
            || next.iter().any(|entry| {
                !known.contains_key(&entry.path) && PathScope::root(&entry.path).contains(&path)
            })
        {
            continue;
        }
        let snapshot = snapshot_directories(&path, options, job)?;
        let (found, summary) =
            index::catalog_snapshot(catalog, &path, shallow_options_recursive(options), job)?;
        tracing::trace!(directory = %path, directories = snapshot.len(), assets = found.len(), scan_complete = summary.scan_complete, "cataloged new directory subtree");
        complete &= summary.scan_complete;
        let seen: HashSet<_> = found.iter().map(|asset| &asset.path).collect();
        candidates.extend(
            catalog
                .under_root(&path)?
                .into_iter()
                .filter(|asset| !seen.contains(&asset.path)),
        );
        next.extend(snapshot);
        scanned.extend(found);
    }
    if complete && !job.is_cancelled() {
        candidates.sort_by_key(|asset| asset.asset_id);
        candidates.dedup_by_key(|asset| asset.asset_id);
        tracing::trace!(
            candidate_assets = candidates.len(),
            "reconciling fast scan candidates"
        );
        prune_jobs::reconcile(
            ReconcileScope::new(root.to_owned(), options),
            services.databases,
            services.image_embedder.dimensions(),
            services.thumbnails,
            ReconcileInput::Candidates(&candidates),
            job,
        )?;
    }
    tracing::debug!(
        unchanged_directories = unchanged_count,
        changed_directories = changed_count,
        missing_directories = missing.len(),
        scanned_assets = scanned.len(),
        scan_complete = complete && !job.is_cancelled(),
        "fast directory check finished"
    );
    next.sort_by(|a, b| a.path.cmp(&b.path));
    Ok((
        scanned,
        index::CatalogSummary {
            cataloged: 0,
            cancelled: job.is_cancelled(),
            scan_complete: complete && !job.is_cancelled(),
        },
        next,
    ))
}

fn shallow_options_recursive(options: &IndexOptions) -> IndexOptions {
    let mut copied = shallow_options(options);
    copied.subdirs = true;
    copied
}

fn embed_images(
    spec: &Spec,
    scope: &PathScope,
    index_videos: bool,
    scanned: &[Asset],
    services: &Services<'_>,
    job: &Arc<Job>,
) -> anyhow::Result<()> {
    let image_spec = image_embeddings::Spec::pending_for(
        scope.clone(),
        spec.retry_failed,
        spec.debug_limit,
        index_videos,
    );
    if !image_embeddings::has_pending(
        &image_spec,
        services.databases,
        services.image_embedder.dimensions(),
        Some(scanned),
    )? {
        return Ok(());
    }
    job.preparing_models(1);
    let model = services
        .image_embedder
        .prepare_with_progress(job.as_ref())?;
    job.models_loaded(1);
    job.check_cancelled()?;
    image_embeddings::run(
        image_spec,
        services.databases,
        services.thumbnails,
        model.as_ref(),
        job.as_ref(),
        Some(scanned),
    )
}

fn recognize_text(
    ocr_request: &mut Option<ocr_models::job::Spec>,
    catalog: &AssetCatalog,
    scanned: &[Asset],
    options: &IndexOptions,
    services: &Services<'_>,
    job: &Arc<Job>,
) -> anyhow::Result<()> {
    let mut ocr = DB::new(&services.databases.ocr)?;
    let selection = index::select_ocr_sources(catalog, &ocr, scanned, options, job.as_ref())?;
    job.check_cancelled()?;
    if selection.is_empty() {
        return Ok(());
    }
    let models = ocr_models(ocr_request, services, job)?;
    let mut models = models.lock().unwrap_or_else(|error| error.into_inner());
    index::recognize_selection(
        catalog,
        &mut ocr,
        &mut models,
        selection,
        options,
        job.as_ref(),
    )?;
    job.check_cancelled()
}

/// The requested OCR pair, loading it only when it is not the pair already in memory. Without a
/// requested pair, whatever is loaded is used.
fn ocr_models(
    ocr_request: &mut Option<ocr_models::job::Spec>,
    services: &Services<'_>,
    job: &Arc<Job>,
) -> anyhow::Result<Arc<std::sync::Mutex<nicegal_core::ocr::PaddleOcrPool>>> {
    if let Some(requested) = ocr_request.take()
        && !requested.is_loaded_in(services.ocr_store)
    {
        services.handle.block_on(ocr_models::job::run(
            requested,
            services.ocr_store,
            services.runtime,
            Arc::clone(job),
        ))?;
    }
    services.ocr_store.snapshot().context(
        "this library recognizes text but no OCR models are loaded; include ocrModels in the scan",
    )
}

fn embed_text(
    spec: &Spec,
    scope: &PathScope,
    services: &Services<'_>,
    job: &Arc<Job>,
) -> anyhow::Result<()> {
    let text_spec = text_embeddings::job::Spec::pending_for(scope.clone(), spec.debug_limit);
    if !text_embeddings::job::has_pending(
        &text_spec,
        &services.databases.ocr,
        services.text_embedder.model().id(),
        services.text_embedder.dimensions(),
    )? {
        return Ok(());
    }
    job.preparing_models(1);
    let model = services.text_embedder.prepare_with_progress(job.as_ref())?;
    job.models_loaded(1);
    job.check_cancelled()?;
    text_embeddings::job::run(
        text_spec,
        &services.databases.ocr,
        model.as_ref(),
        job.as_ref(),
    )
}

#[cfg(test)]
mod tests {
    use nicegal_core::libraries::LibraryOptions;

    use super::*;

    fn folder(path: &str, scan_pending: bool) -> IncludedFolder {
        IncludedFolder {
            path: PathBuf::from(path),
            scan_pending,
            scan_outcome: None,
            scan_error: None,
            last_scan_completed_ns: None,
            scan_request: 1,
        }
    }

    fn walks(library: &Library, pending_only: bool) -> Vec<(String, Vec<String>)> {
        scan_folders(library, pending_only)
            .iter()
            .map(|walk| {
                (
                    walk.folder.path.to_string(),
                    walk.covered
                        .iter()
                        .map(|folder| folder.path.to_string())
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn nested_folders_share_one_walk_and_pending_only_narrows_it() {
        let library = Library {
            id: 1,
            include: vec![
                folder("/a/nested", true),
                folder("/a", false),
                folder("/b", true),
            ],
            exclude: Vec::new(),
            options: LibraryOptions::default(),
        };
        assert_eq!(
            walks(&library, false),
            [
                ("/a".to_owned(), vec!["/a/nested".to_owned()]),
                ("/b".to_owned(), vec![]),
            ]
        );
        // Only the nested folder is pending, so it is walked on its own.
        assert_eq!(
            walks(&library, true),
            [("/a/nested".to_owned(), vec![]), ("/b".to_owned(), vec![])]
        );
    }

    #[test]
    fn filesystem_root_is_not_treated_as_its_own_child() {
        let library = Library {
            id: 1,
            include: vec![folder("/", true), folder("/nested", true)],
            exclude: Vec::new(),
            options: LibraryOptions::default(),
        };
        assert_eq!(
            walks(&library, false),
            [("/".to_owned(), vec!["/nested".to_owned()])]
        );
        assert_eq!(
            walks(&library, true),
            [("/".to_owned(), vec!["/nested".to_owned()])]
        );
    }

    #[test]
    fn requests_need_a_library_and_reject_bad_limits() {
        let parse = |value: serde_json::Value| serde_json::from_value::<Request>(value);
        assert!(parse(serde_json::json!({})).is_err());
        assert!(parse(serde_json::json!({"libraryId": 1, "root": "/a"})).is_err());
        let spec = prepare(parse(serde_json::json!({"libraryId": 1})).unwrap()).unwrap();
        assert_eq!(spec.scan_mode, ScanMode::Full);
        assert_eq!(
            prepare(parse(serde_json::json!({"libraryId": 1, "scanMode": "fast"})).unwrap())
                .unwrap()
                .scan_mode,
            ScanMode::Fast
        );
        assert_eq!(
            prepare(parse(serde_json::json!({"libraryId": 1, "scanMode": "full"})).unwrap())
                .unwrap()
                .scan_mode,
            ScanMode::Full
        );
        assert!(parse(serde_json::json!({"libraryId": 1, "scanMode": "auto"})).is_err());
        assert!(!spec.pending_only && spec.ocr_models.is_none());
        for body in [
            serde_json::json!({"libraryId": 1, "debugLimit": 0}),
            serde_json::json!({"libraryId": 1, "maxDimensions": {"width": 0, "height": 5}}),
        ] {
            assert!(prepare(parse(body).unwrap()).is_err());
        }
    }

    fn request(value: serde_json::Value) -> Spec {
        prepare(serde_json::from_value(value).unwrap()).unwrap()
    }

    #[test]
    fn pending_scans_retry_folders_whose_last_scan_failed() {
        let mut failed = folder("/failed", true);
        failed.scan_outcome = Some(ScanOutcome::Unavailable);
        failed.scan_error = Some("unavailable".to_owned());
        let library = Library {
            id: 1,
            include: vec![failed, folder("/new", true), folder("/current", false)],
            exclude: Vec::new(),
            options: LibraryOptions::default(),
        };
        assert_eq!(scan_folders(&library, true).len(), 2);
        assert_eq!(scan_folders(&library, false).len(), 3);
    }

    #[test]
    fn merged_scan_requests_keep_full_walk_and_retry_intent() {
        let mut pending =
            request(serde_json::json!({ "libraryId": 4, "pendingOnly": true, "scanMode": "fast" }));
        pending.merge(&request(
            serde_json::json!({ "libraryId": 4, "retryFailed": true }),
        ));
        assert!(!pending.pending_only);
        assert!(pending.retry_failed);
        assert_eq!(pending.scan_mode, ScanMode::Full);
    }

    #[test]
    fn full_scans_are_due_on_a_rolling_24_hour_clock() {
        let interval = FULL_SCAN_INTERVAL.as_nanos() as i64;
        let now = interval * 10;
        let mut current = folder("/current", false);
        current.last_scan_completed_ns = Some(now - interval + 1);
        assert!(!full_scan_due(std::iter::once(&current), now));
        current.last_scan_completed_ns = Some(now - interval);
        assert!(full_scan_due(std::iter::once(&current), now));
        current.last_scan_completed_ns = None;
        assert!(full_scan_due(std::iter::once(&current), now));
    }

    #[test]
    fn directory_mtime_detects_new_entries_without_changing_untouched_parent() {
        let temp = tempfile::tempdir().unwrap();
        let root = PathBuf::try_from(temp.path().to_owned()).unwrap();
        let child = root.join("child");
        fs::create_dir(&child).unwrap();
        let root_snapshot = directory_mtime(&root).unwrap();
        let child_snapshot = directory_mtime(&child).unwrap();

        // An unchanged fast check can skip both listings.
        assert_eq!(directory_mtime(&root).unwrap(), root_snapshot);
        assert_eq!(directory_mtime(&child).unwrap(), child_snapshot);

        std::thread::sleep(Duration::from_millis(1100));
        fs::write(child.join("new.jpg"), b"image").unwrap();
        // Adding an entry changes its containing directory, but not the parent directory.
        assert_ne!(directory_mtime(&child).unwrap(), child_snapshot);
        assert_eq!(directory_mtime(&root).unwrap(), root_snapshot);
    }
}
