use std::fs;

use anyhow::Context;
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::db::DB;
use nicegal_core::image_index::ImageIndexDb;
use nicegal_core::index::{IndexEvent, IndexObserver, IndexOptions, IndexPhase, IndexProgressDelta};
use nicegal_core::thumbs::ThumbnailService;
use serde::Deserialize;

use super::error::ApiError;
use super::roots;

const PRUNE_BATCH_SIZE: usize = 128;

/// The precise scan scope, retained across the catalog-first indexing pipeline.
pub(super) struct ReconcileScope {
    root: PathBuf,
    recursive: bool,
    exclude: Vec<glob::Pattern>,
    limited: bool,
}

impl ReconcileScope {
    pub(super) fn new(root: PathBuf, options: &IndexOptions) -> Self {
        Self {
            root,
            recursive: options.subdirs,
            exclude: options.exclude.clone(),
            limited: options.limit.is_some(),
        }
    }

    fn includes(&self, path: &camino::Utf8Path) -> bool {
        if !path.starts_with(&self.root) || (!self.recursive && path.parent() != Some(&self.root)) {
            return false;
        }
        !path
            .ancestors()
            .take_while(|ancestor| ancestor.starts_with(&self.root))
            .any(|ancestor| {
                self.exclude
                    .iter()
                    .any(|pattern| pattern.matches_path(ancestor.as_std_path()))
            })
    }
}

/// Reconcile only after successful discovery. Confirm absence rather than treating unvisited
/// paths as deleted; excluded subtrees, unreadable folders, and disconnected roots are preserved.
pub(super) fn reconcile(
    mut scope: ReconcileScope,
    databases: &super::Databases,
    image_dimensions: usize,
    thumbnails: &ThumbnailService,
    observer: &dyn IndexObserver,
) -> anyhow::Result<bool> {
    if observer.is_cancelled() {
        return Ok(true);
    }
    if scope.limited {
        return Ok(false);
    }
    scope.root = nicegal_core::assets::canonicalize_path(&scope.root)?;
    ensure_root_available(&scope.root)?;
    let assets = AssetCatalog::new(&databases.assets)?;
    let candidates = assets.under_root(&scope.root)?;
    let mut missing = Vec::new();
    // Complete all filesystem checks before deleting any row. An inaccessible subtree must not
    // turn a partial check into a partial automatic purge.
    for asset in &candidates {
        if observer.is_cancelled() {
            return Ok(true);
        }
        if scope.includes(&asset.path) && confirmed_missing(&asset.path, &scope.root)? {
            missing.push(asset);
        }
    }
    if missing.is_empty() {
        return Ok(observer.is_cancelled());
    }
    let mut ocr = DB::new(&databases.ocr)?;
    let mut images = ImageIndexDb::new(&databases.images, image_dimensions)?;
    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::Pruning));
    observer.on_event(IndexEvent::DiscoveryComplete {
        total: missing.len(),
    });
    for chunk in missing.chunks(PRUNE_BATCH_SIZE) {
        if observer.is_cancelled() {
            return Ok(true);
        }
        ensure_root_available(&scope.root)?;
        // Files can reappear after discovery (for example during a move). Recheck each batch.
        let missing_now = chunk
            .iter()
            .copied()
            .filter_map(|asset| match confirmed_missing(&asset.path, &scope.root) {
                Ok(true) => Some(Ok(asset)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if observer.is_cancelled() {
            return Ok(true);
        }
        let deleted = delete_asset_batch(&missing_now, &assets, &mut ocr, &mut images, thumbnails)?;
        observer.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: chunk.len(),
            deleted,
            ..IndexProgressDelta::default()
        }));
    }
    Ok(observer.is_cancelled())
}

fn confirmed_missing(path: &camino::Utf8Path, root: &camino::Utf8Path) -> anyhow::Result<bool> {
    match fs::metadata(path) {
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("checking library file: {path}")),
    }
    // A missing parent directory is a legitimate deletion too, provided its nearest existing
    // ancestor inside the root is readable. Never use `exists()`, which hides permission errors.
    for ancestor in path
        .ancestors()
        .skip(1)
        .take_while(|parent| parent.starts_with(root))
    {
        match fs::read_dir(ancestor) {
            Ok(mut entries) => {
                if let Some(entry) = entries.next() {
                    entry?;
                }
                return Ok(true);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("checking library folder: {ancestor}"));
            }
        }
    }
    anyhow::bail!("library root became unavailable: {root}")
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Request {
    root: PathBuf,
    #[serde(default = "default_true")]
    dry_run: bool,
}

pub(super) struct Spec {
    root: PathBuf,
    dry_run: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct LibraryPurgeRequest {
    root: PathBuf,
}

pub(super) struct LibraryPurgeSpec {
    root: PathBuf,
}

pub(super) fn prepare(request: Request) -> Result<Spec, ApiError> {
    // A root must exist right now, so an unplugged drive is never mistaken for an empty library.
    let root = roots::resolve_root("prune", &request.root)?;
    Ok(Spec {
        root,
        dry_run: request.dry_run,
    })
}

pub(super) fn prepare_library_purge(
    request: LibraryPurgeRequest,
) -> Result<LibraryPurgeSpec, ApiError> {
    let root = roots::resolve_root("library purge", &request.root)?;
    Ok(LibraryPurgeSpec { root })
}

pub(super) fn run(
    spec: Spec,
    asset_database: &PathBuf,
    ocr_database: &PathBuf,
    image_database: &PathBuf,
    image_dimensions: usize,
    thumbnails: &ThumbnailService,
    observer: &dyn IndexObserver,
) -> anyhow::Result<bool> {
    ensure_root_available(&spec.root)?;
    let assets = AssetCatalog::new(asset_database)?;
    let candidates = assets.under_root(&spec.root)?;
    let mut ocr = DB::new(ocr_database)?;
    let mut images = ImageIndexDb::new(image_database, image_dimensions)?;
    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::Pruning));
    observer.on_event(IndexEvent::Discovered {
        count: candidates.len(),
    });
    observer.on_event(IndexEvent::DiscoveryComplete {
        total: candidates.len(),
    });

    for chunk in candidates.chunks(PRUNE_BATCH_SIZE) {
        let mut missing = Vec::new();
        let mut cancelled = false;
        for asset in chunk {
            if observer.is_cancelled() {
                cancelled = true;
                break;
            }
            match asset.path.try_exists() {
                Ok(true) => observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                    phase_completed: 1,
                    processed: 1,
                    skipped: 1,
                    ..IndexProgressDelta::default()
                })),
                Ok(false) => missing.push(asset),
                Err(error) => {
                    observer.on_event(IndexEvent::Error {
                        path: Some(asset.path.clone()),
                        message: format!("checking asset existence failed: {error}"),
                    });
                    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                        phase_completed: 1,
                        processed: 1,
                        failed: 1,
                        ..IndexProgressDelta::default()
                    }));
                }
            }
        }
        if missing.is_empty() {
            if cancelled {
                return Ok(true);
            }
            continue;
        }
        // A removable root can disappear after job validation. Stop before interpreting any
        // further missing children as deletions.
        ensure_root_available(&spec.root)?;
        if spec.dry_run {
            observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                phase_completed: missing.len(),
                processed: missing.len(),
                prune_candidates: missing.len(),
                ..IndexProgressDelta::default()
            }));
            if cancelled {
                return Ok(true);
            }
            continue;
        }
        match delete_asset_batch(&missing, &assets, &mut ocr, &mut images, thumbnails) {
            Ok(deleted) => observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                phase_completed: missing.len(),
                processed: missing.len(),
                prune_candidates: missing.len(),
                deleted,
                ..IndexProgressDelta::default()
            })),
            Err(error) => report_batch_failure(&missing, error, observer),
        }
        if cancelled {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn run_library_purge(
    spec: LibraryPurgeSpec,
    asset_database: &PathBuf,
    ocr_database: &PathBuf,
    image_database: &PathBuf,
    image_dimensions: usize,
    thumbnails: &ThumbnailService,
    observer: &dyn IndexObserver,
) -> anyhow::Result<bool> {
    ensure_library_purge_root_available(&spec.root)?;
    let assets = AssetCatalog::new(asset_database)?;
    // `under_root` uses a separator-delimited prefix, so a purge of `C:/photos` cannot select
    // catalog rows from a sibling such as `C:/photos-old`.
    let candidates = assets.under_root(&spec.root)?;
    let mut ocr = DB::new(ocr_database)?;
    let mut images = ImageIndexDb::new(image_database, image_dimensions)?;
    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::Pruning));
    observer.on_event(IndexEvent::Discovered {
        count: candidates.len(),
    });
    observer.on_event(IndexEvent::DiscoveryComplete {
        total: candidates.len(),
    });

    for chunk in candidates.chunks(PRUNE_BATCH_SIZE) {
        let mut assets_to_delete = Vec::with_capacity(chunk.len());
        let mut cancelled = false;
        for asset in chunk {
            if observer.is_cancelled() {
                cancelled = true;
                break;
            }
            assets_to_delete.push(asset);
        }
        if assets_to_delete.is_empty() {
            return Ok(cancelled);
        }
        match delete_asset_batch(
            &assets_to_delete,
            &assets,
            &mut ocr,
            &mut images,
            thumbnails,
        ) {
            Ok(deleted) => observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                phase_completed: assets_to_delete.len(),
                processed: assets_to_delete.len(),
                deleted,
                ..IndexProgressDelta::default()
            })),
            Err(error) => report_batch_failure(&assets_to_delete, error, observer),
        }
        if cancelled {
            return Ok(true);
        }
    }
    Ok(false)
}

fn delete_asset_batch(
    assets: &[&nicegal_core::assets::Asset],
    catalog: &AssetCatalog,
    ocr: &mut DB,
    images: &mut ImageIndexDb,
    thumbnails: &ThumbnailService,
) -> anyhow::Result<usize> {
    let asset_ids = assets
        .iter()
        .map(|asset| asset.asset_id)
        .collect::<Vec<_>>();
    // Derived rows go first. If a later store fails, every catalog row remains discoverable and a
    // retry safely finishes the already-partial cross-database deletion.
    ocr.delete_assets(&asset_ids)?;
    images.delete_assets(&asset_ids)?;
    thumbnails.delete_assets(asset_ids.clone())?;
    catalog.delete_assets(&asset_ids)
}

fn report_batch_failure(
    assets: &[&nicegal_core::assets::Asset],
    error: anyhow::Error,
    observer: &dyn IndexObserver,
) {
    let message = format!("purging asset batch failed: {error:#}");
    for asset in assets {
        observer.on_event(IndexEvent::Error {
            path: Some(asset.path.clone()),
            message: message.clone(),
        });
    }
    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: assets.len(),
        processed: assets.len(),
        failed: assets.len(),
        ..IndexProgressDelta::default()
    }));
}

fn ensure_root_available(root: &PathBuf) -> anyhow::Result<()> {
    let metadata = fs::metadata(root)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("prune root became unavailable: {root}"))?;
    if !metadata.is_dir() {
        anyhow::bail!("prune root is no longer a directory: {root}");
    }
    let mut entries =
        fs::read_dir(root).with_context(|| format!("reading library root: {root}"))?;
    if let Some(entry) = entries.next() {
        entry?;
    }
    Ok(())
}

fn ensure_library_purge_root_available(root: &PathBuf) -> anyhow::Result<()> {
    let metadata = fs::metadata(root)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("library purge root became unavailable: {root}"))?;
    if !metadata.is_dir() {
        anyhow::bail!("library purge root is no longer a directory: {root}");
    }
    Ok(())
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use tempfile::TempDir;

    struct LibraryFixture {
        _temp: TempDir,
        root: PathBuf,
        databases: super::super::Databases,
        thumbnails: ThumbnailService,
        catalog: AssetCatalog,
    }

    impl LibraryFixture {
        fn new() -> anyhow::Result<Self> {
            let temp = TempDir::new()?;
            let base = PathBuf::try_from(temp.path().to_owned())?;
            let root = base.join("photos");
            fs::create_dir(&root)?;
            let databases = super::super::Databases {
                assets: base.join("assets.db"),
                ocr: base.join("ocr.db"),
                images: base.join("clip.db"),
                thumbnails: base.join("thumbnails.db"),
            };
            let catalog = AssetCatalog::new(&databases.assets)?;
            let thumbnails = ThumbnailService::new(&databases.thumbnails)?;
            Ok(Self {
                _temp: temp,
                root,
                databases,
                thumbnails,
                catalog,
            })
        }

        fn asset(&self, relative: &str) -> anyhow::Result<nicegal_core::assets::Asset> {
            let path = self.root.join(relative);
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(&path, [])?;
            self.catalog.upsert(&path, &fs::metadata(&path)?)
        }

        fn reconcile(
            &self,
            options: IndexOptions,
            observer: &dyn IndexObserver,
        ) -> anyhow::Result<bool> {
            reconcile(
                ReconcileScope::new(self.root.clone(), &options),
                &self.databases,
                512,
                &self.thumbnails,
                observer,
            )
        }
    }

    #[test]
    fn automatic_reconciliation_deletes_missing_catalog_ocr_vectors_and_thumbnails()
    -> anyhow::Result<()> {
        use nicegal_core::db::{OcrResult, TextEmbedding, TextEmbeddingSpace};
        use nicegal_core::thumbs::{ThumbnailDb, ThumbnailEncoding};
        let library = LibraryFixture::new()?;
        let removed = library.asset("removed.png")?;
        let retained = library.asset("retained.png")?;
        let mut ocr = DB::new(&library.databases.ocr)?;
        ocr.save_results(vec![OcrResult {
            asset_id: removed.asset_id,
            path: removed.path.clone(),
            fingerprint: removed.fingerprint,
            exif_taken_ns: None,
            width: 1,
            height: 1,
            contents: "deleted words".to_owned(),
        }])?;
        ocr.set_text_embedding_model(TextEmbeddingSpace::OcrText, "test", 3, false)?;
        ocr.save_text_embeddings(
            TextEmbeddingSpace::OcrText,
            "test",
            vec![TextEmbedding {
                asset_id: removed.asset_id,
                vector: vec![1.0, 0.0, 0.0],
            }],
        )?;
        let thumbs = ThumbnailDb::new(&library.databases.thumbnails)?;
        thumbs.put(
            removed.asset_id,
            128,
            1,
            removed.fingerprint,
            1,
            1,
            ThumbnailEncoding::Png,
            &nicegal_core::imaging::encode_png(1, 1, &[0, 0, 0, 0])?,
        )?;
        assert!(
            thumbs
                .get(removed.asset_id, 128, 1, removed.fingerprint)?
                .is_some()
        );
        fs::remove_file(&removed.path)?;
        assert!(!library.reconcile(IndexOptions::default(), &NeverCancelled)?);
        assert!(library.catalog.get(removed.asset_id)?.is_none());
        assert!(library.catalog.get(retained.asset_id)?.is_some());
        assert!(!ocr.is_indexed(removed.asset_id, removed.fingerprint)?);
        assert!(
            ocr.search_text_vectors(
                &[1.0, 0.0, 0.0],
                TextEmbeddingSpace::OcrText,
                &nicegal_core::db::SearchFilters::new(&library.root),
                10,
                Default::default()
            )?
            .1
            .is_empty()
        );
        assert!(
            thumbs
                .get(removed.asset_id, 128, 1, removed.fingerprint)?
                .is_none()
        );
        assert!(!library.reconcile(IndexOptions::default(), &NeverCancelled)?);
        Ok(())
    }

    #[test]
    fn automatic_reconciliation_preserves_cancelled_limited_and_unavailable_libraries()
    -> anyhow::Result<()> {
        let library = LibraryFixture::new()?;
        let removed = library.asset("removed.png")?;
        fs::remove_file(&removed.path)?;
        assert!(library.reconcile(
            IndexOptions::default(),
            &CancelBeforeSecondAsset(AtomicUsize::new(0))
        )?);
        assert!(library.catalog.get(removed.asset_id)?.is_some());
        assert!(!library.reconcile(
            IndexOptions {
                limit: Some(10),
                ..IndexOptions::default()
            },
            &NeverCancelled
        )?);
        assert!(library.catalog.get(removed.asset_id)?.is_some());
        fs::remove_dir(&library.root)?;
        assert!(
            library
                .reconcile(IndexOptions::default(), &NeverCancelled)
                .is_err()
        );
        assert!(library.catalog.get(removed.asset_id)?.is_some());
        Ok(())
    }

    #[test]
    fn automatic_reconciliation_respects_excluded_ancestors_and_nonrecursive_scans()
    -> anyhow::Result<()> {
        let library = LibraryFixture::new()?;
        let excluded = library.asset(".cache/nested/removed.png")?;
        let nested = library.asset("nested/removed.png")?;
        let direct = library.asset("removed.png")?;
        for asset in [&excluded, &nested, &direct] {
            fs::remove_file(&asset.path)?;
        }
        library.reconcile(
            IndexOptions {
                subdirs: false,
                ..IndexOptions::default()
            },
            &NeverCancelled,
        )?;
        assert!(library.catalog.get(direct.asset_id)?.is_none());
        assert!(library.catalog.get(nested.asset_id)?.is_some());
        library.reconcile(
            IndexOptions {
                exclude: vec![glob::Pattern::new("*/.cache")?],
                ..IndexOptions::default()
            },
            &NeverCancelled,
        )?;
        assert!(library.catalog.get(nested.asset_id)?.is_none());
        assert!(library.catalog.get(excluded.asset_id)?.is_some());
        Ok(())
    }

    #[test]
    fn unreadable_ancestor_aborts_automatic_reconciliation_before_any_deletion()
    -> anyhow::Result<()> {
        let library = LibraryFixture::new()?;
        let removed = library.asset("removed.png")?;
        let inaccessible = library.asset("folder/removed.png")?;
        fs::remove_file(&removed.path)?;
        fs::remove_file(&inaccessible.path)?;
        fs::remove_dir(library.root.join("folder"))?;
        // A path that cannot be traversed is portable across Windows and Unix, unlike ACL tests
        // whose permissions depend on the test runner's elevation.
        fs::write(library.root.join("folder"), [])?;
        assert!(
            library
                .reconcile(IndexOptions::default(), &NeverCancelled)
                .is_err()
        );
        assert!(library.catalog.get(removed.asset_id)?.is_some());
        assert!(library.catalog.get(inaccessible.asset_id)?.is_some());
        Ok(())
    }

    struct NeverCancelled;

    impl IndexObserver for NeverCancelled {
        fn is_cancelled(&self) -> bool {
            false
        }

        fn on_event(&self, _event: IndexEvent) {}
    }

    struct CancelBeforeSecondAsset(AtomicUsize);

    impl IndexObserver for CancelBeforeSecondAsset {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) != 0
        }

        fn on_event(&self, _event: IndexEvent) {}
    }

    #[test]
    fn prune_defaults_to_dry_run() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "root": "C:/gallery"
        }))
        .unwrap();
        assert!(request.dry_run);
    }

    #[test]
    fn unavailable_root_is_rejected_before_job_creation() {
        let missing = PathBuf::try_from(std::env::current_dir().unwrap())
            .unwrap()
            .join("definitely-missing-prune-root");
        let error = prepare(Request {
            root: missing,
            dry_run: false,
        })
        .err()
        .expect("missing root should be rejected");
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(error.code, super::super::error::ErrorCode::InvalidRoot);
    }

    #[test]
    fn library_purge_request_requires_only_a_root() {
        assert!(
            serde_json::from_value::<LibraryPurgeRequest>(serde_json::json!({
                "root": "C:/gallery"
            }))
            .is_ok()
        );
        assert!(serde_json::from_value::<LibraryPurgeRequest>(serde_json::json!({})).is_err());
        assert!(
            serde_json::from_value::<LibraryPurgeRequest>(serde_json::json!({
                "root": "C:/gallery",
                "dryRun": false
            }))
            .is_err()
        );
    }

    #[test]
    fn library_purge_does_not_touch_same_prefix_sibling_root() -> anyhow::Result<()> {
        let temporary = TempDir::new()?;
        let root = PathBuf::try_from(temporary.path().join("photos"))?;
        let sibling = PathBuf::try_from(temporary.path().join("photos-old"))?;
        fs::create_dir(&root)?;
        fs::create_dir(&sibling)?;
        let inside = root.join("inside.png");
        let outside = sibling.join("outside.png");
        fs::write(&inside, [])?;
        fs::write(&outside, [])?;

        let asset_database = PathBuf::try_from(temporary.path().join("assets.db"))?;
        let ocr_database = PathBuf::try_from(temporary.path().join("ocr.db"))?;
        let image_database = PathBuf::try_from(temporary.path().join("clip.db"))?;
        let thumbnail_database = PathBuf::try_from(temporary.path().join("thumbnails.db"))?;
        let thumbnails = ThumbnailService::new(&thumbnail_database)?;
        let catalog = AssetCatalog::new(&asset_database)?;
        let included = catalog.upsert(&inside, &fs::metadata(&inside)?)?;
        let excluded = catalog.upsert(&outside, &fs::metadata(&outside)?)?;

        assert!(!run_library_purge(
            prepare_library_purge(LibraryPurgeRequest { root })
                .expect("temporary root should prepare for library purge"),
            &asset_database,
            &ocr_database,
            &image_database,
            512,
            &thumbnails,
            &NeverCancelled,
        )?);

        let catalog = AssetCatalog::new(&asset_database)?;
        assert!(catalog.get(included.asset_id)?.is_none());
        assert!(catalog.get(excluded.asset_id)?.is_some());
        Ok(())
    }

    #[test]
    fn library_purge_cancellation_stops_before_the_next_asset() -> anyhow::Result<()> {
        let temporary = TempDir::new()?;
        let root = PathBuf::try_from(temporary.path().join("photos"))?;
        fs::create_dir(&root)?;
        let first_path = root.join("first.png");
        let second_path = root.join("second.png");
        fs::write(&first_path, [])?;
        fs::write(&second_path, [])?;

        let asset_database = PathBuf::try_from(temporary.path().join("assets.db"))?;
        let ocr_database = PathBuf::try_from(temporary.path().join("ocr.db"))?;
        let image_database = PathBuf::try_from(temporary.path().join("clip.db"))?;
        let thumbnail_database = PathBuf::try_from(temporary.path().join("thumbnails.db"))?;
        let thumbnails = ThumbnailService::new(&thumbnail_database)?;
        let catalog = AssetCatalog::new(&asset_database)?;
        let first = catalog.upsert(&first_path, &fs::metadata(&first_path)?)?;
        let second = catalog.upsert(&second_path, &fs::metadata(&second_path)?)?;

        assert!(run_library_purge(
            prepare_library_purge(LibraryPurgeRequest { root })
                .expect("temporary root should prepare for library purge"),
            &asset_database,
            &ocr_database,
            &image_database,
            512,
            &thumbnails,
            &CancelBeforeSecondAsset(AtomicUsize::new(0)),
        )?);

        let catalog = AssetCatalog::new(&asset_database)?;
        assert!(catalog.get(first.asset_id)?.is_none());
        assert!(catalog.get(second.asset_id)?.is_some());
        Ok(())
    }
}
