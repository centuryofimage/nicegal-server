use super::jobs::is_cancelled;
use std::sync::{Arc, Mutex};

use camino::Utf8PathBuf as PathBuf;
use glob::Pattern;
use nicegal_core::assets::{Asset, AssetCatalog};
use nicegal_core::db::DB;
use nicegal_core::index::{self, IndexObserver, IndexOptions};
use nicegal_core::ocr::PaddleOcrPool;
use serde::Deserialize;

use super::error::ApiError;
use super::prune_jobs::ReconcileScope;
use super::roots;

const OCR_COMMIT_CHUNK_SIZE: usize = 32;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    root: PathBuf,
    #[serde(default)]
    scan: ScanOptions,
    embed: Option<bool>,
    #[serde(default = "default_true")]
    ocr: bool,
    image: Option<bool>,
}

pub(crate) struct Spec {
    root: PathBuf,
    options: IndexOptions,
    text: bool,
    ocr: bool,
    image: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CatalogSyncRequest {
    root: PathBuf,
    #[serde(default)]
    scan: CatalogScanOptions,
    #[serde(default)]
    image: bool,
}

pub(crate) struct CatalogSyncSpec {
    root: PathBuf,
    options: IndexOptions,
    image: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ScanOptions {
    #[serde(default = "default_true")]
    recursive: bool,
    #[serde(default = "default_excludes")]
    exclude: Vec<String>,
    #[serde(default)]
    force: bool,
    #[serde(default)]
    retry_failed: bool,
    #[serde(default)]
    cleanup: bool,
    max_dimensions: Option<MaxDimensions>,
    debug_limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogScanOptions {
    #[serde(default = "default_true")]
    recursive: bool,
    #[serde(default = "default_excludes")]
    exclude: Vec<String>,
    /// Debug guardrail. Production callers should omit this and catalog the whole root.
    debug_limit: Option<usize>,
    /// Older desktop builds send this flag; delta classification now runs for either value.
    #[serde(rename = "newOnly", default)]
    _legacy_new_only: bool,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            recursive: true,
            exclude: default_excludes(),
            force: false,
            retry_failed: false,
            cleanup: false,
            max_dimensions: None,
            debug_limit: None,
        }
    }
}

impl Default for CatalogScanOptions {
    fn default() -> Self {
        Self {
            recursive: true,
            exclude: default_excludes(),
            debug_limit: None,
            _legacy_new_only: false,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MaxDimensions {
    width: usize,
    height: usize,
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    let image = request.image.unwrap_or(request.embed.unwrap_or(true));
    if !request.ocr && !image {
        return Err(ApiError::bad_request(
            "select text recognition or image search",
        ));
    }
    roots::resolve_root("index", &request.root)?;
    Ok(Spec {
        root: request.root,
        options: index_options(request.scan)?,
        text: request.ocr && request.embed.unwrap_or(true),
        ocr: request.ocr,
        image,
    })
}

pub(crate) fn prepare_catalog_sync(
    request: CatalogSyncRequest,
) -> Result<CatalogSyncSpec, ApiError> {
    let root = roots::resolve_root("catalog sync", &request.root)?;
    Ok(CatalogSyncSpec {
        root,
        options: catalog_sync_options(request.scan)?,
        image: request.image,
    })
}

pub(crate) fn run(
    spec: Spec,
    asset_database: &PathBuf,
    ocr_database: &PathBuf,
    models: Option<&Arc<Mutex<PaddleOcrPool>>>,
    observer: &dyn IndexObserver,
    catalog_step: impl FnOnce(&[Asset]) -> anyhow::Result<()>,
) -> anyhow::Result<index::IndexSummary> {
    let mut assets = AssetCatalog::new(asset_database)?;
    if !spec.ocr {
        let summary =
            index::catalog_dir_with_step(&assets, &spec.root, spec.options, observer, |catalog| {
                cancellation_result(catalog_step(catalog))
            })?;
        return Ok(index::IndexSummary {
            indexed: 0,
            deleted: 0,
            cancelled: summary.cancelled,
            scan_complete: summary.scan_complete,
        });
    }
    let models = models.ok_or_else(|| {
        anyhow::anyhow!("PaddleOCR models were unloaded after the index job was accepted")
    })?;
    let mut ocr = DB::new(ocr_database)?;
    let mut models = models.lock().unwrap_or_else(|error| error.into_inner());
    let summary = index::index_dir_with_catalog_step(
        &mut assets,
        &mut ocr,
        &mut models,
        &spec.root,
        spec.options,
        observer,
        |catalog| cancellation_result(catalog_step(catalog)),
    )?;
    Ok(summary)
}

fn cancellation_result(result: anyhow::Result<()>) -> anyhow::Result<bool> {
    match result {
        Ok(()) => Ok(false),
        Err(error) if is_cancelled(&error) => Ok(true),
        Err(error) => Err(error),
    }
}

pub(crate) fn run_catalog_sync(
    spec: CatalogSyncSpec,
    asset_database: &PathBuf,
    observer: &dyn IndexObserver,
) -> anyhow::Result<index::CatalogDelta> {
    let assets = AssetCatalog::new(asset_database)?;
    index::catalog_delta_dir_observed(&assets, &spec.root, spec.options, observer)
}

impl CatalogSyncSpec {
    pub(crate) fn embeds_images(&self) -> bool {
        self.image
    }
    pub(crate) fn root(&self) -> &PathBuf {
        &self.root
    }
    pub(crate) fn debug_limit(&self) -> Option<usize> {
        self.options.limit
    }
    pub(super) fn reconciliation(&self) -> ReconcileScope {
        ReconcileScope::new(self.root.clone(), &self.options)
    }
}

impl Spec {
    pub(super) fn reconciliation(&self) -> ReconcileScope {
        ReconcileScope::new(self.root.clone(), &self.options)
    }
    pub(crate) fn root(&self) -> &PathBuf {
        &self.root
    }

    pub(crate) fn recognizes_text(&self) -> bool {
        self.ocr
    }

    pub(crate) fn embeds_text(&self) -> bool {
        self.text
    }

    pub(crate) fn embeds_images(&self) -> bool {
        self.image
    }

    pub(crate) fn retry_failed(&self) -> bool {
        self.options.retry_failed
    }

    pub(crate) fn debug_limit(&self) -> Option<usize> {
        self.options.limit
    }
}

fn index_options(scan: ScanOptions) -> Result<IndexOptions, ApiError> {
    let mut options = scan_options(scan.recursive, scan.exclude)?;
    options.rescan = scan.force;
    options.retry_failed = scan.retry_failed;
    options.commit_chunk_size = OCR_COMMIT_CHUNK_SIZE;
    options.cleanup = scan.cleanup;
    if scan.debug_limit == Some(0) {
        return Err(ApiError::bad_request(
            "index debug limit must be greater than zero",
        ));
    }
    options.limit = scan.debug_limit;
    options.max_dimensions = scan
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
    Ok(options)
}

fn catalog_sync_options(scan: CatalogScanOptions) -> Result<IndexOptions, ApiError> {
    let mut options = scan_options(scan.recursive, scan.exclude)?;
    if scan.debug_limit == Some(0) {
        return Err(ApiError::bad_request(
            "catalog debug limit must be greater than zero",
        ));
    }
    options.limit = scan.debug_limit;
    Ok(options)
}

fn scan_options(recursive: bool, exclude: Vec<String>) -> Result<IndexOptions, ApiError> {
    let exclude = exclude
        .into_iter()
        .map(|pattern| {
            Pattern::new(&pattern).map_err(|error| {
                ApiError::bad_request(format!("invalid exclusion pattern {pattern:?}: {error}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(IndexOptions {
        exclude,
        subdirs: recursive,
        ..IndexOptions::default()
    })
}

fn default_true() -> bool {
    true
}

fn default_excludes() -> Vec<String> {
    vec!["*/.cache".to_owned(), "*/.thumb*".to_owned()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_only_scan_does_not_open_ocr_database_or_require_models() {
        crate::api::tests::initialize_test_runtime();
        let temp = tempfile::TempDir::new().unwrap();
        let directory = PathBuf::try_from(temp.path().to_path_buf()).unwrap();
        let root = directory.join("pictures");
        std::fs::create_dir(&root).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "root": root, "ocr": false, "image": true
        }))
        .unwrap();
        struct Observer;
        impl IndexObserver for Observer {}
        let mut called = false;
        let summary = run(
            prepare(request).unwrap(),
            &directory.join("assets.db"),
            &directory.join("ocr.db"),
            None,
            &Observer,
            |catalog| {
                assert!(catalog.is_empty());
                called = true;
                assert!(directory.join("assets.db").exists());
                assert!(!directory.join("ocr.db").exists());
                crate::api::jobs::cancel_if(true)
            },
        )
        .unwrap();
        assert!(summary.scan_complete);
        assert!(called);
        assert!(summary.cancelled);
        assert!(!directory.join("ocr.db").exists());
    }

    #[test]
    fn index_request_uses_stable_scan_defaults() {
        let request: Request = serde_json::from_str(r#"{"root":"/gallery"}"#).unwrap();
        let options = index_options(request.scan).unwrap();
        assert!(options.subdirs);
        assert!(!options.rescan);
        assert!(!options.cleanup);
        assert_eq!(options.commit_chunk_size, OCR_COMMIT_CHUNK_SIZE);
        assert_eq!(options.exclude.len(), 2);
        assert!(request.embed.unwrap_or(true));
    }

    #[test]
    fn index_request_can_disable_default_embedding() {
        let request: Request =
            serde_json::from_str(r#"{"root":"/gallery","embed":false}"#).unwrap();
        assert!(!request.embed.unwrap_or(true));
    }

    #[test]
    fn catalog_sync_request_uses_scan_only_defaults() {
        let request: CatalogSyncRequest = serde_json::from_str(r#"{"root":"/gallery"}"#).unwrap();
        let options = catalog_sync_options(request.scan).unwrap();
        assert!(options.subdirs);
        assert!(!options.rescan);
        assert!(!options.cleanup);
        assert_eq!(options.limit, None);
        assert_eq!(options.exclude.len(), 2);
        assert!(!request.image);

        let unsupported = serde_json::from_str::<CatalogSyncRequest>(
            r#"{"root":"/gallery","scan":{"force":true}}"#,
        );
        assert!(unsupported.is_err());
    }

    #[test]
    fn catalog_sync_accepts_image_embedding_without_a_scan_mode() {
        let request: CatalogSyncRequest =
            serde_json::from_str(r#"{"root":"/gallery","image":true}"#).unwrap();
        assert!(request.image);
        let options = catalog_sync_options(request.scan).unwrap();
        assert!(!options.rescan);
        assert!(!options.cleanup);
        assert!(serde_json::from_str::<CatalogSyncRequest>(
            r#"{"root":"/gallery","scan":{"newOnly":true}}"#
        )
        .is_ok());
    }

    #[test]
    fn catalog_sync_accepts_a_positive_debug_limit() {
        let request: CatalogSyncRequest =
            serde_json::from_str(r#"{"root":"/gallery","scan":{"debugLimit":1000}}"#).unwrap();
        assert_eq!(
            catalog_sync_options(request.scan).unwrap().limit,
            Some(1000)
        );

        let request: CatalogSyncRequest =
            serde_json::from_str(r#"{"root":"/gallery","scan":{"debugLimit":0}}"#).unwrap();
        assert!(catalog_sync_options(request.scan).is_err());
    }

    #[test]
    fn ocr_index_accepts_a_positive_debug_limit() {
        let request: Request =
            serde_json::from_str(r#"{"root":"/gallery","scan":{"debugLimit":100}}"#).unwrap();
        assert_eq!(index_options(request.scan).unwrap().limit, Some(100));

        let request: Request =
            serde_json::from_str(r#"{"root":"/gallery","scan":{"debugLimit":0}}"#).unwrap();
        assert!(index_options(request.scan).is_err());
    }

    #[test]
    fn dimensions_must_be_positive() {
        let scan: ScanOptions =
            serde_json::from_str(r#"{"maxDimensions":{"width":0,"height":100}}"#).unwrap();
        assert!(index_options(scan).is_err());
    }
}
