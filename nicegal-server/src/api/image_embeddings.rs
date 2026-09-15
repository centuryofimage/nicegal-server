//! CLIP image embedding jobs.

use super::{AppState, extract::ApiQuery, run_blocking};
use axum::{
    Json,
    extract::State,
    routing::{MethodRouter, get},
};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::db::SearchFilters;
use nicegal_core::embedding::ImageEmbedder;
use nicegal_core::image_index::{ImageIndexDb, ImageIndexOptions, index_images_observed};
use nicegal_core::index::IndexObserver;
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::roots;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageRequest {
    root: PathBuf,
}

#[derive(Serialize)]
struct CoverageResponse {
    total: usize,
    indexed: usize,
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(coverage)
}

async fn coverage(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<CoverageRequest>,
) -> Result<Json<CoverageResponse>, ApiError> {
    let dimensions = state.image_query_embedder.dimensions();
    run_blocking(move || {
        let root = roots::resolve_root("image embeddings", &request.root)?;
        let db = state.databases.open_images_read_only(dimensions)?;
        let (total, indexed) = db.coverage(&SearchFilters::new(&root))?;
        Ok(Json(CoverageResponse { total, indexed }))
    })
    .await
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    root: PathBuf,
    #[serde(default)]
    force: bool,
    debug_limit: Option<usize>,
}

pub(crate) struct Spec {
    root: PathBuf,
    force: bool,
    retry_failed: bool,
    debug_limit: Option<usize>,
}

impl Spec {
    pub(crate) fn pending_for(
        root: PathBuf,
        retry_failed: bool,
        debug_limit: Option<usize>,
    ) -> Self {
        Self {
            root,
            force: false,
            retry_failed,
            debug_limit,
        }
    }
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    if request.debug_limit == Some(0) {
        return Err(ApiError::bad_request(
            "debugLimit must be greater than zero",
        ));
    }
    let root = roots::resolve_root("image embeddings", &request.root)?;
    Ok(Spec {
        root,
        force: request.force,
        retry_failed: false,
        debug_limit: request.debug_limit,
    })
}

pub(crate) fn run(
    spec: Spec,
    asset_database: &PathBuf,
    image_database: &PathBuf,
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
) -> anyhow::Result<bool> {
    let assets = AssetCatalog::new(asset_database)?;
    let mut images = ImageIndexDb::new(image_database, embedder.dimensions())?;
    index_images_observed(
        &assets,
        &mut images,
        embedder,
        &spec.root,
        ImageIndexOptions {
            force: spec.force,
            retry_failed: spec.retry_failed,
            limit: spec.debug_limit,
        },
        observer,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> Result<Request, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn root_is_required_and_force_defaults_off() {
        assert!(request(serde_json::json!({})).is_err());
        let parsed = request(serde_json::json!({"root": "C:/gallery"})).unwrap();
        assert!(!parsed.force);
        assert!(
            request(serde_json::json!({
                "root": "C:/gallery",
                "unknown": true
            }))
            .is_err()
        );
    }

    #[test]
    fn image_debug_limit_is_positive_and_optional() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().to_str().unwrap();
        let parsed = request(serde_json::json!({"root": root, "debugLimit": 2000})).unwrap();
        assert_eq!(prepare(parsed).unwrap().debug_limit, Some(2000));
        let parsed = request(serde_json::json!({"root": root, "debugLimit": 0})).unwrap();
        assert!(prepare(parsed).is_err());
    }
}
