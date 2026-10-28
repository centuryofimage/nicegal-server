//! Image embedding jobs.

use super::indexing::Services;
use super::{AppState, Databases, extract::ApiQuery, run_blocking};
use crate::api::jobs::cancel_if;
use axum::{
    Json,
    extract::State,
    routing::{MethodRouter, get},
};
use nicegal_core::assets::{Asset, AssetCatalog};
use nicegal_core::db::SearchFilters;
use nicegal_core::embedding::ImageEmbedder;
use nicegal_core::image_index::{ImageIndexDb, ImageIndexOptions, index_selected_images};
use nicegal_core::index::IndexObserver;
use serde::{Deserialize, Serialize};

use super::error::ApiError;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CoverageRequest {
    library_id: i64,
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
    let dimensions = state.models.image_query.dimensions();
    run_blocking(move || {
        let scope = super::libraries::scope(&state.databases, request.library_id)?;
        let db = state.databases.open_images_read_only(dimensions)?;
        let (total, indexed) = db.coverage(&SearchFilters::new(scope))?;
        Ok(Json(CoverageResponse { total, indexed }))
    })
    .await
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    library_id: i64,
    #[serde(default)]
    force: bool,
    debug_limit: Option<usize>,
}

pub(crate) struct Spec {
    library_id: i64,
    force: bool,
    debug_limit: Option<usize>,
}

impl Spec {
    pub(crate) fn library_id(&self) -> i64 {
        self.library_id
    }

    /// The library's cataloged files and what to embed of them, read when the job runs.
    pub(crate) fn resolve(
        &self,
        databases: &Databases,
    ) -> anyhow::Result<(Vec<Asset>, ImageIndexOptions)> {
        let catalog = AssetCatalog::new_read_only(&databases.assets)?;
        let library = super::libraries::stored(&catalog, self.library_id)?;
        let options = ImageIndexOptions {
            force: self.force,
            retry_failed: false,
            limit: self.debug_limit,
            index_videos: library.options.videos,
        };
        Ok((catalog.in_scope(&library.scope())?, options))
    }
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    if request.debug_limit == Some(0) {
        return Err(ApiError::bad_request(
            "debugLimit must be greater than zero",
        ));
    }
    Ok(Spec {
        library_id: request.library_id,
        force: request.force,
        debug_limit: request.debug_limit,
    })
}

/// Embed `sources`, selected by `select_catalog_images`, with a loaded `embedder`.
pub(super) fn run(
    services: &Services<'_>,
    embedder: &ImageEmbedder,
    sources: Vec<Asset>,
    observer: &dyn IndexObserver,
) -> anyhow::Result<()> {
    let databases = services.databases;
    let assets = AssetCatalog::new(&databases.assets)?;
    let mut images = ImageIndexDb::new(&databases.images, embedder.dimensions())?;
    index_selected_images(
        &assets,
        &mut images,
        embedder,
        sources,
        Some(services.thumbnails),
        observer,
    )
    .and_then(cancel_if)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> Result<Request, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn library_is_required_and_force_defaults_off() {
        assert!(request(serde_json::json!({})).is_err());
        let parsed = request(serde_json::json!({"libraryId": 1})).unwrap();
        assert!(!parsed.force);
        assert!(
            request(serde_json::json!({
                "libraryId": 1,
                "unknown": true
            }))
            .is_err()
        );
    }

    #[test]
    fn image_debug_limit_is_positive_and_optional() {
        let parsed = request(serde_json::json!({"libraryId": 1, "debugLimit": 2000})).unwrap();
        assert_eq!(prepare(parsed).unwrap().debug_limit, Some(2000));
        let parsed = request(serde_json::json!({"libraryId": 1, "debugLimit": 0})).unwrap();
        assert!(prepare(parsed).is_err());
    }
}
