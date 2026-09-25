//! Image embedding jobs.

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
use nicegal_core::image_index::{
    ImageIndexDb, ImageIndexOptions, has_pending_catalog_images, index_catalog_images_observed,
    index_images_observed,
};
use nicegal_core::index::IndexObserver;
use nicegal_core::scope::PathScope;
use nicegal_core::thumbs::ThumbnailService;
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
    let dimensions = state.image_query_embedder.dimensions();
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
    /// Set for a standalone job; its scope is read from the library when the job runs.
    library_id: Option<i64>,
    scope: PathScope,
    force: bool,
    retry_failed: bool,
    debug_limit: Option<usize>,
    index_videos: bool,
}

impl Spec {
    pub(crate) fn pending_for(
        scope: PathScope,
        retry_failed: bool,
        debug_limit: Option<usize>,
        index_videos: bool,
    ) -> Self {
        Self {
            library_id: None,
            scope,
            force: false,
            retry_failed,
            debug_limit,
            index_videos,
        }
    }
}

impl Spec {
    pub(crate) fn library_id(&self) -> Option<i64> {
        self.library_id
    }

    /// Read a standalone job's library scope.
    pub(crate) fn resolve(mut self, databases: &Databases) -> anyhow::Result<Self> {
        if let Some(library_id) = self.library_id {
            let catalog = AssetCatalog::new_read_only(&databases.assets)?;
            self.scope = super::libraries::stored(&catalog, library_id)?.scope();
        }
        Ok(self)
    }
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    if request.debug_limit == Some(0) {
        return Err(ApiError::bad_request(
            "debugLimit must be greater than zero",
        ));
    }
    Ok(Spec {
        library_id: Some(request.library_id),
        scope: PathScope::default(),
        force: request.force,
        retry_failed: false,
        debug_limit: request.debug_limit,
        index_videos: true,
    })
}

pub(crate) fn run(
    spec: Spec,
    databases: &Databases,
    thumbnails: &ThumbnailService,
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
    catalog: Option<&[Asset]>,
) -> anyhow::Result<()> {
    let assets = AssetCatalog::new(&databases.assets)?;
    let mut images = ImageIndexDb::new(&databases.images, embedder.dimensions())?;
    let options = ImageIndexOptions {
        force: spec.force,
        retry_failed: spec.retry_failed,
        index_videos: spec.index_videos,
        limit: spec.debug_limit,
        thumbnail_database: Some(databases.thumbnails.clone()),
        thumbnail_service: Some(thumbnails.clone()),
    };
    match catalog {
        Some(catalog) => index_catalog_images_observed(
            &assets,
            &mut images,
            embedder,
            catalog,
            options,
            observer,
        ),
        None => index_images_observed(
            &assets,
            &mut images,
            embedder,
            &spec.scope,
            options,
            observer,
        ),
    }
    .and_then(cancel_if)
}

/// Use the same candidate selection as the embedding pass before preparing its model.
pub(crate) fn has_pending(
    spec: &Spec,
    databases: &Databases,
    dimensions: usize,
    catalog: Option<&[Asset]>,
) -> anyhow::Result<bool> {
    tracing::debug!("checking pending image embeddings");
    let assets = AssetCatalog::new_read_only(&databases.assets)?;
    let images = ImageIndexDb::new_read_only(&databases.images, dimensions, &databases.assets)?;
    tracing::debug!("opened read-only embedding selection databases");
    let owned;
    let catalog = match catalog {
        Some(catalog) => catalog,
        None => {
            owned = assets.in_scope(&spec.scope)?;
            &owned
        }
    };
    let pending = has_pending_catalog_images(
        &assets,
        &images,
        catalog,
        &ImageIndexOptions {
            force: spec.force,
            retry_failed: spec.retry_failed,
            index_videos: spec.index_videos,
            limit: spec.debug_limit,
            thumbnail_database: Some(databases.thumbnails.clone()),
            thumbnail_service: None,
        },
    )?;
    tracing::debug!(pending, "checked pending image embeddings");
    Ok(pending)
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
