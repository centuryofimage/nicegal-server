//! Zero-shot tags for one asset from its stored image vector.
use axum::Json;
use axum::extract::State;
use axum::routing::{MethodRouter, get};
use nicegal_core::cancellation::SearchCancellation;
use nicegal_core::embedding::ImageEmbeddingModel;
use nicegal_core::tags::{ScoredTag, Sensitivity, TagSet, TagSource};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use super::error::{ApiError, ErrorCode};
use super::extract::ApiQuery;
use super::{AppState, run_blocking, search};

const TAGS_PER_KIND: usize = 15;

/// The active model's tag set, loaded on its first request. Loading holds the lock so concurrent
/// requests wait for one download instead of starting their own.
#[derive(Default)]
pub(crate) struct TagSets(Mutex<Option<Arc<TagSet>>>);

impl TagSets {
    fn for_model(&self, model: ImageEmbeddingModel) -> anyhow::Result<Arc<TagSet>> {
        let mut current = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("tag set lock is poisoned"))?;
        if let Some(set) = current.as_ref().filter(|set| set.model() == model) {
            return Ok(Arc::clone(set));
        }
        // Release the previous model's map before loading the next one.
        *current = None;
        let set = Arc::new(TagSet::load(model)?);
        *current = Some(Arc::clone(&set));
        Ok(set)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TagsRequest {
    asset_id: i64,
    /// Leaves out tags rated `blocked`. On unless disabled.
    #[serde(default = "hide_offensive_default")]
    hide_offensive: bool,
}

fn hide_offensive_default() -> bool {
    true
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TagsResponse {
    model: &'static str,
    /// False when the active image model has no published tags; every list is then empty.
    supported: bool,
    simple: Vec<TagResponse>,
    subjects: Vec<TagResponse>,
    vibes: Vec<TagResponse>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TagResponse {
    term: String,
    source: TagSource,
    sensitivity: Sensitivity,
    similarity: f32,
    score: f32,
}

impl From<ScoredTag<'_>> for TagResponse {
    fn from(scored: ScoredTag<'_>) -> Self {
        Self {
            term: scored.tag.term.clone(),
            source: scored.tag.source,
            sensitivity: scored.tag.sensitivity,
            similarity: scored.similarity,
            score: scored.score,
        }
    }
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(tags)
}

async fn tags(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<TagsRequest>,
) -> Result<Json<TagsResponse>, ApiError> {
    run_blocking(move || {
        let model = state.image_query_embedder.model();
        if !TagSet::supports(model) {
            return Ok(Json(TagsResponse {
                model: model.id(),
                supported: false,
                simple: Vec::new(),
                subjects: Vec::new(),
                vibes: Vec::new(),
            }));
        }
        let query = serde_json::from_value(serde_json::json!({
            "components": [{ "assetId": request.asset_id }]
        }))
        .map_err(|error| ApiError::internal(error.into()))?;
        let image = search::resolve_image_query(&state, query, &SearchCancellation::default())
            .map_err(|error| match error.code {
                // The only invalid part of a one-asset query is an asset with no current vector.
                ErrorCode::InvalidRequest => ApiError::bad_request(
                    "This file isn't indexed for the current search model yet.",
                ),
                _ => error,
            })?;
        let set = state.tags.for_model(model).map_err(ApiError::internal)?;
        let ranking = set
            .rank(&image, TAGS_PER_KIND, request.hide_offensive)
            .map_err(ApiError::internal)?;
        Ok(Json(TagsResponse {
            model: model.id(),
            supported: true,
            simple: ranking.simple.into_iter().map(Into::into).collect(),
            subjects: ranking.subjects.into_iter().map(Into::into).collect(),
            vibes: ranking.vibes.into_iter().map(Into::into).collect(),
        }))
    })
    .await
}
