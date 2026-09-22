//! `GET /v1/text-embeddings` — what OCR-text vector search can currently answer for.
//!
//! Vector search degrades silently: a root whose text has never been embedded returns no hits and
//! looks exactly like a root with nothing to find. This route is how the UI tells those apart, and
//! how it decides whether to offer a backfill.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{MethodRouter, get, post};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::db::{SearchFilters, TextEmbeddingSpace};
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::extract::{ApiJson, ApiQuery};
use super::jobs::{JobResponse, JobSpec};
use super::{AppState, roots, run_blocking};

pub(super) mod job;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StatusRequest {
    root: PathBuf,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusResponse {
    /// The text embedding model this build embeds with. Always present, even before anything is
    /// stored.
    embedder: EmbedderResponse,
    /// The model the *stored* vectors belong to. Null until the first backfill. When this differs
    /// from `embedder`, the next text embed job discards the stored vectors and starts over.
    stored: Option<EmbedderResponse>,
    /// OCR rows under the root.
    indexed: usize,
    /// Of those, the ones whose vector matches their current OCR text.
    embedded: usize,
    /// The backlog an embed job would work through.
    pending: usize,
    /// Unix seconds of the newest indexed source file under the root, or null when nothing is
    /// indexed yet. There is no wall-clock record of when OCR indexing actually ran, so this is a
    /// proxy: the most recent `source_modified_ns` among indexed rows, truncated to seconds.
    last_indexed_at: Option<i64>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct EmbedderResponse {
    model: String,
    dimensions: usize,
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(status)
}

pub(super) fn generate_route() -> MethodRouter<AppState> {
    post(create_job)
}

async fn create_job(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<job::Request>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    let spec = JobSpec::TextEmbed(job::prepare(request)?);
    let job = state.jobs.start(spec)?;
    Ok((StatusCode::ACCEPTED, Json(job.response())))
}

async fn status(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<StatusRequest>,
) -> Result<Json<StatusResponse>, ApiError> {
    let requested_root = request.root;
    let databases = state.databases;
    let embedder = EmbedderResponse {
        model: state.embedder.model().id().to_owned(),
        dimensions: state.embedder.dimensions(),
    };

    run_blocking(move || {
        let root = roots::resolve_root("text embeddings", &requested_root)?;
        let db = databases.open_ocr_read_only()?;
        let stored = db
            .text_embedding_model(TextEmbeddingSpace::OcrText)?
            .map(|model| EmbedderResponse {
                model: model.model,
                dimensions: model.dimensions,
            });
        let coverage =
            db.text_embedding_coverage(TextEmbeddingSpace::OcrText, &SearchFilters::new(&root))?;
        Ok(Json(StatusResponse {
            embedder,
            stored,
            indexed: coverage.indexed,
            embedded: coverage.embedded,
            pending: coverage.pending(),
            last_indexed_at: coverage
                .last_indexed_ns
                .map(|ns| ns.div_euclid(1_000_000_000)),
        }))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_is_required_and_unknown_parameters_are_rejected() {
        assert!(serde_urlencoded::from_str::<StatusRequest>("").is_err());
        assert!(serde_urlencoded::from_str::<StatusRequest>("root=C:/gallery&bogus=1").is_err());
        let request: StatusRequest =
            serde_urlencoded::from_str("root=C:/gallery").expect("a root alone is enough");
        assert_eq!(request.root, "C:/gallery");
    }
}
