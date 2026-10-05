//! HTTP routes for creating, watching, and cancelling jobs.
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use tokio_stream::Stream;
use tokio_stream::wrappers::WatchStream;

use super::job::{Job, JobListResponse, JobResponse};
use super::request::{JobRequest, JobSpec};
use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;
use crate::api::identifier::validate_identifier;
use crate::api::ttl_map::Retained;

pub(super) const MAX_REQUEST_ID_LEN: usize = 160;

/// The job a client request ID already started, kept to answer retries of the same request.
pub(super) struct RetainedRequest {
    pub(super) fingerprint: String,
    pub(super) job_id: Option<u64>,
    pub(super) created: Instant,
}

impl Retained for RetainedRequest {
    fn touched_at(&self) -> Instant {
        self.created
    }
}

pub(in crate::api) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/jobs", post(create_job).get(list_jobs))
        .route("/v1/jobs/cancel", post(cancel_jobs))
        .route("/v1/jobs/{job_id}/events", get(job_events))
        .route("/v1/jobs/{job_id}", get(get_job).delete(cancel_job))
}

pub(super) struct JobEventStream {
    pub(super) inner: WatchStream<JobResponse>,
    pub(super) finished: bool,
}

impl Stream for JobEventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.inner).poll_next(context) {
            Poll::Ready(Some(response)) => {
                self.finished = response.status.is_terminal();
                let data = serde_json::to_string(&response)
                    .expect("serializing a job response should not fail");
                Poll::Ready(Some(Ok(Event::default().event("snapshot").data(data))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

pub(super) async fn job_events(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let job = state
        .jobs
        .get(parse_job_id(&job_id)?)
        .ok_or_else(ApiError::job_not_found)?;
    let stream = JobEventStream {
        inner: WatchStream::new(job.subscribe()),
        finished: false,
    };
    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

pub(super) async fn list_jobs(State(state): State<AppState>) -> Json<JobListResponse> {
    Json(state.jobs.list())
}

pub(super) async fn create_job(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    ApiJson(value): ApiJson<serde_json::Value>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    let key = headers
        .get("x-nicegal-request-id")
        .map(|value| value.to_str().map(str::to_owned))
        .transpose()
        .map_err(|_| ApiError::bad_request("invalid request ID"))?;
    if let Some(key) = &key {
        validate_identifier(key, MAX_REQUEST_ID_LEN, "invalid request ID")?;
    }
    let fingerprint =
        serde_json::to_string(&value).map_err(|error| ApiError::internal(error.into()))?;
    let request: JobRequest =
        serde_json::from_value(value).map_err(|error| ApiError::bad_request(error.to_string()))?;
    let spec = request.prepare()?;
    let mut requests = state.jobs.requests.lock().await;
    requests.sweep();
    if let Some(previous) = key.as_ref().and_then(|key| requests.get(key)) {
        let job_id = previous.job_id.ok_or_else(ApiError::job_cancelled)?;
        if previous.fingerprint != fingerprint {
            return Err(ApiError::bad_request(
                "request ID was already used for another job",
            ));
        }
        let job = state.jobs.get(job_id).ok_or_else(ApiError::job_not_found)?;
        return Ok((StatusCode::ACCEPTED, Json(job.response())));
    }
    if key.as_ref().is_some_and(|key| !requests.has_room_for(key)) {
        return Err(ApiError::bad_request("too many retained job requests"));
    }
    let job = start_job(&state, spec).await?;
    if let Some(key) = key {
        requests.insert(
            key,
            RetainedRequest {
                fingerprint,
                job_id: Some(job.id),
                created: Instant::now(),
            },
        );
    }
    Ok((StatusCode::ACCEPTED, Json(job.response())))
}

/// Start a job after checking that the library it names exists, so an unknown library is a
/// `404` rather than a job that fails.
pub(in crate::api) async fn start_job(
    state: &AppState,
    spec: JobSpec,
) -> Result<Arc<Job>, ApiError> {
    if let Some(library_id) = spec.library_id() {
        let databases = Arc::clone(&state.databases);
        crate::api::run_blocking(move || {
            match databases.open_assets_read_only()?.library(library_id)? {
                Some(_) => Ok(()),
                None => Err(ApiError::library_not_found(library_id)),
            }
        })
        .await?;
    }
    state.jobs.start(spec)
}

pub(super) async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobResponse>, ApiError> {
    let job = state
        .jobs
        .get(parse_job_id(&job_id)?)
        .ok_or_else(ApiError::job_not_found)?;
    Ok(Json(job.response()))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct CancelJobsRequest {
    request_ids: Vec<String>,
}

pub(super) async fn cancel_jobs(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<CancelJobsRequest>,
) -> Result<Json<JobListResponse>, ApiError> {
    for id in &request.request_ids {
        validate_identifier(id, MAX_REQUEST_ID_LEN, "invalid request ID")?;
    }
    // Job acceptance holds this mutex through validation and admission. Stop either
    // cancels that admitted job or leaves a tombstone for a request still in transit.
    let mut requests = state.jobs.requests.lock().await;
    requests.sweep();
    let missing: std::collections::BTreeSet<_> = request
        .request_ids
        .iter()
        .filter(|id| requests.get(id).is_none())
        .cloned()
        .collect();
    if requests.iter().count() + missing.len() > crate::api::ttl_map::MAX_ENTRIES {
        return Err(ApiError::bad_request("too many retained job requests"));
    }
    for id in missing {
        requests.insert(
            id,
            RetainedRequest {
                fingerprint: String::new(),
                job_id: None,
                created: Instant::now(),
            },
        );
    }
    Ok(Json(state.jobs.stop()))
}

pub(super) async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    let job = state
        .jobs
        .cancel(parse_job_id(&job_id)?)
        .ok_or_else(ApiError::job_not_found)?;
    let response = job.response();
    let status = if response.status.is_terminal() {
        StatusCode::OK
    } else {
        StatusCode::ACCEPTED
    };
    Ok((status, Json(response)))
}

pub(super) fn wire_job_id(id: u64) -> String {
    format!("{}-{id}", crate::api::instance_id())
}

pub(super) fn parse_job_id(value: &str) -> Result<u64, ApiError> {
    let value = value
        .strip_prefix(&format!("{}-", crate::api::instance_id()))
        .ok_or_else(ApiError::job_not_found)?;
    let id = value
        .parse::<u64>()
        .map_err(|_| ApiError::bad_request("job identifier must be a positive integer"))?;
    if id == 0 {
        return Err(ApiError::bad_request(
            "job identifier must be a positive integer",
        ));
    }
    Ok(id)
}
