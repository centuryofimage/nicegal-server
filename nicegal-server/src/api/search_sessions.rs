//! Ordering belongs to the search service, independently of HTTP or Electron delivery order.
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use axum::{Json, http::HeaderMap};
use nicegal_core::cancellation::SearchCancellation;
use parking_lot::Mutex;
use serde::Deserialize;

use super::{error::ApiError, extract::ApiJson};

static SESSIONS: LazyLock<Mutex<HashMap<String, Session>>> = LazyLock::new(Mutex::default);
const RETENTION: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_SESSIONS: usize = 4096;

struct Session {
    generation: u64,
    closed: bool,
    touched: Instant,
    lanes: HashMap<String, (u64, SearchCancellation)>,
    next_request: u64,
}

impl Session {
    fn cancel(&mut self) {
        for (_, token) in self.lanes.values() {
            token.cancel();
        }
        self.lanes.clear();
        self.closed = true;
    }
}

pub(super) struct SearchLease {
    pub(super) cancellation: SearchCancellation,
    key: Option<(String, String, u64)>,
}

impl Drop for SearchLease {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some((client, lane, request)) = &self.key {
            let mut sessions = SESSIONS.lock();
            if let Some(session) = sessions.get_mut(client) {
                if session.lanes.get(lane).is_some_and(|(id, _)| id == request) {
                    session.lanes.remove(lane);
                }
                session.touched = Instant::now();
            }
        }
    }
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>, ApiError> {
    headers
        .get(name)
        .map(|value| {
            value
                .to_str()
                .map_err(|_| ApiError::bad_request("invalid search session header"))
        })
        .transpose()
}

fn validate_client(client: &str) -> Result<(), ApiError> {
    if client.is_empty()
        || client.len() > 128
        || !client
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_:".contains(&b))
    {
        return Err(ApiError::bad_request("invalid search client"));
    }
    Ok(())
}

fn session<'a>(
    sessions: &'a mut HashMap<String, Session>,
    client: &str,
) -> Result<&'a mut Session, ApiError> {
    sessions.retain(|_, s| !s.lanes.is_empty() || s.touched.elapsed() < RETENTION);
    if !sessions.contains_key(client) && sessions.len() >= MAX_SESSIONS {
        return Err(ApiError::bad_request("too many search sessions"));
    }
    Ok(sessions
        .entry(client.to_owned())
        .or_insert_with(|| Session {
            generation: 0,
            closed: false,
            touched: Instant::now(),
            lanes: HashMap::new(),
            next_request: 0,
        }))
}

pub(super) fn begin(headers: &HeaderMap) -> Result<SearchLease, ApiError> {
    let client = header(headers, "x-nicegal-search-client")?;
    let generation = header(headers, "x-nicegal-search-generation")?;
    let lane = header(headers, "x-nicegal-search-lane")?;
    let cancellation = SearchCancellation::default();
    if client.is_none() && generation.is_none() && lane.is_none() {
        return Ok(SearchLease {
            cancellation,
            key: None,
        });
    }
    let (Some(client), Some(generation), Some(lane)) = (client, generation, lane) else {
        return Err(ApiError::bad_request(
            "search client, generation and lane must be supplied together",
        ));
    };
    validate_client(client)?;
    let generation = generation
        .parse::<u64>()
        .map_err(|_| ApiError::bad_request("invalid search generation"))?;
    if !matches!(lane, "literal" | "files" | "meaning" | "visual") {
        return Err(ApiError::bad_request("invalid search lane"));
    }
    let mut sessions = SESSIONS.lock();
    let session = session(&mut sessions, client)?;
    if generation < session.generation || (generation == session.generation && session.closed) {
        return Err(ApiError::search_superseded());
    }
    if generation > session.generation {
        session.cancel();
        session.generation = generation;
        session.closed = false;
    }
    session.next_request += 1;
    let request = session.next_request;
    if let Some((_, previous)) = session
        .lanes
        .insert(lane.to_owned(), (request, cancellation.clone()))
    {
        previous.cancel();
    }
    session.touched = Instant::now();
    Ok(SearchLease {
        cancellation,
        key: Some((client.to_owned(), lane.to_owned(), request)),
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct CancelRequest {
    client: String,
    through_generation: u64,
}

pub(super) async fn cancel(ApiJson(request): ApiJson<CancelRequest>) -> Result<Json<()>, ApiError> {
    validate_client(&request.client)?;
    let mut sessions = SESSIONS.lock();
    let session = session(&mut sessions, &request.client)?;
    if session.generation <= request.through_generation {
        session.cancel();
        session.generation = request.through_generation;
    }
    session.touched = Instant::now();
    Ok(Json(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(client: &str, generation: u64, lane: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-nicegal-search-client", client.parse().unwrap());
        headers.insert(
            "x-nicegal-search-generation",
            generation.to_string().parse().unwrap(),
        );
        headers.insert("x-nicegal-search-lane", lane.parse().unwrap());
        headers
    }

    async fn cancel_through(client: &str, generation: u64) {
        let _ = cancel(ApiJson(CancelRequest {
            client: client.into(),
            through_generation: generation,
        }))
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn delayed_cancellation_and_requests_cannot_replace_a_newer_search() {
        let client = "test-search-ordering";
        let old = begin(&headers(client, 1, "literal")).unwrap();
        let current = begin(&headers(client, 2, "literal")).unwrap();
        assert!(old.cancellation.check().is_err());
        cancel_through(client, 1).await;
        assert!(begin(&headers(client, 1, "visual")).is_err());
        assert!(current.cancellation.check().is_ok());
        cancel_through(client, 2).await;
        assert!(current.cancellation.check().is_err());
        assert!(begin(&headers(client, 2, "visual")).is_err());
        assert!(begin(&headers(client, 3, "visual")).is_ok());
    }

    #[tokio::test]
    async fn cancellation_before_request_arrival_is_remembered() {
        let client = "test-search-cancel-first";
        cancel_through(client, 8).await;
        assert!(begin(&headers(client, 7, "files")).is_err());
        assert!(begin(&headers(client, 8, "files")).is_err());
        assert!(begin(&headers(client, 9, "files")).is_ok());
    }

    #[tokio::test]
    async fn replacing_a_lane_preserves_other_lanes_and_clients_even_after_old_drop() {
        let client = "test-search-lanes";
        let old = begin(&headers(client, 1, "literal")).unwrap();
        let sibling = begin(&headers(client, 1, "visual")).unwrap();
        let other = begin(&headers("test-search-other-client", 1, "literal")).unwrap();
        let replacement = begin(&headers(client, 1, "literal")).unwrap();
        assert!(old.cancellation.check().is_err());
        drop(old);
        assert!(replacement.cancellation.check().is_ok());
        assert!(sibling.cancellation.check().is_ok());
        cancel_through(client, 1).await;
        assert!(
            replacement.cancellation.check().is_err(),
            "old lease must not remove its replacement"
        );
        assert!(sibling.cancellation.check().is_err());
        assert!(other.cancellation.check().is_ok());
    }

    #[test]
    fn invalid_metadata_does_not_cancel_active_work_and_drop_cancels_its_worker() {
        let client = "test-search-invalid";
        let active = begin(&headers(client, 1, "meaning")).unwrap();
        assert!(begin(&headers(client, 2, "unknown")).is_err());
        let mut incomplete = headers(client, 2, "meaning");
        incomplete.remove("x-nicegal-search-lane");
        assert!(begin(&incomplete).is_err());
        assert!(active.cancellation.check().is_ok());
        let worker = active.cancellation.clone();
        drop(active);
        assert!(worker.check().is_err());
    }
}
