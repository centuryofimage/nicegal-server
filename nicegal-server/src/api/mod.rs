mod assets;
mod catalog;
mod error;
mod extract;
mod image_embeddings;
mod indexing;
mod jobs;
pub(crate) mod models;
mod ocr_models;
mod prune_jobs;
mod roots;
mod runtime;
mod search;
mod text_embeddings;
mod thumbnails;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::HeaderValue;
use axum::http::header::AUTHORIZATION;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use camino::Utf8PathBuf as PathBuf;
use models::{ImageModel, ImageQueryModel as ImageQueryEmbedder, TextModel as TextEmbedder};
mod external_image;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::db::DB;
use nicegal_core::image_index::ImageIndexDb;
use nicegal_core::thumbs::{ThumbnailDb, ThumbnailService};
use serde::Serialize;
use tower_http::trace::TraceLayer;
use tracing::{Span, debug_span, field, info, info_span};

use error::ApiError;

pub(crate) use jobs::JobManager;
pub(crate) use ocr_models::ModelStore;
pub(crate) use runtime::RuntimeSettings;

pub(crate) const API_VERSION: u8 = 1;
/// Process exit code an `ocrModelLoad` job uses to ask the desktop launcher for a clean restart
/// after silently downgrading the configured provider — see `ocr_models::job::run`. Distinct from
/// a crash so the launcher can respawn quietly instead of surfacing an error; the desktop side
/// checks for this exact value in `main/backend/nicegal-server-process.ts`.
pub(crate) const RESTART_EXIT_CODE: i32 = 75;
const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
/// A process-local correlation key for concurrent HTTP spans. It intentionally has no API
/// meaning and only needs to remain distinct for the lifetime of the process.
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// The three independent stores, addressed by path.
///
/// Query handlers open short-lived read-only connections inside the blocking pool. Thumbnail
/// mutations instead go through the process-wide service, whose writer owns its connection.
pub(crate) struct Databases {
    pub(crate) assets: PathBuf,
    pub(crate) images: PathBuf,
    pub(crate) ocr: PathBuf,
    pub(crate) thumbnails: PathBuf,
}
impl Databases {
    fn open_assets_read_only(&self) -> Result<AssetCatalog> {
        AssetCatalog::new_read_only(&self.assets)
    }

    fn open_ocr_read_only(&self) -> Result<DB> {
        DB::new_read_only(&self.ocr)
    }

    /// Open the image index for searching, with the asset catalog attached for filters and
    /// fingerprint currency. `dimensions` comes from the process's loaded image model.
    fn open_images_read_only(&self, dimensions: usize) -> Result<ImageIndexDb> {
        ImageIndexDb::new_read_only(&self.images, dimensions, &self.assets)
    }

    fn open_thumbnails_read_only(&self) -> Result<ThumbnailDb> {
        ThumbnailDb::new_read_only(&self.thumbnails)
    }
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) databases: Arc<Databases>,
    pub(crate) thumbnails: Arc<ThumbnailService>,
    pub(crate) jobs: Arc<JobManager>,
    /// Prepared on request by indexing or model preparation, then shared by searches.
    pub(crate) embedder: Arc<TextEmbedder>,
    /// The image engine's paired text encoder, the query half of the same CLIP model the image
    /// embedder encodes pictures with. Prepared with its pair, and on CPU: see
    /// `main` for why.
    pub(crate) image_query_embedder: Arc<ImageQueryEmbedder>,
    pub(crate) image_embedder: Arc<ImageModel>,
    pub(crate) ocr_models: Arc<ocr_models::ModelStore>,
    /// The process's selected ONNX Runtime distribution and the persisted selection for its next
    /// launch. It is separate from model state because a loaded dynamic library cannot change.
    pub(crate) runtime: Arc<RuntimeSettings>,
}

#[derive(Clone)]
struct AuthState {
    authorization: HeaderValue,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    api_version: u8,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusResponse {
    api_version: u8,
    runtime: runtime::RuntimeStatusResponse,
    ocr_models_loaded: bool,
}

pub(crate) fn router(state: AppState, authorization: HeaderValue) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/status", get(status))
        .route("/v1/models", models::route())
        .route("/v1/runtime", runtime::route())
        .route("/v1/assets", assets::route())
        .merge(catalog::routes())
        .route("/v1/thumbnails", thumbnails::route())
        .route("/v1/thumbnails/generate", thumbnails::generate_route())
        .route(
            "/v1/search",
            search::route().layer(DefaultBodyLimit::max(48 * 1024 * 1024)),
        )
        .route("/v1/text-embeddings", text_embeddings::route())
        .route(
            "/v1/text-embeddings/generate",
            text_embeddings::generate_route(),
        )
        .route("/v1/ocr/models", ocr_models::route())
        .merge(jobs::routes())
        .route_layer(middleware::from_fn_with_state(
            AuthState { authorization },
            authorize,
        ))
        // Unmatched routes answer with the same envelope as everything else, so a client never has
        // to parse a plain-text body to learn what went wrong.
        .method_not_allowed_fallback(method_not_allowed)
        .fallback(route_not_found)
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        // Outermost, so the span covers body limits and authorization rejections too, and so every
        // log line a handler emits is attributed to the request that caused it.
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request| {
                    let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
                    info_span!(
                        "http",
                        request_id,
                        method = %request.method(),
                        path = %request.uri().path(),
                        status = field::Empty,
                    )
                })
                // Status is both recorded onto the span (for a deep dive that turns span-close
                // events on for this target) and logged as its own event here: an explicit event
                // is the only thing that shows up on the console layer, which never renders span
                // open/close lines by design (see `nicegal_core::logging`) — this is the one
                // per-request line a human watching the desktop app's console actually sees.
                .on_request(())
                .on_response(|response: &Response, latency: Duration, span: &Span| {
                    let status = response.status().as_u16();
                    span.record("status", status);
                    info!(
                        status,
                        latency_ms = latency.as_millis() as u64,
                        "request completed"
                    );
                })
                .on_body_chunk(())
                .on_eos(()),
        )
        .with_state(state)
}

async fn authorize(State(auth): State<AuthState>, request: Request, next: Next) -> Response {
    if request.headers().get(AUTHORIZATION) != Some(&auth.authorization) {
        return ApiError::unauthorized().into_response();
    }
    next.run(request).await
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        api_version: API_VERSION,
    })
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    Json(StatusResponse {
        api_version: API_VERSION,
        runtime: runtime::status_response(&state.runtime),
        ocr_models_loaded: state.ocr_models.is_loaded(),
    })
}

async fn route_not_found() -> ApiError {
    ApiError::route_not_found()
}

async fn method_not_allowed() -> ApiError {
    ApiError::method_not_allowed()
}

/// Run one handler's blocking database or filesystem work on the blocking pool.
///
/// The task returns [`ApiError`] directly so a handler can classify its own failures; ordinary
/// [`anyhow`] errors still convert into `500 internal_error` through `?`. The current request
/// span crosses into the blocking worker, where a child span records both pool queueing and work.
async fn run_blocking<T>(
    task: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError>
where
    T: Send + 'static,
{
    let parent = Span::current();
    let queued_at = Instant::now();
    tokio::task::spawn_blocking(move || {
        let queue_wait = queued_at.elapsed();
        let _parent = parent.enter();
        let span = debug_span!("blocking_request", ?queue_wait, work_ms = field::Empty,);
        let _entered = span.enter();
        let work_started = Instant::now();
        let result = task();
        span.record("work_ms", work_started.elapsed().as_millis() as u64);
        result
    })
    .await
    .map_err(|error| ApiError::internal(anyhow!("blocking request task failed: {error}")))?
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode, header};
    use nicegal_core::assets::SourceFingerprint;
    use nicegal_core::db::OcrResult;
    use tempfile::TempDir;
    use tower::ServiceExt as _;

    use super::*;

    const TOKEN: &str = "Bearer test-token";

    pub(super) fn initialize_test_runtime() {
        #[cfg(windows)]
        {
            static RUNTIME: std::sync::Once = std::sync::Once::new();
            RUNTIME.call_once(|| {
                // Select the test binary's bundled CPU runtime without changing PATH in
                // the multithreaded harness. main() normally initializes this for us.
                let executable = std::env::current_exe().unwrap();
                nicegal_core::runtime::initialize_from_dylib(
                    &executable
                        .parent()
                        .unwrap()
                        .join("onnxruntime/directml/onnxruntime.dll"),
                )
                .expect("the bundled CPU runtime initializes");
            });
        }
    }

    /// A router over real databases holding one indexed OCR row, so the search tests
    /// exercise SQLite rather than a stub.
    fn test_router(temp: &TempDir) -> Router {
        test_router_with_preparation(temp, true)
    }

    fn test_router_with_preparation(temp: &TempDir, prepare_models: bool) -> Router {
        initialize_test_runtime();
        let directory = PathBuf::try_from(temp.path().to_path_buf()).unwrap();
        let databases = Arc::new(Databases {
            assets: directory.join("assets.db"),
            images: directory.join("qdrant-clip-vit-b-32.db"),
            ocr: directory.join("ocr.db"),
            thumbnails: directory.join("thumbnails.db"),
        });
        let mut ocr = DB::new(&databases.ocr).unwrap();
        ocr.save_results(vec![OcrResult {
            asset_id: 1,
            path: directory.join("receipt.png"),
            fingerprint: SourceFingerprint {
                modified_ns: 1,
                size: 2,
            },
            exif_taken_ns: None,
            width: 640,
            height: 480,
            contents: "hello world".to_owned(),
        }])
        .unwrap();
        drop(ocr);
        drop(AssetCatalog::new(&databases.assets).unwrap());
        drop(ThumbnailDb::new(&databases.thumbnails).unwrap());

        let embedder = Arc::new(
            TextEmbedder::deferred(nicegal_core::embedding::TextEmbedderOptions::default())
                .without_cached_loading(),
        );
        let image_embedder = Arc::new(ImageModel::deferred(
            nicegal_core::embedding::ImageEmbedderOptions::default(),
        ));
        let image_query_embedder = Arc::new(
            ImageQueryEmbedder::deferred(
                nicegal_core::embedding::ImageQueryEmbedderOptions::default(),
            )
            .without_cached_loading(),
        );
        if prepare_models {
            embedder.prepare().unwrap();
            image_embedder.prepare().unwrap();
            image_query_embedder.prepare().unwrap();
        }
        drop(ImageIndexDb::new(&databases.images, image_query_embedder.dimensions()).unwrap());
        let ocr_models = Arc::new(ModelStore::new(
            nicegal_core::runtime::ExecutionProvider::Cpu,
        ));
        let thumbnails = Arc::new(ThumbnailService::new(&databases.thumbnails).unwrap());
        let runtime = Arc::new(
            RuntimeSettings::load(
                databases.assets.parent().unwrap().join("runtime.json"),
                None,
            )
            .unwrap(),
        );
        runtime.set_onnx_runtime_build_info("test build".to_owned());
        let state = AppState {
            jobs: Arc::new(JobManager::new(
                Arc::clone(&databases),
                Arc::clone(&thumbnails),
                Arc::clone(&embedder),
                Arc::clone(&image_embedder),
                Arc::clone(&image_query_embedder),
                Arc::clone(&ocr_models),
                Arc::clone(&runtime),
            )),
            databases,
            thumbnails,
            embedder,
            image_query_embedder,
            image_embedder,
            ocr_models,
            runtime,
        };
        router(state, HeaderValue::from_static(TOKEN))
    }

    async fn send(router: &Router, method: Method, uri: &str) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::AUTHORIZATION, TOKEN)
            .body(Body::empty())
            .unwrap();
        response_parts(router, request).await
    }

    async fn send_json(
        router: &Router,
        method: Method,
        uri: &str,
        body: &'static str,
    ) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::AUTHORIZATION, TOKEN)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        response_parts(router, request).await
    }

    /// Every response body in this API is JSON; the helper fails loudly if one is not, which is
    /// how a leaked plain-text rejection would be caught.
    async fn response_parts(
        router: &Router,
        request: Request<Body>,
    ) -> (StatusCode, serde_json::Value) {
        let response = router.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = if bytes.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or_else(|error| {
                panic!(
                    "expected a JSON body, got {:?}: {error}",
                    String::from_utf8_lossy(&bytes)
                )
            })
        };
        (status, body)
    }

    fn search_uri(temp: &TempDir, query: &str, kind: &str) -> String {
        let root = temp.path().to_str().unwrap();
        format!(
            "/v1/search?q={}&type={kind}&root={}",
            urlencode(query),
            urlencode(root)
        )
    }

    fn urlencode(value: &str) -> String {
        value
            .bytes()
            .map(|byte| match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    (byte as char).to_string()
                }
                _ => format!("%{byte:02X}"),
            })
            .collect()
    }

    #[tokio::test]
    async fn unprepared_models_leave_browsing_and_literal_search_available() {
        let temp = TempDir::new().unwrap();
        let router = test_router_with_preparation(&temp, false);
        let (status, body) = send(&router, Method::GET, "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        for key in ["text", "clipImage", "clipText"] {
            assert_eq!(body[key]["state"], "notLoaded");
            assert!(body[key]["error"].is_null());
        }
        let (status, _) = send(&router, Method::GET, "/v1/health").await;
        assert_eq!(status, StatusCode::OK);
        let (status, body) =
            send(&router, Method::GET, &search_uri(&temp, "hello", "simple")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        for kind in ["vector", "image"] {
            let (status, body) =
                send(&router, Method::GET, &search_uri(&temp, "hello", kind)).await;
            assert_eq!(status, StatusCode::CONFLICT, "{kind}: {body}");
            assert_eq!(body["error"]["code"], "models_not_ready");
        }
        let (_, body) = send(&router, Method::GET, "/v1/models").await;
        for key in ["text", "clipImage", "clipText"] {
            assert_eq!(body[key]["state"], "notLoaded");
        }
    }

    #[tokio::test]
    async fn health_reports_the_api_version() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(&router, Method::GET, "/v1/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["apiVersion"], API_VERSION);
    }

    #[tokio::test]
    async fn runtime_status_reports_active_and_persisted_provider() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);

        let (status, body) = send(&router, Method::GET, "/v1/status").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["apiVersion"], API_VERSION);
        assert_eq!(body["runtime"]["activeExecutionProvider"], "directml");
        assert_eq!(body["runtime"]["activeRuntimeDistribution"], "directml");
        assert_eq!(body["runtime"]["configuredExecutionProvider"], "directml");
        assert_eq!(body["runtime"]["restartRequired"], false);
        assert_eq!(body["runtime"]["onnxRuntimeBuildInfo"], "test build");
        assert_eq!(body["ocrModelsLoaded"], false);

        let (status, body) = send_json(
            &router,
            Method::PUT,
            "/v1/runtime",
            r#"{"executionProvider":"openvino"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["activeExecutionProvider"], "directml");
        assert_eq!(body["activeRuntimeDistribution"], "directml");
        assert_eq!(body["configuredExecutionProvider"], "openvino");
        assert_eq!(body["restartRequired"], true);

        let (status, body) = send(&router, Method::GET, "/v1/runtime").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["configuredExecutionProvider"], "openvino");
        assert_eq!(body["restartRequired"], true);
    }

    #[tokio::test]
    async fn a_missing_or_wrong_token_is_rejected_with_the_envelope() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        for token in [None, Some("Bearer wrong")] {
            let mut request = Request::builder().method(Method::GET).uri("/v1/health");
            if let Some(token) = token {
                request = request.header(header::AUTHORIZATION, token);
            }
            let (status, body) =
                response_parts(&router, request.body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            assert_eq!(body["error"]["code"], "unauthorized");
        }
    }

    #[tokio::test]
    async fn unknown_routes_and_methods_answer_with_the_envelope() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);

        let (status, body) = send(&router, Method::GET, "/v1/nope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "not_found");

        let (status, body) = send(&router, Method::DELETE, "/v1/health").await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(body["error"]["code"], "method_not_allowed");
    }

    #[tokio::test]
    async fn malformed_query_strings_are_invalid_request_not_plain_text() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);

        // Axum's own Query rejection is plain text; the wrapper must convert it.
        let (status, body) = send(&router, Method::GET, "/v1/search?q=hello").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
        assert!(
            body["error"]["message"].as_str().unwrap().contains("root"),
            "{body}"
        );

        let (status, body) = send(
            &router,
            Method::GET,
            "/v1/thumbnails?assetId=notanumber&requestedSize=1&generatorVersion=1",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn malformed_json_bodies_are_invalid_request_not_plain_text() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);

        let (status, body) = send_json(&router, Method::POST, "/v1/jobs", "{not json").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");

        // A well-formed body of the wrong shape is still the caller's problem.
        let (status, body) =
            send_json(&router, Method::POST, "/v1/jobs", "{\"type\":\"nope\"}").await;
        assert!(status.is_client_error(), "{status}");
        assert_eq!(body["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn a_well_formed_search_returns_its_hits() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) =
            send(&router, Method::GET, &search_uri(&temp, "hello", "simple")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        assert_eq!(body["results"][0]["assetId"], 1);
    }

    #[tokio::test]
    async fn a_query_sqlite_cannot_parse_is_a_400_carrying_its_message() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        for (query, kind, expected) in [
            ("ocr:\"unterminated", "simple", "unterminated string"),
            ("AND", "match", "fts5: syntax error"),
            ("col:foo", "match", "no such column"),
        ] {
            let (status, body) = send(&router, Method::GET, &search_uri(&temp, query, kind)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{query}: {body}");
            assert_eq!(body["error"]["code"], "query_syntax", "{query}");
            let message = body["error"]["message"].as_str().unwrap();
            assert!(message.contains(expected), "{query}: {message}");
        }
    }

    #[tokio::test]
    async fn an_unusable_search_root_is_invalid_root() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let missing = temp.path().join("not-there");
        let uri = format!(
            "/v1/search?q=hello&type=image&root={}",
            urlencode(missing.to_str().unwrap())
        );
        let (status, body) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_root");

        let (status, body) = send(&router, Method::GET, "/v1/search?q=hello&root=relative").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_root");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("absolute"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn an_indexed_root_with_no_matches_is_an_empty_result_not_an_error() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(
            &router,
            Method::GET,
            &search_uri(&temp, "nothingmatchesthis", "simple"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 0);
        assert_eq!(body["results"].as_array().unwrap().len(), 0);
    }

    /// Give the router's OCR database vectors for its rows, the way a completed embed job would.
    fn embed_everything(temp: &TempDir) {
        let ocr_path = PathBuf::try_from(temp.path().join("ocr.db")).unwrap();
        let embedder = nicegal_core::embedding::TextEmbedder::load(
            &nicegal_core::embedding::TextEmbedderOptions::default(),
        )
        .unwrap();
        let mut ocr = DB::new(&ocr_path).unwrap();
        let space = nicegal_core::db::TextEmbeddingSpace::OcrText;
        ocr.set_text_embedding_model(space, embedder.model().id(), embedder.dimensions(), true)
            .unwrap();
        let root = PathBuf::try_from(temp.path().to_path_buf()).unwrap();
        let filters = nicegal_core::db::SearchFilters::new(&root);
        let pending = ocr
            .pending_text_embeddings(space, &filters, 100, 8192)
            .unwrap();
        let items: Vec<_> = pending
            .iter()
            .map(|row| nicegal_core::db::TextEmbedding {
                asset_id: row.asset_id,
                vector: embedder.embed_query(&row.content).unwrap(),
            })
            .collect();
        ocr.save_text_embeddings(space, embedder.model().id(), items)
            .unwrap();
    }

    async fn post_json(
        router: &Router,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header(header::AUTHORIZATION, TOKEN)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        response_parts(router, request).await
    }

    #[tokio::test]
    async fn vector_is_the_default_search_mode_and_is_empty_until_something_is_embedded() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let root = urlencode(temp.path().to_str().unwrap());

        // No `type=`: the default is vector, and nothing is embedded yet.
        let (status, body) = send(
            &router,
            Method::GET,
            &format!("/v1/search?q=hello&root={root}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 0);

        embed_everything(&temp);
        let (status, body) = send(
            &router,
            Method::GET,
            &format!("/v1/search?q=hello&root={root}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        assert_eq!(body["results"][0]["assetId"], 1);
        assert_eq!(body["results"][0]["rank"], 1);
        // Vector hits carry a distance; the text modes do not.
        assert!(body["results"][0]["distance"].is_number(), "{body}");

        let (status, body) =
            send(&router, Method::GET, &search_uri(&temp, "hello", "simple")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body["results"][0]["distance"].is_null(), "{body}");
    }

    #[tokio::test]
    async fn a_distance_ceiling_is_rejected_for_the_text_modes() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let root = urlencode(temp.path().to_str().unwrap());
        let (status, body) = send(
            &router,
            Method::GET,
            &format!("/v1/search?q=hello&type=simple&root={root}&maxDistance=0.5"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("vector"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_combined_search_answers_every_mode_and_fuses_them() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        embed_everything(&temp);

        let (status, body) = post_json(
            &router,
            "/v1/search",
            serde_json::json!({
                "root": temp.path().to_str().unwrap(),
                "queries": [
                    {"key": "semantic", "type": "vector", "q": "hello"},
                    {"key": "literal", "type": "simple", "q": "hello"},
                    {"key": "nothing", "type": "simple", "q": "zzzznomatch"}
                ],
                "fuse": {"method": "rrf"}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");

        let queries = body["queries"].as_array().unwrap();
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0]["key"], "semantic");
        assert_eq!(queries[0]["type"], "vector");
        assert_eq!(queries[0]["total"], 1);
        assert_eq!(queries[1]["total"], 1);
        assert_eq!(
            queries[2]["total"], 0,
            "a mode with no hits is not an error"
        );

        // The asset both modes found is the single fused hit, and it names both sources.
        assert_eq!(body["fused"]["total"], 1);
        assert_eq!(body["fused"]["results"][0]["assetId"], 1);
        assert_eq!(
            body["fused"]["results"][0]["sources"],
            serde_json::json!(["semantic", "literal"])
        );
        assert!(
            body["fused"]["results"][0]["score"].as_f64().unwrap() > 0.0,
            "{body}"
        );
        assert_eq!(
            body["model"]["dimensions"],
            nicegal_core::embedding::TextEmbeddingModel::default().dimensions()
        );
    }

    #[tokio::test]
    async fn a_combined_search_without_a_fuse_block_returns_the_lists_only() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = post_json(
            &router,
            "/v1/search",
            serde_json::json!({
                "root": temp.path().to_str().unwrap(),
                "queries": [{"type": "simple", "q": "hello"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // The key defaults to the mode name.
        assert_eq!(body["queries"][0]["key"], "simple");
        assert!(body.get("fused").is_none(), "{body}");
        // Nothing has been embedded, so there is no stored model to report.
        assert!(body["model"].is_null(), "{body}");
    }

    #[tokio::test]
    async fn a_combined_search_rejects_an_empty_or_ambiguous_query_set() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let root = temp.path().to_str().unwrap();

        for body in [
            serde_json::json!({"root": root, "queries": []}),
            serde_json::json!({"root": root, "queries": [
                {"key": "same", "type": "simple", "q": "a"},
                {"key": "same", "type": "glob", "q": "b"}
            ]}),
            serde_json::json!({"root": root, "queries": [
                {"type": "simple", "q": "a", "weight": 0}
            ]}),
        ] {
            let (status, response) = post_json(&router, "/v1/search", body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {response}");
            assert_eq!(response["error"]["code"], "invalid_request", "{body}");
        }
    }

    #[tokio::test]
    async fn a_query_sqlite_cannot_parse_is_still_query_syntax_in_the_combined_form() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = post_json(
            &router,
            "/v1/search",
            serde_json::json!({
                "root": temp.path().to_str().unwrap(),
                "queries": [{"type": "match", "q": "AND"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "query_syntax");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("fts5: syntax error"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn embedding_status_separates_an_unembedded_root_from_an_empty_one() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let uri = format!(
            "/v1/text-embeddings?root={}",
            urlencode(temp.path().to_str().unwrap())
        );

        let (status, body) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["embedder"]["model"],
            nicegal_core::embedding::TextEmbeddingModel::default().id()
        );
        assert_eq!(
            body["embedder"]["dimensions"],
            nicegal_core::embedding::TextEmbeddingModel::default().dimensions()
        );
        assert!(body["stored"].is_null(), "nothing has been embedded yet");
        assert_eq!(body["indexed"], 1);
        assert_eq!(body["embedded"], 0);
        assert_eq!(body["pending"], 1);
        // The fixture's one OCR row is stamped `modified_ns: 1`, well under a second.
        assert_eq!(body["lastIndexedAt"], 0, "{body}");

        embed_everything(&temp);
        let (status, body) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            body["stored"]["model"],
            nicegal_core::embedding::TextEmbeddingModel::default().id()
        );
        assert_eq!(body["embedded"], 1);
        assert_eq!(body["pending"], 0);
    }

    #[tokio::test]
    async fn an_embed_backfill_starts_a_job_and_a_bad_root_does_not() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);

        let (status, body) = post_json(
            &router,
            "/v1/text-embeddings/generate",
            serde_json::json!({"root": temp.path().to_str().unwrap()}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{body}");
        assert_eq!(body["type"], "textEmbed");

        let (status, body) = post_json(
            &router,
            "/v1/text-embeddings/generate",
            serde_json::json!({"root": "relative/path"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_root");
    }

    #[tokio::test]
    async fn a_time_bound_without_a_timeline_is_refused() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let root = urlencode(temp.path().to_str().unwrap());

        // There is no default timeline: capture and modified answer different questions, so the
        // server refuses to pick one on the caller's behalf.
        for query in ["after=0", "before=100", "after=0&before=100"] {
            let (status, body) = send(
                &router,
                Method::GET,
                &format!("/v1/search?q=hello&type=simple&root={root}&{query}"),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{query}: {body}");
            assert_eq!(body["error"]["code"], "invalid_request", "{query}");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("timeline is required"),
                "{query}: {body}"
            );
        }

        // A timeline with no bound is inert rather than an error, so a client that always sends
        // its preference does not have to special-case an unbounded search.
        let (status, body) = send(
            &router,
            Method::GET,
            &format!("/v1/search?q=hello&type=simple&root={root}&timeline=capture"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
    }

    #[tokio::test]
    async fn time_bounds_narrow_every_search_mode() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        embed_everything(&temp);
        let root = urlencode(temp.path().to_str().unwrap());
        // The fixture row has source_modified_ns = 1 and no EXIF capture time.
        let inside = "after=0&before=2&timeline=modified";
        let outside = "after=2&before=9&timeline=modified";

        for kind in ["simple", "glob", "vector"] {
            let query = if kind == "glob" { "*hello*" } else { "hello" };
            let (status, body) = send(
                &router,
                Method::GET,
                &format!(
                    "/v1/search?q={}&type={kind}&root={root}&{inside}",
                    urlencode(query)
                ),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{kind}: {body}");
            assert_eq!(body["total"], 1, "{kind}: {body}");

            let (status, body) = send(
                &router,
                Method::GET,
                &format!(
                    "/v1/search?q={}&type={kind}&root={root}&{outside}",
                    urlencode(query)
                ),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{kind}: {body}");
            assert_eq!(body["total"], 0, "{kind} ignored the range: {body}");
        }
    }

    #[tokio::test]
    async fn instants_are_decimal_strings_and_a_reversed_range_is_refused() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let root = urlencode(temp.path().to_str().unwrap());

        let (status, body) = send(
            &router,
            Method::GET,
            &format!(
                "/v1/search?q=hello&type=simple&root={root}&after=notanumber&timeline=modified"
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Unix nanoseconds"),
            "{body}"
        );

        let (status, body) = send(
            &router,
            Method::GET,
            &format!(
                "/v1/search?q=hello&type=simple&root={root}&after=9&before=2&timeline=modified"
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("after must be less than before"),
            "{body}"
        );

        // Nanosecond precision survives the round trip, which a JSON number would not.
        let (status, body) = send(
            &router,
            Method::GET,
            &format!(
                "/v1/search?q=hello&type=simple&root={root}&after=1717243200123456789&timeline=capture"
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 0);
    }

    #[tokio::test]
    async fn a_combined_search_applies_the_request_range_and_lets_one_query_override_it() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        embed_everything(&temp);

        let (status, body) = post_json(
            &router,
            "/v1/search",
            serde_json::json!({
                "root": temp.path().to_str().unwrap(),
                "timeline": "modified",
                "after": "2",
                "queries": [
                    {"key": "inherits", "type": "simple", "q": "hello"},
                    // Overriding replaces the whole filter rather than merging, so this query does
                    // not silently inherit `after` from the request.
                    {"key": "overrides", "type": "simple", "q": "hello",
                     "timeline": "modified", "after": "0", "before": "2"},
                    {"key": "vector", "type": "vector", "q": "hello"}
                ]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let queries = body["queries"].as_array().unwrap();
        assert_eq!(queries[0]["total"], 0, "the request range excludes the row");
        assert_eq!(queries[1]["total"], 1, "the override replaces it entirely");
        assert_eq!(
            queries[2]["total"], 0,
            "vector inherits the request range too"
        );
    }

    #[tokio::test]
    async fn a_combined_query_override_still_needs_its_own_timeline() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = post_json(
            &router,
            "/v1/search",
            serde_json::json!({
                "root": temp.path().to_str().unwrap(),
                "timeline": "modified",
                "after": "0",
                "queries": [{"key": "partial", "type": "simple", "q": "hello", "before": "2"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "invalid_request");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("partial"),
            "the message names the query at fault: {body}"
        );
    }

    #[tokio::test]
    async fn a_missing_asset_lookup_is_asset_not_found() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let missing = temp.path().join("absent.png");
        let uri = format!("/v1/assets?path={}", urlencode(missing.to_str().unwrap()));
        let (status, body) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "asset_not_found");
    }

    #[tokio::test]
    async fn catalog_and_metadata_work_without_models_or_source_files() {
        let temp = TempDir::new().unwrap();
        let router = test_router_with_preparation(&temp, false);
        let root = PathBuf::try_from(temp.path().to_path_buf()).unwrap();
        let source = root.join("photo.png");
        std::fs::write(&source, []).unwrap();
        let catalog = AssetCatalog::new(&root.join("assets.db")).unwrap();
        let asset = catalog
            .upsert(&source, &std::fs::metadata(&source).unwrap())
            .unwrap();
        catalog.record_decode_failure(&asset).unwrap();
        let mut ocr = DB::new(&root.join("ocr.db")).unwrap();
        ocr.save_results(vec![OcrResult {
            asset_id: asset.asset_id,
            path: source.clone(),
            fingerprint: asset.fingerprint,
            exif_taken_ns: asset.exif_taken_ns,
            width: 1,
            height: 1,
            contents: "Total: $23.50".to_owned(),
        }])
        .unwrap();
        std::fs::remove_file(&source).unwrap();
        let uri = format!(
            "/v1/catalog?root={}&timeline=capture",
            urlencode(root.as_str())
        );
        let (status, listing) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(listing[0]["id"], asset.asset_id.to_string());
        assert_eq!(listing[0]["sourceSize"], "0");
        assert_eq!(
            listing[0]["modifiedNs"],
            asset.fingerprint.modified_ns.to_string()
        );
        let (_, count) = send(
            &router,
            Method::GET,
            &format!("/v1/catalog/count?root={}", urlencode(root.as_str())),
        )
        .await;
        assert_eq!(count, 1);
        let (_, revision) = send(&router, Method::GET, "/v1/catalog/revision").await;
        assert_eq!(revision, "1");
        let (status, info) = send(
            &router,
            Method::GET,
            &format!("/v1/catalog/metadata?assetId={}", asset.asset_id),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{info}");
        assert_eq!(info["asset"], listing[0]);
        assert_eq!(info["file"]["sourceState"], "missing");
        assert_eq!(info["decodeFailed"], true);
        assert_eq!(info["imageIndexed"], false);
        assert_eq!(info["ocrState"], "indexed");
        assert_eq!(info["ocrText"], "Total: $23.50");
        assert_eq!(info["textState"], "pending");
        let (status, _) = send(&router, Method::GET, "/v1/catalog/metadata?assetId=999").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        for uri in [
            "/v1/catalog?root=relative",
            "/v1/catalog/count?root=relative",
            "/v1/catalog/metadata?assetId=0",
            "/v1/catalog?root=C%3A%2F&timeline=wrong",
        ] {
            assert_eq!(
                send(&router, Method::GET, uri).await.0,
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[tokio::test]
    async fn thumbnail_parameters_are_validated_before_any_lookup() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(
            &router,
            Method::GET,
            "/v1/thumbnails?assetId=0&requestedSize=100&generatorVersion=1",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");

        let (status, body) = send_json(
            &router,
            Method::POST,
            "/v1/thumbnails",
            "{\"assetIds\":[1],\"requiredSize\":4096}",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn an_unknown_job_is_job_not_found_and_a_bad_id_is_invalid_request() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);

        let (status, body) = send(&router, Method::GET, "/v1/jobs/99").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "job_not_found");

        let (status, body) = send(&router, Method::GET, "/v1/jobs/abc").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
    }
}
