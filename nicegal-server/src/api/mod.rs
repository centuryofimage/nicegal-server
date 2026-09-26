mod assets;
mod catalog;
mod error;
mod extract;
mod image_embeddings;
mod image_model;
mod jobs;
mod libraries;
mod library_scan;
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

pub(crate) use image_model::ImageModelSettings;
pub(crate) use jobs::JobManager;
pub(crate) use ocr_models::ModelStore;
pub(crate) use runtime::RuntimeSettings;

pub(crate) const API_VERSION: u8 = 1;
/// Requests a clean launcher restart after the configured provider changes.
pub(crate) const RESTART_EXIT_CODE: i32 = 75;
const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
/// A process-local correlation key for concurrent HTTP spans. It intentionally has no API
/// meaning and only needs to remain distinct for the lifetime of the process.
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// The four independent stores, addressed by path.
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
    /// Text encoder paired with the active image embedding model.
    pub(crate) image_query_embedder: Arc<ImageQueryEmbedder>,
    pub(crate) image_embedder: Arc<ImageModel>,
    pub(crate) ocr_models: Arc<ocr_models::ModelStore>,
    /// The process's selected ONNX Runtime distribution and the persisted selection for its next
    /// launch. It is separate from model state because a loaded dynamic library cannot change.
    pub(crate) runtime: Arc<RuntimeSettings>,
    pub(crate) image_model_settings: Arc<ImageModelSettings>,
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
        .merge(libraries::routes())
        .route("/v1/thumbnails", thumbnails::route())
        .route("/v1/thumbnails/generate", thumbnails::generate_route())
        .route(
            "/v1/search",
            search::route().layer(DefaultBodyLimit::max(48 * 1024 * 1024)),
        )
        .route("/v1/text-embeddings", text_embeddings::route())
        .route("/v1/image-embeddings", image_embeddings::route())
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
        runtime: runtime::status_response(&state),
        ocr_models_loaded: state.ocr_models.is_loaded(),
    })
}

async fn route_not_found() -> ApiError {
    ApiError::route_not_found()
}

async fn method_not_allowed() -> ApiError {
    ApiError::method_not_allowed()
}

/// Cancels request-owned work when its handler is dropped; already-started shared work may finish.
async fn run_cancellable<T: Send + 'static>(
    task: impl FnOnce(nicegal_core::cancellation::SearchCancellation) -> Result<T, ApiError>
    + Send
    + 'static,
) -> Result<T, ApiError> {
    use nicegal_core::cancellation::SearchCancellation;
    struct CancelOnDrop(Option<SearchCancellation>);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            if let Some(cancellation) = &self.0 {
                cancellation.cancel();
                tracing::debug!("cancelled abandoned request");
            }
        }
    }
    let cancellation = SearchCancellation::default();
    let mut guard = CancelOnDrop(Some(cancellation.clone()));
    let result = run_blocking(move || {
        cancellation.check()?;
        task(cancellation)
    })
    .await;
    guard.0 = None;
    result
}

/// Run one handler's blocking database or filesystem work on the blocking pool.
///
/// Handler-classified failures remain [`ApiError`]; other errors become `internal_error`.
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
        // Elsewhere the bundled loader leaves the environment alone; dev.sh copies the
        // runtime next to test binaries in target/<profile>/deps.
        #[cfg(not(windows))]
        {
            static RUNTIME: std::sync::Once = std::sync::Once::new();
            RUNTIME.call_once(|| {
                nicegal_core::runtime::initialize_bundled_runtime(
                    nicegal_core::runtime::ExecutionProvider::Cpu,
                )
                .expect("the bundled CPU runtime initializes");
            });
        }
    }

    /// A router over real databases holding one indexed OCR row, so the search tests
    /// exercise SQLite rather than a stub.
    fn test_router(temp: &TempDir) -> Router {
        test_router_with_preparation(temp, false)
    }

    fn prepared_text_model() -> Arc<TextEmbedder> {
        static MODEL: std::sync::OnceLock<Arc<TextEmbedder>> = std::sync::OnceLock::new();
        Arc::clone(MODEL.get_or_init(|| {
            initialize_test_runtime();
            let model = Arc::new(TextEmbedder::deferred(
                nicegal_core::embedding::TextEmbedderOptions::default(),
            ));
            model.prepare().unwrap();
            model
        }))
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
        AssetCatalog::new(&databases.assets)
            .unwrap()
            .create_library(
                &nicegal_core::libraries::LibraryDefinition {
                    include: vec![directory.clone()],
                    exclude: Vec::new(),
                    options: Default::default(),
                },
                None,
            )
            .unwrap();
        drop(ThumbnailDb::new(&databases.thumbnails).unwrap());

        let embedder = if prepare_models {
            prepared_text_model()
        } else {
            Arc::new(
                TextEmbedder::deferred(nicegal_core::embedding::TextEmbedderOptions::default())
                    .without_cached_loading(),
            )
        };
        let image_embedder = Arc::new(ImageModel::deferred(
            nicegal_core::embedding::ImageEmbedderOptions::default(),
        ));
        let image_query_embedder = Arc::new(
            ImageQueryEmbedder::deferred(
                nicegal_core::embedding::ImageQueryEmbedderOptions::default(),
            )
            .without_cached_loading(),
        );
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
        let image_model_settings = Arc::new(ImageModelSettings::new(Arc::clone(&runtime), None));
        let state = AppState {
            image_model_settings,
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

    /// Every test router has one library, over its temp directory.
    const FIXTURE_LIBRARY: i64 = 1;

    fn search_uri(query: &str, kind: &str) -> String {
        format!(
            "/v1/search?q={}&type={kind}&libraryId={FIXTURE_LIBRARY}",
            urlencode(query)
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
        let (status, body) = send(&router, Method::GET, &search_uri("hello", "ocrSimple")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        for kind in ["vector", "image"] {
            let (status, body) = send(&router, Method::GET, &search_uri("hello", kind)).await;
            assert_eq!(status, StatusCode::CONFLICT, "{kind}: {body}");
            assert_eq!(body["error"]["code"], "models_not_ready");
        }
        let (_, body) = send(&router, Method::GET, "/v1/models").await;
        for key in ["text", "clipImage", "clipText"] {
            assert_eq!(body[key]["state"], "notLoaded");
        }
    }

    #[tokio::test]
    async fn path_search_uses_catalog_without_preparing_models() {
        let temp = TempDir::new().unwrap();
        let router = test_router_with_preparation(&temp, false);
        let directory = PathBuf::try_from(temp.path().join("Trips 2025")).unwrap();
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("Café.jpg");
        std::fs::write(&path, b"image").unwrap();
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db")).unwrap();
        let mut catalog = AssetCatalog::new(&catalog_path).unwrap();
        let asset = catalog.upsert(&path, &path.metadata().unwrap()).unwrap();
        let root = PathBuf::try_from(temp.path().to_path_buf()).unwrap();
        let empty = root.join("Empty folder");
        std::fs::create_dir_all(&empty).unwrap();
        catalog
            .replace_directory_snapshot(
                FIXTURE_LIBRARY,
                &root,
                "[]",
                &[
                    nicegal_core::libraries::DirectorySnapshot {
                        path: root.clone(),
                        modified_ns: 0,
                    },
                    nicegal_core::libraries::DirectorySnapshot {
                        path: directory.clone(),
                        modified_ns: 0,
                    },
                    nicegal_core::libraries::DirectorySnapshot {
                        path: empty.clone(),
                        modified_ns: 1_790_000_000_000_000_000,
                    },
                ],
            )
            .unwrap();
        drop(catalog);

        let (status, folders) = send(
            &router,
            Method::GET,
            &format!("/v1/catalog/folders?libraryId={FIXTURE_LIBRARY}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{folders}");
        assert!(folders.as_array().unwrap().contains(&serde_json::json!({
            "path": empty.as_str(),
            "modifiedNs": "1790000000000000000",
        })));

        let (status, body) = send(&router, Method::GET, &search_uri("TRIPS", "path")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        assert_eq!(body["results"][0]["assetId"], asset.asset_id);
        assert_eq!(body["results"][0]["snippet"], path.as_str());
        let (status, body) = send(&router, Method::GET, &search_uri("CAFÉ", "name")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        assert_eq!(body["results"][0]["snippet"], "Café.jpg");
        let (status, body) = send(&router, Method::GET, &search_uri("Trips", "name")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 0);
        let other = PathBuf::try_from(temp.path().join("Trips 2025-old")).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        let other_path = other.join("Café.jpg");
        std::fs::write(&other_path, b"image").unwrap();
        AssetCatalog::new(&catalog_path)
            .unwrap()
            .upsert(&other_path, &other_path.metadata().unwrap())
            .unwrap();
        let focused = format!(
            "{}&folder={}",
            search_uri("CAFÉ", "name"),
            urlencode(directory.as_str())
        );
        let (status, body) = send(&router, Method::GET, &focused).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        assert_eq!(body["results"][0]["assetId"], asset.asset_id);
        let (status, body) = send(&router, Method::GET, "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["text"]["state"], "notLoaded");
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
        // The platform's default provider: DirectML on Windows, WebGPU or OpenVINO on Linux.
        let default =
            RuntimeSettings::load(temp.path().join("default.json").try_into().unwrap(), None)
                .unwrap()
                .active_execution_provider();
        let (provider, distribution) =
            (default.to_string(), runtime::runtime_distribution(default));

        let (status, body) = send(&router, Method::GET, "/v1/status").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["apiVersion"], API_VERSION);
        assert_eq!(body["runtime"]["activeExecutionProvider"], provider);
        assert_eq!(body["runtime"]["activeRuntimeDistribution"], distribution);
        assert_eq!(body["runtime"]["configuredExecutionProvider"], provider);
        #[cfg(all(windows, feature = "ort-cuda"))]
        assert!(
            body["runtime"]["availableExecutionProviders"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("cuda"))
        );
        assert_eq!(body["runtime"]["restartRequired"], false);
        assert_eq!(body["runtime"]["onnxRuntimeBuildInfo"], "test build");
        assert_eq!(body["ocrModelsLoaded"], false);

        let (status, body) = send_json(
            &router,
            Method::PUT,
            "/v1/runtime",
            // CPU is always compiled and never the default beside an accelerated provider.
            r#"{"executionProvider":"cpu"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["activeExecutionProvider"], provider);
        assert_eq!(body["activeRuntimeDistribution"], distribution);
        assert_eq!(body["configuredExecutionProvider"], "cpu");
        assert_eq!(body["restartRequired"], true);

        #[cfg(all(windows, feature = "ort-cuda"))]
        {
            let (status, body) = send_json(
                &router,
                Method::PUT,
                "/v1/runtime",
                r#"{"executionProvider":"cuda"}"#,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body["configuredExecutionProvider"], "cuda");
            assert_eq!(body["restartRequired"], true);
        }

        let (status, body) = send(&router, Method::GET, "/v1/runtime").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["configuredExecutionProvider"],
            if cfg!(all(windows, feature = "ort-cuda")) {
                "cuda"
            } else {
                "cpu"
            }
        );
        assert_eq!(body["restartRequired"], true);
    }

    #[tokio::test]
    async fn runtime_includes_image_catalog_and_accepts_model_only_updates() {
        use nicegal_core::embedding::ImageEmbeddingModel;
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(&router, Method::GET, "/v1/runtime").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["imageModel"]["activeModel"],
            ImageEmbeddingModel::MetaClip2B32.id()
        );
        let models = body["imageModel"]["models"].as_array().unwrap();
        assert_eq!(models.len(), ImageEmbeddingModel::ALL.len());
        assert!(
            models
                .iter()
                .any(|model| model["id"] == ImageEmbeddingModel::SigLipBetaSwinV2Frozen.id())
        );
        let (status, body) = send_json(
            &router,
            Method::PUT,
            "/v1/runtime",
            r#"{"imageModel":"facebook/metaclip-2-worldwide-b16"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["imageModel"]["restartRequired"], true);
        assert_eq!(
            body["imageModel"]["selectedModel"],
            ImageEmbeddingModel::MetaClip2B16.id()
        );
        for request in [r#"{}"#, r#"{"imageModel":"unknown"}"#] {
            let (status, _) = send_json(&router, Method::PUT, "/v1/runtime", request).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }
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
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("libraryId"),
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
    async fn asset_batch_body_rejections_use_the_error_envelope() {
        let temp = TempDir::new().unwrap();
        let router = test_router_with_preparation(&temp, false);
        for body in ["{not json", r#"{"assetIds":"wrong shape"}"#] {
            let (status, body) = send_json(&router, Method::POST, "/v1/assets", body).await;
            assert!(status.is_client_error());
            assert_eq!(body["error"]["code"], "invalid_request");
        }
        let (status, body) = send(&router, Method::POST, "/v1/assets").await;
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(body["error"]["code"], "unsupported_media_type");

        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/assets")
            .header(header::AUTHORIZATION, TOKEN)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(vec![b' '; MAX_REQUEST_BYTES + 1]))
            .unwrap();
        let (status, body) = response_parts(&router, request).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body["error"]["code"], "payload_too_large");
    }

    #[tokio::test]
    async fn a_well_formed_search_returns_its_hits() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(&router, Method::GET, &search_uri("hello", "ocrSimple")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        assert_eq!(body["results"][0]["assetId"], 1);
    }

    #[tokio::test]
    async fn a_query_sqlite_cannot_parse_is_a_400_carrying_its_message() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        for (query, kind, expected) in [
            ("ocr:\"unterminated", "ocrSimple", "unterminated string"),
            ("AND", "ocrMatch", "fts5: syntax error"),
            ("col:foo", "ocrMatch", "no such column"),
        ] {
            let (status, body) = send(&router, Method::GET, &search_uri(query, kind)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{query}: {body}");
            assert_eq!(body["error"]["code"], "query_syntax", "{query}");
            let message = body["error"]["message"].as_str().unwrap();
            assert!(message.contains(expected), "{query}: {message}");
        }
    }

    #[tokio::test]
    async fn searching_an_unknown_library_is_library_not_found() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(&router, Method::GET, "/v1/search?q=hello&libraryId=99").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "library_not_found");
    }

    #[tokio::test]
    async fn an_indexed_root_with_no_matches_is_an_empty_result_not_an_error() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(
            &router,
            Method::GET,
            &search_uri("nothingmatchesthis", "ocrSimple"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 0);
        assert_eq!(body["results"].as_array().unwrap().len(), 0);
    }

    /// Give the router's OCR database vectors for its rows, the way a completed embed job would.
    fn embed_everything(temp: &TempDir) {
        let ocr_path = PathBuf::try_from(temp.path().join("ocr.db")).unwrap();
        let embedder = prepared_text_model().ready().unwrap();
        let mut ocr = DB::new(&ocr_path).unwrap();
        let space = nicegal_core::db::TextEmbeddingSpace::OcrText;
        ocr.set_text_embedding_model(space, embedder.model().id(), embedder.dimensions(), true)
            .unwrap();
        let root = PathBuf::try_from(temp.path().to_path_buf()).unwrap();
        let filters = nicegal_core::db::SearchFilters::under(&root);
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
        send_value(router, Method::POST, uri, body).await
    }

    async fn send_value(
        router: &Router,
        method: Method,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method(method)
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
        let router = test_router_with_preparation(&temp, true);

        // No `type=`: the default is vector, and nothing is embedded yet.
        let (status, body) = send(
            &router,
            Method::GET,
            &format!("/v1/search?q=hello&libraryId={FIXTURE_LIBRARY}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 0);

        embed_everything(&temp);
        let (status, body) = send(
            &router,
            Method::GET,
            &format!("/v1/search?q=hello&libraryId={FIXTURE_LIBRARY}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
        assert_eq!(body["results"][0]["assetId"], 1);
        assert_eq!(body["results"][0]["rank"], 1);
        // Vector hits carry a distance; the text modes do not.
        assert!(body["results"][0]["distance"].is_number(), "{body}");

        let (status, body) = send(&router, Method::GET, &search_uri("hello", "ocrSimple")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body["results"][0]["distance"].is_null(), "{body}");
    }

    #[tokio::test]
    async fn a_distance_ceiling_is_rejected_for_the_text_modes() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = send(
            &router,
            Method::GET,
            &format!(
                "/v1/search?q=hello&type=ocrSimple&libraryId={FIXTURE_LIBRARY}&maxDistance=0.5"
            ),
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
        let router = test_router_with_preparation(&temp, true);
        embed_everything(&temp);

        let (status, body) = post_json(
            &router,
            "/v1/search",
            serde_json::json!({
                "libraryId": FIXTURE_LIBRARY,
                "queries": [
                    {"key": "semantic", "type": "vector", "q": "hello"},
                    {"key": "literal", "type": "ocrSimple", "q": "hello"},
                    {"key": "nothing", "type": "ocrSimple", "q": "zzzznomatch"}
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
        for (index, kind) in [(0, "vector"), (1, "ocrSimple")] {
            let (single_status, single) =
                send(&router, Method::GET, &search_uri("hello", kind)).await;
            assert_eq!(single_status, StatusCode::OK, "{single}");
            assert_eq!(single["total"], queries[index]["total"]);
            assert_eq!(single["results"], queries[index]["results"]);
        }

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
                "libraryId": FIXTURE_LIBRARY,
                "queries": [{"type": "ocrSimple", "q": "hello"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // The key defaults to the mode name.
        assert_eq!(body["queries"][0]["key"], "ocrSimple");
        assert!(body.get("fused").is_none(), "{body}");
        // Nothing has been embedded, so there is no stored model to report.
        assert!(body["model"].is_null(), "{body}");
    }

    #[tokio::test]
    async fn a_combined_search_rejects_an_empty_or_ambiguous_query_set() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        for body in [
            serde_json::json!({"libraryId": FIXTURE_LIBRARY, "queries": []}),
            serde_json::json!({"libraryId": FIXTURE_LIBRARY, "queries": [
                {"key": "same", "type": "ocrSimple", "q": "a"},
                {"key": "same", "type": "ocrGlob", "q": "b"}
            ]}),
            serde_json::json!({"libraryId": FIXTURE_LIBRARY, "queries": [
                {"type": "ocrSimple", "q": "a", "weight": 0}
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
                "libraryId": FIXTURE_LIBRARY,
                "queries": [{"type": "ocrMatch", "q": "AND"}]
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
    async fn image_embedding_coverage_is_available_without_loading_models() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let uri = format!("/v1/image-embeddings?libraryId={FIXTURE_LIBRARY}");
        let (status, body) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["indexed"], 0);
        assert!(body["total"].is_u64());
        let (status, _) = send(&router, Method::GET, "/v1/image-embeddings").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn embedding_status_separates_an_unembedded_root_from_an_empty_one() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let uri = format!("/v1/text-embeddings?libraryId={FIXTURE_LIBRARY}");

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
    async fn an_embed_backfill_starts_a_job_and_an_unknown_library_does_not() {
        let temp = TempDir::new().unwrap();
        let router = test_router_with_preparation(&temp, true);

        let (status, body) = post_json(
            &router,
            "/v1/text-embeddings/generate",
            serde_json::json!({"libraryId": FIXTURE_LIBRARY}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{body}");
        assert_eq!(body["type"], "textEmbed");

        let (status, body) = post_json(
            &router,
            "/v1/text-embeddings/generate",
            serde_json::json!({"libraryId": 99}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "library_not_found");
    }

    #[tokio::test]
    async fn a_time_bound_without_a_timeline_is_refused() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);

        // There is no default timeline: capture and modified answer different questions, so the
        // server refuses to pick one on the caller's behalf.
        for query in ["after=0", "before=100", "after=0&before=100"] {
            let (status, body) = send(
                &router,
                Method::GET,
                &format!("/v1/search?q=hello&type=ocrSimple&libraryId={FIXTURE_LIBRARY}&{query}"),
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
            &format!(
                "/v1/search?q=hello&type=ocrSimple&libraryId={FIXTURE_LIBRARY}&timeline=capture"
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 1);
    }

    #[tokio::test]
    async fn time_bounds_narrow_every_search_mode() {
        let temp = TempDir::new().unwrap();
        let router = test_router_with_preparation(&temp, true);
        embed_everything(&temp);
        // The fixture row has source_modified_ns = 1 and no EXIF capture time.
        let inside = "after=0&before=2&timeline=modified";
        let outside = "after=2&before=9&timeline=modified";

        for kind in ["ocrSimple", "ocrGlob", "vector"] {
            let query = if kind == "ocrGlob" {
                "*hello*"
            } else {
                "hello"
            };
            let (status, body) = send(
                &router,
                Method::GET,
                &format!(
                    "/v1/search?q={}&type={kind}&libraryId={FIXTURE_LIBRARY}&{inside}",
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
                    "/v1/search?q={}&type={kind}&libraryId={FIXTURE_LIBRARY}&{outside}",
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

        let (status, body) = send(
            &router,
            Method::GET,
            &format!(
                "/v1/search?q=hello&type=ocrSimple&libraryId={FIXTURE_LIBRARY}&after=notanumber&timeline=modified"
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
                "/v1/search?q=hello&type=ocrSimple&libraryId={FIXTURE_LIBRARY}&after=9&before=2&timeline=modified"
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
                "/v1/search?q=hello&type=ocrSimple&libraryId={FIXTURE_LIBRARY}&after=1717243200123456789&timeline=capture"
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["total"], 0);
    }

    #[tokio::test]
    async fn a_combined_search_applies_the_request_range_and_lets_one_query_override_it() {
        let temp = TempDir::new().unwrap();
        let router = test_router_with_preparation(&temp, true);
        embed_everything(&temp);

        let (status, body) = post_json(
            &router,
            "/v1/search",
            serde_json::json!({
                "libraryId": FIXTURE_LIBRARY,
                "timeline": "modified",
                "after": "2",
                "queries": [
                    {"key": "inherits", "type": "ocrSimple", "q": "hello"},
                    // Overriding replaces the whole filter rather than merging, so this query does
                    // not silently inherit `after` from the request.
                    {"key": "overrides", "type": "ocrSimple", "q": "hello",
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
                "libraryId": FIXTURE_LIBRARY,
                "timeline": "modified",
                "after": "0",
                "queries": [{"key": "partial", "type": "ocrSimple", "q": "hello", "before": "2"}]
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
        let uri = format!("/v1/catalog?libraryId={FIXTURE_LIBRARY}&timeline=capture");
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
            &format!("/v1/catalog/count?libraryId={FIXTURE_LIBRARY}"),
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
            "/v1/catalog",
            "/v1/catalog/count",
            "/v1/catalog/metadata?assetId=0",
            "/v1/catalog?libraryId=1&timeline=wrong",
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

    #[tokio::test]
    async fn thumbnail_ensure_bounds_the_batch_and_deduplicates_ids() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let (status, body) = post_json(
            &router,
            "/v1/thumbnails",
            serde_json::json!({
                "assetIds": vec![1; 513], "requiredSize": 128
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_request");
        assert!(body["error"]["message"].as_str().unwrap().contains("512"));

        // A one-pixel GIF keeps this route test independent of model downloads and image tooling.
        use base64::Engine as _;
        let source = PathBuf::try_from(temp.path().join("pixel.gif")).unwrap();
        std::fs::write(
            &source,
            base64::engine::general_purpose::STANDARD
                .decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
                .unwrap(),
        )
        .unwrap();
        let catalog =
            AssetCatalog::new(&PathBuf::try_from(temp.path().join("assets.db")).unwrap()).unwrap();
        let asset = catalog
            .upsert(&source, &std::fs::metadata(&source).unwrap())
            .unwrap();
        for ids in [vec![asset.asset_id], vec![asset.asset_id, asset.asset_id]] {
            let (status, body) = post_json(
                &router,
                "/v1/thumbnails",
                serde_json::json!({
                    "assetIds": ids, "requiredSize": 128
                }),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["assetIds"], serde_json::json!([asset.asset_id]));
        }
        let (status, body) = post_json(
            &router,
            "/v1/thumbnails",
            serde_json::json!({
                "assetIds": [asset.asset_id, 99999], "requiredSize": 128
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "asset_not_found");
    }

    fn library_dir(temp: &TempDir, name: &str) -> String {
        let path = temp.path().join(name);
        std::fs::create_dir_all(&path).unwrap();
        let path = PathBuf::try_from(path).unwrap();
        nicegal_core::assets::canonicalize_path(&path)
            .unwrap()
            .into_string()
    }

    #[tokio::test]
    async fn libraries_are_created_listed_edited_and_deleted() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let photos = library_dir(&temp, "photos");
        let private = library_dir(&temp, "photos/private");
        let phone = library_dir(&temp, "phone");

        let (status, created) = post_json(
            &router,
            "/v1/libraries",
            serde_json::json!({ "include": [photos], "exclude": [private] }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{created}");
        let id = created["id"].as_i64().unwrap();
        assert!(created.get("revision").is_none());
        assert_eq!(created["include"][0]["path"], photos.as_str());
        assert_eq!(created["include"][0]["scanPending"], true);
        assert_eq!(
            created["include"][0]["scanOutcome"],
            serde_json::Value::Null
        );
        assert_eq!(created["exclude"][0], private.as_str());
        assert_eq!(
            (
                created["ocr"].clone(),
                created["image"].clone(),
                created["videos"].clone()
            ),
            (false.into(), true.into(), true.into())
        );
        let scans_of = |jobs: &serde_json::Value, library: &serde_json::Value| {
            jobs["jobs"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|job| job["type"] == "libraryScan" && &job["libraryId"] == library)
                .count()
        };
        let (_, jobs) = send(&router, Method::GET, "/v1/jobs").await;
        assert_eq!(
            scans_of(&jobs, &created["id"]),
            0,
            "creating a library starts no scan"
        );
        let (status, scan) = post_json(
            &router,
            "/v1/jobs",
            serde_json::json!({ "type": "libraryScan", "params": { "libraryId": id, "pendingOnly": true } }),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{scan}");
        assert_eq!(scan["libraryId"], id);

        let (status, listed) = send(&router, Method::GET, "/v1/libraries").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            listed.as_array().unwrap().last().unwrap()["id"],
            created["id"]
        );
        let uri = format!("/v1/libraries/{id}");
        assert_eq!(
            send(&router, Method::GET, &uri).await.1["id"],
            created["id"]
        );

        let edit = serde_json::json!({
            "include": [photos, phone], "exclude": [], "ocr": true, "image": true,
            "videos": false
        });
        let (status, edited) = send_value(&router, Method::PUT, &uri, edit.clone()).await;
        assert_eq!(status, StatusCode::OK, "{edited}");
        assert_eq!(edited["include"][1]["path"], phone.as_str());
        assert_eq!(edited["exclude"], serde_json::json!([]));
        assert_eq!(edited["videos"], false);
        // The new folder awaits its first scan; the one that kept its place keeps its state.
        assert_eq!(edited["include"][1]["scanPending"], true);
        assert_eq!(
            edited["include"][1]["lastScanCompletedNs"],
            serde_json::Value::Null
        );

        // Repeating an edit changes nothing.
        let (status, repeated) = send_value(&router, Method::PUT, &uri, edit).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(repeated["id"], edited["id"]);
        assert_eq!(repeated["ocr"], edited["ocr"]);
        assert_eq!(repeated["include"][1]["path"], edited["include"][1]["path"]);

        let (status, _) = send(&router, Method::DELETE, &uri).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        for method in [Method::GET, Method::DELETE] {
            let (status, missing) = send(&router, method, &uri).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            assert_eq!(missing["error"]["code"], "library_not_found");
        }
    }

    #[tokio::test]
    async fn library_definitions_are_validated_before_anything_is_written() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let photos = library_dir(&temp, "photos");
        let elsewhere = library_dir(&temp, "elsewhere");
        let missing = temp.path().join("missing").to_str().unwrap().to_owned();

        for (body, code) in [
            (serde_json::json!({ "include": [] }), "invalid_request"),
            (
                serde_json::json!({ "include": ["relative"] }),
                "invalid_root",
            ),
            (serde_json::json!({ "include": [missing] }), "invalid_root"),
            (
                serde_json::json!({ "include": [photos], "exclude": [elsewhere] }),
                "invalid_request",
            ),
        ] {
            let (status, error) = post_json(&router, "/v1/libraries", body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(error["error"]["code"], code, "{body}: {error}");
        }
        let (_, listed) = send(&router, Method::GET, "/v1/libraries").await;
        assert_eq!(
            listed.as_array().unwrap().len(),
            1,
            "only the fixture library"
        );
    }

    #[tokio::test]
    async fn imports_are_idempotent_and_offline_folders_stay_editable() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let offline = temp.path().join("unplugged").to_str().unwrap().to_owned();
        let import = serde_json::json!({ "include": [offline], "importKey": offline, "ocr": true });

        // An import may name a folder whose drive is disconnected.
        let (status, first) = post_json(&router, "/v1/libraries", import.clone()).await;
        assert_eq!(status, StatusCode::CREATED, "{first}");
        let (status, again) = post_json(&router, "/v1/libraries", import).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(again["id"], first["id"]);
        assert_eq!(again["include"][0]["path"], first["include"][0]["path"]);
        assert_eq!(again["ocr"], first["ocr"]);

        // Folders the library already has are not stat'ed, so its options remain editable.
        let uri = format!("/v1/libraries/{}", first["id"]);
        let (status, edited) = send_value(
            &router,
            Method::PUT,
            &uri,
            serde_json::json!({ "include": [offline], "ocr": false, "image": true, "videos": true }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{edited}");
        assert_eq!(edited["ocr"], false);
    }

    #[tokio::test]
    async fn library_reads_cover_every_folder_minus_exclusions_even_offline() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let photos = library_dir(&temp, "photos");
        let private = library_dir(&temp, "photos/private");
        let phone = library_dir(&temp, "phone");
        let other = library_dir(&temp, "other");
        let databases_dir = PathBuf::try_from(temp.path().to_path_buf()).unwrap();
        let catalog = AssetCatalog::new(&databases_dir.join("assets.db")).unwrap();
        let mut ids = std::collections::HashMap::new();
        for (name, folder) in [
            ("a", &photos),
            ("b", &private),
            ("c", &phone),
            ("d", &other),
        ] {
            let path = PathBuf::from(folder.as_str()).join(format!("{name}.png"));
            std::fs::write(&path, b"not really a png").unwrap();
            let asset = catalog
                .upsert(&path, &std::fs::metadata(&path).unwrap())
                .unwrap();
            ids.insert(name, asset);
        }
        drop(catalog);
        let mut ocr = DB::new(&databases_dir.join("ocr.db")).unwrap();
        ocr.save_results(
            ["b", "c", "d"]
                .into_iter()
                .map(|name| OcrResult {
                    asset_id: ids[name].asset_id,
                    path: ids[name].path.clone(),
                    fingerprint: ids[name].fingerprint,
                    exif_taken_ns: None,
                    width: 1,
                    height: 1,
                    contents: "needle".to_owned(),
                })
                .collect(),
        )
        .unwrap();
        drop(ocr);

        let (status, library) = post_json(
            &router,
            "/v1/libraries",
            serde_json::json!({ "include": [photos, phone], "exclude": [private] }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{library}");
        let id = library["id"].as_i64().unwrap();
        let listed_ids = |body: &serde_json::Value, field: &str| {
            let mut listed = body
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row[field]
                        .as_str()
                        .map_or_else(|| row[field].to_string(), str::to_owned)
                })
                .collect::<Vec<_>>();
            listed.sort();
            listed
        };
        let expected = |names: &[&str]| {
            let mut expected = names
                .iter()
                .map(|name| ids[name].asset_id.to_string())
                .collect::<Vec<_>>();
            expected.sort();
            expected
        };

        let (status, gallery) =
            send(&router, Method::GET, &format!("/v1/catalog?libraryId={id}")).await;
        assert_eq!(status, StatusCode::OK, "{gallery}");
        assert_eq!(listed_ids(&gallery, "id"), expected(&["a", "c"]));
        let (_, count) = send(
            &router,
            Method::GET,
            &format!("/v1/catalog/count?libraryId={id}"),
        )
        .await;
        assert_eq!(count, 2);

        // A disconnected folder is still searchable from its cached index.
        std::fs::remove_dir_all(&phone).unwrap();
        let (status, found) = send(
            &router,
            Method::GET,
            &format!("/v1/search?q=needle&type=ocrSimple&libraryId={id}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{found}");
        assert_eq!(found["total"], 1);
        assert_eq!(found["results"][0]["assetId"], ids["c"].asset_id);
        let uri = format!(
            "/v1/search?q=needle&type=ocrSimple&libraryId={id}&folder={}",
            urlencode(phone.as_str())
        );
        let (status, focused) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::OK, "{focused}");
        assert_eq!(focused["total"], 1);
        let uri = format!(
            "/v1/search?q=needle&type=ocrSimple&libraryId={id}&folder={}",
            urlencode(private.as_str())
        );
        let (status, excluded) = send(&router, Method::GET, &uri).await;
        assert_eq!(status, StatusCode::OK, "{excluded}");
        assert_eq!(excluded["total"], 0);

        for uri in [
            format!("/v1/catalog?libraryId={}", id + 1),
            format!("/v1/search?q=needle&type=ocrSimple&libraryId={}", id + 1),
        ] {
            let (status, missing) = send(&router, Method::GET, &uri).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
            assert_eq!(missing["error"]["code"], "library_not_found");
        }
    }

    async fn wait_for_job(router: &Router, job: &serde_json::Value) -> serde_json::Value {
        let uri = format!("/v1/jobs/{}", job["jobId"].as_str().unwrap());
        for _ in 0..600 {
            let (status, body) = send(router, Method::GET, &uri).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            if ["completed", "failed", "cancelled"].contains(&body["status"].as_str().unwrap()) {
                return body;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("job did not finish");
    }

    #[tokio::test]
    async fn a_scan_respells_a_folder_imported_while_offline() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        // A spelling canonicalization would change, for a folder that does not exist yet.
        let spelled = format!("{}/./later/", temp.path().to_str().unwrap());
        let (status, library) = post_json(
            &router,
            "/v1/libraries",
            serde_json::json!({ "include": [spelled], "importKey": spelled, "image": false }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{library}");
        let stored = library["include"][0]["path"].as_str().unwrap().to_owned();
        assert!(!stored.ends_with(['/', '\\']), "{stored}");
        if cfg!(windows) {
            assert!(!stored.contains('/'), "{stored}");
        }

        // The drive comes back.
        let later = library_dir(&temp, "later");
        std::fs::write(PathBuf::from(later.as_str()).join("a.mp4"), b"video").unwrap();
        let id = library["id"].as_i64().unwrap();
        let (status, job) = post_json(
            &router,
            "/v1/jobs",
            serde_json::json!({ "type": "libraryScan", "params": { "libraryId": id } }),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{job}");
        let job = wait_for_job(&router, &job).await;
        assert_eq!(job["status"], "completed", "{job}");

        let (_, library) = send(&router, Method::GET, &format!("/v1/libraries/{id}")).await;
        assert_eq!(library["include"][0]["path"], later.as_str());
        assert_eq!(library["include"][0]["scanPending"], false);
        let (_, count) = send(
            &router,
            Method::GET,
            &format!("/v1/catalog/count?libraryId={id}"),
        )
        .await;
        assert_eq!(
            count, 1,
            "the library shows the files cataloged under the canonical path"
        );
    }

    #[tokio::test]
    async fn a_library_scan_catalogs_every_folder_and_reports_each_one() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let photos = library_dir(&temp, "photos");
        let private = library_dir(&temp, "photos/private");
        let phone = library_dir(&temp, "phone");
        let unplugged = library_dir(&temp, "unplugged");
        for (folder, name) in [(&photos, "a.mp4"), (&private, "b.mp4"), (&phone, "c.mp4")] {
            std::fs::write(
                PathBuf::from(folder.as_str()).join(name),
                b"not really a video",
            )
            .unwrap();
        }
        // OCR is on, but nothing here is an image, so no OCR model is needed or loaded.
        let (status, library) = post_json(
            &router,
            "/v1/libraries",
            serde_json::json!({
                "include": [photos, phone, unplugged], "exclude": [private],
                "ocr": true, "image": false
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{library}");
        let id = library["id"].as_i64().unwrap();
        std::fs::remove_dir(&unplugged).unwrap();

        let (status, job) = post_json(
            &router,
            "/v1/jobs",
            serde_json::json!({ "type": "libraryScan", "params": { "libraryId": id } }),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{job}");
        let job = wait_for_job(&router, &job).await;
        assert_eq!(job["status"], "completed", "{job}");
        assert_eq!(
            job["indexStages"],
            serde_json::json!({"ocr": true, "image": false, "text": true})
        );
        let folders = job["folders"].as_array().unwrap();
        let states = folders
            .iter()
            .map(|folder| {
                (
                    folder["path"].as_str().unwrap(),
                    folder["state"].as_str().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            states,
            [
                (photos.as_str(), "completed"),
                (phone.as_str(), "completed"),
                (unplugged.as_str(), "unavailable")
            ]
        );
        assert_eq!(
            folders[0]["discovered"], 1,
            "the excluded folder is never walked"
        );
        assert_eq!(folders[0]["cataloged"], 1);
        assert!(
            folders[2]["error"]
                .as_str()
                .unwrap()
                .contains("cannot read folder")
        );
        assert_eq!(job["progress"]["modelsLoaded"], 0);

        let (_, gallery) = send(&router, Method::GET, &format!("/v1/catalog?libraryId={id}")).await;
        let mut names = gallery
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["displayName"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["a.mp4", "c.mp4"]);

        let (_, library) = send(&router, Method::GET, &format!("/v1/libraries/{id}")).await;
        let include = library["include"].as_array().unwrap();
        for folder in &include[..2] {
            assert_eq!(folder["scanPending"], false, "{folder}");
            assert_eq!(folder["scanOutcome"], serde_json::Value::Null);
            assert_eq!(folder["scanError"], serde_json::Value::Null);
            assert!(folder["lastScanCompletedNs"].is_string(), "{folder}");
        }
        assert_eq!(
            include[2]["scanPending"], true,
            "an offline folder stays pending"
        );
        assert_eq!(include[2]["scanOutcome"], "unavailable");
        assert!(include[2]["scanError"].is_string());

        // Only the offline folder is left for a pending-only scan.
        let (_, job) = post_json(
            &router,
            "/v1/jobs",
            serde_json::json!({ "type": "libraryScan", "params": { "libraryId": id, "pendingOnly": true } }),
        )
        .await;
        let job = wait_for_job(&router, &job).await;
        let paths = job["folders"]
            .as_array()
            .unwrap()
            .iter()
            .map(|folder| folder["path"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(paths, [unplugged.as_str()]);

        let (status, missing) = post_json(
            &router,
            "/v1/jobs",
            serde_json::json!({ "type": "libraryScan", "params": { "libraryId": id + 1 } }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(missing["error"]["code"], "library_not_found");
    }

    #[tokio::test]
    async fn automatic_scan_checks_nested_changes_without_advancing_the_full_scan_clock() {
        let temp = TempDir::new().unwrap();
        let router = test_router(&temp);
        let root = library_dir(&temp, "automatic");
        let nested = library_dir(&temp, "automatic/nested");
        let sibling = library_dir(&temp, "automatic/sibling");
        std::fs::write(PathBuf::from(nested.as_str()).join("old.mp4"), b"video").unwrap();
        std::fs::write(PathBuf::from(sibling.as_str()).join("keep.mp4"), b"video").unwrap();
        let (status, library) = post_json(
            &router,
            "/v1/libraries",
            serde_json::json!({
                "include": [root], "image": false
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{library}");
        let id = library["id"].as_i64().unwrap();
        let scan = |mode: &str| {
            serde_json::json!({
                "type": "libraryScan", "params": { "libraryId": id, "scanMode": mode }
            })
        };
        let (_, first) = post_json(&router, "/v1/jobs", scan("full")).await;
        assert_eq!(wait_for_job(&router, &first).await["status"], "completed");
        let (_, before) = send(&router, Method::GET, &format!("/v1/libraries/{id}")).await;
        let full_clock = before["include"][0]["lastScanCompletedNs"].clone();

        let (_, unchanged) = post_json(&router, "/v1/jobs", scan("fast")).await;
        let unchanged = wait_for_job(&router, &unchanged).await;
        assert_eq!(unchanged["status"], "completed", "{unchanged}");
        assert_eq!(unchanged["folders"][0]["scanMode"], "fast");
        assert_eq!(unchanged["folders"][0]["cataloged"], 0);

        std::fs::remove_file(PathBuf::from(nested.as_str()).join("old.mp4")).unwrap();
        std::fs::write(PathBuf::from(nested.as_str()).join("new.mp4"), b"video").unwrap();
        let (_, changed) = post_json(&router, "/v1/jobs", scan("fast")).await;
        let changed = wait_for_job(&router, &changed).await;
        assert_eq!(changed["status"], "completed", "{changed}");
        assert_eq!(changed["folders"][0]["scanMode"], "fast");
        let (_, gallery) = send(&router, Method::GET, &format!("/v1/catalog?libraryId={id}")).await;
        let mut names = gallery
            .as_array()
            .unwrap()
            .iter()
            .map(|asset| asset["displayName"].as_str().unwrap())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["keep.mp4", "new.mp4"]);

        let added = library_dir(&temp, "automatic/added");
        std::fs::write(PathBuf::from(added.as_str()).join("added.mp4"), b"video").unwrap();
        let (_, added_job) = post_json(&router, "/v1/jobs", scan("fast")).await;
        let added_job = wait_for_job(&router, &added_job).await;
        assert_eq!(added_job["status"], "completed", "{added_job}");
        std::fs::remove_dir_all(&added).unwrap();
        let (_, removed_job) = post_json(&router, "/v1/jobs", scan("fast")).await;
        let removed_job = wait_for_job(&router, &removed_job).await;
        assert_eq!(removed_job["status"], "completed", "{removed_job}");
        let (_, gallery) = send(&router, Method::GET, &format!("/v1/catalog?libraryId={id}")).await;
        let mut names = gallery
            .as_array()
            .unwrap()
            .iter()
            .map(|asset| asset["displayName"].as_str().unwrap())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["keep.mp4", "new.mp4"]);
        let (_, after) = send(&router, Method::GET, &format!("/v1/libraries/{id}")).await;
        assert_eq!(after["include"][0]["lastScanCompletedNs"], full_clock);
    }
}
