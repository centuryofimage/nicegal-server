#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod api;

use std::fs;
use std::io::{self, Write};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use axum::http::HeaderValue;
use camino::Utf8PathBuf as PathBuf;
use clap::Parser;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::db::DB;
use nicegal_core::embedding::{ImageEmbedderOptions, ImageEmbeddingModel};
use nicegal_core::image_index::ImageIndexDb;
use nicegal_core::runtime::ExecutionProvider;
use nicegal_core::thumbs::ThumbnailService;
use serde::Serialize;
use tokio::io::AsyncReadExt;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Location of the canonical asset catalog.
    #[arg(long, env = "NICEGAL_ASSET_DB", value_name = "FILE")]
    asset_database: Option<PathBuf>,

    /// Location of the OCR text index.
    #[arg(long, env = "NICEGAL_OCR_DB", value_name = "FILE")]
    ocr_database: Option<PathBuf>,

    /// Location of the thumbnail database.
    #[arg(long, env = "NICEGAL_THUMBNAIL_DB", value_name = "FILE")]
    thumbnail_database: Option<PathBuf>,

    /// Per-launch bearer token required by every request.
    #[arg(long, env = "NICEGAL_RPC_TOKEN", hide_env_values = true)]
    token: String,

    /// Image/text encoder pair used for image similarity search.
    #[arg(long, env = "NICEGAL_IMAGE_MODEL", value_name = "MODEL")]
    image_model: Option<ImageEmbeddingModel>,

    /// Preferred ONNX Runtime execution provider for this launch.
    #[arg(long, env = "NICEGAL_EXECUTION_PROVIDER", value_name = "PROVIDER")]
    execution_provider: Option<ExecutionProvider>,

    /// Location of the persisted runtime and image-model settings.
    #[arg(long, env = "NICEGAL_RUNTIME_CONFIG", value_name = "FILE")]
    runtime_config: Option<PathBuf>,

    /// Filter directives for the JSON log file. `RUST_LOG` takes precedence; the console is fixed.
    #[arg(long, env = "NICEGAL_LOG", value_name = "DIRECTIVES")]
    log: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReadyResponse {
    api_version: u8,
    endpoint: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let ocr_database = match args.ocr_database {
        Some(path) => path,
        None => default_database("index.db")?,
    };
    let asset_database = match args.asset_database {
        Some(path) => path,
        None => default_database("assets.db")?,
    };
    let thumbnail_database = match args.thumbnail_database {
        Some(path) => path,
        None => default_database("thumbnails.db")?,
    };
    let runtime_config = match args.runtime_config {
        Some(path) => path,
        None => asset_database
            .parent()
            .context("asset database path has no parent directory")?
            .join("runtime.json"),
    };
    let runtime = Arc::new(api::RuntimeSettings::load(
        runtime_config.clone(),
        args.execution_provider,
    )?);
    let image_model = args.image_model.unwrap_or_else(|| runtime.image_model());
    let image_database = asset_database
        .parent()
        .context("asset database path has no parent directory")?
        .join(image_model.database_file_name());
    for database in [
        &asset_database,
        &image_database,
        &ocr_database,
        &thumbnail_database,
        &runtime_config,
    ] {
        if let Some(parent) = database.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating database directory: {parent}"))?;
        }
    }

    let log_directory = asset_database
        .parent()
        .context("asset database path has no parent directory")?;
    let _logging_guards = nicegal_core::logging::init(log_directory, args.log.as_deref())?;
    if args.token.is_empty() {
        bail!("NICEGAL_RPC_TOKEN must not be empty");
    }
    nicegal_core::runtime::initialize_bundled_runtime(runtime.active_execution_provider())?;
    runtime.set_onnx_runtime_build_info(nicegal_core::runtime::onnxruntime_build_info().to_owned());

    let authorization = HeaderValue::from_str(&format!("Bearer {}", args.token))
        .context("NICEGAL_RPC_TOKEN contains invalid HTTP header characters")?;
    // Create and validate every store once during startup so a schema incompatibility fails
    // before readiness, and so the per-request read-only connections always find a schema.
    drop(DB::new(&ocr_database)?);
    let catalog = AssetCatalog::new(&asset_database)?;
    // Video indexing moved from the runtime settings to each library. Copy an older "off" into
    // the libraries before forgetting it, so a crash between the two steps repeats the copy.
    if runtime.legacy_index_videos() == Some(false) {
        catalog.disable_video_indexing()?;
    }
    runtime.clear_legacy_index_videos()?;
    drop(catalog);

    // Models are loaded lazily when a job or request first needs them; `Models` picks each
    // one's provider and how long it stays in memory.
    let models = Arc::new(api::models::Models::new(
        ImageEmbedderOptions {
            model: image_model,
            ..ImageEmbedderOptions::default()
        },
        &runtime,
    ));
    drop(ImageIndexDb::new(
        &image_database,
        models.image.dimensions(),
    )?);

    let databases = Arc::new(api::Databases {
        assets: asset_database,
        images: image_database,
        ocr: ocr_database,
        thumbnails: thumbnail_database,
    });
    let thumbnails = Arc::new(ThumbnailService::new(&databases.thumbnails)?);
    let jobs = Arc::new(api::JobManager::new(
        Arc::clone(&databases),
        Arc::clone(&thumbnails),
        Arc::clone(&models),
    ));
    if let Err(error) = jobs.recover().await {
        tracing::error!(?error, "could not recover background work");
    }
    let state = api::AppState {
        databases,
        thumbnails,
        jobs: Arc::clone(&jobs),
        models,
        runtime,
        tags: Default::default(),
        search_sessions: Default::default(),
        patch_cache: Default::default(),
        external_vectors: Default::default(),
    };
    let model_cleanup = state.start_model_cleanup();
    let app = api::router(state, authorization);

    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .context("binding local HTTP server")?;
    let address = listener.local_addr()?;
    let ready = ReadyResponse {
        api_version: api::API_VERSION,
        endpoint: format!("http://{address}"),
    };
    println!("{}", serde_json::to_string(&ready)?);
    io::stdout().flush()?;
    tracing::info!(endpoint = %ready.endpoint, "listening");

    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(jobs))
        .await
        .context("serving local HTTP requests");
    model_cleanup.abort();
    served?;
    Ok(())
}

fn default_database(file_name: &str) -> Result<PathBuf> {
    let data_dir =
        dirs::data_local_dir().context("the user's local data directory is unavailable")?;
    let data_dir = PathBuf::from_path_buf(data_dir).map_err(|path| {
        anyhow!(
            "local data directory is not valid UTF-8: {}",
            path.display()
        )
    })?;
    Ok(data_dir.join("nicegal-server").join(file_name))
}

async fn shutdown_signal(jobs: Arc<api::JobManager>) {
    let mut stdin = tokio::io::stdin();
    let mut byte = [0_u8; 1];
    tokio::select! {
        _ = async {
            loop {
                match stdin.read(&mut byte).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => (),
                }
            }
        } => (),
        _ = tokio::signal::ctrl_c() => (),
    }
    jobs.cancel_all();
}
