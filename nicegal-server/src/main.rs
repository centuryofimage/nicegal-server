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
use nicegal_core::embedding::{
    ImageEmbedderOptions, ImageEmbeddingModel, ImageQueryEmbedderOptions, TextEmbedderOptions,
    TextEmbeddingModel,
};
use nicegal_core::image_index::ImageIndexDb;
use nicegal_core::runtime::{ExecutionProvider, RuntimeOptions};
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

    /// Text embedding model used for vector search.
    #[arg(long, env = "NICEGAL_EMBED_MODEL", value_name = "MODEL")]
    embed_model: Option<TextEmbeddingModel>,

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
        None => default_database("runtime.json")?,
    };
    let runtime = Arc::new(api::RuntimeSettings::load(
        runtime_config.clone(),
        args.execution_provider,
    )?);
    let image_model_settings = Arc::new(api::ImageModelSettings::new(
        Arc::clone(&runtime),
        args.image_model,
    ));
    let image_model = image_model_settings.active();
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
    drop(AssetCatalog::new(&asset_database)?);

    // Models are loaded lazily when a job or request first needs them.
    let mut embedder_options = TextEmbedderOptions::default();
    if let Some(model) = args.embed_model {
        embedder_options.model = model;
    }
    // The bundled runtime distribution selected above is process-global, so text and image
    // embedders request the same execution provider as OCR.
    embedder_options.runtime = RuntimeOptions {
        execution_provider: runtime.active_execution_provider(),
        ..RuntimeOptions::default()
    };
    let embedder = Arc::new(api::models::TextModel::deferred(embedder_options.clone()));
    let image_embedder_options = ImageEmbedderOptions {
        model: image_model,
        runtime: embedder_options.runtime,
        ..ImageEmbedderOptions::default()
    };
    let image_embedder = Arc::new(api::models::ImageModel::deferred(image_embedder_options));
    // The CLIP pair's text half, on CPU on purpose: query embedding is one short forward pass
    // per search, where a GPU upload costs more than the pass itself, and the accelerator is
    // wanted by indexing and OCR. Its model is the image encoder's paired text encoder, so it is
    // selected with it here and changes only through a restart, exactly like indexing.
    let image_query_embedder_options = ImageQueryEmbedderOptions {
        model: image_model,
        runtime: RuntimeOptions {
            execution_provider: ExecutionProvider::Cpu,
            ..RuntimeOptions::default()
        },
        ..ImageQueryEmbedderOptions::default()
    };
    let image_query_embedder = Arc::new(api::models::ImageQueryModel::deferred(
        image_query_embedder_options,
    ));
    drop(ImageIndexDb::new(
        &image_database,
        image_embedder.dimensions(),
    )?);

    let databases = Arc::new(api::Databases {
        assets: asset_database,
        images: image_database,
        ocr: ocr_database,
        thumbnails: thumbnail_database,
    });
    let thumbnails = Arc::new(ThumbnailService::new(&databases.thumbnails)?);
    let ocr_models = Arc::new(api::ModelStore::new(runtime.active_execution_provider()));
    let jobs = Arc::new(api::JobManager::new(
        Arc::clone(&databases),
        Arc::clone(&thumbnails),
        Arc::clone(&embedder),
        Arc::clone(&image_embedder),
        Arc::clone(&image_query_embedder),
        Arc::clone(&ocr_models),
        Arc::clone(&runtime),
    ));
    let state = api::AppState {
        databases,
        thumbnails,
        jobs: Arc::clone(&jobs),
        embedder,
        image_query_embedder,
        image_embedder,
        image_model_settings,
        ocr_models,
        runtime,
    };
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

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(jobs))
        .await
        .context("serving local HTTP requests")?;
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
