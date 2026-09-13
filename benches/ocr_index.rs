use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use camino::{Utf8Path, Utf8PathBuf};
use image::RgbImage;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::db::{DB, SearchFilters, TextEmbeddingSpace};
use nicegal_core::hub::ModelSource;
use nicegal_core::index::{IndexEvent, IndexObserver, IndexOptions, index_dir_observed};
use nicegal_core::ocr::{PaddleOcrOptions, PaddleOcrPool};
use nicegal_core::runtime::{ExecutionProvider, RuntimeOptions};
use strum::VariantNames;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use walkdir::WalkDir;

struct Arguments {
    corpus: PathBuf,
    detection_model: PathBuf,
    detection_config: PathBuf,
    recognition_model: PathBuf,
    recognition_config: PathBuf,
    provider: ExecutionProvider,
    threads: usize,
    recognition_batch_size: usize,
    detection_max_side: u32,
    replicas: Option<NonZeroUsize>,
    limit: Option<usize>,
    trace_jsonl: Option<PathBuf>,
    runs: usize,
    warmup_runs: usize,
}

struct BenchmarkObserver {
    failures: AtomicUsize,
}

impl BenchmarkObserver {
    fn failures(&self) -> usize {
        self.failures.load(Ordering::Relaxed)
    }
}

impl IndexObserver for BenchmarkObserver {
    fn on_event(&self, event: IndexEvent) {
        if let IndexEvent::Progress(delta) = event {
            self.failures.fetch_add(delta.failed, Ordering::Relaxed);
        }
    }
}

fn main() -> Result<()> {
    if env::args_os().any(|argument| argument == "--help") {
        print_help();
        return Ok(());
    }

    let arguments = parse_arguments()?;
    init_tracing(arguments.trace_jsonl.as_deref())?;
    run(arguments)
}

fn init_tracing(path: Option<&Path>) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,nicegal_core::ocr=debug,nom_exif=off"));
    if let Some(path) = path {
        let file = File::create(path)
            .with_context(|| format!("creating JSONL trace output: {}", path.display()))?;
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(file)
            .try_init()
            .map_err(|error| anyhow!("installing benchmark JSONL tracing subscriber: {error}"))?;
    } else {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(std::io::stderr)
            .try_init()
            .map_err(|error| anyhow!("installing benchmark JSONL tracing subscriber: {error}"))?;
    }
    Ok(())
}

fn run(arguments: Arguments) -> Result<()> {
    if !arguments.corpus.is_dir() {
        bail!(
            "--corpus must name an existing directory: {}",
            arguments.corpus.display()
        );
    }

    let corpus = utf8_path(&arguments.corpus, "--corpus")?;
    let requested_provider = arguments.provider.to_string();
    let runtime_options = RuntimeOptions {
        execution_provider: arguments.provider,
        intra_threads: NonZeroUsize::new(arguments.threads)
            .expect("argument parsing rejects zero thread counts"),
        allow_cpu_fallback: false,
        replicas: arguments.replicas,
    };

    let model_load_started = Instant::now();
    let mut models = PaddleOcrPool::load_with_options(
        ModelSource::local(&arguments.detection_model)?,
        &arguments.detection_model,
        &arguments.detection_config,
        ModelSource::local(&arguments.recognition_model)?,
        &arguments.recognition_model,
        &arguments.recognition_config,
        runtime_options,
    )?;
    let model_load_duration = model_load_started.elapsed();
    let configured_provider = models.execution_provider().to_string();
    let ocr_options = PaddleOcrOptions {
        recognition_batch_size: arguments.recognition_batch_size,
        detection_max_side: arguments.detection_max_side,
        ..PaddleOcrOptions::default()
    };

    let warmup_started = Instant::now();
    let warmup_image = first_supported_image(&corpus)?;
    for _ in 0..arguments.warmup_runs {
        models.scan(&warmup_image, ocr_options)?;
    }
    let warmup_duration = warmup_started.elapsed();

    println!("os={}", env::consts::OS);
    println!("architecture={}", env::consts::ARCH);
    match std::thread::available_parallelism() {
        Ok(parallelism) => println!("logical_parallelism={}", parallelism.get()),
        Err(_) => println!("logical_parallelism=unavailable"),
    }
    println!("requested_provider={requested_provider}");
    println!("configured_provider={configured_provider}");
    println!("threads={}", arguments.threads);
    println!(
        "recognition_batch_size={}",
        arguments.recognition_batch_size
    );
    println!("detection_max_side={}", arguments.detection_max_side);
    let replica_count = models.replicas();
    println!("replicas={replica_count}");
    println!(
        "limit={}",
        arguments
            .limit
            .map_or_else(|| "none".into(), |limit| limit.to_string())
    );
    println!(
        "trace_jsonl={}",
        arguments
            .trace_jsonl
            .as_deref()
            .map_or_else(|| "stderr".into(), |path| path.display().to_string())
    );
    println!("corpus={}", corpus);
    println!("detection_model={}", arguments.detection_model.display());
    println!("detection_config={}", arguments.detection_config.display());
    println!(
        "recognition_model={}",
        arguments.recognition_model.display()
    );
    println!(
        "recognition_config={}",
        arguments.recognition_config.display()
    );
    println!(
        "model_load_seconds={:.6}",
        model_load_duration.as_secs_f64()
    );
    println!("warmup_seconds={:.6}", warmup_duration.as_secs_f64());
    println!(
        "run,configured_provider,threads,replicas,recognition_batch_size,detection_max_side,indexed,failed,elapsed_seconds,images_per_second,ocr_content_digest"
    );

    for run in 1..=arguments.runs {
        let temporary_directory =
            tempfile::tempdir().context("creating temporary benchmark directory")?;
        let database_directory =
            utf8_path(temporary_directory.path(), "temporary benchmark directory")?;
        let mut assets = AssetCatalog::new(&database_directory.join("assets.sqlite"))?;
        let mut db = DB::new(&database_directory.join("ocr.sqlite"))?;
        let observer = BenchmarkObserver {
            failures: AtomicUsize::new(0),
        };

        let started = Instant::now();
        let summary = index_dir_observed(
            &mut assets,
            &mut db,
            &mut models,
            &corpus,
            IndexOptions {
                ocr: ocr_options,
                limit: arguments.limit,
                ..IndexOptions::default()
            },
            &observer,
        )?;
        let elapsed = started.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let images_per_second = summary.indexed as f64 / elapsed_seconds;
        let ocr_content_digest = ocr_content_digest(&db, &corpus)?;

        println!(
            "{run},{configured_provider},{},{replica_count},{},{},{},{},{elapsed_seconds:.6},{images_per_second:.6},{ocr_content_digest:016x}",
            arguments.threads,
            arguments.recognition_batch_size,
            arguments.detection_max_side,
            summary.indexed,
            observer.failures(),
        );
    }

    Ok(())
}

fn ocr_content_digest(db: &DB, corpus: &Utf8Path) -> Result<u64> {
    let pending = db.pending_text_embeddings(
        TextEmbeddingSpace::OcrText,
        &SearchFilters::new(corpus),
        1_000_000,
        16 * 1024 * 1024,
    )?;
    let mut hasher = DefaultHasher::new();
    for item in pending {
        item.asset_id.hash(&mut hasher);
        item.content.hash(&mut hasher);
    }
    Ok(hasher.finish())
}

fn parse_arguments() -> Result<Arguments> {
    let mut corpus = None;
    let mut detection_model = None;
    let mut detection_config = None;
    let mut recognition_model = None;
    let mut recognition_config = None;
    let mut provider = None;
    let mut threads = None;
    let mut recognition_batch_size = None;
    let mut detection_max_side = None;
    let mut replicas = None;
    let mut limit = None;
    let mut trace_jsonl = None;
    let mut runs = None;
    let mut warmup_runs = None;
    let mut values = env::args_os().skip(1);

    while let Some(argument) = values.next() {
        let argument = argument.into_string().map_err(|argument| {
            anyhow!(
                "argument is not valid UTF-8: {}",
                argument.to_string_lossy()
            )
        })?;
        match argument.as_str() {
            "--corpus" => set_once(
                &mut corpus,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--detection-model" => set_once(
                &mut detection_model,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--detection-config" => set_once(
                &mut detection_config,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--recognition-model" => set_once(
                &mut recognition_model,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--recognition-config" => set_once(
                &mut recognition_config,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--provider" => set_once(
                &mut provider,
                next_value(&mut values, &argument)?
                    .parse::<ExecutionProvider>()
                    .map_err(|error| anyhow!("invalid --provider: {error}"))?,
                &argument,
            )?,
            "--threads" => set_once(
                &mut threads,
                parse_nonzero(&next_value(&mut values, &argument)?, &argument)?,
                &argument,
            )?,
            "--recognition-batch-size" => set_once(
                &mut recognition_batch_size,
                parse_nonzero(&next_value(&mut values, &argument)?, &argument)?,
                &argument,
            )?,
            "--detection-max-side" => set_once(
                &mut detection_max_side,
                u32::try_from(parse_nonzero(
                    &next_value(&mut values, &argument)?,
                    &argument,
                )?)
                .map_err(|_| anyhow!("invalid {argument}: value is too large"))?,
                &argument,
            )?,
            "--replicas" => set_once(
                &mut replicas,
                NonZeroUsize::new(parse_nonzero(
                    &next_value(&mut values, &argument)?,
                    &argument,
                )?)
                .expect("parse_nonzero rejects zero"),
                &argument,
            )?,
            "--limit" => set_once(
                &mut limit,
                parse_nonzero(&next_value(&mut values, &argument)?, &argument)?,
                &argument,
            )?,
            "--trace-jsonl" => set_once(
                &mut trace_jsonl,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--runs" => set_once(
                &mut runs,
                parse_nonzero(&next_value(&mut values, &argument)?, &argument)?,
                &argument,
            )?,
            "--warmup-runs" => set_once(
                &mut warmup_runs,
                parse_nonzero(&next_value(&mut values, &argument)?, &argument)?,
                &argument,
            )?,
            // Cargo appends this libtest compatibility flag even for a harness-free bench target.
            "--bench" => {}
            _ if argument.starts_with("--") => bail!("unknown flag: {argument}"),
            _ => bail!("unexpected argument: {argument}"),
        }
    }

    Ok(Arguments {
        corpus: required(corpus, "--corpus")?,
        detection_model: required(detection_model, "--detection-model")?,
        detection_config: required(detection_config, "--detection-config")?,
        recognition_model: required(recognition_model, "--recognition-model")?,
        recognition_config: required(recognition_config, "--recognition-config")?,
        provider: provider.unwrap_or(ExecutionProvider::Cpu),
        threads: threads.unwrap_or(4),
        recognition_batch_size: recognition_batch_size
            .unwrap_or(PaddleOcrOptions::default().recognition_batch_size),
        detection_max_side: detection_max_side
            .unwrap_or(PaddleOcrOptions::default().detection_max_side),
        replicas,
        limit,
        trace_jsonl,
        runs: runs.unwrap_or(3),
        warmup_runs: warmup_runs.unwrap_or(1),
    })
}

fn next_value(values: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<String> {
    let value = values
        .next()
        .ok_or_else(|| anyhow!("missing value for {flag}"))?
        .into_string()
        .map_err(|value| {
            anyhow!(
                "value for {flag} is not valid UTF-8: {}",
                value.to_string_lossy()
            )
        })?;
    if value.starts_with("--") {
        bail!("missing value for {flag}");
    }
    Ok(value)
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<()> {
    if slot.replace(value).is_some() {
        bail!("duplicate flag: {flag}");
    }
    Ok(())
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T> {
    value.ok_or_else(|| anyhow!("missing required flag: {flag}"))
}

fn parse_nonzero(value: &str, flag: &str) -> Result<usize> {
    let value = value
        .parse::<usize>()
        .map_err(|error| anyhow!("invalid value for {flag}: {error}"))?;
    if value == 0 {
        bail!("{flag} must be greater than zero");
    }
    Ok(value)
}

fn first_supported_image(corpus: &Utf8Path) -> Result<RgbImage> {
    for entry in WalkDir::new(corpus).follow_links(true) {
        let entry = entry.context("walking corpus for warmup image")?;
        if !entry.file_type().is_file() || !is_supported_ocr_image(entry.path()) {
            continue;
        }
        let data = std::fs::read(entry.path())
            .with_context(|| format!("opening warmup image: {}", entry.path().display()))?;
        let raster = nicegal_core::imaging::decode(&data)
            .with_context(|| format!("decoding warmup image: {}", entry.path().display()))?;
        let (width, height) = (raster.width(), raster.height());
        return RgbImage::from_raw(width, height, raster.into_rgb_bytes())
            .context("assembling warmup image buffer");
    }
    bail!("--corpus contains no supported OCR image for warmup: {corpus}")
}

fn is_supported_ocr_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpeg" | "jpg" | "gif" | "webp"
            )
        })
}

fn utf8_path(path: &Path, label: &str) -> Result<Utf8PathBuf> {
    Utf8PathBuf::from_path_buf(path.to_owned())
        .map_err(|path| anyhow!("{label} is not valid UTF-8: {}", path.display()))
}

fn print_help() {
    println!(
        "Usage: cargo bench --bench ocr_index -- \\\n  --corpus <DIR> \\\n  --detection-model <FILE> \\\n  --detection-config <FILE> \\\n  --recognition-model <FILE> \\\n  --recognition-config <FILE> \\\n  [--provider <{}>] \\\n  [--threads <N>] [--recognition-batch-size <N>] \\\n  [--trace-jsonl <FILE>] [--runs <N>] [--warmup-runs <N>]\n\n\
Measures the full catalog, decode, PaddleOCR, and SQLite indexing pipeline.\n\
The requested execution provider is loaded without CPU fallback. Tracing uses RUST_LOG and writes\n\
one JSON object per line to stderr or --trace-jsonl.",
        ExecutionProvider::VARIANTS.join("|")
    );
}
