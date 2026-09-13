use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[cfg(windows)]
use std::ffi::c_void;

use anyhow::{Context, Result, anyhow, bail};
use camino::{Utf8Path, Utf8PathBuf};
use image::RgbImage;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::embedding::{ImageEmbedder, ImageEmbedderOptions};
use nicegal_core::image_index::{ImageIndexDb, index_images_observed};
use nicegal_core::index::{IndexEvent, IndexObserver, IndexOptions, catalog_dir_observed};
use nicegal_core::runtime::{ExecutionProvider, RuntimeOptions};
use strum::VariantNames;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use walkdir::WalkDir;

struct Arguments {
    corpus: PathBuf,
    provider: ExecutionProvider,
    threads: usize,
    batch_size: usize,
    runs: usize,
    warmup_runs: usize,
    trace_jsonl: Option<PathBuf>,
}

#[derive(Default)]
struct BenchmarkObserver {
    embedded: AtomicUsize,
    failures: AtomicUsize,
}

struct ProcessMemoryUsage {
    working_set_bytes: usize,
    peak_working_set_bytes: usize,
}

impl IndexObserver for BenchmarkObserver {
    fn on_event(&self, event: IndexEvent) {
        if let IndexEvent::Progress(delta) = event {
            self.embedded.fetch_add(delta.embedded, Ordering::Relaxed);
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
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "warn,nicegal_core::image_index=debug,nicegal_core::embedding::image=debug,nom_exif=off",
        )
    });
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
    let corpus = nicegal_core::assets::canonicalize_path(&utf8_path(&arguments.corpus, "--corpus")?)
        .context("canonicalizing corpus")?;
    nicegal_core::runtime::initialize_bundled_runtime(arguments.provider)?;
    let runtime = RuntimeOptions {
        execution_provider: arguments.provider,
        intra_threads: NonZeroUsize::new(arguments.threads)
            .expect("argument parsing rejects zero thread counts"),
        allow_cpu_fallback: false,
        replicas: None,
    };
    let options = ImageEmbedderOptions {
        max_batch_size: arguments.batch_size,
        runtime,
        ..ImageEmbedderOptions::default()
    };

    let model_load_started = Instant::now();
    let embedder = ImageEmbedder::load(&options).context("loading the CLIP image model")?;
    let model_load_duration = model_load_started.elapsed();

    let temporary = tempfile::tempdir().context("creating benchmark directory")?;
    let database_directory = utf8_path(temporary.path(), "benchmark directory")?;
    let assets = AssetCatalog::new(&database_directory.join("assets.db"))?;
    let catalog_started = Instant::now();
    let catalog_summary = catalog_dir_observed(
        &assets,
        &corpus,
        IndexOptions::default(),
        &BenchmarkObserver::default(),
    )?;
    let catalog_duration = catalog_started.elapsed();

    let warmup_image = first_supported_image(&corpus)?;
    let warmup_pixels = embedder
        .preprocess_image(warmup_image)
        .context("preprocessing CLIP warmup image")?;
    let warmup_started = Instant::now();
    for _ in 0..arguments.warmup_runs {
        embedder.embed_preprocessed_images(vec![warmup_pixels.clone()])?;
    }
    let warmup_duration = warmup_started.elapsed();

    println!("os={}", env::consts::OS);
    println!("architecture={}", env::consts::ARCH);
    match std::thread::available_parallelism() {
        Ok(parallelism) => println!("logical_parallelism={}", parallelism.get()),
        Err(_) => println!("logical_parallelism=unavailable"),
    }
    println!("corpus={corpus}");
    println!("cataloged={}", catalog_summary.cataloged);
    println!("requested_provider={}", arguments.provider);
    println!("configured_provider={}", embedder.execution_provider());
    println!("threads={}", arguments.threads);
    println!("batch_size={}", arguments.batch_size);
    println!("model={}", embedder.model());
    println!("dimensions={}", embedder.dimensions());
    println!(
        "model_load_seconds={:.6}",
        model_load_duration.as_secs_f64()
    );
    println!("catalog_seconds={:.6}", catalog_duration.as_secs_f64());
    println!("warmup_seconds={:.6}", warmup_duration.as_secs_f64());
    println!(
        "trace_jsonl={}",
        arguments
            .trace_jsonl
            .as_deref()
            .map_or_else(|| "stderr".into(), |path| path.display().to_string())
    );
    println!(
        "run,configured_provider,threads,batch_size,embedded,failed,elapsed_seconds,images_per_second,working_set_bytes,peak_working_set_bytes"
    );

    for run in 1..=arguments.runs {
        let database = database_directory.join(format!("image-index-{run}.db"));
        let mut image_index = ImageIndexDb::new(&database, embedder.dimensions())?;
        let observer = BenchmarkObserver::default();
        let started = Instant::now();
        let cancelled = index_images_observed(
            &assets,
            &mut image_index,
            &embedder,
            &corpus,
            false,
            false,
            &observer,
        )?;
        let elapsed = started.elapsed();
        if cancelled {
            bail!("benchmark observer unexpectedly cancelled the run");
        }
        let embedded = observer.embedded.load(Ordering::Relaxed);
        let failed = observer.failures.load(Ordering::Relaxed);
        let elapsed_seconds = elapsed.as_secs_f64();
        let images_per_second = embedded as f64 / elapsed_seconds;
        let memory = process_memory_usage();
        let working_set = memory.as_ref().map_or_else(
            || "unavailable".to_owned(),
            |usage| usage.working_set_bytes.to_string(),
        );
        let peak_working_set = memory.as_ref().map_or_else(
            || "unavailable".to_owned(),
            |usage| usage.peak_working_set_bytes.to_string(),
        );
        println!(
            "{run},{},{},{},{embedded},{failed},{elapsed_seconds:.6},{images_per_second:.6},{working_set},{peak_working_set}",
            embedder.execution_provider(),
            arguments.threads,
            arguments.batch_size,
        );
    }
    Ok(())
}

#[cfg(windows)]
fn process_memory_usage() -> Option<ProcessMemoryUsage> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            counters_size: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: u32::try_from(std::mem::size_of::<ProcessMemoryCounters>()).ok()?,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    // SAFETY: GetCurrentProcess returns a pseudo-handle for this process, and `counters` is a
    // writable C-compatible buffer whose byte size is supplied in both its `cb` field and call.
    let succeeded =
        unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &raw mut counters, counters.cb) } != 0;
    succeeded.then_some(ProcessMemoryUsage {
        working_set_bytes: counters.working_set_size,
        peak_working_set_bytes: counters.peak_working_set_size,
    })
}

#[cfg(not(windows))]
fn process_memory_usage() -> Option<ProcessMemoryUsage> {
    None
}

fn parse_arguments() -> Result<Arguments> {
    let mut corpus = None;
    let mut provider = None;
    let mut threads = None;
    let mut batch_size = None;
    let mut runs = None;
    let mut warmup_runs = None;
    let mut trace_jsonl = None;
    let mut values = env::args_os().skip(1);
    while let Some(argument) = values.next() {
        let argument = argument
            .into_string()
            .map_err(|value| anyhow!("argument is not UTF-8: {}", value.to_string_lossy()))?;
        match argument.as_str() {
            "--corpus" => set_once(
                &mut corpus,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--provider" => set_once(
                &mut provider,
                next_value(&mut values, &argument)?
                    .parse()
                    .map_err(|error| anyhow!("invalid {argument}: {error}"))?,
                &argument,
            )?,
            "--threads" => set_once(
                &mut threads,
                parse_nonzero(&next_value(&mut values, &argument)?, &argument)?,
                &argument,
            )?,
            "--batch-size" => set_once(
                &mut batch_size,
                parse_nonzero(&next_value(&mut values, &argument)?, &argument)?,
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
            "--trace-jsonl" => set_once(
                &mut trace_jsonl,
                PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            "--bench" => {}
            _ if argument.starts_with("--") => bail!("unknown flag: {argument}"),
            _ => bail!("unexpected argument: {argument}"),
        }
    }
    Ok(Arguments {
        corpus: corpus.unwrap_or_else(|| PathBuf::from("../testdata")),
        provider: provider.unwrap_or(ExecutionProvider::Cpu),
        threads: threads.unwrap_or(4),
        batch_size: batch_size.unwrap_or(8),
        runs: runs.unwrap_or(3),
        warmup_runs: warmup_runs.unwrap_or(1),
        trace_jsonl,
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
        if !entry.file_type().is_file() || !is_supported_image(entry.path()) {
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
    bail!("--corpus contains no supported image: {corpus}")
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpeg" | "jpg" | "gif" | "webp" | "bmp"
            )
        })
}

fn utf8_path(path: &Path, label: &str) -> Result<Utf8PathBuf> {
    Utf8PathBuf::from_path_buf(path.to_owned())
        .map_err(|path| anyhow!("{label} is not valid UTF-8: {}", path.display()))
}

fn print_help() {
    println!(
        "Usage: cargo bench --bench image_index -- \\\n  [--corpus <DIR>] [--provider <{}>] [--threads <N>] [--batch-size <N>] \\\n  [--runs <N>] [--warmup-runs <N>] [--trace-jsonl <FILE>]\n\n\
Measures catalog-backed decode, batched CLIP inference, and SQLite-vec ingestion. The corpus\n\
defaults to ../testdata. The requested provider has CPU fallback disabled so a failed setup is\n\
never reported as a mislabeled result. Set RUST_LOG for trace selection; JSONL is written to\n\
stderr unless --trace-jsonl is supplied.",
        ExecutionProvider::VARIANTS.join("|")
    );
}
