use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use nicegal_core::embedding::{TextEmbedder, TextEmbedderOptions};
use nicegal_core::runtime::{ExecutionProvider, RuntimeOptions};
use strum::VariantNames;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

struct Arguments {
    provider: ExecutionProvider,
    threads: usize,
    batch_sizes: Vec<usize>,
    runs: usize,
    warmup_runs: usize,
    trace_jsonl: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let arguments: Vec<OsString> = env::args_os().skip(1).collect();
    if arguments.iter().any(|argument| argument == "--help") {
        print_help();
        return Ok(());
    }
    if !arguments.iter().any(|argument| argument == "--provider") {
        eprintln!("text_embed_index benchmark skipped; pass --provider <PROVIDER> to run it");
        return Ok(());
    }

    let arguments = parse_arguments()?;
    init_tracing(arguments.trace_jsonl.as_deref())?;
    run(arguments)
}

fn init_tracing(path: Option<&Path>) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,nicegal_core::embedding=debug,nom_exif=off"));
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
    let largest_batch = *arguments
        .batch_sizes
        .iter()
        .max()
        .context("--batch-sizes must name at least one size")?;

    let runtime_options = RuntimeOptions {
        execution_provider: arguments.provider,
        intra_threads: std::num::NonZeroUsize::new(arguments.threads)
            .expect("argument parsing rejects zero thread counts"),
        // Disabled so a setup failure fails the run instead of silently reporting a mislabeled
        // CPU result, matching benches/ocr_index.rs.
        allow_cpu_fallback: false,
        replicas: None,
    };
    let embedder_options = TextEmbedderOptions {
        max_batch_size: largest_batch,
        runtime: runtime_options,
        ..TextEmbedderOptions::default()
    };

    let model_load_started = Instant::now();
    let embedder =
        TextEmbedder::load(&embedder_options).context("loading the text embedding model")?;
    let model_load_duration = model_load_started.elapsed();

    let corpus = synthetic_corpus(largest_batch);

    println!("os={}", env::consts::OS);
    println!("architecture={}", env::consts::ARCH);
    match std::thread::available_parallelism() {
        Ok(parallelism) => println!("logical_parallelism={}", parallelism.get()),
        Err(_) => println!("logical_parallelism=unavailable"),
    }
    println!("requested_provider={}", arguments.provider);
    println!("configured_provider={}", embedder.execution_provider());
    println!("threads={}", arguments.threads);
    println!(
        "batch_sizes={}",
        arguments
            .batch_sizes
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("model={}", embedder.model());
    println!("dimensions={}", embedder.dimensions());
    println!(
        "model_load_seconds={:.6}",
        model_load_duration.as_secs_f64()
    );
    println!(
        "trace_jsonl={}",
        arguments
            .trace_jsonl
            .as_deref()
            .map_or_else(|| "stderr".into(), |path| path.display().to_string())
    );
    println!(
        "run,configured_provider,threads,batch_size,rows,bytes,elapsed_seconds,rows_per_second,mib_per_second"
    );

    for &batch_size in &arguments.batch_sizes {
        let texts: Vec<&str> = corpus[..batch_size].iter().map(String::as_str).collect();
        let bytes: usize = texts.iter().map(|text| text.len()).sum();

        for _ in 0..arguments.warmup_runs {
            embedder
                .embed_documents(&texts)
                .with_context(|| format!("warming up batch size {batch_size}"))?;
        }

        for run in 1..=arguments.runs {
            let started = Instant::now();
            let vectors = embedder
                .embed_documents(&texts)
                .with_context(|| format!("embedding batch size {batch_size}"))?;
            let elapsed = started.elapsed();
            if vectors.len() != texts.len() {
                bail!(
                    "expected {} vectors, got {} for batch size {batch_size}",
                    texts.len(),
                    vectors.len()
                );
            }
            let elapsed_seconds = elapsed.as_secs_f64();
            let rows_per_second = texts.len() as f64 / elapsed_seconds;
            let mib_per_second = (bytes as f64 / (1024.0 * 1024.0)) / elapsed_seconds;
            println!(
                "{run},{},{},{batch_size},{},{bytes},{elapsed_seconds:.6},{rows_per_second:.6},{mib_per_second:.6}",
                embedder.execution_provider(),
                arguments.threads,
                texts.len(),
            );
        }
    }

    Ok(())
}

/// Deterministic English-like OCR-shaped text: mostly short receipt/label lines, with a periodic
/// long paragraph to exercise the padding/truncation cost called out in
/// `EMBEDDING_INDEX_PERFORMANCE.md`. Synthetic rather than a real corpus so the bench needs no
/// external OCR database to run.
fn synthetic_corpus(rows: usize) -> Vec<String> {
    const WORDS: &[&str] = &[
        "invoice",
        "total",
        "subtotal",
        "tax",
        "date",
        "receipt",
        "store",
        "quantity",
        "price",
        "item",
        "customer",
        "order",
        "number",
        "warranty",
        "return",
        "policy",
        "thank",
        "you",
        "for",
        "shopping",
        "with",
        "us",
        "card",
        "cash",
        "change",
        "due",
        "paid",
        "balance",
        "account",
        "reference",
        "shipping",
        "address",
        "phone",
        "email",
        "signature",
        "approved",
        "declined",
        "gateway",
        "terminal",
        "batch",
        "auth",
        "code",
        "member",
        "loyalty",
        "points",
        "discount",
        "coupon",
        "expires",
        "valid",
        "until",
        "manager",
        "cashier",
        "register",
    ];

    let mut random_state: u64 = 0x9E3779B97F4A7C15;
    let mut next_random = move || {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 7;
        random_state ^= random_state << 17;
        random_state
    };

    (0..rows)
        .map(|index| {
            // Every 32nd row is a long-text stress case near the model's input cap; the rest are
            // short lines typical of receipts and labels.
            let word_count = if index % 32 == 0 {
                900
            } else {
                3 + (next_random() % 12) as usize
            };
            (0..word_count)
                .map(|_| WORDS[(next_random() % WORDS.len() as u64) as usize])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn parse_arguments() -> Result<Arguments> {
    let mut provider = None;
    let mut threads = None;
    let mut batch_sizes = None;
    let mut runs = None;
    let mut warmup_runs = None;
    let mut trace_jsonl = None;
    let mut values = env::args_os().skip(1);

    while let Some(argument) = values.next() {
        let argument = argument.into_string().map_err(|argument| {
            anyhow!(
                "argument is not valid UTF-8: {}",
                argument.to_string_lossy()
            )
        })?;
        match argument.as_str() {
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
            "--batch-sizes" => set_once(
                &mut batch_sizes,
                next_value(&mut values, &argument)?
                    .split(',')
                    .map(|value| parse_nonzero(value.trim(), &argument))
                    .collect::<Result<Vec<_>>>()?,
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
                std::path::PathBuf::from(next_value(&mut values, &argument)?),
                &argument,
            )?,
            // Cargo appends this libtest compatibility flag even for a harness-free bench target.
            "--bench" => {}
            _ if argument.starts_with("--") => bail!("unknown flag: {argument}"),
            _ => bail!("unexpected argument: {argument}"),
        }
    }

    Ok(Arguments {
        provider: provider.unwrap_or(ExecutionProvider::Cpu),
        threads: threads.unwrap_or(4),
        batch_sizes: batch_sizes.unwrap_or_else(|| vec![1, 8, 32, 64, 128, 256, 512, 1024]),
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

fn required(value: usize, flag: &str) -> Result<usize> {
    if value == 0 {
        bail!("{flag} must be greater than zero");
    }
    Ok(value)
}

fn parse_nonzero(value: &str, flag: &str) -> Result<usize> {
    let value = value
        .parse::<usize>()
        .map_err(|error| anyhow!("invalid value for {flag}: {error}"))?;
    required(value, flag)
}

fn print_help() {
    println!(
        "Usage: cargo bench --bench text_embed_index -- \\\n  [--provider <{}>] \\\n  [--threads <N>] [--batch-sizes <CSV>] \\\n  [--runs <N>] [--warmup-runs <N>] [--trace-jsonl <FILE>]\n\n\
Measures TextEmbedder::embed_documents throughput on a synthetic OCR-shaped corpus across batch\n\
sizes. The requested execution provider is loaded without CPU fallback, so a setup failure fails\n\
the run instead of silently reporting a mislabeled CPU result. Tracing uses RUST_LOG and writes\n\
one JSON object per line to stderr or --trace-jsonl.\n\n\
Default batch sizes match the sweep in EMBEDDING_INDEX_PERFORMANCE.md: 1,8,32,64,128,256,512,1024.",
        ExecutionProvider::VARIANTS.join("|")
    );
}
