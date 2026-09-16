//! Two independent layers are installed on one subscriber:
//!
//! - A **console** layer on stderr: compact human text filtered by [`DEFAULT_DIRECTIVES`].
//! - A **file** layer, JSON, in the caller's log directory: filtered by `RUST_LOG` or `--log`,
//!   with the default directives used when neither is set.
//!
//! Both writers use background I/O. Keep [`LoggingGuards`] alive to flush buffered lines.

use std::fs::{self, OpenOptions};

use anyhow::{Context, Result};
use camino::Utf8Path as Path;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Directives used for the JSON file layer when neither `RUST_LOG` nor an explicit override is
/// given, and unconditionally for the console layer.
const DEFAULT_DIRECTIVES: &str = "warn,nicegal_core=info,nicegal_server=info,nom_exif=off";

const LOG_FILE_NAME: &str = "nicegal-server.log";

const KEPT_GENERATIONS: u32 = 3;

/// Keeps the non-blocking writers' background flush threads alive.
#[must_use]
pub struct LoggingGuards {
    shutdown: Box<dyn Fn() + Send + Sync>,
    _console: WorkerGuard,
    _file: WorkerGuard,
}

impl Drop for LoggingGuards {
    fn drop(&mut self) {
        // Stop events before formatter TLS is destroyed; then WorkerGuard flushes queued lines.
        (self.shutdown)();
    }
}

pub fn init(log_directory: &Path, override_directives: Option<&str>) -> Result<LoggingGuards> {
    fs::create_dir_all(log_directory)
        .with_context(|| format!("creating log directory: {log_directory}"))?;
    let log_path = log_directory.join(LOG_FILE_NAME);
    rotate(&log_path).context("rotating the previous log file")?;

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening log file: {log_path}"))?;
    let (file_writer, file_guard) = tracing_appender::non_blocking(file);
    let (console_writer, console_guard) = tracing_appender::non_blocking(std::io::stderr());

    let file_filter = match (std::env::var("RUST_LOG"), override_directives) {
        (Ok(directives), _) => {
            EnvFilter::try_new(directives).context("RUST_LOG is not a valid tracing filter")?
        }
        (Err(_), Some(directives)) => {
            EnvFilter::try_new(directives).context("log level is not a valid tracing filter")?
        }
        (Err(_), None) => EnvFilter::new(DEFAULT_DIRECTIVES),
    };
    let file_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(file_writer)
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(file_filter);

    let console_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_writer(console_writer)
        .with_ansi(true)
        .with_filter(EnvFilter::new(DEFAULT_DIRECTIVES));

    let (shutdown_filter, shutdown_handle) =
        tracing_subscriber::reload::Layer::new(tracing_subscriber::filter::LevelFilter::TRACE);
    tracing_subscriber::registry()
        .with(file_layer)
        .with(console_layer)
        .with(shutdown_filter)
        .try_init()
        .map_err(|error| anyhow::anyhow!("installing the log writer: {error}"))?;

    Ok(LoggingGuards {
        shutdown: Box::new(move || {
            let _ = shutdown_handle.reload(tracing_subscriber::filter::LevelFilter::OFF);
        }),
        _console: console_guard,
        _file: file_guard,
    })
}

fn rotate(path: &Path) -> Result<()> {
    let oldest = generation_path(path, KEPT_GENERATIONS - 1);
    if oldest.exists() {
        fs::remove_file(&oldest).with_context(|| format!("removing oldest log file: {oldest}"))?;
    }
    for generation in (1..KEPT_GENERATIONS - 1).rev() {
        let from = generation_path(path, generation);
        let to = generation_path(path, generation + 1);
        if from.exists() {
            fs::rename(&from, &to).with_context(|| format!("rotating log file: {from} -> {to}"))?;
        }
    }
    if path.exists() {
        let to = generation_path(path, 1);
        fs::rename(path, &to).with_context(|| format!("rotating log file: {path} -> {to}"))?;
    }
    Ok(())
}

fn generation_path(path: &Path, generation: u32) -> camino::Utf8PathBuf {
    camino::Utf8PathBuf::from(format!("{path}.{generation}"))
}
