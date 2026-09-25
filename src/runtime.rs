//! ONNX Runtime session construction, shared by every model family in this crate.
//!
//! [`RuntimeOptions`] is per model load, but the dynamically loaded runtime library is
//! process-wide. Provider availability is checked against that library before registration.

use std::fmt;
use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ort::ep::{CPU, ExecutionProviderDispatch};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use strum::{Display, EnumString, IntoStaticStr, VariantNames};
use tracing::{Span, field, instrument, warn};

/// An ONNX Runtime execution provider a model can be compiled for.
///
/// The canonical name (what [`std::str::FromStr`] accepts and [`Display`](fmt::Display) reports)
/// is the variant name lowercased, e.g. `OpenVino` <-> `"openvino"`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    Display,
    EnumString,
    IntoStaticStr,
    VariantNames,
)]
#[strum(serialize_all = "lowercase")]
#[strum(parse_err_ty = ParseExecutionProviderError, parse_err_fn = unsupported_execution_provider)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    OpenVino,
    Directml,
    Webgpu,
    CoreML,
}

fn unsupported_execution_provider(provider: &str) -> ParseExecutionProviderError {
    ParseExecutionProviderError(provider.to_owned())
}

/// Error returned when a string does not name an [`ExecutionProvider`] this crate knows about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseExecutionProviderError(String);

impl fmt::Display for ParseExecutionProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unsupported execution provider `{}`; expected one of: {}",
            self.0,
            ExecutionProvider::VARIANTS.join(", ")
        )
    }
}

impl std::error::Error for ParseExecutionProviderError {}

/// ONNX Runtime settings used while compiling one family of models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeOptions {
    /// The preferred execution provider. If it cannot be configured, CPU can be used as a fallback.
    pub execution_provider: ExecutionProvider,
    /// Non-zero thread count used by CPU or OpenVINO as appropriate for the provider.
    pub intra_threads: NonZeroUsize,
    /// Whether an unavailable or failing non-CPU provider should rebuild every model on CPU.
    pub allow_cpu_fallback: bool,
    /// How many independent copies of a model family to compile, or `None` to let the family pick
    /// a count from the provider actually configured. Only families that can keep several
    /// inferences in flight at once read this; the rest compile one copy regardless.
    pub replicas: Option<NonZeroUsize>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::Cpu,
            intra_threads: NonZeroUsize::new(4).expect("four is non-zero"),
            allow_cpu_fallback: true,
            replicas: None,
        }
    }
}

impl RuntimeOptions {
    /// Whether a provider failure should be retried on CPU. A CPU failure never is: the retry
    /// would be the build that just failed.
    fn can_fall_back_to_cpu(self) -> bool {
        self.allow_cpu_fallback && self.execution_provider != ExecutionProvider::Cpu
    }
}

/// Try the requested provider and its allowed fallbacks, returning the provider that succeeded.
pub(crate) fn with_fallback<T>(
    options: RuntimeOptions,
    mut attempt: impl FnMut(ExecutionProvider) -> Result<T>,
) -> Result<(T, ExecutionProvider)> {
    let mut requested = options.execution_provider;
    let mut last_error = match attempt(requested) {
        Ok(value) => return Ok((value, requested)),
        Err(error) => error,
    };
    if !options.can_fall_back_to_cpu() {
        return Err(last_error)
            .with_context(|| format!("loading with the {requested} execution provider"));
    }

    for &fallback in fallback_chain(options.execution_provider) {
        warn!(
            execution_provider = %requested,
            error = %format_args!("{last_error:#}"),
            "execution provider failed; retrying on the next fallback provider"
        );
        requested = fallback;
        match attempt(requested) {
            Ok(value) => return Ok((value, requested)),
            Err(error) => last_error = error,
        }
    }
    Err(last_error).context("rebuilding on the CPU execution provider")
}

/// One model file to compile, with the label naming it in errors and logs.
#[derive(Debug, Clone, Copy)]
pub struct SessionSpec<'a> {
    pub label: &'a str,
    pub path: &'a Path,
}

impl<'a> SessionSpec<'a> {
    pub fn new(label: &'a str, path: &'a Path) -> Self {
        Self { label, path }
    }
}

/// The compiled sessions and the provider they were actually built for, which is not the requested
/// one when a fallback happened.
pub struct LoadedSessions<const N: usize> {
    pub sessions: [Session; N],
    pub execution_provider: ExecutionProvider,
}

/// Initialize `ort` from an application-selected dynamic library before any model is loaded.
///
/// `ort` binds its library process-wide, so executable applications must call this once before
/// constructing a [`Session`] directly or through a library such as FastEmbed.
pub fn initialize_from_dylib(path: &Path) -> Result<()> {
    ort::init_from(path)
        .map_err(|error| anyhow!("{error}"))
        .with_context(|| format!("loading ONNX Runtime from {}", path.display()))?
        .commit();
    Ok(())
}

/// Select the bundled ONNX Runtime distribution next to the current executable.
///
/// Either accelerated distribution includes CPU. CPU therefore uses the broadly applicable
/// DirectML bundle, while OpenVINO selects its own bundle. This must run before any model load.
#[cfg(windows)]
pub fn initialize_bundled_runtime(execution_provider: ExecutionProvider) -> Result<()> {
    let runtime_distribution = match execution_provider {
        ExecutionProvider::OpenVino => "openvino",
        ExecutionProvider::Cpu | ExecutionProvider::Directml => "directml",
        ExecutionProvider::Webgpu => {
            bail!("{execution_provider} is only supported on Linux")
        }
        ExecutionProvider::CoreML => {
            bail!("{execution_provider} is only supported on macOS")
        }
    };
    let executable = std::env::current_exe().context("resolving the executable path")?;
    let executable_directory = executable
        .parent()
        .context("executable path has no parent directory")?;
    let runtime_directory = executable_directory
        .join("onnxruntime")
        .join(runtime_distribution);
    let dylib_path = runtime_directory.join("onnxruntime.dll");
    if !dylib_path.is_file() {
        bail!(
            "the {runtime_distribution} ONNX Runtime distribution is missing: expected {}",
            dylib_path.display()
        );
    }

    let current = std::env::var_os("PATH").unwrap_or_default();
    let path = std::env::join_paths(
        std::iter::once(runtime_directory.clone()).chain(std::env::split_paths(&current)),
    )
    .context("constructing ONNX Runtime DLL search path")?;
    // Called before models or serving threads exist; no other thread can observe a partial update.
    unsafe { std::env::set_var("PATH", path) };
    initialize_from_dylib(&dylib_path)?;
    tracing::info!(
        execution_provider = %execution_provider,
        runtime_distribution,
        path = %dylib_path.display(),
        "initialized ONNX Runtime"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn initialize_bundled_runtime(execution_provider: ExecutionProvider) -> Result<()> {
    let distribution = match execution_provider {
        ExecutionProvider::Cpu if cfg!(feature = "ort-webgpu") => "webgpu",
        ExecutionProvider::Cpu | ExecutionProvider::OpenVino => "openvino",
        ExecutionProvider::Webgpu => "webgpu",
        ExecutionProvider::Directml => bail!("directml is only supported on Windows"),
        ExecutionProvider::CoreML => bail!("coreml is only supported on macOS"),
    };
    let executable = std::env::current_exe().context("resolving the executable path")?;
    let directory = executable
        .parent()
        .context("executable path has no parent directory")?;
    let dylib = directory
        .join("onnxruntime")
        .join(distribution)
        .join("libonnxruntime.so");
    if !dylib.is_file() {
        bail!(
            "the {distribution} ONNX Runtime distribution is missing: expected {}",
            dylib.display()
        );
    }
    initialize_from_dylib(&dylib)?;
    #[cfg(feature = "ort-webgpu")]
    if execution_provider == ExecutionProvider::Webgpu {
        ort::environment::Environment::current()?
            .register_ep_library(
                "webgpu",
                directory.join("onnxruntime/webgpu/libonnxruntime_providers_webgpu.so"),
            )
            .context("registering native WebGPU plugin")?;
    }
    tracing::info!(%execution_provider, distribution, path = %dylib.display(), "initialized ONNX Runtime");
    Ok(())
}

/// Select the bundled macOS ONNX Runtime distribution. The regular 1.30 wheel ships CoreML and
/// CPU together, so both choices load the same library.
#[cfg(target_os = "macos")]
pub fn initialize_bundled_runtime(execution_provider: ExecutionProvider) -> Result<()> {
    match execution_provider {
        ExecutionProvider::CoreML | ExecutionProvider::Cpu => {}
        ExecutionProvider::OpenVino | ExecutionProvider::Directml | ExecutionProvider::Webgpu => {
            bail!("{execution_provider} is not supported on macOS")
        }
    }
    let executable = std::env::current_exe().context("resolving the executable path")?;
    let directory = executable
        .parent()
        .context("executable path has no parent directory")?;
    let runtime_directory = directory.join("onnxruntime/coreml");
    let dylib = runtime_directory.join("libonnxruntime.dylib");
    if !dylib.is_file() {
        bail!(
            "the coreml ONNX Runtime distribution is missing: expected {}",
            dylib.display()
        );
    }
    initialize_from_dylib(&dylib)?;
    tracing::info!(%execution_provider, runtime_distribution = "coreml", path = %dylib.display(), "initialized ONNX Runtime");
    Ok(())
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub fn initialize_bundled_runtime(_execution_provider: ExecutionProvider) -> Result<()> {
    Ok(())
}

/// Build metadata reported by the process-wide ONNX Runtime library selected at startup.
pub fn onnxruntime_build_info() -> &'static str {
    ort::info()
}

/// Compile every model in one family onto a single execution provider.
///
/// All-or-nothing: if one model cannot be built for the requested provider the whole family is
/// rebuilt on the next provider in [`fallback_chain`], so the provider it reports is one that
/// every session in it actually uses.
#[instrument(
    name = "ort_load",
    skip_all,
    fields(
        models = N,
        requested_execution_provider = %options.execution_provider,
        execution_provider = field::Empty
    )
)]
pub fn load_sessions<const N: usize>(
    specs: [SessionSpec<'_>; N],
    options: RuntimeOptions,
) -> Result<LoadedSessions<N>> {
    let (sessions, execution_provider) = with_fallback(options, |provider| {
        compile_all(specs, provider, options.intra_threads)
    })?;
    Span::current().record("execution_provider", field::display(execution_provider));
    Ok(LoadedSessions {
        sessions,
        execution_provider,
    })
}

/// Providers to retry, in order, after `execution_provider` fails to compile. CPU is the final
/// attempt; its failure is returned without another fallback. DirectML tries OpenVINO first, while
/// other non-CPU providers fall straight to CPU.
pub(crate) fn fallback_chain(
    execution_provider: ExecutionProvider,
) -> &'static [ExecutionProvider] {
    match execution_provider {
        ExecutionProvider::Directml => &[ExecutionProvider::OpenVino, ExecutionProvider::Cpu],
        ExecutionProvider::OpenVino | ExecutionProvider::Webgpu | ExecutionProvider::Cpu => {
            &[ExecutionProvider::Cpu]
        }
        ExecutionProvider::CoreML => &[ExecutionProvider::Cpu],
    }
}

fn compile_all<const N: usize>(
    specs: [SessionSpec<'_>; N],
    execution_provider: ExecutionProvider,
    intra_threads: NonZeroUsize,
) -> Result<[Session; N]> {
    let mut sessions = Vec::with_capacity(N);
    for spec in specs {
        sessions.push(
            compile_session(spec.path, execution_provider, intra_threads).with_context(|| {
                format!(
                    "loading the {} model from {}",
                    spec.label,
                    spec.path.display()
                )
            })?,
        );
    }
    Ok(sessions
        .try_into()
        .unwrap_or_else(|_| unreachable!("one session is pushed per spec")))
}

/// A provider ready to register, with the ONNX Runtime intra-op thread count that suits it.
pub(crate) struct ConfiguredProvider {
    pub(crate) dispatch: ExecutionProviderDispatch,
    /// Size of ONNX Runtime's *own* intra-op pool. OpenVINO takes the requested thread count for
    /// a pool of its own, so ORT's is left at one rather than doubling up on cores.
    pub(crate) intra_threads: NonZeroUsize,
}

/// Build a provider dispatch after checking the process-wide runtime library.
pub(crate) fn configure_provider(
    execution_provider: ExecutionProvider,
    intra_threads: NonZeroUsize,
) -> Result<ConfiguredProvider> {
    match execution_provider {
        ExecutionProvider::Cpu => cpu_provider(intra_threads),

        #[cfg(feature = "ort-openvino")]
        ExecutionProvider::OpenVino => openvino_provider(intra_threads),
        #[cfg(not(feature = "ort-openvino"))]
        ExecutionProvider::OpenVino => provider_unavailable(execution_provider, "ort-openvino"),

        #[cfg(feature = "ort-directml")]
        ExecutionProvider::Directml => directml_provider(intra_threads),
        #[cfg(not(feature = "ort-directml"))]
        ExecutionProvider::Directml => provider_unavailable(execution_provider, "ort-directml"),

        #[cfg(feature = "ort-webgpu")]
        ExecutionProvider::Webgpu => Ok(ConfiguredProvider {
            dispatch: ort::ep::WebGPU::default().build().error_on_failure(),
            intra_threads,
        }),
        #[cfg(not(feature = "ort-webgpu"))]
        ExecutionProvider::Webgpu => provider_unavailable(execution_provider, "ort-webgpu"),

        #[cfg(feature = "ort-coreml")]
        ExecutionProvider::CoreML => coreml_provider(intra_threads),
        #[cfg(not(feature = "ort-coreml"))]
        ExecutionProvider::CoreML => provider_unavailable(execution_provider, "ort-coreml"),
    }
}

#[instrument(
    name = "onnx_compile",
    skip_all,
    fields(model_path = %path.display(), execution_provider = %execution_provider)
)]
fn compile_session(
    path: &Path,
    execution_provider: ExecutionProvider,
    intra_threads: NonZeroUsize,
) -> Result<Session> {
    let provider = configure_provider(execution_provider, intra_threads)?;

    let mut builder = Session::builder()?
        .with_intra_threads(provider.intra_threads.get())
        .map_err(builder_error)?
        .with_parallel_execution(false)
        .map_err(builder_error)?
        // WebGPU plugin 0.3.0 cannot initialize ORT 1.30's fused OCR activations.
        // Basic optimization keeps the supported Conv and activation kernels separate.
        .with_optimization_level(if execution_provider == ExecutionProvider::Webgpu {
            GraphOptimizationLevel::Level1
        } else {
            GraphOptimizationLevel::Level3
        })
        .map_err(builder_error)?;
    if std::env::var_os("NICEGAL_ORT_TRACE").is_some() {
        builder = builder
            .with_log_level(ort::logging::LogLevel::Verbose)
            .map_err(builder_error)?
            .with_log_verbosity(1)
            .map_err(builder_error)?;
    }
    let mut builder = if execution_provider == ExecutionProvider::Webgpu {
        let environment = ort::environment::Environment::current()?;
        let device = environment
            .devices()
            .find(|device| {
                device
                    .ep()
                    .is_ok_and(|name| name == "WebGpuExecutionProvider")
            })
            .context("No native WebGPU device is available")?;
        builder
            .with_devices([device], None)
            .map_err(builder_error)?
    } else {
        builder
            .with_execution_providers([provider.dispatch])
            .map_err(builder_error)?
    };
    if provider.intra_threads.get() == 1 {
        // A one-thread ORT pool means the provider below owns the compute; spinning would only
        // take cores from the pool doing the work.
        builder = builder
            .with_intra_op_spinning(false)
            .map_err(builder_error)?;
    }

    builder.commit_from_file(path).with_context(|| {
        format!("compiling the ONNX graph for the {execution_provider} execution provider")
    })
}

fn cpu_provider(intra_threads: NonZeroUsize) -> Result<ConfiguredProvider> {
    Ok(ConfiguredProvider {
        dispatch: CPU::default()
            .with_arena_allocator(true)
            .build()
            .error_on_failure(),
        intra_threads,
    })
}

#[cfg(feature = "ort-openvino")]
fn openvino_provider(intra_threads: NonZeroUsize) -> Result<ConfiguredProvider> {
    use ort::ep::OpenVINO;

    let provider = OpenVINO::default()
        .with_num_threads(intra_threads.get())
        .with_dynamic_shapes(true);
    ensure_available(&provider, ExecutionProvider::OpenVino)?;
    Ok(ConfiguredProvider {
        dispatch: provider.build().error_on_failure(),
        intra_threads: NonZeroUsize::MIN,
    })
}

#[cfg(feature = "ort-directml")]
fn directml_provider(intra_threads: NonZeroUsize) -> Result<ConfiguredProvider> {
    use ort::ep::DirectML;

    let provider = DirectML::default();
    ensure_available(&provider, ExecutionProvider::Directml)?;
    Ok(ConfiguredProvider {
        dispatch: provider.build().error_on_failure(),
        // Unlike OpenVINO, DirectML has no thread-pool knob of its own: it offloads to the GPU,
        // and any node it can't cover falls back to running on ORT's own CPU pool. So the
        // requested thread count is kept rather than collapsed to one.
        intra_threads,
    })
}

#[cfg(feature = "ort-coreml")]
fn coreml_provider(intra_threads: NonZeroUsize) -> Result<ConfiguredProvider> {
    use ort::ep::CoreML;

    let provider = CoreML::default();
    ensure_available(&provider, ExecutionProvider::CoreML)?;
    Ok(ConfiguredProvider {
        dispatch: provider.build().error_on_failure(),
        intra_threads,
    })
}

/// Refuse a provider whose `ort` feature was not compiled into this build.
#[cfg_attr(
    any(
        feature = "ort-openvino",
        feature = "ort-directml",
        feature = "ort-coreml"
    ),
    allow(dead_code)
)]
fn provider_unavailable(
    execution_provider: ExecutionProvider,
    feature: &str,
) -> Result<ConfiguredProvider> {
    bail!("the {execution_provider} execution provider requires the `{feature}` feature")
}

/// Refuse a provider the loaded ONNX Runtime library was not built with.
///
/// An `ort` provider feature only compiles in the Rust side of registration; whether the provider
/// exists belongs to the library loaded at runtime. Unchecked, such a session commits silently on
/// CPU and everything measured from it is mislabelled.
#[cfg(any(
    feature = "ort-openvino",
    feature = "ort-directml",
    feature = "ort-coreml"
))]
fn ensure_available(
    provider: &impl ort::ep::ExecutionProvider,
    name: ExecutionProvider,
) -> Result<()> {
    let available = provider
        .is_available()
        .map_err(|error| anyhow!("{error}"))
        .with_context(|| format!("checking {name} execution provider availability"))?;
    if !available {
        bail!("the loaded ONNX Runtime library was not built with the {name} execution provider");
    }
    Ok(())
}

/// `ort`'s builder errors hand the builder back for recovery, which makes them neither `Send` nor
/// `Sync`, so they do not convert into an `anyhow::Error` on their own.
fn builder_error<R>(error: ort::Error<R>) -> anyhow::Error {
    anyhow!("{error}")
}

#[cfg(test)]
mod tests {
    use super::{ExecutionProvider, RuntimeOptions, fallback_chain};
    use std::{num::NonZeroUsize, str::FromStr};

    #[test]
    fn providers_use_their_declared_fallback_chains() {
        assert_eq!(
            fallback_chain(ExecutionProvider::Directml),
            [ExecutionProvider::OpenVino, ExecutionProvider::Cpu]
        );
        assert_eq!(
            fallback_chain(ExecutionProvider::OpenVino),
            [ExecutionProvider::Cpu]
        );
        assert_eq!(
            fallback_chain(ExecutionProvider::CoreML),
            [ExecutionProvider::Cpu]
        );
    }

    #[test]
    fn execution_provider_parses_and_displays_canonical_names() {
        for (value, expected) in [
            ("cpu", ExecutionProvider::Cpu),
            ("openvino", ExecutionProvider::OpenVino),
            ("directml", ExecutionProvider::Directml),
            ("webgpu", ExecutionProvider::Webgpu),
            ("coreml", ExecutionProvider::CoreML),
        ] {
            assert_eq!(ExecutionProvider::from_str(value).unwrap(), expected);
            assert_eq!(expected.to_string(), value);
        }
        assert!(ExecutionProvider::from_str("CPU").is_err());
    }

    #[test]
    fn runtime_options_default_to_cpu_with_four_threads() {
        assert_eq!(
            RuntimeOptions::default(),
            RuntimeOptions {
                execution_provider: ExecutionProvider::Cpu,
                intra_threads: NonZeroUsize::new(4).unwrap(),
                replicas: None,
                allow_cpu_fallback: true,
            }
        );
    }

    #[test]
    fn only_non_cpu_provider_failures_can_fallback() {
        assert!(!RuntimeOptions::default().can_fall_back_to_cpu());
        assert!(
            RuntimeOptions {
                execution_provider: ExecutionProvider::OpenVino,
                ..RuntimeOptions::default()
            }
            .can_fall_back_to_cpu()
        );
        assert!(
            !RuntimeOptions {
                execution_provider: ExecutionProvider::Directml,
                allow_cpu_fallback: false,
                ..RuntimeOptions::default()
            }
            .can_fall_back_to_cpu()
        );
    }
}
