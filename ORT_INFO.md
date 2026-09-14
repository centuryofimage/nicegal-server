# ONNX Runtime integration notes

## Execution providers

- CPU is always available.
- `ort-directml` (Windows, DirectX 12 GPUs) and `ort-openvino` (Intel CPUs/iGPUs/VPUs) are both
  compiled into the Windows server.
- `onnxruntime-directml` and `onnxruntime-openvino` each ship an `onnxruntime.dll` with only that
  non-CPU provider compiled in. They cannot be merged: the server packages and dynamically loads
  one complete distribution per process. This means `runtime::fallback_chain`'s `directml` →
  `openvino` rung can never actually succeed when the process loaded the `directml` distribution —
  it always lands on `cpu`. `ocr_models::job::run` detects exactly that outcome and restarts the
  process into `openvino` instead of running the rest of the session on CPU; see `RESTART_EXIT_CODE`
  in `api/mod.rs` and INTERNAL_API.md.

## Runtime setup

- Default Linux `ort-webgpu` uses ONNX Runtime 1.30.0 and native WebGPU plugin
  0.3.0, pinned in `requirements-webgpu.txt`. Build with
  `NICEGAL_LINUX_PROVIDERS=webgpu ./build-server.sh`. Only the native shared
  libraries are packaged; Python is a build tool. Startup registers the plugin
  before sessions are created. Both the core and vendored FastEmbed register its
  discovered device through `SessionBuilder::with_devices`; the older built-in
  WebGPU registration API does not select plugin devices. CPU fallback uses the
  same ORT distribution. These runtime versions do not affect Windows providers.
  PaddleOCR sessions use BASIC graph optimization: ORT 1.30's fused convolution
  activations fail to initialize with plugin 0.3.0, while separate kernels pass
  the native parity tests. FastEmbed's embedding sessions retain their existing
  optimization level.

- `build-server.cmd` creates the `.venv-directml` and `.venv-openvino` environments from their
  corresponding requirements files, skipping each once installed.
- `build.rs` copies both distributions to `onnxruntime/directml/` and `onnxruntime/openvino/` next
  to the executable (and the test executable directory). The OpenVINO directory also contains its
  plugin and support DLLs. `NICEGAL_DIRECTML_ORT_LIB_PATH`,
  `NICEGAL_OPENVINO_ORT_LIB_PATH`, and `NICEGAL_OPENVINO_LIB_PATH` override their respective
  source directories.
- The server resolves the selected distribution from `current_exe()` and calls
  `ort::init_from` before FastEmbed or PaddleOCR can create a session. `load-dynamic` then fixes
  that `onnxruntime` library for the process lifetime.
- `GET /v1/runtime` returns `activeExecutionProvider`, `activeRuntimeDistribution`,
  `onnxRuntimeBuildInfo`, `configuredExecutionProvider`, and `restartRequired`.
  `onnxRuntimeBuildInfo` comes from `ort::info()` and identifies the loaded runtime's build.
  `PUT /v1/runtime` persists an `executionProvider` for the next launch; it never attempts to
  replace the loaded DLL or live sessions.

## Important gotchas

- Opt-in `ort-profiling` requires ORT API 25 or newer. With this feature,
  `NICEGAL_ORT_PROFILE_DIR` records ONNX profiles for the first three image batches
  (including warmup), using `RunOptions::enable_profiling`. Leave both unset for
  throughput measurements. This feature is not enabled in the standard Windows
  build, whose older runtimes retain the existing API 24 requirement.

- With `load-dynamic`, `ort-sys/disable-linking` means upstream `copy-dylibs` does not run.
  `build.rs` copies the two distributions itself, and startup selects an absolute DLL path.
- The OpenVINO wheel has no `onnxruntime.lib`; compile-time dynamic linking would need generating
  or obtaining that MSVC import library.
- The `onnxruntime-openvino` wheel does not bundle OpenVINO itself — its provider DLL imports
  `openvino.dll`, so the separate `openvino` PyPI package (and its OpenVINO/TBB/plugin DLLs) is
  required alongside it.
- Loading `onnxruntime.dll` by absolute path is not sufficient when its dependencies are outside
  the process's DLL search locations. Startup prepends the selected distribution directory before
  loading it, and all related DLLs are kept within that directory.
- Runtime selection is process-global: `ort::init_from`/first session use must happen before either
  PaddleOCR or FastEmbed creates a session.
- Production fallback (`RuntimeOptions::allow_cpu_fallback`) rebuilds both OCR sessions on CPU and
  reports CPU as configured; the benchmark disables it so a setup failure fails the run instead of
  silently producing a mislabeled CPU result.
- OpenVINO uses its own requested thread count with ORT's own intra-op pool limited to one (and
  spinning disabled) so the two pools don't compete for cores. DirectML has no thread-pool knob of
  its own, so ORT's pool keeps the requested thread count for whatever falls back to CPU. CPU
  itself defaults to four intra-op threads and sequential execution.
- Provider selection, session compilation, and the fallback live in `src/runtime.rs`; Hugging Face
  file resolution in `src/hub.rs`. Both are shared by every model family, and `RuntimeOptions` is
  passed per load so families need not agree on a thread budget.
- FastEmbed (the embedder) shares this the same way OCR does: `fastembed` 6.0.0 pins the exact
  `ort` version this crate does, so `runtime::configure_provider`'s `ExecutionProviderDispatch` is
  handed straight into `TextInitOptions::with_execution_providers`, and `runtime::with_fallback`
  gives it the same request/CPU-fallback contract as `load_sessions` without going through
  `compile_session` (FastEmbed builds its own session internally). See
  `benches/EMBEDDING_INDEX_PERFORMANCE.md`'s "GPU acceleration" section for the batch-size ceiling this
  forced: DirectML fails hard (a device-level error, not a catchable allocation failure) at a
  batch size well below where CPU's arena OOMs, and below where either provider's throughput was
  still improving.
- A provider's `ort` Cargo feature only compiles in its Rust registration code; whether the
  provider actually exists belongs to the loaded runtime. Both are checked before a session is
  committed, so a build with the feature but not the library fails instead of silently running on
  CPU.

## Platform support

Windows packages include DirectML and OpenVINO. Linux x64 defaults to WebGPU
with CPU fallback and requires a working Vulkan driver/loader. An OpenVINO build
can be selected with `NICEGAL_LINUX_PROVIDERS=openvino`. The later September 14
user direction removes the untested CUDA/MIGraphX paths, superseding their
earlier inclusion during the Linux branch merge. Saved unavailable provider
selections migrate to the current build's default. Explicit unavailable CLI/API
choices are rejected. macOS runtime packaging is not supported.
