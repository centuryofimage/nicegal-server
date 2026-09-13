# nicegal-server

A Rust library and local HTTP server for a high-performance searchable image gallery.

The workspace contains the `nicegal-core` library, `nicegal-server` HTTP server, and
`nicegal-cli` search command. Environment variables use the `NICEGAL_` prefix.
Standalone tools default to the local data directory's `nicegal-server` folder.

Development documentation: [internal HTTP API](INTERNAL_API.md).

## License

The original application code in this repository is licensed under
[PolyForm Internal Use 1.0.0](LICENSE). The modified third-party code in `vendor/`
remains licensed under Apache-2.0, with its license files and attribution notices
retained there. These application terms do not replace the licenses applicable
to separately identified components. Model weights downloaded during operation
are subject to their respective publishers' licenses.

## PaddleOCR runtime

The OCR runtime uses PaddleOCR PP-OCRv6 ONNX models through `ort` 2.0.0-rc.13. Model repository IDs are supplied by the API caller, downloaded through `hf-hub` into the standard Hugging Face cache, and compiled for ONNX Runtime. The server exposes download and compilation progress through its existing job API.

CPU inference is always available and uses four intra-op threads by default rather than ONNX
Runtime's machine-wide default. The Windows server ships both accelerated provider registrations
and two separate `onnxruntime` distributions:

```text
build-server.cmd   # Windows: DirectML and OpenVINO distributions
./build-server.sh  # Linux x64: OpenVINO distribution, including CPU
```

`onnxruntime-directml` and `onnxruntime-openvino` each need their own `onnxruntime.dll`, since one
runtime build only ever has one non-CPU provider. `build-server.cmd` creates both isolated `uv`
venvs when needed:

```text
uv venv --python 3.13 .venv-directml
uv pip install --python .venv-directml -r requirements-directml.txt

uv venv --python 3.13 .venv-openvino
uv pip install --python .venv-openvino -r requirements-openvino.txt
```

`build.rs` copies the DLL sets beside the compiled server as
`onnxruntime/directml/` and `onnxruntime/openvino/`; the latter includes OpenVINO's own DLLs. The
server resolves one of those directories from its executable path before any ONNX model loads, so
the Electron deployment must preserve both folders. To replace a wheel distribution, set
`NICEGAL_DIRECTML_ORT_LIB_PATH` or `NICEGAL_OPENVINO_ORT_LIB_PATH`; set
`NICEGAL_OPENVINO_LIB_PATH` to replace the OpenVINO support DLL directory.

On Windows, `build.rs` also copies the Visual C++ runtime DLLs beside the server and test
executables so deployment does not require a separate Visual C++ Redistributable installation.
It uses `VCToolsRedistDir` when set, otherwise locates Visual Studio with `vswhere` and reads its
default redistributable version. Set `NICEGAL_CRT_LIB_PATH` to the target architecture's
`Microsoft.VC143.CRT` directory to override discovery (for example, in CI). Distribute these
DLLs beside `nicegal-server.exe`, preserve Microsoft's redistribution terms, and refresh them
when updating the bundled runtime. The Electron packaging filter already includes these DLLs.

`GET /v1/runtime` reports the provider active in this process, the actual ONNX Runtime DLL
distribution selected for it, its build metadata, and the persisted selection for the next launch.
`GET /v1/status` additionally reports whether OCR models are loaded. `PUT /v1/runtime` with
`{"executionProvider":"cpu"|"directml"|"openvino"}` changes the persisted selection;
`restartRequired` is true until the server restarts with it. The command-line
`--execution-provider` / `NICEGAL_EXECUTION_PROVIDER` override still applies to that launch
only.

On Linux x64, `build-server.sh` provisions `.venv-openvino` with uv and Python 3.13.
The Linux wheel includes OpenVINO's native libraries, so a separate OpenVINO Python
package is unnecessary. The script stages the shared libraries under
`target/{debug,release}/onnxruntime/openvino/` (and each profile's `deps/` for tests).
Distribute `nicegal-server` with that `onnxruntime/openvino/` directory intact;
Python is not needed at runtime. Linux defaults to OpenVINO and can fall back to CPU.

Ubuntu 24.04 build prerequisites are Rust 1.98.0, uv, `build-essential`,
`pkg-config`, `libssl-dev`, `libclang-dev`, `cmake`, and `nasm`. After running
`./build-server.sh`, verify both providers with a small local ONNX model:

```bash
./dev.sh test -p nicegal-core --features ort-openvino --test linux_runtime -- --ignored
```

This test disables fallback to prove both OpenVINO and CPU can execute inference.

`RuntimeOptions` selects the provider and thread count. Production callers may allow a failed
accelerator registration or model compilation to rebuild both OCR sessions on CPU. Callers that
need a definitive provider result, including the benchmark below, disable that fallback.

After loading a detector/recognizer pair, an `ocrIndex` job scans and catalogs a root, decodes images through a bounded preprocessing queue, runs PaddleOCR detection and batched recognition, commits OCR rows in small durable chunks, then incrementally embeds pending OCR text by default. Set its `embed` parameter to `false` to stop after OCR; the standalone text embedding job remains available for explicit backfills such as a model change. Thumbnail generation remains separate and lazy.

### Index pipeline design

Discovery and cataloging finish before derived OCR work starts, so slow decoding or inference
cannot delay otherwise usable gallery entries. `ort` sessions require
mutable access and ONNX Runtime already owns its inference threads, so cloning model pairs across
workers would duplicate memory and oversubscribe the CPU. Instead, up to four workers perform only
image decoding and feed a channel bounded to two decoded images per worker. One owner performs
detection and recognition, recognition crops are grouped by aspect ratio and batched, and one
writer commits every 32 images. Decode completion is intentionally out of scan order so a slow
JPEG does not leave inference idle. The queue bounds full-resolution memory independently of catalog discovery.

### Index benchmark

`ocr_index` exercises the real catalog, bounded decode, detection, recognition, and SQLite write
pipeline. Every measured run uses fresh temporary databases while reusing the loaded and warmed
model sessions. Model-load and warmup durations are reported separately from indexing throughput.
The benchmark refuses CPU fallback so a missing provider DLL cannot produce a misleading
accelerator result.

```text
dev.cmd bench --bench ocr_index -- \
  --corpus C:/gallery \
  --detection-model C:/models/det/inference.onnx \
  --detection-config C:/models/det/inference.yml \
  --recognition-model C:/models/rec/inference.onnx \
  --recognition-config C:/models/rec/inference.yml \
  --provider cpu \
  --threads 4 \
  --warmup-runs 1 \
  --runs 3
```

Build the matching feature when selecting a non-CPU provider, for example:

```text
dev.cmd bench --bench ocr_index --features ort-directml -- \
  <model and corpus arguments> --provider directml --threads 4
```

Keep the corpus, models, warmup count, and run count fixed when sweeping provider and thread
settings. The benchmark prints OS, architecture, logical parallelism, requested/configured
provider, paths, and setup durations followed by stable CSV rows. Compare the per-run
`images_per_second` values rather than model-load time; retain every row so variance and thermal
throttling remain visible.

## CLI usage

The CLI currently searches an existing OCR database:

```text
Usage: nicegal-cli [OPTIONS] <QUERIES>...

Arguments:
  <QUERIES>...  Strings to search for

Options:
  -d, --database <FILE>     Location of the OCR index database
  -x, --exclude <PATTERN>   Exclude indexed paths matching this glob
  -l, --limit <LIMIT>       Maximum number of results [default: 100]
  -s, --search-type <TYPE>  simple, match, glob, or regex [default: simple]
      --pwd <PWD>           Set working directory (hidden integration option)
  -h, --help                Print help
  -V, --version             Print version
```

## Building

On Windows x64, install Rust 1.98.0 (MSVC), Visual Studio C++ build tools, and uv.
Clone this repository and build the server with its runtime libraries:

```text
git clone https://github.com/centuryofimage/nicegal-server.git
cd nicegal-server
build-server.cmd
dev.cmd build --workspace --release --locked
```

On Windows, use `build-server.cmd` for a deployable server; it creates both provider venvs and
packages their DLLs as described above. No system OCR library or language-data installation is
required.

## OCR API

Start the server with `NICEGAL_RPC_TOKEN` set, create an `ocrModelLoad` job with explicit detector and recognizer repository IDs, then create an `ocrIndex` job for an absolute gallery root. Index creation is rejected until models are loaded. See the [Jobs section](INTERNAL_API.md#jobs) for requests, progress fields, scheduling, and cancellation behavior.

## Tests and benchmarks

From the backend workspace, run `dev.cmd test --locked --workspace` and
`dev.cmd clippy --locked --workspace --all-targets` after provisioning runtimes
with `build-server.cmd`. Benchmark corpora are supplied by the caller and are
not included in this repository.

- [OCR batching and measurements](BATCHING.md)
- [CLIP image indexing](benches/CLIP_INDEX.md)
- [Text embedding benchmark procedure](benches/EMBEDDING_INDEX_PERFORMANCE.md)
- [Vector search measurements](benches/SEARCH_REVIEW.md)
- [ONNX Runtime integration](ORT_INFO.md)

## Verify downloads

Verify a downloaded server ZIP with GitHub CLI:

```powershell
gh attestation verify PATH_TO_DOWNLOADED_ZIP --repo centuryofimage/nicegal-server
```

The build-provenance attestation identifies the repository, source commit, and
GitHub Actions workflow that produced the archive. It is separate from Windows
Authenticode signing.
