# Embedding Index Benchmark Procedure

This procedure measures the OCR-text embedding backfill independently from OCR indexing. It covers throughput, resource use, batching, incremental behavior, cancellation, storage growth, and basic vector-quality attributes.

## Current contract

- Model: `BAAI/bge-small-en-v1.5` through FastEmbed/ONNX Runtime.
- Vector width: 384 normalized `f32` values.
- Execution provider: synced with whatever OCR is using (`GET /v1/runtime`'s `activeExecutionProvider`) — DirectML by default on Windows, with the same OpenVINO-then-CPU fallback chain OCR uses if DirectML can't load. See "GPU acceleration" below for why this default was chosen and its limits.
- Default/effective maximum batch: 64 rows. The API accepts `batchSize` from 1 through 512, but values above 64 are currently clamped to 64 — see "GPU acceleration" for why.
- Input cap: 8192 UTF-8 bytes per OCR row, truncated on a character boundary.
- Only rows under the requested absolute `root` are considered.
- `force: false` embeds only missing or stale rows. `force: true` clears current OCR-text vectors and rebuilds them.
- One resource-intensive job may run at a time. Embedding inference runs without holding a SQLite write transaction; vectors are saved after each batch.

Use copied databases for benchmarking. A forced run intentionally replaces vectors in the OCR database.

## Prepare a representative corpus

Start with an OCR database that has already been indexed. Record:

- total OCR rows and total OCR text bytes;
- text-length percentiles, especially rows near the 8192-byte cap;
- language/content mix;
- database location and storage type;
- CPU, logical-core count, RAM, Windows version, build commit, and power plan.

Use at least two corpus shapes when possible:

1. **Typical gallery:** the expected OCR length and language distribution.
2. **Long-text stress:** many rows near the input cap, which exposes tokenization and padding cost.

Do not include model download time in indexing throughput. Measure cold and warm startup separately instead.

## Launch the benchmark server

Use a release build and explicit copied databases:

```bat
set NICEGAL_RPC_TOKEN=embedding-benchmark

dev.cmd run --release -p nicegal-server -- ^
  --asset-database C:\bench\assets.db ^
  --ocr-database C:\bench\ocr.db ^
  --thumbnail-database C:\bench\thumbnails.db
```

The server prints one readiness object containing its ephemeral endpoint. Use that endpoint below as `%ENDPOINT%`.

For a non-destructive cold-cache startup measurement, point `HF_HOME` at a new empty benchmark directory before launching. For warm startup, restart with the same directory. Record process-start-to-readiness time for both cases. Do not delete the user's normal Hugging Face cache.

## Run an embedding backfill

Check the initial coverage:

```bat
curl.exe -sS --get "%ENDPOINT%/v1/text-embeddings" ^
  --data-urlencode "root=C:/gallery" ^
  -H "Authorization: Bearer embedding-benchmark"
```

Start a full rebuild:

```bat
curl.exe -sS -X POST "%ENDPOINT%/v1/text-embeddings/generate" ^
  -H "Authorization: Bearer embedding-benchmark" ^
  -H "Content-Type: application/json" ^
  --data-binary "{\"root\":\"C:/gallery\",\"force\":true,\"batchSize\":256}"
```

Record the returned `jobId`. Poll until terminal:

```bat
curl.exe -sS "%ENDPOINT%/v1/jobs/JOB_ID" ^
  -H "Authorization: Bearer embedding-benchmark"
```

A valid completed run has:

- `status: "completed"` and `phase: "finished"`;
- `progress.failed == 0` and an empty `errors` array;
- `progress.processed == progress.embedded + progress.skipped`;
- final coverage with `pending == 0` and `indexed == embedded` for a stable corpus.

Calculate sustained throughput as:

```text
embedded rows / wall seconds from the 202 response to the terminal snapshot
embedded UTF-8 bytes after the 8192-byte per-row cap / the same interval
```

Exclude server startup and model loading. Run each configuration at least three times on fresh copies of the same pre-embedding OCR database; report the median and range. The first inference may include additional ONNX initialization, so keep it as a separately reported cold-job result rather than silently discarding it.

## Batch-size sweep

Benchmark `batchSize` values `1, 8, 32, 64, 128, 256, 512` using a fresh database copy or `force: true` before every run. Do not sweep past 512 without reading "GPU acceleration" below first — 1024 is a known CPU-arena-OOM trigger.

For each size, record:

| Batch | Rows/s | MiB text/s | Elapsed | Peak working set | CPU utilization | Cancel latency | Failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | | | |
| 8 | | | | | | | |
| 32 | | | | | | | |
| 64 | | | | | | | |
| 128 | | | | | | | |
| 256 | | | | | | | |
| 512 | | | | | | | |

Watch `nicegal-server` in Windows Performance Monitor or Task Manager. Capture peak working set, private bytes, total CPU utilization, thread count, and disk write throughput. Sample frequently enough to catch per-batch peaks.

Choose the smallest batch whose throughput is close to the plateau. Larger batches increase memory use and cancellation latency because cancellation is observed between batches, not during an ONNX inference call.

## GPU acceleration

`src/runtime.rs`'s DirectML/OpenVINO/CPU provider selection also applies to the embedder, since `fastembed` 6.0.0 pins the exact `ort` version this crate does and accepts the same `ExecutionProviderDispatch`. `benches/text_embed_index.rs` (`dev.cmd bench --bench text_embed_index -- --provider <cpu|directml|openvino>`) measures `TextEmbedder::embed_documents` in isolation on a synthetic OCR-shaped corpus (mostly short lines, one row in 32 padded out near the 8192-byte cap), independent of the server/HTTP path this document otherwise benchmarks.

One reference run (BGE-small-en-v1.5, Windows, DirectML vs. CPU, 2 runs per batch after 1 warmup run):

| Batch | CPU rows/s | DirectML rows/s | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | ~12.3 | ~128 | ~10x |
| 8 | ~11.4 | ~138 | ~12x |
| 32 | ~6.0 (noisy) | ~236 | ~40x |
| 64 | ~8.7 | ~252 | ~29x |
| 128 | ~9.4 | ~249 | ~26x |
| 256 | ~9.5 | **DirectML crashed** | — |
| 1024 | **CPU crashed** | not attempted | — |

Two failures, both real and both load-bearing for the batch-size default:

- **CPU, batch 1024:** the ONNX arena tried to allocate one ~12 GiB buffer and failed. Attention cost is `batch * heads * seq^2 * 4 bytes`; at batch 1024 with sequences padded to the model's 512-token max that is `1024 * 12 * 512^2 * 4` = exactly the requested size in the error. Every row in a batch pads to the *longest* row in that batch, so this triggers from a single near-cap OCR row landing in an otherwise-short batch, not from the whole batch being long.
- **DirectML, batch 256:** a hard device-level error (`DmlCommandRecorder.cpp`: "The GPU will not respond to more commands, most likely because of an invalid command"), not a graceful allocation failure. This is a DirectML execution-provider limit, not a memory-budget one — it showed up at a quarter of the CPU failure's batch size, and throughput had already plateaued by batch 32-64 on both providers, so there is no throughput reason to run anywhere near it.

Given that, `TextEmbedderOptions::default().max_batch_size` is 64: comfortably below both cliffs, at the measured plateau, and small enough (a few hundred MiB of attention activations at most) to stay reasonable on the 16 GB machines this also has to run on. Re-run the sweep above (and re-check the two failure points) after any FastEmbed, `ort`, or DirectML runtime upgrade before trusting this table — both failure modes are specific to this stack's current versions.

## Behavioral and operational checks

1. **Incremental no-op:** rerun with `force: false`. It should discover zero pending rows and finish quickly without changing stored coverage.
2. **Stale-row repair:** re-index a small changed subset, confirm `pending` increases by that subset, then run `force: false` and confirm only those rows are embedded.
3. **Cancellation:** start a forced run, send `DELETE /v1/jobs/JOB_ID`, and record time until `cancelled`. Confirm completed batches remain stored and a subsequent non-forced run finishes the backlog.
4. **Concurrent search:** issue a fixed vector query repeatedly during backfill. Record p50/p95/p99 latency and errors before and during the job. Search should remain available because inference does not hold the SQLite write lock.
5. **Job exclusion:** while embedding is active, starting another resource-intensive job should return `409 job_busy`.
6. **Storage growth:** record the combined size of `ocr.db`, `ocr.db-wal`, and `ocr.db-shm` before and after. Measure again after a controlled WAL checkpoint on the disposable copy if compact database size matters.
7. **Restart durability:** stop and restart the server after completion. Coverage should still report the same model, dimensions, and embedded count.

## Vector attributes and retrieval sanity

Performance numbers are not sufficient if the vectors are malformed or unhelpful. Verify:

- every stored vector has 384 finite values;
- sampled vector L2 norms are approximately 1.0;
- repeated embedding of unchanged text is stable within a small floating-point tolerance;
- different representative texts do not produce identical vectors;
- a small labeled query set returns semantically relevant OCR rows near the top;
- non-English and very long OCR samples are called out separately, because the current model is English and long rows are truncated.

For the labeled query set, report Recall@1, Recall@5, and Recall@10, plus several failure examples. Keep this set fixed between model or runtime changes so quality regressions are visible alongside throughput changes.

## Result header

Attach this metadata to every result set:

```text
Commit:
Build profile: release
Model: BAAI/bge-small-en-v1.5
ONNX Runtime/FastEmbed versions:
CPU / logical cores / RAM:
Storage:
Windows version / power plan:
HF cache: cold | warm
Corpus rows / OCR bytes / length percentiles:
Root filter:
Batch size:
Force: true | false
Elapsed / rows per second / MiB per second:
Peak working set / CPU / disk write rate:
Failures / cancellation latency / search p95:
Database+WAL size before and after:
Notes:
```

Retain the raw job JSON, coverage JSON, resource-monitor export, and labeled-query results with the summary. Those artifacts make later comparisons auditable.
