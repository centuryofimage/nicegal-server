# CLIP Image Index Benchmarks

This document records the CLIP image-index benchmark runs. The benchmark measures the complete
catalog-backed indexing path:

1. catalog lookup and source selection;
2. parallel full-resolution image decode;
3. FastEmbed-compatible CPU preprocessing and ONNX Runtime image inference; and
4. SQLite-vec vector ingestion.

The benchmark does not include model download time in the reported indexing elapsed time. Model
load time is printed separately. The vendored FastEmbed API separates model-configured
preprocessing from inference: `preprocess_image` spans cover the CPU resize/crop/normalize step,
while `embed_preprocessed_images` covers tensor batching and the ONNX Runtime call.

## Result summary

The small corpus is useful for provider and batch comparisons. The camera corpus is more
representative of the intended workload because it contains 12 MP phone photographs.

| Corpus | Provider | Best tested batch | Images | Median images/s | Median elapsed | Peak working set |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Small gallery | DirectML | 16 | 106 | **130.3** | 0.814 s | 456 MiB |
| Small gallery | OpenVINO | 16 | 106 | 46.2 | 2.293 s | 1.69 GiB |
| Phone camera | DirectML | 16 | 146 | **12.3** | 11.877 s | 1.58 GiB |

The camera run indexed 146 images from 153 cataloged assets; the remaining seven assets were
unsupported videos.

## Small gallery: provider and batch sweep

Corpus: a small gallery corpus, 106 supported images, CLIP
`Qdrant/clip-ViT-B-32`, 512 dimensions, four runtime threads, three measured runs after one
warmup run. Working set values are process peak working set, sampled at the end of each run from
Windows `GetProcessMemoryInfo`.

| Provider | Batch | Median images/s | Run range images/s | Median elapsed | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: |
| DirectML | 1 | 100.6 | 80.7–104.9 | 1.054 s | 459 MiB |
| DirectML | 4 | 39.0 | 33.7–39.4 | 2.719 s | 474 MiB |
| DirectML | 8 | 49.5 | 43.7–50.3 | 2.141 s | 456 MiB |
| DirectML | 16 | **130.3** | 124.7–132.5 | **0.814 s** | 456 MiB |
| OpenVINO | 1 | 36.8 | 36.8–36.8 | 2.881 s | 1.69 GiB |
| OpenVINO | 4 | 43.4 | 42.3–44.3 | 2.444 s | 1.69 GiB |
| OpenVINO | 8 | 45.0 | 44.7–45.2 | 2.354 s | 1.69 GiB |
| OpenVINO | 16 | **46.2** | 45.5–46.3 | **2.293 s** | 1.69 GiB |

### Small-gallery trace breakdown

The detailed DirectML batch-8 trace shows that the indexing stages are already instrumented:

| Stage | Measurement |
| --- | ---: |
| `decode_image`, median per image | 1.77 ms |
| `decode_image`, maximum observed | 13.2 ms |
| `embed_images`, median full batch of eight | 160 ms |
| `save_image_embeddings`, median full batch | 0.30 ms |
| `image_index`, complete run | 2.14 s |

These baseline `embed_images` measurements include FastEmbed preprocessing and model inference.
The current pipeline emits separate `preprocess_image` and `embed_preprocessed_images` spans,
while retaining the per-image decode spans and per-batch SQLite write spans at `debug` level.

## Phone camera: large-image batch comparison

Corpus: a phone-camera corpus, 153 cataloged assets, 146 supported
images, 12 MP photos sampled at 4000x3000, CLIP `Qdrant/clip-ViT-B-32`, 512 dimensions, four
runtime threads, three measured runs after one warmup run.

| Provider | Batch | Median images/s | Run range images/s | Median elapsed | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: |
| DirectML | 1 | 10.6 | 10.3–11.9 | 13.818 s | 669 MiB |
| DirectML | 16 | **12.3** | 12.1–12.3 | **11.877 s** | **1.58 GiB** |

### Camera-corpus trace breakdown

| Batch | Decode median/image | Decode maximum | `embed_images` median/full batch | SQLite save median/batch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 102 ms | 159 ms | 86.6 ms | 0.18 ms |
| 16 | 105 ms | 159 ms | 1.25 s | 0.47 ms |

Full-resolution JPEG decode is the dominant cost for this corpus. Batch 16 is approximately 16%
faster than batch 1, but its decoded-image queue requires substantially more memory. The current
vendored FastEmbed integration queues normalized tensors in the selected model's configured shape
(currently `[3, 224, 224]` for this CLIP model) rather than full-resolution RGB images.

## Vendored FastEmbed preprocessing

`fastembed` is vendored at `vendor/fastembed` from the 6.0.0 crate release. The fork adds two
general APIs suitable for upstream:

- `ImageEmbedding::preprocessor()` returns a cheap, cloneable `ImagePreprocessor` configured
  from the same model `preprocessor_config.json`;
- `ImagePreprocessor::preprocess` produces an owned `ndarray::Array3<f32>`, and
  `ImageEmbedding::embed_preprocessed` accepts a batch of those arrays.

The application keeps its decoder out of FastEmbed: decode workers use `nicegal_core::imaging` first,
then use the model-owned preprocessor. That preserves the gallery's format handling without
coupling FastEmbed to an application-specific image decoder. It also lets four CPU workers discard
each full-resolution image immediately after it becomes a normalized tensor, while the single
mutable ONNX session continues the prior batch on the GPU.

## Method and metadata

| Field | Value |
| --- | --- |
| Model | `Qdrant/clip-ViT-B-32` |
| Vector dimensions | 512 |
| Runtime threads | 4 |
| Providers tested | DirectML, OpenVINO |
| Batches tested | 1, 4, 8, 16 |
| Runs/configuration | 3 measured runs, 1 warmup run |
| Build | optimized `bench` profile via `dev.cmd` |
| Host logical parallelism | 20 |
| Runtime cache | warm Hugging Face cache |
| Failures | 0 in every completed run |

The benchmark command has the following form:

```console
set RUST_LOG=warn,nicegal_core::image_index=debug,nicegal_core::embedding::image=debug
dev.cmd bench --bench image_index -- ^
  --corpus C:/gallery ^
  --provider directml --batch-size 16 ^
  --trace-jsonl clip-camera-dml-b16.jsonl
```

The benchmark's JSONL traces contain:

- model-load and provider-selection events;
- one `decode_image` span per decoded source;
- one `preprocess_image` span per source;
- one `embed_preprocessed_images` span per inference batch;
- one `save_image_embeddings` span per SQLite-vec write; and
- the enclosing `image_index` span.

ONNX Runtime's own session profiling is not enabled. FastEmbed 6.0.0 owns the CLIP
`ort::Session` privately and does not expose the session builder's profiling configuration or the
session lifecycle needed to call `end_profiling`. The application-level spans therefore measure
the usable pipeline stages, while provider-kernel attribution would require a FastEmbed API
extension or a locally owned CLIP session.

## Conclusions

- DirectML is substantially faster than OpenVINO on this machine and uses less memory for the
  small corpus.
- Batch 16 was the best tested setting for both corpora, although the small-corpus result has
  unusually large batch-to-batch variance at smaller sizes.
- Full-resolution decode, not SQLite ingestion, becomes the limiting stage on 12 MP photos.
- These measurements use the baseline pipeline before normalized tensor queuing; they do not
  establish the throughput or memory use of the current implementation.
