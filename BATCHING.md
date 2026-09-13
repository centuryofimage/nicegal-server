# OCR Batching and Measurements

## Current pipeline

Detection processes one image per inference, with dimensions aligned to multiples
of 32 and a default maximum side of 960 pixels. Recognition operates on text-line
crops from that image, sorted by aspect ratio and grouped into batches of eight.
Each batch is padded to its widest crop, with a minimum recognition width of 320.
Recognition does not combine crops from different images.

The detector and recognizer retain their float input buffers with the loaded model
pair, growing capacity only when needed. Resize images, crop images, result vectors,
and ONNX Runtime outputs have separate allocations. CPU runtime sessions use the
arena allocator and graph optimization level 3.

## Benchmark controls

Use the `ocr_index` benchmark from the backend workspace, with caller-supplied
images and detector/recognizer model files. See the [README](README.md#index-benchmark)
for a complete command. The harness exposes `--recognition-batch-size`,
`--detection-max-side`, `--threads`, `--replicas`, `--runs`, and `--trace-jsonl`.

Recognition batch sweeps commonly use 1, 4, 8, 16, and 32 crops. Compare the OCR
content digest as well as throughput: changing a tensor shape must not silently
change the recognized text. Provider selection is explicit and benchmark runs
disable CPU fallback.

JSONL traces record crops per image, actual recognition batch occupancy and width,
detection shapes, inference time, preprocessing time, and input-buffer allocations.
Measure model loading separately from warm indexing throughput.

## Recognition occupancy

A 250-image mixed-gallery corpus on CPU with four threads and batch size eight
had 39.6% mean occupancy. Of 286 recognition batches, 132 contained one crop and
257 used width 320. Detection inference accounted for 66.4% of a stable run;
recognition inference accounted for 16.1%.

In a larger corpus, 1,770 processed images produced 2,376 recognition batches with
mean occupancy 4.5/8 (56%). Of those batches, 596 contained one crop and 1,878 used
width 320. Recognition inference accounted for 316 seconds of a 1,549-second OCR
phase (20.4%). These measurements describe those corpora, not a universal speedup
from larger batches.

## Detector resolution and session replicas

The following CPU measurements use 490 images, four intra-op threads per session,
and one measured run per setting:

| Replicas | Maximum side | Wall time | Images/s | Summed CPU | Detection/image | Content digest |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2000 | 311.8 s | 1.57 | 311 s | 362 ms | `d1fba762150c532c` |
| 1 | 960 | 142.0 s | 3.45 | 141 s | 165 ms | `d1fba762150c532c` |
| 4 | 2000 | 209.0 s | 2.34 | 834 s | 928 ms | `d1fba762150c532c` |
| 4 | 1280 | 140.1 s | 3.50 | 556 s | 654 ms | `d1fba762150c532c` |
| 4 | 960 | 132.4 s | 3.70 | 527 s | 518 ms | `d1fba762150c532c` |

At one replica, reducing the maximum side to 960 produced 2.20x throughput with
identical text on this corpus. Four replicas at that resolution improved wall
time by about 7% while using substantially more CPU. The application defaults to
one replica; the replica option is retained for benchmarks.

## Measurement limits

Separate CPU-provider runs on the smaller corpus varied from 47.3 to 134.8 seconds
without a corresponding implementation change. That variability prevents attributing
a reliable throughput gain to buffer reuse or arena allocation alone. Record host
load, hardware, provider/runtime versions, corpus, thread count, warmup, and repeated
runs alongside any result. The recorded CPU results do not establish DirectML or
OpenVINO throughput.
