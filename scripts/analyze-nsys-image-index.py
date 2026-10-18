"""Summarize an image-index Nsight Systems capture exported to SQLite.

Usage: python -I scripts/analyze-nsys-image-index.py target/nsys-image-index/image-index.sqlite

Prints each NVTX range's count and time, GPU metrics averaged inside the inference stages, the
GPU-idle time inside a batch but outside the model stages, and per-batch stage durations. GPU
metrics exist only when the capture ran elevated.
"""

import bisect
import sqlite3
import statistics
import sys

# GPU metric name prefix -> short column label.
METRICS = {
    "GR Active": "GR%",
    "SMs Active": "SM%",
    "SM Issue": "issue%",
    "Tensor Active": "tensor%",
    "Compute Warps in Flight [Throughput": "warps%",
    "DRAM Read": "dram_rd%",
    "DRAM Write": "dram_wr%",
}
STAGES = ["image_index", "flush_batch", "embed_preprocessed_images", "gemma_vision", "gemma_text"]
MODEL_STAGES = ("gemma_vision", "gemma_text")
IDLE_BELOW = 10


def main(path: str) -> None:
    db = sqlite3.connect(path)
    ranges: dict[str, list[tuple[int, int, int]]] = {}
    for name, start, end, tid in db.execute(
        """select coalesce(s.value, e.text), e.start, e.end, e.globalTid from NVTX_EVENTS e
           left join StringIds s on s.id = e.textId where e.end is not null"""
    ):
        ranges.setdefault(name, []).append((start, end, tid))
    if not ranges:
        sys.exit("no NVTX ranges: build the server with --features nvtx")

    print(f"{'range':32} {'count':>6} {'total ms':>10} {'median ms':>10} {'threads':>8}")
    for name, spans in sorted(ranges.items(), key=lambda item: -sum(e - s for s, e, _ in item[1])):
        durations = [(e - s) / 1e6 for s, e, _ in spans]
        print(f"{name:32} {len(spans):6} {sum(durations):10.1f} {statistics.median(durations):10.2f} "
              f"{len({tid for _, _, tid in spans}):8}")

    series = metric_series(db)
    print()
    if series:
        print(f"{'GPU metrics, mean in range':32} " + " ".join(f"{label:>9}" for label in series) + f" {'idle%':>6}")
        for name in STAGES:
            if name in ranges:
                print_metrics(name, [(s, e) for s, e, _ in ranges[name]], series)
    else:
        print("no GPU metrics (capture without elevation)")

    gaps = []
    for start, end, _ in ranges.get("flush_batch", []):
        inner = sorted((s, e) for stage in MODEL_STAGES for s, e, _ in ranges.get(stage, [])
                       if s >= start and e <= end)
        cursor = start
        for s, e in inner:
            if s > cursor:
                gaps.append((cursor, s))
            cursor = max(cursor, e)
        if end > cursor:
            gaps.append((cursor, end))
    if series and gaps:
        print_metrics("flush_batch outside model stages", gaps, series)
    print(f"\nflush_batch time outside model stages: {sum(e - s for s, e in gaps) / 1e6:.1f} ms")
    batches = sorted((s, e) for s, e, _ in ranges.get("flush_batch", []))
    between = [(b[0] - a[1]) / 1e6 for a, b in zip(batches, batches[1:])]
    if between:
        print(f"between batches: max {max(between):.2f} ms, total {sum(between):.1f} ms")
    for stage in MODEL_STAGES:
        if stage in ranges:
            per_batch = [round((e - s) / 1e6, 1) for s, e, _ in sorted(ranges[stage])]
            print(f"{stage} per batch ms: {per_batch}")


def metric_series(db: sqlite3.Connection) -> dict[str, tuple[list[int], list[float]]]:
    try:
        names = db.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS").fetchall()
    except sqlite3.OperationalError:
        return {}
    series = {}
    for prefix, label in METRICS.items():
        match = next((metric for metric, name in names if name.startswith(prefix)), None)
        if match is None:
            continue
        rows = db.execute("select timestamp, value from GPU_METRICS where metricId = ? order by timestamp",
                          (match,)).fetchall()
        series[label] = ([t for t, _ in rows], [v for _, v in rows])
    return series


def print_metrics(name: str, spans: list[tuple[int, int]], series: dict) -> None:
    def samples(label: str) -> list[float]:
        times, values = series[label]
        return [v for s, e in spans for v in values[bisect.bisect_left(times, s):bisect.bisect_right(times, e)]]
    columns = [f"{statistics.mean(v):9.0f}" if (v := samples(label)) else f"{'-':>9}" for label in series]
    busy = samples("GR%") if "GR%" in series else []
    idle = f"{100 * sum(v < IDLE_BELOW for v in busy) / len(busy):6.0f}" if busy else f"{'-':>6}"
    print(f"{name:32} " + " ".join(columns) + f" {idle}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
