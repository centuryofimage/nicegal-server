# Vector Search Measurements

The CLIP search additions copied an expensive OCR search pattern: run a scalar
cosine scan to count matches, then run it again to rank and return results.
SQLite can also evaluate the distance expression again for filtering and sorting
when the scoring CTE is inlined. In sqlite-vec 0.1.9, scalar vector access calls
`vec0_get_vector_data`, which locates the chunk, opens a blob, allocates a vector
buffer, reads it, and closes the blob for each evaluation. This is substantially
more overhead than the dedicated KNN scan.

## Query implementation

- OCR and image database connections allow up to 256 MiB of memory mapping for
  their main file. This is a mapping budget, not a reserved heap allocation.
  File-backed pages can be shared across connections and reclaimed by the OS.
  SQLite may map less or fall back to ordinary reads on unsupported platforms.
- Both vector searches materialize just asset IDs, distances, and modification
  timestamps once, then reuse these for the exact count and limited ranking.
- OCR contents and result metadata are fetched after selecting hits, avoiding
  materializing every candidate's OCR text.
- A left join preserves the total when no hits are returned, including a zero
  library-level limit. Regression tests cover this and an empty distance range.
- OCR distance ties now have a deterministic asset-ID tie breaker, like images.

## Measurements

Read-only connections to the existing application databases; no schema changes
or writes to application data. Python via `uv run --with sqlite-vec==0.1.9`,
SQLite 3.53.1, sqlite-vec 0.1.9. Queries use one stored vector as their input,
include all paths, and apply the production fingerprint joins. These timings
exclude model inference, highlighting, HTTP, and JSON serialization. The Python
SQLite build is not necessarily identical to the Rust application's bundled build.

The image database contained 71,928 vectors; OCR contained 52,181. Initial image
count-plus-top-50 runs without mapping took 17.38, 16.53, and 16.90 seconds, with
a later 32-second outlier. Enabling mapping reduced that operation to about a
second. An unfiltered KNN top-50 reference took 0.14 seconds, but does not provide
the same filtering or total-count contract.

The following are medians of three executions, **with mapping enabled on both
versions**, isolating the query rewrite. Timings varied with machine load.

| Search | Limit | Ceiling | Old SQL (ms) | New SQL (ms) |
| --- | ---: | ---: | ---: | ---: |
| Image | 50 | Unbounded | 1,036 | 793 |
| Image | 50 | 0.5 | 1,740 | 762 |
| Image | 100,000 | Unbounded | 2,196 | 1,134 |
| Image | 100,000 | 0.5 | 1,798 | 871 |
| OCR | 50 | Unbounded | 972 | 513 |
| OCR | 50 | 0.5 | 943 | 502 |
| OCR | 100,000 | Unbounded | 2,346 | 874 |
| OCR | 100,000 | 0.5 | 2,535 | 845 |

Totals matched in all cases. Full result sets matched including distances and
OCR metadata/snippets; image top-50 sets also matched. OCR adds a defined order
for previously unspecified ties. Existing tests cover roots, exclusions,
timestamps, stale vectors, snapshots, and distance ceilings.

## Measurement limits

The API defaults to 100,000 hits and permits 250,000, with an exact total before
the limit. Those semantics are expensive even with ANN: a typical small top-k
lookup cannot provide an exact count under a distance ceiling. These measurements apply to that exact-count contract.

This sqlite-vec version has a 4,096 KNN k cap. Simply replacing scalar SQL with
`MATCH ... k = limit` breaks large requests; joining root/fingerprint filters
after a globally limited KNN search can also under-return. KNN top-k lookup is therefore not equivalent to the measured query.

Inspect `embed_image_query`, `search_image_vectors`, `search_text_vectors`, and
the request blocking spans in the application trace. OCR highlighting and large
response serialization can remain significant after the SQL improvement.
Also remeasure when database files grow beyond the mapping budget or have a
large WAL: these results do not establish performance at larger scales.

## Application trace and cancellation

A reference application trace measured CLIP embedding at 5–8 ms, vector search
at 495–558 ms for 71,822 matches, and complete image HTTP requests at 552–575 ms.

GET and POST search retain a cancellation guard in the async request future.
Dropping that future interrupts the request's OCR and image connections through
rusqlite's `InterruptHandle`. A progress callback checks a persistent atomic flag
every 1,000 SQLite VM operations to cover cancellation between statements, when
an interrupt alone has no effect. Workers also check cancellation before starting
work, between query components, after inference, and between combined modes.

Cancellation is limited to request-owned readers; indexing jobs and other
requests use separate connections. It does not preempt an ONNX inference or
arbitrary Rust code already executing. Transport cancellation must drop the
handler future to trigger the guard. Expected cancellation is logged at debug
level rather than as a server failure.

Additional tests cover interrupting executing SQL, cancellation while SQLite is
idle, rejection of late connection registration, dropping an awaiting request,
and successful completion leaving its cancellation token intact.
