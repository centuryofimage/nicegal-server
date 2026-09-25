# Internal API

`nicegal-server` is the local RPC host for the controlling desktop process. It binds an
ephemeral IPv4 loopback port and prints one JSON readiness line containing `apiVersion` and
`endpoint`. Every request requires the token from `NICEGAL_RPC_TOKEN` as a bearer token.
Closing the server's stdin starts graceful shutdown.

## Frontend integration at a glance

Electron owns the configured database paths. Rust owns catalog and metadata policy:

| Frontend operation | Interface |
| --- | --- |
| Define libraries and their folders; clients request scans | `/v1/libraries`, see [Libraries](#libraries) |
| Enumerate/sort the library | `GET /v1/catalog?libraryId=...&timeline=modified\|capture` |
| List indexed folders for the tree | `GET /v1/catalog/folders?libraryId=...` |
| Count a library | `GET /v1/catalog/count?libraryId=...` |
| Inspect a photo or video | `GET /v1/catalog/metadata?assetId=...` |
| Poll for catalog changes | `GET /v1/catalog/revision` |
| Load thumbnail blobs | Read-only SQLite on `thumbnails.db`, normally behind `thumb://` |
| Search OCR text or indexed images (one mode) | `GET /v1/search` |
| Search several ways at once | `POST /v1/search` |
| Underline what matched in an OCR result | `highlights`, see [Highlights](#highlights) |
| Check vector-search coverage | `GET /v1/text-embeddings` |
| Check visual-search coverage | `GET /v1/image-embeddings?libraryId=...` — `{total,indexed}`; cataloged images and videos with current vectors for the active image model, excluding stale fingerprints |
| Start, monitor, and cancel work | `/v1/jobs` |
| Backfill thumbnail variants | `POST /v1/thumbnails/generate`, then poll the returned job |
| Explicitly backfill OCR text embeddings | `POST /v1/text-embeddings/generate`, then poll the returned job |
| Backfill CLIP image embeddings | `POST /v1/jobs` with type `imageEmbed` |
| Preview/delete missing files | `POST /v1/jobs` with type `pruneMissing` |
| Remove a configured library and its derived data | `POST /v1/jobs` with type `libraryPurge` |
| Resolve one known path | `GET /v1/assets?path=...` |

Electron does not open `assets.db`: listings, counts, revision checks, asset resolution (including
`original://`) and metadata use Rust HTTP APIs. The thumbnail blob reader remains direct SQLite;
production Electron does not fetch thumbnail bytes over HTTP.

Vector search is the primary search mode; glob, FTS, and regex remain available and are reached by
naming them explicitly. OCR-text vectors live in the OCR store and are produced by the server
itself. Select the **OCR** text model with `--embed-model`/`NICEGAL_EMBED_MODEL`; BGE small
English v1.5 is the default and currently the only supported OCR model. That selection does not
affect image search. CLIP image vectors live in `qdrant-clip-vit-b-32.db`, derived from the full
publisher/model slug. The server creates no model sessions and downloads no models before
readiness. Models are prepared when indexing is requested. Browsing, catalog sync, thumbnails, and literal OCR search remain available
without models. Schema validation and bundled runtime initialization still occur before readiness.

Configure the three independent stores with `--asset-database`/`NICEGAL_ASSET_DB`,
`--ocr-database`/`NICEGAL_OCR_DB`, and
`--thumbnail-database`/`NICEGAL_THUMBNAIL_DB`. Schema versions are strict. A database with a
nonzero version other than the documented version is rejected; this project does not migrate or
accept the old image-owned schemas. The image-vector store is derived automatically beside the
configured asset catalog from the active model filename.

### Search model preparation

`GET /v1/models` returns `{text, clipImage, clipText}`. Each entry contains
`{state: "notLoaded" | "preparing" | "ready" | "failed", error: string | null}`.
The state describes sessions in this process, not disk-cache availability: a restart resets it to
`notLoaded`. Failed preparation retains its diagnostic until the next preparation attempt, which
clears the error and enters `preparing`; a successful attempt enters `ready`. Status reads do not
wait for download or compilation. OCR continues to use `GET /v1/ocr/models` and its existing
`ocrModelLoad` job.

`POST /v1/jobs` with `{"type":"modelPrepare","params":{}}` prepares the text, CLIP image,
and paired CLIP text sessions without indexing a root. It is the Settings prepare/retry action.
An `imageEmbed` job prepares the CLIP pair; `textEmbed` prepares the text session; an
`libraryScan` prepares whichever of them its pending work needs. Already prepared sessions are reused.
Failed jobs can be retried by starting the same request again. Preparation uses the usual
single-active-job scheduling and reports failures through both the job error and model status.

Embedding preparation uses `loadingModels`: `progress.total` is the number of required
sessions (1, 2, or 3), and `phaseCompleted`/`modelsLoaded` advances after each session.
FastEmbed combines downloading/cache resolution and session compilation, so label this phase
"Preparing search models (may download)" with indeterminate activity while each session loads.
There are no per-byte FastEmbed download percentages. Cancellation is honored between sessions;
an in-flight synchronous model download or compilation must finish first. Successfully prepared
sessions remain reusable after cancellation; cancellation does not delete cached model files.

Vector and CLIP searches lazily reload their text-query encoder from disk after a restart.
An already set-up library can be searched immediately without reindexing or visiting Settings. `/v1/models` reports `preparing` while the search request waits for compilation, then
`ready`. The CLIP image encoder and OCR recognition sessions are not needed to search existing
vectors and remain unloaded until indexing/setup requests them.

The query loader resolves the ONNX file, all declared external/additional files, and all four
tokenizer JSON files through `hf_hub::CacheRepo`, then loads those paths directly. It constructs
no HTTP client and never falls back to the downloading loader. Missing files return HTTP 409
`models_not_ready` with indexing/setup guidance; corrupt files produce the same actionable API
code and retain diagnostics in model status. Explicit setup/indexing remains the only download
path. A search cancelled during synchronous compilation stops after that compilation finishes;
its completed session remains reusable. Existing advanced weighted image-query request shapes
are unchanged.

OCR model-load jobs use the process-wide execution provider — the persisted selection from
`PUT /v1/runtime` (see below), or the `--execution-provider`/`NICEGAL_EXECUTION_PROVIDER` flag
overriding it for one launch. It defaults to `directml` when nothing has been persisted yet. The
provider is chosen when models load, so changing it normally requires restarting the server before
loading models again.

`runtime::fallback_chain` walks `directml` → `openvino` → `cpu` when the requested provider fails
to compile, but the `openvino` rung only ever succeeds if the *loaded* runtime distribution has
OpenVINO compiled in — see "Runtime setup" below: a process launched with `directml` only ever
lands on `cpu` in practice, since the two accelerated providers ship in separate, mutually
exclusive `onnxruntime.dll` distributions. When that specific case happens — `directml` requested,
`cpu` actually loaded — the `ocrModelLoad` job itself persists `openvino` as the next launch's
provider and exits with `RESTART_EXIT_CODE` (see `api::RESTART_EXIT_CODE` in `nicegal-server`) to
ask the desktop launcher for an immediate, transparent restart into it, rather than running the
rest of the session on CPU. The desktop launcher (`main/backend/nicegal-server-process.ts`) recognizes
this exit code and respawns without reporting a crash.

All four stores use WAL mode. Their separation and the stable columns explicitly documented below
are part of the desktop read contract. Any table or column shape change requires a `user_version`
bump; readers should reject versions they do not support.

Writable startup connections apply supported migrations in one transaction. Asset catalog
versions 2–7 upgrade to 8; OCR version 8 upgrades to 9. Read-only connections require the
current version and never migrate. Unsupported versions are rejected without changes.

## Asset catalog schema (version 8)

The asset catalog owns canonical identity. `asset_id` is stable across rescans and source changes
because updates conflict on the unique absolute path without replacing the row.

```sql
CREATE TABLE assets(
    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    source_modified_ns INTEGER NOT NULL,
    source_created_ns INTEGER,
    exif_taken_ns INTEGER,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    media_kind TEXT NOT NULL,
    media_format TEXT NOT NULL,
    width INTEGER CHECK(width > 0),
    height INTEGER CHECK(height > 0),
    is_animated INTEGER NOT NULL CHECK(is_animated IN (0, 1)),
    frame_count INTEGER CHECK(frame_count > 0),
    duration_ms INTEGER CHECK(duration_ms >= 0),
    metadata_version INTEGER NOT NULL DEFAULT 2 CHECK(metadata_version > 0)
);
CREATE INDEX assets_modified_idx
    ON assets(source_modified_ns, asset_id);
CREATE INDEX assets_taken_idx
    ON assets(COALESCE(exif_taken_ns, source_modified_ns), asset_id);
CREATE TABLE decode_failure_state(
    asset_id INTEGER PRIMARY KEY,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0)
);
CREATE TABLE catalog_meta(
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    revision INTEGER NOT NULL CHECK(revision >= 0)
);
```

Scans and catalog upserts accept PNG, JPEG, GIF, WebP, and BMP images, plus MP4, M4V, MOV, MKV,
WebM, AVI, MPG, MPEG, TS, and M2TS videos. Lookups, listings, timelines, and counts include both
media kinds. Image dimensions, GIF frame count, and duration are best-effort. Video dimensions,
frame count, and duration are probed from the container; unsupported or corrupt videos remain in
the catalog with unknown optional metadata. For images, `exif_taken_ns` is Unix nanoseconds parsed from `DateTimeOriginal`, including
`SubSecTimeOriginal` and `OffsetTimeOriginal` when present; an absent EXIF offset deterministically
means UTC. For videos the same capture-timeline column holds container `creation_time` only when
it parses as RFC 3339; the inspector labels this as video metadata, separate from EXIF. Probe
failure leaves optional columns null and does not remove the asset. GIF posters
are static PNGs produced from the first decoded/composited frame; original animated GIF bytes
never belong in the thumbnail database.

These columns are Rust implementation details. Gallery listing uses these indexed orders
(descending timestamp and descending asset ID for the desktop):

```sql
SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size,
       media_kind, media_format, width, height, is_animated, frame_count, duration_ms
  FROM assets
 ORDER BY source_modified_ns, asset_id;

SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size,
       media_kind, media_format, width, height, is_animated, frame_count, duration_ms
  FROM assets
 ORDER BY COALESCE(exif_taken_ns, source_modified_ns), asset_id;
```

`catalog_meta` contains exactly one row (`singleton = 1`). Its `revision` increases in the same
transaction as every changed catalog row and does not change for an unchanged fingerprint, so a
`GET /v1/catalog/revision` can cheaply poll it before refreshing the listing.

### Libraries

Library definitions live in the catalog database (version 8). A library owns no assets: its scope
is the union of its included folders minus its excluded folders, evaluated against `assets.path`
on every read. Editing folders therefore writes only these rows; overlapping libraries share every
cataloged file and its search data, and removing a folder or library never deletes indexed data.

```sql
CREATE TABLE libraries(
    library_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_sequence INTEGER NOT NULL CHECK(scan_sequence >= 0),
    index_ocr INTEGER NOT NULL CHECK(index_ocr IN (0, 1)),
    index_image INTEGER NOT NULL CHECK(index_image IN (0, 1)),
    import_key TEXT UNIQUE
);
CREATE TABLE library_folders(
    library_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    excluded INTEGER NOT NULL CHECK(excluded IN (0, 1)),
    position INTEGER NOT NULL CHECK(position >= 0),
    scan_requested INTEGER NOT NULL CHECK(scan_requested >= 0),
    scan_completed INTEGER NOT NULL CHECK(scan_completed >= 0),
    scan_error TEXT,
    last_scan_completed_ns INTEGER,
    scan_outcome TEXT
        CHECK(scan_outcome IN ('unavailable', 'incomplete', 'cancelled', 'failed')),
    PRIMARY KEY(library_id, path)
) WITHOUT ROWID;
```

Path membership is one rule, `scope::PathScope`, shared by catalog listing, counts, search,
coverage, reconciliation, and deletion: a literal, case-sensitive prefix ending at a path separator
(`/` or `\` on Windows, `/` elsewhere). `D:\Photos` covers `D:\Photos\a.png` but not
`D:\Photos-old\a.png` or `D:\Photos` itself, and SQL wildcard characters in folder names are
literal. Stored paths are canonical, so callers pass canonical folders. Each folder renders as a
byte range on the path column, which SQLite answers from the path index. An exclusion wins over
every include.

An included folder has an outstanding scan while `scan_requested > scan_completed`. Creating a
library requests a scan of each folder. An edit requests scans only where visible files can grow:
new includes, includes that contained a removed exclusion, and every include when OCR or image
indexing is switched on. Hiding files (removing a folder, adding an exclusion, switching an index
off) requests nothing. Each request takes the next `scan_sequence` number, and a scan completes
only the request number it started from. Numbers are never reused, so neither a request made
during a scan nor a folder removed and re-added during one is marked complete by it. These
counters are internal; the API reports `scanPending`, `scanOutcome`, and `scanError`.
`scan_outcome` is why the latest attempt stopped short, and `scan_error` its text; NULL means no
failed attempt, and a complete scan clears both. Version 8 added `scan_outcome`; a version-7 folder
with a `scan_error` migrates to `failed` and keeps its text. The backend never starts a
scan on its own: a client decides when a library is brought up to date, normally with a
`pendingOnly` scan when the user creates, edits, or opens it.

## OCR schema (version 9)

OCR is a derived store for eligible images only. Videos are excluded even during force, rescan,
and retry. It receives `asset_id` from the catalog and never allocates gallery IDs.
`source_path` is denormalized only for directory filtering and cleanup.

```sql
CREATE TABLE ocr_results(
    asset_id INTEGER PRIMARY KEY NOT NULL CHECK(asset_id > 0),
    source_path TEXT NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    exif_taken_ns INTEGER,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    mark_delete INTEGER NOT NULL DEFAULT 0 CHECK(mark_delete IN (0, 1)),
    content TEXT NOT NULL
);
CREATE INDEX ocr_modified_idx ON ocr_results(source_modified_ns);
CREATE INDEX ocr_taken_idx ON ocr_results(COALESCE(exif_taken_ns, source_modified_ns));
CREATE VIRTUAL TABLE ocr_results_fts USING fts5(
    content,
    content=ocr_results,
    content_rowid=asset_id,
    tokenize='trigram case_sensitive 0'
);
```

The insert, update, and delete triggers in `src/db_create.sql` keep the external-content FTS table
synchronized. Search results expose the unchanged catalog-owned `asset_id`.

`exif_taken_ns` is denormalized from the asset catalog for the same reason `source_path` is: search
filters on it, and the read-only search connection opens the OCR store alone. The two indexes match
the two timeline expressions exactly, so a time-bounded search is a range scan.

### Search filters

The path scope (see [Libraries](#libraries)) and the time range are one value (`db::SearchFilters`) shared by every
search mode rather than re-derived per mode. It renders its `WHERE` fragments and the named
parameters they bind *together*, so a filter cannot reach one mode's SQL without its parameter. Any
new filter is added once there and every mode — FTS, glob, regex, OCR vectors, and image vectors —
gets it.

### Vector search

`sqlite-vec` is compiled into the binary and registered with `sqlite3_auto_extension`, so every
connection this process opens has `vec0` and the `vec_*` scalar functions available. No extension
file is loaded at runtime.

```sql
CREATE TABLE ocr_embedding_model(
    space TEXT PRIMARY KEY CHECK(space = 'ocrText'),
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0)
);
CREATE TABLE ocr_embedding_state(
    space TEXT NOT NULL CHECK(space = 'ocrText'),
    asset_id INTEGER NOT NULL REFERENCES ocr_results(asset_id) ON DELETE CASCADE,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    PRIMARY KEY(space, asset_id)
);
-- One per space, created on first use, not by db_create.sql. See below.
CREATE VIRTUAL TABLE ocr_embeddings_ocr_text USING vec0(
    asset_id INTEGER PRIMARY KEY,
    embedding FLOAT[<dimensions>] distance_metric=cosine
);
```

The OCR database currently has one OCR-text embedding space, `ocrText`. It holds one text model at a time;
declaring a different model discards its stored vectors rather than mixing incompatible coordinate
systems.

The `ocr_embeddings_*` tables are the only ones not created by `db_create.sql`: `vec0` bakes the
dimension count into its DDL, and that count is a property of the configured model rather than of
the schema version. Each is created the first time its space declares a model. **A reader must
treat an absent `ocr_embeddings_*` as "nothing has been embedded into that space yet", not as a
damaged database.** Everything a reader needs to query in plain SQL — coverage, backlog, which rows
are current — lives in `ocr_embedding_state`, which `db_create.sql` does create.

`ocr_embedding_state` carries the OCR fingerprint each vector was computed from, and search joins
it back to `ocr_results` on that fingerprint. Re-running OCR over a changed file therefore retires
its vector automatically: the row drops out of vector search and reappears in the backlog until it
is embedded again. Deleting an OCR row cascades the state row away; `vec0` is a virtual table and
cannot be a foreign key target, so its rows are swept explicitly by `sweep_deletions` and
`delete_asset`. Writers must open with `PRAGMA foreign_keys = ON`, which `DB::new` does.

Vector search is an **exact** scan (`vec_distance_cosine` over the joined rows), not a `vec0`
`MATCH ... k = ?` lookup. `k` is applied before any join, so under a root filter an approximate
top-k would silently return fewer than `limit` rows whenever the nearest vectors live outside the
searched directory. `vec0` is brute force either way, so exactness costs a constant factor rather
than an algorithm.

OCR-text embeddings are produced by `src/embedding/` through FastEmbed and BGE small English v1.5. The model
is downloaded into the standard Hugging Face cache on first use and reused on subsequent launches.
Its 384-dimensional normalized vectors are stored with the stable model identifier
`BAAI/bge-small-en-v1.5`.

### Image embedding index (version 3)

Visual ingestion and search are independent of OCR and use one database per model. MetaCLIP2 B/32
is the default; its normalized 512-wide vectors are stored in
`facebook-metaclip-2-worldwide-b32.db`. Text requests to `type=image` use the active model's paired
text encoder in the same vector space. It is deliberately loaded on CPU: an image query is one
short text forward pass, while the configured accelerator remains available for OCR and
image-indexing work.

`GET /v1/search` keeps `type=image` text-only through `q`. `POST /v1/search` additionally supports
signed asset-ID and text components; see [Composite image queries](#composite-image-queries).
Precomputed binary vectors and a general image-to-vector upload endpoint are not accepted.

```sql
CREATE TABLE image_embedding_state(
    asset_id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL,
    sampling_version INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE image_embeddings USING vec0(
    embedding_id INTEGER PRIMARY KEY,
    asset_id INTEGER,
    +timestamp_ms INTEGER,
    embedding FLOAT[512] distance_metric=cosine
);
CREATE TABLE image_embedding_samples(
    embedding_id INTEGER PRIMARY KEY,
    asset_id INTEGER NOT NULL,
    timestamp_ms INTEGER
);
CREATE INDEX image_embedding_samples_asset_idx
    ON image_embedding_samples(asset_id, timestamp_ms);
CREATE INDEX image_embedding_video_samples_idx
    ON image_embedding_samples(asset_id, timestamp_ms) WHERE timestamp_ms IS NOT NULL;
```

The state fingerprint comes directly from the asset catalog. `sampling_version` is zero for still
images and the current video sampling policy version for videos. Each still has one vector with
null `timestamp_ms`; each video can have several vectors sharing its `asset_id`, each with the
actual sampled presentation timestamp. `embedding_id` is the `vec0` row identifier. The ordinary
sample table gives per-asset and video-timestamp reads indexed keys without scanning the vector
table. A video's vectors, sample keys, and state are replaced together in one transaction. Search
scores current samples, picks each asset's lowest-distance sample before
applying the result limit or counting `total`, and returns one hit per asset with the winning
`timestampMs` for videos. Coverage counts assets, not sample vectors.

A small decode pool feeds bounded batches of RGB image and video samples ahead of a single
FastEmbed/ONNX inference lane. FastEmbed applies the CLIP preprocessor and submits
`[B, 3, 224, 224]`; full batches use `B = 8`, while the final partial batch may be smaller.
ONNX Runtime owns model execution threading. Video samples are decoded by the shared bounded
FFmpeg service described in [FFMPEG.md](FFMPEG.md); videos never enter OCR.

## Thumbnail schema (version 4)

Thumbnails are static derived artifacts. Fixed maximum-edge buckets are 128, 256, 512, and 1024
physical pixels. `generator_version` invalidates generator changes independently from UI size
requests.

```sql
CREATE TABLE thumbnails(
    asset_id INTEGER NOT NULL CHECK(asset_id > 0),
    size_bucket INTEGER NOT NULL CHECK(size_bucket IN (128, 256, 512, 1024)),
    generator_version INTEGER NOT NULL CHECK(generator_version > 0),
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    encoding TEXT NOT NULL CHECK(encoding IN ('image/jpeg', 'image/png', 'image/webp')),
    data BLOB NOT NULL,
    PRIMARY KEY(asset_id, size_bucket, generator_version)
) WITHOUT ROWID;
CREATE TABLE video_thumbnails(
    asset_id INTEGER NOT NULL CHECK(asset_id > 0),
    timestamp_ms INTEGER NOT NULL CHECK(timestamp_ms >= 0),
    size_bucket INTEGER NOT NULL CHECK(size_bucket IN (128, 256, 512, 1024)),
    sampling_version INTEGER NOT NULL CHECK(sampling_version > 0),
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    encoding TEXT NOT NULL CHECK(encoding IN ('image/jpeg', 'image/png', 'image/webp')),
    data BLOB NOT NULL,
    PRIMARY KEY(asset_id, timestamp_ms, size_bucket)
) WITHOUT ROWID;
```

Production Electron access is direct read-only SQLite access from the main process. Given
`:asset_id`, `:generator_version`, the current source fingerprint, and `:requested_size`, use:

```sql
SELECT size_bucket, generator_version, width, height, encoding, data
FROM thumbnails
WHERE asset_id = :asset_id
  AND generator_version = :generator_version
  AND source_modified_ns = :source_modified_ns
  AND source_size = :source_size
ORDER BY CASE WHEN size_bucket >= :requested_size THEN 0 ELSE 1 END,
         CASE WHEN size_bucket >= :requested_size THEN size_bucket END ASC,
         CASE WHEN size_bucket < :requested_size THEN size_bucket END DESC
LIMIT 1;
```

This chooses the smallest adequate current bucket, then the largest current smaller fallback.
Changing display size only changes `:requested_size`; it does not invalidate stored variants.
Every `thumbnails` column shown above remains the stable direct-read surface for the desktop.
Video embedding ingestion eagerly stores every sampled frame in `video_thumbnails` at every
bucket, keyed by asset, actual timestamp, and bucket; reads also require the current source
fingerprint and sampling version. The first sample is copied into `thumbnails` as the gallery
poster, so the existing Electron reader still finds it. The matched frame is fetched separately
with `GET /v1/thumbnails?timestampMs=...`. The two databases commit separately: thumbnails are
stored before vectors, and a failed or interrupted indexing attempt can be retried.

Generator version 1 creates still-image variants lazily by default. A library index job catalogs
media and OCRs eligible images; the gallery calls the synchronous ensure endpoint for its visible
image IDs. Full-library generation remains available through the thumbnail backfill job. Still images are
decoded once per asset for all missing buckets; opaque results are JPEG quality 85 and alpha-bearing
results are PNG. GIFs remain static PNG first-frame posters. A failed thumbnail decode is recorded
on the job but does not discard the catalog row or prevent OCR. Current source/version variants are
skipped, source changes overwrite the same generator key, and old generator versions remain until a
successful backfill for that library explicitly requests `sweepStale`.

## Local HTTP API

**Reads address a library.** `GET /v1/catalog`, `GET /v1/catalog/count`, both search forms,
`GET /v1/text-embeddings`, and `GET /v1/image-embeddings` require `libraryId`; an unknown library
is `404 library_not_found`. The read covers the library's stored scope, its included folders minus
its exclusions (see [Libraries](#libraries)), and never touches the filesystem, so a library whose
folders are offline still lists, counts, and searches its cached index.

Version 1 routes:

- `GET /v1/health`
- `GET /v1/status` reports the API version and process/runtime status:
  ```json
  {
    "apiVersion": 1,
    "runtime": {
      "activeExecutionProvider": "directml",
      "activeRuntimeDistribution": "directml",
      "onnxRuntimeBuildInfo": "ORT Build Info: ...",
      "configuredExecutionProvider": "openvino",
      "indexVideos": true,
      "ocrModels": {
        "detection": { "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx", "revision": "main", "filename": "inference.onnx", "configFilename": "inference.yml" },
        "recognition": { "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx", "revision": "main", "filename": "inference.onnx", "configFilename": "inference.yml" }
      },
      "restartRequired": true
    },
    "ocrModelsLoaded": false
  }
  ```
  `activeExecutionProvider` is the provider requested in this process and
  `activeRuntimeDistribution` identifies the ONNX Runtime DLL set actually loaded for it;
  `onnxRuntimeBuildInfo` is ONNX Runtime's own release/commit/build-flags diagnostic string;
  `configuredExecutionProvider` is the persisted choice for the next launch. They differ after a
  setting changes and while a command-line provider override is in effect. `indexVideos` and
  `ocrModels` are the saved defaults every scan uses unless its request supplies them. `ocrModelsLoaded`
  avoids a follow-up request when the client only needs to decide whether OCR indexing is available;
  use `GET /v1/ocr/models` for loaded model identities and their actual session provider.
- `GET /v1/runtime` returns the `runtime` object shown above. `PUT /v1/runtime` accepts
  `executionProvider`, `imageModel`, `indexVideos`, and `ocrModels` (the pair shape shown under
  Jobs). Scan options persist in `runtime.json`; explicit scans may also supply them and update
  the saved choices. The built-in defaults are video indexing enabled and the PaddleOCR v6 small
  detector/recognizer pair. The frontend can omit both options on ordinary scans.
  The execution provider selection persists for the next launch.
  Saved selections absent from the current build migrate to its bundled default at startup;
  explicit unsupported selections are rejected. Retired CUDA/MIGraphX settings also migrate.
  The runtime's `availableExecutionProviders` lists choices supported by this platform and build;
  the desktop shows those choices and the update endpoint rejects unavailable ones. It returns the
  same runtime object with `restartRequired: true` when a restart is needed. It never tries to
  unload or replace ONNX Runtime in the current process. The desktop applies both
  provider and model selections through its shared backend restart flow, leaving
  the app open. Provider and image-model updates are rejected while a job is active; scan-option
  updates can be saved while work runs and apply to scans that start afterwards.
- `GET /v1/catalog?libraryId=<id>&timeline=modified|capture` returns the desktop gallery
  array, including videos. `timeline` defaults to modified; capture falls back to modified when
  image EXIF or video container creation time is absent.
  Both sorts descend with asset ID as the descending tie-breaker. Each row has `id`, `path`, `displayName`, `extension`,
  `modifiedNs`, `createdNs`, `captureNs`, `sourceSize`, `mediaKind`, `mediaFormat`, `width`, `height`,
  `animated`, `frameCount`, `durationMs`. IDs, nanosecond timestamps and byte sizes are decimal
  strings; unavailable timestamps/dimensions/frame count/duration are null.
- `GET /v1/catalog/count?libraryId=<id>` returns a JSON integer without materializing rows.
  `GET /v1/catalog/revision` returns a decimal-string revision.
- `GET /v1/catalog/folders?libraryId=<id>` returns a path-sorted JSON array of
  `{ "path": "D:\Photos\Trips", "modifiedNs": "1790000000000000000" }` from completed scan
  snapshots, plus configured included roots. `modifiedNs` is the directory's modification time at
  its latest completed scan, as a decimal string, or `null` for a root that has not finished one.
  It includes empty directories and filters excluded paths. A newly added root appears immediately; its subfolders appear after its
  first complete scan. No source directory is walked by this read.
- `GET /v1/libraries` lists every library in creation order; `GET /v1/libraries/<id>` returns one.
  A library is
  ```json
  {
    "id": 3,
    "include": [{
      "path": "D:\\Photos",
      "scanPending": false,
      "scanOutcome": null,
      "scanError": null,
      "lastScanCompletedNs": "1790000000000000000"
    }],
    "exclude": ["D:\\Photos\\Private"],
    "ocr": false,
    "image": true
  }
  ```
  Libraries have no stored name; clients derive a label from the folders. `scanPending` means the folder was added or an edit revealed more
  of it (a removed exclusion, or OCR or image indexing switched on) and no complete `libraryScan`
  of it has finished since. `scanOutcome` is why the latest scan of the folder stopped short:
  `unavailable` (the folder is offline or unreadable), `incomplete` (unreadable entries, or the
  walk reached its `debugLimit`), `cancelled`, or `failed` (cleanup or any other job failure); null
  when no attempt failed. `scanError` is human-readable detail for it, not meant to be parsed.
  Both clear on the next complete scan, and a new scan request for the folder clears them too.
  `lastScanCompletedNs` is when a complete full scan last finished, as decimal Unix nanoseconds.
  `ocr` and `image` choose which search indexes `libraryScan` maintains for the library, so a
  client saves a changed choice here before starting a scan; the scan request has no such
  options.
  `POST /v1/libraries` accepts `{include, exclude?, ocr?, image?, importKey?}` and returns `201`
  with the new library. `ocr` defaults to false and `image` to true. Folders are canonicalized and
  must exist, except in a request with `importKey`, which keeps a missing folder so an import can
  preserve a library on a disconnected drive. A missing folder is stored with the platform's
  separators and no trailing separator; its case and links cannot be resolved offline, so the first
  `libraryScan` that finds the folder reachable replaces it (and its exclusions) with the canonical
  spelling, keeping its scan state. Repeating a request with the same `importKey` returns the
  existing library with `200` and changes nothing. Creating a library starts no scan; its folders
  are `scanPending` until a client requests one.
  `PUT /v1/libraries/<id>` accepts `{include, exclude?, ocr, image}`, replaces the definition, and
  returns the updated library. Folders the library already has are not re-read, so an offline
  folder stays editable. There is no edit conflict check: the server serves one client, so the
  last write wins. An edit starts no scan; folders it revealed are `scanPending` until a client
  requests one.
  `DELETE /v1/libraries/<id>` returns `204` and keeps every cataloged file and its search data.
  Definitions need at least one included folder, may not repeat a folder, and each exclusion must
  be inside an included folder; an included folder inside an exclusion is rejected. These return
  `400 invalid_request`; a relative, missing, or non-directory folder is `400 invalid_root`.
- `GET /v1/catalog/metadata?assetId=<positive-id>` returns `{asset, file, ocrState, ocrText,
  textState, imageIndexed, decodeFailed}`. `asset` has the gallery shape above. `file` contains `sourceState`
  (`current`, `changed`, `missing`, `unavailable`), Windows `attributes`, selected image `exif`
  and video `video` fields as separate `{label,value}` arrays, and a nullable diagnostic `error`.
  Video fields include codec, frame rate, bit rate, container creation time, title, and HDR indication
  when available. Detailed file probing happens on demand; changed sources omit probe fields
  instead of mixing live data with saved catalog dimensions.
  Unsupported/no EXIF is an empty list, not a failure. Field text is bounded to 4096 characters.
  `ocrText` is the complete OCR result for the current source fingerprint, or null when OCR has
  not indexed the current file; a successful image with no recognized text returns an empty string.
  `ocrState` is `indexed|stale|notIndexed`; `textState` is `embedded|pending|noText|notIndexed` and
  matches the persisted OCR fingerprint/embedding state, excluding marked-for-deletion OCR rows.
  `imageIndexed` means a CLIP vector matches the catalog fingerprint in the active model's store.
  `decodeFailed` means this exact catalog fingerprint has a recorded decode failure. Other past
  job errors are not a durable per-asset error history. Index status describes the catalog snapshot;
  `file.sourceState` separately reports source changes since the last scan. No model is loaded.
  Unknown IDs return `asset_not_found`.
- `GET /v1/assets?path=<absolute-path>` resolves the path before lookup and returns the canonical
  catalog record. It includes the full `path` plus zero-I/O path derivatives (`displayName`,
  `folderPath`, and `extension`), media metadata, `exifTakenNs`, `sourceCreatedNs`, and the current
  source fingerprint. Its `indexState` is `indexed`, `stale`, or `notIndexed`: stale OCR belongs
  to an older source fingerprint and should be presented as “Needs reindex.” Nanosecond timestamps
  are decimal strings so JavaScript does not lose integer precision. Electron obtains canonical IDs
  through this route rather than inferring SQLite row IDs.
- `POST /v1/assets` accepts `{"assetIds":[...]}` for up to 512 known catalog IDs and returns
  `{"assets":[...], "missingAssetIds":[...]}` in request order. It does not stat files or enumerate
  a root, so the renderer can resolve visible search/gallery rows in bounded batches.
- `GET /v1/search?q=<query>&type=vector|image|name|path|ocrSimple|ocrMatch|ocrGlob|regex&libraryId=<id>&limit=<n>`
  runs **one** mode and returns
  `{total, results: [{assetId, timestampMs?, snippet, rank, distance?, highlights?}]}`. `total`
  counts all matches before the requested result cap; the default cap is 100,000 and the maximum is
  250,000. `rank` is the 1-based position in this mode's own ranking. An optional
  `folder=<absolute-directory>` intersects every mode with a literal,
  separator-delimited subtree inside the library. The same top-level `folder` field is available
  in `POST /v1/search`; filtering happens before each mode's result limit. Exclusions still win,
  and a folder outside the library returns zero results. `distance` is present only for
  `type=vector` and `type=image` and is a cosine distance (0 identical, 1 orthogonal, 2 opposite)
  in that mode's distinct vector space. `type=vector` searches OCR-text vectors; `type=image`
  embeds `q` with the active image model's paired CPU text encoder and searches that model's current
  image and video sample vectors. Image hits have an empty `snippet` and no `highlights`; video
  hits carry the best-matching sample's actual `timestampMs` (still-image hits omit it). There is
  one hit per asset, and `total` counts assets after choosing each video's best sample. Resolve
  metadata through `POST /v1/assets` and use the timestamp for an exact matched-frame thumbnail.
  `ocrSimple` and `ocrMatch` retain FTS rank order; `ocrGlob` ranks assets by matching-token
  count, then recency. For `type=ocrGlob`, `q` is matched case-insensitively against each complete OCR word token: `*` matches
  any number of characters and `?` matches one. Thus `dre*` is a word-prefix search (it does not
  match `andrew`), while `*cat*` finds a word containing `cat`. The backend scans the FTS5 word
  vocabulary and follows matching term postings rather than scanning every OCR text blob; glob
  snippets are the first 512 characters of OCR text, so they need not contain the matching token at
  all — see [Highlights](#highlights) for what is claimed about a snippet's contents and what is
  only estimated. `q` is capped at 4096 bytes. **`type` now defaults to `vector`**; the text
  modes and image mode are opt-in. `maxDistance=<0..2>` bounds either vector neighbourhood and is
  rejected for the text modes. Image mode applies the library scope and time filters to the asset
  catalog and excludes an embedding as soon as its source fingerprint is stale. `before`, `after`,
  and `timeline` bound the search in time; see
  [Time filtering](#time-filtering). A query SQLite cannot parse is `400 query_syntax` carrying
  SQLite's own message; see [Errors](#errors).
  OCR-mode `highlights` marks where in `snippet` this query matched; image hits omit it because
  they have no OCR snippet. See [Highlights](#highlights).
  `type=name` searches the filename, and `type=path` searches the stored full media path, as
  case-insensitive literal substrings inside the selected library. Both use the catalog's path
  trigram index for suitable terms and a scoped scan for short or wildcard-character terms.
  Path separators are normalized; the matched name or path is returned as the snippet.
  Name hits use numeric, case-insensitive ordering for ASCII filenames with asset ID as a stable
  tie-breaker. Non-ASCII names currently use lowercase lexical order, which can differ from the
  renderer's former locale-aware sort.
  Optional `pathContains=<text>` on GET, or top-level `pathContains` on POST, filters every mode
  by a case-insensitive literal substring of the full indexed path. It normalizes path separators
  and applies before each mode's result limit. It can be combined with `folder`, time bounds, and
  any search type. The path text is capped at 4096 bytes.
- `POST /v1/search` runs several modes and optionally fuses them. OCR modes share one OCR-store
  snapshot; `type=image` reads its independent CLIP index on a separate read-only connection. This
  is the route for a client that searches more than one way at once; see
  [Combined search](#combined-search)
- `GET /v1/text-embeddings?libraryId=<id>` reports what vector search can currently answer for:
  `{embedder: {model, dimensions}, stored: {model, dimensions} | null, indexed, embedded, pending}`.
  `embedder` is what this build produces, `stored` is what the database holds (null before the first
  backfill), and a difference between them means the next text embed job discards and rebuilds. This is
  how the UI tells "nothing matched" apart from "nothing has been embedded"
- `POST /v1/text-embeddings/generate` starts the text embedding backfill job. Its JSON body is
  `{libraryId, force:false, batchSize?}`. `force` re-embeds rows that already have a current vector;
  `batchSize` is 1 to 512 and is additionally clamped to what the backend accepts, so omitting it
  is normally right. The same job can be created through `POST /v1/jobs` with type `textEmbed`
- `POST /v1/jobs` starts a typed background job and returns `202 Accepted`. Types are
  `modelPrepare`, `ocrModelLoad`, `libraryScan`, `thumbnailGenerate`, `textEmbed`, `imageEmbed`,
  `pruneMissing`, and `libraryPurge`. Every type except `modelPrepare` and `ocrModelLoad` takes a
  `libraryId`; an unknown library is `404 library_not_found` and no job is created. A job reads its
  library's folders when it runs. `imageEmbed` accepts `{libraryId, force:false, debugLimit?}`
- `GET /v1/jobs` lists the active and retained recent jobs
- `GET /v1/jobs/<job-id>` returns one job's current state and progress
- `GET /v1/jobs/<job-id>/events` streams `snapshot` server-sent events whenever job state changes
- `DELETE /v1/jobs/<job-id>` requests cooperative cancellation
- `POST /v1/thumbnails/generate` starts a thumbnail backfill job. Its JSON body is
  `{libraryId, buckets:[1024], force:false, sweepStale:false, timeline:"modified",
  range:{fromNs,toNs}}`. Only catalog assets in the library's scope are selected before applying
  `timeline`, `range`, and `buckets`. `timeline` is `modified` or `capture`; capture uses
  `COALESCE(exif_taken_ns, source_modified_ns)`. Range bounds are optional decimal strings
  containing Unix nanoseconds, with an inclusive `fromNs` and exclusive `toNs`. `sweepStale`
  still requires an unbounded backfill of every bucket, and removes stale generators only for
  the library's assets. The same job can be created through `POST /v1/jobs` with type
  `thumbnailGenerate`
- `POST /v1/thumbnails` synchronously ensures variants for a visible image set. Its body is
  `{assetIds:[...], requiredSize:<physical-pixels>}` with at most 512 IDs before deduplication.
  Abandoned requests stop between generation chunks; shared work already underway may finish.
  `requiredSize` is 1 through 1024; it selects
  the smallest adequate fixed bucket. The response is `200` only after every requested current
  variant has been committed to `thumbnails.db`, and returns `{assetIds, requiredSize, sizeBucket,
  generatorVersion}`. Clients then read the bytes directly from SQLite.
- `PUT /v1/thumbnails?assetId=<id>&sizeBucket=<bucket>&generatorVersion=<version>&width=<actual-width>&height=<actual-height>&encoding=<mime-type>` with static encoded bytes. The server fully decodes the body, requires its decoded dimensions to equal `width` and `height`, and limits both edges to `sizeBucket`.
- `GET /v1/thumbnails?assetId=<id>&requestedSize=<physical-pixels>&generatorVersion=<version>`
  reads the standard gallery poster. For a video search hit, append `timestampMs=<nonnegative-ms>`
  to read the exact cached frame from `video_thumbnails`; an image asset rejects `timestampMs`.
  The matched frame is looked up by current source fingerprint and sampling version.
- `DELETE /v1/thumbnails?assetId=<id>` deletes every bucket and generator version for the asset

A successful GET uses the stored MIME type as `Content-Type` and returns
`X-NicegalServer-Asset-Id`, `X-NicegalServer-Size-Bucket`, `X-NicegalServer-Generator-Version`,
`X-NicegalServer-Thumbnail-Width`, and `X-NicegalServer-Thumbnail-Height`. PUT and GET resolve the source
path through the asset catalog and fingerprint the current file; stale variants are not returned.

### Time filtering

Every search mode accepts the same three parameters, on `GET /v1/search`, on the body of
`POST /v1/search`, and on any single query inside that body:

| Parameter | Meaning |
| --- | --- |
| `after` | Inclusive lower bound, Unix **nanoseconds** as a decimal string |
| `before` | Exclusive upper bound, Unix **nanoseconds** as a decimal string |
| `timeline` | `capture` or `modified`. Required whenever `before` or `after` is given |

Instants are decimal strings of Unix nanoseconds, matching `fromNs`/`toNs` on the thumbnail backfill
job, because a JSON number loses the low digits of a nanosecond timestamp in JavaScript. `capture`
is `COALESCE(exif_taken_ns, source_modified_ns)`; `modified` is `source_modified_ns`.

**`timeline` has no default.** "When the photo was taken" and "when the file last changed" are
different questions, and a result set produced by silently picking one cannot be interpreted by the
caller, so a bound without a timeline is `400 invalid_request`. A `timeline` with no bound is inert
rather than an error, so a client that always sends its preference needs no special case.

The range is half-open, so adjacent ranges tile without double-counting the boundary instant.
`after` greater than or equal to `before` is `400 invalid_request` rather than a silently empty
result.

There is no `during`; a caller that means "this day" or "this month" sends the two bounds it
resolves to, which keeps every timezone and calendar decision in the frontend where the user's
locale actually is.

In `POST /v1/search`, a query that names **any** of the three owns its whole time filter and does
not merge with the request-level one. A partial override cannot therefore inherit a timeline it did
not ask for; a query naming none of them inherits the request-level filter intact.

### Combined search

Clients that search several ways at once send **one** `POST /v1/search` rather than one `GET` per
mode. OCR modes run in one OCR-store WAL read transaction. `type=image` reads the independent CLIP
index in the same request on its own read-only connection, because no SQLite snapshot can span the
two database files. The same query text is embedded at most once per vector engine; fused ranking
is computed where both result lists already exist instead of being shipped across the socket to be
recombined. Vector queries are also why the combined form is a POST: a distance ceiling and
per-mode weights do not belong in a query string.

```json
{
  "libraryId": 3,
  "limit": 100,
  "queries": [
    { "key": "semantic", "type": "vector", "q": "coffee receipt", "maxDistance": 1.2, "weight": 2 },
    { "key": "visual", "type": "image", "maxDistance": 0.8,
      "imageQuery": { "components": [
        { "assetId": 101, "weight": 1 },
        { "assetId": 202, "weight": -1 },
        { "text": "a coffee on a marble table", "weight": 2 }
      ] } },
    { "key": "literal",  "type": "ocrSimple", "q": "coffee" },
    { "key": "files",    "type": "ocrGlob",   "q": "*invoice*", "limit": 50,
      "timeline": "modified", "after": "1717243200000000000" }
  ],
  "timeline": "capture",
  "after": "1704067200000000000",
  "before": "1735689600000000000",
  "fuse": { "method": "rrf", "k": 60, "limit": 50 }
}
```

At most 6 queries. `key` names the block in the response and its weight in `fuse`, defaulting to the
mode name; duplicate keys are `400 invalid_request`. A query's `limit` overrides the request-level
one. `weight` must be positive and defaults to 1. Omitting `fuse` returns the per-mode lists only.

```json
{
  "model": { "model": "BAAI/bge-small-en-v1.5", "dimensions": 384 },
  "imageModel": { "model": "facebook/metaclip-2-worldwide-b32", "dimensions": 512 },
  "queries": [
    { "key": "semantic", "type": "vector", "total": 812,
      "results": [{ "assetId": 41, "snippet": "...", "rank": 1, "distance": 0.21,
                    "highlights": [{ "start": 12, "end": 18, "kind": "exact" }] }] },
    { "key": "visual", "type": "image", "total": 24,
      "results": [{ "assetId": 52, "snippet": "", "rank": 1, "distance": 0.18 }] },
    { "key": "literal", "type": "ocrSimple", "total": 12,
      "results": [{ "assetId": 41, "snippet": "coffee ...", "rank": 1,
                    "highlights": [{ "start": 0, "end": 6, "kind": "indexed" }] }] }
  ],
  "fused": {
    "total": 18,
    "results": [{ "assetId": 41, "score": 0.032, "rank": 1, "sources": ["semantic", "literal"],
                  "snippet": "...", "highlights": [{ "start": 12, "end": 18, "kind": "exact" }] }]
  }
}
```

`model` is the model the *stored OCR-text* vectors belong to, null before the first backfill, so a
client can tell an empty OCR vector result from an unembedded library without a second request.
`imageModel` is always the restart-scoped image encoder whose vectors `type=image` searches. Each
block's `total` is that mode's match count before its `limit`, exactly as the
single-mode route reports it. `fused.total` is the number of distinct assets any mode returned,
before `fuse.limit`.

OCR modes' hits carry `highlights` in the same shape; see [Highlights](#highlights). Image hits
have no OCR text and therefore omit `highlights`. A fused hit carries the snippet and highlights of
the first mode that returned that asset — spans are offsets into one particular snippet, so the two
always travel together. If image is first, the fused hit has an empty snippet and no highlights.

### Composite image queries

`POST /v1/search` lets an image query combine current indexed images and CLIP text in one signed
direction:

```json
{
  "libraryId": 3,
  "queries": [{
    "key": "visual",
    "type": "image",
    "imageQuery": {
      "components": [
        { "assetId": 101, "weight": 1 },
        { "assetId": 202, "weight": -1 },
        { "text": "snow", "weight": 2 },
        { "text": "sports car", "weight": -2 }
      ]
    }
  }]
}
```

Every component has exactly one source: a positive `assetId`, non-empty `text`, or
`externalImage: { "bytesBase64": "..." }`, plus a finite,
nonzero `weight` whose magnitude is at most 100. At most 16 components may participate, including
the legacy `q` text when present. Asset IDs are read from the active model's image index and must
have a current catalog fingerprint; missing, stale, or not-yet-indexed assets reject the request
rather than silently changing its meaning. Reference assets do **not** need to be in the library:
the library scope and timeline filters constrain returned results, not the examples used to
compose the query. Component resolution and the image-index scan share one read transaction, so an
asset cannot be accepted as current and then become stale halfway through that same request.

The server normalizes each component vector, takes their signed weighted sum, then normalizes the
sum before cosine search:

$$
\operatorname{normalize}\left(\sum_i w_i \cdot \operatorname{normalize}(v_i)\right).
$$

Positive components pull results toward their meaning. Negative components are CLIP direction
arithmetic, not boolean exclusions: `+beach - people` seeks a beach-like, people-unlike direction;
it does not prove a returned image has no person. Components that cancel to a zero direction are
rejected.

The existing `q` field remains a backward-compatible positive text component for `type=image`.
It is optional only when `imageQuery.components` supplies at least one component. OCR `vector`,
`ocrSimple`, `ocrMatch`, `ocrGlob`, `path`, and `regex` still require `q` and reject `imageQuery`. `GET /v1/search`
does not accept components; use POST for structured composition.

External images are native-picker snapshots sent as standard padded base64 (no data-URL prefix).
The server accepts JPEG, PNG, GIF (first frame), WebP, and BMP, at most 16 MiB per image,
32 MiB of decoded source bytes across the request, and 40 million pixels per image. The search
route accepts at most 48 MiB of JSON; other routes retain the 8 MiB limit. Invalid payloads reject
the whole search with `invalid_request`; validation messages identify the query key and zero-based
component index. Component failures also include `error.componentIndex` (zero-based) for inline UI
feedback; unrelated errors omit it. Images never enter the catalog or any file store. A process-local 32-entry LRU
caches vectors by SHA-256 of the original bytes and active model identity (about 64 KiB of vectors
for CLIP ViT-B/32), without retaining image bytes. Weight changes reuse cached vectors; changed
image bytes and model changes do not. External queries can load an already cached image encoder,
but never download models. An unprepared cache returns `models_not_ready` as for text searches.

### Fusion

Fusion is reciprocal rank fusion: an asset scores `weight / (k + rank)` in every mode that returned
it, summed across modes. Ranks are combined rather than the underlying scores because FTS5 rank and
cosine distance are not on a common scale and cannot be made comparable by normalising them. Larger
`k` damps the top of each list, so agreement between modes outweighs one mode's confidence; `k`
defaults to 60. `sources` names the modes that returned the asset, which is the signal the fused
list exists to surface. Ties break on ascending `assetId`, so the order is deterministic.

A library that has never been indexed, a mode with no matches, and a library with nothing embedded
are all empty results rather than errors. A query SQLite cannot parse fails the whole request with
`400 query_syntax`, since a partial ranking would be misleading.

### Highlights

Every OCR-returning mode answers in one shape: `snippet` is plain OCR text, and `highlights` says
which parts of it matched, so a client has one renderer for OCR results. `type=image` is the
intentional exception: it returns the picture's `assetId`, distance, and rank with an empty
`snippet` and no `highlights`, because no OCR text caused the semantic image match.

```json
{ "assetId": 41, "snippet": "a coffee shop receipt", "rank": 1,
  "highlights": [{ "start": 2, "end": 8, "kind": "indexed" }] }
```

`start` and `end` are **character** offsets into that hit's own `snippet`, half-open,
non-overlapping, in reading order. They are characters rather than UTF-8 bytes because the client
slicing them is JavaScript, and OCR text carries accents and curly quotes. The field is omitted
when there is nothing to highlight.

The two routes to that field are not equally trustworthy, which is what `kind` is for. `ocrSimple` and
`ocrMatch` come from FTS5, which *knows* which terms it matched: it wraps them in delimiters, and the
server converts those markers into spans and hands back the cleaned text. **Their snippets no longer
contain `[` and `]`** — that markup became the spans. OCR `vector` has no such report — it returns
a ranking and no reason for it — so its spans are estimated after retrieval by matching the query's
words against the OCR text that came back:

| `kind`    | matched because                                     | example                       |
| --------- | --------------------------------------------------- | ----------------------------- |
| `indexed` | FTS5 reported this token as a match — not a guess    | `dre*` ↔ `dresses`            |
| `exact`   | the words are equal, ignoring case and punctuation   | `receipts` ↔ `Receipts`       |
| `stem`    | the words share a crudely stripped stem              | `inspecting` ↔ `inspected`    |
| `prefix`  | a query term (3+ characters) begins the word         | `regul` ↔ `regulations`       |
| `fuzzy`   | one character edit apart, both 5+ characters         | `regulations` ↔ `regulatlons` |

`indexed` is evidence; the four below it are inference, listed in descending confidence. Estimated
spans never filter or reorder a result — the hit was already chosen, semantically, and this is only
an explanation of it. So a correct hit can carry no highlights at all, and `fuzzy` in particular
will occasionally underline a word the query had nothing to do with. Treat the lower rungs as a
hint and style them accordingly. Neighbouring estimated matches merge into one span reported at its
best `kind`, so a two-word query that matches two adjacent words returns one highlight; FTS5's
markers are passed through as it drew them. Query words that carry no information (`the`, `and`,
`of`, and similar) are dropped, so a query made only of those highlights nothing.

One caveat on `indexed`: the delimiters are ordinary `[` and `]`, so a bracketed citation in the OCR
text itself is indistinguishable from a marker and comes back as a span with its brackets removed.
The parse is conservative about the rest — an unpaired delimiter is left in the text as the literal
character it probably is — and the cost of the remaining ambiguity is one wrongly underlined word.

### Errors

Every failure, including an unmatched route, a rejected query string, and a malformed body,
answers with one envelope:

```json
{ "error": { "code": "query_syntax", "message": "fts5: syntax error near \"AND\"" } }
```

`message` is the displayable string; the renderer shows it nearly verbatim. It never contains a
stack trace or a path the caller did not supply. The rule the whole surface follows is that `4xx`
means the caller can fix the request and `5xx` means the server or its databases are broken, so a
user-input problem is never reported as a `500`.

| `code` | HTTP | Cause |
| --- | --- | --- |
| `invalid_request` | 400 | A rejected parameter, body, or path component. `message` names the parameter and why |
| `invalid_root` | 400 | A root that is relative, missing, unreadable, or not a directory |
| `query_syntax` | 400 | SQLite could not parse the search query. `message` is SQLite's own text, folded onto one line |
| `unauthorized` | 401 | Missing or wrong bearer token |
| `not_found` | 404 | No such route in this API version |
| `asset_not_found` | 404 | The catalog has no matching path or ID, or the source file is gone |
| `thumbnail_not_found` | 404 | No current variant exists for the asset |
| `job_not_found` | 404 | The job never existed or is no longer retained |
| `library_not_found` | 404 | No library has the requested ID |
| `method_not_allowed` | 405 | The route exists but not for this method |
| `job_busy` | 409 | Another resource-intensive job is already active |
| `payload_too_large` | 413 | The body exceeded 48 MiB for search, or 8 MiB for other routes |
| `unsupported_media_type` | 415 | A JSON route received a body that was not JSON |
| `shutting_down` | 503 | The server is shutting down and will not start another job |
| `internal_error` | 500 | A genuine server or database failure. The cause is logged, not returned |

`query_syntax` covers every way SQLite can reject the query text: FTS5 syntax (`fts5: syntax error
near "AND"`), an unterminated string, an unknown column prefix (`no such column: ocr`), and an
invalid `regex` pattern. The text is cryptic but honest, and it is the only description of which
part of the query is wrong; the renderer prefixes it with `Query syntax — `. The classification is
structural rather than a message match: the search statements are static, so a SQLite logic error
raised while *executing* one is always attributable to the bound query, while a preparation-stage
failure stays an `internal_error`.

A library that has never been indexed is not an error. `{"total": 0, "results": []}` is the
honest answer, and the gallery already knows from its own catalog whether its folders have been
scanned. The same holds for a library whose text has never been embedded; `GET /v1/text-embeddings` is how
the UI distinguishes that case from "nothing matched".

Vector search has no notion of "no matches". It ranks every embedded row by distance and returns
the nearest ones, so a query with no semantic relation to the library still comes back with hits at
large distances. A caller that wants "close enough" rather than "closest" sets `maxDistance`; the
text modes have no such parameter and reject it.

### Jobs

The server runs one resource-intensive job at a time. Other job types return `409 Conflict` with
`job_busy` while one runs. A `libraryScan` instead returns `202` with status `queued` if the worker
is occupied, so a client can request a scan without first waiting for the slot. At most one scan
waits, and the newest request wins: a request for the queued library returns the same job ID and
merges into it (`scanMode: "full"` wins over `"fast"`; a request without `pendingOnly` wins over
one with it; `force` and `retryFailed` stay enabled if
either request enabled them), while a request for another library cancels the queued scan and
takes its place. A queued scan lives only in server memory; the folders it would have scanned stay
`scanPending` for the next request. Job state is held in memory for
the server's lifetime, with at most 32 recent jobs retained. Electron should retain the returned
`jobId`, or rediscover it with the collection GET, then poll the item GET while the status is
nonterminal. For live UI counters, prefer the SSE item-events route: it sends the current snapshot
immediately, coalesces updates for slow consumers so they receive the newest state, sends a
keepalive every 15 seconds, and closes after delivering a terminal snapshot. The regular item GET
is the reconnect and non-streaming fallback. Every library job snapshot, including entries from
`GET /v1/jobs`, has `libraryId`; jobs not tied to a library omit it.

Job creation uses a stable typed envelope so future OCR inference, CLIP, thumbnail, or maintenance
jobs can share the lifecycle API. Download and compile a PaddleOCR detector and recognizer by
posting explicit Hugging Face model IDs:

```json
{
  "type": "ocrModelLoad",
  "params": {
    "detection": {
      "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx",
      "revision": "main",
      "filename": "inference.onnx",
      "configFilename": "inference.yml"
    },
    "recognition": {
      "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
      "revision": "main",
      "filename": "inference.onnx",
      "configFilename": "inference.yml"
    }
  }
}
```

`modelId` is required for both models and is never selected inside the binary. `revision` is
optional and defaults to the repository's `main` branch. `filename` is optional and defaults to
`inference.onnx`; `configFilename` defaults to `inference.yml`. The configuration supplies the
detector thresholds and normalization, recognition input shape, and character dictionary, so these
are kept with the selected model rather than hardcoded for one repository. Files resolve through
the standard `HF_HOME` cache. A cache hit does not contact the network; a miss reports byte progress
while `hf-hub` downloads the file. The two sessions are then compiled with level-3 graph
optimization for the configured ONNX Runtime execution provider and kept in server memory.

`GET /v1/ocr/models` returns `{}` before a successful load and this shape afterward:

```json
{
  "loaded": {
    "detection": {
      "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx",
      "revision": "main",
      "filename": "inference.onnx"
    },
    "recognition": {
      "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
      "revision": "main",
      "filename": "inference.onnx"
    },
    "executionProvider": "cpu"
  }
}
```

`executionProvider` is the provider the sessions were actually compiled for, not the one requested:
a build that falls back reports `cpu`.

### Library scans

`libraryScan` brings one library up to date: it walks and catalogs the library's folders, removes
confirmed-missing files, and then runs the search indexes the library has enabled.

```json
{
  "type": "libraryScan",
  "params": {
    "libraryId": 3,
    "retryFailed": false
  }
}
```

Only `libraryId` is required. `scanMode` is `"full"` by default. `"fast"` runs a full scan for
each included folder whose last full scan is at least 24 hours old, is pending, has a failed
attempt, or lacks a valid directory snapshot. Other folders stat their known directories and
enumerate only changed ones. A quick check does not advance `lastScanCompletedNs`. The client
requests automatic scans at startup and when the rolling full-scan deadline passes; the server
does not schedule them itself. The library's stored `ocr` and `image` options decide which indexes
run (see the library endpoints), so save a changed choice with `PUT /v1/libraries/<id>` before
scanning. OCR text always gets its text embeddings.

**Folders.** An explicit scan walks every included folder once, recursively; a folder nested inside another
included folder is covered by that walk. The library's excluded folders are skipped with their
whole subtree, as are the default `*/.cache` and `*/.thumb*` folders; a directory link or junction
that resolves into an excluded folder is skipped too. Before walking, a reachable folder stored
with a non-canonical spelling (from an offline import) is respelled canonically. `pendingOnly: true`
walks only folders with `scanPending` set, including ones whose last scan failed; it is what a
client may request when a library is created or edited. It is independent of `scanMode`. A folder that is
offline or unreadable is reported `unavailable` and skipped; the other folders still scan and its
cached entries are never removed. After a complete walk of a folder, catalog entries inside it
(and outside its exclusions) whose files are confirmed missing are removed with their OCR, vectors,
and thumbnails; files are rechecked before each deletion batch. A walk with unreadable entries,
one that reaches its `debugLimit`, or cancellation is incomplete and removes nothing.

**Models load only for work that needs them.** Image vectors are prepared only when cataloged
files lack current vectors, OCR runs only for images without current text, and text embeddings
only for text without current vectors. The OCR pair is downloaded and loaded inside the job, and
only when some image needs recognition and the loaded pair differs from `ocrModels` (the same shape
as an `ocrModelLoad` request). When omitted, the saved pair is used, including after restart or
provider fallback. A scan that finds nothing to do loads no
model at all.

**Options.** `indexVideos` defaults to the saved runtime setting (initially true) and includes video frames in image indexing; videos are
cataloged either way. `force` re-runs OCR for images that already have current text. `retryFailed`
retries sources whose earlier decode failed. `maxDimensions` skips OCR for larger images without
removing earlier text. `debugLimit` caps each folder's walk for development runs; a walk that reaches it with files
left over is incomplete, and one that finishes under it counts as complete.

**Progress.** Snapshots carry `indexStages: {ocr, image, text}` from the library and a `folders`
list in walk order:

```json
"folders": [
  { "path": "D:\\Photos", "scanMode": "full", "state": "completed", "discovered": 812, "cataloged": 12, "failed": 0, "error": null },
  { "path": "E:\\Phone", "state": "unavailable", "discovered": 0, "cataloged": 0, "failed": 0,
    "error": "cannot read folder E:\\Phone: ..." }
]
```

A folder moves `queued` → `scanning` → `scanned` (walked, cataloged, and cleaned up) →
`completed` (every enabled index has caught up). A walk that could not finish ends `incomplete`,
an offline folder `unavailable`, a cleanup failure `failed`, and a cancelled walk `cancelled`; each
of these carries `error`. `discovered`, `cataloged`, and `failed` count that folder's walk; `cataloged` includes files that
were already current, as the job-wide counter does. The
job-wide `phase` and `progress` describe the current step as for every job; the library-wide
steps after the walks (image embedding, OCR, text embedding) are not attributed to one folder.
Only `completed` folders clear `scanPending`; completed full scans record `lastScanCompletedNs`,
while completed automatic directory checks leave it unchanged. Other outcomes record
`scanOutcome` and `scanError` on the library and stay pending. A folder edit made while a scan runs stays pending
after the scan finishes.

Phase order is, per folder, `scanning` → `cataloging` → optional `pruning`, then
`imageEmbedding` (image libraries), optional `downloadingModels` → `loadingModels`, `ocr`, and
`textEmbedding` (OCR libraries), then `finished`. Each model-preparation step appears only when
that model is needed. Catalog rows commit in batches during `cataloging`, so the gallery can show
new files while the scan continues; OCR rows commit every 32 images. Thumbnail generation remains a
separate job.

show these updates.

| Job type | Phase | Fields that move in the phase | `total` in the phase | Honest within-phase UI |
| --- | --- | --- | --- | --- |
| Any | `queued` | None | `null` | Queued state, not a progress bar. |
| `ocrModelLoad` | `downloadingModels` | `downloadedBytes`, `downloadTotalBytes`; `processed` advances after each ONNX inference file resolves; transfer totals accumulate only for network downloads | `null` | Use `downloadedBytes / downloadTotalBytes` only when `downloadTotalBytes > 0`; otherwise show indeterminate download/cache preparation. |
| `ocrModelLoad` | `loadingModels` | `modelsLoaded`, `phaseCompleted` | `2` (detector and recognizer sessions) | `phaseCompleted / total` or “loading N of 2”. |
| `ocrModelLoad` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `libraryScan` | `scanning` | `discovered` (also on the current folder); when a folder's walk ends, `total` and `phaseCompleted` become its discovered count | `null` while walking; the folder's final count when its walk ends | Show “discovered N” per folder while indeterminate. |
| `libraryScan` | `cataloging` | `phaseCompleted`; `cataloged` and `failed` (also on the current folder) | Media discovered in the folder | `phaseCompleted / total`. |
| `libraryScan` | `pruning` | `phaseCompleted`, `pruneCandidates`, `deleted` | Confirmed-missing entries in the folder | `phaseCompleted / total`. Omitted when nothing is missing. |
| `libraryScan` | `imageEmbedding` | `phaseCompleted`, `processed`, `embedded`, `failed` | Pending current image vectors among the scanned files | `phaseCompleted / total`. |
| `libraryScan` | `downloadingModels`, `loadingModels` | As for `ocrModelLoad` | As for `ocrModelLoad` | Shown only when the OCR pair must be loaded. |
| `libraryScan` | `ocr` | `phaseCompleted`, `processed`, `skipped`, `failed`; `indexed` advances when committed OCR chunks save | Scanned files, including those skipped as current or not images | `phaseCompleted / total`. |
| `libraryScan` | `textEmbedding` | `phaseCompleted`, `processed`, `embedded`, `skipped`, `failed` | Pending current `ocrText` vectors in the library | `phaseCompleted / total`. |
| `libraryScan` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state; `folders` holds each folder's outcome. |
| `textEmbed` | `textEmbedding` | `phaseCompleted`, `processed`, `embedded`, `skipped`, `failed` | Pending current `ocrText` vectors in the library after an optional force clear | `phaseCompleted / total`; a zero backlog is immediately complete. |
| `textEmbed` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `imageEmbed` | `imageEmbedding` | `phaseCompleted`, `processed`, `embedded`, `failed` | Pending current CLIP image vectors in the library | `phaseCompleted / total`; a zero backlog is immediately complete. |
| `imageEmbed` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `thumbnailGenerate` | `thumbnails` | `phaseCompleted`, `processed`, `thumbnailsGenerated`, `thumbnailFailures` | Target assets selected by its request | `phaseCompleted / total`; `thumbnailsGenerated` may exceed the numerator because one asset can produce several buckets. |
| `thumbnailGenerate` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `pruneMissing` | `pruning` | `phaseCompleted`, `processed`, `pruneCandidates`, `deleted`, `skipped`, `failed` | Catalog candidates in the library's available folders | `phaseCompleted / total`; `pruneCandidates` is the subset found missing, not the numerator. |
| `pruneMissing` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `libraryPurge` | `pruning` | `phaseCompleted`, `processed`, `deleted`, `failed` | Library assets no other library covers | `phaseCompleted / total`; `deleted` and `failed` are cumulative per-asset outcomes. |
| `libraryPurge` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |

Cancellation after OCR but before or during text embedding leaves a `libraryScan` cancelled without
starting more text embedding batches, and its folders stay pending.

Cancellation is cooperative between model files, scan entries, catalog entries, image decodes,
inference calls, embedding batches, and library-purge assets. `libraryScan` checks it before the
next walk entry and catalog upsert, preserving every prior committed row without starting derived
work. `libraryPurge` checks before beginning its next asset, so cancellation can leave a partial
purge but never begins another asset after it is requested. `hf-hub` 0.4 has no cancellation token
for an in-flight transfer, and ONNX Runtime does not interrupt session compilation or an executing
inference call; an operation already executing is allowed to finish. Decoder output already in the
bounded queue is drained without starting more inference, and an embedding batch already handed to
the model is allowed to finish before the next backlog query sees cancellation. Closing server
stdin requests cancellation of all active jobs before runtime shutdown.

Every HTTP handler opens its own short-lived database connection for the duration of its blocking
work rather than sharing one process-wide connection. Asset lookup, search, and thumbnail requests
therefore remain responsive during background jobs. WAL readers never block, so the remaining
serialization is SQLite's one-writer-per-file rule, which applies per write statement. All stores
use WAL mode and a bounded busy timeout; the asset, OCR, and thumbnail databases remain independent
and are not one cross-database atomic transaction.

### Staleness and explicit pruning

Source freshness is always based on the required `(source_modified_ns, source_size)` fingerprint.
An unchanged file keeps its catalog ID and can skip current derived work. A changed file keeps its
ID, updates the catalog fingerprint, and makes old OCR/thumbnail fingerprints ineligible while new
derived data is generated.

Missing source files remain cataloged by default. This is intentional for removable and temporarily
offline storage. To inspect or delete them, create a typed job:

```json
{
  "type": "pruneMissing",
  "params": {
    "libraryId": 3,
    "dryRun": true
  }
}
```

Only the library's folders that are readable right now are checked; an offline folder is reported
in `errors` and its entries are kept, so an unplugged drive cannot be mistaken for deleted files.
Excluded folders are not checked. `dryRun` defaults to true. A dry run reports `pruneCandidates`
without deleting; the caller must explicitly send `"dryRun": false` to remove rows. Deletion stops
if a checked folder disappears during the job. OCR and thumbnail rows are removed before the
catalog row, and each catalog deletion increments `catalog_meta.revision`.

To remove one library's indexed data, create a `libraryPurge` job:

```json
{
  "type": "libraryPurge",
  "params": {
    "libraryId": 3
  }
}
```

The purge selects the library's catalog assets (its folders minus its exclusions) and keeps every
asset another library also covers, so shared files stay searchable there. Coverage is rechecked
before each deletion batch, so a folder another library gains while the purge runs keeps its
data. Original files are never
touched, so the library's folders need not be available. The library definition itself is left in
place; delete it with `DELETE /v1/libraries/<id>` afterwards. For each asset, OCR text/text vectors and CLIP image vectors are
deleted first, then all thumbnails, and finally its catalog row. Each completed catalog deletion advances
`catalog_meta.revision`. The work is not cross-database atomic: a deletion failure leaves the
catalog row in place, and cancellation may leave already completed assets removed. Re-running the
purge is safe and resumes from the rows that remain. Deleting a library without purging keeps all
of its indexed data.

After every resource-intensive job, the server removes derived rows whose catalog IDs no longer
exist, runs bounded incremental vacuum and `PRAGMA optimize` on all four stores, and requests a
passive WAL checkpoint. This common epilogue is best-effort: a maintenance failure is logged but
does not replace the job's own result. `libraryPurge` itself still deletes derived rows before the
catalog row, so interruption leaves a retryable catalog record rather than creating a new orphan.

## Verification

From the `nicegal-server` directory:

```sh
dev.cmd test --locked --workspace
dev.cmd build --locked -p nicegal-server
node scripts/rpc-smoke.mjs
node scripts/gallery-api-playground.mjs [SERVER_EXE] --root=../testdata --exclude=video [--ocr] [--no-image] [--keep]
```

The playground is a manual walkthrough rather than a framework integration test. It creates a
library over `--root` with the given exclusions, scans it (printing each folder's state and which
models loaded), checks that no excluded file reached the catalog, runs an image search, rescans to
show that nothing is reloaded or re-indexed, then removes the exclusions and runs a `pendingOnly`
scan that picks up the revealed files. `--keep` preserves its temporary databases for inspection.

The Windows build packages ONNX Runtime libraries; model weights are downloaded separately
when a model preparation or indexing request requires them.

## Image model selection

`GET /v1/runtime` includes `imageModel: {activeModel, selectedModel, restartRequired,
models}`. Top-level `restartRequired` covers both the provider and model. Each catalog entry contains `id`, `name`, `dimensions`, `license`,
`url`, and `available`. Published exports download normally; local development overrides can be
placed under `NICEGAL_LOCAL_MODELS_DIR`.

`PUT /v1/runtime` accepts `{"imageModel":"facebook/metaclip-2-worldwide-b32"}`
and returns runtime status. `executionProvider`, `imageModel`, `indexVideos`, and `ocrModels`
are optional, but at least one must be supplied. Unknown or unavailable image models are rejected,
as are provider or image-model changes while an indexing job is active. The caller restarts the backend to
activate the selection. Provider and image-model selections persist together in the runtime
configuration, so a combined update either saves both selections or neither. Existing
`image-model.json` files are read when the runtime record has no `imageModel`; the next
successful settings update incorporates that selection into the runtime record. The
legacy file is retained but no longer consulted once `imageModel` is present. Older
server versions that reject unknown runtime settings fields cannot read the migrated
record. `--image-model` / `NICEGAL_IMAGE_MODEL` overrides only the active
model for that launch, useful for sequential evaluation scripts.

All image models now use byte-intermediate convolution for downscaling, retaining
their configured filter, crop, and normalization. Upscaling keeps float intermediates.
Existing vectors remain valid and are not automatically rebuilt.

### Selective indexing

A library's `ocr` and `image` options select its indexes. `image` alone catalogs files and runs
only image embeddings, without loading PaddleOCR or BGE. `ocr` alone runs OCR and its text
embeddings, without loading the image model or its text tower. With both off, a scan only
catalogs and cleans up. Existing indexed results are retained when an index is switched off;
cleanup still removes missing files after a complete walk.

`libraryScan` snapshots include `indexStages: {ocr, image, text}` so progress displays only stages
that can run, including when a client attaches to an existing job.
