# Internal API

`nicegal-server` is the local RPC host for the controlling desktop process. It binds an
ephemeral IPv4 loopback port and prints one JSON readiness line containing `apiVersion` and
`endpoint`. Every request requires the token from `NICEGAL_RPC_TOKEN` as a bearer token.
Closing the server's stdin starts graceful shutdown.

## Frontend integration at a glance

Electron owns the configured database paths. Rust owns catalog and metadata policy:

| Frontend operation | Interface |
| --- | --- |
| Enumerate/sort the library | `GET /v1/catalog?root=...&timeline=modified\|capture` |
| Count a library | `GET /v1/catalog/count?root=...` |
| Inspect a photo | `GET /v1/catalog/metadata?assetId=...` |
| Poll for catalog changes | `GET /v1/catalog/revision` |
| Load thumbnail blobs | Read-only SQLite on `thumbnails.db`, normally behind `thumb://` |
| Search OCR text or indexed images (one mode) | `GET /v1/search` |
| Search several ways at once | `POST /v1/search` |
| Underline what matched in an OCR result | `highlights`, see [Highlights](#highlights) |
| Check vector-search coverage | `GET /v1/text-embeddings` |
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
`ocrIndex` job with `embed:true` prepares all three. Already prepared sessions are reused.
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

All three files use WAL mode. Their separation and the stable columns explicitly documented below
are part of the desktop read contract. Any table or column shape change requires a `user_version`
bump; readers should reject versions they do not support.

## Asset catalog schema (version 5)

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

`media_kind` is currently `image` or `video`. Image dimensions, GIF frame count, and duration are
best-effort. `exif_taken_ns` is Unix nanoseconds parsed from `DateTimeOriginal`, including
`SubSecTimeOriginal` and `OffsetTimeOriginal` when present; an absent EXIF offset deterministically
means UTC. Probe failure leaves optional columns null and does not remove the asset. Video paths
and basic format classification are cataloged; video probing and poster generation are not supported. GIF posters are static PNGs produced from the first decoded/composited frame; original
animated GIF or video bytes never belong in the thumbnail database.

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

## OCR schema (version 8)

OCR is a derived store. It receives `asset_id` from the catalog and never allocates gallery IDs.
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

The root, the exclusion, and the time range are one value (`db::SearchFilters`) shared by every
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

### CLIP image index (version 1)

CLIP image ingestion and search are independent of OCR and use a model-specific database. The
initial image model is `Qdrant/clip-ViT-B-32`, stored in `qdrant-clip-vit-b-32.db`; its vectors are
normalized and 512-wide. Text requests to `type=image` use its paired FastEmbed CLIP text encoder,
`Qdrant/clip-ViT-B-32-text`, in that same 512-dimensional space. It is deliberately loaded on CPU:
an image query is one short text forward pass, while the configured accelerator remains available
for OCR and image-indexing work.

`GET /v1/search` keeps `type=image` text-only through `q`. `POST /v1/search` additionally supports
signed asset-ID and text components; see [Composite image queries](#composite-image-queries).
Precomputed binary vectors and a general image-to-vector upload endpoint are not accepted.

```sql
CREATE TABLE image_embedding_state(
    asset_id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL
);
CREATE VIRTUAL TABLE image_embeddings USING vec0(
    asset_id INTEGER PRIMARY KEY,
    embedding FLOAT[512] distance_metric=cosine
);
```

The state fingerprint comes directly from the asset catalog. A small decode pool feeds one bounded
batch of decoded RGB images ahead of a single FastEmbed/ONNX inference lane. FastEmbed applies the
CLIP preprocessor and submits `[B, 3, 224, 224]`; full batches use `B = 8`, while the final partial
batch may be smaller. ONNX Runtime owns model execution threading.

## Thumbnail schema (version 3)

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
Every thumbnail column shown above is a stable direct-read surface.

Generator version 1 creates variants lazily by default. An OCR index job catalogs and OCRs media
without generating thumbnails; the gallery calls the synchronous ensure endpoint for its visible
image IDs. Full-library generation remains available through the thumbnail backfill job. Still images are
decoded once per asset for all missing buckets; opaque results are JPEG quality 85 and alpha-bearing
results are PNG. GIFs remain static PNG first-frame posters. A failed thumbnail decode is recorded
on the job but does not discard the catalog row or prevent OCR. Current source/version variants are
skipped, source changes overwrite the same generator key, and old generator versions remain until a
successful backfill for that library explicitly requests `sweepStale`.

## Local HTTP API

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
      "restartRequired": true
    },
    "ocrModelsLoaded": false
  }
  ```
  `activeExecutionProvider` is the provider requested in this process and
  `activeRuntimeDistribution` identifies the ONNX Runtime DLL set actually loaded for it;
  `onnxRuntimeBuildInfo` is ONNX Runtime's own release/commit/build-flags diagnostic string;
  `configuredExecutionProvider` is the persisted choice for the next launch. They differ after a
  setting changes and while a command-line provider override is in effect. `ocrModelsLoaded`
  avoids a follow-up request when the client only needs to decide whether OCR indexing is available;
  use `GET /v1/ocr/models` for loaded model identities and their actual session provider.
- `GET /v1/runtime` returns the `runtime` object shown above. `PUT /v1/runtime` accepts exactly
  `{"executionProvider":"cpu"|"directml"|"openvino"}` and persists that selection. It returns the
  same runtime object with `restartRequired: true` when a restart is needed. It never tries to
  unload or replace ONNX Runtime in the current process.
- `GET /v1/catalog?root=<absolute-root>&timeline=modified|capture` returns the desktop gallery
  array. `timeline` defaults to modified; capture falls back to modified when EXIF time is absent.
  Both sorts descend with asset ID as the descending tie-breaker. Root matching uses literal path
  boundaries (not SQL wildcards); trailing separators are accepted. Reads do not stat the root,
  so offline libraries still show saved rows. Each row has `id`, `path`, `displayName`, `extension`,
  `modifiedNs`, `createdNs`, `captureNs`, `sourceSize`, `mediaKind`, `mediaFormat`, `width`, `height`,
  `animated`, `frameCount`, `durationMs`. IDs, nanosecond timestamps and byte sizes are decimal
  strings; unavailable timestamps/dimensions/frame count/duration are null.
- `GET /v1/catalog/count?root=<absolute-root>` returns a JSON integer without materializing rows.
  `GET /v1/catalog/revision` returns a decimal-string revision.
- `GET /v1/catalog/metadata?assetId=<positive-id>` returns `{asset, file, ocrState, textState,
  imageIndexed, decodeFailed}`. `asset` has the gallery shape above. `file` contains `sourceState`
  (`current`, `changed`, `missing`, `unavailable`), Windows `attributes`, selected EXIF fields as
  `{label,value}` pairs, and a nullable diagnostic `error`. Detailed file probing happens on demand;
  changed sources omit EXIF instead of mixing live camera data with saved catalog dimensions.
  Unsupported/no EXIF is an empty list, not a failure. Field text is bounded to 4096 characters.
  `ocrState` is `indexed|stale|notIndexed`; `textState` is `embedded|pending|noText|notIndexed` and
  matches the persisted OCR fingerprint/embedding state, excluding marked-for-deletion OCR rows.
  `imageIndexed` means a CLIP vector matches the catalog fingerprint in the active model's store.
  `decodeFailed` means this exact catalog fingerprint has a recorded decode failure. Other past
  job errors are not a durable per-asset error history. Index status describes the catalog snapshot;
  `file.sourceState` separately reports source changes since the last scan. No model is loaded.
  Unknown IDs return `asset_not_found`. No database schema or migration is added.
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
- `GET /v1/search?q=<query>&type=vector|image|simple|match|glob|regex&root=<absolute-path>&limit=<n>`
  runs **one** mode and returns
  `{total, results: [{assetId, snippet, rank, distance?, highlights?}]}`. `total`
  counts all matches before the requested result cap; the default cap is 100,000 and the maximum is
  250,000. `rank` is the 1-based position in this mode's own ranking. `distance` is present only for
  `type=vector` and `type=image` and is a cosine distance (0 identical, 1 orthogonal, 2 opposite)
  in that mode's distinct vector space. `type=vector` searches OCR-text vectors; `type=image`
  embeds `q` with the CPU `Qdrant/clip-ViT-B-32-text` encoder and searches current
  `Qdrant/clip-ViT-B-32` image vectors. Image hits have an empty `snippet` and no `highlights`;
  resolve their image metadata through `POST /v1/assets`. Simple and match text modes retain FTS
  rank order; glob ranks assets by matching-token count, then recency. For
  `type=glob`, `q` is matched case-insensitively against each complete OCR word token: `*` matches
  any number of characters and `?` matches one. Thus `dre*` is a word-prefix search (it does not
  match `andrew`), while `*cat*` finds a word containing `cat`. The backend scans the FTS5 word
  vocabulary and follows matching term postings rather than scanning every OCR text blob; glob
  snippets are the first 512 characters of OCR text, so they need not contain the matching token at
  all — see [Highlights](#highlights) for what is claimed about a snippet's contents and what is
  only estimated. `q` is capped at 4096 bytes. **`type` now defaults to `vector`**; the text
  modes and image mode are opt-in. `maxDistance=<0..2>` bounds either vector neighbourhood and is
  rejected for the text modes. Image mode applies `root`, `exclude`, and time filters to the asset
  catalog and excludes an embedding as soon as its source fingerprint is stale. `before`, `after`,
  and `timeline` bound the search in time; see
  [Time filtering](#time-filtering). A query SQLite cannot parse is `400 query_syntax` carrying
  SQLite's own message, and an unusable root is `400 invalid_root`; see [Errors](#errors).
  OCR-mode `highlights` marks where in `snippet` this query matched; image hits omit it because
  they have no OCR snippet. See [Highlights](#highlights).
- `POST /v1/search` runs several modes and optionally fuses them. OCR modes share one OCR-store
  snapshot; `type=image` reads its independent CLIP index on a separate read-only connection. This
  is the route for a client that searches more than one way at once; see
  [Combined search](#combined-search)
- `GET /v1/text-embeddings?root=<absolute-path>` reports what vector search can currently answer for:
  `{embedder: {model, dimensions}, stored: {model, dimensions} | null, indexed, embedded, pending}`.
  `embedder` is what this build produces, `stored` is what the database holds (null before the first
  backfill), and a difference between them means the next text embed job discards and rebuilds. This is
  how the UI tells "nothing matched" apart from "nothing has been embedded"
- `POST /v1/text-embeddings/generate` starts the text embedding backfill job. Its JSON body is
  `{root, force:false, batchSize?}`. `force` re-embeds rows that already have a current vector;
  `batchSize` is 1 to 512 and is additionally clamped to what the backend accepts, so omitting it
  is normally right. The same job can be created through `POST /v1/jobs` with type `textEmbed`
- `POST /v1/jobs` starts a typed background job and returns `202 Accepted`. Types are
  `modelPrepare`, `ocrModelLoad`, `ocrIndex`, `catalogSync`, `thumbnailGenerate`, `textEmbed`, `imageEmbed`,
  `pruneMissing`, and `libraryPurge`
- `GET /v1/jobs` lists the active and retained recent jobs
- `GET /v1/jobs/<job-id>` returns one job's current state and progress
- `GET /v1/jobs/<job-id>/events` streams `snapshot` server-sent events whenever job state changes
- `DELETE /v1/jobs/<job-id>` requests cooperative cancellation
- `POST /v1/thumbnails/generate` starts a thumbnail backfill job. Its JSON body is
  `{root, buckets:[1024], force:false, sweepStale:false, timeline:"modified",
  range:{fromNs,toNs}}`; `root` is required and must name an existing absolute directory.
  It is canonicalized, then only catalog assets below that root are selected before applying
  `timeline`, `range`, and `buckets`. `timeline` is `modified` or `capture`; capture uses
  `COALESCE(exif_taken_ns, source_modified_ns)`. Range bounds are optional decimal strings
  containing Unix nanoseconds, with an inclusive `fromNs` and exclusive `toNs`. `sweepStale`
  still requires an unbounded backfill of every bucket, and removes stale generators only for
  the selected root's assets. The same job can be created through `POST /v1/jobs` with type
  `thumbnailGenerate`
- `POST /v1/thumbnails` synchronously ensures variants for a visible image set. Its body is
  `{assetIds:[...], requiredSize:<physical-pixels>}`. `requiredSize` is 1 through 1024; it selects
  the smallest adequate fixed bucket. The response is `200` only after every requested current
  variant has been committed to `thumbnails.db`, and returns `{assetIds, requiredSize, sizeBucket,
  generatorVersion}`. Clients then read the bytes directly from SQLite.
- `PUT /v1/thumbnails?assetId=<id>&sizeBucket=<bucket>&generatorVersion=<version>&width=<actual-width>&height=<actual-height>&encoding=<mime-type>` with static encoded bytes. The server fully decodes the body, requires its decoded dimensions to equal `width` and `height`, and limits both edges to `sizeBucket`.
- `GET /v1/thumbnails?assetId=<id>&requestedSize=<physical-pixels>&generatorVersion=<version>`
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
  "root": "C:/gallery",
  "exclude": "C:/gallery/tmp",
  "limit": 100,
  "queries": [
    { "key": "semantic", "type": "vector", "q": "coffee receipt", "maxDistance": 1.2, "weight": 2 },
    { "key": "visual", "type": "image", "maxDistance": 0.8,
      "imageQuery": { "components": [
        { "assetId": 101, "weight": 1 },
        { "assetId": 202, "weight": -1 },
        { "text": "a coffee on a marble table", "weight": 2 }
      ] } },
    { "key": "literal",  "type": "simple", "q": "coffee" },
    { "key": "files",    "type": "glob",   "q": "*invoice*", "limit": 50,
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
  "imageModel": { "model": "Qdrant/clip-ViT-B-32", "dimensions": 512 },
  "queries": [
    { "key": "semantic", "type": "vector", "total": 812,
      "results": [{ "assetId": 41, "snippet": "...", "rank": 1, "distance": 0.21,
                    "highlights": [{ "start": 12, "end": 18, "kind": "exact" }] }] },
    { "key": "visual", "type": "image", "total": 24,
      "results": [{ "assetId": 52, "snippet": "", "rank": 1, "distance": 0.18 }] },
    { "key": "literal", "type": "simple", "total": 12,
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
`imageModel` is always the restart-scoped image encoder whose 512-wide CLIP vectors `type=image`
searches. Each block's `total` is that mode's match count before its `limit`, exactly as the
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
  "root": "C:/gallery",
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
rather than silently changing its meaning. Reference assets do **not** need to sit under `root`:
`root`, `exclude`, and timeline filters constrain returned results, not the examples used to
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
`simple`, `match`, `glob`, and `regex` still require `q` and reject `imageQuery`. `GET /v1/search`
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

A root that has never been indexed, a mode with no matches, and a library with nothing embedded are
all empty results rather than errors. A query SQLite cannot parse fails the whole request with
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

The two routes to that field are not equally trustworthy, which is what `kind` is for. `simple` and
`match` come from FTS5, which *knows* which terms it matched: it wraps them in delimiters, and the
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
| `method_not_allowed` | 405 | The route exists but not for this method |
| `job_busy` | 409 | Another resource-intensive job is already active |
| `ocr_models_not_loaded` | 409 | `ocrIndex` was requested before a detector and recognizer were loaded |
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

A search root that exists but has never been indexed is not an error. `{"total": 0, "results": []}`
is the honest answer, and the gallery already knows from its own catalog whether that root has been
scanned. The same holds for a root whose text has never been embedded; `GET /v1/text-embeddings` is how
the UI distinguishes that case from "nothing matched".

Vector search has no notion of "no matches". It ranks every embedded row by distance and returns
the nearest ones, so a query with no semantic relation to the library still comes back with hits at
large distances. A caller that wants "close enough" rather than "closest" sets `maxDistance`; the
text modes have no such parameter and reject it.

### Jobs

The server currently runs one resource-intensive job at a time, including `catalogSync`. Starting
another returns `409 Conflict` with the error code `job_busy`. Job state is held in memory for the
server's lifetime, with at most 32 recent jobs retained. Electron should retain the returned
`jobId`, or rediscover it with the collection GET, then poll the item GET while the status is
nonterminal. For live UI counters, prefer the SSE item-events route: it sends the current snapshot
immediately, coalesces updates for slow consumers so they receive the newest state, sends a
keepalive every 15 seconds, and closes after delivering a terminal snapshot. The regular item GET
is the reconnect and non-streaming fallback.

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

To scan a library into the primary catalog for browsing before any OCR preparation, create this
model-free job:

```json
{
  "type": "catalogSync",
  "params": {
    "root": "C:/absolute/gallery",
    "scan": {
      "recursive": true,
      "exclude": ["*/.cache", "*/.thumb*"]
    }
  }
}
```

`root` is required, must name an existing absolute directory, and is canonicalized before scanning.
`scan` is optional; `recursive` defaults to true and `exclude` defaults to the two patterns shown.
`debugLimit`, when supplied, must be a positive integer and caps the number of catalogable media
discovered for a development run. Production callers omit it to scan the complete root. It walks
the production scanner and then upserts catalogable media with the normal fingerprint and
canonical-path rules. Each changed row and its catalog_meta.revision increment commit
independently, so the existing direct SQLite readers can see a partial, internally consistent
catalog while the job is still running. Existing paths retain
their stable `assetId`; unchanged fingerprints are no-ops and do not advance the revision.

`catalogSync` never loads OCR models or generates OCR text, embeddings, or thumbnails. A completed
scan now automatically removes confirmed-deleted catalog entries and their OCR/text-vector,
CLIP-vector, and thumbnail records. Unavailable roots, traversal errors,
cancelled scans, and any `debugLimit` preserve existing entries. Recursive and exclusion scope also
apply to reconciliation; an excluded folder's descendants are retained. Filesystem checks finish
before deletion starts, with root and file availability checked again before each bounded batch.
Derived stores are deleted first, so an interrupted cross-database deletion retains the catalog
entry for the next update to finish. Its worker phase order is
`scanning` → `cataloging` → optional `pruning` → `finished`; it shares ordinary cooperative cancellation and the single
active-job slot with every other job. Cancellation leaves already committed catalog rows visible and
stops before another scan entry or catalog upsert.

After the pair is loaded, scan and OCR a gallery root with:

```json
{
  "type": "ocrIndex",
  "params": {
    "root": "C:/absolute/gallery",
    "embed": true,
    "scan": {
      "recursive": true,
      "exclude": ["*/.cache", "*/.thumb*"],
      "force": false,
      "cleanup": false,
      "maxDimensions": { "width": 12000, "height": 12000 }
    }
  }
}
```

`embed` is optional and defaults to `true`. After a successful, uncancelled OCR run (including its
existing cleanup step and the same automatic missing-file reconciliation as `catalogSync`), the same
job first incrementally embeds pending CLIP image vectors and then
pending `ocrText` vectors under the root. Both use fingerprint-aware backlogs and do not
force-rebuild current vectors. Set `"embed": false` to finish after OCR. Embedding item failures
are retained in this job's `errors` and increase its `failed` counter; they remain pending for a
later retry. This does **not** generate thumbnails.

`scan` and all its fields are optional. `recursive` defaults to true and the two exclusions shown
are the defaults. `force` defaults to false. When true, it re-runs OCR for every otherwise eligible
image under the requested root, including unchanged OCR rows and unchanged source revisions whose
image decode previously failed. It still honors `recursive`, `exclude`, and `maxDimensions`, and it
does not force-rebuild current image or OCR-text vectors. A normal run already retries OCR inference
failures, because they leave no OCR row; cached decode failures retry when their source fingerprint
changes or when `force` is true. `cleanup` is the legacy OCR-only sweep and applies only to complete
recursive scans with no exclusions; automatic confirmed-missing cleanup does not require this flag.
Sources removed between discovery and OCR/CLIP decoding count as skipped rather than failed and do
not acquire cached decode failures. Permission, malformed-media, and inference errors remain visible.
`maxDimensions` is optional and skips larger images without removing an earlier good OCR
result. A request made before model preparation returns `409 ocr_models_not_loaded`.

Discovery and cataloging finish before OCR starts. Image decoding is reported as part of the OCR
phase, decoder completion may differ from filesystem order, recognition is batched, and OCR rows
are committed every 32 images. With the default `embed: true`, phase order is `scanning` →
`cataloging` → `ocr` → `cleanup` → optional `pruning` → `imageEmbedding` → `textEmbedding` → `finished`; `cleanup` is
retained in its existing place, before image and text embedding. With `embed: false`, it is `scanning` →
`cataloging` → `ocr` → `cleanup` → optional `pruning` → `finished`. Thumbnail generation remains a separate job.

Start or retry CLIP image ingestion without rerunning OCR through the generic job endpoint:

```json
{
  "type": "imageEmbed",
  "params": {
    "root": "C:/absolute/gallery",
    "force": false
  }
}
```

`force` defaults to false and overwrites current vectors for assets under the requested root.

Start an OCR-text embedding backfill through its feature endpoint:

```http
POST /v1/text-embeddings/generate
Content-Type: application/json

{
  "root": "C:/absolute/gallery",
  "force": false
}
```

The response is `202 Accepted` with a regular job snapshot whose `type` is `textEmbed`. The equivalent
generic job request is:

```json
{
  "type": "textEmbed",
  "params": {
    "root": "C:/absolute/gallery",
    "force": false,
    "batchSize": 128
  }
}
```

`batchSize` is optional and normally should be omitted so the configured backend chooses its
preferred maximum. If supplied, it must be between 1 and 512 and is clamped to the backend limit.
`force` defaults to false; true discards current OCR-text vectors before rebuilding them. A model
identifier or dimension change also replaces the stored embedding space automatically. Use
`GET /v1/text-embeddings?root=<absolute-path>` before or after the job to inspect `indexed`, `embedded`,
and `pending` coverage.

The create and item endpoints return this shape:

```json
{
  "jobId": "1",
  "type": "ocrModelLoad",
  "status": "running",
  "phase": "downloadingModels",
  "progress": {
    "discovered": 0,
    "total": null,
    "phaseCompleted": 0,
    "processed": 0,
    "cataloged": 0,
    "thumbnailsGenerated": 0,
    "thumbnailFailures": 0,
    "pruneCandidates": 0,
    "embedded": 0,
    "indexed": 0,
    "skipped": 0,
    "failed": 0,
    "deleted": 0,
    "downloadedBytes": 7340032,
    "downloadTotalBytes": 9880512,
    "modelsLoaded": 0
  },
  "errors": []
}
```

Statuses are `queued`, `running`, `cancelling`, `cancelled`, `completed`, and `failed`. Terminal
statuses are `cancelled`, `completed`, and `failed`. `total` is a `u64` and **phase-local**: it is
reset to `null` whenever a worker phase changes, then becomes a known denominator only for phases
that can count their work. `phaseCompleted`, a new `u64`, is also reset at every worker phase
change and is the only generic numerator for `total`. The terminal `finished` snapshot retains its
last worker-phase values. A UI must use `phaseCompleted / total` when the table says a denominator
is known, rather than dividing lifetime counters such as `processed`, `indexed`, or `embedded` by
it. Those lifetime counters never reset and remain useful for labels and final summaries.

`discovered` is phase-local only while scanning: it is the count of catalogable media found so far.
`processed`, `cataloged`, `thumbnailsGenerated`, `thumbnailFailures`, `pruneCandidates`, `embedded`,
`indexed`, `skipped`, `failed`, and `deleted` are lifetime counters for the job. `errors` retains up
to 100 item failures. A failed job also contains an `error` string.

`itemsPerSecond` measures completed work per second since the current phase began and resets
on phase changes. During OCR it excludes that phase's skipped assets, so a resumed job's
previously indexed images advance progress without inflating OCR speed. It is `null` until
non-skipped OCR work completes. The progress bar still includes skipped assets.

| Job type | Phase | Fields that move in the phase | `total` in the phase | Honest within-phase UI |
| --- | --- | --- | --- | --- |
| Any | `queued` | None | `null` | Queued state, not a progress bar. |
| `ocrModelLoad` | `downloadingModels` | `downloadedBytes`, `downloadTotalBytes`; `processed` advances after each ONNX inference file resolves; transfer totals accumulate only for network downloads | `null` | Use `downloadedBytes / downloadTotalBytes` only when `downloadTotalBytes > 0`; otherwise show indeterminate download/cache preparation. |
| `ocrModelLoad` | `loadingModels` | `modelsLoaded`, `phaseCompleted` | `2` (detector and recognizer sessions) | `phaseCompleted / total` or “loading N of 2”. |
| `ocrModelLoad` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `catalogSync` | `scanning` | `discovered`; when walking ends, `total` and `phaseCompleted` become the final discovered count | `null` while walking; final count of catalogable media when discovery completes | Show “discovered N” while indeterminate; the final scan snapshot is complete. |
| `catalogSync` | `cataloging` | `phaseCompleted`; `cataloged` for successful upserts and `failed` for failed metadata/upserts | Count of media discovered by scanning | `phaseCompleted / total`. |
| `catalogSync` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `ocrIndex` | `scanning` | `discovered`; when walking ends, `total` and `phaseCompleted` become the final discovered count | `null` while walking; final count of catalogable media when discovery completes | Show “discovered N” while indeterminate; the final scan snapshot is complete. |
| `ocrIndex` | `cataloging` | `phaseCompleted`; `cataloged` for successful upserts and `failed` for failed metadata/upserts | Count of media discovered by scanning | `phaseCompleted / total`. |
| `ocrIndex` | `ocr` | `phaseCompleted`, `processed`, `skipped`, `failed`; `indexed` advances when committed OCR chunks save | Count of successfully cataloged media, including non-OCR images that are skipped | `phaseCompleted / total`; display cumulative `indexed` separately if useful. |
| `ocrIndex` | `cleanup` | `phaseCompleted`, `deleted` | `1` (the one cleanup sweep, even if cleanup was disabled and deletes zero rows) | A one-step completion indicator, or simply “finalizing”. |
| `catalogSync` or `ocrIndex` | `pruning` | `phaseCompleted`, `deleted` | Confirmed-missing entries in the completed scan scope | `phaseCompleted / total`; `deleted` supplies the compact removal summary. Phase is omitted when nothing is missing. |
| `ocrIndex` with `embed: true` | `imageEmbedding` | `phaseCompleted`, `processed`, `embedded`, `failed` | Pending current CLIP image vectors under the root | `phaseCompleted / total`; `embedded` remains cumulative for the job. |
| `ocrIndex` with `embed: true` | `textEmbedding` | `phaseCompleted`, `processed`, `embedded`, `skipped`, `failed` | Pending current `ocrText` vectors under the root | `phaseCompleted / total`; `embedded` remains cumulative for the job. |
| `ocrIndex` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `textEmbed` | `textEmbedding` | `phaseCompleted`, `processed`, `embedded`, `skipped`, `failed` | Pending current `ocrText` vectors under the requested root after an optional force clear | `phaseCompleted / total`; a zero backlog is immediately complete. |
| `textEmbed` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `imageEmbed` | `imageEmbedding` | `phaseCompleted`, `processed`, `embedded`, `failed` | Pending current CLIP image vectors under the requested root | `phaseCompleted / total`; a zero backlog is immediately complete. |
| `imageEmbed` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `thumbnailGenerate` | `thumbnails` | `phaseCompleted`, `processed`, `thumbnailsGenerated`, `thumbnailFailures` | Target assets selected by its request | `phaseCompleted / total`; `thumbnailsGenerated` may exceed the numerator because one asset can produce several buckets. |
| `thumbnailGenerate` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `pruneMissing` | `pruning` | `phaseCompleted`, `processed`, `pruneCandidates`, `deleted`, `skipped`, `failed` | Catalog candidates under the requested root | `phaseCompleted / total`; `pruneCandidates` is the subset found missing, not the numerator. |
| `pruneMissing` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |
| `libraryPurge` | `pruning` | `phaseCompleted`, `processed`, `deleted`, `failed` | Catalog assets under the requested root | `phaseCompleted / total`; `deleted` and `failed` are cumulative per-asset outcomes. |
| `libraryPurge` | `finished` | None | Last value is retained until the terminal snapshot | Terminal state, no progress bar. |

An `ocrIndex` does not enter `textEmbedding` when `embed` is false, and cancellation after OCR but
before or during text embedding leaves the job cancelled without starting more text embedding batches.

Cancellation is cooperative between model files, scan entries, catalog entries, image decodes,
inference calls, embedding batches, and library-purge assets. `catalogSync` checks it before the
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
    "root": "E:/Photos",
    "dryRun": true
  }
}
```

The root must be an absolute, currently existing directory, so an unplugged drive cannot be
mistaken for an empty library. `dryRun` defaults to true. A dry run reports `pruneCandidates`
without deleting; the caller must explicitly send `"dryRun": false` to remove rows. Deletion is
scoped to catalog paths below the canonical root. OCR and thumbnail rows are removed before the
catalog row, and each catalog deletion increments `catalog_meta.revision`.

To destructively remove every cataloged asset for one library, create a `libraryPurge` job:

```json
{
  "type": "libraryPurge",
  "params": {
    "root": "E:/Photos"
  }
}
```

`root` is required, absolute, currently existing, and canonicalized before the assets are selected.
Only catalog paths below that canonical directory are eligible; paths under same-prefix sibling
directories are never selected. For each asset, OCR text/text vectors and CLIP image vectors are
deleted first, then all thumbnails, and finally its catalog row. Each completed catalog deletion advances
`catalog_meta.revision`. The work is not cross-database atomic: a deletion failure leaves the
catalog row in place, and cancellation may leave already completed assets removed. Re-running the
same root is safe and resumes from the rows that remain. Unregistering a library without deleting
data is renderer-local and does not create a backend job.

Removing orphaned derived rows whose catalog IDs no longer exist and explicit SQLite
checkpoint/optimization work are outside `libraryPurge`.

## Verification

From the `nicegal-server` directory:

```sh
dev.cmd test --locked --workspace
dev.cmd build --locked -p nicegal-server
node scripts/rpc-smoke.mjs
node scripts/gallery-api-playground.mjs [TESTDATA_ROOT] [SERVER_EXE] [--keep]
```

The playground is a temporary manual validation tool rather than a framework integration test. It
indexes `../testdata/pink` by default, consumes live job SSE, reads both catalog timeline orders
through `node:sqlite`, runs modified/capture thumbnail ranges, exercises search and single-asset
lookup, and summarizes thumbnail buckets. It never invokes pruning. `--keep` preserves its
temporary databases for inspection.

The Windows build packages ONNX Runtime libraries; model weights are downloaded separately
when a model preparation or indexing request requires them.
