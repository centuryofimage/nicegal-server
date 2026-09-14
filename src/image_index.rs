//! Model-specific CLIP image-vector storage and ingestion.
//!
//! Each model owns a database file, so incompatible coordinate spaces never share a table. The
//! database tracks source fingerprints directly from the asset catalog; OCR success or failure has
//! no bearing on image-vector coverage.

use std::collections::HashSet;
use std::ops::Deref;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use crossbeam_channel::{Receiver, Sender, bounded};
use image::RgbImage;
use rusqlite::types::Value;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use tracing::{debug, error, info, instrument};

use crate::assets::{Asset, AssetCatalog, MediaKind, SourceFingerprint};
use crate::db::{
    FilterScope, SearchFilters, UNBOUNDED_DISTANCE, bind_named, configure_vector_reads,
    register_glob, register_vector_extension, vector_to_blob,
};
use crate::embedding::ImageEmbedder;
use crate::index::{
    IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta, aborted, send_unless_aborted,
};
use crate::schema::{check_schema_read_only, open_schema};

const SCHEMA_VERSION: i32 = 1;
const SCHEMA_LABEL: &str = "image embedding database";
const MAX_EMBEDDING_DIMENSIONS: usize = 65_536;
const MAX_DECODE_WORKERS: usize = 4;

pub struct ImageIndexDb {
    conn: Connection,
    dimensions: usize,
}

/// A consistent read view of one image index and its attached asset catalog.
///
/// Image-query components and their neighbour search must share this transaction: otherwise a
/// component could be current when read, become stale while the request is running, and still
/// affect a search that correctly excludes it. Dropping an uncommitted snapshot rolls it back.
pub struct ImageIndexReadSnapshot<'db> {
    db: &'db ImageIndexDb,
    active: bool,
}

impl Deref for ImageIndexReadSnapshot<'_> {
    type Target = ImageIndexDb;

    fn deref(&self) -> &Self::Target {
        self.db
    }
}

impl ImageIndexReadSnapshot<'_> {
    /// Cleanly complete this read-only transaction. Errors and unwinding roll back through
    /// [`Drop`] instead.
    pub fn commit(&mut self) -> Result<()> {
        self.db
            .conn
            .execute_batch("COMMIT")
            .context("committing image index read snapshot")?;
        self.active = false;
        Ok(())
    }
}

impl Drop for ImageIndexReadSnapshot<'_> {
    fn drop(&mut self) {
        if self.active {
            // There is no useful recovery path in Drop. A best-effort rollback only releases the
            // SQLite snapshot; the original error remains the one returned to the caller.
            let _ = self.db.conn.execute_batch("ROLLBACK");
        }
    }
}

#[derive(Debug)]
struct StoredImageEmbedding {
    asset_id: i64,
    path: PathBuf,
    fingerprint: SourceFingerprint,
    vector: Vec<f32>,
}

impl ImageIndexDb {
    /// Attach cancellation to a reader owned exclusively by one search request.
    pub fn set_search_cancellation(
        &self,
        cancellation: &crate::cancellation::SearchCancellation,
    ) -> Result<()> {
        cancellation.register(&self.conn)
    }

    /// Open the database belonging to one model and ensure its fixed-width vector table exists.
    pub fn new(path: &Path, dimensions: usize) -> Result<Self> {
        if dimensions == 0 || dimensions > MAX_EMBEDDING_DIMENSIONS {
            bail!("image embedding dimensions must be between 1 and {MAX_EMBEDDING_DIMENSIONS}");
        }
        register_vector_extension();
        let conn = Connection::open(path)
            .with_context(|| format!("opening image embedding database: {path}"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "auto_vacuum", "FULL")?;
        configure_vector_reads(&conn)?;
        conn.pragma_update(None, "journal_mode", "wal")?;
        conn.pragma_update(None, "synchronous", "normal")?;
        open_schema(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("image_index_create.sql"),
        )?;
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS image_embeddings USING vec0(\
                 asset_id INTEGER PRIMARY KEY, \
                 embedding FLOAT[{dimensions}] distance_metric=cosine\
             );"
        ))
        .context("creating the image embedding vector table")?;
        let db = Self { conn, dimensions };
        db.check_vector_width()?;
        Ok(db)
    }

    /// Open one model's database for searching, with the asset catalog attached read-only under
    /// the `catalog` schema alias.
    ///
    /// Search needs the catalog's rows and not only the vectors: root, exclude, and time filters
    /// run against catalog columns, and a vector only answers while its recorded fingerprint
    /// still matches the catalog's current row for that asset. A read-only main connection opens
    /// its attached databases read-only as well, which the tests pin.
    pub fn new_read_only(path: &Path, dimensions: usize, catalog: &Path) -> Result<Self> {
        if dimensions == 0 || dimensions > MAX_EMBEDDING_DIMENSIONS {
            bail!("image embedding dimensions must be between 1 and {MAX_EMBEDDING_DIMENSIONS}");
        }
        register_vector_extension();
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening image embedding database: {path}"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        configure_vector_reads(&conn)?;
        check_schema_read_only(&conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
        let db = Self { conn, dimensions };
        db.check_vector_width()?;
        db.conn
            .execute("ATTACH DATABASE ?1 AS catalog", [catalog.as_str()])
            .with_context(|| format!("attaching the asset catalog: {catalog}"))?;
        register_glob(&db.conn)?;
        Ok(db)
    }

    /// Start a deferred SQLite read transaction spanning this image database and its attached
    /// catalog. The first lookup establishes one view that every component read and vector search
    /// in the request shares.
    pub fn begin_read_snapshot(&self) -> Result<ImageIndexReadSnapshot<'_>> {
        self.conn
            .execute_batch("BEGIN")
            .context("starting image index read snapshot")?;
        Ok(ImageIndexReadSnapshot {
            db: self,
            active: true,
        })
    }

    /// The vec0 table bakes the model's width into its DDL, so a database written by another model
    /// is refused rather than searched with incomparable vectors.
    fn check_vector_width(&self) -> Result<()> {
        let schema: String = self
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name = 'image_embeddings'",
                [],
                |row| row.get(0),
            )
            .context("reading the image embedding vector schema")?;
        if !schema.contains(&format!("embedding FLOAT[{}]", self.dimensions)) {
            bail!(
                "image embedding database vector width does not match the model's {} dimensions",
                self.dimensions
            );
        }
        Ok(())
    }

    fn current_asset_ids(&self, fingerprints: &[(i64, SourceFingerprint)]) -> Result<HashSet<i64>> {
        let mut current = HashSet::with_capacity(fingerprints.len());
        let mut statement = self.conn.prepare_cached(
            "SELECT EXISTS(\
                 SELECT 1 FROM image_embedding_state \
                  WHERE asset_id = ?1 AND source_modified_ns = ?2 AND source_size = ?3\
             )",
        )?;
        for (asset_id, fingerprint) in fingerprints {
            let source_size = i64::try_from(fingerprint.size)
                .context("source byte size exceeds SQLite's integer range")?;
            let exists: bool = statement
                .query_row((*asset_id, fingerprint.modified_ns, source_size), |row| {
                    row.get(0)
                })?;
            if exists {
                current.insert(*asset_id);
            }
        }
        Ok(current)
    }

    #[instrument(
        name = "save_image_embeddings",
        level = "debug",
        skip_all,
        fields(batch = items.len())
    )]
    fn save_embeddings(&mut self, items: Vec<StoredImageEmbedding>) -> Result<usize> {
        for item in &items {
            if item.vector.len() != self.dimensions {
                bail!(
                    "asset {} has {} dimensions but this image index requires {}",
                    item.asset_id,
                    item.vector.len(),
                    self.dimensions
                );
            }
            if item.vector.iter().any(|value| !value.is_finite()) {
                bail!(
                    "asset {} has a vector containing a non-finite value",
                    item.asset_id
                );
            }
        }

        let tx = self.conn.transaction()?;
        let stored = items.len();
        {
            let mut state = tx.prepare_cached(
                "INSERT INTO image_embedding_state \
                     (asset_id, source_path, source_modified_ns, source_size) \
                 VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(asset_id) DO UPDATE SET \
                     source_path = excluded.source_path, \
                     source_modified_ns = excluded.source_modified_ns, \
                     source_size = excluded.source_size",
            )?;
            let mut clear =
                tx.prepare_cached("DELETE FROM image_embeddings WHERE asset_id = ?1")?;
            let mut insert = tx.prepare_cached(
                "INSERT INTO image_embeddings (asset_id, embedding) VALUES (?1, ?2)",
            )?;
            for item in items {
                let source_size = i64::try_from(item.fingerprint.size)
                    .context("source byte size exceeds SQLite's integer range")?;
                state.execute((
                    item.asset_id,
                    item.path.as_str(),
                    item.fingerprint.modified_ns,
                    source_size,
                ))?;
                clear.execute([item.asset_id])?;
                insert.execute((item.asset_id, vector_to_blob(&item.vector)))?;
            }
        }
        tx.commit()?;
        Ok(stored)
    }

    pub fn clear(&mut self) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let cleared = tx.execute("DELETE FROM image_embeddings", [])?;
        tx.execute("DELETE FROM image_embedding_state", [])?;
        tx.commit()?;
        Ok(cleared)
    }

    pub fn delete_assets(&mut self, asset_ids: &[i64]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut deleted = 0;
        {
            let mut vectors =
                tx.prepare_cached("DELETE FROM image_embeddings WHERE asset_id = ?1")?;
            let mut state =
                tx.prepare_cached("DELETE FROM image_embedding_state WHERE asset_id = ?1")?;
            for asset_id in asset_ids {
                if *asset_id <= 0 {
                    bail!("asset identifiers must be greater than zero");
                }
                deleted += vectors.execute([asset_id])?;
                state.execute([asset_id])?;
            }
        }
        tx.commit()?;
        Ok(deleted)
    }

    /// Nearest neighbours of `vector` among this model's current image embeddings, restricted to
    /// `filters`.
    ///
    /// Same contract as the OCR store's vector search: an exact cosine scan rather than a `vec0`
    /// `MATCH ... k = ?` lookup, because `k` is applied before any join and a root filter would
    /// make a globally limited top-k silently under-return. `total` counts matches before `limit`.
    ///
    /// A vector answers only while its fingerprint still matches the catalog's row for the asset,
    /// so a changed-but-not-yet-re-embedded source drops out of results exactly like a re-OCR'd
    /// row does in the OCR store's vector search.
    #[instrument(
        name = "search_image_vectors",
        level = "debug",
        skip_all,
        fields(dimensions = vector.len(), limit, total = tracing::field::Empty)
    )]
    pub fn search_vectors(
        &self,
        vector: &[f32],
        filters: &SearchFilters<'_>,
        limit: usize,
        options: &ImageVectorSearchOptions,
    ) -> Result<(usize, Vec<ImageVectorHit>)> {
        if vector.len() != self.dimensions {
            bail!(
                "query vector has {} dimensions but this image index requires {}",
                vector.len(),
                self.dimensions
            );
        }
        let bound = filters.bind(FilterScope::CATALOG_ROWS)?;
        let scored = format!(
            r#"
            WITH scored AS MATERIALIZED (
                SELECT image_embeddings.asset_id AS asset_id,
                       vec_distance_cosine(image_embeddings.embedding, :query) AS distance,
                       catalog.assets.source_modified_ns AS source_modified_ns
                  FROM image_embeddings
                  INNER JOIN image_embedding_state
                          ON image_embedding_state.asset_id = image_embeddings.asset_id
                  INNER JOIN catalog.assets
                          ON catalog.assets.asset_id = image_embeddings.asset_id
                         AND catalog.assets.source_modified_ns
                              = image_embedding_state.source_modified_ns
                         AND catalog.assets.source_size = image_embedding_state.source_size
                 WHERE 1{filters}
            )"#,
            filters = bound.sql
        );

        // Materialize only IDs, distances and tie breakers: scalar reads from vec0 open a
        // vector blob per evaluation. Counting and sorting must reuse those scores.
        let mut shared = bound.params;
        shared.push((":query", Value::Blob(vector_to_blob(vector))));
        shared.push((
            ":max_distance",
            Value::Real(options.max_distance.unwrap_or(UNBOUNDED_DISTANCE)),
        ));

        let mut params = shared;
        params.push((
            ":limit",
            Value::Integer(
                i64::try_from(limit).context("search limit exceeds SQLite's integer range")?,
            ),
        ));

        let mut statement = self.conn.prepare_cached(&format!(
            r#"{scored}
            , hits AS (
            SELECT asset_id, distance, source_modified_ns
              FROM scored
             WHERE distance <= :max_distance
             ORDER BY distance ASC, source_modified_ns DESC, asset_id ASC
             LIMIT :limit
            )
            SELECT totals.total, hits.asset_id, hits.distance
              FROM (SELECT count(*) AS total FROM scored WHERE distance <= :max_distance) totals
              LEFT JOIN hits ON 1
             ORDER BY hits.distance ASC, hits.source_modified_ns DESC, hits.asset_id ASC"#
        ))?;
        let mut rows = statement
            .query(bind_named(&params).as_slice())
            .context("querying the image embedding vector index")?;
        let mut total = 0_i64;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            total = row.get(0)?;
            // LEFT JOIN retains the total for an empty result, including limit = 0.
            if let Some(asset_id) = row.get::<_, Option<i64>>(1)? {
                results.push(ImageVectorHit {
                    asset_id,
                    distance: row.get(2)?,
                });
            }
        }

        tracing::Span::current().record("total", total);
        Ok((
            usize::try_from(total).context("image vector search result count exceeds usize")?,
            results,
        ))
    }

    /// Return one asset's vector only while it is current with the attached catalog.
    ///
    /// Query-reference assets deliberately are not constrained to a search root: an image from
    /// one library may be the example used to search another. They must still belong to this
    /// model's index and match the catalog fingerprint, or this answers `None` rather than
    /// accidentally comparing an obsolete or incompatible vector.
    pub fn current_vector(&self, asset_id: i64) -> Result<Option<Vec<f32>>> {
        let bytes: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT image_embeddings.embedding \
                   FROM image_embeddings \
                  INNER JOIN image_embedding_state \
                          ON image_embedding_state.asset_id = image_embeddings.asset_id \
                  INNER JOIN catalog.assets \
                          ON catalog.assets.asset_id = image_embeddings.asset_id \
                         AND catalog.assets.source_modified_ns \
                              = image_embedding_state.source_modified_ns \
                         AND catalog.assets.source_size = image_embedding_state.source_size \
                  WHERE image_embeddings.asset_id = ?1",
                [asset_id],
                |row| row.get(0),
            )
            .optional()
            .context("reading a current image query vector")?;
        bytes.map(|bytes| self.vector_from_blob(&bytes)).transpose()
    }

    fn vector_from_blob(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let expected_bytes = self
            .dimensions
            .checked_mul(std::mem::size_of::<f32>())
            .context("image vector dimensions overflow the byte length")?;
        if bytes.len() != expected_bytes {
            bail!(
                "image index returned a {}-byte vector, expected {expected_bytes} bytes for {} dimensions",
                bytes.len(),
                self.dimensions
            );
        }
        let (chunks, remainder) = bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty(), "the length check left no remainder");
        Ok(chunks.iter().copied().map(f32::from_le_bytes).collect())
    }

    #[cfg(test)]
    fn vector_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM image_embeddings", [], |row| {
                row.get(0)
            })
            .context("counting image embeddings")?;
        usize::try_from(count).context("image embedding count exceeds usize")
    }
}

/// Options for [`ImageIndexDb::search_vectors`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageVectorSearchOptions {
    /// Drop neighbours further than this cosine distance. `None` keeps every neighbour.
    pub max_distance: Option<f64>,
}

/// One nearest neighbour of an image query.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageVectorHit {
    pub asset_id: i64,
    /// Cosine distance from the query vector: 0 identical, 1 orthogonal, 2 opposite.
    pub distance: f64,
}

/// Flags for an incremental image-indexing pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageIndexOptions {
    pub force: bool,
    pub retry_failed: bool,
    pub limit: Option<usize>,
}

/// Incrementally embed cataloged images beneath `root`.
///
/// Decode workers overlap source I/O with a single batched inference lane. ONNX Runtime owns all
/// model execution parallelism; no session replicas are created.
#[instrument(
    name = "image_index",
    skip_all,
    fields(root = %root, model = %embedder.model(), sources = tracing::field::Empty)
)]
pub fn index_images_observed(
    catalog: &AssetCatalog,
    db: &mut ImageIndexDb,
    embedder: &ImageEmbedder,
    root: &Path,
    options: ImageIndexOptions,
    observer: &dyn IndexObserver,
) -> Result<bool> {
    let ImageIndexOptions {
        force,
        retry_failed,
        limit,
    } = options;
    let assets = catalog.under_root(root)?;
    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
    let images = assets
        .into_iter()
        .filter(is_embedding_image)
        .collect::<Vec<_>>();
    let current = if force {
        let asset_ids = images
            .iter()
            .map(|asset| asset.asset_id)
            .collect::<Vec<_>>();
        db.delete_assets(&asset_ids)?;
        HashSet::new()
    } else {
        let fingerprints = images
            .iter()
            .map(|asset| (asset.asset_id, asset.fingerprint))
            .collect::<Vec<_>>();
        db.current_asset_ids(&fingerprints)?
    };
    let decode_failed = if force || retry_failed {
        HashSet::new()
    } else {
        let fingerprints = images
            .iter()
            .map(|asset| (asset.asset_id, asset.fingerprint))
            .collect::<Vec<_>>();
        catalog.current_decode_failure_asset_ids(&fingerprints)?
    };
    let mut sources = images
        .into_iter()
        .filter(|asset| {
            !current.contains(&asset.asset_id) && !decode_failed.contains(&asset.asset_id)
        })
        .collect::<Vec<_>>();
    if let Some(limit) = limit {
        sources.truncate(limit);
    }
    tracing::Span::current().record("sources", sources.len());
    observer.on_event(IndexEvent::Discovered {
        count: sources.len(),
    });
    observer.on_event(IndexEvent::DiscoveryComplete {
        total: sources.len(),
    });
    if sources.is_empty() {
        return Ok(observer.is_cancelled());
    }

    BoundedImagePipeline {
        catalog,
        db,
        embedder,
        observer,
    }
    .run(sources)
}

fn is_embedding_image(asset: &Asset) -> bool {
    asset.media_kind == MediaKind::Image
        && matches!(
            asset.media_format.as_str(),
            "png" | "jpeg" | "gif" | "webp" | "bmp"
        )
}

struct BoundedImagePipeline<'a> {
    catalog: &'a AssetCatalog,
    db: &'a mut ImageIndexDb,
    embedder: &'a ImageEmbedder,
    observer: &'a dyn IndexObserver,
}

impl BoundedImagePipeline<'_> {
    fn run(self, sources: Vec<Asset>) -> Result<bool> {
        let batch_size = self.embedder.max_batch_size().min(sources.len());
        let decode_workers = decode_worker_count().min(sources.len());
        // One normalized batch may wait while one batch is inferred. Decode workers use the
        // model-owned FastEmbed preprocessor, then discard their full-resolution source image.
        let (outcome_sender, outcome_receiver) = bounded::<DecodeOutcome>(batch_size);
        let (abort_sender, abort_receiver) = bounded::<()>(1);
        let next = AtomicUsize::new(0);
        let mut pending = Vec::with_capacity(batch_size);
        let mut fatal = None;
        let mut write_error = None;
        let mut cancelled = false;

        std::thread::scope(|scope| {
            let mut workers = Vec::with_capacity(decode_workers);
            let mut abort_sender = Some(abort_sender);
            for _ in 0..decode_workers {
                let outcomes = outcome_sender.clone();
                let abort = abort_receiver.clone();
                let sources = &sources;
                let next = &next;
                workers.push(scope.spawn(move || {
                    guard_decode_worker(&outcomes, &abort, || {
                        decode_sources(
                            sources,
                            next,
                            self.embedder,
                            self.observer,
                            &outcomes,
                            &abort,
                        );
                    });
                }));
            }
            drop(outcome_sender);

            while let Ok(outcome) = outcome_receiver.recv() {
                if !cancelled && self.observer.is_cancelled() {
                    cancelled = true;
                    pending.clear();
                    abort_sender.take();
                }
                let stopping = cancelled || write_error.is_some() || fatal.is_some();
                match outcome {
                    DecodeOutcome::Success(decoded) if !stopping => {
                        pending.push(decoded);
                        if pending.len() == batch_size
                            && let Err(error) =
                                flush_batch(self.db, self.embedder, self.observer, &mut pending)
                        {
                            write_error = Some(error);
                            abort_sender.take();
                        }
                    }
                    DecodeOutcome::Failure {
                        asset,
                        message,
                        cache_decode_failure,
                    } if !stopping => {
                        self.observer.on_event(IndexEvent::ActiveAsset {
                            path: asset.path.clone(),
                            active: false,
                        });
                        let recorded = if cache_decode_failure {
                            self.catalog.record_decode_failure(&asset)
                        } else {
                            Ok(())
                        };
                        match recorded {
                            Ok(()) => report_failure(self.observer, asset.path, message),
                            Err(error) => {
                                write_error = Some(error);
                                abort_sender.take();
                            }
                        }
                    }
                    DecodeOutcome::Fatal(message) => {
                        fatal.get_or_insert(message);
                        abort_sender.take();
                    }
                    DecodeOutcome::Success(_) | DecodeOutcome::Failure { .. } => {}
                }
            }

            drop(abort_sender);
            for worker in workers {
                if worker.join().is_err() {
                    fatal.get_or_insert_with(|| "an image decode worker panicked".to_owned());
                }
            }
        });

        if let Some(error) = write_error {
            return Err(error);
        }
        if let Some(message) = fatal {
            bail!("{message}");
        }
        cancelled |= self.observer.is_cancelled();
        if !cancelled {
            flush_batch(self.db, self.embedder, self.observer, &mut pending)?;
        }
        info!(
            decode_workers,
            batch_size,
            cancelled = self.observer.is_cancelled(),
            "CLIP image indexing complete"
        );
        Ok(cancelled)
    }
}

struct DecodedImage {
    asset: Asset,
    pixels: ndarray::Array3<f32>,
}

enum DecodeOutcome {
    Success(DecodedImage),
    Failure {
        asset: Asset,
        message: String,
        cache_decode_failure: bool,
    },
    Fatal(String),
}

fn decode_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .saturating_sub(1)
        .clamp(1, MAX_DECODE_WORKERS)
}

fn decode_sources(
    sources: &[Asset],
    next: &AtomicUsize,
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
    outcomes: &Sender<DecodeOutcome>,
    abort: &Receiver<()>,
) {
    loop {
        if observer.is_cancelled() || aborted(abort) {
            break;
        }
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(asset) = sources.get(index) else {
            break;
        };
        observer.on_event(IndexEvent::ActiveAsset {
            path: asset.path.clone(),
            active: true,
        });
        let outcome = match prepare_image(
            &asset.path,
            embedder.model() != crate::embedding::ImageEmbeddingModel::ClipVitB32,
            embedder.model().is_deepghs(),
            |image| embedder.preprocess_image(image),
        ) {
            Ok(pixels) => DecodeOutcome::Success(DecodedImage {
                asset: asset.clone(),
                pixels,
            }),
            Err(PreparationError::Decode(error))
                if crate::index::is_missing_source_error(&error) =>
            {
                crate::index::report_disappeared_source(observer, &asset.path);
                continue;
            }
            Err(PreparationError::Decode(error)) => DecodeOutcome::Failure {
                asset: asset.clone(),
                message: format!("decoding image for CLIP failed: {error:#}"),
                cache_decode_failure: true,
            },
            Err(PreparationError::Preprocess(error)) => DecodeOutcome::Failure {
                asset: asset.clone(),
                message: format!("preprocessing image for CLIP failed: {error:#}"),
                cache_decode_failure: false,
            },
        };
        if !send_unless_aborted(outcomes, outcome, abort) {
            break;
        }
    }
}

#[instrument(
    name = "decode_image",
    level = "debug",
    skip_all,
    fields(path = %path)
)]
fn decode_image(path: &Path, accurate: bool, white_background: bool) -> Result<RgbImage> {
    let data = std::fs::read(path).with_context(|| format!("opening image: {path}"))?;
    let raster = if accurate {
        crate::imaging::decode_accurate(&data)
    } else {
        crate::imaging::decode(&data)
    }
    .with_context(|| format!("decoding image: {path}"))?;
    let (width, height) = (raster.width(), raster.height());
    let pixels = if white_background {
        raster.flatten_rgb([255, 255, 255])
    } else {
        raster.into_rgb_bytes()
    };
    RgbImage::from_raw(width, height, pixels).context("assembling decoded image buffer")
}

enum PreparationError {
    Decode(anyhow::Error),
    Preprocess(anyhow::Error),
}

fn prepare_image(
    path: &Path,
    accurate: bool,
    white_background: bool,
    preprocess: impl FnOnce(RgbImage) -> Result<ndarray::Array3<f32>>,
) -> Result<ndarray::Array3<f32>, PreparationError> {
    let image = decode_image(path, accurate, white_background).map_err(PreparationError::Decode)?;
    // A model-specific transform failure says nothing about whether OCR can decode the file.
    preprocess(image).map_err(PreparationError::Preprocess)
}

fn flush_batch(
    db: &mut ImageIndexDb,
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
    pending: &mut Vec<DecodedImage>,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(pending);
    let (assets, pixels): (Vec<_>, Vec<_>) = batch
        .into_iter()
        .map(|decoded| (decoded.asset, decoded.pixels))
        .unzip();
    let vectors = match embedder.embed_preprocessed_images(pixels) {
        Ok(vectors) => vectors,
        Err(error) => {
            let message = format!("CLIP image inference failed: {error:#}");
            for asset in assets {
                observer.on_event(IndexEvent::ActiveAsset {
                    path: asset.path.clone(),
                    active: false,
                });
                report_failure(observer, asset.path, message.clone());
            }
            return Err(error).context("running the CLIP image embedding batch");
        }
    };
    let attempted = assets.len();
    let active_paths = assets
        .iter()
        .map(|asset| asset.path.clone())
        .collect::<Vec<_>>();
    let items = assets
        .into_iter()
        .zip(vectors)
        .map(|(asset, vector)| StoredImageEmbedding {
            asset_id: asset.asset_id,
            path: asset.path,
            fingerprint: asset.fingerprint,
            vector,
        })
        .collect();
    let stored = db.save_embeddings(items)?;
    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
        processed: attempted,
        phase_completed: attempted,
        embedded: stored,
        skipped: attempted - stored,
        ..IndexProgressDelta::default()
    }));
    for path in active_paths {
        observer.on_event(IndexEvent::ActiveAsset {
            path,
            active: false,
        });
    }
    debug!(stored, "saved a CLIP image embedding batch");
    Ok(())
}

fn report_failure(observer: &dyn IndexObserver, path: PathBuf, message: String) {
    error!(path = %path, "{message}");
    observer.on_event(IndexEvent::Error {
        path: Some(path),
        message,
    });
    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
        processed: 1,
        phase_completed: 1,
        failed: 1,
        ..IndexProgressDelta::default()
    }));
}

fn guard_decode_worker(
    outcomes: &Sender<DecodeOutcome>,
    abort: &Receiver<()>,
    body: impl FnOnce(),
) {
    let Err(panic) = catch_unwind(AssertUnwindSafe(body)) else {
        return;
    };
    let message = panic
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panicked".to_owned());
    send_unless_aborted(outcomes, DecodeOutcome::Fatal(message), abort);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn preprocessing_errors_do_not_classify_valid_images_as_decode_failures() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("valid.jpg"))?;
        std::fs::write(
            &path,
            crate::imaging::test_support::jpeg_with_orientation(1)?,
        )?;
        assert!(matches!(
            prepare_image(&path, false, false, |_| anyhow::bail!(
                "model transform failed"
            )),
            Err(PreparationError::Preprocess(_))
        ));
        std::fs::write(&path, b"invalid image")?;
        assert!(matches!(
            prepare_image(&path, false, false, |_| panic!(
                "invalid pixels cannot reach the preprocessor"
            )),
            Err(PreparationError::Decode(_))
        ));
        Ok(())
    }

    use crate::assets::Timeline;
    use crate::db::TimeRange;

    /// One catalog row: id, path, `source_modified_ns`, `exif_taken_ns`, size.
    type CatalogRow = (i64, PathBuf, i64, Option<i64>, u64);

    /// Create an asset catalog with the given rows, exactly as the indexing pipeline would have
    /// cataloged them. The catalog's upsert API fingerprints real files, which a vector search
    /// fixture does not need; the row shape it stores is what search joins against.
    fn catalog_with_rows(temp: &TempDir, rows: &[CatalogRow]) -> Result<PathBuf> {
        let path = PathBuf::try_from(temp.path().join("assets.db"))?;
        drop(AssetCatalog::new(&path)?);
        let conn = Connection::open(&path)?;
        for (asset_id, path, modified_ns, exif_taken_ns, size) in rows {
            conn.execute(
                "INSERT INTO assets(asset_id, path, source_modified_ns, exif_taken_ns, \
                     source_size, media_kind, media_format, is_animated) \
                 VALUES (?1, ?2, ?3, ?4, ?5, 'image', 'png', 0)",
                (
                    asset_id,
                    path.as_str(),
                    modified_ns,
                    exif_taken_ns,
                    *size as i64,
                ),
            )?;
        }
        Ok(path)
    }

    /// An image index whose state records the fingerprint each vector was computed from.
    fn image_index_with_vectors(
        temp: &TempDir,
        dimensions: usize,
        vectors: &[(i64, PathBuf, i64, u64, Vec<f32>)],
    ) -> Result<PathBuf> {
        let path = PathBuf::try_from(temp.path().join("images.db"))?;
        let mut db = ImageIndexDb::new(&path, dimensions)?;
        db.save_embeddings(
            vectors
                .iter()
                .map(
                    |(asset_id, path, modified_ns, size, vector)| StoredImageEmbedding {
                        asset_id: *asset_id,
                        path: path.clone(),
                        fingerprint: SourceFingerprint {
                            modified_ns: *modified_ns,
                            size: *size,
                        },
                        vector: vector.clone(),
                    },
                )
                .collect(),
        )?;
        Ok(path)
    }

    #[test]
    fn model_database_tracks_fingerprints_and_replaces_vectors() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("clip.db"))?;
        let mut db = ImageIndexDb::new(&path, 3)?;
        let fingerprint = SourceFingerprint {
            modified_ns: 10,
            size: 20,
        };
        db.save_embeddings(vec![StoredImageEmbedding {
            asset_id: 1,
            path: PathBuf::from("C:/gallery/a.png"),
            fingerprint,
            vector: vec![1.0, 0.0, 0.0],
        }])?;
        assert_eq!(db.vector_count()?, 1);
        assert_eq!(
            db.current_asset_ids(&[(1, fingerprint)])?,
            HashSet::from([1])
        );

        db.save_embeddings(vec![StoredImageEmbedding {
            asset_id: 1,
            path: PathBuf::from("C:/gallery/a.png"),
            fingerprint: SourceFingerprint {
                modified_ns: 11,
                size: 20,
            },
            vector: vec![0.0, 1.0, 0.0],
        }])?;
        assert_eq!(db.vector_count()?, 1);
        assert!(db.current_asset_ids(&[(1, fingerprint)])?.is_empty());
        Ok(())
    }

    #[test]
    fn deleting_assets_removes_clip_vectors_and_fingerprint_state() -> Result<()> {
        let temp = TempDir::new()?;
        let path = image_index_with_vectors(
            &temp,
            3,
            &[
                (
                    1,
                    PathBuf::from("C:/gallery/deleted.png"),
                    10,
                    20,
                    vec![1.0, 0.0, 0.0],
                ),
                (
                    2,
                    PathBuf::from("C:/gallery/retained.png"),
                    10,
                    20,
                    vec![0.0, 1.0, 0.0],
                ),
            ],
        )?;
        let mut db = ImageIndexDb::new(&path, 3)?;
        assert_eq!(db.delete_assets(&[1])?, 1);
        assert_eq!(db.vector_count()?, 1);
        assert!(
            db.current_asset_ids(&[(
                1,
                SourceFingerprint {
                    modified_ns: 10,
                    size: 20
                }
            )])?
            .is_empty()
        );
        let vectors: i64 = db.conn.query_row(
            "SELECT count(*) FROM image_embeddings WHERE asset_id = 1",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(vectors, 0);
        assert_eq!(db.delete_assets(&[1])?, 0);
        Ok(())
    }

    #[test]
    fn database_rejects_a_different_vector_width() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("clip.db"))?;
        drop(ImageIndexDb::new(&path, 3)?);
        assert!(ImageIndexDb::new(&path, 4).is_err());
        Ok(())
    }

    /// The shared fixture: five cataloged images, of which one lives outside the searched root,
    /// one inside the excluded subtree, and one whose recorded fingerprint no longer matches the
    /// catalog's row for it.
    fn search_fixture(temp: &TempDir) -> Result<(PathBuf, PathBuf, PathBuf)> {
        let root = PathBuf::try_from(temp.path().join("gallery"))?;
        let outside = PathBuf::try_from(temp.path().join("elsewhere"))?;
        let catalog = catalog_with_rows(
            temp,
            &[
                (1, root.join("a.png"), 100, None, 10),
                (2, root.join("b.png"), 200, Some(50), 20),
                (3, root.join("skip").join("c.png"), 300, None, 30),
                (4, outside.join("d.png"), 400, None, 40),
                (5, root.join("e.png"), 500, None, 50),
            ],
        )?;
        // Asset 5's vector was computed from a fingerprint the catalog no longer has, so search
        // must drop it like the OCR store drops a re-OCR'd row's vector.
        let images = image_index_with_vectors(
            temp,
            3,
            &[
                (1, root.join("a.png"), 100, 10, vec![1.0, 0.0, 0.0]),
                (2, root.join("b.png"), 200, 20, vec![0.0, 1.0, 0.0]),
                (
                    3,
                    root.join("skip").join("c.png"),
                    300,
                    30,
                    vec![0.0, 1.0, 0.0],
                ),
                (4, outside.join("d.png"), 400, 40, vec![1.0, 0.0, 0.0]),
                (5, root.join("e.png"), 999, 50, vec![1.0, 0.0, 0.0]),
            ],
        )?;
        Ok((catalog, images, root))
    }

    #[test]
    fn image_search_orders_neighbours_and_drops_stale_and_outside_rows() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            100,
            &ImageVectorSearchOptions::default(),
        )?;

        // Asset 4 is outside the root and asset 5 is stale; both drop out of the total. The two
        // orthogonal neighbours tie on distance and break on recency.
        assert_eq!(total, 3);
        let ids = hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 3, 2]);
        assert!(hits[0].distance < 1e-6, "{:?}", hits[0].distance);
        Ok(())
    }

    #[test]
    fn current_query_vector_requires_a_matching_catalog_fingerprint() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, _) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;

        assert_eq!(reader.current_vector(1)?, Some(vec![1.0, 0.0, 0.0]));
        // Asset 5 is still in both tables but its catalog fingerprint changed after embedding.
        assert_eq!(reader.current_vector(5)?, None);
        assert_eq!(reader.current_vector(999)?, None);
        Ok(())
    }

    #[test]
    fn a_component_and_its_image_search_share_one_catalog_snapshot() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let mut snapshot = reader.begin_read_snapshot()?;

        // This first read establishes the image index + attached catalog view for the component.
        assert_eq!(snapshot.current_vector(1)?, Some(vec![1.0, 0.0, 0.0]));
        let writer = Connection::open(&catalog)?;
        writer.execute(
            "UPDATE assets SET source_modified_ns = 101 WHERE asset_id = 1",
            [],
        )?;

        // The request consistently sees the old-current pair rather than accepting the old
        // component then mixing it with a newly stale result query.
        assert_eq!(snapshot.current_vector(1)?, Some(vec![1.0, 0.0, 0.0]));
        let (total, hits) = snapshot.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            100,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert_eq!(hits[0].asset_id, 1);
        snapshot.commit()?;

        // A later request establishes a new view and rejects the now-stale component.
        assert_eq!(reader.current_vector(1)?, None);
        Ok(())
    }

    #[test]
    fn image_search_honours_the_exclude_the_ceiling_the_limit_and_the_timeline() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let query = [1.0, 0.0, 0.0];

        let excluded = root.join("skip");
        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root).with_exclude(Some(excluded.as_str())),
            100,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2, "the excluded subtree does not answer");
        assert_eq!(
            hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1, 2]
        );

        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root),
            100,
            &ImageVectorSearchOptions {
                max_distance: Some(0.5),
            },
        )?;
        assert_eq!(total, 1, "the ceiling drops the orthogonal neighbours");
        assert_eq!(
            hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1]
        );

        // `limit` caps the returned rows without changing how many matched.
        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root),
            2,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert_eq!(hits.len(), 2);

        // Capture is `COALESCE(exif_taken_ns, source_modified_ns)`, so the after bound drops
        // asset 2 (taken at 50) and keeps assets 1 and 3.
        let time = TimeRange {
            timeline: Timeline::Capture,
            after_ns: Some(60),
            before_ns: None,
        };
        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root).with_time(Some(time)),
            100,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2);
        assert_eq!(
            hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1, 3]
        );
        Ok(())
    }

    #[test]
    fn image_search_preserves_totals_when_no_hits_are_returned() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            0,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert!(hits.is_empty());

        let (total, hits) = reader.search_vectors(
            &[-1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            10,
            &ImageVectorSearchOptions {
                max_distance: Some(0.0),
            },
        )?;
        assert_eq!(total, 0);
        assert!(hits.is_empty());
        Ok(())
    }

    #[test]
    fn image_search_refuses_a_query_vector_of_the_wrong_width() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        assert!(
            reader
                .search_vectors(
                    &[1.0, 0.0],
                    &SearchFilters::new(&root),
                    10,
                    &ImageVectorSearchOptions::default(),
                )
                .is_err(),
            "a vector from another width is not comparable"
        );
        Ok(())
    }

    #[test]
    fn a_read_only_search_connection_cannot_write_the_attached_catalog() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        // Read-only flags propagate to attached databases; if they ever stop doing so, this
        // fails open and the reader could mutate the catalog.
        assert!(
            reader
                .conn
                .execute(
                    "INSERT INTO catalog.assets(asset_id, path, source_modified_ns, \
                         source_size, media_kind, media_format, is_animated) \
                     VALUES (99, 'x', 1, 1, 'image', 'png', 0)",
                    [],
                )
                .is_err(),
            "the attached catalog must be read-only"
        );
        let (total, _) = reader.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            10,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        Ok(())
    }
}
