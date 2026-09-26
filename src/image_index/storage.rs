use crate::assets::{Asset, MediaKind, SourceFingerprint};
use crate::db::{
    FilterScope, SearchFilters, UNBOUNDED_DISTANCE, bind_named, configure_vector_reads,
    configure_vector_writer, register_vector_extension, vector_to_blob,
};
use crate::schema::{check_schema_read_only, open_schema_with_migrations};
use crate::storage::{READ_ONLY_FLAGS, configure_reader, maintain, validate_asset_ids};
use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension, Transaction, params_from_iter};
use std::collections::HashSet;
use std::ops::Deref;
use tracing::instrument;

const SCHEMA_VERSION: i32 = 4;
const SCHEMA_LABEL: &str = "image embedding database";
const MIGRATIONS: &[(i32, &str)] = &[
    (1, "VACUUM;"),
    (
        2,
        "ALTER TABLE image_embedding_state ADD COLUMN sampling_version INTEGER NOT NULL DEFAULT 0;",
    ),
    (
        3,
        r#"CREATE TABLE image_embedding_samples(
               embedding_id INTEGER PRIMARY KEY,
               asset_id INTEGER NOT NULL,
               timestamp_ms INTEGER
           );
           CREATE INDEX image_embedding_samples_asset_idx
               ON image_embedding_samples(asset_id, timestamp_ms);
           CREATE INDEX image_embedding_video_samples_idx
               ON image_embedding_samples(asset_id, timestamp_ms) WHERE timestamp_ms IS NOT NULL;
           CREATE TABLE image_embedding_sample_migration(
               singleton INTEGER PRIMARY KEY,
               complete INTEGER NOT NULL
           );
           INSERT INTO image_embedding_sample_migration VALUES (1, 0);"#,
    ),
];
const MAX_EMBEDDING_DIMENSIONS: usize = 65_536;

pub struct ImageIndexDb {
    pub(super) conn: Connection,
    dimensions: usize,
}

/// A consistent read view of one image index and its attached asset catalog.
///
/// Image-query components and their neighbour search must share this transaction: otherwise a
/// component could be current when read, become stale while the request is running, and still
/// affect a search that correctly excludes it. Dropping an uncommitted snapshot rolls it back.
pub struct ImageIndexReadSnapshot<'db> {
    snapshot: crate::storage::ReadSnapshot<&'db ImageIndexDb>,
}

impl Deref for ImageIndexReadSnapshot<'_> {
    type Target = ImageIndexDb;

    fn deref(&self) -> &Self::Target {
        self.snapshot.target
    }
}

impl ImageIndexReadSnapshot<'_> {
    /// Cleanly complete this read-only transaction. Errors and unwinding roll back through
    /// [`Drop`] instead.
    pub fn commit(&mut self) -> Result<()> {
        self.snapshot
            .commit()
            .context("committing image index read snapshot")
    }
}

#[derive(Debug)]
pub(super) struct StoredImageEmbedding {
    pub(super) asset_id: i64,
    pub(super) path: PathBuf,
    pub(super) fingerprint: SourceFingerprint,
    pub(super) vector: Vec<f32>,
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
        configure_vector_writer(&conn)?;
        open_schema_with_migrations(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("../image_index_create.sql"),
            MIGRATIONS,
        )?;
        ensure_vector_table(&conn, dimensions)?;
        ensure_sample_rows(&conn)?;
        let db = Self { conn, dimensions };
        db.check_vector_width()?;
        Ok(db)
    }

    /// Open one model's database for searching, with the asset catalog attached read-only under
    /// the `catalog` schema alias.
    ///
    /// Catalog columns provide filters and fingerprint currency for stored vectors.
    pub fn new_read_only(path: &Path, dimensions: usize, catalog: &Path) -> Result<Self> {
        if dimensions == 0 || dimensions > MAX_EMBEDDING_DIMENSIONS {
            bail!("image embedding dimensions must be between 1 and {MAX_EMBEDDING_DIMENSIONS}");
        }
        register_vector_extension();
        let conn = Connection::open_with_flags(path, READ_ONLY_FLAGS)
            .with_context(|| format!("opening image embedding database: {path}"))?;
        configure_reader(&conn)?;
        configure_vector_reads(&conn)?;
        check_schema_read_only(&conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
        ensure_sample_rows_ready(&conn)?;
        let db = Self { conn, dimensions };
        db.check_vector_width()?;
        db.conn
            .execute("ATTACH DATABASE ?1 AS catalog", [catalog.as_str()])
            .with_context(|| format!("attaching the asset catalog: {catalog}"))?;
        Ok(db)
    }

    /// Start a deferred SQLite read transaction spanning this image database and its attached
    /// catalog. The first lookup establishes one view that every component read and vector search
    /// in the request shares.
    pub fn begin_read_snapshot(&self) -> Result<ImageIndexReadSnapshot<'_>> {
        Ok(ImageIndexReadSnapshot {
            snapshot: crate::storage::ReadSnapshot::begin(self, |db| &db.conn)
                .context("starting image index read snapshot")?,
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
        if !schema.contains("embedding_id INTEGER PRIMARY KEY")
            || !schema.contains("+timestamp_ms INTEGER")
        {
            bail!("image embedding database vector table migration is incomplete");
        }
        Ok(())
    }

    pub(super) fn current_asset_ids(
        &self,
        fingerprints: &[(i64, SourceFingerprint)],
    ) -> Result<HashSet<i64>> {
        crate::storage::matching_fingerprint_ids(
            &self.conn,
            "SELECT EXISTS(\
                 SELECT 1 FROM image_embedding_state \
                  WHERE asset_id = ?1 AND source_modified_ns = ?2 AND source_size = ?3\
                    AND sampling_version = 0\
             )",
            fingerprints,
        )
    }

    /// Video currency also includes the decoder's sampling policy.
    pub(super) fn current_video_asset_ids(
        &self,
        fingerprints: &[(i64, SourceFingerprint)],
    ) -> Result<HashSet<i64>> {
        crate::storage::matching_fingerprint_ids(
            &self.conn,
            &format!(
                "SELECT EXISTS(SELECT 1 FROM image_embedding_state WHERE asset_id = ?1 AND source_modified_ns = ?2 AND source_size = ?3 AND sampling_version = {})",
                crate::video::SAMPLING_VERSION
            ),
            fingerprints,
        )
    }

    /// Sample keys currently exposed to search for selected videos. No vector rows are read.
    pub(super) fn video_sample_timestamps_for_assets(
        &self,
        asset_ids: &std::collections::HashSet<i64>,
    ) -> Result<std::collections::HashMap<i64, Vec<i64>>> {
        let mut timestamps = std::collections::HashMap::new();
        if asset_ids.is_empty() {
            return Ok(timestamps);
        }
        // This scans only the narrow ordinary table, ordered by its asset index.
        let mut statement = self.conn.prepare_cached(
            "SELECT asset_id, timestamp_ms FROM image_embedding_samples WHERE timestamp_ms IS NOT NULL ORDER BY asset_id, timestamp_ms",
        )?;
        let rows =
            statement.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        for row in rows {
            let (asset_id, timestamp) = row?;
            if asset_ids.contains(&asset_id) {
                timestamps
                    .entry(asset_id)
                    .or_insert_with(Vec::new)
                    .push(timestamp);
            }
        }
        Ok(timestamps)
    }

    #[instrument(
        name = "save_image_embeddings",
        level = "debug",
        skip_all,
        fields(batch = items.len())
    )]
    pub(super) fn save_embeddings(&mut self, items: Vec<StoredImageEmbedding>) -> Result<usize> {
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
                     (asset_id, source_path, source_modified_ns, source_size, sampling_version) \
                 VALUES (?1, ?2, ?3, ?4, 0) \
                 ON CONFLICT(asset_id) DO UPDATE SET \
                     source_path = excluded.source_path, \
                     source_modified_ns = excluded.source_modified_ns, \
                     source_size = excluded.source_size, \
                     sampling_version = excluded.sampling_version",
            )?;
            let mut insert = tx.prepare_cached(
                "INSERT INTO image_embeddings (embedding_id, asset_id, timestamp_ms, embedding) VALUES (?1, ?2, NULL, ?3)",
            )?;
            for item in items {
                let source_size = i64::try_from(item.fingerprint.size)
                    .context("source byte size exceeds SQLite's integer range")?;
                delete_asset_samples(&tx, item.asset_id)?;
                let embedding_id = insert_sample(&tx, item.asset_id, None)?;
                insert.execute((embedding_id, item.asset_id, vector_to_blob(&item.vector)))?;
                state.execute((
                    item.asset_id,
                    item.path.as_str(),
                    item.fingerprint.modified_ns,
                    source_size,
                ))?;
            }
        }
        tx.commit()?;
        Ok(stored)
    }

    /// Replace every sample for one video and its currency record as one transaction.
    pub(super) fn save_video_embeddings(
        &mut self,
        asset: &Asset,
        samples: Vec<(i64, Vec<f32>)>,
    ) -> Result<usize> {
        if asset.media_kind != MediaKind::Video {
            bail!("asset {} is not a video", asset.asset_id);
        }
        if samples.is_empty() {
            bail!("video {} has no samples", asset.asset_id);
        }
        for (timestamp_ms, vector) in &samples {
            if *timestamp_ms < 0
                || vector.len() != self.dimensions
                || vector.iter().any(|value| !value.is_finite())
            {
                bail!(
                    "video {} has an invalid sample at {timestamp_ms} ms",
                    asset.asset_id
                );
            }
        }
        let source_size = i64::try_from(asset.fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let tx = self.conn.transaction()?;
        delete_asset_samples(&tx, asset.asset_id)?;
        {
            let mut insert = tx.prepare_cached("INSERT INTO image_embeddings (embedding_id, asset_id, timestamp_ms, embedding) VALUES (?1, ?2, ?3, ?4)")?;
            for (timestamp_ms, vector) in &samples {
                let embedding_id = insert_sample(&tx, asset.asset_id, Some(*timestamp_ms))?;
                insert.execute((
                    embedding_id,
                    asset.asset_id,
                    timestamp_ms,
                    vector_to_blob(vector),
                ))?;
            }
        }
        tx.execute(
            "INSERT INTO image_embedding_state (asset_id, source_path, source_modified_ns, source_size, sampling_version) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(asset_id) DO UPDATE SET source_path = excluded.source_path, source_modified_ns = excluded.source_modified_ns, source_size = excluded.source_size, sampling_version = excluded.sampling_version",
            (asset.asset_id, asset.path.as_str(), asset.fingerprint.modified_ns, source_size, i64::from(crate::video::SAMPLING_VERSION)),
        )?;
        tx.commit()?;
        Ok(samples.len())
    }

    pub fn clear(&mut self) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let cleared = tx.execute("DELETE FROM image_embeddings", [])?;
        tx.execute("DELETE FROM image_embedding_samples", [])?;
        tx.execute("DELETE FROM image_embedding_state", [])?;
        tx.commit()?;
        Ok(cleared)
    }

    pub fn delete_assets(&mut self, asset_ids: &[i64]) -> Result<usize> {
        validate_asset_ids(asset_ids)?;
        if asset_ids.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.transaction()?;
        let ids = std::iter::repeat_n("?", asset_ids.len()).collect::<Vec<_>>().join(",");
        let deleted = tx.execute(
            &format!("DELETE FROM image_embeddings WHERE embedding_id IN (SELECT embedding_id FROM image_embedding_samples WHERE asset_id IN ({ids}))"),
            params_from_iter(asset_ids),
        )?;
        tx.execute(
            &format!("DELETE FROM image_embedding_samples WHERE asset_id IN ({ids})"),
            params_from_iter(asset_ids),
        )?;
        tx.execute(
            &format!("DELETE FROM image_embedding_state WHERE asset_id IN ({ids})"),
            params_from_iter(asset_ids),
        )?;
        tx.commit()?;
        Ok(deleted)
    }

    /// Remove vector state whose owning catalog asset no longer exists, plus any vector that has
    /// lost its local state row after an interrupted write.
    #[tracing::instrument(level = "debug", skip(self))]
    pub fn prune_orphans(&mut self, catalog: &Path) -> Result<usize> {
        self.conn
            .execute(
                "ATTACH DATABASE ?1 AS maintenance_catalog",
                [catalog.as_str()],
            )
            .with_context(|| format!("attaching asset catalog for image pruning: {catalog}"))?;
        let result = (|| {
            let tx = self.conn.transaction()?;
            let orphan_ids = tx
                .prepare_cached(
                    r#"SELECT samples.embedding_id
                   FROM image_embedding_samples AS samples
                  WHERE NOT EXISTS (
                      SELECT 1 FROM image_embedding_state AS state
                       WHERE state.asset_id = samples.asset_id
                  ) OR NOT EXISTS (
                      SELECT 1 FROM maintenance_catalog.assets AS assets
                       WHERE assets.asset_id = samples.asset_id
                  )"#,
                )?
                .query_map([], |row| row.get::<_, i64>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let mut deleted = 0;
            {
                let mut vectors =
                    tx.prepare_cached("DELETE FROM image_embeddings WHERE embedding_id = ?1")?;
                let mut samples = tx.prepare_cached(
                    "DELETE FROM image_embedding_samples WHERE embedding_id = ?1",
                )?;
                for id in orphan_ids {
                    deleted += vectors.execute([id])?;
                    samples.execute([id])?;
                }
            }
            tx.execute(
                "DELETE FROM image_embedding_state
                 WHERE NOT EXISTS (
                     SELECT 1 FROM maintenance_catalog.assets
                     WHERE assets.asset_id = image_embedding_state.asset_id
                 )",
                [],
            )?;
            tx.commit().context("pruning orphaned image embeddings")?;
            Ok(deleted)
        })();
        self.conn
            .execute_batch("DETACH DATABASE maintenance_catalog")
            .context("detaching asset catalog after image pruning")?;
        result
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub fn maintain(&self) -> Result<()> {
        maintain(&self.conn).context("maintaining image embedding database")
    }

    /// Nearest asset neighbours of `vector` among current image and video samples, restricted to
    /// `filters`.
    ///
    /// Uses an exact cosine scan because vec0 applies `k` before joined filters, which can
    /// under-return. `total` counts current, filtered matches before `limit`.
    #[instrument(
        name = "search_image_vectors",
        level = "debug",
        skip_all,
        fields(dimensions = vector.len(), limit, total = tracing::field::Empty)
    )]
    pub fn search_vectors(
        &self,
        vector: &[f32],
        filters: &SearchFilters,
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
                       image_embeddings.timestamp_ms AS timestamp_ms,
                       image_embeddings.embedding_id AS embedding_id,
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
                         AND ((catalog.assets.media_kind = 'image' AND image_embedding_state.sampling_version = 0 AND image_embeddings.timestamp_ms IS NULL)
                           OR (catalog.assets.media_kind = 'video' AND image_embedding_state.sampling_version = {sampling_version} AND image_embeddings.timestamp_ms IS NOT NULL))
                 WHERE 1{filters}
            )"#,
            filters = bound.sql,
            sampling_version = crate::video::SAMPLING_VERSION,
        );

        // Materialize only IDs, distances and tie breakers: scalar reads from vec0 open a
        // vector blob per evaluation. Counting and sorting must reuse those scores.
        let params = bound.extend([
            (":query", Value::Blob(vector_to_blob(vector))),
            (
                ":max_distance",
                Value::Real(options.max_distance.unwrap_or(UNBOUNDED_DISTANCE)),
            ),
            (
                ":limit",
                Value::Integer(
                    i64::try_from(limit).context("search limit exceeds SQLite's integer range")?,
                ),
            ),
        ]);

        let mut statement = self.conn.prepare_cached(&format!(
            r#"{scored}
            , winners AS MATERIALIZED (
                SELECT asset_id, timestamp_ms, distance, source_modified_ns,
                       row_number() OVER (PARTITION BY asset_id ORDER BY distance ASC, timestamp_ms ASC, embedding_id ASC) AS sample_rank
                  FROM scored
            ), eligible AS (
                SELECT asset_id, timestamp_ms, distance, source_modified_ns
                  FROM winners
                 WHERE sample_rank = 1 AND distance <= :max_distance
            ), hits AS (
            SELECT asset_id, timestamp_ms, distance, source_modified_ns
              FROM eligible
             ORDER BY distance ASC, source_modified_ns DESC, asset_id ASC
             LIMIT :limit
            )
            SELECT totals.total, hits.asset_id, hits.distance, hits.timestamp_ms
              FROM (SELECT count(*) AS total FROM eligible) totals
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
                    timestamp_ms: row.get(3)?,
                });
            }
        }

        tracing::Span::current().record("total", total);
        Ok((
            usize::try_from(total).context("image vector search result count exceeds usize")?,
            results,
        ))
    }

    /// Current image-search coverage, including videos, excluding stale vectors and other roots.
    pub fn coverage(&self, filters: &SearchFilters) -> Result<(usize, usize)> {
        let bound = filters.bind(FilterScope::CATALOG_ROWS)?;
        let sql = format!(
            "SELECT count(*), count(image_embedding_state.asset_id) \
             FROM catalog.assets \
             LEFT JOIN image_embedding_state \
               ON image_embedding_state.asset_id = catalog.assets.asset_id \
              AND image_embedding_state.source_modified_ns = catalog.assets.source_modified_ns \
              AND image_embedding_state.source_size = catalog.assets.source_size \
              AND ((catalog.assets.media_kind = 'image' AND image_embedding_state.sampling_version = 0) \
                OR (catalog.assets.media_kind = 'video' AND image_embedding_state.sampling_version = {})) \
             WHERE catalog.assets.media_kind IN ('image', 'video'){}",
            crate::video::SAMPLING_VERSION,
            bound.sql
        );
        let (total, indexed): (i64, i64) =
            self.conn
                .query_row(&sql, bind_named(&bound.params).as_slice(), |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?;
        Ok((usize::try_from(total)?, usize::try_from(indexed)?))
    }

    /// Whether this asset has a current image vector or at least one current video sample.
    pub fn is_asset_indexed(&self, asset_id: i64) -> Result<bool> {
        self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM image_embedding_samples AS samples
                   JOIN image_embedding_state ON image_embedding_state.asset_id = samples.asset_id
                   JOIN catalog.assets ON catalog.assets.asset_id = samples.asset_id
                  WHERE samples.asset_id = ?1
                    AND catalog.assets.source_modified_ns = image_embedding_state.source_modified_ns
                    AND catalog.assets.source_size = image_embedding_state.source_size
                    AND ((catalog.assets.media_kind = 'image' AND image_embedding_state.sampling_version = 0 AND samples.timestamp_ms IS NULL)
                      OR (catalog.assets.media_kind = 'video' AND image_embedding_state.sampling_version = ?2 AND samples.timestamp_ms IS NOT NULL)))",
                (asset_id, crate::video::SAMPLING_VERSION),
                |row| row.get(0),
            )
            .context("checking current visual embedding coverage")
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
                   FROM image_embedding_samples AS samples \
                  INNER JOIN image_embedding_state \
                          ON image_embedding_state.asset_id = samples.asset_id \
                  INNER JOIN catalog.assets \
                          ON catalog.assets.asset_id = samples.asset_id \
                         AND catalog.assets.source_modified_ns \
                              = image_embedding_state.source_modified_ns \
                         AND catalog.assets.source_size = image_embedding_state.source_size \
                  CROSS JOIN image_embeddings
                  WHERE samples.asset_id = ?1
                    AND image_embeddings.embedding_id = samples.embedding_id
                    AND samples.timestamp_ms IS NULL
                    AND catalog.assets.media_kind = 'image'
                    AND image_embedding_state.sampling_version = 0",
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
    pub(super) fn vector_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM image_embeddings", [], |row| {
                row.get(0)
            })
            .context("counting image embeddings")?;
        usize::try_from(count).context("image embedding count exceeds usize")
    }
}

fn insert_sample(tx: &Transaction<'_>, asset_id: i64, timestamp_ms: Option<i64>) -> Result<i64> {
    tx.execute(
        "INSERT INTO image_embedding_samples(asset_id, timestamp_ms) VALUES (?1, ?2)",
        (asset_id, timestamp_ms),
    )?;
    Ok(tx.last_insert_rowid())
}

fn delete_asset_samples(tx: &Transaction<'_>, asset_id: i64) -> Result<usize> {
    let ids = tx
        .prepare_cached("SELECT embedding_id FROM image_embedding_samples WHERE asset_id = ?1")?
        .query_map([asset_id], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut deleted = 0;
    let mut vectors = tx.prepare_cached("DELETE FROM image_embeddings WHERE embedding_id = ?1")?;
    for id in ids {
        deleted += vectors.execute([id])?;
    }
    tx.execute(
        "DELETE FROM image_embedding_samples WHERE asset_id = ?1",
        [asset_id],
    )?;
    Ok(deleted)
}

fn ensure_vector_table(conn: &Connection, dimensions: usize) -> Result<()> {
    let definition = format!(
        r#"CREATE VIRTUAL TABLE image_embeddings USING vec0(
             embedding_id INTEGER PRIMARY KEY,
             asset_id INTEGER,
             +timestamp_ms INTEGER,
             embedding FLOAT[{dimensions}] distance_metric=cosine
         );"#
    );
    let old_schema: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'image_embeddings'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    match old_schema {
        None => conn
            .execute_batch(&definition)
            .context("creating the image embedding vector table"),
        Some(schema) if schema.contains("embedding_id INTEGER PRIMARY KEY") => Ok(()),
        Some(schema) => {
            if !schema.contains(&format!("embedding FLOAT[{dimensions}]")) {
                bail!(
                    "image embedding database vector width does not match the model's {dimensions} dimensions"
                );
            }
            let tx = conn.unchecked_transaction()?;
            tx.execute_batch("CREATE TEMP TABLE old_image_embeddings AS SELECT asset_id, embedding FROM image_embeddings; DROP TABLE image_embeddings;")?;
            tx.execute_batch(&definition)?;
            tx.execute_batch("INSERT INTO image_embeddings (embedding_id, asset_id, timestamp_ms, embedding) SELECT asset_id, asset_id, NULL, embedding FROM old_image_embeddings; DROP TABLE old_image_embeddings;")?;
            tx.commit()
                .context("migrating image vectors to sample rows")
        }
    }
}

/// Finish the one-time copy from either the image-only vec0 layout or this branch's video layout.
/// The marker makes an interrupted copy safe to retry without scanning vectors on later opens.
fn ensure_sample_rows(conn: &Connection) -> Result<()> {
    let complete: bool = conn.query_row(
        "SELECT complete FROM image_embedding_sample_migration WHERE singleton = 1",
        [],
        |row| row.get(0),
    )?;
    if complete {
        return Ok(());
    }
    let tx = conn.unchecked_transaction()?;
    tx.execute(
        "INSERT INTO image_embedding_samples(embedding_id, asset_id, timestamp_ms)
         SELECT embedding_id, asset_id, timestamp_ms FROM image_embeddings",
        [],
    )?;
    tx.execute(
        "UPDATE image_embedding_sample_migration SET complete = 1 WHERE singleton = 1",
        [],
    )?;
    tx.commit().context("copying image sample keys")
}

fn ensure_sample_rows_ready(conn: &Connection) -> Result<()> {
    let complete: bool = conn.query_row(
        "SELECT complete FROM image_embedding_sample_migration WHERE singleton = 1",
        [],
        |row| row.get(0),
    )?;
    if !complete {
        bail!("image embedding sample migration has not completed");
    }
    Ok(())
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
    /// Actual presentation time for the winning video frame; absent for images.
    pub timestamp_ms: Option<i64>,
}

#[cfg(test)]
mod sample_tests {
    use super::*;
    use crate::assets::AssetCatalog;
    use crate::{thumbs::ThumbnailDb, video};
    use tempfile::TempDir;

    #[test]
    fn deleted_or_partial_video_thumbnails_make_current_vectors_pending() -> Result<()> {
        let temp = TempDir::new()?;
        let (index_path, _) = paths(&temp)?;
        let thumbnail_path = PathBuf::try_from(temp.path().join("thumbnails.db"))?;
        let asset = video(1);
        let mut db = ImageIndexDb::new(&index_path, 2)?;
        db.save_video_embeddings(&asset, vec![(100, vec![1.0, 0.0]), (200, vec![0.0, 1.0])])?;
        db.save_video_embeddings(&video(2), vec![(300, vec![1.0, 0.0])])?;
        assert_eq!(
            db.video_sample_timestamps_for_assets(&HashSet::from([asset.asset_id]))?,
            std::collections::HashMap::from([(asset.asset_id, vec![100, 200])])
        );
        let cache = ThumbnailDb::new(&thumbnail_path)?;
        let samples = [100, 200].map(|timestamp_ms| video::VideoSample {
            timestamp_ms,
            raster: crate::imaging::Raster::Rgb {
                width: 2,
                height: 2,
                pixels: vec![42; 12],
            },
        });
        cache.store_video_samples(&asset, &samples)?;
        let vectors = db.current_video_asset_ids(&[(asset.asset_id, asset.fingerprint)])?;
        assert_eq!(
            super::super::videos_with_complete_thumbnails(
                &db,
                &cache,
                std::slice::from_ref(&asset),
                &vectors
            )?,
            HashSet::from([asset.asset_id]),
        );

        // A single missing size is enough to make a search timestamp unusable.
        Connection::open(&thumbnail_path)?.execute(
            "DELETE FROM video_thumbnails WHERE asset_id = ?1 AND timestamp_ms = 200 AND size_bucket = 256",
            [asset.asset_id],
        )?;
        assert!(
            super::super::videos_with_complete_thumbnails(
                &db,
                &cache,
                std::slice::from_ref(&asset),
                &vectors
            )?
            .is_empty()
        );

        cache.store_video_samples(&asset, &samples)?;
        Connection::open(&thumbnail_path)?.execute(
            "DELETE FROM thumbnails WHERE asset_id = ?1 AND size_bucket = 256",
            [asset.asset_id],
        )?;
        assert!(
            super::super::videos_with_complete_thumbnails(
                &db,
                &cache,
                std::slice::from_ref(&asset),
                &vectors
            )?
            .is_empty()
        );

        cache.store_video_samples(&asset, &samples)?;
        // Poster frame: 4 sizes plus 4 gallery thumbnails; the second frame omits 1024.
        assert_eq!(cache.delete_asset(asset.asset_id)?, 11);
        assert!(
            super::super::videos_with_complete_thumbnails(&db, &cache, &[asset], &vectors)?
                .is_empty()
        );
        Ok(())
    }

    fn paths(temp: &TempDir) -> Result<(PathBuf, PathBuf)> {
        Ok((
            PathBuf::try_from(temp.path().join("vectors.db"))?,
            PathBuf::try_from(temp.path().join("assets.db"))?,
        ))
    }

    fn video(id: i64) -> Asset {
        Asset {
            asset_id: id,
            path: PathBuf::from("C:/gallery").join(format!("video{id}.mp4")),
            fingerprint: SourceFingerprint {
                modified_ns: 10,
                size: 100,
            },
            source_created_ns: None,
            exif_taken_ns: None,
            media_kind: MediaKind::Video,
            media_format: "mp4".into(),
            width: None,
            height: None,
            is_animated: false,
            frame_count: None,
            duration_ms: None,
            metadata_version: 0,
        }
    }

    #[test]
    fn video_samples_search_by_best_frame_then_replace_and_delete() -> Result<()> {
        let temp = TempDir::new()?;
        let (index, catalog) = paths(&temp)?;
        drop(AssetCatalog::new(&catalog)?);
        let catalog_conn = Connection::open(&catalog)?;
        for id in [1, 2] {
            let asset = video(id);
            catalog_conn.execute(
                "INSERT INTO assets(asset_id, path, source_modified_ns, source_size, media_kind, media_format, is_animated) VALUES (?1, ?2, 10, 100, 'video', 'mp4', 0)",
                (id, asset.path.as_str()),
            )?;
        }
        let mut db = ImageIndexDb::new(&index, 2)?;
        db.save_video_embeddings(
            &video(1),
            vec![
                (300, vec![1.0, 0.0]),
                (100, vec![1.0, 0.0]),
                (200, vec![0.0, 1.0]),
            ],
        )?;
        db.save_video_embeddings(&video(2), vec![(400, vec![0.8, 0.6])])?;
        assert_eq!(db.vector_count()?, 4);
        let reader = ImageIndexDb::new_read_only(&index, 2, &catalog)?;
        let filters = SearchFilters::under(Path::new("C:/gallery"));
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0],
            &filters,
            1,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2);
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].asset_id, hits[0].timestamp_ms), (1, Some(100)));
        let filtered = filters.clone().with_path_contains(Some("VIDEO2.MP4"));
        let (filtered_total, filtered_hits) = reader.search_vectors(
            &[1.0, 0.0],
            &filtered,
            1,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(filtered_total, 1);
        assert_eq!(filtered_hits[0].asset_id, 2);
        assert_eq!(reader.coverage(&filters)?, (2, 2));
        assert!(reader.is_asset_indexed(1)?);
        assert!(reader.current_vector(1)?.is_none());
        drop(reader);

        db.save_video_embeddings(&video(1), vec![(900, vec![0.0, 1.0])])?;
        assert_eq!(db.vector_count()?, 2);
        let reader = ImageIndexDb::new_read_only(&index, 2, &catalog)?;
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0],
            &filters,
            2,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2);
        assert_eq!((hits[0].asset_id, hits[0].timestamp_ms), (2, Some(400)));
        drop(reader);
        assert_eq!(db.delete_assets(&[1])?, 1);
        assert_eq!(db.vector_count()?, 1);
        let reader = ImageIndexDb::new_read_only(&index, 2, &catalog)?;
        assert!(!reader.is_asset_indexed(1)?);
        assert!(reader.is_asset_indexed(2)?);
        Ok(())
    }

    #[test]
    fn legacy_image_vector_migrates_without_loss() -> Result<()> {
        let temp = TempDir::new()?;
        let (index, catalog) = paths(&temp)?;
        drop(AssetCatalog::new(&catalog)?);
        register_vector_extension();
        let conn = Connection::open(&index)?;
        conn.execute_batch("CREATE TABLE image_embedding_state(asset_id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, source_modified_ns INTEGER NOT NULL, source_size INTEGER NOT NULL); PRAGMA user_version = 2; CREATE VIRTUAL TABLE image_embeddings USING vec0(asset_id INTEGER PRIMARY KEY, embedding FLOAT[2] distance_metric=cosine);")?;
        let image_path = PathBuf::from("C:/gallery").join("a.png");
        conn.execute(
            "INSERT INTO image_embedding_state VALUES (1, ?1, 10, 100)",
            [image_path.as_str()],
        )?;
        conn.execute(
            "INSERT INTO image_embeddings(asset_id, embedding) VALUES (1, ?1)",
            [vector_to_blob(&[1.0, 0.0])],
        )?;
        drop(conn);
        Connection::open(&catalog)?.execute("INSERT INTO assets(asset_id, path, source_modified_ns, source_size, media_kind, media_format, is_animated) VALUES (1, ?1, 10, 100, 'image', 'png', 0)", [image_path.as_str()])?;
        let db = ImageIndexDb::new(&index, 2)?;
        assert_eq!(db.vector_count()?, 1);
        let reader = ImageIndexDb::new_read_only(&index, 2, &catalog)?;
        assert_eq!(reader.current_vector(1)?, Some(vec![1.0, 0.0]));
        assert!(reader.is_asset_indexed(1)?);
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0],
            &SearchFilters::under(Path::new("C:/gallery")),
            10,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 1);
        assert_eq!(hits[0].timestamp_ms, None);
        Ok(())
    }

    #[test]
    fn video_layout_migrates_sample_keys_without_reembedding() -> Result<()> {
        let temp = TempDir::new()?;
        let (index, catalog) = paths(&temp)?;
        drop(AssetCatalog::new(&catalog)?);
        register_vector_extension();
        let conn = Connection::open(&index)?;
        conn.execute_batch(
            r#"CREATE TABLE image_embedding_state(
                   asset_id INTEGER PRIMARY KEY,
                   source_path TEXT NOT NULL,
                   source_modified_ns INTEGER NOT NULL,
                   source_size INTEGER NOT NULL,
                   sampling_version INTEGER NOT NULL DEFAULT 0
               );
               PRAGMA user_version = 3;
               CREATE VIRTUAL TABLE image_embeddings USING vec0(
                   embedding_id INTEGER PRIMARY KEY,
                   asset_id INTEGER,
                   +timestamp_ms INTEGER,
                   embedding FLOAT[2] distance_metric=cosine
               );"#,
        )?;
        conn.execute(
            "INSERT INTO image_embeddings(embedding_id, asset_id, timestamp_ms, embedding) VALUES (7, 1, 200, ?1)",
            [vector_to_blob(&[1.0, 0.0])],
        )?;
        conn.execute(
            "INSERT INTO image_embeddings(embedding_id, asset_id, timestamp_ms, embedding) VALUES (8, 1, 100, ?1)",
            [vector_to_blob(&[0.0, 1.0])],
        )?;
        drop(conn);

        let db = ImageIndexDb::new(&index, 2)?;
        assert_eq!(db.vector_count()?, 2);
        assert_eq!(
            db.video_sample_timestamps_for_assets(&HashSet::from([1]))?,
            std::collections::HashMap::from([(1, vec![100, 200])])
        );
        assert_eq!(
            db.conn.query_row(
                "SELECT complete FROM image_embedding_sample_migration WHERE singleton = 1",
                [],
                |row| row.get::<_, i64>(0),
            )?,
            1
        );
        drop(db);
        ImageIndexDb::new_read_only(&index, 2, &catalog)?;
        Ok(())
    }

    #[test]
    fn decoded_video_search_hit_resolves_the_exact_stored_thumbnail() -> Result<()> {
        let temp = TempDir::new()?;
        let gallery = PathBuf::try_from(temp.path().join("gallery"))?;
        std::fs::create_dir(&gallery)?;
        let video_path = gallery.join("keyframes.mp4");
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/video/keyframes.mp4");
        std::fs::copy(fixture, &video_path)?;

        let (index_path, catalog_path) = paths(&temp)?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&video_path, &std::fs::metadata(&video_path)?)?;
        assert_eq!(asset.media_kind, MediaKind::Video);
        let samples = video::samples(&asset.path, 64, || false)?;
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.timestamp_ms)
                .collect::<Vec<_>>(),
            vec![0, 2000, 4000]
        );

        let thumbnail_path = PathBuf::try_from(temp.path().join("thumbnails.db"))?;
        let thumbnails = ThumbnailDb::new(&thumbnail_path)?;
        thumbnails.store_video_samples(&asset, &samples)?;

        let mut index = ImageIndexDb::new(&index_path, 2)?;
        let vectors = [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]];
        index.save_video_embeddings(
            &asset,
            samples
                .iter()
                .zip(vectors)
                .map(|(sample, vector)| (sample.timestamp_ms, vector.to_vec()))
                .collect(),
        )?;
        let reader = ImageIndexDb::new_read_only(&index_path, 2, &catalog_path)?;
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0],
            &SearchFilters::under(&gallery),
            10,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].asset_id, asset.asset_id);
        assert_eq!(hits[0].timestamp_ms, Some(samples[1].timestamp_ms));
        let matching = thumbnails.get_video_sample(
            hits[0].asset_id,
            hits[0].timestamp_ms.unwrap(),
            128,
            asset.fingerprint,
        )?;
        assert!(matching.is_some());
        assert!(
            thumbnails
                .get_video_sample(asset.asset_id, 1999, 128, asset.fingerprint)?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn reader_rejects_pending_vector_migration_and_writer_repairs_it() -> Result<()> {
        let temp = TempDir::new()?;
        let (index, catalog) = paths(&temp)?;
        drop(AssetCatalog::new(&catalog)?);
        register_vector_extension();
        let conn = Connection::open(&index)?;
        conn.execute_batch("CREATE TABLE image_embedding_state(asset_id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, source_modified_ns INTEGER NOT NULL, source_size INTEGER NOT NULL, sampling_version INTEGER NOT NULL DEFAULT 0); PRAGMA user_version = 3; CREATE VIRTUAL TABLE image_embeddings USING vec0(asset_id INTEGER PRIMARY KEY, embedding FLOAT[2] distance_metric=cosine);")?;
        drop(conn);
        let error = ImageIndexDb::new_read_only(&index, 2, &catalog)
            .err()
            .context("reader unexpectedly accepted incomplete migration")?;
        assert!(error.to_string().contains("schema version 3"));
        drop(ImageIndexDb::new(&index, 2)?);
        ImageIndexDb::new_read_only(&index, 2, &catalog)?;
        Ok(())
    }
}
