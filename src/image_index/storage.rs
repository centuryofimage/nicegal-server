use crate::assets::SourceFingerprint;
use crate::db::{
    FilterScope, SearchFilters, UNBOUNDED_DISTANCE, bind_named, configure_vector_reads,
    register_glob, register_vector_extension, vector_to_blob,
};
use crate::schema::{check_schema_read_only, open_schema_with_migrations};
use crate::storage::{
    READ_ONLY_FLAGS, configure_reader, configure_writer, maintain, validate_asset_ids,
};
use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;
use std::ops::Deref;
use tracing::instrument;

const SCHEMA_VERSION: i32 = 2;
const SCHEMA_LABEL: &str = "image embedding database";
const MIGRATIONS: &[(i32, &str)] = &[(1, "VACUUM;")];
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
        configure_writer(&conn)?;
        configure_vector_reads(&conn)?;
        open_schema_with_migrations(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("../image_index_create.sql"),
            MIGRATIONS,
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
             )",
            fingerprints,
        )
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
        validate_asset_ids(asset_ids)?;
        let tx = self.conn.transaction()?;
        let mut deleted = 0;
        {
            let mut vectors =
                tx.prepare_cached("DELETE FROM image_embeddings WHERE asset_id = ?1")?;
            let mut state =
                tx.prepare_cached("DELETE FROM image_embedding_state WHERE asset_id = ?1")?;
            for asset_id in asset_ids {
                deleted += vectors.execute([asset_id])?;
                state.execute([asset_id])?;
            }
        }
        tx.commit()?;
        Ok(deleted)
    }

    /// Remove vector state whose owning catalog asset no longer exists, plus any vector that has
    /// lost its local state row after an interrupted write.
    pub fn prune_orphans(&mut self, catalog: &Path) -> Result<usize> {
        self.conn
            .execute(
                "ATTACH DATABASE ?1 AS maintenance_catalog",
                [catalog.as_str()],
            )
            .with_context(|| format!("attaching asset catalog for image pruning: {catalog}"))?;
        let result = (|| {
            let tx = self.conn.transaction()?;
            let deleted = tx.execute(
                "DELETE FROM image_embeddings
                 WHERE NOT EXISTS (
                     SELECT 1 FROM image_embedding_state
                     WHERE image_embedding_state.asset_id = image_embeddings.asset_id
                 ) OR NOT EXISTS (
                     SELECT 1 FROM maintenance_catalog.assets
                     WHERE assets.asset_id = image_embeddings.asset_id
                 )",
                [],
            )?;
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

    pub fn maintain(&self) -> Result<()> {
        maintain(&self.conn).context("maintaining image embedding database")
    }

    /// Nearest neighbours of `vector` among this model's current image embeddings, restricted to
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

    /// Current image-search coverage, excluding videos, stale vectors and other roots.
    pub fn coverage(&self, filters: &SearchFilters<'_>) -> Result<(usize, usize)> {
        let bound = filters.bind(FilterScope::CATALOG_ROWS)?;
        let sql = format!(
            "SELECT count(*), count(image_embedding_state.asset_id) \
             FROM catalog.assets \
             LEFT JOIN image_embedding_state \
               ON image_embedding_state.asset_id = catalog.assets.asset_id \
              AND image_embedding_state.source_modified_ns = catalog.assets.source_modified_ns \
              AND image_embedding_state.source_size = catalog.assets.source_size \
             WHERE catalog.assets.media_kind = 'image'{}",
            bound.sql
        );
        let (total, indexed): (i64, i64) =
            self.conn
                .query_row(&sql, bind_named(&bound.params).as_slice(), |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?;
        Ok((usize::try_from(total)?, usize::try_from(indexed)?))
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
