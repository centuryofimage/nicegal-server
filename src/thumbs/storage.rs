//! Persistence for thumbnail variants; generation and concurrency live in the service module.
use super::{
    DecodedThumbnail, Thumbnail, ThumbnailEncoding, validate_key, validate_static_thumbnail,
};
use crate::assets::SourceFingerprint;
use crate::schema::{check_schema_read_only, open_schema_with_migrations};
use crate::storage::{
    READ_ONLY_FLAGS, configure_reader, configure_writer, maintain, validate_asset_ids,
};
use anyhow::{Context, Result, bail};
use camino::Utf8Path as Path;
use rusqlite::{Connection, OptionalExtension};
use tracing::{debug_span, field, trace_span};
// Keep this in sync with THUMBNAIL_SCHEMA_VERSION in the Electron frontend's
// src/main/backend/thumbnail-reader.ts; it reads this database directly.
const SCHEMA_VERSION: i32 = 4;
const SCHEMA_LABEL: &str = "thumbnail database";
const MIGRATIONS: &[(i32, &str)] = &[(3, "VACUUM;")];

pub struct ThumbnailDb {
    conn: Connection,
}

impl ThumbnailDb {
    pub fn new(path: &Path) -> Result<Self> {
        let span = debug_span!("thumbnail_db_open", path = %path, read_only = false);
        let _entered = span.enter();
        let conn = Connection::open(path)
            .with_context(|| format!("opening thumbnail database: {path}"))?;
        configure_writer(&conn)?;
        open_schema_with_migrations(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("../thumbs_create.sql"),
            MIGRATIONS,
        )?;
        Ok(Self { conn })
    }

    /// Open a query-only connection for serving stored variants. Unlike [`ThumbnailDb::new`] this
    /// never creates the schema, so a reader cannot bring an empty database into existence.
    pub fn new_read_only(path: &Path) -> Result<Self> {
        let span = debug_span!("thumbnail_db_open", path = %path, read_only = true);
        let _entered = span.enter();
        let conn = Connection::open_with_flags(path, READ_ONLY_FLAGS)
            .with_context(|| format!("opening thumbnail database read-only: {path}"))?;
        configure_reader(&conn)?;
        check_schema_read_only(&conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
        Ok(Self { conn })
    }

    /// Store or replace exactly one size and generator variant.
    pub fn put(&self, thumbnail: DecodedThumbnail) -> Result<()> {
        self.store_batch(&[thumbnail])
    }

    /// Persist a bounded decoded batch atomically. Every row is validated before the transaction
    /// begins, so malformed input cannot leave an earlier row from the same batch committed.
    pub fn store_batch(&self, thumbnails: &[DecodedThumbnail]) -> Result<()> {
        for thumbnail in thumbnails {
            validate_key(
                thumbnail.asset_id,
                thumbnail.size_bucket,
                thumbnail.generator_version,
            )?;
            validate_static_thumbnail(
                thumbnail.size_bucket,
                thumbnail.width,
                thumbnail.height,
                thumbnail.encoding,
                &thumbnail.data,
            )?;
            i64::try_from(thumbnail.fingerprint.size)
                .context("source byte size exceeds SQLite's integer range")?;
        }
        if thumbnails.is_empty() {
            return Ok(());
        }

        let span = debug_span!(
            "thumbnail_store_batch",
            variants = thumbnails.len(),
            data_bytes = thumbnails
                .iter()
                .map(|thumbnail| thumbnail.data.len())
                .sum::<usize>(),
        );
        let _entered = span.enter();
        let transaction = self.conn.unchecked_transaction()?;
        let mut statement = transaction.prepare_cached(
            "INSERT INTO thumbnails (
                 asset_id, size_bucket, generator_version, source_modified_ns,
                 source_size, width, height, encoding, data
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(asset_id, size_bucket, generator_version) DO UPDATE SET
                 source_modified_ns = excluded.source_modified_ns,
                 source_size = excluded.source_size,
                 width = excluded.width,
                 height = excluded.height,
                 encoding = excluded.encoding,
                 data = excluded.data",
        )?;
        for thumbnail in thumbnails {
            statement.execute((
                thumbnail.asset_id,
                thumbnail.size_bucket,
                thumbnail.generator_version,
                thumbnail.fingerprint.modified_ns,
                i64::try_from(thumbnail.fingerprint.size)?,
                thumbnail.width,
                thumbnail.height,
                thumbnail.encoding.content_type(),
                &thumbnail.data,
            ))?;
        }
        drop(statement);
        transaction.commit().context("committing thumbnail batch")?;
        Ok(())
    }

    pub fn has_current(
        &self,
        asset_id: i64,
        size_bucket: u16,
        generator_version: u32,
        fingerprint: SourceFingerprint,
    ) -> Result<bool> {
        validate_key(asset_id, size_bucket, generator_version)?;
        let span = trace_span!(
            "thumbnail_cache_lookup",
            asset_id,
            size_bucket,
            generator_version,
            current = field::Empty,
        );
        let _entered = span.enter();
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let current = self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM thumbnails WHERE asset_id = ?1 AND size_bucket = ?2 AND generator_version = ?3 AND source_modified_ns = ?4 AND source_size = ?5)",
                (
                    asset_id,
                    size_bucket,
                    generator_version,
                    fingerprint.modified_ns,
                    source_size,
                ),
                |row| row.get(0),
            )
            .context("checking current thumbnail variant")?;
        span.record("current", current);
        Ok(current)
    }

    /// Remove variants from older generators after the current generator has been backfilled.
    pub fn sweep_old_generators(&self, generator_version: u32) -> Result<usize> {
        if generator_version == 0 {
            bail!("generator version must be greater than zero");
        }
        self.conn
            .execute(
                "DELETE FROM thumbnails WHERE generator_version <> ?1",
                [generator_version],
            )
            .context("sweeping stale thumbnail generator versions")
    }

    /// Remove variants from older generators for the specified catalog assets only.
    pub fn sweep_old_generators_for_assets(
        &self,
        generator_version: u32,
        asset_ids: impl IntoIterator<Item = i64>,
    ) -> Result<usize> {
        if generator_version == 0 {
            bail!("generator version must be greater than zero");
        }
        let transaction = self.conn.unchecked_transaction()?;
        let mut statement = transaction.prepare_cached(
            "DELETE FROM thumbnails WHERE asset_id = ?1 AND generator_version <> ?2",
        )?;
        let mut deleted = 0;
        for asset_id in asset_ids {
            if asset_id <= 0 {
                bail!("asset identifier must be greater than zero");
            }
            deleted += statement.execute((asset_id, generator_version))?;
        }
        drop(statement);
        transaction
            .commit()
            .context("sweeping scoped stale thumbnail generators")?;
        Ok(deleted)
    }

    /// Select the smallest current bucket that satisfies the requested physical size.
    /// If none is large enough, return the largest current smaller bucket.
    pub fn get(
        &self,
        asset_id: i64,
        requested_physical_size: u32,
        generator_version: u32,
        fingerprint: SourceFingerprint,
    ) -> Result<Option<Thumbnail>> {
        if asset_id <= 0 {
            bail!("asset identifier must be greater than zero");
        }
        if requested_physical_size == 0 {
            bail!("requested physical size must be greater than zero");
        }
        if generator_version == 0 {
            bail!("generator version must be greater than zero");
        }
        let span = trace_span!(
            "thumbnail_cache_get",
            asset_id,
            requested_physical_size,
            generator_version,
            hit = field::Empty,
            size_bucket = field::Empty,
            data_bytes = field::Empty,
        );
        let _entered = span.enter();
        let requested = i64::from(requested_physical_size);
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let thumbnail = self.conn
            .query_row(
                "SELECT size_bucket, generator_version, width, height, encoding, data \
                 FROM thumbnails \
                 WHERE asset_id = ?1 AND generator_version = ?2 AND source_modified_ns = ?3 AND source_size = ?4 \
                 ORDER BY CASE WHEN size_bucket >= ?5 THEN 0 ELSE 1 END, \
                          CASE WHEN size_bucket >= ?5 THEN size_bucket END ASC, \
                          CASE WHEN size_bucket < ?5 THEN size_bucket END DESC \
                 LIMIT 1",
                (
                    asset_id,
                    generator_version,
                    fingerprint.modified_ns,
                    source_size,
                    requested,
                ),
                |row| {
                    let encoding: String = row.get(4)?;
                    Ok((
                        row.get::<_, u16>(0)?,
                        row.get::<_, u32>(1)?,
                        row.get::<_, u32>(2)?,
                        row.get::<_, u32>(3)?,
                        encoding,
                        row.get::<_, Vec<u8>>(5)?,
                    ))
                },
            )
            .optional()?
            .map(|(size_bucket, generator_version, width, height, encoding, data)| -> Result<Thumbnail> {
                Ok(Thumbnail {
                    size_bucket,
                    generator_version,
                    width,
                    height,
                    encoding: ThumbnailEncoding::parse(&encoding)?,
                    data,
                })
            })
            .transpose()
            .context("loading thumbnail variant")?;
        if let Some(thumbnail) = &thumbnail {
            span.record("hit", true);
            span.record("size_bucket", thumbnail.size_bucket);
            span.record("data_bytes", thumbnail.data.len());
        } else {
            span.record("hit", false);
        }
        Ok(thumbnail)
    }

    /// Delete every size and generator variant belonging to an asset.
    pub fn delete_asset(&self, asset_id: i64) -> Result<usize> {
        self.delete_assets(&[asset_id])
    }

    pub fn delete_assets(&self, asset_ids: &[i64]) -> Result<usize> {
        validate_asset_ids(asset_ids)?;
        if asset_ids.is_empty() {
            return Ok(0);
        }
        let transaction = self.conn.unchecked_transaction()?;
        let deleted = {
            let mut statement =
                transaction.prepare("DELETE FROM thumbnails WHERE asset_id = ?1")?;
            asset_ids.iter().try_fold(0usize, |deleted, asset_id| {
                statement.execute([asset_id]).map(|count| deleted + count)
            })?
        };
        transaction
            .commit()
            .context("committing asset thumbnail deletions")?;
        Ok(deleted)
    }

    pub fn prune_orphans(&self, catalog: &Path) -> Result<usize> {
        self.conn
            .execute(
                "ATTACH DATABASE ?1 AS maintenance_catalog",
                [catalog.as_str()],
            )
            .with_context(|| format!("attaching asset catalog for thumbnail pruning: {catalog}"))?;
        let result = self
            .conn
            .execute(
                "DELETE FROM thumbnails
                 WHERE NOT EXISTS (
                     SELECT 1 FROM maintenance_catalog.assets
                     WHERE assets.asset_id = thumbnails.asset_id
                 )",
                [],
            )
            .context("pruning orphaned thumbnails");
        self.conn
            .execute_batch("DETACH DATABASE maintenance_catalog")
            .context("detaching asset catalog after thumbnail pruning")?;
        result
    }

    pub fn maintain(&self) -> Result<()> {
        maintain(&self.conn).context("maintaining thumbnail database")
    }
}
