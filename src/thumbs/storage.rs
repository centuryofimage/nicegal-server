//! Persistence for thumbnail variants; generation and concurrency live in the service module.
use super::{
    DecodedThumbnail, Thumbnail, ThumbnailEncoding, validate_key, validate_static_thumbnail,
};
use crate::assets::{Asset, MediaKind, SourceFingerprint};
use crate::schema::{check_schema_read_only, open_schema_with_migrations};
use crate::storage::{
    READ_ONLY_FLAGS, configure_reader, configure_writer, maintain, validate_asset_ids,
};
use crate::{poster, video};
use anyhow::{Context, Result, bail};
use camino::Utf8Path as Path;
use rusqlite::{Connection, OptionalExtension};
use tracing::{debug_span, field, trace_span};
// Keep this in sync with THUMBNAIL_SCHEMA_VERSION in the Electron frontend's
// src/main/backend/thumbnail-reader.ts; it reads this database directly.
const SCHEMA_VERSION: i32 = 4;
const SCHEMA_LABEL: &str = "thumbnail database";
const MIGRATIONS: &[(i32, &str)] = &[(3, "VACUUM;")];
const VIDEO_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS video_thumbnails(
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
CREATE INDEX IF NOT EXISTS video_thumbnail_assets_idx ON video_thumbnails(asset_id);";

pub struct ThumbnailDb {
    conn: Connection,
}

/// Encoded poster buckets are prepared off the SQLite writer thread.
pub struct EncodedVideoSamples {
    rows: Vec<(i64, Vec<poster::StaticPoster>)>,
}

pub fn encode_video_samples(
    asset: &Asset,
    samples: &[video::VideoSample],
) -> Result<EncodedVideoSamples> {
    encode_video_samples_with_output(asset, samples, video::VideoOutputOptions::default())
}

pub fn encode_video_samples_with_output(
    asset: &Asset,
    samples: &[video::VideoSample],
    output: video::VideoOutputOptions,
) -> Result<EncodedVideoSamples> {
    if asset.media_kind != MediaKind::Video {
        bail!("asset {} is not a video", asset.asset_id);
    }
    if asset.asset_id <= 0 {
        bail!("asset identifier must be greater than zero");
    }
    if samples.is_empty() {
        bail!("video has no sample frames");
    }
    let span = debug_span!("video_sample_encode", path = %asset.path, samples = samples.len());
    let _entered = span.enter();
    let mut rows = Vec::with_capacity(samples.len());
    let mut seen_timestamps = std::collections::HashSet::new();
    for sample in samples {
        if sample.timestamp_ms < 0 {
            bail!("video sample timestamp must be nonnegative");
        }
        if !seen_timestamps.insert(sample.timestamp_ms) {
            continue;
        }
        let posters = poster::raster_buckets_with_output_at(
            &asset.path,
            &sample.raster,
            &super::SIZE_BUCKETS,
            output,
        )?;
        rows.push((sample.timestamp_ms, posters));
    }
    Ok(EncodedVideoSamples { rows })
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
        // Version 4 is also consumed directly by Electron. Additive table creation keeps its
        // existing reader compatible with databases created before video thumbnails existed.
        conn.execute_batch(VIDEO_TABLE_SQL)?;
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

    /// Replace all indexed samples for this video and refresh its default gallery poster.
    /// Each timestamp has every public size bucket, encoded from the decoded raster.
    pub fn store_video_samples(&self, asset: &Asset, samples: &[video::VideoSample]) -> Result<()> {
        self.store_video_samples_with_output(asset, samples, video::VideoOutputOptions::default())
    }

    pub fn store_video_samples_with_output(
        &self,
        asset: &Asset,
        samples: &[video::VideoSample],
        output: video::VideoOutputOptions,
    ) -> Result<()> {
        let encoded = encode_video_samples_with_output(asset, samples, output)?;
        self.store_encoded_video_samples(asset, &encoded)
    }

    pub fn store_encoded_video_samples(
        &self,
        asset: &Asset,
        encoded: &EncodedVideoSamples,
    ) -> Result<()> {
        let span = debug_span!("video_sample_store", path = %asset.path, asset_id = asset.asset_id, samples = encoded.rows.len());
        let _entered = span.enter();
        if asset.media_kind != MediaKind::Video {
            bail!("asset {} is not a video", asset.asset_id);
        }
        if asset.asset_id <= 0 {
            bail!("asset identifier must be greater than zero");
        }
        if encoded.rows.is_empty() {
            bail!("video has no sample frames");
        }
        let source_size = i64::try_from(asset.fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let write_span =
            debug_span!("video_sample_write", path = %asset.path, samples = encoded.rows.len());
        let _writing = write_span.enter();
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM video_thumbnails WHERE asset_id = ?1",
            [asset.asset_id],
        )?;
        {
            let mut insert = tx.prepare_cached(
                "INSERT INTO video_thumbnails (asset_id, timestamp_ms, size_bucket, sampling_version, source_modified_ns, source_size, width, height, encoding, data) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"
            )?;
            for (timestamp_ms, posters) in &encoded.rows {
                for (&bucket, poster) in super::SIZE_BUCKETS.iter().zip(posters) {
                    insert.execute((
                        asset.asset_id,
                        timestamp_ms,
                        bucket,
                        video::SAMPLING_VERSION,
                        asset.fingerprint.modified_ns,
                        source_size,
                        poster.width,
                        poster.height,
                        poster.encoding.content_type(),
                        &poster.data,
                    ))?;
                }
            }
        }
        // The first indexed frame is also the standard gallery poster, readable by the
        // unchanged desktop thumbnail reader.
        for (&bucket, poster) in super::SIZE_BUCKETS.iter().zip(&encoded.rows[0].1) {
            tx.execute(
                "INSERT INTO thumbnails (asset_id, size_bucket, generator_version, source_modified_ns, source_size, width, height, encoding, data) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) ON CONFLICT(asset_id, size_bucket, generator_version) DO UPDATE SET source_modified_ns=excluded.source_modified_ns, source_size=excluded.source_size, width=excluded.width, height=excluded.height, encoding=excluded.encoding, data=excluded.data",
                (asset.asset_id, bucket, super::GENERATOR_VERSION, asset.fingerprint.modified_ns,
                    source_size, poster.width, poster.height, poster.encoding.content_type(), &poster.data),
            )?;
        }
        tx.commit().context("committing video thumbnails")?;
        Ok(())
    }

    /// Read an exact indexed video frame without opening or decoding the source video.
    pub fn get_video_sample(
        &self,
        asset_id: i64,
        timestamp_ms: i64,
        requested_physical_size: u32,
        fingerprint: SourceFingerprint,
    ) -> Result<Option<Thumbnail>> {
        if asset_id <= 0 || timestamp_ms < 0 || requested_physical_size == 0 {
            bail!("invalid video sample thumbnail key");
        }
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let row = self.conn.query_row(
            "SELECT size_bucket, width, height, encoding, data FROM video_thumbnails WHERE asset_id=?1 AND timestamp_ms=?2 AND sampling_version=?3 AND source_modified_ns=?4 AND source_size=?5 ORDER BY CASE WHEN size_bucket >= ?6 THEN 0 ELSE 1 END, CASE WHEN size_bucket >= ?6 THEN size_bucket END ASC, CASE WHEN size_bucket < ?6 THEN size_bucket END DESC LIMIT 1",
            (asset_id, timestamp_ms, video::SAMPLING_VERSION, fingerprint.modified_ns, source_size, i64::from(requested_physical_size)),
            |row| Ok((row.get::<_, u16>(0)?, row.get::<_, u32>(1)?, row.get::<_, u32>(2)?, row.get::<_, String>(3)?, row.get::<_, Vec<u8>>(4)?)),
        ).optional()?;
        row.map(|(size_bucket, width, height, encoding, data)| {
            Ok(Thumbnail {
                size_bucket,
                generator_version: super::GENERATOR_VERSION,
                width,
                height,
                encoding: ThumbnailEncoding::parse(&encoding)?,
                data,
            })
        })
        .transpose()
    }

    /// Whether every indexed frame still has all persisted sizes and the gallery poster.
    pub(crate) fn has_current_video_samples(
        &self,
        asset: &Asset,
        timestamps: &[i64],
    ) -> Result<bool> {
        if timestamps.is_empty() {
            return Ok(false);
        }
        let source_size = i64::try_from(asset.fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let mut statement = self.conn.prepare_cached(
            "SELECT timestamp_ms, size_bucket FROM video_thumbnails WHERE asset_id = ?1 AND sampling_version = ?2 AND source_modified_ns = ?3 AND source_size = ?4",
        )?;
        let variants = statement
            .query_map(
                (
                    asset.asset_id,
                    video::SAMPLING_VERSION,
                    asset.fingerprint.modified_ns,
                    source_size,
                ),
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, u16>(1)?)),
            )?
            .collect::<rusqlite::Result<std::collections::HashSet<_>>>()?;
        if !timestamps.iter().all(|timestamp| {
            super::SIZE_BUCKETS
                .iter()
                .all(|bucket| variants.contains(&(*timestamp, *bucket)))
        }) {
            return Ok(false);
        }
        let mut poster = self.conn.prepare_cached(
            "SELECT size_bucket FROM thumbnails WHERE asset_id = ?1 AND generator_version = ?2 AND source_modified_ns = ?3 AND source_size = ?4",
        )?;
        let poster_buckets = poster
            .query_map(
                (
                    asset.asset_id,
                    super::GENERATOR_VERSION,
                    asset.fingerprint.modified_ns,
                    source_size,
                ),
                |row| row.get::<_, u16>(0),
            )?
            .collect::<rusqlite::Result<std::collections::HashSet<_>>>()?;
        Ok(super::SIZE_BUCKETS
            .iter()
            .all(|bucket| poster_buckets.contains(bucket)))
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
        let mut deleted = {
            let mut statement =
                transaction.prepare("DELETE FROM thumbnails WHERE asset_id = ?1")?;
            asset_ids.iter().try_fold(0usize, |deleted, asset_id| {
                statement.execute([asset_id]).map(|count| deleted + count)
            })?
        };
        {
            let mut statement =
                transaction.prepare("DELETE FROM video_thumbnails WHERE asset_id = ?1")?;
            for asset_id in asset_ids {
                deleted += statement.execute([asset_id])?;
            }
        }
        transaction
            .commit()
            .context("committing asset thumbnail deletions")?;
        Ok(deleted)
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub fn prune_orphans(&self, catalog: &Path) -> Result<usize> {
        self.conn
            .execute(
                "ATTACH DATABASE ?1 AS maintenance_catalog",
                [catalog.as_str()],
            )
            .with_context(|| format!("attaching asset catalog for thumbnail pruning: {catalog}"))?;
        // Resolve orphan IDs from the covering index before touching blob-bearing table rows.
        // A direct DELETE with NOT EXISTS visits every WITHOUT ROWID table row even when none match.
        let result = self
            .conn
            .execute(
                "DELETE FROM thumbnails
                 WHERE asset_id IN (
                     SELECT asset_id FROM thumbnails AS candidates
                     WHERE NOT EXISTS (
                         SELECT 1 FROM maintenance_catalog.assets
                         WHERE assets.asset_id = candidates.asset_id
                     )
                 )",
                [],
            )
            .context("pruning orphaned thumbnails");
        let video_result = self
            .conn
            .execute(
                "DELETE FROM video_thumbnails
             WHERE asset_id IN (
                 SELECT DISTINCT candidates.asset_id
                   FROM video_thumbnails AS candidates INDEXED BY video_thumbnail_assets_idx
                  WHERE NOT EXISTS (
                      SELECT 1 FROM maintenance_catalog.assets
                       WHERE assets.asset_id = candidates.asset_id
                  )
             )",
                [],
            )
            .context("pruning orphaned video thumbnails");
        self.conn
            .execute_batch("DETACH DATABASE maintenance_catalog")
            .context("detaching asset catalog after thumbnail pruning")?;
        Ok(result? + video_result?)
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub fn maintain(&self) -> Result<()> {
        maintain(&self.conn).context("maintaining thumbnail database")
    }
}
