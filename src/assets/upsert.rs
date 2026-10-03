use std::collections::HashMap;
use std::fs;
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};

use super::media::{
    MediaProbe, is_catalog_media, is_catalog_video, metadata_version_for, probe_media,
};
use super::{ASSET_COLUMNS, Asset, AssetCatalog, MediaKind, asset_from_row, timestamp_ns};
use crate::scope::PathScope;
use crate::storage::bind_named;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceFingerprint {
    pub modified_ns: i64,
    pub size: u64,
}

impl SourceFingerprint {
    pub fn from_metadata(metadata: &fs::Metadata) -> Result<Self> {
        let modified = metadata
            .modified()
            .context("reading source modification time")?
            .duration_since(UNIX_EPOCH)
            .context("source modification time predates the Unix epoch")?;
        Ok(Self {
            modified_ns: i64::try_from(modified.as_nanos())
                .context("source modification time exceeds SQLite's integer range")?,
            size: metadata.len(),
        })
    }
}

/// Metadata needed to classify a filesystem entry without opening its media bytes.
pub(crate) struct CatalogScanEntry {
    fingerprint: SourceFingerprint,
    source_created_ns: Option<i64>,
    metadata_version: i32,
    media_kind: MediaKind,
}

impl CatalogScanEntry {
    pub(crate) fn is_current(&self, metadata: &fs::Metadata) -> Result<bool> {
        Ok(
            self.fingerprint == SourceFingerprint::from_metadata(metadata)?
                && self.source_created_ns == timestamp_ns(metadata.created().ok())
                && self.metadata_version == metadata_version_for(self.media_kind),
        )
    }
}

#[derive(Debug, Default)]
pub(crate) struct CatalogUpsertTimings {
    pub fingerprint: Duration,
    pub lookup: Duration,
    pub probe: Duration,
    pub dimensions: Duration,
    pub exif: Duration,
    pub animation: Duration,
    pub store: Duration,
    pub transaction_begin: Duration,
    pub row_upsert: Duration,
    pub revision_update: Duration,
    pub commit: Duration,
    pub unchanged: bool,
}

impl CatalogUpsertTimings {
    pub fn record(&self, span: &tracing::Span) {
        for (name, duration) in [
            ("fingerprint_us", self.fingerprint),
            ("lookup_us", self.lookup),
            ("probe_us", self.probe),
            ("dimensions_us", self.dimensions),
            ("exif_us", self.exif),
            ("animation_us", self.animation),
            ("store_us", self.store),
            ("transaction_begin_us", self.transaction_begin),
            ("row_upsert_us", self.row_upsert),
            ("revision_update_us", self.revision_update),
            ("commit_us", self.commit),
        ] {
            span.record(
                name,
                u64::try_from(duration.as_micros()).unwrap_or(u64::MAX),
            );
        }
    }

    pub fn accumulate(&mut self, timings: Self) {
        self.fingerprint += timings.fingerprint;
        self.lookup += timings.lookup;
        self.probe += timings.probe;
        self.dimensions += timings.dimensions;
        self.exif += timings.exif;
        self.animation += timings.animation;
        self.store += timings.store;
        self.transaction_begin += timings.transaction_begin;
        self.row_upsert += timings.row_upsert;
        self.revision_update += timings.revision_update;
        self.commit += timings.commit;
    }
}

pub(crate) enum PreparedCatalogAsset {
    Unchanged(Asset),
    MetadataChanged(Asset),
    Changed(PreparedCatalogChange),
}

pub(crate) struct PreparedCatalogChange {
    path: PathBuf,
    fingerprint: SourceFingerprint,
    source_created_ns: Option<i64>,
    probe: MediaProbe,
    source_size: i64,
    frame_count: Option<i64>,
    duration_ms: Option<i64>,
}

impl AssetCatalog {
    /// Insert or refresh an asset while preserving the ID already assigned to its path.
    /// Optional media probing failures produce nullable metadata rather than dropping the asset.
    pub fn upsert(&self, path: &Path, metadata: &fs::Metadata) -> Result<Asset> {
        let (prepared, _prepare_timings) = self.prepare_upsert_timed(path, metadata)?;
        let (mut assets, _store_timings) = self.store_prepared_batch(vec![prepared])?;
        Ok(assets
            .pop()
            .expect("a one-row catalog batch must return one asset"))
    }

    pub(crate) fn prepare_upsert_timed(
        &self,
        path: &Path,
        metadata: &fs::Metadata,
    ) -> Result<(PreparedCatalogAsset, CatalogUpsertTimings)> {
        self.prepare_upsert_timed_with_probe(path, metadata, |path| {
            crate::video::probe_catalog(path).ok()
        })
    }

    /// Avoid scheduling a probe for a video whose catalog metadata is already current.
    pub(crate) fn video_needs_probe(&self, path: &Path, metadata: &fs::Metadata) -> Result<bool> {
        let fingerprint = SourceFingerprint::from_metadata(metadata)?;
        Ok(!self.get_by_path(path)?.is_some_and(|existing| {
            existing.fingerprint == fingerprint
                && existing.metadata_version == metadata_version_for(existing.media_kind)
        }))
    }

    /// `path` is stored as spelled. The catalog walk builds it from a library's canonical root
    /// and the names it lists, so a linked folder's files keep the link's path and the link acts
    /// like a folder. A file reachable through two paths is two assets.
    pub(crate) fn prepare_upsert_timed_with_probe(
        &self,
        path: &Path,
        metadata: &fs::Metadata,
        video_probe: impl FnOnce(&Path) -> Option<crate::video::VideoMetadata>,
    ) -> Result<(PreparedCatalogAsset, CatalogUpsertTimings)> {
        if !path.is_absolute() {
            bail!("asset path must be absolute: {path}");
        }
        if !is_catalog_media(path) {
            bail!("unsupported media format: {path}");
        }
        let mut timings = CatalogUpsertTimings::default();
        let path = path.to_owned();

        let started = Instant::now();
        let fingerprint = SourceFingerprint::from_metadata(metadata)?;
        timings.fingerprint = started.elapsed();
        let source_created_ns = timestamp_ns(metadata.created().ok());

        let started = Instant::now();
        if let Some(mut existing) = self.get_by_path(&path)?
            && existing.fingerprint == fingerprint
            && existing.metadata_version == metadata_version_for(existing.media_kind)
        {
            timings.lookup = started.elapsed();
            if existing.source_created_ns == source_created_ns {
                timings.unchanged = true;
                return Ok((PreparedCatalogAsset::Unchanged(existing), timings));
            }
            existing.source_created_ns = source_created_ns;
            return Ok((PreparedCatalogAsset::MetadataChanged(existing), timings));
        }
        timings.lookup = started.elapsed();

        let started = Instant::now();
        let (probe, probe_timings) = probe_media(
            &path,
            if is_catalog_video(&path) {
                Some(video_probe(&path))
            } else {
                None
            },
        );
        timings.probe = started.elapsed();
        timings.dimensions = probe_timings.dimensions;
        timings.exif = probe_timings.exif;
        timings.animation = probe_timings.animation;
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let frame_count = probe.frame_count.map(i64::from);
        let duration_ms = probe
            .duration_ms
            .map(i64::try_from)
            .transpose()
            .context("media duration exceeds SQLite's integer range")?;

        Ok((
            PreparedCatalogAsset::Changed(PreparedCatalogChange {
                path,
                fingerprint,
                source_created_ns,
                probe,
                source_size,
                frame_count,
                duration_ms,
            }),
            timings,
        ))
    }

    pub(crate) fn store_prepared_batch(
        &self,
        prepared: Vec<PreparedCatalogAsset>,
    ) -> Result<(Vec<Asset>, CatalogUpsertTimings)> {
        if prepared
            .iter()
            .all(|asset| matches!(asset, PreparedCatalogAsset::Unchanged(_)))
        {
            return Ok((
                prepared
                    .into_iter()
                    .map(|asset| match asset {
                        PreparedCatalogAsset::Unchanged(asset) => asset,
                        PreparedCatalogAsset::MetadataChanged(_)
                        | PreparedCatalogAsset::Changed(_) => unreachable!(),
                    })
                    .collect(),
                CatalogUpsertTimings::default(),
            ));
        }

        let mut timings = CatalogUpsertTimings::default();
        let started = Instant::now();
        let tx = self.conn.unchecked_transaction()?;
        timings.transaction_begin = started.elapsed();

        let started = Instant::now();
        let mut statement = tx.prepare(
            "INSERT INTO assets (
                 path, source_modified_ns, source_created_ns, exif_taken_ns, source_size,
                 media_kind, media_format, width, height, is_animated, frame_count,
                 duration_ms, metadata_version
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
             ON CONFLICT(path) DO UPDATE SET
                 source_modified_ns = excluded.source_modified_ns,
                 source_created_ns = excluded.source_created_ns,
                 exif_taken_ns = excluded.exif_taken_ns,
                 source_size = excluded.source_size,
                 media_kind = excluded.media_kind,
                 media_format = excluded.media_format,
                 width = excluded.width,
                 height = excluded.height,
                 is_animated = excluded.is_animated,
                 frame_count = excluded.frame_count,
                 duration_ms = excluded.duration_ms,
                 metadata_version = excluded.metadata_version
             RETURNING asset_id",
        )?;
        let mut metadata_statement =
            tx.prepare("UPDATE assets SET source_created_ns = ?1 WHERE asset_id = ?2")?;
        let mut assets = Vec::with_capacity(prepared.len());
        for prepared in prepared {
            let change = match prepared {
                PreparedCatalogAsset::Unchanged(asset) => {
                    assets.push(asset);
                    continue;
                }
                PreparedCatalogAsset::MetadataChanged(asset) => {
                    metadata_statement.execute((asset.source_created_ns, asset.asset_id))?;
                    assets.push(asset);
                    continue;
                }
                PreparedCatalogAsset::Changed(change) => change,
            };
            let asset_id = statement
                .query_row(
                    (
                        change.path.as_str(),
                        change.fingerprint.modified_ns,
                        change.source_created_ns,
                        change.probe.exif_taken_ns,
                        change.source_size,
                        change.probe.kind.as_str(),
                        change.probe.format.as_str(),
                        change.probe.width,
                        change.probe.height,
                        change.probe.is_animated,
                        change.frame_count,
                        change.duration_ms,
                        metadata_version_for(change.probe.kind),
                    ),
                    |row| row.get(0),
                )
                .with_context(|| format!("storing asset catalog entry for {}", change.path))?;
            assets.push(Asset {
                asset_id,
                path: change.path,
                fingerprint: change.fingerprint,
                source_created_ns: change.source_created_ns,
                exif_taken_ns: change.probe.exif_taken_ns,
                media_kind: change.probe.kind,
                media_format: change.probe.format,
                width: change.probe.width,
                height: change.probe.height,
                is_animated: change.probe.is_animated,
                frame_count: change.probe.frame_count,
                duration_ms: change.probe.duration_ms,
                metadata_version: metadata_version_for(change.probe.kind),
            });
        }
        drop(statement);
        drop(metadata_statement);
        timings.row_upsert = started.elapsed();

        let started = Instant::now();
        tx.execute(
            "UPDATE catalog_meta SET revision = revision + 1 WHERE singleton = 1",
            [],
        )?;
        timings.revision_update = started.elapsed();

        let started = Instant::now();
        tx.commit()?;
        timings.commit = started.elapsed();
        timings.store = timings.transaction_begin
            + timings.row_upsert
            + timings.revision_update
            + timings.commit;

        Ok((assets, timings))
    }

    /// Read the small catalog fields needed to classify a directory walk.
    pub(crate) fn scan_entries_under_root(
        &self,
        root: &Path,
    ) -> Result<HashMap<PathBuf, CatalogScanEntry>> {
        if !root.is_absolute() {
            bail!("asset root must be absolute: {root}");
        }
        let scope = PathScope::root(root).bind("path");
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT path, source_modified_ns, source_size, source_created_ns, metadata_version, media_kind
             FROM assets WHERE {}",
            scope.sql
        ))?;
        let rows = statement.query_map(bind_named(&scope.params).as_slice(), |row| {
            Ok((
                PathBuf::from(row.get::<_, String>(0)?),
                CatalogScanEntry {
                    fingerprint: SourceFingerprint {
                        modified_ns: row.get(1)?,
                        size: row.get::<_, i64>(2)?.try_into().map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                2,
                                rusqlite::types::Type::Integer,
                                Box::new(error),
                            )
                        })?,
                    },
                    source_created_ns: row.get(3)?,
                    metadata_version: row.get(4)?,
                    media_kind: MediaKind::from_db(&row.get::<_, String>(5)?).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            5,
                            rusqlite::types::Type::Text,
                            error.into(),
                        )
                    })?,
                },
            ))
        })?;
        rows.collect::<rusqlite::Result<HashMap<_, _>>>()
            .context("listing catalog scan entries under root")
    }

    /// Return catalog rows under `root` that were not present in a completed filesystem scan.
    /// The scan set lives only on this connection and never rewrites persistent asset rows.
    pub fn unseen_under_root(
        &mut self,
        root: &Path,
        seen_asset_ids: impl IntoIterator<Item = i64>,
    ) -> Result<Vec<Asset>> {
        if !root.is_absolute() {
            bail!("asset root must be absolute: {root}");
        }
        let transaction = self.conn.transaction()?;
        transaction.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS scan_seen_assets(
                asset_id INTEGER PRIMARY KEY
            ) WITHOUT ROWID;
            DELETE FROM scan_seen_assets;",
        )?;
        {
            let mut insert = transaction
                .prepare_cached("INSERT OR IGNORE INTO scan_seen_assets(asset_id) VALUES (?1)")?;
            for asset_id in seen_asset_ids {
                insert.execute([asset_id])?;
            }
        }
        let unseen = {
            let scope = PathScope::root(root).bind("path");
            let mut statement = transaction.prepare(&format!(
                "SELECT {ASSET_COLUMNS} FROM assets
                 WHERE {}
                   AND NOT EXISTS (
                       SELECT 1 FROM scan_seen_assets
                       WHERE scan_seen_assets.asset_id = assets.asset_id
                   )
                 ORDER BY asset_id",
                scope.sql
            ))?;
            let rows =
                statement.query_and_then(bind_named(&scope.params).as_slice(), asset_from_row)?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        transaction.commit()?;
        Ok(unseen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::media::{METADATA_VERSION, VIDEO_METADATA_VERSION};
    use crate::assets::test_support::catalog_revision;
    use crate::assets::*;
    use std::fs::{self, File};
    use tempfile::TempDir;

    #[test]
    fn videos_are_cataloged_and_visible_even_when_metadata_probe_fails() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_path_buf())?;
        let catalog = AssetCatalog::new(&root.join("assets.db"))?;
        let mut videos = Vec::new();
        for extension in [
            "mp4", "MOV", "avi", "mkv", "m4v", "webm", "mpg", "mpeg", "m2ts",
        ] {
            let path = root.join(format!("clip.{extension}"));
            fs::write(&path, b"video placeholder")?;
            assert!(is_catalog_media(&path));
            let asset = catalog.upsert(&path, &fs::metadata(&path)?)?;
            assert_eq!(asset.media_kind, MediaKind::Video);
            assert_eq!(asset.media_format, extension.to_ascii_lowercase());
            assert_eq!(
                (asset.width, asset.height, asset.duration_ms),
                (None, None, None)
            );
            assert!(!is_ocr_image(&asset));
            videos.push(asset);
        }
        let mut disguised_video = videos[0].clone();
        disguised_video.media_format = "png".to_owned();
        assert!(!is_ocr_image(&disguised_video));
        for extension in ["png", "JPG", "jpeg", "gif", "webp", "bmp"] {
            assert!(is_catalog_media(&root.join(format!("image.{extension}"))));
        }
        let image = root.join("photo.png");
        fs::write(&image, b"malformed images remain catalogable")?;
        let asset = catalog.upsert(&image, &fs::metadata(&image)?)?;
        assert_eq!(catalog.get(videos[0].asset_id)?, Some(videos[0].clone()));
        assert_eq!(
            catalog.get_by_path(&videos[0].path)?,
            Some(videos[0].clone())
        );
        assert_eq!(
            catalog.get_many(&[videos[0].asset_id, asset.asset_id])?,
            vec![videos[0].clone(), asset.clone()]
        );
        videos.push(asset.clone());
        assert_eq!(catalog.all()?, videos);
        assert_eq!(catalog.under_root(&root)?, videos);
        assert_eq!(catalog.scan_entries_under_root(&root)?.len(), videos.len());
        assert_eq!(
            catalog.count_gallery(&PathScope::root(&root))?,
            videos.len() as i64
        );
        for timeline in [Timeline::Modified, Timeline::Capture] {
            assert_eq!(
                catalog
                    .list_gallery(&PathScope::root(&root), timeline)?
                    .len(),
                videos.len()
            );
            assert_eq!(
                catalog.timeline_range(timeline, None, None)?.len(),
                videos.len()
            );
            assert!(
                catalog
                    .timeline_range(timeline, Some(0), Some(20))?
                    .is_empty()
            );
        }
        Ok(())
    }

    #[test]
    fn stale_catalog_metadata_is_reprobed_without_a_source_change() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("oriented.jpg"))?;
        fs::write(
            &source,
            crate::imaging::test_support::jpeg_with_orientation(6)?,
        )?;

        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let first = catalog.upsert(&source, &fs::metadata(&source)?)?;
        catalog.conn.execute(
            "UPDATE assets SET width = 3, height = 2, metadata_version = 1 WHERE asset_id = ?1",
            [first.asset_id],
        )?;

        let refreshed = catalog.upsert(&source, &fs::metadata(&source)?)?;

        assert_eq!((refreshed.width, refreshed.height), (Some(2), Some(3)));
        assert_eq!(refreshed.metadata_version, METADATA_VERSION);
        Ok(())
    }

    #[test]
    fn old_video_metadata_is_reprobed_without_refreshing_images() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_path_buf())?;
        let video = root.join("keyframes.mp4");
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/video/keyframes.mp4"),
            &video,
        )?;
        let image = root.join("photo.jpg");
        fs::write(
            &image,
            crate::imaging::test_support::jpeg_with_orientation(1)?,
        )?;

        let catalog = AssetCatalog::new(&root.join("assets.db"))?;
        let video_asset = catalog.upsert(&video, &fs::metadata(&video)?)?;
        let image_asset = catalog.upsert(&image, &fs::metadata(&image)?)?;
        assert_eq!(
            (video_asset.width, video_asset.height),
            (Some(128), Some(96))
        );
        catalog.conn.execute(
            "UPDATE assets SET width = NULL, height = NULL, metadata_version = 2 WHERE asset_id = ?1",
            [video_asset.asset_id],
        )?;

        let entries = catalog.scan_entries_under_root(&root)?;
        assert!(!entries[&video].is_current(&fs::metadata(&video)?)?);
        assert!(entries[&image].is_current(&fs::metadata(&image)?)?);
        let refreshed = catalog.upsert(&video, &fs::metadata(&video)?)?;
        assert_eq!((refreshed.width, refreshed.height), (Some(128), Some(96)));
        assert_eq!(refreshed.metadata_version, VIDEO_METADATA_VERSION);
        assert_eq!(image_asset.metadata_version, METADATA_VERSION);
        Ok(())
    }

    #[test]
    fn catalog_records_creation_time_and_path_derivatives() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("Summer.photo.PNG"))?;
        File::create(&source)?;
        let metadata = fs::metadata(&source)?;
        let expected_created_ns = timestamp_ns(metadata.created().ok());
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;

        let asset = catalog.upsert(&source, &metadata)?;

        assert_eq!(asset.source_created_ns, expected_created_ns);
        assert_eq!(asset.display_name(), "Summer.photo.PNG");
        assert_eq!(asset.extension(), Some("PNG"));
        assert_eq!(catalog.get(asset.asset_id)?, Some(asset.clone()));

        // Creation time is independent from the content fingerprint. A stale value is refreshed
        // with a narrow catalog update, without doing another media probe.
        catalog.conn.execute(
            "UPDATE assets SET source_created_ns = -1 WHERE asset_id = ?1",
            [asset.asset_id],
        )?;
        let (prepared, _) = catalog.prepare_upsert_timed(&source, &metadata)?;
        assert!(matches!(prepared, PreparedCatalogAsset::MetadataChanged(_)));
        let (refreshed, _) = catalog.store_prepared_batch(vec![prepared])?;
        assert_eq!(refreshed[0].source_created_ns, expected_created_ns);
        Ok(())
    }

    #[test]
    fn preserves_id_and_keeps_malformed_media() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("broken.gif"))?;
        File::create(&source)?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;

        let first = catalog.upsert(&source, &fs::metadata(&source)?)?;
        let second = catalog.upsert(&source, &fs::metadata(&source)?)?;

        assert_eq!(first.asset_id, second.asset_id);
        assert_eq!(first.media_format, "gif");
        assert_eq!(first.width, None);
        assert!(!first.is_animated);
        assert_eq!(catalog.get(first.asset_id)?, Some(first));
        Ok(())
    }

    #[test]
    fn catalog_revision_changes_only_when_a_source_changes() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("broken.png"))?;
        File::create(&source)?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;

        assert_eq!(catalog_revision(&catalog)?, 0);
        catalog.upsert(&source, &fs::metadata(&source)?)?;
        assert_eq!(catalog_revision(&catalog)?, 1);
        catalog.upsert(&source, &fs::metadata(&source)?)?;
        assert_eq!(catalog_revision(&catalog)?, 1);
        Ok(())
    }

    #[test]
    fn prepared_batch_commits_once_and_preserves_asset_order() -> Result<()> {
        let temp = TempDir::new()?;
        let first = PathBuf::try_from(temp.path().join("first.png"))?;
        let second = PathBuf::try_from(temp.path().join("second.png"))?;
        File::create(&first)?;
        File::create(&second)?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;

        let prepared = [&first, &second]
            .into_iter()
            .map(|path| {
                catalog
                    .prepare_upsert_timed(path, &fs::metadata(path)?)
                    .map(|(asset, _timings)| asset)
            })
            .collect::<Result<Vec<_>>>()?;
        let (assets, timings) = catalog.store_prepared_batch(prepared)?;

        assert_eq!(
            assets
                .iter()
                .map(|asset| asset.path.as_path())
                .collect::<Vec<_>>(),
            vec![first.as_path(), second.as_path()]
        );
        assert_eq!(catalog_revision(&catalog)?, 1);
        assert!(timings.commit > Duration::ZERO);

        let prepared = [&first, &second]
            .into_iter()
            .map(|path| {
                catalog
                    .prepare_upsert_timed(path, &fs::metadata(path)?)
                    .map(|(asset, _timings)| asset)
            })
            .collect::<Result<Vec<_>>>()?;
        let (_assets, timings) = catalog.store_prepared_batch(prepared)?;
        assert_eq!(catalog_revision(&catalog)?, 1);
        assert_eq!(timings.store, Duration::ZERO);

        assert_eq!(
            catalog.delete_assets(
                &assets
                    .iter()
                    .map(|asset| asset.asset_id)
                    .collect::<Vec<_>>()
            )?,
            2
        );
        assert_eq!(catalog_revision(&catalog)?, 2);
        Ok(())
    }

    #[test]
    fn temporary_seen_set_returns_only_unseen_assets_inside_root() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().join("gallery"))?;
        let sibling = PathBuf::try_from(temp.path().join("gallery-other"))?;
        fs::create_dir_all(&root)?;
        fs::create_dir_all(&sibling)?;
        let seen_path = root.join("seen.bmp");
        let unseen_path = root.join("unseen.bmp");
        let sibling_path = sibling.join("sibling.bmp");
        for path in [&seen_path, &unseen_path, &sibling_path] {
            File::create(path)?;
        }
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let mut catalog = AssetCatalog::new(&catalog_path)?;
        let seen = catalog.upsert(&seen_path, &fs::metadata(&seen_path)?)?;
        let unseen = catalog.upsert(&unseen_path, &fs::metadata(&unseen_path)?)?;
        catalog.upsert(&sibling_path, &fs::metadata(&sibling_path)?)?;

        assert_eq!(
            catalog.unseen_under_root(&root, [seen.asset_id])?,
            vec![unseen.clone()]
        );
        assert!(
            catalog
                .unseen_under_root(&root, [seen.asset_id, unseen.asset_id])?
                .is_empty()
        );
        Ok(())
    }
}
