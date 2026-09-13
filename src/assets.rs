use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use nom_exif::{ExifTag, read_exif};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params_from_iter};

use crate::imaging::{ExifOrientation, orientation_from_exif};
use crate::schema::{check_schema_read_only, open_schema};

const SCHEMA_VERSION: i32 = 5;
const SCHEMA_LABEL: &str = "asset catalog";
// Creation time is filesystem metadata, not media-probe output. Keeping this version unchanged
// lets version-3 catalogs backfill it with a narrow update instead of re-reading every image.
const METADATA_VERSION: i32 = 2;
const GALLERY_ROOT_PREDICATE: &str = r"(path = ?1 OR (substr(path, 1, length(?1)) = ?1 AND (substr(?1, -1) IN ('/', '\') OR substr(path, length(?1) + 1, 1) IN ('/', '\'))))";

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

/// Resolve a filesystem path while retaining the normal Windows path spelling used by clients.
pub fn canonicalize_path(path: &Path) -> Result<PathBuf> {
    let path = PathBuf::try_from(
        fs::canonicalize(path).with_context(|| format!("canonicalizing path: {path}"))?,
    )
    .with_context(|| format!("canonical path is not valid UTF-8: {path}"))?;
    Ok(normalize_windows_verbatim_path(path))
}

// EVAL: is this the right way to do this? is there a crate or another api to do this in a canonical way?
#[cfg(windows)]
fn normalize_windows_verbatim_path(path: PathBuf) -> PathBuf {
    let path = path.as_str();
    if let Some(remainder) = path
        .strip_prefix("//?/UNC/")
        .or_else(|| path.strip_prefix(r"\\?\UNC\"))
    {
        return PathBuf::from(format!("//{remainder}"));
    }
    if let Some(remainder) = path
        .strip_prefix("//?/")
        .or_else(|| path.strip_prefix(r"\\?\"))
    {
        return PathBuf::from(remainder);
    }
    PathBuf::from(path)
}

#[cfg(not(windows))]
fn normalize_windows_verbatim_path(path: PathBuf) -> PathBuf {
    path
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    // Retained for legacy catalog compatibility only; videos are no longer admitted or listed.
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timeline {
    Modified,
    Capture,
}

impl Timeline {
    /// The SQL instant this timeline orders and filters on, qualified by `table`.
    ///
    /// Both the asset catalog and the OCR store spell these columns the same way, so one
    /// definition serves both and the two can never disagree about what "capture" means.
    pub fn expression(self, table: &str) -> String {
        match self {
            Self::Modified => format!("{table}.source_modified_ns"),
            Self::Capture => format!("COALESCE({table}.exif_taken_ns, {table}.source_modified_ns)"),
        }
    }
}

impl MediaKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }

    fn from_db(value: &str) -> Result<Self> {
        match value {
            "image" => Ok(Self::Image),
            "video" => Ok(Self::Video),
            _ => bail!("invalid media kind in asset catalog: {value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub asset_id: i64, // EVAL: signed? does it matter with sqlite anyway?
    pub path: PathBuf,
    pub fingerprint: SourceFingerprint,
    pub source_created_ns: Option<i64>,
    pub exif_taken_ns: Option<i64>,
    pub media_kind: MediaKind,
    pub media_format: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub is_animated: bool,
    pub frame_count: Option<u32>,
    pub duration_ms: Option<u64>,
    pub(crate) metadata_version: i32,
}

impl Asset {
    /// The final path component suitable for compact gallery labels.
    pub fn display_name(&self) -> &str {
        self.path.file_name().unwrap_or(self.path.as_str())
    }

    /// The filename extension as written on disk, without its leading dot.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension()
    }
}

#[derive(Debug)]
struct MediaProbe {
    kind: MediaKind,
    format: String,
    exif_taken_ns: Option<i64>,
    width: Option<u32>,
    height: Option<u32>,
    is_animated: bool,
    frame_count: Option<u32>,
    duration_ms: Option<u64>,
}

pub struct AssetCatalog {
    conn: Connection,
}

#[derive(Debug, Default)]
pub(crate) struct CatalogUpsertTimings {
    pub canonicalize: Duration,
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
    pub fn new(path: &Path) -> Result<Self> {
        let conn =
            Connection::open(path).with_context(|| format!("opening asset catalog: {path}"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "auto_vacuum", "FULL")?;
        conn.pragma_update(None, "journal_mode", "wal")?;
        conn.pragma_update(None, "synchronous", "normal")?;
        migrate_schema(&conn)?;
        open_schema(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("assets_create.sql"),
        )?;
        Ok(Self { conn })
    }

    /// Open a query-only connection suitable for concurrent lookups while an index job writes.
    /// Unlike [`AssetCatalog::new`] this never creates the schema, so a caller cannot silently
    /// read an empty catalog it just brought into existence.
    pub fn new_read_only(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening asset catalog read-only: {path}"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        check_schema_read_only(&conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
        Ok(Self { conn })
    }

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
        if !path.is_absolute() {
            bail!("asset path must be absolute: {path}");
        }
        if !is_catalog_media(path) {
            bail!("unsupported image format: {path}");
        }
        let mut timings = CatalogUpsertTimings::default();
        let started = Instant::now();
        let path = canonicalize_path(path)?;
        timings.canonicalize = started.elapsed();

        let started = Instant::now();
        let fingerprint = SourceFingerprint::from_metadata(metadata)?;
        timings.fingerprint = started.elapsed();
        let source_created_ns = timestamp_ns(metadata.created().ok());

        let started = Instant::now();
        if let Some(mut existing) = self.get_by_path(&path)?
            && existing.fingerprint == fingerprint
            && existing.metadata_version == METADATA_VERSION
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
        let (probe, probe_timings) = probe_media(&path);
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
            "INSERT INTO assets (path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13) \
             ON CONFLICT(path) DO UPDATE SET source_modified_ns=excluded.source_modified_ns, source_created_ns=excluded.source_created_ns, exif_taken_ns=excluded.exif_taken_ns, source_size=excluded.source_size, media_kind=excluded.media_kind, media_format=excluded.media_format, width=excluded.width, height=excluded.height, is_animated=excluded.is_animated, frame_count=excluded.frame_count, duration_ms=excluded.duration_ms, metadata_version=excluded.metadata_version \
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
                        METADATA_VERSION,
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
                metadata_version: METADATA_VERSION,
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

    pub fn get(&self, asset_id: i64) -> Result<Option<Asset>> {
        self.conn
            .query_row(
                "SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version FROM assets WHERE media_kind = 'image' AND asset_id = ?1",
                [asset_id],
                asset_from_row,
            )
            .optional()
            .context("loading asset catalog entry")
    }

    pub fn get_by_path(&self, path: &Path) -> Result<Option<Asset>> {
        self.conn
            .query_row(
                "SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version FROM assets WHERE media_kind = 'image' AND path = ?1",
                [path.as_str()],
                asset_from_row,
            )
            .optional()
            .with_context(|| format!("loading asset catalog entry for {path}"))
    }

    /// Look up a bounded batch of asset IDs without touching the filesystem.
    ///
    /// Results follow the request order and omit IDs that no longer exist, so callers can retain
    /// viewport order while handling a concurrently pruned catalog.
    pub fn get_many(&self, asset_ids: &[i64]) -> Result<Vec<Asset>> {
        const QUERY_CHUNK_SIZE: usize = 512;

        validate_asset_ids(asset_ids)?;
        if asset_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut found = HashMap::with_capacity(asset_ids.len());
        for asset_ids in asset_ids.chunks(QUERY_CHUNK_SIZE) {
            let placeholders = std::iter::repeat_n("?", asset_ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version FROM assets WHERE media_kind = 'image' AND asset_id IN ({placeholders})"
            );
            let mut statement = self.conn.prepare(&sql)?;
            let rows = statement.query_and_then(params_from_iter(asset_ids), asset_from_row)?;
            for asset in rows {
                let asset = asset?;
                found.insert(asset.asset_id, asset);
            }
        }

        Ok(asset_ids
            .iter()
            .filter_map(|asset_id| found.remove(asset_id))
            .collect())
    }

    pub fn all(&self) -> Result<Vec<Asset>> {
        let mut statement = self.conn.prepare_cached(
            "SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version FROM assets WHERE media_kind = 'image' ORDER BY asset_id",
        )?;
        let rows = statement.query_and_then([], asset_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("listing asset catalog")
    }

    pub fn timeline_range(
        &self,
        timeline: Timeline,
        from_ns: Option<i64>,
        to_ns: Option<i64>,
    ) -> Result<Vec<Asset>> {
        if matches!((from_ns, to_ns), (Some(from), Some(to)) if from >= to) {
            bail!("timeline range start must be less than its end");
        }
        let expression = timeline.expression("assets");
        let (predicate, parameters): (&str, Vec<i64>) = match (from_ns, to_ns) {
            (None, None) => ("", Vec::new()),
            (Some(from), None) => ("AND {time} >= ?1", vec![from]),
            (None, Some(to)) => ("AND {time} < ?1", vec![to]),
            (Some(from), Some(to)) => ("AND {time} >= ?1 AND {time} < ?2", vec![from, to]),
        };
        let predicate = predicate.replace("{time}", &expression);
        let sql = format!(
            "SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version FROM assets WHERE media_kind = 'image' {predicate} ORDER BY {expression}, asset_id"
        );
        let mut statement = self.conn.prepare(&sql)?;
        let rows = statement.query_and_then(params_from_iter(parameters), asset_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("listing asset timeline range")
    }

    pub fn under_root(&self, root: &Path) -> Result<Vec<Asset>> {
        if !root.is_absolute() {
            bail!("asset root must be absolute: {root}");
        }
        let mut statement = self.conn.prepare_cached(
            "SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version FROM assets WHERE media_kind = 'image' AND path LIKE ?1 ESCAPE '#' ORDER BY asset_id",
        )?;
        let rows = statement.query_and_then([path_prefix_like(root)], asset_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("listing assets under root")
    }

    /// Desktop reads use literal path boundaries, including roots containing SQL wildcards.
    pub fn list_gallery(&self, root: &Path, timeline: Timeline) -> Result<Vec<Asset>> {
        let order = timeline.expression("assets");
        let mut statement = self.conn.prepare(&format!(
            "SELECT asset_id, path, source_modified_ns, source_created_ns, exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated, frame_count, duration_ms, metadata_version FROM assets WHERE media_kind = 'image' AND {GALLERY_ROOT_PREDICATE} ORDER BY {order} DESC, asset_id DESC"
        ))?;
        Ok(statement
            .query_and_then(
                [root.as_str().trim_end_matches(['/', '\\'])],
                asset_from_row,
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn count_gallery(&self, root: &Path) -> Result<i64> {
        Ok(self.conn.query_row(
            &format!("SELECT COUNT(*) FROM assets WHERE media_kind = 'image' AND {GALLERY_ROOT_PREDICATE}"),
            [root.as_str().trim_end_matches(['/', '\\'])],
            |row| row.get(0),
        )?)
    }

    pub fn revision(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT revision FROM catalog_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?)
    }

    /// Asset IDs whose current source fingerprint previously failed during image decoding.
    /// A changed source is deliberately absent, so fixing or replacing a broken file retries it.
    pub fn current_decode_failure_asset_ids(
        &self,
        fingerprints: &[(i64, SourceFingerprint)],
    ) -> Result<HashSet<i64>> {
        let mut failed = HashSet::with_capacity(fingerprints.len());
        let mut statement = self.conn.prepare_cached(
            "SELECT EXISTS(\
                 SELECT 1 FROM decode_failure_state \
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
                failed.insert(*asset_id);
            }
        }
        Ok(failed)
    }

    /// Remember that this exact revision could not be decoded. This state is shared by OCR and
    /// image embedding so a malformed image produces one useful error instead of one per job.
    pub fn record_decode_failure(&self, asset: &Asset) -> Result<()> {
        let source_size = i64::try_from(asset.fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        self.conn.execute(
            "INSERT INTO decode_failure_state(asset_id, source_modified_ns, source_size) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(asset_id) DO UPDATE SET \
                 source_modified_ns = excluded.source_modified_ns, \
                 source_size = excluded.source_size",
            (asset.asset_id, asset.fingerprint.modified_ns, source_size),
        )?;
        Ok(())
    }

    pub fn delete_asset(&self, asset_id: i64) -> Result<bool> {
        Ok(self.delete_assets(&[asset_id])? > 0)
    }

    pub fn delete_assets(&self, asset_ids: &[i64]) -> Result<usize> {
        validate_asset_ids(asset_ids)?;
        if asset_ids.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.unchecked_transaction()?;
        let deleted = {
            let mut failures =
                tx.prepare("DELETE FROM decode_failure_state WHERE asset_id = ?1")?;
            let mut statement = tx.prepare("DELETE FROM assets WHERE asset_id = ?1")?;
            asset_ids.iter().try_fold(0usize, |deleted, asset_id| {
                failures.execute([asset_id])?;
                Ok::<_, rusqlite::Error>(deleted + statement.execute([asset_id])?)
            })?
        };
        if deleted > 0 {
            tx.execute(
                "UPDATE catalog_meta SET revision = revision + 1 WHERE singleton = 1",
                [],
            )?;
        }
        tx.commit()?;
        Ok(deleted)
    }
}

fn validate_asset_ids(asset_ids: &[i64]) -> Result<()> {
    if asset_ids.iter().any(|asset_id| *asset_id <= 0) {
        bail!("asset identifiers must be greater than zero");
    }
    Ok(())
}

fn path_prefix_like(path: &Path) -> String {
    let mut prefix = path.as_str().to_owned();
    if !prefix.ends_with(['/', '\\']) {
        prefix.push(std::path::MAIN_SEPARATOR);
    }
    format!(
        "{}%",
        prefix
            .replace('#', "##")
            .replace('%', "#%")
            .replace('_', "#_")
    )
}

fn asset_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Asset> {
    let kind: String = row.get(6)?;
    let path: String = row.get(1)?;
    let source_size: i64 = row.get(5)?;
    let duration_ms: Option<i64> = row.get(12)?;
    Ok(Asset {
        asset_id: row.get(0)?,
        path: PathBuf::from(path),
        fingerprint: SourceFingerprint {
            modified_ns: row.get(2)?,
            size: u64::try_from(source_size).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    5,
                    rusqlite::types::Type::Integer,
                    Box::new(error),
                )
            })?,
        },
        source_created_ns: row.get(3)?,
        exif_taken_ns: row.get(4)?,
        media_kind: MediaKind::from_db(&kind).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, error.into())
        })?,
        media_format: row.get(7)?,
        width: row.get(8)?,
        height: row.get(9)?,
        is_animated: row.get(10)?,
        frame_count: row.get(11)?,
        duration_ms: duration_ms
            .map(u64::try_from)
            .transpose()
            .map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    12,
                    rusqlite::types::Type::Integer,
                    Box::new(error),
                )
            })?,
        metadata_version: row.get(13)?,
    })
}

fn timestamp_ns(time: Option<std::time::SystemTime>) -> Option<i64> {
    i64::try_from(time?.duration_since(UNIX_EPOCH).ok()?.as_nanos()).ok()
}

#[derive(Debug, Default)]
struct MediaProbeTimings {
    dimensions: Duration,
    exif: Duration,
    animation: Duration,
}

fn probe_media(path: &Path) -> (MediaProbe, MediaProbeTimings) {
    let mut timings = MediaProbeTimings::default();
    let extension = path.extension().unwrap_or_default().to_ascii_lowercase();

    let started = Instant::now();
    let (sniffed_format, dimensions) = probe_image_header(path);
    timings.dimensions = started.elapsed();

    // The extension is only a fallback: a still image's real format comes from its header, so a
    // misnamed file (a JPEG saved with a `.png` extension, say) is catalogued as what it actually
    // is rather than what its name claims.
    let format = sniffed_format.unwrap_or_else(|| match extension.as_str() {
        "jpg" => "jpeg".to_owned(),
        "" => "unknown".to_owned(),
        value => value.to_owned(),
    });

    let started = Instant::now();
    let gif_metadata = if format == "gif" {
        probe_gif(path).ok()
    } else {
        None
    };
    timings.animation = started.elapsed();

    let started = Instant::now();
    let (exif_taken_ns, orientation) =
        probe_exif_metadata(path).unwrap_or((None, ExifOrientation::Identity));
    timings.exif = started.elapsed();

    (
        MediaProbe {
            kind: MediaKind::Image,
            format,
            exif_taken_ns,
            width: dimensions.map(|(width, height)| {
                if orientation.swaps_dimensions() {
                    height
                } else {
                    width
                }
            }),
            height: dimensions.map(|(width, height)| {
                if orientation.swaps_dimensions() {
                    width
                } else {
                    height
                }
            }),
            is_animated: gif_metadata.is_some_and(|value| value.0 > 1),
            frame_count: gif_metadata.map(|value| value.0),
            duration_ms: gif_metadata.map(|value| value.1),
        },
        timings,
    )
}

/// Detect the real still-image format from its magic bytes, along with its pixel dimensions read
/// from the same header. `None` for either half if the file can't be opened, or its header isn't
/// a still-image format nicegal-server decodes; callers fall back to the file extension.
fn probe_image_header(path: &Path) -> (Option<String>, Option<(u32, u32)>) {
    let Ok(file) = File::open(path) else {
        return (None, None);
    };
    let mut reader = BufReader::new(file);
    let Ok(kind) = imagesize::reader_type(&mut reader) else {
        return (None, None);
    };
    let dimensions = kind.reader_size(&mut reader).ok().and_then(|size| {
        Some((
            u32::try_from(size.width).ok()?,
            u32::try_from(size.height).ok()?,
        ))
    });
    (image_type_name(kind), dimensions)
}

fn image_type_name(kind: imagesize::ImageType) -> Option<String> {
    Some(
        match kind {
            imagesize::ImageType::Jpeg => "jpeg",
            imagesize::ImageType::Png => "png",
            imagesize::ImageType::Gif => "gif",
            imagesize::ImageType::Webp => "webp",
            imagesize::ImageType::Bmp => "bmp",
            _ => return None,
        }
        .to_owned(),
    )
}

fn probe_exif_metadata(path: &Path) -> Result<(Option<i64>, ExifOrientation)> {
    let exif =
        read_exif(path.as_std_path()).with_context(|| format!("reading EXIF data: {path}"))?;
    let orientation = orientation_from_exif(&exif);
    let Some(datetime) = exif
        .get(ExifTag::DateTimeOriginal)
        .and_then(|value| value.as_datetime())
        .or_else(|| {
            exif.get(ExifTag::ModifyDate)
                .and_then(|value| value.as_datetime())
        })
    else {
        return Ok((None, orientation));
    };
    let timestamp = match datetime.aware() {
        Some(datetime) => datetime.timestamp_nanos_opt(),
        None => datetime.into_naive().and_utc().timestamp_nanos_opt(),
    }
    .context("EXIF capture time exceeds the nanosecond timestamp range")?;
    Ok((Some(timestamp), orientation))
}

fn migrate_schema(conn: &Connection) -> Result<()> {
    let version: i32 = conn
        .query_row("SELECT user_version FROM pragma_user_version", [], |row| {
            row.get(0)
        })
        .context("reading asset catalog schema version")?;
    match version {
        2 => conn
            .execute_batch(
                "BEGIN;
                 ALTER TABLE assets ADD COLUMN metadata_version INTEGER NOT NULL DEFAULT 1 CHECK(metadata_version > 0);
                 ALTER TABLE assets ADD COLUMN source_created_ns INTEGER;
                 CREATE TABLE decode_failure_state(asset_id INTEGER PRIMARY KEY, source_modified_ns INTEGER NOT NULL, source_size INTEGER NOT NULL CHECK(source_size >= 0));
                 PRAGMA user_version = 5;
                 COMMIT;",
            )
            .context("migrating asset catalog schema from version 2 to 5")?,
        3 => conn
            .execute_batch(
                "BEGIN;
                 ALTER TABLE assets ADD COLUMN source_created_ns INTEGER;
                 CREATE TABLE decode_failure_state(asset_id INTEGER PRIMARY KEY, source_modified_ns INTEGER NOT NULL, source_size INTEGER NOT NULL CHECK(source_size >= 0));
                 PRAGMA user_version = 5;
                 COMMIT;",
            )
            .context("migrating asset catalog schema from version 3 to 5")?,
        4 => conn
            .execute_batch(
                "BEGIN;
                 CREATE TABLE decode_failure_state(
                     asset_id INTEGER PRIMARY KEY,
                     source_modified_ns INTEGER NOT NULL,
                     source_size INTEGER NOT NULL CHECK(source_size >= 0)
                 );
                 PRAGMA user_version = 5;
                 COMMIT;",
            )
            .context("migrating asset catalog schema from version 4 to 5")?,
        _ => {}
    }
    Ok(())
}

fn probe_gif(path: &Path) -> Result<(u32, u64)> {
    let file = File::open(path).with_context(|| format!("opening GIF: {path}"))?;
    let mut options = gif::DecodeOptions::new();
    // Cataloging needs frame control data, not pixels. Skipping LZW decompression avoids allocating
    // a frame-sized buffer and decoding every image block merely to count frames and add delays.
    options.skip_frame_decoding(true);
    let mut decoder = options
        .read_info(file)
        .with_context(|| format!("decoding GIF header: {path}"))?;
    let mut frame_count = 0_u32;
    let mut duration_ms = 0_u64;
    while let Some(frame) = decoder
        .next_frame_info()
        .with_context(|| format!("reading GIF frame metadata: {path}"))?
    {
        frame_count = frame_count
            .checked_add(1)
            .context("GIF frame count overflow")?;
        duration_ms = duration_ms.saturating_add(u64::from(frame.delay) * 10);
    }
    if frame_count == 0 {
        bail!("GIF contains no decodable frames: {path}");
    }
    Ok((frame_count, duration_ms))
}

pub fn is_catalog_media(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        matches!(
            extension.to_ascii_lowercase().as_str(),
            "png" | "jpeg" | "jpg" | "gif" | "webp" | "bmp"
        )
    })
}

pub fn is_ocr_image(asset: &Asset) -> bool {
    asset.media_kind == MediaKind::Image
        && matches!(asset.media_format.as_str(), "png" | "jpeg" | "gif" | "webp")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn videos_are_not_admitted_and_legacy_rows_are_hidden_without_deletion() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_path_buf())?;
        let catalog = AssetCatalog::new(&root.join("assets.db"))?;
        for extension in ["mp4", "MOV", "avi", "mkv", "m4v", "webm"] {
            let path = root.join(format!("clip.{extension}"));
            fs::write(&path, b"video placeholder")?;
            assert!(!is_catalog_media(&path));
            assert!(catalog.upsert(&path, &fs::metadata(&path)?).is_err());
        }
        for extension in ["png", "JPG", "jpeg", "gif", "webp", "bmp"] {
            assert!(is_catalog_media(&root.join(format!("image.{extension}"))));
        }
        let video = root.join("legacy.mp4");
        catalog.conn.execute("INSERT INTO assets(asset_id, path, source_modified_ns, source_size, media_kind, media_format, is_animated) VALUES (1, ?1, 10, 0, 'video', 'mp4', 0)", [video.as_str()])?;
        let image = root.join("photo.png");
        fs::write(&image, b"malformed images remain catalogable")?;
        let asset = catalog.upsert(&image, &fs::metadata(&image)?)?;
        assert!(catalog.get(1)?.is_none());
        assert!(catalog.get_by_path(&video)?.is_none());
        assert_eq!(catalog.get_many(&[1, asset.asset_id])?, vec![asset.clone()]);
        assert_eq!(catalog.all()?, vec![asset.clone()]);
        assert_eq!(catalog.under_root(&root)?, vec![asset.clone()]);
        assert_eq!(catalog.count_gallery(&root)?, 1);
        for timeline in [Timeline::Modified, Timeline::Capture] {
            assert_eq!(catalog.list_gallery(&root, timeline)?, vec![asset.clone()]);
            assert_eq!(
                catalog.timeline_range(timeline, None, None)?,
                vec![asset.clone()]
            );
            assert!(
                catalog
                    .timeline_range(timeline, Some(0), Some(20))?
                    .is_empty()
            );
        }
        let rows: i64 = catalog
            .conn
            .query_row("SELECT COUNT(*) FROM assets", [], |row| row.get(0))?;
        assert_eq!(
            rows, 2,
            "legacy metadata is retained, not destructively migrated"
        );
        Ok(())
    }

    #[test]
    fn gallery_reads_preserve_literal_root_boundaries_and_timeline_order() -> Result<()> {
        let temp = TempDir::new()?;
        let base = PathBuf::try_from(temp.path().to_path_buf())?;
        let catalog = AssetCatalog::new(&base.join("assets.db"))?;
        let root = base.join("photos_%");
        // These paths intentionally do not exist: catalog browsing must work offline.
        for (id, path, modified, capture) in [
            (1, root.join("one.png"), 10, Some(30)),
            (2, root.join("nested/two.png"), 20, None),
            (3, root.join("three.png"), 20, Some(30)),
            (4, base.join("photos_%sibling/four.png"), 99, None),
            (5, base.join("photos_ab/five.png"), 99, None),
        ] {
            catalog.conn.execute("INSERT INTO assets(asset_id, path, source_modified_ns, exif_taken_ns, source_size, media_kind, media_format, is_animated) VALUES (?1, ?2, ?3, ?4, 0, 'image', 'png', 0)", (id, path.as_str(), modified, capture))?;
        }
        let ids = |assets: Vec<Asset>| {
            assets
                .into_iter()
                .map(|asset| asset.asset_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(catalog.count_gallery(&root)?, 3);
        assert_eq!(
            ids(catalog.list_gallery(&root, Timeline::Modified)?),
            [3, 2, 1]
        );
        assert_eq!(
            ids(catalog.list_gallery(&root, Timeline::Capture)?),
            [3, 1, 2]
        );
        let trailing = PathBuf::from(format!("{root}/"));
        assert_eq!(catalog.count_gallery(&trailing)?, 3);
        assert_eq!(catalog.count_gallery(&base.join("empty"))?, 0);
        assert_eq!(catalog.revision()?, 0);
        catalog.delete_assets(&[1])?;
        assert_eq!(catalog.revision()?, 1);
        assert_eq!(catalog.count_gallery(&root)?, 2);
        Ok(())
    }

    #[test]
    fn gif_probe_counts_frames_and_delays_without_decoding_pixels() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("animated.gif"))?;
        {
            let mut file = File::create(&source)?;
            let mut encoder = gif::Encoder::new(&mut file, 1, 1, &[0, 0, 0])?;
            for delay in [3, 7, 11] {
                encoder.write_frame(&gif::Frame {
                    delay,
                    width: 1,
                    height: 1,
                    buffer: Cow::Borrowed(&[0]),
                    ..gif::Frame::default()
                })?;
            }
        }

        assert_eq!(probe_gif(&source)?, (3, 210));

        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;
        assert!(asset.is_animated);
        assert_eq!(asset.frame_count, Some(3));
        assert_eq!(asset.duration_ms, Some(210));
        Ok(())
    }

    #[test]
    fn media_format_comes_from_the_header_not_the_extension() -> Result<()> {
        let temp = TempDir::new()?;
        // A real JPEG wearing a `.png` extension, the exact mislabeling this catalog guards
        // against: the format it records must match the bytes, not the filename.
        let source = PathBuf::try_from(temp.path().join("mislabeled.png"))?;
        let jpeg = crate::imaging::encode_jpeg(4, 2, &[0u8; 4 * 2 * 3], 85.0)?;
        fs::write(&source, jpeg)?;

        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;

        assert_eq!(asset.media_format, "jpeg");
        assert_eq!((asset.width, asset.height), (Some(4), Some(2)));
        Ok(())
    }

    #[test]
    fn jpeg_orientation_sets_visual_catalog_dimensions() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("oriented.jpg"))?;
        fs::write(
            &source,
            crate::imaging::test_support::jpeg_with_orientation(6)?,
        )?;

        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;

        assert_eq!((asset.width, asset.height), (Some(2), Some(3)));
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
    fn migrates_existing_catalogs_to_decode_failure_version_five() -> Result<()> {
        let temp = TempDir::new()?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let conn = Connection::open(&catalog_path)?;
        conn.execute_batch(
            "CREATE TABLE assets(asset_id INTEGER PRIMARY KEY);
             PRAGMA user_version = 2;",
        )?;
        drop(conn);

        let catalog = AssetCatalog::new(&catalog_path)?;
        let version: i32 =
            catalog
                .conn
                .query_row("SELECT user_version FROM pragma_user_version", [], |row| {
                    row.get(0)
                })?;
        let has_metadata_version: bool = catalog.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('assets') WHERE name = 'metadata_version')",
            [],
            |row| row.get(0),
        )?;
        let has_source_created_ns: bool = catalog.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('assets') WHERE name = 'source_created_ns')",
            [],
            |row| row.get(0),
        )?;

        assert_eq!(version, SCHEMA_VERSION);
        assert!(has_metadata_version);
        assert!(has_source_created_ns);
        Ok(())
    }

    #[test]
    fn decode_failures_only_apply_to_the_recorded_source_revision() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("broken.png"))?;
        File::create(&source)?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;

        catalog.record_decode_failure(&asset)?;
        let failures =
            catalog.current_decode_failure_asset_ids(&[(asset.asset_id, asset.fingerprint)])?;
        assert!(failures.contains(&asset.asset_id));

        let changed = SourceFingerprint {
            modified_ns: asset.fingerprint.modified_ns + 1,
            ..asset.fingerprint
        };
        let failures = catalog.current_decode_failure_asset_ids(&[(asset.asset_id, changed)])?;
        assert!(!failures.contains(&asset.asset_id));
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
    fn canonical_path_aliases_share_an_asset_id() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("broken.png"))?;
        File::create(&source)?;
        let alias_parent = PathBuf::try_from(temp.path().join("alias"))?;
        std::fs::create_dir(&alias_parent)?;
        let source_alias = alias_parent.join("..").join("broken.png");
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;

        let direct = catalog.upsert(&source, &fs::metadata(&source)?)?;
        let through_alias = catalog.upsert(&source_alias, &fs::metadata(&source_alias)?)?;

        assert_eq!(direct.asset_id, through_alias.asset_id);
        assert_eq!(direct.path, through_alias.path);
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
    fn batch_lookup_preserves_requested_order_and_omits_missing_assets() -> Result<()> {
        let temp = TempDir::new()?;
        let first_path = PathBuf::try_from(temp.path().join("first.png"))?;
        let second_path = PathBuf::try_from(temp.path().join("second.png"))?;
        File::create(&first_path)?;
        File::create(&second_path)?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let first = catalog.upsert(&first_path, &fs::metadata(&first_path)?)?;
        let second = catalog.upsert(&second_path, &fs::metadata(&second_path)?)?;

        let assets = catalog.get_many(&[second.asset_id, 999, first.asset_id])?;

        assert_eq!(
            assets
                .iter()
                .map(|asset| asset.asset_id)
                .collect::<Vec<_>>(),
            vec![second.asset_id, first.asset_id]
        );
        assert!(catalog.get_many(&[-1]).is_err());
        Ok(())
    }

    #[test]
    fn both_gallery_timeline_orders_use_covering_indexes() -> Result<()> {
        let temp = TempDir::new()?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;

        assert!(
            query_plan(&catalog, "source_modified_ns, asset_id")?.contains("assets_modified_idx")
        );
        assert!(
            query_plan(
                &catalog,
                "COALESCE(exif_taken_ns, source_modified_ns), asset_id"
            )?
            .contains("assets_taken_idx")
        );
        Ok(())
    }

    #[test]
    fn timeline_ranges_are_half_open_and_use_the_selected_timestamp() -> Result<()> {
        let temp = TempDir::new()?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        for (id, modified, taken) in [
            (1_i64, 100_i64, Some(400_i64)),
            (2, 200, None),
            (3, 300, Some(100)),
        ] {
            catalog.conn.execute(
                "INSERT INTO assets(asset_id, path, source_modified_ns, exif_taken_ns, source_size, media_kind, media_format, is_animated) VALUES (?1, ?2, ?3, ?4, 0, 'image', 'png', 0)",
                (id, format!("C:/gallery/{id}.png"), modified, taken),
            )?;
        }

        let modified = catalog.timeline_range(Timeline::Modified, Some(200), Some(300))?;
        assert_eq!(
            modified
                .iter()
                .map(|asset| asset.asset_id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        let capture = catalog.timeline_range(Timeline::Capture, Some(150), Some(450))?;
        assert_eq!(
            capture
                .iter()
                .map(|asset| asset.asset_id)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        Ok(())
    }

    #[test]
    fn root_listing_does_not_match_a_sibling_with_the_same_prefix() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().join("gallery"))?;
        let sibling = PathBuf::try_from(temp.path().join("gallery-other"))?;
        fs::create_dir_all(&root)?;
        fs::create_dir_all(&sibling)?;
        let inside = root.join("inside.bmp");
        let outside = sibling.join("outside.bmp");
        File::create(&inside)?;
        File::create(&outside)?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let inside = catalog.upsert(&inside, &fs::metadata(&inside)?)?;
        catalog.upsert(&outside, &fs::metadata(&outside)?)?;

        let listed = catalog.under_root(&canonicalize_path(&root)?)?;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].asset_id, inside.asset_id);
        Ok(())
    }

    fn query_plan(catalog: &AssetCatalog, order: &str) -> Result<String> {
        Ok(catalog.conn.query_row(
            &format!("EXPLAIN QUERY PLAN SELECT asset_id FROM assets ORDER BY {order}"),
            [],
            |row| row.get(3),
        )?)
    }

    fn catalog_revision(catalog: &AssetCatalog) -> Result<i64> {
        Ok(catalog.conn.query_row(
            "SELECT revision FROM catalog_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?)
    }

    #[cfg(windows)]
    #[test]
    fn normalizes_windows_verbatim_paths() {
        assert_eq!(
            normalize_windows_verbatim_path(PathBuf::from("//?/C:/gallery/image.png")),
            PathBuf::from("C:/gallery/image.png")
        );
        assert_eq!(
            normalize_windows_verbatim_path(PathBuf::from("//?/UNC/server/share/image.png")),
            PathBuf::from("//server/share/image.png")
        );
    }
}
