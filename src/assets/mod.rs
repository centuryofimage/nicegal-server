use std::time::UNIX_EPOCH;

use anyhow::{Result, bail};
use camino::Utf8PathBuf as PathBuf;
use rusqlite::Connection;

mod media;
mod paths;
mod query;
mod schema;
mod search;
mod upsert;

pub use media::{is_catalog_media, is_catalog_video, is_ocr_image};
pub use paths::canonicalize_path;
pub use search::{FileSearchField, FileSearchHit};
pub use upsert::SourceFingerprint;
pub(crate) use upsert::{CatalogScanEntry, CatalogUpsertTimings};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
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

/// A folder for the library tree: a configured root or a directory from a completed scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FolderEntry {
    pub path: PathBuf,
    pub modified_ns: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    /// SQLite row identifier; SQLite exposes rowids as signed 64-bit integers.
    pub asset_id: i64,
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

pub struct AssetCatalog {
    pub(crate) conn: Connection,
}

// Keep this projection in the positional order consumed by asset_from_row.
const ASSET_COLUMNS: &str = "asset_id, path, source_modified_ns, source_created_ns,
    exif_taken_ns, source_size, media_kind, media_format, width, height, is_animated,
    frame_count, duration_ms, metadata_version";

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

#[cfg(test)]
mod test_support {
    use super::*;

    pub(super) fn catalog_revision(catalog: &AssetCatalog) -> Result<i64> {
        Ok(catalog.conn.query_row(
            "SELECT revision FROM catalog_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?)
    }
}
