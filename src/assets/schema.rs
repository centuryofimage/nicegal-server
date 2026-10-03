use anyhow::{Context, Result};
use camino::Utf8Path as Path;
use rusqlite::Connection;

use super::AssetCatalog;
use crate::schema::{check_schema_read_only, open_schema_with_migrations};
use crate::storage::{READ_ONLY_FLAGS, configure_reader, configure_writer, maintain};

const SCHEMA_VERSION: i32 = 11;
const SCHEMA_LABEL: &str = "asset catalog";
const MIGRATIONS: &[(i32, &str)] = &[
    (
        2,
        "ALTER TABLE assets ADD COLUMN metadata_version INTEGER NOT NULL DEFAULT 1 CHECK(metadata_version > 0);",
    ),
    (
        3,
        "ALTER TABLE assets ADD COLUMN source_created_ns INTEGER;",
    ),
    (
        4,
        "CREATE TABLE decode_failure_state(
        asset_id INTEGER PRIMARY KEY,
        source_modified_ns INTEGER NOT NULL,
        source_size INTEGER NOT NULL CHECK(source_size >= 0)
    );",
    ),
    (5, "VACUUM;"),
    (6, include_str!("../migrations/assets_6_to_7.sql")),
    (7, include_str!("../migrations/assets_7_to_8.sql")),
    (8, include_str!("../migrations/assets_8_to_9.sql")),
    (9, include_str!("../migrations/assets_9_to_10.sql")),
    (10, include_str!("../migrations/assets_10_to_11.sql")),
];

impl AssetCatalog {
    pub fn new(path: &Path) -> Result<Self> {
        let conn =
            Connection::open(path).with_context(|| format!("opening asset catalog: {path}"))?;
        configure_writer(&conn)?;
        open_schema_with_migrations(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("../assets_create.sql"),
            MIGRATIONS,
        )?;
        Ok(Self { conn })
    }

    /// Open a query-only connection suitable for concurrent lookups while an index job writes.
    /// Unlike [`AssetCatalog::new`] this never creates the schema, so a caller cannot silently
    /// read an empty catalog it just brought into existence.
    pub fn new_read_only(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(path, READ_ONLY_FLAGS)
            .with_context(|| format!("opening asset catalog read-only: {path}"))?;
        configure_reader(&conn)?;
        check_schema_read_only(&conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
        Ok(Self { conn })
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub fn maintain(&self) -> Result<()> {
        maintain(&self.conn).context("maintaining asset catalog")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::*;
    use crate::scope::PathScope;
    use std::collections::HashSet;
    use tempfile::TempDir;

    #[test]
    fn migrates_existing_catalogs_through_the_current_version() -> Result<()> {
        for old_version in 2..SCHEMA_VERSION {
            let temp = TempDir::new()?;
            let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
            let conn = Connection::open(&catalog_path)?;
            conn.execute_batch(
                "CREATE TABLE assets(asset_id INTEGER PRIMARY KEY, path TEXT);
             INSERT INTO assets VALUES (42, 'legacy.png');
             PRAGMA user_version = 2;",
            )?;
            for (from, sql) in MIGRATIONS {
                if *from < old_version {
                    conn.execute_batch(sql)?;
                }
            }
            conn.pragma_update(None, "user_version", old_version)?;
            drop(conn);

            let catalog = AssetCatalog::new(&catalog_path)?;
            let version: i32 = catalog.conn.query_row(
                "SELECT user_version FROM pragma_user_version",
                [],
                |row| row.get(0),
            )?;
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
            let retained: (i64, i32, Option<i64>) = catalog.conn.query_row(
                "SELECT asset_id, metadata_version, source_created_ns FROM assets",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            assert_eq!(retained, (42, 1, None));
            catalog
                .conn
                .execute("INSERT INTO decode_failure_state VALUES (42, 1, 1)", [])?;
            assert_eq!(catalog.libraries()?, []);
            drop(catalog);
            AssetCatalog::new(&catalog_path)?;
        }
        Ok(())
    }

    /// Every schema object except SQLite's own, with whitespace folded so formatting differences
    /// in the DDL text do not count.
    fn schema_of(conn: &Connection) -> Result<Vec<(String, String, String)>> {
        let mut statement = conn.prepare(
            "SELECT type, name, coalesce(sql, '') FROM sqlite_master
             WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" "),
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// A real version-6 catalog, created from the schema that version shipped with, migrates to
    /// exactly the schema a new catalog gets, keeps every row, and reopens without migrating again.
    #[test]
    fn a_version_six_catalog_migrates_to_the_fresh_schema_and_keeps_its_rows() -> Result<()> {
        let temp = TempDir::new()?;
        // Library roots are validated as absolute on the host platform.
        let root = if cfg!(windows) {
            "C:/photos"
        } else {
            "/photos"
        };
        let (image, clip) = (format!("{root}/a.png"), format!("{root}/clip.mp4"));
        let old_path = PathBuf::try_from(temp.path().join("old.db"))?;
        let conn = Connection::open(&old_path)?;
        configure_writer(&conn)?;
        conn.execute_batch(include_str!("../../tests/fixtures/schema/assets_v6.sql"))?;
        conn.execute_batch(&format!(
            "INSERT INTO assets(asset_id, path, source_modified_ns, source_created_ns,
                 exif_taken_ns, source_size, media_kind, media_format, width, height,
                 is_animated, frame_count, duration_ms, metadata_version)
             VALUES (7, '{image}', 10, 11, 12, 13, 'image', 'png', 640, 480, 0,
                     NULL, NULL, 2),
                    (9, '{clip}', 20, NULL, NULL, 30, 'video', 'mp4', 1920,
                     1080, 0, 300, 10000, 3);
             INSERT INTO decode_failure_state VALUES (7, 10, 13);
             UPDATE catalog_meta SET revision = 42;",
        ))?;
        drop(conn);
        // Readers never migrate: they refuse the old version until a writer has upgraded it.
        assert!(AssetCatalog::new_read_only(&old_path).is_err());

        let migrated = AssetCatalog::new(&old_path)?;
        let fresh_path = PathBuf::try_from(temp.path().join("fresh.db"))?;
        let fresh = AssetCatalog::new(&fresh_path)?;
        assert_eq!(schema_of(&migrated.conn)?, schema_of(&fresh.conn)?);
        let version: i32 =
            migrated
                .conn
                .query_row("SELECT user_version FROM pragma_user_version", [], |row| {
                    row.get(0)
                })?;
        assert_eq!(version, SCHEMA_VERSION);

        let assets = migrated.all()?;
        assert_eq!(
            assets
                .iter()
                .map(|asset| (asset.asset_id, asset.path.as_str()))
                .collect::<Vec<_>>(),
            [(7, image.as_str()), (9, clip.as_str())]
        );
        assert_eq!(assets[1].duration_ms, Some(10_000));
        assert_eq!(migrated.revision()?, 42);
        assert_eq!(
            migrated.current_decode_failure_asset_ids(&[(7, assets[0].fingerprint)])?,
            HashSet::from([7])
        );
        assert_eq!(migrated.libraries()?, []);
        let integrity: String = migrated
            .conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
        assert_eq!(integrity, "ok");
        drop(migrated);

        // The migrated catalog now serves readers and reopens as current, and its new tables work.
        let reader = AssetCatalog::new_read_only(&old_path)?;
        assert_eq!(reader.count_gallery(&PathScope::root(Path::new(root)))?, 2);
        drop(reader);
        let mut reopened = AssetCatalog::new(&old_path)?;
        let library = reopened.create_library(
            &crate::libraries::LibraryDefinition {
                include: vec![PathBuf::from(root)],
                exclude: Vec::new(),
                options: Default::default(),
            },
            Some(root),
        )?;
        assert!(matches!(library, crate::libraries::Created::New(_)));
        assert_eq!(
            reopened.revision()?,
            42,
            "library edits are not catalog changes"
        );
        Ok(())
    }

    /// A version-7 catalog gains the structured scan outcome; a folder whose failure was only
    /// recorded as text becomes a generic failure and keeps the text.
    #[test]
    fn a_version_seven_catalog_migrates_folder_errors_to_outcomes() -> Result<()> {
        use crate::libraries::ScanOutcome;

        let temp = TempDir::new()?;
        let old_path = PathBuf::try_from(temp.path().join("old.db"))?;
        let conn = Connection::open(&old_path)?;
        configure_writer(&conn)?;
        conn.execute_batch(include_str!("../../tests/fixtures/schema/assets_v7.sql"))?;
        conn.execute_batch(
            "INSERT INTO libraries VALUES (1, 2, 0, 1, NULL);
             INSERT INTO library_folders VALUES
                 (1, 'C:/offline', 0, 0, 1, 0, 'drive is offline', NULL),
                 (1, 'C:/photos', 0, 1, 2, 2, NULL, 42);",
        )?;
        drop(conn);

        let migrated = AssetCatalog::new(&old_path)?;
        let fresh = AssetCatalog::new(&PathBuf::try_from(temp.path().join("fresh.db"))?)?;
        assert_eq!(schema_of(&migrated.conn)?, schema_of(&fresh.conn)?);
        let library = migrated.library(1)?.expect("the library survives");
        let folders = library
            .include
            .iter()
            .map(|folder| {
                (
                    folder.path.as_str(),
                    folder.scan_pending,
                    folder.scan_outcome,
                    folder.scan_error.as_deref(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            folders,
            [
                (
                    "C:/offline",
                    true,
                    Some(ScanOutcome::Failed),
                    Some("drive is offline")
                ),
                ("C:/photos", false, None, None),
            ]
        );
        Ok(())
    }
}
