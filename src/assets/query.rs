use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use rusqlite::{OptionalExtension, params_from_iter};

use super::{
    ASSET_COLUMNS, Asset, AssetCatalog, FolderEntry, SourceFingerprint, Timeline, asset_from_row,
};
use crate::scope::PathScope;
use crate::storage::{bind_named, validate_asset_ids};

impl AssetCatalog {
    /// The scan snapshot is updated once per completed folder scan, not once per indexed file.
    /// Keep configured roots visible before the first scan and while a drive is offline. A folder's
    /// time is its directory modification time from the latest completed scan, or `None` before
    /// one; nested included roots can snapshot the same directory, so the newest time wins.
    pub fn list_folders(&self, library_id: i64, scope: &PathScope) -> Result<Vec<FolderEntry>> {
        let mut folders: BTreeMap<PathBuf, Option<i64>> = scope
            .include()
            .iter()
            .map(|path| (path.clone(), None))
            .collect();
        let mut statement = self.conn.prepare(
            "SELECT path, modified_ns FROM library_directory_snapshots WHERE library_id = ?1",
        )?;
        let rows = statement.query_map([library_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let (path, modified_ns) = row?;
            let path = PathBuf::from(path);
            if scope.contains_folder(&path) {
                let entry = folders.entry(path).or_default();
                *entry = (*entry).max(Some(modified_ns));
            }
        }
        Ok(folders
            .into_iter()
            .map(|(path, modified_ns)| FolderEntry { path, modified_ns })
            .collect())
    }

    pub fn get(&self, asset_id: i64) -> Result<Option<Asset>> {
        self.conn
            .query_row(
                &format!("SELECT {ASSET_COLUMNS} FROM assets WHERE asset_id = ?1"),
                [asset_id],
                asset_from_row,
            )
            .optional()
            .context("loading asset catalog entry")
    }

    pub fn get_by_path(&self, path: &Path) -> Result<Option<Asset>> {
        self.conn
            .query_row(
                &format!("SELECT {ASSET_COLUMNS} FROM assets WHERE path = ?1"),
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
                "SELECT {ASSET_COLUMNS} FROM assets
                 WHERE asset_id IN ({placeholders})"
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
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT {ASSET_COLUMNS} FROM assets ORDER BY asset_id"
        ))?;
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
            "SELECT {ASSET_COLUMNS} FROM assets
             WHERE 1 = 1 {predicate} ORDER BY {expression}, asset_id"
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
        self.in_scope(&PathScope::root(root))
    }

    /// Every asset in `scope`, in asset ID order.
    pub fn in_scope(&self, scope: &PathScope) -> Result<Vec<Asset>> {
        let scope = scope.bind("path");
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT {ASSET_COLUMNS} FROM assets
             WHERE {} ORDER BY asset_id",
            scope.sql
        ))?;
        let rows =
            statement.query_and_then(bind_named(&scope.params).as_slice(), asset_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("listing assets in scope")
    }

    /// Every asset in `scope`, newest first on `timeline`. Scope membership is a literal path
    /// match, so an offline library remains browsable.
    pub fn list_gallery(&self, scope: &PathScope, timeline: Timeline) -> Result<Vec<Asset>> {
        let order = timeline.expression("assets");
        let scope = scope.bind("path");
        let mut statement = self.conn.prepare(&format!(
            "SELECT {ASSET_COLUMNS} FROM assets
             WHERE {}
             ORDER BY {order} DESC, asset_id DESC",
            scope.sql
        ))?;
        Ok(statement
            .query_and_then(bind_named(&scope.params).as_slice(), asset_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn count_gallery(&self, scope: &PathScope) -> Result<i64> {
        let scope = scope.bind("path");
        Ok(self.conn.query_row(
            &format!("SELECT COUNT(*) FROM assets WHERE {}", scope.sql),
            bind_named(&scope.params).as_slice(),
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
        crate::storage::matching_fingerprint_ids(
            &self.conn,
            "SELECT EXISTS(\
                 SELECT 1 FROM decode_failure_state \
                  WHERE asset_id = ?1 AND source_modified_ns = ?2 AND source_size = ?3\
             )",
            fingerprints,
        )
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
        let ids = std::iter::repeat_n("?", asset_ids.len())
            .collect::<Vec<_>>()
            .join(",");
        tx.execute(
            &format!("DELETE FROM decode_failure_state WHERE asset_id IN ({ids})"),
            params_from_iter(asset_ids),
        )?;
        let deleted = tx.execute(
            &format!("DELETE FROM assets WHERE asset_id IN ({ids})"),
            params_from_iter(asset_ids),
        )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::*;
    use std::fs::{self, File};
    use tempfile::TempDir;

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
        assert_eq!(catalog.count_gallery(&PathScope::root(&root))?, 3);
        assert_eq!(
            ids(catalog.list_gallery(&PathScope::root(&root), Timeline::Modified)?),
            [3, 2, 1]
        );
        assert_eq!(
            ids(catalog.list_gallery(&PathScope::root(&root), Timeline::Capture)?),
            [3, 1, 2]
        );
        let trailing = PathBuf::from(format!("{root}/"));
        assert_eq!(catalog.count_gallery(&PathScope::root(&trailing))?, 3);
        assert_eq!(
            catalog.count_gallery(&PathScope::root(&base.join("empty")))?,
            0
        );
        assert_eq!(catalog.revision()?, 0);
        catalog.delete_assets(&[1])?;
        assert_eq!(catalog.revision()?, 1);
        assert_eq!(catalog.count_gallery(&PathScope::root(&root))?, 2);
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
}
