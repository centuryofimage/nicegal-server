use std::cmp::Ordering;

use anyhow::Result;
use camino::Utf8PathBuf as PathBuf;

use super::AssetCatalog;
use crate::cancellation::SearchCancellation;
use crate::db::{FilterScope, SearchFilters};
use crate::file_filter::PathPattern;
use crate::storage::bind_named;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSearchHit {
    pub asset_id: i64,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSearchField {
    Name,
    Path,
}

/// Compare lowercased UTF-8 names and paths by ASCII numeric runs and byte order otherwise.
/// Using the same ordering for every string keeps mixed ASCII and Unicode results transitive.
fn natural_file_cmp(left: &[u8], right: &[u8]) -> Ordering {
    let (mut a, mut b) = (0, 0);
    while a < left.len() && b < right.len() {
        if left[a].is_ascii_digit() && right[b].is_ascii_digit() {
            let start_a = a;
            let start_b = b;
            while a < left.len() && left[a].is_ascii_digit() {
                a += 1;
            }
            while b < right.len() && right[b].is_ascii_digit() {
                b += 1;
            }
            let digits_a = &left[start_a..a];
            let digits_b = &right[start_b..b];
            let trimmed_a = digits_a
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&digits_a[digits_a.len() - 1..], |index| &digits_a[index..]);
            let trimmed_b = digits_b
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&digits_b[digits_b.len() - 1..], |index| &digits_b[index..]);
            let compared = trimmed_a
                .len()
                .cmp(&trimmed_b.len())
                .then(trimmed_a.cmp(trimmed_b));
            if compared != Ordering::Equal {
                return compared;
            }
            continue;
        }
        let compared = left[a].cmp(&right[b]);
        if compared != Ordering::Equal {
            return compared;
        }
        a += 1;
        b += 1;
    }
    left.len()
        .saturating_sub(a)
        .cmp(&right.len().saturating_sub(b))
}

impl AssetCatalog {
    pub fn set_search_cancellation(&self, cancellation: &SearchCancellation) -> Result<()> {
        cancellation.register(&self.conn)
    }

    /// Search indexed paths inside the same library scope used by gallery listing. FTS5 narrows
    /// candidates by the pattern's longest ordinary ASCII run; short, Unicode, and
    /// wildcard-character runs use a scoped scan. The final comparison is always
    /// [`PathPattern`]'s, so no match inherits SQLite LIKE's case or wildcard behavior. An empty
    /// query returns every file the filters admit.
    pub fn search_files(
        &self,
        filters: &SearchFilters,
        field: FileSearchField,
        query: &str,
        limit: usize,
        cancellation: &SearchCancellation,
    ) -> Result<(usize, Vec<FileSearchHit>)> {
        cancellation.check()?;
        let pattern = match field {
            FileSearchField::Name => PathPattern::name(query),
            FileSearchField::Path => PathPattern::path(query),
        };
        let literal = pattern.required_literal();
        let indexed = literal.len() >= 4 && literal.is_ascii() && !literal.contains(['%', '_']);
        let bound = filters.bind(FilterScope::FILE_ROWS)?;
        let mut params = bound.params;
        let mut where_sql = format!("1{}", bound.sql);
        if indexed {
            where_sql.push_str(" AND f.path LIKE :path_pattern");
            params.push((
                ":path_pattern".to_owned(),
                rusqlite::types::Value::Text(format!("%{literal}%")),
            ));
        }
        let from = if indexed {
            "asset_path_fts f JOIN assets a ON a.asset_id = f.rowid"
        } else {
            "assets a"
        };
        let mut statement = self.conn.prepare(&format!(
            "SELECT a.asset_id, a.path FROM {from} WHERE {where_sql}"
        ))?;
        let rows = statement.query_map(bind_named(&params).as_slice(), |row| {
            Ok(FileSearchHit {
                asset_id: row.get(0)?,
                path: PathBuf::from(row.get::<_, String>(1)?),
            })
        })?;
        let mut matches = Vec::new();
        for row in rows {
            cancellation.check()?;
            let hit = row?;
            if pattern.matches(hit.path.as_str()) {
                matches.push(hit);
            }
        }
        let total = matches.len();
        let mut matches = matches
            .into_iter()
            .map(|hit| {
                let text = match field {
                    FileSearchField::Name => hit.path.file_name().unwrap_or(hit.path.as_str()),
                    FileSearchField::Path => hit.path.as_str(),
                };
                let sort_key = text.to_lowercase();
                (hit, sort_key)
            })
            .collect::<Vec<_>>();
        matches.sort_unstable_by(|(left, left_key), (right, right_key)| {
            natural_file_cmp(left_key.as_bytes(), right_key.as_bytes())
                .then(left.asset_id.cmp(&right.asset_id))
        });
        matches.truncate(limit);
        Ok((total, matches.into_iter().map(|(hit, _)| hit).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::*;
    use crate::file_filter::FileFilter;
    use crate::scope::PathScope;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn path_search_uses_scope_and_updates_its_index() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_path_buf())?;
        let catalog = AssetCatalog::new(&root.join("assets.db"))?;
        let trips = root.join("Trips 2025");
        let excluded = trips.join("Excluded");
        fs::create_dir_all(&excluded)?;
        let wanted = trips.join("Café_100%.jpg");
        let hidden = excluded.join("Café_100%.jpg");
        let outside = root.join("Elsewhere.jpg");
        for path in [&wanted, &hidden, &outside] {
            fs::write(path, b"image")?;
            catalog.upsert(path, &fs::metadata(path)?)?;
        }
        let scope = PathScope::root(&root).with_exclude([excluded]);
        let cancellation = SearchCancellation::default();
        let find = |term: &str| {
            catalog.search_files(
                &SearchFilters::new(scope.clone()),
                FileSearchField::Path,
                term,
                10,
                &cancellation,
            )
        };
        for term in ["TRIPS", "Café_100%", "Trips 2025\\Café"] {
            let (total, hits) = find(term)?;
            assert_eq!(total, 1, "{term}");
            assert_eq!(hits[0].path, wanted, "{term}");
        }
        let filters = SearchFilters::new(scope.clone())
            .with_files([FileFilter::path("TRIPS 2025\\CAFÉ", false)]);
        let (total, hits) =
            catalog.search_files(&filters, FileSearchField::Name, "Café", 1, &cancellation)?;
        assert_eq!(total, 1);
        assert_eq!(hits[0].path, wanted);
        assert_eq!(find("elsewhere")?.0, 1);
        catalog.delete_asset(catalog.get_by_path(&wanted)?.unwrap().asset_id)?;
        assert_eq!(find("trips")?.0, 0);
        Ok(())
    }

    #[test]
    fn name_search_ignores_folder_matches_and_sorts_numbers_naturally() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_path_buf())?;
        let catalog = AssetCatalog::new(&root.join("assets.db"))?;
        let folder = root.join("photo folder");
        fs::create_dir_all(&folder)?;
        for name in ["photo10.jpg", "photo2.jpg", "other.jpg"] {
            let path = folder.join(name);
            fs::write(&path, b"image")?;
            catalog.upsert(&path, &fs::metadata(&path)?)?;
        }
        let cancellation = SearchCancellation::default();
        let scope = PathScope::root(&root);
        let (total, hits) = catalog.search_files(
            &SearchFilters::new(scope),
            FileSearchField::Name,
            "PHOTO",
            10,
            &cancellation,
        )?;
        assert_eq!(total, 2);
        assert_eq!(
            hits.iter()
                .map(|hit| hit.path.file_name().unwrap())
                .collect::<Vec<_>>(),
            ["photo2.jpg", "photo10.jpg"]
        );
        Ok(())
    }

    #[test]
    fn file_search_honors_wildcards_extensions_and_exclusions() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::try_from(temp.path().to_path_buf())?;
        let catalog = AssetCatalog::new(&root.join("assets.db"))?;
        let folder = root.join("2024");
        fs::create_dir_all(&folder)?;
        for name in [
            "IMG_0001.JPG",
            "IMG_0002.png",
            "clip.mp4",
            "IMG_notes.txt.jpg",
        ] {
            let path = folder.join(name);
            fs::write(&path, b"image")?;
            catalog.upsert(&path, &fs::metadata(&path)?)?;
        }
        let cancellation = SearchCancellation::default();
        let scope = PathScope::root(&root);
        let names = |filters: SearchFilters, field, query: &str| -> Result<Vec<String>> {
            let (_, hits) = catalog.search_files(&filters, field, query, 10, &cancellation)?;
            Ok(hits
                .into_iter()
                .map(|hit| hit.path.file_name().unwrap().to_owned())
                .collect())
        };
        let all = || SearchFilters::new(scope.clone());
        assert_eq!(
            names(all(), FileSearchField::Name, "img_????.*")?,
            ["IMG_0001.JPG", "IMG_0002.png"]
        );
        assert_eq!(
            names(all(), FileSearchField::Path, "*\\2024\\*.mp4")?,
            ["clip.mp4"]
        );
        assert_eq!(
            names(
                all().with_files([FileFilter::extensions(["jpg"], false)]),
                FileSearchField::Path,
                ""
            )?,
            ["IMG_0001.JPG", "IMG_notes.txt.jpg"]
        );
        assert_eq!(
            names(
                all().with_files([
                    FileFilter::extensions(["mp4"], true),
                    FileFilter::path("img_*.jpg", true),
                ]),
                FileSearchField::Name,
                "img"
            )?,
            ["IMG_0002.png"]
        );
        Ok(())
    }
}
