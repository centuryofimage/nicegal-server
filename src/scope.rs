//! Which cataloged paths a library, search, or maintenance request covers.
//!
//! Every store that filters by location renders its predicate from a [`PathScope`], so the
//! gallery, search, coverage, and cleanup can never disagree about which files are inside.

use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use rusqlite::types::Value;

/// Characters that separate path components in stored paths. Windows accepts both spellings;
/// elsewhere a backslash is an ordinary filename character.
#[cfg(windows)]
const SEPARATORS: &[char] = &['/', '\\'];
#[cfg(not(windows))]
const SEPARATORS: &[char] = &['/'];

/// Descendants of any included directory that are not descendants of an excluded one.
///
/// Membership is a literal, case-sensitive, separator-delimited prefix of the stored path: `D:\a`
/// covers `D:\a\b.png` but neither `D:\a-old\b.png` nor `D:\a` itself. Stored paths are
/// canonical, so callers must pass canonical directories (or ones read back from the catalog).
/// Overlapping includes are harmless because each row is tested once, and an exclusion wins over
/// every include.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PathScope {
    include: Vec<PathBuf>,
    exclude: Vec<PathBuf>,
}

impl PathScope {
    pub fn new(include: Vec<PathBuf>, exclude: Vec<PathBuf>) -> Self {
        Self { include, exclude }
    }

    /// Everything underneath one directory.
    pub fn root(root: &Path) -> Self {
        Self::new(vec![root.to_owned()], Vec::new())
    }

    pub fn with_exclude(mut self, exclude: impl IntoIterator<Item = PathBuf>) -> Self {
        self.exclude.extend(exclude);
        self
    }

    pub fn include(&self) -> &[PathBuf] {
        &self.include
    }

    pub fn exclude(&self) -> &[PathBuf] {
        &self.exclude
    }

    /// Intersect library membership with one folder without widening an included root.
    pub fn focused(&self, folder: &Path) -> Self {
        let include = self
            .include
            .iter()
            .filter_map(|root| {
                if same_or_descendant(folder, root) {
                    Some(folder.to_owned())
                } else if same_or_descendant(root, folder) {
                    Some(root.clone())
                } else {
                    None
                }
            })
            .collect();
        Self::new(include, self.exclude.clone())
    }

    /// A configured root can be shown in the tree even though it contains no assets yet.
    pub fn contains_folder(&self, folder: &Path) -> bool {
        (self
            .include
            .iter()
            .any(|root| same_or_descendant(folder, root))
            || self
                .include
                .iter()
                .any(|root| same_or_descendant(root, folder)))
            && !self
                .exclude
                .iter()
                .any(|root| same_or_descendant(folder, root))
    }

    /// The Rust spelling of [`PathScope::bind`], for filtering paths that are not in SQLite.
    pub fn contains(&self, path: &Path) -> bool {
        let under = |directory: &PathBuf| {
            let prefix = trim_separators(directory);
            path.as_str()
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with(SEPARATORS))
        };
        self.include.iter().any(under) && !self.exclude.iter().any(under)
    }

    /// Render the scope as one parenthesized SQL predicate on `column`, plus the named parameters
    /// it references. Each directory becomes a half-open range per separator, which is exact for
    /// SQLite's default byte-wise `BINARY` collation and can use an index on the column.
    pub(crate) fn bind(&self, column: &str) -> BoundScope {
        let mut params = Vec::new();
        let mut ranges = |directories: &[PathBuf]| {
            let ranges = directories
                .iter()
                .flat_map(|directory| {
                    let prefix = trim_separators(directory);
                    SEPARATORS.iter().map(move |separator| {
                        let upper = char::from(*separator as u8 + 1);
                        (format!("{prefix}{separator}"), format!("{prefix}{upper}"))
                    })
                })
                .map(|(low, high)| {
                    let index = params.len() / 2;
                    let (low_name, high_name) = (
                        format!(":scope_{index}_low"),
                        format!(":scope_{index}_high"),
                    );
                    let sql = format!("({column} >= {low_name} AND {column} < {high_name})");
                    params.push((low_name, Value::Text(low)));
                    params.push((high_name, Value::Text(high)));
                    sql
                })
                .collect::<Vec<_>>();
            if ranges.is_empty() {
                "0".to_owned()
            } else {
                ranges.join(" OR ")
            }
        };
        let include = ranges(&self.include);
        let sql = if self.exclude.is_empty() {
            format!("({include})")
        } else {
            let exclude = ranges(&self.exclude);
            format!("(({include}) AND NOT ({exclude}))")
        };
        BoundScope { sql, params }
    }
}

fn same_or_descendant(path: &Path, directory: &Path) -> bool {
    let prefix = trim_separators(directory);
    path.as_str() == prefix
        || path
            .as_str()
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with(SEPARATORS))
}

/// A directory's spelling without trailing separators, so `C:\` and `/` become prefixes that a
/// single separator completes.
fn trim_separators(directory: &Path) -> &str {
    directory.as_str().trim_end_matches(SEPARATORS)
}

/// The SQL a [`PathScope`] expands to, paired with exactly the parameters it references.
#[derive(Debug, Clone)]
pub(crate) struct BoundScope {
    pub(crate) sql: String,
    pub(crate) params: Vec<(String, Value)>,
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rusqlite::Connection;

    use super::*;
    use crate::storage::bind_named;

    /// Assert that SQLite and [`PathScope::contains`] agree with `expected` for every path.
    fn check(scope: &PathScope, cases: &[(&str, bool)]) -> Result<()> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("CREATE TABLE paths(path TEXT NOT NULL UNIQUE)")?;
        for (path, _) in cases {
            conn.execute("INSERT INTO paths VALUES (?1)", [path])?;
        }
        let bound = scope.bind("paths.path");
        let mut statement = conn.prepare(&format!(
            "SELECT path FROM paths WHERE {} ORDER BY path",
            bound.sql
        ))?;
        let matched = statement
            .query_map(bind_named(&bound.params).as_slice(), |row| {
                row.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut expected = cases
            .iter()
            .filter(|(_, inside)| *inside)
            .map(|(path, _)| path.to_string())
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(matched, expected, "{scope:?}");
        for (path, inside) in cases {
            assert_eq!(
                scope.contains(Path::new(path)),
                *inside,
                "{path} in {scope:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_root_covers_descendants_only_at_literal_separator_boundaries() -> Result<()> {
        let cases = [
            ("/gallery_100%#/photo.png", true),
            ("/gallery_100%#/nested/photo.png", true),
            ("/gallery_100%#", false),
            ("/gallery_100%#-old/photo.png", false),
            ("/galleryX100%#/photo.png", false),
            ("/gallery_100anything#/photo.png", false),
            ("/Gallery_100%#/photo.png", false),
            ("/elsewhere/photo.png", false),
        ];
        check(&PathScope::root(Path::new("/gallery_100%#")), &cases)?;
        check(&PathScope::root(Path::new("/gallery_100%#/")), &cases)?;
        Ok(())
    }

    #[test]
    fn filesystem_roots_cover_everything_beneath_them() -> Result<()> {
        check(
            &PathScope::root(Path::new("/")),
            &[("/photo.png", true), ("/a/b.png", true)],
        )?;
        check(
            &PathScope::root(Path::new("C:/")),
            &[("C:/photo.png", true), ("D:/photo.png", false)],
        )?;
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn windows_accepts_either_separator() -> Result<()> {
        let cases = [
            (r"C:\gallery\photo.png", true),
            (r"C:\gallery/photo.png", true),
            (r"C:\gallery/mixed\photo.png", true),
            (r"C:\gallery-old\photo.png", false),
            (r"C:\gallery]\photo.png", false),
            (r"C:\gallery0\photo.png", false),
        ];
        check(&PathScope::root(Path::new(r"C:\gallery")), &cases)?;
        check(&PathScope::root(Path::new(r"C:\gallery\")), &cases)?;
        check(
            &PathScope::root(Path::new(r"C:\")),
            &[(r"C:\photo.png", true), ("C:/photo.png", true)],
        )?;
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn backslash_is_a_filename_character_outside_windows() -> Result<()> {
        check(
            &PathScope::root(Path::new("/gallery")),
            &[("/gallery/photo.png", true), ("/gallery\\photo.png", false)],
        )
    }

    #[test]
    fn includes_union_and_exclusions_win() -> Result<()> {
        let scope = PathScope::new(
            vec!["/a".into(), "/b".into(), "/a/nested".into()],
            vec!["/a/private".into(), "/b/private".into()],
        );
        check(
            &scope,
            &[
                ("/a/one.png", true),
                ("/a/nested/two.png", true),
                ("/b/three.png", true),
                ("/a/private/four.png", false),
                ("/a/private/deeper/five.png", false),
                ("/a/private-not/six.png", true),
                ("/b/private/seven.png", false),
                ("/c/eight.png", false),
            ],
        )
    }

    #[test]
    fn folder_focus_intersects_includes_and_keeps_exclusions() -> Result<()> {
        let library = PathScope::new(
            vec!["/photos".into(), "/phone".into()],
            vec!["/photos/private".into()],
        );
        let focus = library.focused(Path::new("/photos/trips"));
        check(
            &focus,
            &[
                ("/photos/trips/a.jpg", true),
                ("/photos/trips-old/b.jpg", false),
                ("/phone/c.jpg", false),
            ],
        )?;
        let parent = library.focused(Path::new("/"));
        check(
            &parent,
            &[
                ("/photos/a.jpg", true),
                ("/phone/b.jpg", true),
                ("/photos/private/c.jpg", false),
            ],
        )?;
        assert!(
            !library
                .focused(Path::new("/other"))
                .contains(Path::new("/other/a.jpg"))
        );
        Ok(())
    }

    #[test]
    fn a_scope_without_includes_covers_nothing() -> Result<()> {
        check(
            &PathScope::new(Vec::new(), vec!["/a".into()]),
            &[("/a/one.png", false), ("/b/two.png", false)],
        )?;
        check(&PathScope::default(), &[("/a/one.png", false)])
    }

    #[test]
    fn range_predicates_use_the_path_index() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("CREATE TABLE paths(path TEXT NOT NULL UNIQUE)")?;
        let bound = PathScope::root(Path::new("/gallery")).bind("path");
        let plan = conn
            .prepare(&format!(
                "EXPLAIN QUERY PLAN SELECT path FROM paths WHERE {}",
                bound.sql
            ))?
            .query_map(bind_named(&bound.params).as_slice(), |row| {
                row.get::<_, String>(3)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        assert!(
            plan.iter()
                .any(|step| step.contains("USING COVERING INDEX")),
            "{plan:?}"
        );
        Ok(())
    }
}
