//! Library definitions: which folders a library includes and excludes, and the search indexing it
//! wants. Stored beside the asset catalog because every consumer of a library is a catalog query.
//!
//! A library never owns assets. Its scope is a [`PathScope`] evaluated against catalog paths at
//! query time, so editing folders changes what a library shows without rewriting any asset or
//! derived data, and overlapping libraries share every cataloged file.

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior};

use crate::assets::AssetCatalog;
use crate::scope::PathScope;

/// Which search indexes a library prepares for its files. Cataloging always happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LibraryOptions {
    pub ocr: bool,
    pub image: bool,
}

impl Default for LibraryOptions {
    /// Image search is cheap enough to run by default; text recognition is an explicit choice.
    fn default() -> Self {
        Self {
            ocr: false,
            image: true,
        }
    }
}

/// The caller-editable part of a library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryDefinition {
    pub include: Vec<PathBuf>,
    pub exclude: Vec<PathBuf>,
    pub options: LibraryOptions,
}

impl LibraryDefinition {
    /// Check the structural rules that do not need the filesystem. Callers canonicalize paths
    /// first; this compares spellings literally, as [`PathScope`] does.
    pub fn validate(&self) -> Result<()> {
        if self.include.is_empty() {
            bail!("a library needs at least one included folder");
        }
        for path in self.include.iter().chain(&self.exclude) {
            if !path.is_absolute() {
                bail!("library folders must be absolute: {path}");
            }
        }
        let all = self.include.iter().chain(&self.exclude).collect::<Vec<_>>();
        for (index, path) in all.iter().enumerate() {
            if all[..index].iter().any(|other| same_folder(other, path)) {
                bail!("a folder appears more than once in the library: {path}");
            }
        }
        for exclude in &self.exclude {
            if !self
                .include
                .iter()
                .any(|include| PathScope::root(include).contains(exclude))
            {
                bail!("an excluded folder must be inside an included folder: {exclude}");
            }
        }
        for include in &self.include {
            if let Some(exclude) = self
                .exclude
                .iter()
                .find(|exclude| PathScope::root(exclude).contains(include))
            {
                bail!("included folder {include} is inside excluded folder {exclude}");
            }
        }
        Ok(())
    }
}

/// Why the latest attempt to scan an included folder stopped short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOutcome {
    /// The folder itself could not be opened, for example because its drive is offline.
    Unavailable,
    /// The walk finished without seeing every file, so nothing it missed was removed.
    Incomplete,
    /// The scan was cancelled before it finished.
    Cancelled,
    /// Anything else went wrong.
    Failed,
}

impl ScanOutcome {
    /// The value stored in `library_folders.scan_outcome` and reported by the API.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            Self::Incomplete => "incomplete",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "unavailable" => Some(Self::Unavailable),
            "incomplete" => Some(Self::Incomplete),
            "cancelled" => Some(Self::Cancelled),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

impl ToSql for ScanOutcome {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for ScanOutcome {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let text = value.as_str()?;
        Self::parse(text)
            .ok_or_else(|| FromSqlError::Other(format!("unknown scan outcome {text:?}").into()))
    }
}

/// An included folder and whether a scan it needs is still outstanding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncludedFolder {
    pub path: PathBuf,
    /// A definition change asked for this folder to be scanned, and no scan has finished since.
    pub scan_pending: bool,
    /// Why the most recent scan of this folder stopped short, cleared by the next complete one.
    pub scan_outcome: Option<ScanOutcome>,
    /// Human-readable detail for `scan_outcome`, cleared with it.
    pub scan_error: Option<String>,
    /// Unix nanoseconds when a scan of this folder last finished completely.
    pub last_scan_completed_ns: Option<i64>,
    /// The folder's latest scan request number, unique within the library. A scan reads it when
    /// it starts and hands it back to [`AssetCatalog::complete_folder_scan`], so a request made
    /// while the scan runs stays pending.
    pub scan_request: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Library {
    pub id: i64,
    pub include: Vec<IncludedFolder>,
    pub exclude: Vec<PathBuf>,
    pub options: LibraryOptions,
}

/// A known directory from a completed full scan. The scope key invalidates snapshots after
/// exclusion edits without depending on the filesystem's timestamp behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorySnapshot {
    pub path: PathBuf,
    pub modified_ns: i64,
}

impl Library {
    /// The catalog rows this library shows.
    pub fn scope(&self) -> PathScope {
        PathScope::new(
            self.include
                .iter()
                .map(|folder| folder.path.clone())
                .collect(),
            self.exclude.clone(),
        )
    }

    pub fn definition(&self) -> LibraryDefinition {
        LibraryDefinition {
            include: self
                .include
                .iter()
                .map(|folder| folder.path.clone())
                .collect(),
            exclude: self.exclude.clone(),
            options: self.options,
        }
    }
}

/// What happened to a create request.
#[derive(Debug)]
pub enum Created {
    New(Library),
    /// A library with the same import key already exists; it is returned unchanged.
    Existing(Library),
}

/// Folder spellings that differ only by trailing separators name the same folder.
fn same_folder(left: &Path, right: &Path) -> bool {
    let trim = |path: &Path| path.as_str().trim_end_matches(['/', '\\']).to_owned();
    trim(left) == trim(right)
}

impl AssetCatalog {
    pub fn directory_snapshot(
        &self,
        library_id: i64,
        folder: &Path,
        scope_key: &str,
    ) -> Result<Vec<DirectorySnapshot>> {
        let mut statement = self.conn.prepare_cached(
            "SELECT path, modified_ns FROM library_directory_snapshots
             WHERE library_id = ?1 AND folder_path = ?2 AND scope_key = ?3 ORDER BY path",
        )?;
        statement
            .query_map((library_id, folder.as_str(), scope_key), |row| {
                Ok(DirectorySnapshot {
                    path: PathBuf::from(row.get::<_, String>(0)?),
                    modified_ns: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("reading directory snapshot")
    }

    pub fn replace_directory_snapshot(
        &mut self,
        library_id: i64,
        folder: &Path,
        scope_key: &str,
        directories: &[DirectorySnapshot],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM library_directory_snapshots WHERE library_id = ?1 AND folder_path = ?2",
            (library_id, folder.as_str()),
        )?;
        {
            let mut insert = tx.prepare_cached(
                "INSERT INTO library_directory_snapshots
                 (library_id, folder_path, path, modified_ns, scope_key)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for directory in directories {
                insert.execute((
                    library_id,
                    folder.as_str(),
                    directory.path.as_str(),
                    directory.modified_ns,
                    scope_key,
                ))?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn libraries(&self) -> Result<Vec<Library>> {
        let ids = self
            .conn
            .prepare_cached("SELECT library_id FROM libraries ORDER BY library_id")?
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        ids.into_iter()
            .map(|id| {
                read_library(&self.conn, id)?
                    .with_context(|| format!("library {id} disappeared while listing"))
            })
            .collect()
    }

    pub fn library(&self, id: i64) -> Result<Option<Library>> {
        read_library(&self.conn, id)
    }

    /// Create a library whose included folders all await their first scan. With an
    /// `import_key`, repeating the request returns the library it created the first time.
    pub fn create_library(
        &mut self,
        definition: &LibraryDefinition,
        import_key: Option<&str>,
    ) -> Result<Created> {
        definition.validate()?;
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(key) = import_key
            && let Some(id) = tx
                .query_row(
                    "SELECT library_id FROM libraries WHERE import_key = ?1",
                    [key],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
        {
            let existing = read_library(&tx, id)?.context("imported library disappeared")?;
            return Ok(Created::Existing(existing));
        }
        tx.execute(
            "INSERT INTO libraries(scan_sequence, index_ocr, index_image, import_key)
             VALUES (0, ?1, ?2, ?3)",
            (definition.options.ocr, definition.options.image, import_key),
        )?;
        let id = tx.last_insert_rowid();
        write_folders(&tx, id, definition, &[])?;
        let library = read_library(&tx, id)?.context("created library disappeared")?;
        tx.commit()?;
        Ok(Created::New(library))
    }

    /// Replace a library's definition, returning `None` for an unknown library. Folders whose
    /// visible contents can grow are marked for a scan: new includes, includes that contained a
    /// removed exclusion, and every include when a search option is switched on.
    pub fn update_library(
        &mut self,
        id: i64,
        definition: &LibraryDefinition,
    ) -> Result<Option<Library>> {
        definition.validate()?;
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let Some(current) = read_library(&tx, id)? else {
            return Ok(None);
        };
        if current.definition() == *definition {
            return Ok(Some(current));
        }

        let enabled = (definition.options.ocr && !current.options.ocr)
            || (definition.options.image && !current.options.image);
        let revealed = current
            .exclude
            .iter()
            .filter(|old| !definition.exclude.iter().any(|new| same_folder(old, new)))
            .collect::<Vec<_>>();
        let rescan = definition
            .include
            .iter()
            .filter(|include| {
                enabled
                    || revealed
                        .iter()
                        .any(|revealed| PathScope::root(include).contains(revealed))
            })
            .cloned()
            .collect::<Vec<_>>();

        tx.execute(
            "UPDATE libraries SET index_ocr = ?2, index_image = ?3 WHERE library_id = ?1",
            (id, definition.options.ocr, definition.options.image),
        )?;
        write_folders(&tx, id, definition, &rescan)?;
        let library = read_library(&tx, id)?.context("updated library disappeared")?;
        tx.commit()?;
        Ok(Some(library))
    }

    /// Record that a scan of an included folder finished completely. `scan_request` is the
    /// folder's [`IncludedFolder::scan_request`] when the scan started; later requests stay
    /// pending. A folder removed from the library meanwhile is ignored.
    pub fn complete_folder_scan(
        &self,
        id: i64,
        path: &Path,
        scan_request: i64,
        completed_ns: i64,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE library_folders
             SET scan_completed = max(scan_completed, ?3), scan_outcome = NULL,
                 scan_error = NULL, last_scan_completed_ns = ?4
             WHERE library_id = ?1 AND path = ?2 AND excluded = 0",
            (id, path.as_str(), scan_request, completed_ns),
        )?;
        Ok(())
    }

    /// A successful quick check clears the outstanding request but preserves the full-scan clock.
    pub fn complete_folder_check(&self, id: i64, path: &Path, scan_request: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE library_folders SET scan_completed = max(scan_completed, ?3),
             scan_outcome = NULL, scan_error = NULL
             WHERE library_id = ?1 AND path = ?2 AND excluded = 0",
            (id, path.as_str(), scan_request),
        )?;
        Ok(())
    }

    /// Record why a scan of an included folder could not finish. The folder stays pending.
    pub fn fail_folder_scan(
        &self,
        id: i64,
        path: &Path,
        outcome: ScanOutcome,
        message: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE library_folders SET scan_outcome = ?3, scan_error = ?4
             WHERE library_id = ?1 AND path = ?2 AND excluded = 0",
            (id, path.as_str(), outcome, message),
        )?;
        Ok(())
    }

    /// Mark an attempted request as stopped without hiding a newer edit's scan request.
    pub fn fail_folder_scan_request(
        &self,
        id: i64,
        path: &Path,
        scan_request: i64,
        outcome: ScanOutcome,
        message: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE library_folders SET scan_outcome = ?4, scan_error = ?5
             WHERE library_id = ?1 AND path = ?2 AND excluded = 0
               AND scan_requested = ?3 AND scan_requested > scan_completed
               AND scan_outcome IS NULL",
            (id, path.as_str(), scan_request, outcome, message),
        )?;
        Ok(())
    }

    /// Replace stored folder spellings with canonical ones, keeping each folder's scan state.
    /// A folder saved while offline keeps the spelling it was given, which can differ in case or
    /// separators from the canonical paths the catalog records, so scans call this once a folder
    /// is reachable. Respellings that would leave an invalid definition are not applied. Returns
    /// `None` for an unknown library.
    pub fn respell_folders(
        &mut self,
        id: i64,
        spellings: &[(PathBuf, PathBuf)],
    ) -> Result<Option<Library>> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let Some(current) = read_library(&tx, id)? else {
            return Ok(None);
        };
        // Spellings compare as strings: `Path` equality ignores the very differences (`.`
        // components, separator style) that respelling exists to remove.
        let spellings = spellings
            .iter()
            .filter(|(old, new)| old.as_str() != new.as_str())
            .collect::<Vec<_>>();
        let respell = |path: &PathBuf| {
            spellings
                .iter()
                .find(|(old, _)| old.as_str() == path.as_str())
                .map_or_else(|| path.clone(), |(_, new)| new.clone())
        };
        let before = current.definition();
        let mut definition = before.clone();
        definition.include = before.include.iter().map(respell).collect();
        definition.exclude = before.exclude.iter().map(respell).collect();
        let changed = before.include.iter().chain(&before.exclude).any(|path| {
            spellings
                .iter()
                .any(|(old, _)| old.as_str() == path.as_str())
        });
        if !changed || definition.validate().is_err() {
            return Ok(Some(current));
        }
        let mut rename = tx.prepare_cached(
            "UPDATE library_folders SET path = ?3 WHERE library_id = ?1 AND path = ?2",
        )?;
        for (old, new) in spellings {
            rename.execute((id, old.as_str(), new.as_str()))?;
        }
        drop(rename);
        // The paths inside snapshots use the old spelling. The next scan will rebuild them.
        tx.execute(
            "DELETE FROM library_directory_snapshots WHERE library_id = ?1",
            [id],
        )?;
        let library = read_library(&tx, id)?.context("respelled library disappeared")?;
        tx.commit()?;
        Ok(Some(library))
    }

    /// Forget a library. Cataloged files and their search data are kept.
    pub fn delete_library(&mut self, id: i64) -> Result<bool> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM library_directory_snapshots WHERE library_id = ?1",
            [id],
        )?;
        tx.execute("DELETE FROM library_folders WHERE library_id = ?1", [id])?;
        let deleted = tx.execute("DELETE FROM libraries WHERE library_id = ?1", [id])? > 0;
        tx.commit()?;
        Ok(deleted)
    }
}

/// Make `library_id`'s folder rows match `definition`, keeping the scan state of folders that
/// stay included. New includes start pending; `rescan` includes become pending again.
fn write_folders(
    tx: &Transaction<'_>,
    library_id: i64,
    definition: &LibraryDefinition,
    rescan: &[PathBuf],
) -> Result<()> {
    let existing = tx
        .prepare_cached(
            "SELECT path, excluded, scan_requested, scan_completed, scan_outcome, scan_error,
                    last_scan_completed_ns
             FROM library_folders WHERE library_id = ?1",
        )?
        .query_map([library_id], |row| {
            Ok((
                PathBuf::from(row.get::<_, String>(0)?),
                row.get::<_, bool>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<ScanOutcome>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<i64>>(6)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    tx.execute(
        "DELETE FROM library_folders WHERE library_id = ?1",
        [library_id],
    )?;
    let mut sequence: i64 = tx.query_row(
        "SELECT scan_sequence FROM libraries WHERE library_id = ?1",
        [library_id],
        |row| row.get(0),
    )?;
    let mut insert = tx.prepare_cached(
        "INSERT INTO library_folders
            (library_id, path, excluded, position, scan_requested, scan_completed, scan_outcome,
             scan_error, last_scan_completed_ns)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
    )?;
    let folders = definition
        .include
        .iter()
        .map(|path| (path, false))
        .chain(definition.exclude.iter().map(|path| (path, true)));
    for (position, (path, excluded)) in folders.enumerate() {
        let previous = existing
            .iter()
            .find(|(old, old_excluded, ..)| *old_excluded == excluded && same_folder(old, path));
        let (mut requested, completed, mut outcome, mut error, last_completed) = match previous {
            Some((_, _, requested, completed, outcome, error, last_completed)) => (
                *requested,
                *completed,
                *outcome,
                error.clone(),
                *last_completed,
            ),
            None => (0, 0, None, None, None),
        };
        // A fresh number, never one an earlier row for this path held, so a scan that started
        // before the folder was removed and re-added cannot complete this request.
        if !excluded && (previous.is_none() || rescan.iter().any(|folder| folder == path)) {
            sequence += 1;
            requested = sequence;
            outcome = None;
            error = None;
        }
        insert.execute((
            library_id,
            path.as_str(),
            excluded,
            i64::try_from(position)?,
            requested,
            completed,
            outcome,
            error,
            last_completed,
        ))?;
    }
    tx.execute(
        "DELETE FROM library_directory_snapshots
         WHERE library_id = ?1 AND folder_path NOT IN
           (SELECT path FROM library_folders WHERE library_id = ?1 AND excluded = 0)",
        [library_id],
    )?;
    tx.execute(
        "UPDATE libraries SET scan_sequence = ?2 WHERE library_id = ?1",
        (library_id, sequence),
    )?;
    Ok(())
}

fn read_library(conn: &rusqlite::Connection, id: i64) -> Result<Option<Library>> {
    let Some((ocr, image)) = conn
        .prepare_cached("SELECT index_ocr, index_image FROM libraries WHERE library_id = ?1")?
        .query_row([id], |row| {
            Ok((row.get::<_, bool>(0)?, row.get::<_, bool>(1)?))
        })
        .optional()?
    else {
        return Ok(None);
    };
    let mut include = Vec::new();
    let mut exclude = Vec::new();
    let mut statement = conn.prepare_cached(
        "SELECT path, excluded, scan_requested > scan_completed, scan_outcome, scan_error,
                last_scan_completed_ns, scan_requested
         FROM library_folders WHERE library_id = ?1 ORDER BY position",
    )?;
    let mut rows = statement.query([id])?;
    while let Some(row) = rows.next()? {
        let path = PathBuf::from(row.get::<_, String>(0)?);
        if row.get::<_, bool>(1)? {
            exclude.push(path);
        } else {
            include.push(IncludedFolder {
                path,
                scan_pending: row.get(2)?,
                scan_outcome: row.get(3)?,
                scan_error: row.get(4)?,
                last_scan_completed_ns: row.get(5)?,
                scan_request: row.get(6)?,
            });
        }
    }
    Ok(Some(Library {
        id,
        include,
        exclude,
        options: LibraryOptions { ocr, image },
    }))
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn catalog(temp: &TempDir) -> Result<AssetCatalog> {
        AssetCatalog::new(&PathBuf::try_from(temp.path().join("assets.db"))?)
    }

    /// Unix-style test paths are not absolute on Windows without a drive.
    const DRIVE: &str = if cfg!(windows) { "C:" } else { "" };

    fn abs(path: &str) -> PathBuf {
        if path.starts_with('/') {
            PathBuf::from(format!("{DRIVE}{path}"))
        } else {
            PathBuf::from(path)
        }
    }

    fn definition(include: &[&str], exclude: &[&str]) -> LibraryDefinition {
        LibraryDefinition {
            include: include.iter().map(|path| abs(path)).collect(),
            exclude: exclude.iter().map(|path| abs(path)).collect(),
            options: LibraryOptions::default(),
        }
    }

    fn created(outcome: Created) -> Library {
        match outcome {
            Created::New(library) => library,
            Created::Existing(library) => panic!("unexpected existing library {library:?}"),
        }
    }

    fn updated(outcome: Option<Library>) -> Library {
        outcome.expect("the library exists")
    }

    fn pending(library: &Library) -> Vec<&str> {
        library
            .include
            .iter()
            .filter(|folder| folder.scan_pending)
            .map(|folder| &folder.path.as_str()[DRIVE.len()..])
            .collect()
    }

    /// Complete a scan of every included folder, as a finished library scan does.
    fn complete_scans(catalog: &AssetCatalog, id: i64) -> Result<()> {
        let library = catalog.library(id)?.expect("the library exists");
        for folder in &library.include {
            catalog.complete_folder_scan(id, &folder.path, folder.scan_request, 7)?;
        }
        Ok(())
    }

    #[test]
    fn definitions_reject_structural_mistakes() {
        for (include, exclude, message) in [
            (&[][..], &[][..], "at least one included folder"),
            (&["relative"], &[], "must be absolute"),
            (&["/a", "/a/"], &[], "more than once"),
            (&["/a"], &["/a"], "more than once"),
            (&["/a"], &["/b/c"], "must be inside an included folder"),
            (&["/a"], &["/a-old/c"], "must be inside an included folder"),
            (
                &["/a", "/a/private/kept"],
                &["/a/private"],
                "is inside excluded",
            ),
        ] {
            let error = definition(include, exclude)
                .validate()
                .expect_err(message)
                .to_string();
            assert!(error.contains(message), "{error}");
        }
        definition(&["/a", "/a/nested", "/b"], &["/a/private", "/b/c"])
            .validate()
            .unwrap();
    }

    #[test]
    fn libraries_round_trip_and_scope_their_folders() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let library = created(catalog.create_library(
            &definition(&["/photos", "/phone"], &["/photos/private"]),
            None,
        )?);
        assert_eq!(pending(&library), ["/photos", "/phone"]);
        assert_eq!(catalog.library(library.id)?, Some(library.clone()));
        assert_eq!(catalog.libraries()?, std::slice::from_ref(&library));

        let scope = library.scope();
        assert!(scope.contains(&abs("/phone/a.png")));
        assert!(!scope.contains(&abs("/photos/private/b.png")));

        assert!(catalog.delete_library(library.id)?);
        assert!(!catalog.delete_library(library.id)?);
        assert_eq!(catalog.libraries()?, []);
        let orphans: i64 =
            catalog
                .conn
                .query_row("SELECT count(*) FROM library_folders", [], |row| row.get(0))?;
        assert_eq!(orphans, 0);
        Ok(())
    }

    #[test]
    fn an_import_key_makes_creation_idempotent() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let first = created(catalog.create_library(&definition(&["/a"], &[]), Some("legacy:/a"))?);
        let Created::Existing(again) =
            catalog.create_library(&definition(&["/a"], &[]), Some("legacy:/a"))?
        else {
            panic!("a repeated import must not create another library");
        };
        assert_eq!(again, first);
        created(catalog.create_library(&definition(&["/a"], &[]), None)?);
        assert_eq!(catalog.libraries()?.len(), 2);
        Ok(())
    }

    #[test]
    fn an_unchanged_update_changes_nothing() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let library = created(catalog.create_library(&definition(&["/a"], &[]), None)?);
        let edited = updated(catalog.update_library(library.id, &definition(&["/a", "/b"], &[]))?);
        assert_eq!(pending(&edited), ["/a", "/b"]);
        let same = updated(catalog.update_library(library.id, &edited.definition())?);
        assert_eq!(same, edited);
        assert!(
            catalog
                .update_library(library.id + 1, &definition(&["/c"], &[]))?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn a_request_made_during_a_scan_survives_its_completion() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let library = created(catalog.create_library(&definition(&["/a"], &["/a/x"]), None)?);
        let started = library.include[0].clone();

        // The scan is running when an edit reveals more of the folder.
        updated(catalog.update_library(library.id, &definition(&["/a"], &[]))?);
        catalog.fail_folder_scan(
            library.id,
            &started.path,
            ScanOutcome::Incomplete,
            "one file was unreadable",
        )?;
        let failed = &catalog.library(library.id)?.unwrap().include[0];
        assert_eq!(failed.scan_outcome, Some(ScanOutcome::Incomplete));
        assert_eq!(
            failed.scan_error.as_deref(),
            Some("one file was unreadable")
        );
        catalog.complete_folder_scan(library.id, &started.path, started.scan_request, 42)?;

        let folder = &catalog.library(library.id)?.unwrap().include[0];
        assert!(
            folder.scan_pending,
            "the edit's request is still outstanding"
        );
        assert_eq!(folder.scan_outcome, None);
        assert_eq!(folder.scan_error, None);
        assert_eq!(folder.last_scan_completed_ns, Some(42));

        catalog.complete_folder_scan(library.id, &folder.path, folder.scan_request, 43)?;
        assert!(!catalog.library(library.id)?.unwrap().include[0].scan_pending);
        Ok(())
    }

    #[test]
    fn stopped_scan_does_not_poison_a_newer_request() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let library = created(catalog.create_library(&definition(&["/a"], &["/a/private"]), None)?);
        let old = library.include[0].clone();
        updated(catalog.update_library(library.id, &definition(&["/a"], &[]))?);
        catalog.fail_folder_scan_request(
            library.id,
            &old.path,
            old.scan_request,
            ScanOutcome::Cancelled,
            "cancelled",
        )?;
        let current = catalog.library(library.id)?.unwrap().include[0].clone();
        assert!(current.scan_pending);
        assert_eq!(current.scan_outcome, None);
        assert_eq!(current.scan_error, None);

        // The current request records its stop, and a later stop does not overwrite the first.
        for (outcome, message) in [
            (ScanOutcome::Cancelled, "cancelled"),
            (ScanOutcome::Failed, "later"),
        ] {
            catalog.fail_folder_scan_request(
                library.id,
                &current.path,
                current.scan_request,
                outcome,
                message,
            )?;
        }
        let stopped = &catalog.library(library.id)?.unwrap().include[0];
        assert_eq!(stopped.scan_outcome, Some(ScanOutcome::Cancelled));
        assert_eq!(stopped.scan_error.as_deref(), Some("cancelled"));
        Ok(())
    }

    #[test]
    fn a_scan_from_before_a_folder_was_readded_cannot_complete_it() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let library = created(catalog.create_library(&definition(&["/a", "/b"], &[]), None)?);
        let started = library.include[1].clone();

        // While a scan of /b runs, /b is removed and added back.
        updated(catalog.update_library(library.id, &definition(&["/a"], &[]))?);
        updated(catalog.update_library(library.id, &definition(&["/a", "/b"], &[]))?);
        catalog.complete_folder_scan(library.id, &started.path, started.scan_request, 42)?;

        let readded = catalog.library(library.id)?.unwrap();
        assert_eq!(pending(&readded), ["/a", "/b"]);
        assert!(readded.include[1].scan_request > started.scan_request);
        Ok(())
    }

    #[test]
    fn respelling_keeps_scan_state_and_skips_invalid_results() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let library = created(catalog.create_library(
            &definition(&["/photos", "/other"], &["/photos/private"]),
            None,
        )?);
        catalog.fail_folder_scan(
            library.id,
            &abs("/photos"),
            ScanOutcome::Unavailable,
            "offline",
        )?;

        let respelled = catalog
            .respell_folders(
                library.id,
                &[
                    (abs("/photos"), abs("/Photos")),
                    (abs("/photos/private"), abs("/Photos/Private")),
                ],
            )?
            .unwrap();
        assert_eq!(
            respelled.definition(),
            definition(&["/Photos", "/other"], &["/Photos/Private"])
        );
        assert_eq!(
            respelled.include[0].scan_outcome,
            Some(ScanOutcome::Unavailable)
        );
        assert_eq!(respelled.include[0].scan_error.as_deref(), Some("offline"));
        assert_eq!(
            respelled.include[0].scan_request,
            library.include[0].scan_request
        );

        // An include respelled without its exclusion would leave the exclusion outside it.
        let unchanged = catalog
            .respell_folders(library.id, &[(abs("/Photos"), abs("/photos"))])?
            .unwrap();
        assert_eq!(unchanged, respelled);
        assert!(catalog.respell_folders(library.id + 1, &[])?.is_none());
        Ok(())
    }

    #[test]
    fn only_edits_that_reveal_files_request_scans() -> Result<()> {
        let temp = TempDir::new()?;
        let mut catalog = catalog(&temp)?;
        let library = created(catalog.create_library(
            &definition(&["/a", "/b"], &["/a/private", "/b/private"]),
            None,
        )?);
        complete_scans(&catalog, library.id)?;

        // Adding an exclusion or removing a folder only hides files.
        let hidden = updated(
            catalog.update_library(library.id, &definition(&["/a"], &["/a/private", "/a/more"]))?,
        );
        assert!(pending(&hidden).is_empty());

        // Adding a folder scans only that folder.
        let added = updated(catalog.update_library(
            library.id,
            &definition(&["/a", "/c"], &["/a/private", "/a/more"]),
        )?);
        assert_eq!(pending(&added), ["/c"]);
        complete_scans(&catalog, library.id)?;

        // Removing an exclusion scans the folders that contained it.
        let revealed =
            updated(catalog.update_library(library.id, &definition(&["/a", "/c"], &["/a/more"]))?);
        assert_eq!(pending(&revealed), ["/a"]);
        complete_scans(&catalog, library.id)?;

        // Switching a search index on prepares every folder; switching it off prepares none.
        let mut options = revealed.definition();
        options.options.ocr = true;
        assert_eq!(
            pending(&updated(catalog.update_library(library.id, &options)?)),
            ["/a", "/c"]
        );
        complete_scans(&catalog, library.id)?;
        options.options.ocr = false;
        assert!(pending(&updated(catalog.update_library(library.id, &options)?)).is_empty());
        Ok(())
    }
}
