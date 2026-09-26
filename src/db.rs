use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Once;

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension, params_from_iter};
use sqlite_vec::sqlite3_vec_init;
use tracing::{Span, field, info, instrument};

use crate::assets::{SourceFingerprint, Timeline};
use crate::schema::{check_schema_read_only, open_schema_with_migrations};
use crate::scope::PathScope;
pub(crate) use crate::storage::bind_named;
use crate::storage::{
    READ_ONLY_FLAGS, configure_reader, configure_writer, maintain, validate_asset_ids,
};

const SCHEMA_VERSION: i32 = 10;
const SCHEMA_LABEL: &str = "OCR database";
const MIGRATIONS: &[(i32, &str)] = &[
    (8, include_str!("migrations/ocr_8_to_9.sql")),
    (9, "VACUUM;"),
];

/// Whether OCR data exists for an asset and still matches its catalog fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrIndexState {
    Indexed,
    Stale,
}

/// A search that failed because of the caller's query text rather than the index or the server.
///
/// SQLite reports FTS5 syntax problems, unknown column prefixes, and invalid REGEXP patterns as a
/// plain `SQLITE_ERROR` while *executing* a statement. The search statements are static and their
/// only caller-controlled input is the bound query string, so an execution-stage logic error is
/// always attributable to that query. Preparation-stage failures stay ordinary errors, which keeps
/// a damaged schema from being reported to the user as a bad query.
#[derive(Debug)]
pub struct QuerySyntaxError {
    message: String,
}

impl QuerySyntaxError {
    /// SQLite's own text, collapsed onto one line for display.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for QuerySyntaxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for QuerySyntaxError {}

/// SQLite's message when `error` was caused by the query text, otherwise `None`.
pub fn query_syntax_message(error: &anyhow::Error) -> Option<&str> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<QuerySyntaxError>())
        .map(QuerySyntaxError::message)
}

/// Tag an execution-stage SQLite logic error as a query problem, leaving every other failure
/// (I/O, corruption, busy timeouts) an ordinary error.
fn execution_error(error: rusqlite::Error) -> anyhow::Error {
    match error {
        rusqlite::Error::SqliteFailure(code, Some(message))
            if code.extended_code == rusqlite::ffi::SQLITE_ERROR =>
        {
            anyhow::Error::new(QuerySyntaxError {
                message: collapse_whitespace(&message),
            })
        }
        other => anyhow::Error::new(other),
    }
}

/// SQLite embeds newlines and alignment carets in some messages (notably regex parse errors).
/// Callers render these inline, so fold the runs of whitespace into single spaces.
fn collapse_whitespace(message: &str) -> String {
    message.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SearchType {
    Simple,
    Match,
    Glob,
    #[cfg(feature = "regex")]
    Regex,
}

/// What [`DB::search`] wraps each matched term of an FTS5 snippet in.
///
/// These are the markers the CLI prints and the markers
/// [`crate::highlight::from_marked`] turns back into spans, so they are named here rather than
/// spelled into the SQL twice and parsed from a third place.
pub const SNIPPET_OPEN: char = '[';
/// The closing half of [`SNIPPET_OPEN`].
pub const SNIPPET_CLOSE: char = ']';

/// A half-open instant range on one of the catalog's two timelines.
///
/// There is deliberately no default timeline. "When was this taken" and "when did this file last
/// change" are different questions with different answers, and silently picking one produces a
/// result set the caller cannot interpret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    pub timeline: Timeline,
    /// Inclusive lower bound, Unix nanoseconds.
    pub after_ns: Option<i64>,
    /// Exclusive upper bound, Unix nanoseconds. Half-open so adjacent ranges tile without
    /// double-counting the boundary instant.
    pub before_ns: Option<i64>,
}

impl TimeRange {
    fn validate(&self) -> Result<()> {
        if let (Some(after), Some(before)) = (self.after_ns, self.before_ns)
            && after >= before
        {
            bail!("search range afterNs must be less than beforeNs");
        }
        Ok(())
    }
}

/// Everything that narrows a search without being the query itself.
///
/// Every search mode takes this same value, so a filter is written once here rather than being
/// re-derived in each mode's SQL. [`SearchFilters::bind`] is what makes that safe: it emits the
/// `WHERE` fragments and the parameters they reference *together*, so the two can never drift
/// apart the way hand-numbered positional parameters do.
#[derive(Debug, Clone)]
pub struct SearchFilters {
    /// Only assets inside this scope.
    pub scope: PathScope,
    /// Restrict to a span on one timeline. `None` searches every instant.
    pub time: Option<TimeRange>,
    /// Case-insensitive substring of the indexed full path.
    pub path_contains: Option<String>,
}

impl SearchFilters {
    /// Filters with no bounds beyond the scope.
    pub fn new(scope: PathScope) -> Self {
        Self {
            scope,
            time: None,
            path_contains: None,
        }
    }

    /// Everything underneath one directory.
    pub fn under(root: &Path) -> Self {
        Self::new(PathScope::root(root))
    }

    pub fn with_time(mut self, time: Option<TimeRange>) -> Self {
        self.time = time;
        self
    }

    pub fn with_path_contains(mut self, path: Option<&str>) -> Self {
        self.path_contains = path
            .filter(|value| !value.is_empty())
            .map(|value| value.replace('\\', "/").to_lowercase());
        self
    }

    /// Render the filters as SQL fragments plus the named parameters they bind, against `rows`.
    /// Which rows a search narrows is the only thing that differs between the OCR store and the
    /// asset catalog; the fragments themselves are these and only these.
    pub(crate) fn bind(&self, rows: FilterScope) -> Result<BoundFilters> {
        let scope = self.scope.bind(rows.path_column);
        let mut sql = format!(
            "
               AND {}",
            scope.sql
        );
        let mut params = scope.params;

        if let Some(needle) = &self.path_contains {
            sql.push_str(&format!(
                " AND nicegal_path_contains({}, :path_contains)",
                rows.path_column
            ));
            params.push((":path_contains".to_owned(), Value::Text(needle.clone())));
        }

        if let Some(time) = self.time {
            time.validate()?;
            // The expression matches an index created beside the scoped table, one per timeline,
            // so a bounded search is a range scan rather than a table scan.
            let instant = time.timeline.expression(rows.table);
            if let Some(after) = time.after_ns {
                sql.push_str(&format!(
                    "
               AND {instant} >= :after_ns"
                ));
                params.push((":after_ns".to_owned(), Value::Integer(after)));
            }
            if let Some(before) = time.before_ns {
                sql.push_str(&format!(
                    "
               AND {instant} < :before_ns"
                ));
                params.push((":before_ns".to_owned(), Value::Integer(before)));
            }
        }

        Ok(BoundFilters { sql, params })
    }
}

/// The rows a [`SearchFilters`] narrows: the table its time filters order, and the qualified path
/// column its scope matches. Named once here so the same filter cannot render one way for the OCR
/// store and another way for the asset catalog.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterScope {
    /// The table, schema-qualified when it lives in an attached database.
    table: &'static str,
    /// The fully qualified column holding each row's source path.
    path_column: &'static str,
}

impl FilterScope {
    /// Rows of the OCR store.
    pub(crate) const OCR_ROWS: Self = Self {
        table: "ocr_results",
        path_column: "ocr_results.source_path",
    };

    /// Rows of the asset catalog, on a connection that has the catalog attached under the
    /// `catalog` schema alias. The image index's reader is the one that does.
    pub(crate) const CATALOG_ROWS: Self = Self {
        table: "catalog.assets",
        path_column: "catalog.assets.path",
    };
}

/// The SQL a [`SearchFilters`] expands to, paired with exactly the parameters it references.
/// Read by the image index's search path as well, so the two stores filter identically.
#[derive(Debug, Clone)]
pub(crate) struct BoundFilters {
    /// `AND`-joined fragments, ready to append to a `WHERE` clause that already has a term.
    pub(crate) sql: String,
    pub(crate) params: Vec<(String, Value)>,
}

impl BoundFilters {
    /// The filter parameters plus a mode's own, in one list to bind.
    pub(crate) fn extend<const N: usize>(
        &self,
        extra: [(&'static str, Value); N],
    ) -> Vec<(String, Value)> {
        extend_params(self.params.clone(), extra)
    }
}

fn extend_params<const N: usize>(
    mut params: Vec<(String, Value)>,
    extra: [(&'static str, Value); N],
) -> Vec<(String, Value)> {
    params.extend(extra.map(|(name, value)| (name.to_owned(), value)));
    params
}

/// Which coordinate system a vector belongs to.
///
/// Vectors are only comparable within a space: each has its own model, its own dimension count, and
/// its own `vec0` table. Image embeddings live in their own model-specific database instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TextEmbeddingSpace {
    /// Vectors of the OCR text extracted from an asset.
    OcrText,
}

impl TextEmbeddingSpace {
    /// Every space, for the maintenance paths that must touch all of them.
    pub const ALL: [Self; 1] = [Self::OcrText];

    /// The wire and storage spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OcrText => "ocrText",
        }
    }

    /// Parse the wire spelling. Named `from_wire` rather than `from_str` so it is not mistaken
    /// for the `FromStr` trait method, which would imply a `parse::<TextEmbeddingSpace>()` that does
    /// not exist.
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "ocrText" => Some(Self::OcrText),
            _ => None,
        }
    }

    /// The `vec0` table holding this space's vectors. One table per space because `vec0` fixes the
    /// dimension count in its DDL and two spaces do not share one.
    fn table(self) -> &'static str {
        match self {
            Self::OcrText => "ocr_embeddings_ocr_text",
        }
    }
}

impl fmt::Display for TextEmbeddingSpace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// OCR-derived data keyed by IDs owned by the asset catalog.
pub struct DB {
    conn: Connection,
}

impl DB {
    /// Attach cancellation to a reader owned exclusively by one search request.
    pub fn set_search_cancellation(
        &self,
        cancellation: &crate::cancellation::SearchCancellation,
    ) -> Result<()> {
        cancellation.register(&self.conn)
    }

    pub fn new(path: &Path) -> Result<Self> {
        if !path.try_exists()? {
            info!(path = %path, "creating a new OCR database");
        }
        register_vector_extension();
        let conn = Connection::open(path)?;
        configure_vector_writer(&conn)?;
        // `ocr_embedding_state` cascades from `ocr_results`, which SQLite only honours with
        // enforcement switched on. It is per-connection, so writers must set it every open.
        conn.pragma_update(None, "foreign_keys", true)?;
        #[cfg(feature = "regex")]
        register_regex(&conn)?;
        register_word_glob(&conn)?;

        open_schema_with_migrations(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("db_create.sql"),
            MIGRATIONS,
        )?;
        Ok(Self { conn })
    }

    /// Open a query-only connection suitable for concurrent HTTP search while an index job writes.
    pub fn new_read_only(path: &Path) -> Result<Self> {
        register_vector_extension();
        let conn = Connection::open_with_flags(path, READ_ONLY_FLAGS)?;
        configure_reader(&conn)?;
        configure_vector_reads(&conn)?;
        #[cfg(feature = "regex")]
        register_regex(&conn)?;
        register_word_glob(&conn)?;
        check_schema_read_only(&conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
        Ok(Self { conn })
    }

    /// Remove OCR rows and vectors whose owning catalog asset no longer exists.
    #[tracing::instrument(level = "debug", skip(self))]
    pub fn prune_orphans(&mut self, catalog: &Path) -> Result<usize> {
        self.conn
            .execute(
                "ATTACH DATABASE ?1 AS maintenance_catalog",
                [catalog.as_str()],
            )
            .with_context(|| format!("attaching asset catalog for OCR pruning: {catalog}"))?;
        let result = (|| {
            let deleted = self
                .conn
                .execute(
                    "DELETE FROM ocr_results
                 WHERE NOT EXISTS (
                     SELECT 1 FROM maintenance_catalog.assets
                     WHERE assets.asset_id = ocr_results.asset_id
                 )",
                    [],
                )
                .context("pruning orphaned OCR rows")?;
            self.sweep_text_embeddings()?;
            Ok(deleted)
        })();
        self.conn
            .execute_batch("DETACH DATABASE maintenance_catalog")
            .context("detaching asset catalog after OCR pruning")?;
        result
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub fn maintain(&self) -> Result<()> {
        maintain(&self.conn).context("maintaining OCR database")
    }

    /// Current OCR text's embedding status: None means no current OCR row, the pair is
    /// (has nonempty text, has a matching text embedding).
    pub fn asset_text_embedding_status(
        &self,
        asset_id: i64,
        fingerprint: SourceFingerprint,
    ) -> Result<Option<(bool, bool)>> {
        Ok(self
            .conn
            .query_row(
                "SELECT trim(content) <> '', EXISTS(
                SELECT 1 FROM ocr_embedding_state e
                WHERE e.asset_id = ocr_results.asset_id AND e.space = ?4
                  AND e.source_modified_ns = ocr_results.source_modified_ns
                  AND e.source_size = ocr_results.source_size
             ) FROM ocr_results
             WHERE asset_id = ?1 AND source_modified_ns = ?2 AND source_size = ?3
               AND mark_delete = FALSE",
                (
                    asset_id,
                    fingerprint.modified_ns,
                    i64::try_from(fingerprint.size)?,
                    TextEmbeddingSpace::OcrText.as_str(),
                ),
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?)
    }

    /// Return recognized text only when it belongs to the catalog's current source fingerprint.
    /// A stale OCR row is useful for search-state reporting, but must not be shown as text from
    /// the current file.
    pub fn current_asset_text(
        &self,
        asset_id: i64,
        fingerprint: SourceFingerprint,
    ) -> Result<Option<String>> {
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        Ok(self
            .conn
            .query_row(
                "SELECT content FROM ocr_results
                 WHERE asset_id = ?1 AND source_modified_ns = ?2 AND source_size = ?3
                   AND mark_delete = FALSE",
                (asset_id, fingerprint.modified_ns, source_size),
                |row| row.get(0),
            )
            .optional()?)
    }

    pub fn is_indexed(&self, asset_id: i64, fingerprint: SourceFingerprint) -> Result<bool> {
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM ocr_results WHERE asset_id = ?1 AND source_modified_ns = ?2 AND source_size = ?3)",
                (asset_id, fingerprint.modified_ns, source_size),
                |row| row.get(0),
            )
            .context("checking OCR source fingerprint")
    }

    pub fn current_asset_ids(
        &self,
        fingerprints: &[(i64, SourceFingerprint)],
    ) -> Result<HashSet<i64>> {
        Ok(self
            .index_states(fingerprints)?
            .into_iter()
            .filter_map(|(asset_id, state)| (state == OcrIndexState::Indexed).then_some(asset_id))
            .collect())
    }

    /// Return OCR state for catalog assets which have an OCR row.
    ///
    /// IDs absent from the returned map have never been indexed; a stale row is deliberately
    /// distinct from that state so the UI can offer re-indexing rather than imply no work exists.
    pub fn index_states(
        &self,
        fingerprints: &[(i64, SourceFingerprint)],
    ) -> Result<HashMap<i64, OcrIndexState>> {
        const QUERY_CHUNK_SIZE: usize = 512;

        let mut expected = HashMap::with_capacity(fingerprints.len());
        for &(asset_id, fingerprint) in fingerprints {
            if asset_id <= 0 {
                bail!("asset identifiers must be greater than zero");
            }
            i64::try_from(fingerprint.size)
                .context("source byte size exceeds SQLite's integer range")?;
            expected.insert(asset_id, fingerprint);
        }

        let mut states = HashMap::with_capacity(fingerprints.len());
        for chunk in fingerprints.chunks(QUERY_CHUNK_SIZE) {
            let placeholders = std::iter::repeat_n("?", chunk.len())
                .collect::<Vec<_>>()
                .join(",");
            let mut statement = self.conn.prepare(&format!(
                "SELECT asset_id, source_modified_ns, source_size \
                   FROM ocr_results WHERE asset_id IN ({placeholders})"
            ))?;
            let rows = statement.query_map(
                params_from_iter(chunk.iter().map(|(asset_id, _)| asset_id)),
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )?;
            for row in rows {
                let (asset_id, modified_ns, source_size) = row?;
                let source_size =
                    u64::try_from(source_size).context("stored OCR source size is negative")?;
                if let Some(expected) = expected.get(&asset_id) {
                    let state = if expected
                        == &(SourceFingerprint {
                            modified_ns,
                            size: source_size,
                        }) {
                        OcrIndexState::Indexed
                    } else {
                        OcrIndexState::Stale
                    };
                    states.insert(asset_id, state);
                }
            }
        }
        Ok(states)
    }

    #[instrument(name = "save_results", level = "debug", skip_all, fields(rows = results.len()))]
    pub fn save_results(&mut self, results: Vec<OcrResult>) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let rowchanges = {
            let mut statement = tx.prepare_cached(
                "INSERT INTO ocr_results (
                     asset_id, source_path, source_modified_ns, exif_taken_ns,
                     source_size, width, height, content
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(asset_id) DO UPDATE SET
                     source_path = excluded.source_path,
                     source_modified_ns = excluded.source_modified_ns,
                     exif_taken_ns = excluded.exif_taken_ns,
                     source_size = excluded.source_size,
                     width = excluded.width,
                     height = excluded.height,
                     content = excluded.content,
                     mark_delete = FALSE",
            )?;
            let mut changed = 0;
            for result in results {
                let source_size = i64::try_from(result.fingerprint.size)
                    .context("source byte size exceeds SQLite's integer range")?;
                changed += statement.execute((
                    result.asset_id,
                    result.path.as_str(),
                    result.fingerprint.modified_ns,
                    result.exif_taken_ns,
                    source_size,
                    result.width,
                    result.height,
                    result.contents,
                ))?;
            }
            changed
        };
        tx.commit()?;
        Ok(rowchanges)
    }

    pub fn mark_for_deletion(&mut self, path: &Path) -> Result<()> {
        if !path.is_dir() {
            bail!("path passed to mark_for_deletion must be a directory: {path}");
        }
        self.conn.execute(
            "UPDATE ocr_results SET mark_delete = FALSE WHERE mark_delete = TRUE",
            [],
        )?;
        let scope = PathScope::root(path).bind("source_path");
        self.conn.execute(
            &format!(
                "UPDATE ocr_results SET mark_delete = TRUE WHERE {}",
                scope.sql
            ),
            bind_named(&scope.params).as_slice(),
        )?;
        Ok(())
    }

    pub fn unmark_asset(&mut self, asset_id: i64) -> Result<()> {
        self.unmark_assets(&[asset_id])?;
        Ok(())
    }

    pub fn unmark_assets(&mut self, asset_ids: &[i64]) -> Result<usize> {
        validate_asset_ids(asset_ids)?;
        if asset_ids.is_empty() {
            return Ok(0);
        }
        let tx = self.conn.transaction()?;
        let changed = {
            let mut statement =
                tx.prepare("UPDATE ocr_results SET mark_delete = FALSE WHERE asset_id = ?1")?;
            asset_ids.iter().try_fold(0usize, |changed, asset_id| {
                statement.execute([asset_id]).map(|count| changed + count)
            })?
        };
        tx.commit()?;
        Ok(changed)
    }

    #[instrument(name = "sweep_deletions", level = "debug", skip_all)]
    pub fn sweep_deletions(&mut self) -> Result<usize> {
        let deleted = self
            .conn
            .execute("DELETE FROM ocr_results WHERE mark_delete = TRUE", [])
            .context("deleting stale OCR results")?;
        // The cascade reaches `ocr_embedding_state`; `vec0` is a virtual table, so its rows are
        // only reachable through the explicit sweep.
        self.sweep_text_embeddings()?;
        Ok(deleted)
    }

    pub fn delete_asset(&mut self, asset_id: i64) -> Result<usize> {
        self.delete_assets(&[asset_id])
    }

    pub fn delete_assets(&mut self, asset_ids: &[i64]) -> Result<usize> {
        validate_asset_ids(asset_ids)?;
        if asset_ids.is_empty() {
            return Ok(0);
        }
        let mut vector_tables = Vec::new();
        for space in TextEmbeddingSpace::ALL {
            if self.has_vector_table(space)? {
                vector_tables.push(space.table());
            }
        }
        let tx = self.conn.transaction()?;
        let ids = std::iter::repeat_n("?", asset_ids.len()).collect::<Vec<_>>().join(",");
        let deleted = tx.execute(
            &format!("DELETE FROM ocr_results WHERE asset_id IN ({ids})"),
            params_from_iter(asset_ids),
        )?;
        for table in vector_tables {
            tx.execute(
                &format!("DELETE FROM {table} WHERE asset_id IN ({ids})"),
                params_from_iter(asset_ids),
            )
            .context("deleting asset embeddings")?;
        }
        tx.execute(
            &format!("DELETE FROM ocr_embedding_state WHERE asset_id IN ({ids})"),
            params_from_iter(asset_ids),
        )
        .context("deleting asset embedding state")?;
        tx.commit()?;
        Ok(deleted)
    }

    #[instrument(
        name = "search",
        level = "debug",
        skip_all,
        fields(kind = ?kind, queries = queries.len(), limit, hits = field::Empty)
    )]
    pub fn search(
        &mut self,
        queries: Vec<&str>,
        filters: &SearchFilters,
        limit: usize,
        kind: SearchType,
    ) -> Result<Vec<SearchResult>> {
        if kind == SearchType::Glob {
            return self.search_glob(queries, filters, limit);
        }
        let bound = filters.bind(FilterScope::OCR_ROWS)?;
        let mut statement = self.conn.prepare_cached(&format!(
            r#"
            SELECT ocr_results.asset_id,
                   snippet(ocr_results_fts, -1, '{open}', '{close}', '..', 64),
                   ocr_results.source_path, ocr_results.source_modified_ns,
                   ocr_results.width, ocr_results.height, bm25(ocr_results_fts)
              {source}
             ORDER BY RANK, ocr_results.source_modified_ns DESC
             LIMIT :limit;
            "#,
            source = ocr_search_source(kind, &bound.sql),
            open = SNIPPET_OPEN,
            close = SNIPPET_CLOSE,
        ))?;
        let params = bound.extend([
            (":query", Value::Text(search_query(&queries, kind))),
            (
                ":limit",
                Value::Integer(
                    i64::try_from(limit).context("search limit exceeds SQLite's integer range")?,
                ),
            ),
        ]);
        let results = (|| -> rusqlite::Result<Vec<SearchResult>> {
            statement
                .query_and_then(
                    bind_named(&params).as_slice(),
                    |row| -> rusqlite::Result<SearchResult> {
                        Ok(SearchResult {
                            asset_id: row.get(0)?,
                            contents: row.get(1)?,
                            path: row.get(2)?,
                            modified_ns: row.get(3)?,
                            width: row.get(4)?,
                            height: row.get(5)?,
                            score: row.get(6)?,
                        })
                    },
                )?
                .collect()
        })()
        .map_err(execution_error)
        .context("querying OCR index")?;
        Span::current().record("hits", results.len());
        Ok(results)
    }

    #[instrument(
        name = "search_count",
        level = "debug",
        skip_all,
        fields(kind = ?kind, queries = queries.len(), total = field::Empty)
    )]
    pub fn search_count(
        &mut self,
        queries: Vec<&str>,
        filters: &SearchFilters,
        kind: SearchType,
    ) -> Result<usize> {
        if kind == SearchType::Glob {
            return self.search_glob_count(queries, filters);
        }
        let bound = filters.bind(FilterScope::OCR_ROWS)?;
        let mut statement = self.conn.prepare_cached(&format!(
            r#"
            SELECT count(*)
              {source};
            "#,
            source = ocr_search_source(kind, &bound.sql),
        ))?;
        let params = bound.extend([(":query", Value::Text(search_query(&queries, kind)))]);
        let count: i64 = statement
            .query_row(bind_named(&params).as_slice(), |row| row.get(0))
            .map_err(execution_error)
            .context("counting OCR search results")?;
        Span::current().record("total", count);
        usize::try_from(count).context("OCR search result count exceeds usize")
    }

    /// Search the word-token vocabulary, then follow only postings for matching terms. The primary
    /// OCR FTS table is trigram-indexed for the existing substring search modes, so it cannot
    /// distinguish a complete word from a substring; this sidecar uses Unicode word tokens.
    fn search_glob(
        &mut self,
        queries: Vec<&str>,
        filters: &SearchFilters,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let bound = filters.bind(FilterScope::OCR_ROWS)?;
        let mut statement = self.conn.prepare_cached(&format!(
            r#"
            {matching_terms},
            matching_documents AS (
                SELECT vocabulary.doc, count(*) AS matching_words
                  FROM ocr_results_words_vocab_instance AS vocabulary
                  INNER JOIN matching_terms USING (term)
                 WHERE vocabulary.col = 'content'
                 GROUP BY vocabulary.doc
            )
            SELECT ocr_results.asset_id, substr(ocr_results.content, 1, 512),
                   ocr_results.source_path, ocr_results.source_modified_ns,
                   ocr_results.width, ocr_results.height, matching_documents.matching_words
              {source}
             ORDER BY matching_documents.matching_words DESC, ocr_results.source_modified_ns DESC
             LIMIT :limit;
            "#,
            matching_terms = GLOB_MATCHING_TERMS,
            source = glob_search_source(&bound.sql),
        ))?;
        let params = bound.extend([
            (
                ":query",
                Value::Text(search_query(&queries, SearchType::Glob)),
            ),
            (
                ":limit",
                Value::Integer(
                    i64::try_from(limit).context("search limit exceeds SQLite's integer range")?,
                ),
            ),
        ]);
        let results = (|| -> rusqlite::Result<Vec<SearchResult>> {
            statement
                .query_and_then(
                    bind_named(&params).as_slice(),
                    |row| -> rusqlite::Result<SearchResult> {
                        let matching_words: i64 = row.get(6)?;
                        Ok(SearchResult {
                            asset_id: row.get(0)?,
                            contents: row.get(1)?,
                            path: row.get(2)?,
                            modified_ns: row.get(3)?,
                            width: row.get(4)?,
                            height: row.get(5)?,
                            score: matching_words as f64,
                        })
                    },
                )?
                .collect()
        })()
        .map_err(execution_error)
        .context("querying OCR word index")?;
        Span::current().record("hits", results.len());
        Ok(results)
    }

    fn search_glob_count(&mut self, queries: Vec<&str>, filters: &SearchFilters) -> Result<usize> {
        let bound = filters.bind(FilterScope::OCR_ROWS)?;
        let mut statement = self.conn.prepare_cached(&format!(
            r#"
            {matching_terms},
            matching_documents AS (
                SELECT DISTINCT vocabulary.doc
                  FROM ocr_results_words_vocab_instance AS vocabulary
                  INNER JOIN matching_terms USING (term)
                 WHERE vocabulary.col = 'content'
            )
            SELECT count(*)
              {source};
            "#,
            matching_terms = GLOB_MATCHING_TERMS,
            source = glob_search_source(&bound.sql),
        ))?;
        let params = bound.extend([(
            ":query",
            Value::Text(search_query(&queries, SearchType::Glob)),
        )]);
        let count: i64 = statement
            .query_row(bind_named(&params).as_slice(), |row| row.get(0))
            .map_err(execution_error)
            .context("counting OCR word index results")?;
        Span::current().record("total", count);
        usize::try_from(count).context("OCR search result count exceeds usize")
    }

    /// Count and fetch one bounded result set from the same WAL snapshot.
    pub fn search_with_count(
        &mut self,
        queries: Vec<&str>,
        filters: &SearchFilters,
        limit: usize,
        kind: SearchType,
    ) -> Result<(usize, Vec<SearchResult>)> {
        self.read_snapshot(|db| {
            let total = db.search_count(queries.clone(), filters, kind)?;
            let results = db.search(queries, filters, limit, kind)?;
            Ok((total, results))
        })
    }

    /// Run several queries against one WAL snapshot so a client combining search modes cannot see
    /// an index job commit between them and rank two inconsistent result sets together.
    pub fn read_snapshot<T>(&mut self, queries: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let mut snapshot = crate::storage::ReadSnapshot::begin(self, |db| &db.conn)
            .context("starting OCR read snapshot")?;
        let value = queries(snapshot.target)?;
        snapshot.commit().context("committing OCR read snapshot")?;
        Ok(value)
    }

    /// The model a space's stored vectors were produced by, or `None` when nothing has been
    /// embedded into it.
    ///
    /// A query must be embedded with this exact model; vectors from two models share a coordinate
    /// space only by coincidence, so each space deliberately holds one at a time.
    pub fn text_embedding_model(
        &self,
        space: TextEmbeddingSpace,
    ) -> Result<Option<StoredTextEmbeddingModel>> {
        self.conn
            .query_row(
                "SELECT model, dimensions FROM ocr_embedding_model WHERE space = ?1",
                [space.as_str()],
                |row| {
                    Ok(StoredTextEmbeddingModel {
                        model: row.get(0)?,
                        dimensions: row.get::<_, i64>(1)? as usize,
                    })
                },
            )
            .optional()
            .context("reading the OCR embedding model")
    }

    /// Declare the model a space's future vectors are produced by, creating its vector table on
    /// first use.
    ///
    /// The `ocr_embeddings_*` tables are the only ones this schema creates outside
    /// `db_create.sql`: `vec0` bakes the dimension count into its DDL, and that count is a property
    /// of the caller's model rather than of the schema version. Readers therefore treat an absent
    /// table as "nothing embedded into this space yet" rather than as a damaged database.
    ///
    /// Switching a space's model invalidates its stored vectors, so it requires `reset`.
    pub fn set_text_embedding_model(
        &mut self,
        space: TextEmbeddingSpace,
        model: &str,
        dimensions: usize,
        reset: bool,
    ) -> Result<()> {
        if model.trim().is_empty() {
            bail!("embedding model name must not be empty");
        }
        if dimensions == 0 || dimensions > MAX_TEXT_EMBEDDING_DIMENSIONS {
            bail!("embedding dimensions must be between 1 and {MAX_TEXT_EMBEDDING_DIMENSIONS}");
        }
        let current = self.text_embedding_model(space)?;
        let changing = current
            .as_ref()
            .is_some_and(|existing| existing.model != model || existing.dimensions != dimensions);
        if changing && !reset {
            let existing = current.as_ref().expect("a change implies a current model");
            bail!(
                "{space} embeddings already use model {} with {} dimensions; re-embedding under {model} requires an explicit reset",
                existing.model,
                existing.dimensions
            );
        }

        let dimension_column =
            i64::try_from(dimensions).context("embedding dimensions exceed SQLite's range")?;
        let vectors = space.table();
        let tx = self.conn.transaction()?;
        if changing {
            tx.execute_batch(&format!("DROP TABLE IF EXISTS {vectors};"))
                .context("clearing embeddings for a new model")?;
            tx.execute(
                "DELETE FROM ocr_embedding_state WHERE space = ?1",
                [space.as_str()],
            )?;
        }
        tx.execute(
            "INSERT INTO ocr_embedding_model (space, model, dimensions) VALUES (?1, ?2, ?3) \
             ON CONFLICT(space) DO UPDATE SET model = excluded.model, dimensions = excluded.dimensions",
            (space.as_str(), model, dimension_column),
        )?;
        tx.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {vectors} USING vec0(\
                 asset_id INTEGER PRIMARY KEY, \
                 embedding FLOAT[{dimensions}] distance_metric=cosine\
             );"
        ))
        .context("creating the OCR embedding vector table")?;
        tx.commit()?;
        Ok(())
    }

    /// Store one vector per asset in `space`, replacing any previous vector for the same asset.
    ///
    /// An asset with no OCR row is skipped rather than rejected: the catalog and the OCR store are
    /// pruned independently, so a backlog a client is halfway through embedding can legitimately
    /// name an asset that has just been swept. The returned count is the vectors actually stored.
    #[instrument(
        name = "save_text_embeddings",
        level = "debug",
        skip_all,
        fields(space = %space, rows = items.len(), stored = field::Empty)
    )]
    pub fn save_text_embeddings(
        &mut self,
        space: TextEmbeddingSpace,
        model: &str,
        items: Vec<TextEmbedding>,
    ) -> Result<usize> {
        let Some(declared) = self.text_embedding_model(space)? else {
            bail!("no embedding model has been declared for the {space} space");
        };
        if declared.model != model {
            bail!(
                "{space} embeddings are stored under model {}, not {model}",
                declared.model
            );
        }
        for item in &items {
            if item.vector.len() != declared.dimensions {
                bail!(
                    "asset {} has {} dimensions but model {} produces {}",
                    item.asset_id,
                    item.vector.len(),
                    declared.model,
                    declared.dimensions
                );
            }
            if item.vector.iter().any(|value| !value.is_finite()) {
                bail!(
                    "asset {} has a vector containing a non-finite value",
                    item.asset_id
                );
            }
        }

        let vectors = space.table();
        let tx = self.conn.transaction()?;
        let mut stored = 0;
        {
            // The fingerprint is copied from the OCR row, which is what makes a later re-OCR
            // silently retire the vector: search joins the two back together on it.
            let mut state = tx.prepare_cached(
                "INSERT INTO ocr_embedding_state (space, asset_id, source_modified_ns, source_size) \
                 SELECT ?1, asset_id, source_modified_ns, source_size FROM ocr_results WHERE asset_id = ?2 \
                 ON CONFLICT(space, asset_id) DO UPDATE SET source_modified_ns = excluded.source_modified_ns, source_size = excluded.source_size",
            )?;
            // vec0 implements plain INSERT and DELETE, not upsert, so replacement is two steps.
            let mut clear =
                tx.prepare_cached(&format!("DELETE FROM {vectors} WHERE asset_id = ?1"))?;
            let mut insert = tx.prepare_cached(&format!(
                "INSERT INTO {vectors} (asset_id, embedding) VALUES (?1, ?2)"
            ))?;
            for item in items {
                if state.execute((space.as_str(), item.asset_id))? == 0 {
                    continue;
                }
                clear.execute([item.asset_id])?;
                insert.execute((item.asset_id, vector_to_blob(&item.vector)))?;
                stored += 1;
            }
        }
        tx.commit()?;
        Span::current().record("stored", stored);
        Ok(stored)
    }

    /// Drop every stored vector in `space`, keeping its declared model. This is what a forced
    /// re-embed of an unchanged model does: the model is still the right one, but its output is
    /// being redone.
    pub fn clear_text_embeddings(&mut self, space: TextEmbeddingSpace) -> Result<usize> {
        if !self.has_vector_table(space)? {
            return Ok(0);
        }
        let tx = self.conn.transaction()?;
        let cleared = tx.execute(&format!("DELETE FROM {}", space.table()), [])?;
        tx.execute(
            "DELETE FROM ocr_embedding_state WHERE space = ?1",
            [space.as_str()],
        )?;
        tx.commit()?;
        Ok(cleared)
    }

    /// Drop `asset_id`'s vector in every space, whether or not its OCR row still exists.
    pub fn delete_text_embedding(&mut self, asset_id: i64) -> Result<usize> {
        let mut deleted = 0;
        for space in TextEmbeddingSpace::ALL {
            if !self.has_vector_table(space)? {
                continue;
            }
            deleted += self
                .conn
                .execute(
                    &format!("DELETE FROM {} WHERE asset_id = ?1", space.table()),
                    [asset_id],
                )
                .context("deleting an asset embedding")?;
        }
        self.conn
            .execute(
                "DELETE FROM ocr_embedding_state WHERE asset_id = ?1",
                [asset_id],
            )
            .context("deleting an asset embedding state")?;
        Ok(deleted)
    }

    /// Delete vectors whose OCR row is gone. `ocr_embedding_state` cascades on its own; `vec0` is
    /// a virtual table and cannot be a foreign key target, so its rows are swept explicitly.
    pub fn sweep_text_embeddings(&mut self) -> Result<usize> {
        let mut swept = 0;
        for space in TextEmbeddingSpace::ALL {
            if !self.has_vector_table(space)? {
                continue;
            }
            let vectors = space.table();
            swept += self
                .conn
                .execute(
                    &format!(
                        "DELETE FROM {vectors} WHERE asset_id IN (\
                             SELECT {vectors}.asset_id FROM {vectors} \
                             LEFT JOIN ocr_embedding_state \
                                    ON ocr_embedding_state.asset_id = {vectors}.asset_id \
                                   AND ocr_embedding_state.space = ?1 \
                              WHERE ocr_embedding_state.asset_id IS NULL\
                         )"
                    ),
                    [space.as_str()],
                )
                .context("sweeping orphaned OCR embeddings")?;
        }
        Ok(swept)
    }

    /// How much of the OCR text matching `filters` currently has a usable vector in `space`.
    ///
    /// Takes the same [`SearchFilters`] as search does, so "coverage of what I am about to search"
    /// is answerable exactly, including a time range.
    pub fn text_embedding_coverage(
        &self,
        space: TextEmbeddingSpace,
        filters: &SearchFilters,
    ) -> Result<TextEmbeddingCoverage> {
        let bound = filters.bind(FilterScope::OCR_ROWS)?;
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT count(*), count(ocr_embedding_state.asset_id), max(ocr_results.source_modified_ns) \
               FROM ocr_results \
               LEFT JOIN ocr_embedding_state \
                      ON ocr_embedding_state.asset_id = ocr_results.asset_id \
                     AND ocr_embedding_state.space = :space \
                     AND ocr_embedding_state.source_modified_ns = ocr_results.source_modified_ns \
                     AND ocr_embedding_state.source_size = ocr_results.source_size \
              WHERE ocr_results.mark_delete = FALSE{filters}",
            filters = bound.sql,
        ))?;
        let params = bound.extend([(":space", Value::Text(space.as_str().to_owned()))]);
        let (indexed, embedded, last_indexed_ns): (i64, i64, Option<i64>) = statement
            .query_row(bind_named(&params).as_slice(), |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .context("counting OCR embedding coverage")?;
        Ok(TextEmbeddingCoverage {
            indexed: usize::try_from(indexed).context("OCR row count exceeds usize")?,
            embedded: usize::try_from(embedded).context("embedded row count exceeds usize")?,
            last_indexed_ns,
        })
    }

    /// OCR rows matching `filters` with no current vector in `space`, newest first, with the text
    /// to embed. This is the backlog a client works through after indexing.
    pub fn pending_text_embeddings(
        &self,
        space: TextEmbeddingSpace,
        filters: &SearchFilters,
        limit: usize,
        max_content_bytes: usize,
    ) -> Result<Vec<PendingTextEmbedding>> {
        let bound = filters.bind(FilterScope::OCR_ROWS)?;
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT ocr_results.asset_id, substr(ocr_results.content, 1, :max_content_bytes) \
               FROM ocr_results \
               LEFT JOIN ocr_embedding_state \
                      ON ocr_embedding_state.asset_id = ocr_results.asset_id \
                     AND ocr_embedding_state.space = :space \
                     AND ocr_embedding_state.source_modified_ns = ocr_results.source_modified_ns \
                     AND ocr_embedding_state.source_size = ocr_results.source_size \
              WHERE ocr_embedding_state.asset_id IS NULL \
                AND ocr_results.mark_delete = FALSE \
                AND trim(ocr_results.content) <> ''{filters} \
              ORDER BY ocr_results.source_modified_ns DESC, ocr_results.asset_id DESC \
              LIMIT :limit",
            filters = bound.sql,
        ))?;
        let params = bound.extend([
            (":space", Value::Text(space.as_str().to_owned())),
            (
                ":max_content_bytes",
                Value::Integer(
                    i64::try_from(max_content_bytes)
                        .context("embedding content cap exceeds SQLite's range")?,
                ),
            ),
            (
                ":limit",
                Value::Integer(
                    i64::try_from(limit)
                        .context("embedding backlog limit exceeds SQLite's range")?,
                ),
            ),
        ]);
        let rows = statement
            .query_map(bind_named(&params).as_slice(), |row| {
                Ok(PendingTextEmbedding {
                    asset_id: row.get(0)?,
                    content: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("listing the OCR embedding backlog")?;
        Ok(rows)
    }

    /// Nearest neighbours of `vector` by cosine distance, restricted to `path`.
    ///
    /// The scan is exact rather than a `vec0` `MATCH ... k = ?` lookup because `k` is applied
    /// before any join: under a root filter a globally limited top-k would silently return fewer than
    /// `limit` rows whenever the nearest vectors live outside the searched directory. `vec0` is
    /// brute force either way, so exactness costs a constant factor rather than an algorithm.
    #[instrument(
        name = "search_text_vectors",
        level = "debug",
        skip_all,
        fields(space = %space, dimensions = vector.len(), limit, total = field::Empty)
    )]
    pub fn search_text_vectors(
        &mut self,
        vector: &[f32],
        space: TextEmbeddingSpace,
        filters: &SearchFilters,
        limit: usize,
        options: TextVectorSearchOptions,
    ) -> Result<(usize, Vec<TextVectorSearchResult>)> {
        let Some(declared) = self.text_embedding_model(space)? else {
            // Nothing has ever been embedded in this space. An empty result is the honest answer,
            // exactly as it is for a root that has never been indexed.
            return Ok((0, Vec::new()));
        };
        if vector.len() != declared.dimensions {
            bail!(
                "query vector has {} dimensions but model {} produces {}",
                vector.len(),
                declared.model,
                declared.dimensions
            );
        }
        if !self.has_vector_table(space)? {
            return Ok((0, Vec::new()));
        }

        let bound = filters.bind(FilterScope::OCR_ROWS)?;
        let vectors = space.table();
        let scored = format!(
            r#"
            WITH scored AS MATERIALIZED (
                SELECT ocr_results.asset_id AS asset_id,
                       vec_distance_cosine({vectors}.embedding, :query) AS distance,
                       ocr_results.source_modified_ns AS source_modified_ns
                  FROM {vectors}
                  INNER JOIN ocr_results
                          ON ocr_results.asset_id = {vectors}.asset_id
                  INNER JOIN ocr_embedding_state
                          ON ocr_embedding_state.asset_id = {vectors}.asset_id
                         AND ocr_embedding_state.space = :space
                         AND ocr_embedding_state.source_modified_ns = ocr_results.source_modified_ns
                         AND ocr_embedding_state.source_size = ocr_results.source_size
                 WHERE ocr_results.mark_delete = FALSE{filters}
            )"#,
            filters = bound.sql,
        );

        let shared = bound.extend([
            (":query", Value::Blob(vector_to_blob(vector))),
            (":space", Value::Text(space.as_str().to_owned())),
            (
                ":max_distance",
                Value::Real(options.max_distance.unwrap_or(UNBOUNDED_DISTANCE)),
            ),
        ]);

        // Score once for both the total and the ranking. Fetch OCR contents only after
        // limiting hits, so the materialized scan does not copy the library's entire text.
        let mut statement = self.conn.prepare_cached(&format!(
            r#"{scored}
            , hits AS (
                SELECT asset_id, distance, source_modified_ns
                  FROM scored
                 WHERE distance <= :max_distance
                 ORDER BY distance ASC, source_modified_ns DESC, asset_id ASC
                 LIMIT :limit
            )
            SELECT totals.total, hits.asset_id, hits.distance, ocr_results.source_path,
                   hits.source_modified_ns, ocr_results.width, ocr_results.height,
                   substr(ocr_results.content, 1, :snippet_bytes)
              FROM (SELECT count(*) AS total FROM scored WHERE distance <= :max_distance) totals
              LEFT JOIN hits ON 1
              LEFT JOIN ocr_results ON ocr_results.asset_id = hits.asset_id
             ORDER BY hits.distance ASC, hits.source_modified_ns DESC, hits.asset_id ASC"#
        ))?;
        let params = extend_params(
            shared,
            [
                (
                    ":snippet_bytes",
                    Value::Integer(
                        i64::try_from(options.snippet_bytes)
                            .context("embedding snippet cap exceeds SQLite's range")?,
                    ),
                ),
                (
                    ":limit",
                    Value::Integer(
                        i64::try_from(limit)
                            .context("search limit exceeds SQLite's integer range")?,
                    ),
                ),
            ],
        );
        let (total, results) = (|| -> rusqlite::Result<_> {
            let mut rows = statement.query(bind_named(&params).as_slice())?;
            let mut total = 0_i64;
            let mut results = Vec::new();
            while let Some(row) = rows.next()? {
                total = row.get(0)?;
                // The total survives even when no hits are returned (including limit = 0).
                if let Some(asset_id) = row.get::<_, Option<i64>>(1)? {
                    results.push(TextVectorSearchResult {
                        asset_id,
                        distance: row.get(2)?,
                        path: row.get(3)?,
                        modified_ns: row.get(4)?,
                        width: row.get(5)?,
                        height: row.get(6)?,
                        contents: row.get(7)?,
                    });
                }
            }
            Ok((total, results))
        })()
        .map_err(execution_error)
        .context("querying the OCR vector index")?;

        Span::current().record("total", total);
        Ok((
            usize::try_from(total).context("vector search result count exceeds usize")?,
            results,
        ))
    }

    fn has_vector_table(&self, space: TextEmbeddingSpace) -> Result<bool> {
        self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name = ?1)",
                [space.table()],
                |row| row.get(0),
            )
            .context("checking for the OCR embedding vector table")
    }
}

fn search_query(queries: &[&str], kind: SearchType) -> String {
    if kind == SearchType::Simple {
        format!(r#""{}""#, queries.join(" ").replace('*', "\\*"))
    } else {
        queries.join(" ")
    }
}

const GLOB_MATCHING_TERMS: &str = "WITH matching_terms AS (
    SELECT term FROM ocr_results_words_vocab WHERE rust_word_glob(:query, term)
)";

fn glob_search_source(filters: &str) -> String {
    format!(
        "FROM matching_documents \
        INNER JOIN ocr_results ON matching_documents.doc = ocr_results.asset_id \
        WHERE 1 = 1{filters}"
    )
}

/// Keep result and count queries on identical joins and predicates.
fn ocr_search_source(kind: SearchType, filters: &str) -> String {
    format!(
        "FROM ocr_results_fts \
        INNER JOIN ocr_results ON ocr_results_fts.rowid = ocr_results.asset_id \
        WHERE ocr_results_fts.content {} :query{}",
        search_operator(kind),
        filters
    )
}

fn search_operator(kind: SearchType) -> &'static str {
    match kind {
        SearchType::Simple | SearchType::Match => "MATCH",
        SearchType::Glob => "GLOB",
        #[cfg(feature = "regex")]
        SearchType::Regex => "REGEXP",
    }
}

/// Cosine distance is bounded by 2, so this stands in for "no ceiling" without branching the SQL.
/// Shared by both vector-search stores: the OCR space and the image index.
pub(crate) const UNBOUNDED_DISTANCE: f64 = 1_000.0;
/// Wide enough for every current text embedding model, narrow enough that a bad `dimensions`
/// cannot ask SQLite to build an absurd `vec0` table.
pub const MAX_TEXT_EMBEDDING_DIMENSIONS: usize = 8192;

/// The single embedding model a database's vectors belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredTextEmbeddingModel {
    pub model: String,
    pub dimensions: usize,
}

/// One asset's vector, ready to store.
#[derive(Debug, Clone)]
pub struct TextEmbedding {
    pub asset_id: i64,
    pub vector: Vec<f32>,
}

/// An OCR row still waiting for a vector, with the text to embed.
#[derive(Debug, Clone)]
pub struct PendingTextEmbedding {
    pub asset_id: i64,
    pub content: String,
}

/// How much of a root's OCR text is currently searchable by vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEmbeddingCoverage {
    /// OCR rows under the root.
    pub indexed: usize,
    /// Of those, the ones whose vector matches their current OCR fingerprint.
    pub embedded: usize,
    /// The newest `source_modified_ns` among indexed rows, Unix nanoseconds. There is no
    /// wall-clock record of when OCR actually ran, so this is a proxy: "the most recently changed
    /// file this root has indexed text for." `None` when nothing is indexed.
    pub last_indexed_ns: Option<i64>,
}

impl TextEmbeddingCoverage {
    /// Rows that still need embedding before vector search covers the whole root.
    pub fn pending(self) -> usize {
        self.indexed.saturating_sub(self.embedded)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextVectorSearchOptions {
    /// Drop neighbours further than this cosine distance. `None` keeps every neighbour.
    pub max_distance: Option<f64>,
    /// Leading bytes of OCR text returned with each hit. Vector search has no matched term to
    /// centre a snippet on, so the excerpt is simply the start of the text.
    pub snippet_bytes: usize,
}

impl Default for TextVectorSearchOptions {
    fn default() -> Self {
        Self {
            max_distance: None,
            snippet_bytes: 240,
        }
    }
}

#[derive(Debug)]
pub struct TextVectorSearchResult {
    pub asset_id: i64,
    /// Cosine distance: 0 is identical, 1 is orthogonal, 2 is opposite.
    pub distance: f64,
    pub path: String,
    pub modified_ns: i64,
    pub width: u32,
    pub height: u32,
    pub contents: String,
}

/// `sqlite-vec` reads a blob as a tightly packed little-endian `f32` array regardless of the
/// host's byte order, so the conversion is explicit rather than a transmute of the slice.
pub(crate) fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

/// Scalar vector scans repeatedly open blobs in vec0's chunk tables. Mapping the main database
/// avoids cycling these reads through SQLite's small default page cache. This is a per-connection
/// address-space budget, not an eager allocation; SQLite can use less or disable mapping on
/// platforms that do not support it. WAL and normal read-transaction semantics still apply.
pub(crate) fn configure_vector_reads(conn: &Connection) -> Result<()> {
    conn.pragma_update(Some("main"), "mmap_size", 256 * 1024 * 1024)?;
    Ok(())
}

/// Set the page size before a new vector store enters WAL mode. Existing databases keep their
/// original page size; SQLite cannot change it while they are in WAL mode anyway.
pub(crate) fn configure_vector_writer(conn: &Connection) -> Result<()> {
    let pages: i64 = conn.pragma_query_value(None, "page_count", |row| row.get(0))?;
    if pages == 0 {
        conn.pragma_update(None, "page_size", 65_536)?;
    }
    configure_writer(conn)?;
    configure_vector_reads(conn)
}

/// `sqlite-vec` ships as a statically linked SQLite extension. `sqlite3_auto_extension` installs
/// it into every connection this process opens afterwards, so it has to run before the first open
/// and exactly once.
pub(crate) fn register_vector_extension() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the extension entry point sqlite-vec exports for exactly
        // this call, and the transmute only restores the C signature the FFI declaration erased.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::os::raw::c_int,
            >(
                sqlite3_vec_init as *const ()
            )));
        }
    });
}

#[derive(Debug)]
pub struct OcrResult {
    pub asset_id: i64,
    pub path: PathBuf,
    pub fingerprint: SourceFingerprint,
    /// Copied from the asset catalog so search can filter on the capture timeline without
    /// attaching a second database.
    pub exif_taken_ns: Option<i64>,
    pub width: u32,
    pub height: u32,
    pub contents: String,
}

#[derive(Debug)]
pub struct SearchResult {
    pub asset_id: i64,
    pub path: String,
    pub modified_ns: i64,
    pub width: u32,
    pub height: u32,
    pub contents: String,
    /// This mode's own relevance score, in whatever units its retrieval method produces: FTS5
    /// `bm25()` for [`SearchType::Simple`] and [`SearchType::Match`] (lower is a better match), or
    /// the count of matching word-term occurrences for [`SearchType::Glob`] (higher is better —
    /// term frequency, not distinct term count). The two are not on a comparable scale — meaningful
    /// only as an ordering within one mode, the same way cosine `distance` is meaningful only within
    /// vector search.
    pub score: f64,
}

#[cfg(feature = "regex")]
fn register_regex(db: &Connection) -> Result<()> {
    use regex::Regex;
    use rusqlite::functions::FunctionFlags;
    use std::sync::Arc;
    db.create_scalar_function(
        "regexp",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        move |ctx| {
            assert_eq!(ctx.len(), 2, "called with unexpected number of arguments");
            let regexp: Arc<Regex> = ctx
                .get_or_create_aux(0, |value| -> Result<_> { Ok(Regex::new(value.as_str()?)?) })?;
            let text = ctx
                .get_raw(1)
                .as_str()
                .map_err(|error| rusqlite::Error::UserFunctionError(error.into()))?;
            Ok(regexp.is_match(text))
        },
    )?;
    Ok(())
}

fn register_word_glob(db: &Connection) -> Result<()> {
    use glob::Pattern;
    use rusqlite::functions::FunctionFlags;
    use std::sync::Arc;
    db.create_scalar_function(
        "rust_word_glob",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        move |ctx| {
            assert_eq!(ctx.len(), 2, "called with unexpected number of arguments");
            // unicode61 normalizes indexed terms to lower case. Normalize the user pattern once
            // when SQLite binds it, then reuse the compiled glob across the vocabulary scan.
            let pattern: Arc<Pattern> = ctx.get_or_create_aux(0, |value| -> Result<_> {
                Ok(Pattern::new(&value.as_str()?.to_lowercase())?)
            })?;
            let text = ctx
                .get_raw(1)
                .as_str()
                .map_err(|error| rusqlite::Error::UserFunctionError(error.into()))?;
            Ok(pattern.matches(text))
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn vector_page_size_applies_only_to_new_databases() -> Result<()> {
        let temp = TempDir::new()?;
        let fresh_path = temp.path().join("fresh.db");
        let fresh = Connection::open(&fresh_path)?;
        configure_vector_writer(&fresh)?;
        fresh.execute_batch("CREATE TABLE marker(id INTEGER PRIMARY KEY)")?;
        assert_eq!(
            fresh.pragma_query_value(None, "page_size", |row| row.get::<_, i64>(0))?,
            65_536
        );
        assert_eq!(
            fresh.pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))?,
            "wal"
        );
        drop(fresh);

        let existing_path = temp.path().join("existing.db");
        let existing = Connection::open(&existing_path)?;
        existing.pragma_update(None, "page_size", 4_096)?;
        existing.execute_batch("CREATE TABLE marker(id INTEGER PRIMARY KEY)")?;
        assert_eq!(
            existing.pragma_query_value(None, "page_size", |row| row.get::<_, i64>(0))?,
            4_096
        );
        configure_vector_writer(&existing)?;
        assert_eq!(
            existing.pragma_query_value(None, "page_size", |row| row.get::<_, i64>(0))?,
            4_096
        );
        assert_eq!(
            existing.pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))?,
            "wal"
        );
        Ok(())
    }

    const SPACE: TextEmbeddingSpace = TextEmbeddingSpace::OcrText;

    #[test]
    fn fresh_and_migrated_ocr_indexes_ignore_bookkeeping_updates() -> Result<()> {
        for migrate in [false, true] {
            let temp = TempDir::new()?;
            let path = PathBuf::try_from(temp.path().join("ocr.db"))?;
            if migrate {
                let conn = Connection::open(&path)?;
                let version_eight = include_str!("db_create.sql")
                    .replace("\r\n", "\n")
                    .replace(
                        "AFTER UPDATE OF asset_id, content ON ocr_results\nWHEN old.asset_id IS NOT new.asset_id OR old.content IS NOT new.content BEGIN",
                        "AFTER UPDATE ON ocr_results BEGIN",
                    )
                    .replace("PRAGMA user_version = 10;", "PRAGMA user_version = 8;");
                conn.execute_batch(&version_eight)?;
                conn.execute("INSERT INTO ocr_results(asset_id, source_path, source_modified_ns, source_size, width, height, content) VALUES (1, 'retained.png', 1, 1, 1, 1, 'original')", [])?;
                let before = conn.total_changes();
                conn.execute(
                    "UPDATE ocr_results SET mark_delete = FALSE WHERE asset_id = 1",
                    [],
                )?;
                assert!(
                    conn.total_changes() - before > 1,
                    "fixture must reproduce the old trigger's FTS churn"
                );
            }
            let mut db = DB::new(&path)?;
            if migrate {
                let content: String = db.conn.query_row(
                    "SELECT content FROM ocr_results WHERE asset_id = 1",
                    [],
                    |row| row.get(0),
                )?;
                assert_eq!(content, "original");
            } else {
                db.save_results(vec![result(&temp, 1, "original")?])?;
            }
            check_schema_read_only(&db.conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
            for sql in [
                "UPDATE ocr_results SET mark_delete = TRUE WHERE asset_id = 1",
                "UPDATE ocr_results SET mark_delete = FALSE WHERE asset_id = 1",
                "UPDATE ocr_results SET source_modified_ns = 2 WHERE asset_id = 1",
                "UPDATE ocr_results SET content = content WHERE asset_id = 1",
            ] {
                let before = db.conn.total_changes();
                db.conn.execute(sql, [])?;
                assert_eq!(
                    db.conn.total_changes() - before,
                    1,
                    "FTS churn for {sql}, migrated={migrate}"
                );
            }
            db.conn.execute(
                "UPDATE ocr_results SET content = 'replacement', asset_id = 2 WHERE asset_id = 1",
                [],
            )?;
            for table in ["ocr_results_fts", "ocr_results_words_fts"] {
                let rows: i64 = db.conn.query_row(
                    &format!("SELECT count(*) FROM {table} WHERE {table} MATCH 'original'"),
                    [],
                    |row| row.get(0),
                )?;
                assert_eq!(rows, 0);
                let id: i64 = db.conn.query_row(
                    &format!("SELECT rowid FROM {table} WHERE {table} MATCH 'replacement'"),
                    [],
                    |row| row.get(0),
                )?;
                assert_eq!(id, 2);
            }
            db.conn
                .execute("DELETE FROM ocr_results WHERE asset_id = 2", [])?;
            for table in ["ocr_results_fts", "ocr_results_words_fts"] {
                let rows: i64 = db.conn.query_row(
                    &format!("SELECT count(*) FROM {table} WHERE {table} MATCH 'replacement'"),
                    [],
                    |row| row.get(0),
                )?;
                assert_eq!(rows, 0);
            }
            drop(db);
            // Reopening an already upgraded database must be a no-op.
            DB::new(&path)?;
        }
        Ok(())
    }

    #[test]
    fn read_snapshot_releases_transaction_after_errors_and_panics() -> Result<()> {
        let temp = TempDir::new()?;
        let mut db = test_db(&temp)?;
        let failure = db.read_snapshot::<()>(|_| anyhow::bail!("query failed"));
        assert!(failure.is_err());
        assert!(db.conn.is_autocommit());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = db.read_snapshot::<()>(|_| panic!("query panicked"));
        }));
        assert!(panic.is_err());
        assert!(db.conn.is_autocommit());
        db.read_snapshot(|db| {
            assert!(!db.conn.is_autocommit());
            Ok(())
        })?;
        assert!(db.conn.is_autocommit());
        Ok(())
    }

    fn test_db(temp: &TempDir) -> Result<DB> {
        DB::new(&PathBuf::try_from(temp.path().join("ocr.db"))?)
    }

    fn result(temp: &TempDir, asset_id: i64, contents: &str) -> Result<OcrResult> {
        Ok(OcrResult {
            asset_id,
            path: PathBuf::try_from(temp.path().join(format!("{asset_id}.png")))?,
            fingerprint: SourceFingerprint {
                modified_ns: 987_654_321,
                size: 1234,
            },
            exif_taken_ns: None,
            width: 640,
            height: 480,
            contents: contents.to_owned(),
        })
    }

    fn searchable_db(temp: &TempDir) -> Result<(DB, PathBuf)> {
        let mut db = test_db(temp)?;
        db.save_results(vec![result(temp, 1, "hello world")?])?;
        Ok((db, PathBuf::try_from(temp.path().to_path_buf())?))
    }

    #[test]
    fn malformed_queries_are_reported_as_query_syntax_errors() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = searchable_db(&temp)?;
        // SQLite words these three differently; none of them is the caller's or the index's fault.
        let cases = [
            (
                SearchType::Simple,
                r#"ocr:"unterminated"#,
                "unterminated string",
            ),
            (SearchType::Match, "AND", r#"fts5: syntax error near "AND""#),
            (SearchType::Match, "col:foo", "no such column: col"),
        ];
        for (kind, query, expected) in cases {
            let error = db
                .search_with_count(vec![query], &SearchFilters::under(&root), 10, kind)
                .expect_err(query);
            assert_eq!(query_syntax_message(&error), Some(expected), "{query}");
        }
        Ok(())
    }

    #[cfg(feature = "regex")]
    #[test]
    fn invalid_regex_patterns_are_reported_on_one_line() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = searchable_db(&temp)?;
        let error = db
            .search_with_count(
                vec!["(unclosed"],
                &SearchFilters::under(&root),
                10,
                SearchType::Regex,
            )
            .expect_err("an unclosed group is not a valid pattern");
        let message = query_syntax_message(&error).expect("regex failures are query failures");
        assert!(!message.contains('\n'), "{message}");
        assert!(message.contains("unclosed group"), "{message}");
        Ok(())
    }

    #[test]
    fn well_formed_queries_are_not_query_syntax_errors() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = searchable_db(&temp)?;
        let (total, results) = db.search_with_count(
            vec!["hello"],
            &SearchFilters::under(&root),
            10,
            SearchType::Simple,
        )?;
        assert_eq!(total, 1);
        assert_eq!(results.len(), 1);

        // A schema failure is a server problem, so it must not be classified as a bad query.
        let missing = PathBuf::try_from(temp.path().join("absent.db"))?;
        let error = match DB::new_read_only(&missing) {
            Ok(_) => panic!("an absent database cannot be opened read-only"),
            Err(error) => error,
        };
        assert_eq!(query_syntax_message(&error), None);
        Ok(())
    }

    #[test]
    fn glob_search_matches_case_insensitive_whole_words() -> Result<()> {
        let temp = TempDir::new()?;
        let mut db = test_db(&temp)?;
        db.save_results(vec![
            result(&temp, 1, "Dreams arrive before dawn.")?,
            result(&temp, 2, "Andrew only.")?,
            result(&temp, 3, "A CATalogue is on the shelf.")?,
        ])?;
        let filters = SearchFilters::under(Path::from_path(temp.path()).unwrap());

        let mut found = |pattern: &str| -> Result<(usize, Vec<i64>)> {
            let (total, results) =
                db.search_with_count(vec![pattern], &filters, 10, SearchType::Glob)?;
            Ok((
                total,
                results.into_iter().map(|result| result.asset_id).collect(),
            ))
        };

        // A plain glob is still an exact word search.
        assert_eq!(found("dreams")?, (1, vec![1]));
        // Prefix matching is word-boundary aware, so it must not find "Andrew".
        assert_eq!(found("dre*")?, (1, vec![1]));
        assert_eq!(found("DRE*")?, (1, vec![1]));
        assert_eq!(found("dr?ams")?, (1, vec![1]));
        assert_eq!(found("*cat*")?, (1, vec![3]));
        Ok(())
    }

    #[test]
    fn search_results_carry_a_relevance_score() -> Result<()> {
        let temp = TempDir::new()?;
        let mut db = test_db(&temp)?;
        db.save_results(vec![
            result(&temp, 1, "receipt receipt receipt")?,
            result(&temp, 2, "receipt")?,
        ])?;
        let filters = SearchFilters::under(Path::from_path(temp.path()).unwrap());

        // FTS5 bm25() is ascending (more negative is a better match), and `search` already orders
        // by it — the repeated term should score at least as well as the single occurrence.
        let matches = db.search(vec!["receipt"], &filters, 10, SearchType::Match)?;
        assert_eq!(
            matches.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(matches[0].score <= matches[1].score, "{matches:?}");

        // Glob's score is the count of matching term occurrences, so the repeated term scores
        // higher, the opposite polarity from bm25 — `search_glob` already orders by it descending.
        let globs = db.search(vec!["receipt"], &filters, 10, SearchType::Glob)?;
        assert_eq!(
            globs.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!((globs[0].score, globs[1].score), (3.0, 1.0), "{globs:?}");
        Ok(())
    }

    #[test]
    fn current_fingerprint_is_indexed() -> Result<()> {
        let temp = TempDir::new()?;
        let mut db = test_db(&temp)?;
        let first = result(&temp, 71, "first")?;
        let second = result(&temp, 72, "second")?;
        let first_fingerprint = first.fingerprint;
        let mut stale_fingerprint = second.fingerprint;
        stale_fingerprint.modified_ns += 1;
        db.save_results(vec![first, second])?;

        assert!(db.is_indexed(71, first_fingerprint)?);
        assert!(!db.is_indexed(73, first_fingerprint)?);
        assert_eq!(
            db.current_asset_ids(&[
                (71, first_fingerprint),
                (72, stale_fingerprint),
                (73, first_fingerprint),
            ])?,
            HashSet::from([71])
        );
        assert_eq!(
            db.index_states(&[
                (71, first_fingerprint),
                (72, stale_fingerprint),
                (73, first_fingerprint),
            ])?,
            HashMap::from([(71, OcrIndexState::Indexed), (72, OcrIndexState::Stale),])
        );
        Ok(())
    }

    #[test]
    fn batch_deletion_uses_catalog_ids() -> Result<()> {
        let temp = TempDir::new()?;
        let mut db = test_db(&temp)?;
        let first = result(&temp, 11, "delete one")?;
        let second = result(&temp, 12, "delete two")?;
        let third = result(&temp, 13, "keep")?;
        let fingerprints = [first.fingerprint, second.fingerprint, third.fingerprint];
        db.save_results(vec![first, second, third])?;
        assert_eq!(db.delete_assets(&[11, 12])?, 2);
        assert!(!db.is_indexed(11, fingerprints[0])?);
        assert!(!db.is_indexed(12, fingerprints[1])?);
        assert!(db.is_indexed(13, fingerprints[2])?);
        Ok(())
    }

    #[test]
    fn deletion_does_not_match_a_sibling_directory_with_the_same_prefix() -> Result<()> {
        let temp = TempDir::new()?;
        let mut db = test_db(&temp)?;
        let directory = PathBuf::try_from(temp.path().join("gallery"))?;
        let sibling = PathBuf::try_from(temp.path().join("gallery-old"))?;
        let mut inside = result(&temp, 11, "inside")?;
        inside.path = directory.join("inside.png");
        let mut outside = result(&temp, 12, "outside")?;
        outside.path = sibling.join("outside.png");
        db.save_results(vec![inside, outside])?;

        std::fs::create_dir_all(&directory)?;
        db.mark_for_deletion(&directory)?;

        assert_eq!(db.sweep_deletions()?, 1);
        assert!(db.is_indexed(
            12,
            SourceFingerprint {
                modified_ns: 987_654_321,
                size: 1234,
            }
        )?);
        Ok(())
    }

    #[test]
    fn deleting_assets_removes_text_vectors_and_embedding_state() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, _) = embedded_db(&temp)?;
        db.delete_assets(&[1])?;
        let vectors: i64 = db.conn.query_row(
            &format!("SELECT count(*) FROM {} WHERE asset_id = 1", SPACE.table()),
            [],
            |row| row.get(0),
        )?;
        let state: i64 = db.conn.query_row(
            "SELECT count(*) FROM ocr_embedding_state WHERE asset_id = 1",
            [],
            |row| row.get(0),
        )?;
        assert_eq!((vectors, state), (0, 0));
        Ok(())
    }

    /// A three-dimensional model keeps the expected neighbour order obvious by inspection.
    fn embedded_db(temp: &TempDir) -> Result<(DB, PathBuf)> {
        let mut db = test_db(temp)?;
        db.save_results(vec![
            result(temp, 1, "due north")?,
            result(temp, 2, "due east")?,
            result(temp, 3, "due south")?,
        ])?;
        db.set_text_embedding_model(SPACE, "test-model", 3, false)?;
        db.save_text_embeddings(
            SPACE,
            "test-model",
            vec![
                TextEmbedding {
                    asset_id: 1,
                    vector: vec![1.0, 0.0, 0.0],
                },
                TextEmbedding {
                    asset_id: 2,
                    vector: vec![0.0, 1.0, 0.0],
                },
                TextEmbedding {
                    asset_id: 3,
                    vector: vec![-1.0, 0.0, 0.0],
                },
            ],
        )?;
        Ok((db, PathBuf::try_from(temp.path().to_path_buf())?))
    }

    #[test]
    fn inspector_embedding_status_tracks_indexing_lifecycle_and_search() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;
        let original = result(&temp, 1, "due north")?;
        let fingerprint = original.fingerprint;
        assert_eq!(
            db.asset_text_embedding_status(1, fingerprint)?,
            Some((true, true))
        );
        assert_eq!(
            db.current_asset_text(1, fingerprint)?,
            Some("due north".to_owned())
        );
        assert_eq!(db.asset_text_embedding_status(999, fingerprint)?, None);
        assert_eq!(db.current_asset_text(999, fingerprint)?, None);

        // Both parts of the source fingerprint matter, even before a rescan runs.
        let changed = SourceFingerprint {
            modified_ns: fingerprint.modified_ns + 1,
            ..fingerprint
        };
        let resized = SourceFingerprint {
            size: fingerprint.size + 1,
            ..fingerprint
        };
        assert_eq!(db.asset_text_embedding_status(1, changed)?, None);
        assert_eq!(db.current_asset_text(1, changed)?, None);
        assert_eq!(db.asset_text_embedding_status(1, resized)?, None);

        let mut rescanned = result(&temp, 1, "due north, rescanned")?;
        rescanned.fingerprint = changed;
        db.save_results(vec![rescanned])?;
        assert_eq!(
            db.asset_text_embedding_status(1, changed)?,
            Some((true, false))
        );
        assert!(
            db.pending_text_embeddings(SPACE, &SearchFilters::under(&root), 10, 4096)?
                .iter()
                .any(|row| row.asset_id == 1)
        );
        let (_, hits) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert!(!hits.iter().any(|row| row.asset_id == 1));

        db.save_text_embeddings(
            SPACE,
            "test-model",
            vec![TextEmbedding {
                asset_id: 1,
                vector: vec![1.0, 0.0, 0.0],
            }],
        )?;
        assert_eq!(
            db.asset_text_embedding_status(1, changed)?,
            Some((true, true))
        );
        let (_, hits) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert!(hits.iter().any(|row| row.asset_id == 1));

        // Empty OCR is successful OCR, but must not be presented as embedding backlog.
        let empty = result(&temp, 4, "   ")?;
        db.save_results(vec![empty])?;
        assert_eq!(
            db.asset_text_embedding_status(4, fingerprint)?,
            Some((false, false))
        );
        assert!(
            !db.pending_text_embeddings(SPACE, &SearchFilters::under(&root), 10, 4096)?
                .iter()
                .any(|row| row.asset_id == 4)
        );

        // A model reset invalidates every old vector without touching the source/catalog.
        db.set_text_embedding_model(SPACE, "replacement-model", 4, true)?;
        assert_eq!(
            db.asset_text_embedding_status(1, changed)?,
            Some((true, false))
        );
        db.mark_for_deletion(&root)?;
        assert_eq!(db.asset_text_embedding_status(1, changed)?, None);
        db.delete_assets(&[1])?;
        assert_eq!(db.asset_text_embedding_status(1, changed)?, None);
        Ok(())
    }

    fn ids(results: &[TextVectorSearchResult]) -> Vec<i64> {
        results.iter().map(|hit| hit.asset_id).collect()
    }

    #[test]
    fn vector_search_orders_neighbours_by_cosine_distance() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;
        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        // Identical, then orthogonal, then opposite.
        assert_eq!(ids(&results), vec![1, 2, 3]);
        assert!(results[0].distance < 1e-6, "{:?}", results[0].distance);
        assert!((results[1].distance - 1.0).abs() < 1e-6, "{results:?}");
        assert!((results[2].distance - 2.0).abs() < 1e-6, "{results:?}");
        // The excerpt is the head of the OCR text, since there is no matched term to centre on.
        assert_eq!(results[0].contents, "due north");
        Ok(())
    }

    #[test]
    fn vector_search_preserves_totals_when_no_hits_are_returned() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;
        let (total, hits) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            0,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert!(hits.is_empty());
        let (total, hits) = db.search_text_vectors(
            &[0.0, 0.0, 1.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions {
                max_distance: Some(0.0),
                ..TextVectorSearchOptions::default()
            },
        )?;
        assert_eq!(total, 0);
        assert!(hits.is_empty());
        Ok(())
    }

    #[test]
    fn a_distance_ceiling_bounds_both_the_results_and_the_total() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;
        let options = TextVectorSearchOptions {
            max_distance: Some(1.5),
            ..TextVectorSearchOptions::default()
        };
        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            options,
        )?;
        assert_eq!(total, 2, "the opposite vector is past the ceiling");
        assert_eq!(ids(&results), vec![1, 2]);

        // `limit` caps the returned rows without changing the count of what matched.
        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            1,
            options,
        )?;
        assert_eq!(total, 2);
        assert_eq!(ids(&results), vec![1]);
        Ok(())
    }

    #[test]
    fn vector_search_respects_the_root_and_the_exclude_glob() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;
        let excluded = root.join("skip");
        let mut moved = result(&temp, 2, "due east")?;
        moved.path = excluded.join("2.png");
        db.save_results(vec![moved])?;
        db.save_text_embeddings(
            SPACE,
            "test-model",
            vec![TextEmbedding {
                asset_id: 2,
                vector: vec![0.0, 1.0, 0.0],
            }],
        )?;

        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::new(PathScope::root(&root).with_exclude([excluded.clone()])),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2);
        assert_eq!(ids(&results), vec![1, 3]);

        // A root that contains nothing embedded is an empty answer, not an error.
        let elsewhere = PathBuf::try_from(temp.path().join("elsewhere"))?;
        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&elsewhere),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 0);
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn searching_before_anything_is_embedded_is_empty_rather_than_an_error() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = searchable_db(&temp)?;
        assert_eq!(db.text_embedding_model(SPACE)?, None);
        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 0);
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn a_query_vector_of_the_wrong_width_is_refused() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;
        let error = db
            .search_text_vectors(
                &[1.0, 0.0],
                SPACE,
                &SearchFilters::under(&root),
                10,
                TextVectorSearchOptions::default(),
            )
            .expect_err("a two-dimensional query cannot search a three-dimensional model");
        assert!(format!("{error}").contains("2 dimensions"), "{error}");
        assert!(
            db.save_text_embeddings(
                SPACE,
                "test-model",
                vec![TextEmbedding {
                    asset_id: 1,
                    vector: vec![1.0, 0.0],
                }]
            )
            .is_err()
        );
        assert!(
            db.save_text_embeddings(
                SPACE,
                "other-model",
                vec![TextEmbedding {
                    asset_id: 1,
                    vector: vec![1.0, 0.0, 0.0],
                }]
            )
            .is_err(),
            "vectors from another model are not comparable"
        );
        Ok(())
    }

    #[test]
    fn re_running_ocr_retires_the_vector_until_it_is_embedded_again() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;
        assert_eq!(
            db.text_embedding_coverage(SPACE, &SearchFilters::under(&root))?,
            TextEmbeddingCoverage {
                indexed: 3,
                embedded: 3,
                last_indexed_ns: Some(987_654_321),
            }
        );

        let mut rescanned = result(&temp, 1, "due north, rescanned")?;
        rescanned.fingerprint.modified_ns += 1;
        db.save_results(vec![rescanned])?;

        let coverage = db.text_embedding_coverage(SPACE, &SearchFilters::under(&root))?;
        assert_eq!(coverage.embedded, 2);
        assert_eq!(coverage.pending(), 1);
        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2, "the stale vector does not answer searches");
        assert_eq!(ids(&results), vec![2, 3]);

        let pending = db.pending_text_embeddings(SPACE, &SearchFilters::under(&root), 10, 4096)?;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].asset_id, 1);
        assert_eq!(pending[0].content, "due north, rescanned");

        db.save_text_embeddings(
            SPACE,
            "test-model",
            vec![TextEmbedding {
                asset_id: 1,
                vector: vec![1.0, 0.0, 0.0],
            }],
        )?;
        assert_eq!(
            db.text_embedding_coverage(SPACE, &SearchFilters::under(&root))?
                .pending(),
            0
        );
        assert!(
            db.pending_text_embeddings(SPACE, &SearchFilters::under(&root), 10, 4096)?
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn deleting_ocr_rows_takes_their_vectors_with_them() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;

        db.delete_asset(2)?;
        assert_eq!(
            db.text_embedding_coverage(SPACE, &SearchFilters::under(&root))?
                .embedded,
            2
        );

        db.mark_for_deletion(Path::from_path(temp.path()).unwrap())?;
        db.unmark_asset(1)?;
        assert_eq!(db.sweep_deletions()?, 1);

        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 1);
        assert_eq!(ids(&results), vec![1]);
        // Nothing was left behind for a later asset id to collide with.
        assert_eq!(db.sweep_text_embeddings()?, 0);
        Ok(())
    }

    #[test]
    fn switching_models_needs_an_explicit_reset_and_clears_every_vector() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = embedded_db(&temp)?;

        let error = db
            .set_text_embedding_model(SPACE, "other-model", 4, false)
            .expect_err("silently discarding an embedded library is not acceptable");
        assert!(format!("{error}").contains("explicit reset"), "{error}");
        assert_eq!(
            db.text_embedding_coverage(SPACE, &SearchFilters::under(&root))?
                .embedded,
            3
        );

        db.set_text_embedding_model(SPACE, "other-model", 4, true)?;
        assert_eq!(
            db.text_embedding_model(SPACE)?,
            Some(StoredTextEmbeddingModel {
                model: "other-model".to_owned(),
                dimensions: 4,
            })
        );
        assert_eq!(
            db.text_embedding_coverage(SPACE, &SearchFilters::under(&root))?
                .embedded,
            0
        );
        assert_eq!(
            db.pending_text_embeddings(SPACE, &SearchFilters::under(&root), 10, 4096)?
                .len(),
            3
        );

        // Re-declaring the same model is a no-op, not a reset.
        db.save_text_embeddings(
            SPACE,
            "other-model",
            vec![TextEmbedding {
                asset_id: 1,
                vector: vec![1.0, 0.0, 0.0, 0.0],
            }],
        )?;
        db.set_text_embedding_model(SPACE, "other-model", 4, false)?;
        assert_eq!(
            db.text_embedding_coverage(SPACE, &SearchFilters::under(&root))?
                .embedded,
            1
        );
        Ok(())
    }

    #[test]
    fn embeddings_survive_a_reopen_and_are_visible_to_a_read_only_connection() -> Result<()> {
        let temp = TempDir::new()?;
        let (db, root) = embedded_db(&temp)?;
        drop(db);

        let mut reader = DB::new_read_only(&PathBuf::try_from(temp.path().join("ocr.db"))?)?;
        let (total, results) = reader.search_text_vectors(
            &[0.0, 1.0, 0.0],
            SPACE,
            &SearchFilters::under(&root),
            10,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert_eq!(ids(&results)[0], 2);
        Ok(())
    }

    /// Three rows an hour apart on the modified timeline, with asset 2 carrying a capture time a
    /// year earlier so the two timelines disagree about its position.
    fn timed_db(temp: &TempDir) -> Result<(DB, PathBuf)> {
        let mut db = test_db(temp)?;
        let mut rows = Vec::new();
        for (asset_id, modified_ns, exif_taken_ns) in [
            (1i64, HOUR, None),
            (2, 2 * HOUR, Some(-YEAR)),
            (3, 3 * HOUR, None),
        ] {
            let mut row = result(temp, asset_id, "needle")?;
            row.fingerprint.modified_ns = modified_ns;
            row.exif_taken_ns = exif_taken_ns;
            rows.push(row);
        }
        db.save_results(rows)?;
        Ok((db, PathBuf::try_from(temp.path().to_path_buf())?))
    }

    const HOUR: i64 = 3_600_000_000_000;
    const YEAR: i64 = 31_536_000 * 1_000_000_000;

    fn on(timeline: Timeline, after_ns: Option<i64>, before_ns: Option<i64>) -> TimeRange {
        TimeRange {
            timeline,
            after_ns,
            before_ns,
        }
    }

    fn found(db: &mut DB, filters: &SearchFilters) -> Result<Vec<i64>> {
        let results = db.search(vec!["needle"], filters, 40, SearchType::Simple)?;
        Ok(results.iter().map(|hit| hit.asset_id).collect())
    }

    #[test]
    fn a_time_range_is_half_open_on_the_modified_timeline() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = timed_db(&temp)?;
        let base = SearchFilters::under(&root);

        // No range at all is every instant.
        assert_eq!(found(&mut db, &base)?, vec![3, 2, 1]);

        // `after` is inclusive and `before` is exclusive, so a bound landing exactly on a row
        // includes it at the bottom and excludes it at the top.
        let range =
            base.clone()
                .with_time(Some(on(Timeline::Modified, Some(2 * HOUR), Some(3 * HOUR))));
        assert_eq!(found(&mut db, &range)?, vec![2]);

        let open_top = base
            .clone()
            .with_time(Some(on(Timeline::Modified, Some(2 * HOUR), None)));
        assert_eq!(found(&mut db, &open_top)?, vec![3, 2]);

        let open_bottom =
            base.clone()
                .with_time(Some(on(Timeline::Modified, None, Some(2 * HOUR))));
        assert_eq!(found(&mut db, &open_bottom)?, vec![1]);

        // Adjacent half-open ranges tile without double-counting the boundary instant.
        let lower = base
            .clone()
            .with_time(Some(on(Timeline::Modified, None, Some(2 * HOUR))));
        let upper = base
            .clone()
            .with_time(Some(on(Timeline::Modified, Some(2 * HOUR), None)));
        let mut tiled = found(&mut db, &lower)?;
        tiled.extend(found(&mut db, &upper)?);
        tiled.sort_unstable();
        assert_eq!(tiled, vec![1, 2, 3]);
        Ok(())
    }

    #[test]
    fn the_two_timelines_place_the_same_asset_differently() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = timed_db(&temp)?;
        let base = SearchFilters::under(&root);

        // Asset 2 was modified inside the window but captured a year before it.
        let window = (Some(0), Some(4 * HOUR));
        assert_eq!(
            found(
                &mut db,
                &base
                    .clone()
                    .with_time(Some(on(Timeline::Modified, window.0, window.1)))
            )?,
            vec![3, 2, 1]
        );
        assert_eq!(
            found(
                &mut db,
                &base
                    .clone()
                    .with_time(Some(on(Timeline::Capture, window.0, window.1)))
            )?,
            vec![3, 1],
            "capture time puts asset 2 outside the window"
        );

        // Assets without EXIF fall back to their modified time on the capture timeline, which is
        // what makes `capture` usable on a library that is only partly tagged.
        assert_eq!(
            found(
                &mut db,
                &base
                    .clone()
                    .with_time(Some(on(Timeline::Capture, Some(-2 * YEAR), Some(0))))
            )?,
            vec![2]
        );
        Ok(())
    }

    #[test]
    fn every_search_mode_honours_the_same_filters() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = timed_db(&temp)?;
        db.set_text_embedding_model(SPACE, "test-model", 3, false)?;
        db.save_text_embeddings(
            SPACE,
            "test-model",
            (1..=3)
                .map(|asset_id| TextEmbedding {
                    asset_id,
                    vector: vec![1.0, 0.0, 0.0],
                })
                .collect(),
        )?;

        let filters = SearchFilters::under(&root).with_time(Some(on(
            Timeline::Modified,
            Some(2 * HOUR),
            Some(3 * HOUR),
        )));

        // The point of the shared builder: one filter definition, identical narrowing everywhere.
        assert_eq!(found(&mut db, &filters)?, vec![2]);
        assert_eq!(
            db.search_count(vec!["needle"], &filters, SearchType::Simple)?,
            1
        );
        assert_eq!(
            db.search_count(vec!["*needle*"], &filters, SearchType::Glob)?,
            1
        );
        let (total, results) = db.search_text_vectors(
            &[1.0, 0.0, 0.0],
            SPACE,
            &filters,
            40,
            TextVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 1);
        assert_eq!(results[0].asset_id, 2);

        // Coverage and the embedding backlog narrow the same way, so "how much of what I am about
        // to search is embedded" is answerable exactly.
        assert_eq!(
            db.text_embedding_coverage(SPACE, &filters)?,
            TextEmbeddingCoverage {
                indexed: 1,
                embedded: 1,
                last_indexed_ns: Some(2 * HOUR),
            }
        );
        Ok(())
    }

    #[test]
    fn a_reversed_range_is_rejected_rather_than_silently_empty() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = timed_db(&temp)?;
        let filters = SearchFilters::under(&root).with_time(Some(on(
            Timeline::Modified,
            Some(3 * HOUR),
            Some(HOUR),
        )));
        let error = db
            .search(vec!["needle"], &filters, 40, SearchType::Simple)
            .expect_err("a range that ends before it starts is a caller mistake");
        assert!(format!("{error}").contains("less than"), "{error}");
        Ok(())
    }

    #[test]
    fn filters_compose_with_the_exclude_glob() -> Result<()> {
        let temp = TempDir::new()?;
        let (mut db, root) = timed_db(&temp)?;
        let excluded = root.join("skip");
        let mut moved = result(&temp, 3, "needle")?;
        moved.fingerprint.modified_ns = 3 * HOUR;
        moved.path = excluded.join("3.png");
        db.save_results(vec![moved])?;

        let filters = SearchFilters::new(PathScope::root(&root).with_exclude([excluded.clone()]))
            .with_time(Some(on(Timeline::Modified, Some(2 * HOUR), None)));
        assert_eq!(found(&mut db, &filters)?, vec![2]);
        Ok(())
    }

    #[test]
    fn search_preserves_catalog_owned_asset_id() -> Result<()> {
        let temp = TempDir::new()?;
        let mut db = test_db(&temp)?;
        let catalog_id = 8_765_432;
        db.save_results(vec![
            result(&temp, 10, "haystack")?,
            result(&temp, catalog_id, "haystack needle")?,
        ])?;

        let root = SearchFilters::under(Path::from_path(temp.path()).unwrap());
        let results = db.search(vec!["needle"], &root, 40, SearchType::Simple)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, catalog_id);
        assert_eq!(
            db.search_count(vec!["needle"], &root, SearchType::Simple)?,
            1
        );
        Ok(())
    }
}
