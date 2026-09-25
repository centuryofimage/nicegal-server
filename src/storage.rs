//! Small SQLite policies shared by the catalog and derived stores.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use rusqlite::functions::FunctionFlags;
use rusqlite::types::Value;
use rusqlite::{Connection, OpenFlags, ToSql};

pub(crate) const READ_ONLY_FLAGS: OpenFlags =
    OpenFlags::SQLITE_OPEN_READ_ONLY.union(OpenFlags::SQLITE_OPEN_NO_MUTEX);

pub(crate) fn configure_reader(conn: &Connection) -> Result<()> {
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.create_scalar_function(
        "nicegal_path_contains",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |context| {
            let path: String = context.get(0)?;
            let needle: String = context.get(1)?;
            Ok(path.replace('\\', "/").to_lowercase().contains(&needle))
        },
    )?;
    Ok(())
}

pub(crate) fn configure_writer(conn: &Connection) -> Result<()> {
    configure_reader(conn)?;
    conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;
    conn.pragma_update(None, "journal_mode", "wal")?;
    conn.pragma_update(None, "synchronous", "normal")?;
    Ok(())
}

/// Perform bounded, low-impact maintenance after a resource-intensive job releases its
/// transactions. Incremental vacuuming only visits free pages already tracked by SQLite, while a
/// passive checkpoint never waits for readers or writers to leave the WAL.
#[tracing::instrument(level = "debug", skip(conn))]
pub(crate) fn maintain(conn: &Connection) -> Result<()> {
    conn.execute_batch("PRAGMA optimize; PRAGMA incremental_vacuum(1000);")?;
    conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(()))?;
    Ok(())
}

pub(crate) fn validate_asset_ids(asset_ids: &[i64]) -> Result<()> {
    if asset_ids.iter().any(|asset_id| *asset_id <= 0) {
        bail!("asset identifiers must be greater than zero");
    }
    Ok(())
}

/// rusqlite rejects a named parameter the statement does not mention, so owned parameter lists
/// are only borrowed as `ToSql` at the point of binding.
pub(crate) fn bind_named<N: AsRef<str>>(params: &[(N, Value)]) -> Vec<(&str, &dyn ToSql)> {
    params
        .iter()
        .map(|(name, value)| (name.as_ref(), value as &dyn ToSql))
        .collect()
}

/// Rolls back on errors, failed commits, and unwinding while retaining access to the owner.
pub(crate) struct ReadSnapshot<T> {
    pub target: T,
    connection: fn(&T) -> &Connection,
    active: bool,
}

impl<T> ReadSnapshot<T> {
    pub fn begin(target: T, connection: fn(&T) -> &Connection) -> Result<Self> {
        connection(&target).execute_batch("BEGIN DEFERRED")?;
        Ok(Self {
            target,
            connection,
            active: true,
        })
    }

    pub fn commit(&mut self) -> Result<()> {
        (self.connection)(&self.target).execute_batch("COMMIT")?;
        self.active = false;
        Ok(())
    }
}

impl<T> Drop for ReadSnapshot<T> {
    fn drop(&mut self) {
        if self.active {
            let _ = (self.connection)(&self.target).execute_batch("ROLLBACK");
        }
    }
}

/// Execute a fingerprint-existence query with the same checked SQLite bindings for each source.
pub(crate) fn matching_fingerprint_ids(
    conn: &Connection,
    sql: &str,
    fingerprints: &[(i64, crate::assets::SourceFingerprint)],
) -> Result<std::collections::HashSet<i64>> {
    let mut matches = std::collections::HashSet::with_capacity(fingerprints.len());
    let mut statement = conn.prepare_cached(sql)?;
    for (asset_id, fingerprint) in fingerprints {
        let size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        if statement.query_row((*asset_id, fingerprint.modified_ns, size), |row| {
            row.get::<_, bool>(0)
        })? {
            matches.insert(*asset_id);
        }
    }
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_commit_rolls_back_when_snapshot_drops() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
            CREATE TABLE parent (id INTEGER PRIMARY KEY);
            CREATE TABLE child (parent_id INTEGER REFERENCES parent(id)
                DEFERRABLE INITIALLY DEFERRED);",
        )?;
        {
            let mut snapshot = ReadSnapshot::begin(&conn, |conn| conn)?;
            conn.execute("INSERT INTO child VALUES (1)", [])?;
            assert!(snapshot.commit().is_err());
        }
        assert!(conn.is_autocommit());
        assert_eq!(
            conn.query_row("SELECT count(*) FROM child", [], |row| row.get::<_, i64>(0))?,
            0
        );
        Ok(())
    }

    #[test]
    fn writers_use_incremental_vacuum_and_maintenance_checkpoints_wal() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        configure_writer(&conn)?;
        assert_eq!(
            conn.pragma_query_value(None, "auto_vacuum", |row| row.get::<_, i64>(0))?,
            2
        );
        maintain(&conn)?;
        Ok(())
    }
}
