use anyhow::{Context, Result, bail};
use rusqlite::{Connection, Transaction, TransactionBehavior};

/// Each entry upgrades its version to the next. DDL/data steps commit together, while a special
/// vacuum-only step runs immediately before that transaction because SQLite forbids `VACUUM`
/// inside one. This helper owns `user_version` in both cases.
pub(crate) fn open_schema_with_migrations(
    conn: &Connection,
    label: &str,
    schema_version: i32,
    create_sql: &str,
    migrations: &[(i32, &str)],
) -> Result<()> {
    let user_version = read_version(conn)?;
    // VACUUM cannot run inside the transaction below. A vacuum-only migration is used when a
    // schema version changes SQLite file-layout policy (currently auto_vacuum); run it first,
    // after configure_writer has selected the new policy, then record the version normally.
    let mut reachable_version = user_version;
    let mut requires_vacuum = false;
    while reachable_version != 0 && reachable_version < schema_version {
        let Some((_, sql)) = migrations
            .iter()
            .find(|(from, _)| *from == reachable_version)
        else {
            break;
        };
        requires_vacuum |= sql.trim().eq_ignore_ascii_case("VACUUM;");
        reachable_version += 1;
    }
    if reachable_version == schema_version && requires_vacuum {
        conn.execute_batch("VACUUM;")
            .with_context(|| format!("vacuuming {label} for schema migration"))?;
    }
    match user_version {
        0 => conn
            .execute_batch(create_sql)
            .with_context(|| format!("creating {label} tables"))?,
        version if version == schema_version => {}
        _ => {
            // Acquire the writer lock before checking again: another opener may have migrated
            // this database since the initial version read.
            let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
            let mut version = read_version(&tx)?;
            while version != schema_version {
                let sql = migrations
                    .iter()
                    .find(|(from, _)| *from == version && version < schema_version)
                    .map(|(_, sql)| *sql)
                    .with_context(|| format!(
                        "{label} schema version {version} is incompatible with {schema_version}; create a new database"
                    ))?;
                if !sql.trim().eq_ignore_ascii_case("VACUUM;") {
                    tx.execute_batch(sql).with_context(|| {
                        format!(
                            "migrating {label} schema from version {version} to {}",
                            version + 1
                        )
                    })?;
                }
                version += 1;
                tx.pragma_update(None, "user_version", version)?;
            }
            tx.commit()
                .with_context(|| format!("committing {label} migrations"))?;
        }
    }
    Ok(())
}

/// Refuse a read-only connection to anything but the exact schema version this build expects. A
/// reader never creates a schema, so `user_version == 0` is rejected the same as any other
/// mismatch instead of silently treating an empty database as current.
pub(crate) fn check_schema_read_only(
    conn: &Connection,
    label: &str,
    schema_version: i32,
) -> Result<()> {
    let user_version = read_version(conn)?;
    if user_version != schema_version {
        bail!(
            "{label} schema version {user_version} is incompatible with {schema_version}; create a new database"
        );
    }
    Ok(())
}

fn read_version(conn: &Connection) -> Result<i32> {
    conn.query_row("SELECT user_version FROM pragma_user_version", [], |row| {
        row.get(0)
    })
    .context("reading schema version")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_or_incomplete_migrations_roll_back_every_step() -> Result<()> {
        for last_step in [None, Some("INSERT INTO missing_table VALUES (1);")] {
            let conn = Connection::open_in_memory()?;
            conn.execute_batch("CREATE TABLE original(value INTEGER); PRAGMA user_version = 1;")?;
            let mut migrations = vec![(1, "ALTER TABLE original ADD COLUMN added INTEGER;")];
            if let Some(sql) = last_step {
                migrations.push((2, sql));
            }
            assert!(open_schema_with_migrations(&conn, "test", 3, "", &migrations).is_err());
            assert_eq!(read_version(&conn)?, 1);
            let columns: i64 = conn.query_row(
                "SELECT count(*) FROM pragma_table_info('original')",
                [],
                |row| row.get(0),
            )?;
            assert_eq!(columns, 1);
            assert!(conn.is_autocommit());
        }
        Ok(())
    }

    #[test]
    fn unsupported_versions_and_readers_do_not_migrate() -> Result<()> {
        for version in [1, 4] {
            let conn = Connection::open_in_memory()?;
            conn.pragma_update(None, "user_version", version)?;
            assert!(
                open_schema_with_migrations(&conn, "test", 3, "", &[(2, "SELECT 1;")]).is_err()
            );
            assert_eq!(read_version(&conn)?, version);
        }
        let conn = Connection::open_in_memory()?;
        conn.pragma_update(None, "user_version", 2)?;
        assert!(check_schema_read_only(&conn, "test", 3).is_err());
        assert_eq!(read_version(&conn)?, 2);
        Ok(())
    }

    #[test]
    fn vacuum_only_migration_activates_incremental_auto_vacuum() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("CREATE TABLE original(value INTEGER); PRAGMA user_version = 3;")?;
        conn.pragma_update(None, "auto_vacuum", "INCREMENTAL")?;

        open_schema_with_migrations(&conn, "test", 4, "", &[(3, "VACUUM;")])?;

        assert_eq!(read_version(&conn)?, 4);
        assert_eq!(
            conn.pragma_query_value(None, "auto_vacuum", |row| row.get::<_, i64>(0))?,
            2
        );
        Ok(())
    }
}
