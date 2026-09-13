use anyhow::{Context, Result, bail};
use rusqlite::Connection;

/// Bring a freshly opened, writable connection's schema up to date: create it from `create_sql`
/// if the database is brand new (`user_version` is still 0), accept it unchanged if already on
/// `schema_version`, or refuse anything else rather than silently reading a schema this build
/// doesn't understand.
///
/// This helper deliberately has no generic migration path. A schema that supports an older
/// version must migrate it before calling this function; all other mismatches remain incompatible.
pub(crate) fn open_schema(
    conn: &Connection,
    label: &str,
    schema_version: i32,
    create_sql: &str,
) -> Result<()> {
    let user_version: i32 = conn
        .query_row("SELECT user_version FROM pragma_user_version", [], |row| {
            row.get(0)
        })
        .context("reading schema version")?;
    match user_version {
        0 => conn
            .execute_batch(create_sql)
            .with_context(|| format!("creating {label} tables"))?,
        version if version == schema_version => {}
        version => bail!(
            "{label} schema version {version} is incompatible with {schema_version}; create a new database"
        ),
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
    let user_version: i32 = conn
        .query_row("SELECT user_version FROM pragma_user_version", [], |row| {
            row.get(0)
        })
        .context("reading schema version")?;
    if user_version != schema_version {
        bail!(
            "{label} schema version {user_version} is incompatible with {schema_version}; create a new database"
        );
    }
    Ok(())
}
