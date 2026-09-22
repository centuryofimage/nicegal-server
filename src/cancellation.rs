//! Cooperative cancellation for request-owned SQLite readers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use rusqlite::{Connection, InterruptHandle};

#[derive(Debug)]
struct SearchCancelled;

impl std::fmt::Display for SearchCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("search cancelled")
    }
}

impl std::error::Error for SearchCancelled {}

/// Distinguish expected cancellation from a database failure in request logs.
pub fn is_cancellation(error: &anyhow::Error) -> bool {
    error.is::<SearchCancelled>()
        || error
            .downcast_ref::<rusqlite::Error>()
            .is_some_and(|error| {
                error.sqlite_error_code() == Some(rusqlite::ErrorCode::OperationInterrupted)
            })
}

/// A request's cancellation state, shared with its blocking worker.
/// Connections registered here must belong exclusively to that request.
#[derive(Clone, Default)]
pub struct SearchCancellation {
    cancelled: Arc<AtomicBool>,
    handles: Arc<Mutex<Vec<InterruptHandle>>>,
}

impl SearchCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        for handle in self
            .handles
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            handle.interrupt();
        }
    }

    pub fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(SearchCancelled.into());
        }
        Ok(())
    }

    pub(crate) fn register(&self, conn: &Connection) -> Result<()> {
        // sqlite3_interrupt is transient: cancellation between statements must also be
        // observed by future statements. Capture just the flag, not the handle collection.
        let cancelled = Arc::clone(&self.cancelled);
        conn.progress_handler(1_000, Some(move || cancelled.load(Ordering::Acquire)))?;
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        self.check()?;
        handles.push(conn.get_interrupt_handle());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_QUERY: &str = "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL SELECT x+1 FROM n WHERE x < 1000000000) SELECT sum(x) FROM n";

    #[test]
    fn cancellation_between_statements_is_not_lost() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        let cancel = SearchCancellation::default();
        cancel.register(&conn)?;
        cancel.cancel();
        let error = conn
            .query_row(LONG_QUERY, [], |row| row.get::<_, i64>(0))
            .unwrap_err();
        assert_eq!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::OperationInterrupted)
        );
        assert!(cancel.register(&Connection::open_in_memory()?).is_err());
        Ok(())
    }

    #[test]
    fn cancellation_interrupts_running_sql_without_affecting_other_readers() -> Result<()> {
        let cancel = SearchCancellation::default();
        let conn = Connection::open_in_memory()?;
        cancel.register(&conn)?;
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        // Signal from inside SQLite, then use only InterruptHandle for this test.
        let mut started = Some(started_tx);
        conn.progress_handler(
            1_000,
            Some(move || {
                if let Some(tx) = started.take() {
                    tx.send(()).unwrap();
                }
                false
            }),
        )?;
        let worker =
            std::thread::spawn(move || conn.query_row(LONG_QUERY, [], |row| row.get::<_, i64>(0)));
        started_rx.recv_timeout(std::time::Duration::from_secs(5))?;
        cancel.cancel();
        let error = worker.join().unwrap().unwrap_err();
        assert_eq!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::OperationInterrupted)
        );
        let other = Connection::open_in_memory()?;
        assert_eq!(
            other.query_row("SELECT 42", [], |row| row.get::<_, i64>(0))?,
            42
        );
        Ok(())
    }
}
