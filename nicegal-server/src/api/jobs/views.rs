//! Libraries client windows show, which drive automatic scans.
use std::sync::Arc;
use std::time::Instant;

use super::job::JobKind;
use super::manager::JobManager;
use crate::api::error::ApiError;
use crate::api::ttl_map::Retained;

/// The library a client window shows. A viewed library is released explicitly when the window
/// closes or navigates, so such views never age out.
#[derive(Clone, Copy)]
pub(super) struct LibraryView {
    pub(super) generation: u64,
    pub(super) library: Option<i64>,
    pub(super) updated: Instant,
}

impl Retained for LibraryView {
    fn touched_at(&self) -> Instant {
        self.updated
    }

    fn pinned(&self) -> bool {
        self.library.is_some()
    }
}

impl JobManager {
    /// A view is client-owned; scan admission and cancellation remain server-owned.
    pub(in crate::api) fn set_view(
        self: &Arc<Self>,
        client: String,
        generation: u64,
        library: Option<i64>,
    ) -> Result<(), ApiError> {
        let _transition = self.view_changes.lock();
        let (stop, start) = {
            let mut registry = self.registry();
            registry.views.sweep();
            let previous = registry.views.get(&client).copied();
            if previous.is_some_and(|view| generation <= view.generation) {
                return Ok(());
            }
            if previous.is_none() && !registry.views.has_room_for(&client) {
                return Err(ApiError::bad_request("too many library views"));
            }
            let viewed_elsewhere = |id: i64| {
                registry
                    .views
                    .iter()
                    .any(|(other, view)| *other != client && view.library == Some(id))
            };
            let stop = previous
                .and_then(|view| view.library)
                .filter(|id| Some(*id) != library && !viewed_elsewhere(*id));
            let already_viewed = library
                .is_some_and(|id| registry.views.values().any(|view| view.library == Some(id)));
            let start = library
                .filter(|_| !already_viewed)
                .map(|id| (id, registry.visited.contains(&id)));
            (stop, start)
        };
        // Scheduling can fail, so the view is committed only once it succeeds; a retry with the
        // same generation must not find the failed attempt already recorded.
        if let Some((id, pending_only)) = start {
            self.schedule_scan(id, pending_only)?;
        }
        {
            let mut registry = self.registry();
            registry.views.insert(
                client,
                LibraryView {
                    generation,
                    library,
                    updated: Instant::now(),
                },
            );
            if let Some((id, _)) = start {
                registry.visited.insert(id);
            }
        }
        if let Some(id) = stop {
            let ids: Vec<_> = self
                .registry()
                .jobs
                .values()
                .filter(|job| {
                    job.kind == JobKind::LibraryScan
                        && job.library_id == Some(id)
                        && !job.response().status.is_terminal()
                })
                .map(|job| job.id)
                .collect();
            for id in ids {
                self.cancel(id);
            }
        }
        Ok(())
    }

    pub(in crate::api) fn schedule_viewed_scan(
        self: &Arc<Self>,
        library: i64,
    ) -> Result<(), ApiError> {
        let _transition = self.view_changes.lock();
        let viewed = self
            .registry()
            .views
            .values()
            .any(|view| view.library == Some(library));
        if viewed {
            self.schedule_scan(library, true)?;
        }
        Ok(())
    }
}
