mod job;
mod manager;
mod progress;
mod request;
mod resume;
mod routes;
#[cfg(test)]
mod tests;
mod views;

pub(super) use job::{FolderState, IndexStages, Job, JobResponse};
pub(crate) use manager::JobManager;
pub(super) use request::JobSpec;
pub(super) use routes::{routes, start_job};

#[derive(Debug)]
pub(crate) struct JobCancelled;

impl std::fmt::Display for JobCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("job cancelled")
    }
}

impl std::error::Error for JobCancelled {}

pub(crate) fn cancel_if(cancelled: bool) -> anyhow::Result<()> {
    if cancelled {
        Err(JobCancelled.into())
    } else {
        Ok(())
    }
}

pub(crate) fn is_cancelled(error: &anyhow::Error) -> bool {
    error.downcast_ref::<JobCancelled>().is_some() || nicegal_core::hub::is_cancellation(error)
}
