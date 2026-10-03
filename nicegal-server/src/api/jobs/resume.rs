//! The interrupted thumbnail job record kept beside the asset database.
use std::sync::Arc;

use camino::Utf8PathBuf as PathBuf;

use super::manager::JobManager;
use super::request::{JobRequest, JobSpec};

impl JobManager {
    pub(super) fn resume_path(&self) -> PathBuf {
        self.databases.assets.with_extension("pending-job.json")
    }

    pub(super) fn clear_resume(&self) {
        if let Err(error) = std::fs::remove_file(self.resume_path())
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::error!(?error, "could not clear interrupted job record");
        }
    }

    pub(super) fn resume_thumbnail_job(self: &Arc<Self>) -> anyhow::Result<()> {
        match std::fs::read(self.resume_path()) {
            Ok(bytes) => {
                let request: JobRequest = serde_json::from_slice(&bytes)?;
                let spec = request
                    .prepare()
                    .map_err(|error| anyhow::anyhow!(error.message))?;
                if matches!(spec, JobSpec::ThumbnailGenerate(_)) {
                    self.start(spec)
                        .map_err(|error| anyhow::anyhow!(error.message))?;
                } else {
                    anyhow::bail!("invalid interrupted job record");
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        Ok(())
    }

    pub(crate) async fn recover(self: &Arc<Self>) -> anyhow::Result<()> {
        if let Err(error) = self.resume_thumbnail_job() {
            tracing::error!(?error, "could not resume thumbnail work");
        }
        Ok(())
    }
}
