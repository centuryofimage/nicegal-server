//! Resolving model files from Hugging Face.
//!
//! Every model family names its files as [`ModelSource`]s and resolves them through the one
//! standard cache, so a file downloaded for one family is never fetched again for another.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use hf_hub::api::tokio::{Api, ApiBuilder, Progress};
use hf_hub::{Cache, Repo, RepoType};

/// The Hugging Face cache this process reads and writes.
///
/// Configured entirely by the environment (`HF_HOME`, `HF_HUB_CACHE`), so every consumer agrees on
/// one location without any of them choosing a directory of its own.
pub fn cache() -> Cache {
    Cache::from_env()
}

/// The directory backing [`cache`], for logging and for backends that take a path.
pub fn cache_dir() -> PathBuf {
    cache().path().clone()
}

/// An API client for downloading on a cache miss.
///
/// hf-hub's own progress bar is off: callers report progress through the [`Progress`] they pass to
/// [`ModelSource::get_with_progress`], so downloads surface wherever that caller reports work.
pub fn api() -> Result<Api> {
    ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .context("building the Hugging Face API client")
}

/// One versioned file in a Hugging Face model repository.
///
/// Repository IDs are supplied by the caller rather than compiled into the application. This lets
/// the UI select a model release and makes a future model upgrade independent of a binary release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSource {
    pub model_id: String,
    pub revision: Option<String>,
    pub filename: String,
}

impl ModelSource {
    pub fn repo(&self) -> Repo {
        match &self.revision {
            Some(revision) => {
                Repo::with_revision(self.model_id.clone(), RepoType::Model, revision.clone())
            }
            None => Repo::model(self.model_id.clone()),
        }
    }

    /// Describe a file already on disk, for callers that bypass the hub entirely. The path stands
    /// in for the repository ID so a local model still reports where it came from.
    pub fn local(path: &Path) -> Result<Self> {
        let filename = path
            .file_name()
            .and_then(|filename| filename.to_str())
            .filter(|filename| !filename.is_empty())
            .ok_or_else(|| anyhow!("model path has no UTF-8 filename: {}", path.display()))?;
        Ok(Self {
            model_id: path.display().to_string(),
            revision: None,
            filename: filename.to_owned(),
        })
    }

    /// Resolve this file through Hugging Face's standard cache, downloading only on a cache miss.
    pub async fn get_with_progress<P>(
        &self,
        api: &Api,
        cache: &Cache,
        progress: P,
    ) -> Result<PathBuf>
    where
        P: Progress + Clone + Send + Sync + 'static,
    {
        let repo = self.repo();
        if let Some(path) = cache.repo(repo.clone()).get(&self.filename) {
            return Ok(path);
        }

        api.repo(repo)
            .download_with_progress(&self.filename, progress)
            .await
            .with_context(|| {
                format!(
                    "downloading {}/{} from Hugging Face",
                    self.model_id, self.filename
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_file_is_described_by_its_own_path() {
        let source = ModelSource::local(Path::new("models/det/inference.onnx")).unwrap();
        assert_eq!(source.filename, "inference.onnx");
        assert_eq!(source.revision, None);
        assert!(source.model_id.ends_with("inference.onnx"));
    }

    #[test]
    fn a_directory_is_not_a_model_file() {
        assert!(ModelSource::local(Path::new("models/det/..")).is_err());
    }
}
