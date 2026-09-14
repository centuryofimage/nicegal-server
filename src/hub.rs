//! Resolving model files from Hugging Face.
//!
//! Every model family names its files as [`ModelSource`]s and resolves them through the one
//! standard cache, so a file downloaded for one family is never fetched again for another.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use hf_hub::api::tokio::{Api, ApiBuilder, Progress};
use hf_hub::{Cache, Repo, RepoType};
use tracing::{debug, error, info};

#[cfg(windows)]
mod windows;

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
            debug!(model_id = %self.model_id, filename = %self.filename, path = %path.display(), "Hugging Face cache hit");
            return Ok(path);
        }
        if let Some(path) = self.cached_in(cache) {
            return Ok(path);
        }

        let started = self.log_download_start();
        let result = api
            .repo(repo)
            .download_with_progress(&self.filename, progress.clone())
            .await
            .with_context(|| {
                format!(
                    "downloading {}/{} from Hugging Face",
                    self.model_id, self.filename
                )
            });
        #[cfg(windows)]
        let result = match result {
            Err(original) if windows::is_connection_error(&original) => {
                let source = self.clone();
                let cache = cache.clone();
                let path = tokio::task::spawn_blocking(move || {
                    windows::fallback(&source, &cache, original)
                })
                .await
                .context("joining Windows download fallback")?;
                if let Ok(path) = &path {
                    let mut progress = progress;
                    let size = std::fs::metadata(path)?.len() as usize;
                    progress.init(size, &self.filename).await;
                    progress.update(size).await;
                    progress.finish().await;
                }
                path
            }
            other => other,
        };
        self.log_download_result(started, &result);
        result
    }

    /// Resolve from the same cache in a blocking model-preparation worker. A search may use
    /// `cached()` instead, keeping the no-network-on-search contract.
    pub fn get_sync(&self) -> Result<PathBuf> {
        if let Some(path) = self.cached() {
            return Ok(path);
        }
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_progress(false)
            .build()
            .context("building the Hugging Face API client")?;
        let started = self.log_download_start();
        let result = api.repo(self.repo()).get(&self.filename).with_context(|| {
            format!(
                "downloading {}/{} from Hugging Face",
                self.model_id, self.filename
            )
        });
        #[cfg(windows)]
        let result = match result {
            Err(original) if windows::is_connection_error(&original) => {
                windows::fallback(self, &cache(), original)
            }
            other => other,
        };
        self.log_download_result(started, &result);
        result
    }

    fn log_download_start(&self) -> Instant {
        info!(
            model_id = %self.model_id,
            revision = self.revision.as_deref().unwrap_or("main"),
            filename = %self.filename,
            "Starting Hugging Face download"
        );
        Instant::now()
    }

    fn log_download_result(&self, started: Instant, result: &Result<PathBuf>) {
        match result {
            Ok(path) => info!(
                model_id = %self.model_id,
                revision = self.revision.as_deref().unwrap_or("main"),
                filename = %self.filename,
                elapsed_ms = started.elapsed().as_millis() as u64,
                path = %path.display(),
                "Hugging Face download complete"
            ),
            Err(error) => error!(
                model_id = %self.model_id,
                revision = self.revision.as_deref().unwrap_or("main"),
                filename = %self.filename,
                elapsed_ms = started.elapsed().as_millis() as u64,
                error_chain = %format!("{error:#}"),
                "Hugging Face download failed"
            ),
        }
    }

    pub fn cached(&self) -> Option<PathBuf> {
        let cache = cache();
        self.cached_in(&cache)
    }

    fn cached_in(&self, cache: &Cache) -> Option<PathBuf> {
        let repo = self.repo();
        if let Some(path) = cache.repo(repo.clone()).get(&self.filename) {
            return Some(path);
        }
        // hf-hub 0.4 resolves a revision through refs/<revision>, whereas Python's
        // huggingface_hub stores a commit-pinned download directly under snapshots/<SHA>
        // without creating that refs entry. Read the existing snapshot in either case.
        let revision = self.revision.as_deref()?;
        if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        let path = cache
            .path()
            .join(repo.folder_name())
            .join("snapshots")
            .join(revision)
            .join(&self.filename);
        path.is_file().then_some(path)
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

    #[test]
    #[ignore = "requires Hugging Face network access"]
    fn pinned_deepghs_metadata_downloads_into_the_shared_cache() {
        let source = ModelSource {
            model_id: "deepghs/siglip_beta".into(),
            revision: Some("03aa79c8a4a6c41e06ca87aa6e44fee563b2491d".into()),
            filename: "smilingwolf/siglip_eva02_base_2025_05_02_21h53m54s/meta.json".into(),
        };
        let path = source.get_sync().unwrap();
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(metadata["image_embedding_width"], 768);
        assert!(source.cached().is_some());
    }
}
