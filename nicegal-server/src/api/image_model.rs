use anyhow::{Context, Result, bail};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::embedding::ImageEmbeddingModel;
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, sync::Mutex};

pub(crate) struct ImageModelSettings {
    path: PathBuf,
    active: ImageEmbeddingModel,
    selected: Mutex<ImageEmbeddingModel>,
}
impl ImageModelSettings {
    pub(crate) fn load(path: PathBuf, override_model: Option<ImageEmbeddingModel>) -> Result<Self> {
        let selected = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<Selection>(&bytes)?.model.parse()?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ImageEmbeddingModel::default()
            }
            Err(error) => return Err(error).context("reading image model settings"),
        };
        Ok(Self {
            path,
            active: override_model.unwrap_or(selected),
            selected: Mutex::new(selected),
        })
    }
    pub(crate) fn active(&self) -> ImageEmbeddingModel {
        self.active
    }
    pub(super) fn status(&self) -> ModelStatus {
        let selected = *self
            .selected
            .lock()
            .expect("image model settings lock poisoned");
        ModelStatus {
            active_model: self.active.id(),
            selected_model: selected.id(),
            supports_text_queries: self.active.supports_text_queries(),
            restart_required: self.active != selected,
            models: ImageEmbeddingModel::SELECTABLE
                .into_iter()
                .map(|model| ModelInfo {
                    id: model.id(),
                    name: model.name(),
                    dimensions: model.dimensions(),
                    license: model.license(),
                    url: model.source_url(),
                    available: model.available(),
                    supports_text_queries: model.supports_text_queries(),
                })
                .collect(),
        }
    }
    pub(super) fn set(&self, model: ImageEmbeddingModel) -> Result<ModelStatus> {
        if !ImageEmbeddingModel::SELECTABLE.contains(&model) {
            bail!("{} is retired from the model selector", model);
        }
        // Serialize updates, including the atomic replacement, under one lock.
        let mut selected = self
            .selected
            .lock()
            .expect("image model settings lock poisoned");
        let parent = self
            .path
            .parent()
            .context("image model settings path has no parent")?;
        fs::create_dir_all(parent)?;
        let temporary = self.path.with_extension("json.tmp");
        let mut file = fs::File::create(&temporary)?;
        file.write_all(&serde_json::to_vec_pretty(&Selection {
            model: model.id().to_owned(),
        })?)?;
        file.sync_all()?;
        drop(file);
        fs::rename(temporary, &self.path)?;
        *selected = model;
        drop(selected);
        Ok(self.status())
    }
}
#[derive(Deserialize, Serialize)]
struct Selection {
    model: String,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ModelStatus {
    active_model: &'static str,
    selected_model: &'static str,
    supports_text_queries: bool,
    pub(super) restart_required: bool,
    models: Vec<ModelInfo>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    id: &'static str,
    name: &'static str,
    dimensions: usize,
    license: &'static str,
    url: String,
    available: bool,
    supports_text_queries: bool,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_persists_without_replacing_active_model() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = PathBuf::try_from(temp.path().join("image-model.json"))?;
        let settings = ImageModelSettings::load(path.clone(), None)?;
        let status = settings.set(ImageEmbeddingModel::SigLip2Base256)?;
        assert_eq!(status.active_model, ImageEmbeddingModel::MetaClip2B32.id());
        assert!(status.restart_required);
        let restarted = ImageModelSettings::load(path.clone(), None)?;
        assert_eq!(restarted.active(), ImageEmbeddingModel::SigLip2Base256);
        assert!(!restarted.status().restart_required);
        let overridden = ImageModelSettings::load(path, Some(ImageEmbeddingModel::MetaClip2B16))?;
        assert_eq!(overridden.active(), ImageEmbeddingModel::MetaClip2B16);
        assert_eq!(
            overridden.status().selected_model,
            ImageEmbeddingModel::SigLip2Base256.id()
        );
        assert!(overridden.set(ImageEmbeddingModel::LaionClipB32).is_err());
        assert!(
            !overridden
                .status()
                .models
                .iter()
                .any(|model| model.id == ImageEmbeddingModel::LaionClipB32.id())
        );
        Ok(())
    }
}
