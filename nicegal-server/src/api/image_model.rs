use nicegal_core::embedding::ImageEmbeddingModel;
use serde::Serialize;
use std::sync::Arc;

use super::RuntimeSettings;

pub(crate) struct ImageModelSettings {
    active: ImageEmbeddingModel,
    runtime: Arc<RuntimeSettings>,
}
impl ImageModelSettings {
    pub(crate) fn new(
        runtime: Arc<RuntimeSettings>,
        override_model: Option<ImageEmbeddingModel>,
    ) -> Self {
        Self {
            active: override_model.unwrap_or_else(|| runtime.image_model()),
            runtime,
        }
    }
    pub(crate) fn active(&self) -> ImageEmbeddingModel {
        self.active
    }
    pub(super) fn status(&self) -> ModelStatus {
        let selected = self.runtime.image_model();
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
    use anyhow::Result;
    use camino::Utf8PathBuf as PathBuf;
    #[test]
    fn selection_persists_without_replacing_active_model() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let path = PathBuf::try_from(temp.path().join("runtime.json"))?;
        let runtime = Arc::new(RuntimeSettings::load(path.clone(), None)?);
        let settings = ImageModelSettings::new(Arc::clone(&runtime), None);
        runtime.update(None, Some(ImageEmbeddingModel::SigLip2Base256))?;
        let status = settings.status();
        assert_eq!(status.active_model, ImageEmbeddingModel::MetaClip2B32.id());
        assert!(status.restart_required);
        let restarted =
            ImageModelSettings::new(Arc::new(RuntimeSettings::load(path.clone(), None)?), None);
        assert_eq!(restarted.active(), ImageEmbeddingModel::SigLip2Base256);
        assert!(!restarted.status().restart_required);
        let overridden =
            ImageModelSettings::new(runtime.clone(), Some(ImageEmbeddingModel::MetaClip2B16));
        assert_eq!(overridden.active(), ImageEmbeddingModel::MetaClip2B16);
        assert_eq!(
            overridden.status().selected_model,
            ImageEmbeddingModel::SigLip2Base256.id()
        );
        assert!(
            runtime
                .update(None, Some(ImageEmbeddingModel::LaionClipB32))
                .is_err()
        );
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
