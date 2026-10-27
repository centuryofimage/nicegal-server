use nicegal_core::embedding::ImageEmbeddingModel;
use serde::Serialize;

/// The image model this process embeds with, which is fixed at startup, against the one saved in
/// settings for the next start.
pub(super) fn status(active: ImageEmbeddingModel, selected: ImageEmbeddingModel) -> ModelStatus {
    ModelStatus {
        active_model: active.id(),
        selected_model: selected.id(),
        supports_text_queries: active.supports_text_queries(),
        restart_required: active != selected,
        models: ImageEmbeddingModel::ALL
            .into_iter()
            .map(|model| ModelInfo {
                id: model.id(),
                name: model.name(),
                dimensions: model.dimensions(),
                license: model.license(),
                url: model.source_url(),
                available: model.available(),
                supports_text_queries: model.supports_text_queries(),
                patch_features: model.supports_patch_features(),
            })
            .collect(),
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
    /// Whether POST /v1/image-embeddings/patches serves this model.
    patch_features: bool,
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_different_selection_requires_a_restart() {
        let unchanged = status(
            ImageEmbeddingModel::MetaClip2B32,
            ImageEmbeddingModel::MetaClip2B32,
        );
        assert!(!unchanged.restart_required);
        assert_eq!(unchanged.models.len(), ImageEmbeddingModel::ALL.len());

        let changed = status(
            ImageEmbeddingModel::MetaClip2B16,
            ImageEmbeddingModel::SigLip2Base256,
        );
        assert!(changed.restart_required);
        assert_eq!(changed.active_model, ImageEmbeddingModel::MetaClip2B16.id());
        assert_eq!(
            changed.selected_model,
            ImageEmbeddingModel::SigLip2Base256.id()
        );
    }
}
