//! CLIP image embedding jobs.

use camino::Utf8PathBuf as PathBuf;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::embedding::ImageEmbedder;
use nicegal_core::image_index::{ImageIndexDb, index_images_observed};
use nicegal_core::index::IndexObserver;
use serde::Deserialize;

use super::error::ApiError;
use super::roots;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    root: PathBuf,
    #[serde(default)]
    force: bool,
}

pub(crate) struct Spec {
    root: PathBuf,
    force: bool,
    retry_failed: bool,
}

impl Spec {
    pub(crate) fn pending_for(root: PathBuf, retry_failed: bool) -> Self {
        Self {
            root,
            force: false,
            retry_failed,
        }
    }
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    let root = roots::resolve_root("image embeddings", &request.root)?;
    Ok(Spec {
        root,
        force: request.force,
        retry_failed: false,
    })
}

pub(crate) fn run(
    spec: Spec,
    asset_database: &PathBuf,
    image_database: &PathBuf,
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
) -> anyhow::Result<bool> {
    let assets = AssetCatalog::new(asset_database)?;
    let mut images = ImageIndexDb::new(image_database, embedder.dimensions())?;
    index_images_observed(
        &assets,
        &mut images,
        embedder,
        &spec.root,
        spec.force,
        spec.retry_failed,
        observer,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> Result<Request, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn root_is_required_and_force_defaults_off() {
        assert!(request(serde_json::json!({})).is_err());
        let parsed = request(serde_json::json!({"root": "C:/gallery"})).unwrap();
        assert!(!parsed.force);
        assert!(
            request(serde_json::json!({
                "root": "C:/gallery",
                "unknown": true
            }))
            .is_err()
        );
    }
}
