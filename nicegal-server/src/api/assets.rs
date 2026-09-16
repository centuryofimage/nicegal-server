use std::collections::{HashMap, HashSet};

use anyhow::Context;
use axum::Json;
use axum::extract::State;
use axum::routing::{MethodRouter, get};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::assets::{Asset, MediaKind, canonicalize_path};
use nicegal_core::db::OcrIndexState;
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::extract::{ApiJson, ApiQuery};
use super::{AppState, run_blocking};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetLookup {
    path: PathBuf,
}

const MAX_BATCH_ASSETS: usize = 512;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AssetBatchLookup {
    asset_ids: Vec<i64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AssetBatchResponse {
    assets: Vec<AssetResponse>,
    missing_asset_ids: Vec<i64>,
}

/// The current relationship between an asset and its persisted OCR data.
///
/// The renderer should present `stale` as “Needs reindex”; it means OCR exists, but came from an
/// older source fingerprint. `notIndexed` means no OCR row exists for the asset.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum IndexStateResponse {
    Indexed,
    Stale,
    NotIndexed,
}

impl From<Option<OcrIndexState>> for IndexStateResponse {
    fn from(state: Option<OcrIndexState>) -> Self {
        match state {
            Some(OcrIndexState::Indexed) => Self::Indexed,
            Some(OcrIndexState::Stale) => Self::Stale,
            None => Self::NotIndexed,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AssetResponse {
    asset_id: i64,
    path: PathBuf,
    display_name: String,
    folder_path: Option<PathBuf>,
    extension: Option<String>,
    source_modified_ns: String,
    source_created_ns: Option<String>,
    exif_taken_ns: Option<String>,
    source_size: u64,
    media_kind: &'static str,
    media_format: String,
    width: Option<u32>,
    height: Option<u32>,
    is_animated: bool,
    frame_count: Option<u32>,
    duration_ms: Option<u64>,
    index_state: IndexStateResponse,
}

impl AssetResponse {
    fn new(asset: Asset, index_state: IndexStateResponse) -> Self {
        let display_name = asset.display_name().to_owned();
        let extension = asset.extension().map(str::to_owned);
        let folder_path = asset.path.parent().map(|path| path.to_path_buf());
        Self {
            asset_id: asset.asset_id,
            path: asset.path,
            display_name,
            folder_path,
            extension,
            source_modified_ns: asset.fingerprint.modified_ns.to_string(),
            source_created_ns: asset.source_created_ns.map(|value| value.to_string()),
            exif_taken_ns: asset.exif_taken_ns.map(|value| value.to_string()),
            source_size: asset.fingerprint.size,
            media_kind: match asset.media_kind {
                MediaKind::Image => "image",
                MediaKind::Video => "video",
            },
            media_format: asset.media_format,
            width: asset.width,
            height: asset.height,
            is_animated: asset.is_animated,
            frame_count: asset.frame_count,
            duration_ms: asset.duration_ms,
            index_state,
        }
    }
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(find_by_path).post(find_by_ids)
}

async fn find_by_path(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<AssetLookup>,
) -> Result<Json<AssetResponse>, ApiError> {
    if !request.path.is_absolute() {
        return Err(ApiError::bad_request("asset path must be absolute"));
    }
    let response = run_blocking(move || {
        // A path that is simply gone is a lookup miss, not a failure to resolve it.
        let path = canonicalize_path(&request.path).map_err(|error| {
            if is_not_found(&error) {
                ApiError::asset_not_found()
            } else {
                ApiError::internal(error)
            }
        })?;
        let catalog = state.databases.open_assets_read_only()?;
        let asset = catalog
            .get_by_path(&path)
            .context("looking up asset by path")?
            .ok_or_else(ApiError::asset_not_found)?;
        let states = state
            .databases
            .open_ocr_read_only()?
            .index_states(&[(asset.asset_id, asset.fingerprint)])?;
        let index_state = states.get(&asset.asset_id).copied().into();
        Ok(AssetResponse::new(asset, index_state))
    })
    .await?;
    Ok(Json(response))
}

async fn find_by_ids(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<AssetBatchLookup>,
) -> Result<Json<AssetBatchResponse>, ApiError> {
    validate_batch_request(&request)?;
    let response = run_blocking(move || {
        let catalog = state.databases.open_assets_read_only()?;
        let assets = catalog.get_many(&request.asset_ids)?;
        let states = state.databases.open_ocr_read_only()?.index_states(
            &assets
                .iter()
                .map(|asset| (asset.asset_id, asset.fingerprint))
                .collect::<Vec<_>>(),
        )?;
        Ok(batch_response(request.asset_ids, assets, states))
    })
    .await?;
    Ok(Json(response))
}

fn validate_batch_request(request: &AssetBatchLookup) -> Result<(), ApiError> {
    if request.asset_ids.len() > MAX_BATCH_ASSETS {
        return Err(ApiError::bad_request(format!(
            "at most {MAX_BATCH_ASSETS} asset IDs may be requested at once"
        )));
    }
    if request.asset_ids.iter().any(|asset_id| *asset_id <= 0) {
        return Err(ApiError::bad_request(
            "asset identifiers must be greater than zero",
        ));
    }
    let unique = request.asset_ids.iter().collect::<HashSet<_>>();
    if unique.len() != request.asset_ids.len() {
        return Err(ApiError::bad_request("asset identifiers must be unique"));
    }
    Ok(())
}

fn batch_response(
    requested_asset_ids: Vec<i64>,
    assets: Vec<Asset>,
    states: HashMap<i64, OcrIndexState>,
) -> AssetBatchResponse {
    let found = assets
        .iter()
        .map(|asset| asset.asset_id)
        .collect::<HashSet<_>>();
    AssetBatchResponse {
        assets: assets
            .into_iter()
            .map(|asset| {
                let index_state = states.get(&asset.asset_id).copied().into();
                AssetResponse::new(asset, index_state)
            })
            .collect(),
        missing_asset_ids: requested_asset_ids
            .into_iter()
            .filter(|asset_id| !found.contains(asset_id))
            .collect(),
    }
}

/// Whether an error chain bottoms out in a missing filesystem entry.
fn is_not_found(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use anyhow::Result;
    use nicegal_core::assets::AssetCatalog;
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn response_serializes_catalogued_display_metadata() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("Summer.photo.PNG"))?;
        fs::write(&source, [])?;
        let catalog_path = PathBuf::try_from(temp.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let asset = catalog.upsert(&source, &fs::metadata(&source)?)?;
        let expected_path = asset.path.clone();
        let expected_folder = asset.path.parent().map(|path| path.to_path_buf());
        let expected_created = asset.source_created_ns;

        let response = serde_json::to_value(AssetResponse::new(
            asset.clone(),
            IndexStateResponse::Indexed,
        ))?;

        assert_eq!(response["path"], expected_path.as_str());
        assert_eq!(response["displayName"], "Summer.photo.PNG");
        assert_eq!(
            response["folderPath"],
            expected_folder
                .as_ref()
                .map(|path| serde_json::Value::String(path.to_string()))
                .unwrap_or(serde_json::Value::Null)
        );
        assert_eq!(response["extension"], "PNG");
        assert_eq!(
            response["sourceCreatedNs"],
            expected_created
                .map(|value| serde_json::Value::String(value.to_string()))
                .unwrap_or(serde_json::Value::Null)
        );
        assert_eq!(response["indexState"], "indexed");

        let batch = serde_json::to_value(batch_response(
            vec![asset.asset_id, 999],
            vec![asset.clone()],
            HashMap::from([(asset.asset_id, OcrIndexState::Stale)]),
        ))?;
        assert_eq!(batch["assets"][0]["assetId"], asset.asset_id);
        assert_eq!(batch["assets"][0]["indexState"], "stale");
        assert_eq!(batch["missingAssetIds"], serde_json::json!([999]));
        Ok(())
    }

    #[test]
    fn batch_requests_reject_duplicate_and_oversized_identifiers() {
        let duplicate = AssetBatchLookup {
            asset_ids: vec![4, 4],
        };
        assert!(validate_batch_request(&duplicate).is_err());

        let oversized = AssetBatchLookup {
            asset_ids: vec![1; MAX_BATCH_ASSETS + 1],
        };
        assert!(validate_batch_request(&oversized).is_err());
    }
}
