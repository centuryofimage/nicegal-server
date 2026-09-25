use axum::{Json, Router, extract::State, routing::get};
use camino::Utf8PathBuf;
use nicegal_core::assets::{Asset, Timeline};
use nicegal_core::metadata::{FileMetadata, SourceState};
use serde::{Deserialize, Serialize};

use super::{
    AppState, assets::IndexStateResponse, error::ApiError, extract::ApiQuery, libraries,
    run_blocking,
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Listing {
    library_id: i64,
    #[serde(default)]
    timeline: CatalogTimeline,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
enum CatalogTimeline {
    #[default]
    Modified,
    Capture,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Lookup {
    asset_id: i64,
}

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/catalog", get(list))
        .route("/v1/catalog/folders", get(folders))
        .route("/v1/catalog/count", get(count))
        .route("/v1/catalog/revision", get(revision))
        .route("/v1/catalog/metadata", get(metadata))
}

async fn folders(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<Listing>,
) -> Result<Json<Vec<String>>, ApiError> {
    Ok(Json(
        run_blocking(move || {
            let scope = libraries::scope(&state.databases, request.library_id)?;
            Ok(state
                .databases
                .open_assets_read_only()?
                .list_folders(request.library_id, &scope)?
                .into_iter()
                .map(Utf8PathBuf::into_string)
                .collect())
        })
        .await?,
    ))
}

async fn list(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<Listing>,
) -> Result<Json<Vec<GalleryAssetResponse>>, ApiError> {
    Ok(Json(
        run_blocking(move || {
            let scope = libraries::scope(&state.databases, request.library_id)?;
            let catalog = state.databases.open_assets_read_only()?;
            let timeline = match request.timeline {
                CatalogTimeline::Modified => Timeline::Modified,
                CatalogTimeline::Capture => Timeline::Capture,
            };
            Ok(catalog
                .list_gallery(&scope, timeline)?
                .into_iter()
                .map(GalleryAssetResponse::from)
                .collect())
        })
        .await?,
    ))
}

async fn count(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<Listing>,
) -> Result<Json<i64>, ApiError> {
    Ok(Json(
        run_blocking(move || {
            let scope = libraries::scope(&state.databases, request.library_id)?;
            Ok(state
                .databases
                .open_assets_read_only()?
                .count_gallery(&scope)?)
        })
        .await?,
    ))
}

async fn revision(State(state): State<AppState>) -> Result<Json<String>, ApiError> {
    Ok(Json(
        run_blocking(move || {
            Ok(state
                .databases
                .open_assets_read_only()?
                .revision()?
                .to_string())
        })
        .await?,
    ))
}

/// Match the desktop GalleryAsset contract without a per-field JSON tree for a whole library.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GalleryAssetResponse {
    id: String,
    path: Utf8PathBuf,
    display_name: String,
    extension: Option<String>,
    modified_ns: String,
    created_ns: Option<String>,
    capture_ns: Option<String>,
    source_size: String,
    media_kind: &'static str,
    media_format: String,
    width: Option<u32>,
    height: Option<u32>,
    animated: bool,
    frame_count: Option<u32>,
    duration_ms: Option<u64>,
}

impl From<Asset> for GalleryAssetResponse {
    fn from(asset: Asset) -> Self {
        Self {
            id: asset.asset_id.to_string(),
            display_name: asset.display_name().to_owned(),
            extension: asset.extension().map(str::to_owned),
            path: asset.path,
            modified_ns: asset.fingerprint.modified_ns.to_string(),
            created_ns: asset.source_created_ns.map(|v| v.to_string()),
            capture_ns: asset.exif_taken_ns.map(|v| v.to_string()),
            source_size: asset.fingerprint.size.to_string(),
            media_kind: match asset.media_kind {
                nicegal_core::assets::MediaKind::Image => "image",
                nicegal_core::assets::MediaKind::Video => "video",
            },
            media_format: asset.media_format,
            width: asset.width,
            height: asset.height,
            animated: asset.is_animated,
            frame_count: asset.frame_count,
            duration_ms: asset.duration_ms,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MetadataResponse {
    asset: GalleryAssetResponse,
    file: FileMetadataResponse,
    ocr_state: IndexStateResponse,
    ocr_text: Option<String>,
    image_indexed: bool,
    text_state: &'static str,
    decode_failed: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileMetadataResponse {
    source_state: &'static str,
    attributes: Vec<&'static str>,
    exif: Vec<MetadataField>,
    video: Vec<MetadataField>,
    error: Option<String>,
}

#[derive(Serialize)]
struct MetadataField {
    label: &'static str,
    value: String,
}

impl From<FileMetadata> for FileMetadataResponse {
    fn from(file: FileMetadata) -> Self {
        Self {
            source_state: match file.source_state {
                SourceState::Current => "current",
                SourceState::Changed => "changed",
                SourceState::Missing => "missing",
                SourceState::Unavailable => "unavailable",
            },
            attributes: file.attributes,
            exif: file
                .exif
                .into_iter()
                .map(|field| MetadataField {
                    label: field.label,
                    value: field.value,
                })
                .collect(),
            video: file
                .video
                .into_iter()
                .map(|field| MetadataField {
                    label: field.label,
                    value: field.value,
                })
                .collect(),
            error: file.error,
        }
    }
}

async fn metadata(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<Lookup>,
) -> Result<Json<MetadataResponse>, ApiError> {
    if request.asset_id <= 0 {
        return Err(ApiError::bad_request(
            "asset identifier must be greater than zero",
        ));
    }
    Ok(Json(
        run_blocking(move || {
            let catalog = state.databases.open_assets_read_only()?;
            let asset = catalog
                .get(request.asset_id)?
                .ok_or_else(ApiError::asset_not_found)?;
            let fingerprints = [(asset.asset_id, asset.fingerprint)];
            let ocr = state.databases.open_ocr_read_only()?;
            let ocr_state: IndexStateResponse = ocr
                .index_states(&fingerprints)?
                .get(&asset.asset_id)
                .copied()
                .into();
            let ocr_text = ocr.current_asset_text(asset.asset_id, asset.fingerprint)?;
            let decode_failed = catalog
                .current_decode_failure_asset_ids(&fingerprints)?
                .contains(&asset.asset_id);
            let text_state =
                match ocr.asset_text_embedding_status(asset.asset_id, asset.fingerprint)? {
                    None => "notIndexed",
                    Some((false, _)) => "noText",
                    Some((true, true)) => "embedded",
                    Some((true, false)) => "pending",
                };
            let image_indexed = state
                .databases
                .open_images_read_only(state.image_embedder.dimensions())?
                .is_asset_indexed(asset.asset_id)?;
            let file = nicegal_core::metadata::inspect(&asset).into();
            Ok(MetadataResponse {
                asset: asset.into(),
                file,
                ocr_state,
                ocr_text,
                image_indexed,
                text_state,
                decode_failed,
            })
        })
        .await?,
    ))
}
