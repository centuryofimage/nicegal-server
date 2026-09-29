//! Per-patch image features for visualizations, returned as safetensors so a renderer can view
//! each tensor as a `Float32Array` over the response buffer without parsing numbers.
use anyhow::Context;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::{MethodRouter, post};
use nicegal_core::assets::MediaKind;
use nicegal_core::cancellation::SearchCancellation;
use nicegal_core::embedding::PatchFeatures;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::error::{ApiError, ErrorCode};
use super::extract::ApiJson;
use super::search::{self, ImageQueryRequest};
use super::{AppState, patch_cache, run_blocking};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PatchRequest {
    asset_id: i64,
    /// Resolved exactly as a `type=image` search resolves it and returned as `query`.
    image_query: Option<ImageQueryRequest>,
}

pub(super) fn route() -> MethodRouter<AppState> {
    post(patches)
}

pub(super) fn scores_route() -> MethodRouter<AppState> {
    post(patch_scores)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ScoresRequest {
    targets: Vec<ScoreTarget>,
    image_query: ImageQueryRequest,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ScoreTarget {
    asset_id: i64,
    timestamp_ms: Option<i64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScoresResponse {
    model: &'static str,
    results: Vec<ScoreResult>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ScoreResult {
    Match(ScoreMatch),
    Error {
        #[serde(rename = "assetId")]
        asset_id: i64,
        #[serde(rename = "timestampMs", skip_serializing_if = "Option::is_none")]
        timestamp_ms: Option<i64>,
        error: ErrorCode,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScoreMatch {
    asset_id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp_ms: Option<i64>,
    rows: usize,
    columns: usize,
    region: [f64; 4],
    pooled: f32,
    scores: Vec<f32>,
}

fn score_features(target: ScoreTarget, features: &PatchFeatures, query: &[f32]) -> ScoreResult {
    if features.dimensions != query.len() {
        return ScoreResult::Error {
            asset_id: target.asset_id,
            timestamp_ms: target.timestamp_ms,
            error: ErrorCode::InternalError,
        };
    }
    let dot = |vector: &[f32]| vector.iter().zip(query).map(|(a, b)| a * b).sum();
    ScoreResult::Match(ScoreMatch {
        asset_id: target.asset_id,
        timestamp_ms: target.timestamp_ms,
        rows: features.rows,
        columns: features.columns,
        region: features.region,
        pooled: dot(&features.embedding),
        scores: features
            .patches
            .chunks_exact(features.dimensions)
            .map(dot)
            .collect(),
    })
}

async fn patch_scores(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ScoresRequest>,
) -> Result<axum::Json<ScoresResponse>, ApiError> {
    if request.targets.len() > 32
        || request
            .targets
            .iter()
            .any(|target| target.asset_id <= 0 || target.timestamp_ms.is_some_and(|time| time < 0))
    {
        return Err(ApiError::bad_request(
            "targets must contain at most 32 positive asset IDs and nonnegative timestamps",
        ));
    }
    let response = run_blocking(move || {
        let model = state.image_embedder.ready_or_cached()?;
        if !model.supports_patch_features() {
            return Err(ApiError::bad_request(format!(
                "{} does not provide patch features",
                model.model()
            )));
        }
        let query = search::resolve_image_query(
            &state,
            request.image_query,
            &SearchCancellation::default(),
        )?;
        let assets: HashMap<_, _> = state
            .databases
            .open_assets_read_only()?
            .get_many(&request.targets.iter().map(|target| target.asset_id).collect::<Vec<_>>())?
            .into_iter()
            .map(|asset| (asset.asset_id, asset))
            .collect();
        let model_id = model.model().id();
        let mut results: Vec<Option<ScoreResult>> =
            (0..request.targets.len()).map(|_| None).collect();
        let mut missing = Vec::new();
        let thumbnails = if request.targets.iter().any(|target| target.timestamp_ms.is_some()) {
            Some(state.databases.open_thumbnails_read_only()?)
        } else {
            None
        };
        for (index, target) in request.targets.iter().copied().enumerate() {
            let asset_id = target.asset_id;
            let Some(asset) = assets.get(&asset_id) else {
                results[index] = Some(ScoreResult::Error {
                    asset_id,
                    timestamp_ms: target.timestamp_ms,
                    error: ErrorCode::AssetNotFound,
                });
                continue;
            };
            if (asset.media_kind == MediaKind::Video) != target.timestamp_ms.is_some() {
                results[index] = Some(ScoreResult::Error {
                    asset_id,
                    timestamp_ms: target.timestamp_ms,
                    error: ErrorCode::InvalidRequest,
                });
            } else if let Some(features) = patch_cache::get(
                model_id,
                asset_id,
                target.timestamp_ms,
                asset.fingerprint,
            ) {
                results[index] = Some(score_features(target, &features, &query));
            } else {
                let frame = if let Some(timestamp_ms) = target.timestamp_ms {
                    thumbnails.as_ref().unwrap().get_video_sample(
                        asset_id,
                        timestamp_ms,
                        256,
                        asset.fingerprint,
                    )?
                } else {
                    None
                };
                if target.timestamp_ms.is_some() && frame.is_none() {
                    results[index] = Some(ScoreResult::Error {
                        asset_id,
                        timestamp_ms: target.timestamp_ms,
                        error: ErrorCode::ThumbnailNotFound,
                    });
                } else {
                    missing.push((index, target, asset, frame.map(|frame| frame.data)));
                }
            }
        }
        let decoded: Vec<_> = missing
            .into_par_iter()
            .map(|(index, target, asset, frame)| {
                let bytes = frame.or_else(|| std::fs::read(&asset.path).ok());
                let raster = bytes.and_then(|bytes| model.decode_image(&bytes).ok());
                (index, target, asset, raster)
            })
            .collect();
        let mut ready = Vec::new();
        for (index, target, asset, raster) in decoded {
            if let Some(raster) = raster {
                ready.push((index, target, asset, raster));
            } else {
                results[index] = Some(ScoreResult::Error {
                    asset_id: asset.asset_id,
                    timestamp_ms: target.timestamp_ms,
                    error: ErrorCode::InternalError,
                });
            }
        }
        for chunk in ready.chunks(model.max_batch_size()) {
            let rasters = chunk.iter().map(|(_, _, _, raster)| raster.clone()).collect();
            match model.patch_features_batch(rasters) {
                Ok(features) => {
                    for ((index, target, asset, _), features) in chunk.iter().zip(features) {
                        let cached = patch_cache::insert(
                            model_id,
                            asset.asset_id,
                            target.timestamp_ms,
                            asset.fingerprint,
                            features,
                        );
                        results[*index] = Some(score_features(*target, &cached, &query));
                    }
                }
                Err(error) => {
                    tracing::warn!(error = %error, "patch batch inference failed");
                    // A bad image must not discard usable neighbors in the same batch.
                    for (index, target, asset, raster) in chunk {
                        results[*index] = Some(match model.patch_features(raster.clone()) {
                            Ok(features) => {
                                let cached = patch_cache::insert(model_id, asset.asset_id, target.timestamp_ms, asset.fingerprint, features);
                                score_features(*target, &cached, &query)
                            }
                            Err(error) => {
                                tracing::warn!(asset_id = asset.asset_id, error = %error, "patch inference failed");
                                ScoreResult::Error { asset_id: asset.asset_id, timestamp_ms: target.timestamp_ms, error: ErrorCode::InternalError }
                            }
                        });
                    }
                }
            }
        }
        Ok(ScoresResponse {
            model: model_id,
            results: results.into_iter().map(Option::unwrap).collect(),
        })
    })
    .await?;
    Ok(axum::Json(response))
}

async fn patches(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<PatchRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if request.asset_id <= 0 {
        return Err(ApiError::bad_request("assetId must be greater than zero"));
    }
    let body = run_blocking(move || {
        let model = state.image_embedder.ready_or_cached()?;
        if !model.supports_patch_features() {
            return Err(ApiError::bad_request(format!(
                "{} does not provide patch features",
                model.model()
            )));
        }
        let asset = state
            .databases
            .open_assets_read_only()?
            .get_many(&[request.asset_id])?
            .pop()
            .ok_or_else(|| ApiError::asset_id_not_found(request.asset_id))?;
        if asset.media_kind == MediaKind::Video {
            return Err(ApiError::bad_request(
                "patch features are available for still images only",
            ));
        }
        let query = request
            .image_query
            .map(|query| search::resolve_image_query(&state, query, &SearchCancellation::default()))
            .transpose()?;
        if let Some(features) = patch_cache::get(
            model.model().id(),
            request.asset_id,
            None,
            asset.fingerprint,
        ) {
            return safetensors(
                &features,
                query.as_deref(),
                model.model().id(),
                request.asset_id,
            )
            .map_err(ApiError::internal);
        }
        let bytes = std::fs::read(&asset.path)
            .with_context(|| format!("reading {}", asset.path))
            .map_err(ApiError::internal)?;
        let raster = model
            .decode_image(&bytes)
            .with_context(|| format!("decoding {}", asset.path))
            .map_err(ApiError::internal)?;
        let features = patch_cache::insert(
            model.model().id(),
            request.asset_id,
            None,
            asset.fingerprint,
            model.patch_features(raster).map_err(ApiError::internal)?,
        );
        safetensors(
            &features,
            query.as_deref(),
            model.model().id(),
            request.asset_id,
        )
        .map_err(ApiError::internal)
    })
    .await?;
    Ok(([(header::CONTENT_TYPE, "application/octet-stream")], body))
}

/// `u64` little-endian header length, a JSON header padded with spaces to 8-byte alignment, then
/// little-endian `F32` tensors: `patches [rows, columns, dimensions]`, `embedding [dimensions]`,
/// and `query [dimensions]` when requested.
fn safetensors(
    features: &PatchFeatures,
    query: Option<&[f32]>,
    model: &str,
    asset_id: i64,
) -> anyhow::Result<Vec<u8>> {
    let mut tensors = vec![
        (
            "patches",
            vec![features.rows, features.columns, features.dimensions],
            features.patches.as_slice(),
        ),
        (
            "embedding",
            vec![features.dimensions],
            features.embedding.as_slice(),
        ),
    ];
    if let Some(query) = query {
        tensors.push(("query", vec![query.len()], query));
    }
    let metadata = HashMap::from([
        ("model".to_owned(), model.to_owned()),
        ("assetId".to_owned(), asset_id.to_string()),
        ("method".to_owned(), features.method.to_owned()),
        (
            "region".to_owned(),
            serde_json::to_string(&features.region)?,
        ),
    ]);
    // Encode the scalar bytes explicitly so the wire format is portable to big-endian hosts.
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, values)| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        })
        .collect();
    let views = tensors
        .into_iter()
        .zip(&bytes)
        .map(|((name, shape, _), bytes)| {
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, bytes)
                .map(|view| (name, view))
        })
        .collect::<Result<Vec<_>, _>>()?;
    safetensors::tensor::serialize(views, Some(metadata)).context("serializing image patch tensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_response_has_row_major_cells_and_per_item_error() {
        let features = PatchFeatures {
            rows: 1,
            columns: 2,
            dimensions: 2,
            patches: vec![1.0, 0.0, 0.0, 1.0],
            embedding: vec![0.6, 0.8],
            region: [0.0, 0.25, 1.0, 0.5],
            method: "clearclip",
        };
        let response = ScoresResponse {
            model: "m",
            results: vec![
                score_features(
                    ScoreTarget {
                        asset_id: 7,
                        timestamp_ms: None,
                    },
                    &features,
                    &[0.0, 1.0],
                ),
                ScoreResult::Error {
                    asset_id: 8,
                    timestamp_ms: Some(2000),
                    error: ErrorCode::AssetNotFound,
                },
            ],
        };
        let json = serde_json::to_value(response).unwrap();
        assert!((json["results"][0]["pooled"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        let mut scored = json["results"][0].clone();
        scored.as_object_mut().unwrap().remove("pooled");
        assert_eq!(
            scored,
            serde_json::json!({
                "assetId": 7, "rows": 1, "columns": 2,
                "region": [0.0, 0.25, 1.0, 0.5], "scores": [0.0, 1.0]
            })
        );
        assert_eq!(
            json["results"][1],
            serde_json::json!({"assetId": 8, "timestampMs": 2000, "error": "asset_not_found"})
        );
    }

    #[test]
    fn safetensors_layout_is_aligned_and_self_describing() {
        let features = PatchFeatures {
            rows: 1,
            columns: 2,
            dimensions: 2,
            patches: vec![1.0, 0.0, 0.0, 1.0],
            embedding: vec![0.6, 0.8],
            region: [0.0, 0.25, 1.0, 0.5],
            method: "clearclip",
        };
        for query in [None, Some([0.0, -1.0].as_slice())] {
            let body = safetensors(&features, query, "m", 7).unwrap();
            let (header_length, metadata) = safetensors::SafeTensors::read_metadata(&body).unwrap();
            assert_eq!(header_length % 8, 0);
            let metadata = metadata.metadata().as_ref().unwrap();
            assert_eq!(metadata["assetId"], "7");
            assert_eq!(metadata["model"], "m");
            assert_eq!(metadata["method"], "clearclip");
            assert_eq!(metadata["region"], "[0.0,0.25,1.0,0.5]");
            let tensors = safetensors::SafeTensors::deserialize(&body).unwrap();
            assert_eq!(tensors.len(), if query.is_some() { 3 } else { 2 });
            for (name, shape, expected) in [
                ("patches", vec![1, 2, 2], features.patches.as_slice()),
                ("embedding", vec![2], features.embedding.as_slice()),
            ]
            .into_iter()
            .chain(query.map(|values| ("query", vec![2], values)))
            {
                let tensor = tensors.tensor(name).unwrap();
                assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
                assert_eq!(tensor.shape(), shape);
                let values: Vec<f32> = tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect();
                assert_eq!(values, expected);
            }
        }
    }
}
