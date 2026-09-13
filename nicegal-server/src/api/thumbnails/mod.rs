use std::fs;
use std::io;

use anyhow::Context;
use axum::Json;
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, get as get_route, post};
use nicegal_core::assets::{AssetCatalog, MediaKind, SourceFingerprint};
use nicegal_core::thumbs::{
    DecodedThumbnail, GENERATOR_VERSION, SIZE_BUCKETS, Thumbnail, ThumbnailEncoding,
    validate_static_thumbnail,
};
use serde::{Deserialize, Serialize};
use tracing::{Instrument, debug_span, field};

use super::error::ApiError;
use super::extract::{ApiBytes, ApiJson, ApiQuery};
use super::jobs::{JobResponse, JobSpec};
use super::{AppState, Databases, run_blocking};

pub(super) mod job;

const WIDTH_HEADER: HeaderName = HeaderName::from_static("x-nicegal-server-thumbnail-width");
const HEIGHT_HEADER: HeaderName = HeaderName::from_static("x-nicegal-server-thumbnail-height");
const ASSET_ID_HEADER: HeaderName = HeaderName::from_static("x-nicegal-server-asset-id");
const SIZE_BUCKET_HEADER: HeaderName = HeaderName::from_static("x-nicegal-server-size-bucket");
const GENERATOR_VERSION_HEADER: HeaderName =
    HeaderName::from_static("x-nicegal-server-generator-version");

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThumbnailLookup {
    asset_id: i64,
    requested_size: u32,
    generator_version: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ThumbnailId {
    asset_id: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EnsureThumbnails {
    asset_ids: Vec<i64>,
    required_size: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EnsuredThumbnails {
    asset_ids: Vec<i64>,
    required_size: u32,
    size_bucket: u16,
    generator_version: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PutThumbnail {
    asset_id: i64,
    size_bucket: u16,
    generator_version: u32,
    width: u32,
    height: u32,
    encoding: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ThumbnailReference {
    asset_id: i64,
    size_bucket: u16,
    generator_version: u32,
}

pub(super) fn route() -> MethodRouter<AppState> {
    get_route(get).post(ensure).put(put).delete(delete)
}

pub(super) fn generate_route() -> MethodRouter<AppState> {
    post(create_job)
}

async fn create_job(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<job::Request>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    let spec = JobSpec::ThumbnailGenerate(job::prepare(request)?);
    let job = state.jobs.start(spec)?;
    Ok((StatusCode::ACCEPTED, Json(job.response())))
}

/// Ensure a visible set of thumbnails exists before the client reads the SQLite data plane.
/// This is intentionally synchronous: a successful response is a durability acknowledgement,
/// not a queued job reference.
async fn ensure(
    State(state): State<AppState>,
    ApiJson(mut request): ApiJson<EnsureThumbnails>,
) -> Result<Json<EnsuredThumbnails>, ApiError> {
    if request.asset_ids.is_empty() {
        return Err(ApiError::bad_request(
            "at least one asset identifier is required",
        ));
    }
    let requested_assets = request.asset_ids.len();
    let size_bucket = bucket_for_required_size(request.required_size)?;
    request.asset_ids.sort_unstable();
    request.asset_ids.dedup();
    for asset_id in &request.asset_ids {
        validate_asset_id(*asset_id)?;
    }
    let asset_ids = request.asset_ids.clone();
    let databases = state.databases;
    let thumbnails = state.thumbnails;
    let span = debug_span!(
        "thumbnail_ensure",
        requested_assets,
        assets = asset_ids.len(),
        required_size = request.required_size,
        size_bucket,
        generator_version = GENERATOR_VERSION,
    );
    run_blocking(move || {
        let assets = {
            let span = debug_span!(
                "thumbnail_asset_load",
                assets = asset_ids.len(),
                source_bytes = field::Empty,
            );
            let _entered = span.enter();
            let catalog = databases.open_assets_read_only()?;
            let mut assets = Vec::with_capacity(asset_ids.len());
            let mut source_bytes = 0u64;
            for asset_id in &asset_ids {
                let mut asset = catalog
                    .get(*asset_id)
                    .context("looking up asset for thumbnail generation")?
                    .ok_or_else(|| ApiError::asset_id_not_found(*asset_id))?;
                if asset.media_kind != MediaKind::Image {
                    return Err(ApiError::bad_request(format!(
                        "asset {asset_id} is not an image"
                    )));
                }
                let metadata = fs::metadata(&asset.path).map_err(|error| {
                    if error.kind() == io::ErrorKind::NotFound {
                        ApiError::asset_id_not_found(*asset_id)
                    } else {
                        ApiError::internal(
                            anyhow::Error::from(error)
                                .context(format!("reading source metadata for asset {asset_id}")),
                        )
                    }
                })?;
                asset.fingerprint = SourceFingerprint::from_metadata(&metadata)?;
                source_bytes = source_bytes.saturating_add(asset.fingerprint.size);
                assets.push(asset);
            }
            span.record("source_bytes", source_bytes);
            assets
        };
        for outcome in
            thumbnails.generate(&assets, &[size_bucket], GENERATOR_VERSION, false, || false)
        {
            outcome.result.with_context(|| {
                format!("generating thumbnail for asset {}", outcome.asset.asset_id)
            })?;
        }
        Ok(())
    })
    .instrument(span)
    .await?;
    Ok(Json(EnsuredThumbnails {
        asset_ids: request.asset_ids,
        required_size: request.required_size,
        size_bucket,
        generator_version: GENERATOR_VERSION,
    }))
}

async fn put(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<PutThumbnail>,
    ApiBytes(body): ApiBytes,
) -> Result<Json<ThumbnailReference>, ApiError> {
    validate_asset_id(request.asset_id)?;
    validate_bucket(request.size_bucket)?;
    validate_generator_version(request.generator_version)?;
    if request.width == 0 || request.height == 0 {
        return Err(ApiError::bad_request(
            "thumbnail width and height must be greater than zero",
        ));
    }
    if body.is_empty() {
        return Err(ApiError::bad_request("thumbnail body must not be empty"));
    }
    let encoding = ThumbnailEncoding::parse(&request.encoding)
        .map_err(|error| ApiError::bad_request(error.to_string()))?;

    let asset_id = request.asset_id;
    let databases = state.databases;
    let thumbnails = state.thumbnails;
    run_blocking(move || {
        // Decoding the submitted bytes is the caller's problem to fix, so it stays a 400 even
        // though it runs alongside the storage work.
        validate_static_thumbnail(
            request.size_bucket,
            request.width,
            request.height,
            encoding,
            &body,
        )
        .map_err(|error| ApiError::bad_request(format!("{error:#}")))?;

        let fingerprint = current_fingerprint(&databases, asset_id)?
            .ok_or_else(|| ApiError::asset_id_not_found(asset_id))?;
        thumbnails
            .put(DecodedThumbnail {
                asset_id,
                fingerprint,
                size_bucket: request.size_bucket,
                generator_version: request.generator_version,
                width: request.width,
                height: request.height,
                encoding,
                data: body.to_vec(),
            })
            .context("storing submitted thumbnail")?;
        Ok(())
    })
    .await?;

    Ok(Json(ThumbnailReference {
        asset_id,
        size_bucket: request.size_bucket,
        generator_version: request.generator_version,
    }))
}

async fn get(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<ThumbnailLookup>,
) -> Result<Response, ApiError> {
    validate_asset_id(request.asset_id)?;
    if request.requested_size == 0 {
        return Err(ApiError::bad_request(
            "requested physical size must be greater than zero",
        ));
    }
    validate_generator_version(request.generator_version)?;
    let asset_id = request.asset_id;
    let databases = state.databases;
    let thumbnail = run_blocking(move || {
        let Some(fingerprint) = current_fingerprint(&databases, asset_id)? else {
            return Ok(None);
        };
        let db = databases.open_thumbnails_read_only()?;
        Ok(db
            .get(
                asset_id,
                request.requested_size,
                request.generator_version,
                fingerprint,
            )
            .context("reading stored thumbnail")?)
    })
    .await?
    .ok_or_else(ApiError::thumbnail_not_found)?;

    Ok(thumbnail_response(asset_id, thumbnail))
}

async fn delete(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<ThumbnailId>,
) -> Result<StatusCode, ApiError> {
    validate_asset_id(request.asset_id)?;
    let thumbnails = state.thumbnails;
    run_blocking(move || {
        thumbnails
            .delete_asset(request.asset_id)
            .context("deleting stored thumbnails")?;
        Ok(())
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// The current on-disk fingerprint of an asset's source, or `None` when the catalog or the file
/// system no longer has it. Stale variants must never be served, so both misses look alike here.
fn current_fingerprint(
    databases: &Databases,
    asset_id: i64,
) -> Result<Option<SourceFingerprint>, ApiError> {
    let catalog: AssetCatalog = databases.open_assets_read_only()?;
    let Some(asset) = catalog
        .get(asset_id)
        .context("looking up asset for thumbnail request")?
    else {
        return Ok(None);
    };
    let metadata = match fs::metadata(&asset.path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(anyhow::Error::from(error)
                .context(format!("reading source metadata for asset {asset_id}"))
                .into());
        }
    };
    Ok(Some(SourceFingerprint::from_metadata(&metadata)?))
}

fn thumbnail_response(asset_id: i64, thumbnail: Thumbnail) -> Response {
    let content_type = thumbnail.encoding.content_type();
    let mut response = thumbnail.data.into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    insert_integer_header(response.headers_mut(), ASSET_ID_HEADER, asset_id);
    insert_integer_header(
        response.headers_mut(),
        SIZE_BUCKET_HEADER,
        thumbnail.size_bucket,
    );
    insert_integer_header(
        response.headers_mut(),
        GENERATOR_VERSION_HEADER,
        thumbnail.generator_version,
    );
    insert_integer_header(response.headers_mut(), WIDTH_HEADER, thumbnail.width);
    insert_integer_header(response.headers_mut(), HEIGHT_HEADER, thumbnail.height);
    response
}

fn insert_integer_header(
    headers: &mut axum::http::HeaderMap,
    name: HeaderName,
    value: impl ToString,
) {
    headers.insert(
        name,
        HeaderValue::from_str(&value.to_string()).expect("integer is a valid header value"),
    );
}

/// The smallest fixed bucket that covers the caller's physical pixel size.
fn bucket_for_required_size(required_size: u32) -> Result<u16, ApiError> {
    let largest = *SIZE_BUCKETS
        .last()
        .expect("the bucket list is a non-empty constant");
    if required_size == 0 || required_size > u32::from(largest) {
        return Err(ApiError::bad_request(format!(
            "required physical size must be between 1 and {largest} pixels"
        )));
    }
    Ok(*SIZE_BUCKETS
        .iter()
        .find(|bucket| u32::from(**bucket) >= required_size)
        .expect("required size is bounded by the largest thumbnail bucket"))
}

fn validate_asset_id(asset_id: i64) -> Result<(), ApiError> {
    if asset_id <= 0 {
        return Err(ApiError::bad_request(
            "asset identifier must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_bucket(size_bucket: u16) -> Result<(), ApiError> {
    if !SIZE_BUCKETS.contains(&size_bucket) {
        return Err(ApiError::bad_request(
            "size bucket must be one of 128, 256, 512, or 1024",
        ));
    }
    Ok(())
}

fn validate_generator_version(generator_version: u32) -> Result<(), ApiError> {
    if generator_version == 0 {
        return Err(ApiError::bad_request(
            "generator version must be greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::error::ErrorCode;
    use super::*;

    #[test]
    fn required_size_selects_the_smallest_adequate_bucket() {
        assert_eq!(bucket_for_required_size(1).unwrap(), 128);
        assert_eq!(bucket_for_required_size(128).unwrap(), 128);
        assert_eq!(bucket_for_required_size(129).unwrap(), 256);
        assert_eq!(bucket_for_required_size(1024).unwrap(), 1024);
    }

    #[test]
    fn out_of_range_required_size_is_rejected_with_the_documented_bounds() {
        for size in [0, 1025] {
            let error = bucket_for_required_size(size).expect_err("out of range");
            assert_eq!(error.code, ErrorCode::InvalidRequest);
            assert!(error.message.contains("between 1 and 1024"), "{error:?}");
        }
    }

    #[test]
    fn key_parameters_are_validated_before_any_database_work() {
        assert!(validate_asset_id(1).is_ok());
        for asset_id in [0, -1] {
            assert_eq!(
                validate_asset_id(asset_id).unwrap_err().code,
                ErrorCode::InvalidRequest
            );
        }
        assert!(validate_bucket(256).is_ok());
        assert_eq!(
            validate_bucket(300).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
        assert!(validate_generator_version(1).is_ok());
        assert_eq!(
            validate_generator_version(0).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
    }
}
