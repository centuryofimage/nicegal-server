//! Extractors that reject with the API's error envelope.
//!
//! Axum's own `Query`, `Json`, and `Bytes` rejections answer with a plain-text body, which the
//! desktop client cannot parse and which previously leaked out of the JSON contract for any
//! malformed request. These wrappers are the only extractors the routes use.

use axum::body::Bytes;
use axum::extract::{FromRequest, FromRequestParts, Json, Query, Request};
use axum::http::request::Parts;
use serde::de::DeserializeOwned;

use super::error::ApiError;

/// Query-string parameters, rejecting with `400 invalid_request`.
#[derive(Debug)]
pub(super) struct ApiQuery<T>(pub(super) T);

impl<T, S> FromRequestParts<S> for ApiQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let Query(value) = Query::<T>::from_request_parts(parts, state).await?;
        Ok(Self(value))
    }
}

/// A JSON body, rejecting with the envelope while keeping axum's status for an unsupported media
/// type or an oversized body.
#[derive(Debug)]
pub(super) struct ApiJson<T>(pub(super) T);

impl<T, S> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(request, state).await?;
        Ok(Self(value))
    }
}

/// A raw body, used for thumbnail uploads.
#[derive(Debug)]
pub(super) struct ApiBytes(pub(super) Bytes);

impl<S> FromRequest<S> for ApiBytes
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        Ok(Self(Bytes::from_request(request, state).await?))
    }
}
