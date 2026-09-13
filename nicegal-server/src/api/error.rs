//! The single error envelope every route answers with.
//!
//! The contract is deliberately narrow: `4xx` means the caller can fix the request, `5xx` means
//! the server or its databases are broken. `code` carries the machine-readable distinction and
//! `message` is the displayable string the desktop UI shows nearly verbatim, so it never contains
//! a stack trace or a path the caller did not supply.

use axum::Json;
use axum::extract::rejection::{BytesRejection, JsonRejection, QueryRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorCode {
    /// A rejected parameter, body, or path component.
    InvalidRequest,
    /// A root that is missing, relative, or not a directory.
    InvalidRoot,
    /// SQLite could not parse the caller's search query.
    QuerySyntax,
    Unauthorized,
    NotFound,
    AssetNotFound,
    ThumbnailNotFound,
    JobNotFound,
    JobBusy,
    OcrModelsNotLoaded,
    ModelsNotReady,
    MethodNotAllowed,
    PayloadTooLarge,
    UnsupportedMediaType,
    ShuttingDown,
    InternalError,
}

impl ErrorCode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::InvalidRoot => "invalid_root",
            Self::QuerySyntax => "query_syntax",
            Self::Unauthorized => "unauthorized",
            Self::NotFound => "not_found",
            Self::AssetNotFound => "asset_not_found",
            Self::ThumbnailNotFound => "thumbnail_not_found",
            Self::JobNotFound => "job_not_found",
            Self::JobBusy => "job_busy",
            Self::ModelsNotReady => "models_not_ready",
            Self::OcrModelsNotLoaded => "ocr_models_not_loaded",
            Self::MethodNotAllowed => "method_not_allowed",
            Self::PayloadTooLarge => "payload_too_large",
            Self::UnsupportedMediaType => "unsupported_media_type",
            Self::ShuttingDown => "shutting_down",
            Self::InternalError => "internal_error",
        }
    }
}

/// The wire form is written by hand so [`ErrorCode::as_str`] stays the one place these strings
/// are defined; the desktop client matches on them.
impl Serialize for ErrorCode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    pub(crate) error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorDetail {
    pub(crate) code: ErrorCode,
    pub(crate) message: String,
    #[serde(rename = "componentIndex", skip_serializing_if = "Option::is_none")]
    pub(crate) component_index: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) code: ErrorCode,
    pub(crate) message: String,
    pub(crate) component_index: Option<usize>,
}

impl ApiError {
    fn new(status: StatusCode, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            component_index: None,
        }
    }

    pub(crate) fn models_not_ready(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, ErrorCode::ModelsNotReady, message)
    }

    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, ErrorCode::InvalidRequest, message)
    }

    /// A root the caller can correct: relative, missing, unreadable, or not a directory.
    pub(crate) fn invalid_root(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, ErrorCode::InvalidRoot, message)
    }

    /// SQLite's own text for a query it could not parse. Cryptic but honest, and the only way the
    /// UI can tell the user which part of their query is wrong.
    pub(crate) fn query_syntax(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, ErrorCode::QuerySyntax, message)
    }

    pub(crate) fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            ErrorCode::Unauthorized,
            "a valid bearer token is required",
        )
    }

    pub(crate) fn route_not_found() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorCode::NotFound,
            "no such route in this API version",
        )
    }

    pub(crate) fn method_not_allowed() -> Self {
        Self::new(
            StatusCode::METHOD_NOT_ALLOWED,
            ErrorCode::MethodNotAllowed,
            "the route does not support this method",
        )
    }

    pub(crate) fn thumbnail_not_found() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorCode::ThumbnailNotFound,
            "no current thumbnail exists for the asset",
        )
    }

    pub(crate) fn asset_not_found() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorCode::AssetNotFound,
            "the asset catalog contains no matching path",
        )
    }

    pub(crate) fn asset_id_not_found(asset_id: i64) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorCode::AssetNotFound,
            format!("asset {asset_id} is not in the catalog"),
        )
    }

    pub(crate) fn job_not_found() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            ErrorCode::JobNotFound,
            "the job does not exist or is no longer retained",
        )
    }

    pub(crate) fn job_busy() -> Self {
        Self::new(
            StatusCode::CONFLICT,
            ErrorCode::JobBusy,
            "another resource-intensive job is already active",
        )
    }

    pub(crate) fn ocr_models_not_loaded() -> Self {
        Self::new(
            StatusCode::CONFLICT,
            ErrorCode::OcrModelsNotLoaded,
            "PaddleOCR models are not loaded; complete an ocrModelLoad job first",
        )
    }

    pub(crate) fn shutting_down() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorCode::ShuttingDown,
            "the server is shutting down and cannot start another job",
        )
    }

    /// The only path that produces a `5xx`. The cause is logged in full and deliberately does not
    /// reach the client.
    pub(crate) fn internal(error: anyhow::Error) -> Self {
        if nicegal_core::cancellation::is_cancellation(&error) {
            // The abandoned search has no client waiting for this response.
            tracing::debug!("search worker stopped after cancellation");
        } else {
            tracing::error!(error = %format_args!("{error:#}"), "request failed");
        }
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorCode::InternalError,
            "the request could not be completed",
        )
    }
}

/// Lets a handler's blocking work use `?` on ordinary [`anyhow`] results without deciding, at
/// every call site, that the failure is internal; a handler that knows better builds a `4xx`
/// itself.
impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self::internal(error)
    }
}

impl From<QueryRejection> for ApiError {
    fn from(rejection: QueryRejection) -> Self {
        Self::bad_request(rejection.body_text())
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self::new(
            rejection.status(),
            code_for_body_status(rejection.status()),
            rejection.body_text(),
        )
    }
}

impl From<BytesRejection> for ApiError {
    fn from(rejection: BytesRejection) -> Self {
        Self::new(
            rejection.status(),
            code_for_body_status(rejection.status()),
            rejection.body_text(),
        )
    }
}

fn code_for_body_status(status: StatusCode) -> ErrorCode {
    match status {
        StatusCode::PAYLOAD_TOO_LARGE => ErrorCode::PayloadTooLarge,
        StatusCode::UNSUPPORTED_MEDIA_TYPE => ErrorCode::UnsupportedMediaType,
        _ => ErrorCode::InvalidRequest,
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: self.code,
                    message: self.message,
                    component_index: self.component_index,
                },
            }),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_codes_are_stable() {
        // These strings are the client's contract; changing one is a breaking API change.
        let expected = [
            (ErrorCode::InvalidRequest, "invalid_request"),
            (ErrorCode::InvalidRoot, "invalid_root"),
            (ErrorCode::QuerySyntax, "query_syntax"),
            (ErrorCode::Unauthorized, "unauthorized"),
            (ErrorCode::NotFound, "not_found"),
            (ErrorCode::AssetNotFound, "asset_not_found"),
            (ErrorCode::ThumbnailNotFound, "thumbnail_not_found"),
            (ErrorCode::JobNotFound, "job_not_found"),
            (ErrorCode::JobBusy, "job_busy"),
            (ErrorCode::MethodNotAllowed, "method_not_allowed"),
            (ErrorCode::PayloadTooLarge, "payload_too_large"),
            (ErrorCode::UnsupportedMediaType, "unsupported_media_type"),
            (ErrorCode::ShuttingDown, "shutting_down"),
            (ErrorCode::InternalError, "internal_error"),
        ];
        for (code, wire) in expected {
            assert_eq!(code.as_str(), wire);
            assert_eq!(serde_json::to_string(&code).unwrap(), format!("\"{wire}\""));
        }
    }

    #[test]
    fn the_envelope_nests_the_code_and_message_under_error() {
        let body = serde_json::to_value(ErrorResponse {
            error: ErrorDetail {
                code: ErrorCode::QuerySyntax,
                component_index: None,
                message: "fts5: syntax error near \"AND\"".to_owned(),
            },
        })
        .unwrap();
        assert_eq!(body["error"]["code"], "query_syntax");
        assert_eq!(body["error"]["message"], "fts5: syntax error near \"AND\"");
    }

    #[test]
    fn user_fixable_causes_never_produce_a_server_error() {
        for error in [
            ApiError::bad_request("bad"),
            ApiError::invalid_root("bad root"),
            ApiError::query_syntax("fts5: syntax error"),
            ApiError::asset_not_found(),
            ApiError::thumbnail_not_found(),
            ApiError::job_not_found(),
            ApiError::job_busy(),
        ] {
            assert!(error.status.is_client_error(), "{error:?}");
        }
    }

    #[test]
    fn oversized_and_mistyped_bodies_keep_their_own_status() {
        assert_eq!(
            code_for_body_status(StatusCode::PAYLOAD_TOO_LARGE),
            ErrorCode::PayloadTooLarge
        );
        assert_eq!(
            code_for_body_status(StatusCode::UNSUPPORTED_MEDIA_TYPE),
            ErrorCode::UnsupportedMediaType
        );
        assert_eq!(
            code_for_body_status(StatusCode::UNPROCESSABLE_ENTITY),
            ErrorCode::InvalidRequest
        );
    }
}
