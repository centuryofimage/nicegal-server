//! Library definitions: named-by-folder groups of included and excluded directories.
//!
//! A library's scope is evaluated against catalog paths on every read, so these endpoints only
//! write definitions. Folders that gain visible files are marked `scanPending`; the client
//! requests the scan when it wants the library brought up to date.

use std::io::ErrorKind;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::assets::AssetCatalog;
use nicegal_core::libraries::{Created, Library, LibraryDefinition, LibraryOptions, ScanOutcome};
use nicegal_core::scope::PathScope;
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::extract::ApiJson;
use super::{AppState, Databases, roots, run_blocking};

/// A library a job was started for. It existed when the job was accepted, so its absence here
/// means it was deleted since.
pub(super) fn stored(catalog: &AssetCatalog, library_id: i64) -> anyhow::Result<Library> {
    catalog
        .library(library_id)?
        .ok_or_else(|| anyhow::anyhow!("library {library_id} was deleted"))
}

/// The catalog rows a read of `library_id` covers. Reads only the stored definition, never the
/// filesystem, so a library whose folders are offline still answers from its cached index.
pub(super) fn scope(databases: &Databases, library_id: i64) -> Result<PathScope, ApiError> {
    Ok(databases
        .open_assets_read_only()?
        .library(library_id)?
        .ok_or_else(|| ApiError::library_not_found(library_id))?
        .scope())
}

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/libraries", get(list).post(create))
        .route(
            "/v1/libraries/{library_id}",
            get(read).put(update).delete(remove),
        )
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CreateRequest {
    include: Vec<PathBuf>,
    #[serde(default)]
    exclude: Vec<PathBuf>,
    #[serde(default)]
    ocr: Option<bool>,
    #[serde(default)]
    image: Option<bool>,
    #[serde(default)]
    videos: Option<bool>,
    /// Makes the request idempotent: a repeat returns the library the first request created.
    /// Import requests may also name folders that are currently unavailable.
    #[serde(default)]
    import_key: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct UpdateRequest {
    include: Vec<PathBuf>,
    #[serde(default)]
    exclude: Vec<PathBuf>,
    ocr: bool,
    image: bool,
    videos: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LibraryResponse {
    id: i64,
    include: Vec<FolderResponse>,
    exclude: Vec<PathBuf>,
    ocr: bool,
    image: bool,
    videos: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FolderResponse {
    path: PathBuf,
    scan_pending: bool,
    /// Why the latest scan attempt stopped short: `unavailable`, `incomplete`, `cancelled` or
    /// `failed`. `scan_error` is its human-readable detail.
    scan_outcome: Option<&'static str>,
    scan_error: Option<String>,
    /// Decimal Unix nanoseconds, like the catalog's timestamps.
    last_scan_completed_ns: Option<String>,
}

impl From<Library> for LibraryResponse {
    fn from(library: Library) -> Self {
        Self {
            id: library.id,
            include: library
                .include
                .into_iter()
                .map(|folder| FolderResponse {
                    path: folder.path,
                    scan_pending: folder.scan_pending,
                    scan_outcome: folder.scan_outcome.map(ScanOutcome::as_str),
                    scan_error: folder.scan_error,
                    last_scan_completed_ns: folder.last_scan_completed_ns.map(|ns| ns.to_string()),
                })
                .collect(),
            exclude: library.exclude,
            ocr: library.options.ocr,
            image: library.options.image,
            videos: library.options.videos,
        }
    }
}

async fn list(State(state): State<AppState>) -> Result<Json<Vec<LibraryResponse>>, ApiError> {
    let libraries = run_blocking(move || {
        Ok(state
            .databases
            .open_assets_read_only()?
            .libraries()?
            .into_iter()
            .map(LibraryResponse::from)
            .collect())
    })
    .await?;
    Ok(Json(libraries))
}

async fn read(
    State(state): State<AppState>,
    Path(library_id): Path<i64>,
) -> Result<Json<LibraryResponse>, ApiError> {
    let library = run_blocking(move || {
        state
            .databases
            .open_assets_read_only()?
            .library(library_id)?
            .ok_or_else(|| ApiError::library_not_found(library_id))
    })
    .await?;
    Ok(Json(library.into()))
}

async fn create(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<CreateRequest>,
) -> Result<(StatusCode, Json<LibraryResponse>), ApiError> {
    run_blocking(move || {
        let defaults = LibraryOptions::default();
        let allow_missing = request.import_key.is_some();
        let definition = LibraryDefinition {
            include: resolve_folders(request.include, &[], allow_missing)?,
            exclude: resolve_folders(request.exclude, &[], allow_missing)?,
            options: LibraryOptions {
                ocr: request.ocr.unwrap_or(defaults.ocr),
                image: request.image.unwrap_or(defaults.image),
                videos: request.videos.unwrap_or(defaults.videos),
            },
        };
        definition
            .validate()
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let mut catalog = AssetCatalog::new(&state.databases.assets)?;
        Ok(
            match catalog.create_library(&definition, request.import_key.as_deref())? {
                Created::New(library) => (StatusCode::CREATED, Json(library.into())),
                Created::Existing(library) => (StatusCode::OK, Json(library.into())),
            },
        )
    })
    .await
}

/// Replace a library's definition. There is no edit conflict check: the server has exactly one
/// client, the app's UI, so the last write is the intended one.
async fn update(
    State(state): State<AppState>,
    Path(library_id): Path<i64>,
    ApiJson(request): ApiJson<UpdateRequest>,
) -> Result<Json<LibraryResponse>, ApiError> {
    run_blocking(move || {
        let mut catalog = AssetCatalog::new(&state.databases.assets)?;
        let current = catalog
            .library(library_id)?
            .ok_or_else(|| ApiError::library_not_found(library_id))?;
        let known = current.definition();
        let definition = LibraryDefinition {
            include: resolve_folders(request.include, &known.include, false)?,
            exclude: resolve_folders(request.exclude, &known.exclude, false)?,
            options: LibraryOptions {
                ocr: request.ocr,
                image: request.image,
                videos: request.videos,
            },
        };
        definition
            .validate()
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        catalog
            .update_library(library_id, &definition)?
            .map(|library| Json(library.into()))
            .ok_or_else(|| ApiError::library_not_found(library_id))
    })
    .await
}

async fn remove(
    State(state): State<AppState>,
    Path(library_id): Path<i64>,
) -> Result<StatusCode, ApiError> {
    run_blocking(move || {
        if AssetCatalog::new(&state.databases.assets)?.delete_library(library_id)? {
            Ok(StatusCode::NO_CONTENT)
        } else {
            Err(ApiError::library_not_found(library_id))
        }
    })
    .await
}

/// Canonicalize requested folders. A folder the library already has keeps its stored spelling
/// without touching the filesystem, so a library on an offline drive stays editable. With
/// `allow_missing`, a folder that does not exist is kept with [`offline_spelling`], which lets an
/// import preserve a library whose drive is disconnected; its first scan once the drive is back
/// replaces that spelling with the canonical one.
fn resolve_folders(
    requested: Vec<PathBuf>,
    known: &[PathBuf],
    allow_missing: bool,
) -> Result<Vec<PathBuf>, ApiError> {
    requested
        .into_iter()
        .map(|path| {
            if !path.is_absolute() {
                return Err(ApiError::invalid_root(format!(
                    "library folder must be absolute: {path}"
                )));
            }
            if known.contains(&path) {
                return Ok(path);
            }
            if allow_missing
                && std::fs::metadata(&path).is_err_and(|error| error.kind() == ErrorKind::NotFound)
            {
                return Ok(offline_spelling(&path));
            }
            roots::resolve_root("library folder", &path)
        })
        .collect()
}

/// The spelling kept for a folder that cannot be canonicalized now: the platform's separators and
/// no trailing separator. Case and links cannot be resolved until the folder is reachable.
fn offline_spelling(path: &camino::Utf8Path) -> PathBuf {
    let spelled = if cfg!(windows) {
        path.as_str().replace('/', "\\")
    } else {
        path.as_str().to_owned()
    };
    let trimmed = spelled.trim_end_matches(['/', '\\']);
    // Keep a bare root such as `C:\` or `/` intact.
    if trimmed.is_empty() || trimmed.ends_with(':') {
        PathBuf::from(spelled)
    } else {
        PathBuf::from(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_folders_use_platform_separators_without_a_trailing_one() {
        if cfg!(windows) {
            assert_eq!(
                offline_spelling("D:/Photos/2024/".into()),
                "D:\\Photos\\2024"
            );
            assert_eq!(offline_spelling("D:\\".into()), "D:\\");
        } else {
            assert_eq!(offline_spelling("/media/photos/".into()), "/media/photos");
            assert_eq!(offline_spelling("/".into()), "/");
        }
    }
}
