//! Shared validation for the caller-supplied root directories that search, indexing, and pruning
//! all take.
//!
//! A root the caller can correct is never a server error: an unplugged drive, a typo, or a
//! relative path all answer `400 invalid_root` with a cause the UI can show.

use std::fs;

use camino::{Utf8Path, Utf8PathBuf as PathBuf};

use super::error::ApiError;

/// Validate `root` and return its canonical spelling. `purpose` names the root in the message
/// ("search", "index", "prune").
pub(super) fn resolve_root(purpose: &str, root: &Utf8Path) -> Result<PathBuf, ApiError> {
    if !root.is_absolute() {
        return Err(ApiError::invalid_root(format!(
            "{purpose} root must be absolute"
        )));
    }
    let metadata = fs::metadata(root)
        .map_err(|error| ApiError::invalid_root(format!("cannot read {purpose} root: {error}")))?;
    if !metadata.is_dir() {
        return Err(ApiError::invalid_root(format!(
            "{purpose} root must be a directory"
        )));
    }
    nicegal_core::assets::canonicalize_path(root)
        .map_err(|error| ApiError::invalid_root(format!("cannot resolve {purpose} root: {error}")))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::super::error::ErrorCode;
    use super::*;

    fn crate_dir() -> PathBuf {
        PathBuf::try_from(std::env::current_dir().unwrap()).unwrap()
    }

    #[test]
    fn unusable_roots_are_invalid_root_rather_than_internal_errors() {
        let relative = resolve_root("search", Utf8Path::new("gallery")).expect_err("relative");
        assert_eq!(relative.status, StatusCode::BAD_REQUEST);
        assert_eq!(relative.code, ErrorCode::InvalidRoot);
        assert!(relative.message.contains("search root"), "{relative:?}");

        let missing = resolve_root("index", &crate_dir().join("definitely-missing-root"))
            .expect_err("missing root");
        assert_eq!(missing.code, ErrorCode::InvalidRoot);
        assert!(
            missing.message.contains("cannot read index root"),
            "{missing:?}"
        );

        let file = resolve_root("prune", &crate_dir().join("Cargo.toml"))
            .expect_err("a file is not a directory");
        assert_eq!(file.code, ErrorCode::InvalidRoot);
        assert!(file.message.contains("must be a directory"), "{file:?}");
    }

    #[test]
    fn a_real_directory_resolves_to_an_absolute_canonical_path() {
        let resolved = resolve_root("search", &crate_dir()).expect("the crate directory is a root");
        assert!(resolved.is_absolute());
    }
}
