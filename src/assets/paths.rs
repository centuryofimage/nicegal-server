use std::fs;

use anyhow::{Context, Result};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};

/// Resolve a filesystem path while retaining the normal Windows path spelling used by clients.
pub fn canonicalize_path(path: &Path) -> Result<PathBuf> {
    let path = PathBuf::try_from(
        fs::canonicalize(path).with_context(|| format!("canonicalizing path: {path}"))?,
    )
    .with_context(|| format!("canonical path is not valid UTF-8: {path}"))?;
    Ok(normalize_windows_verbatim_path(path))
}

// `fs::canonicalize` returns verbatim paths on Windows; strip that transport-only prefix so paths
// use the spelling accepted and returned by the API.
#[cfg(windows)]
fn normalize_windows_verbatim_path(path: PathBuf) -> PathBuf {
    let path = path.as_str();
    if let Some(remainder) = path
        .strip_prefix("//?/UNC/")
        .or_else(|| path.strip_prefix(r"\\?\UNC\"))
    {
        return PathBuf::from(format!("//{remainder}"));
    }
    if let Some(remainder) = path
        .strip_prefix("//?/")
        .or_else(|| path.strip_prefix(r"\\?\"))
    {
        return PathBuf::from(remainder);
    }
    PathBuf::from(path)
}

#[cfg(not(windows))]
fn normalize_windows_verbatim_path(path: PathBuf) -> PathBuf {
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn normalizes_windows_verbatim_paths() {
        assert_eq!(
            normalize_windows_verbatim_path(PathBuf::from("//?/C:/gallery/image.png")),
            PathBuf::from("C:/gallery/image.png")
        );
        assert_eq!(
            normalize_windows_verbatim_path(PathBuf::from("//?/UNC/server/share/image.png")),
            PathBuf::from("//server/share/image.png")
        );
    }
}
