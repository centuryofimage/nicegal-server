//! Retire only known superseded encoder files after their replacement session loads.
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

fn plain_directory(path: &Path) -> Result<()> {
    if !std::fs::symlink_metadata(path)?.file_type().is_dir() {
        bail!("cache directory is a link or is not a directory");
    }
    Ok(())
}

fn referenced_files(
    directory: &Path,
    retired: &HashSet<PathBuf>,
    output: &mut HashSet<PathBuf>,
) -> Result<()> {
    plain_directory(directory)?;
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if retired.contains(&entry.path()) {
            continue;
        }
        if entry.file_type()?.is_dir() {
            referenced_files(&entry.path(), retired, output)?;
        } else if let Ok(target) = entry.path().canonicalize() {
            output.insert(target);
        } else {
            // An unreadable entry makes proving an unreferenced blob impossible.
            bail!("cannot resolve cached file {}", entry.path().display());
        }
    }
    Ok(())
}

pub(crate) fn retire_files(
    cache: &Path,
    repo: &str,
    revision: &str,
    filenames: &[&str],
) -> Result<()> {
    if !revision.bytes().all(|b| b.is_ascii_hexdigit())
        || revision.len() != 40
        || repo.split('/').count() != 2
        || !repo
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"/-_.".contains(&b))
        || repo
            .split('/')
            .any(|s| s.is_empty() || s == "." || s == "..")
        || filenames
            .iter()
            .any(|s| s.contains(['/', '\\', ':']) || *s == "." || *s == "..")
    {
        bail!("invalid retired export path");
    }
    let root = cache.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = root.join("snapshots");
    let old = snapshots.join(revision);
    if !old.exists() {
        return Ok(());
    }
    for path in [&root, &snapshots, &old] {
        plain_directory(path)?;
    }
    let blobs = root.join("blobs");
    plain_directory(&blobs)?;
    let canonical_blobs = blobs.canonicalize()?;
    let mut candidates = HashSet::new();
    let mut retired = HashSet::new();
    for filename in filenames {
        let path = old.join(filename);
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            bail!("retired encoder is a directory");
        }
        if let Ok(target) = path.canonicalize()
            && target.parent() == Some(canonical_blobs.as_path())
        {
            candidates.insert(target);
        }
        retired.insert(path);
    }
    // Preserve any blob referenced by another snapshot, including derived graph caches.
    let mut referenced = HashSet::new();
    referenced_files(&snapshots, &retired, &mut referenced)?;
    for blob in candidates.difference(&referenced) {
        std::fs::remove_file(blob).with_context(|| format!("retiring blob {}", blob.display()))?;
    }
    // Keep the snapshot links if a mapped blob could not be removed, so a later load
    // can retry rather than leaving an unreachable multi-GB orphan in blobs/.
    for path in retired {
        std::fs::remove_file(&path).with_context(|| format!("retiring {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal() {
        assert!(retire_files(Path::new("unused"), "a/b", "../bad", &["image.onnx"]).is_err());
        assert!(
            retire_files(
                Path::new("unused"),
                "../b",
                &"a".repeat(40),
                &["image.onnx"]
            )
            .is_err()
        );
        assert!(
            retire_files(
                Path::new("unused"),
                "a/b",
                &"a".repeat(40),
                &["../image.onnx"]
            )
            .is_err()
        );
    }

    #[test]
    fn preserves_shared_blobs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("models--a--b");
        let old = root.join("snapshots").join("a".repeat(40));
        let new = root.join("snapshots").join("b".repeat(40));
        std::fs::create_dir_all(&old)?;
        std::fs::create_dir_all(&new)?;
        std::fs::create_dir(root.join("blobs"))?;
        for name in ["shared", "unused"] {
            std::fs::write(root.join("blobs").join(name), name)?;
        }
        fn link(source: &Path, target: &Path) -> std::io::Result<()> {
            #[cfg(windows)]
            {
                std::os::windows::fs::symlink_file(source, target)
            }
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(source, target)
            }
        }
        link(&root.join("blobs/shared"), &old.join("text.onnx"))?;
        link(&root.join("blobs/shared"), &new.join("text.onnx"))?;
        link(&root.join("blobs/unused"), &old.join("image.onnx"))?;
        retire_files(
            temp.path(),
            "a/b",
            &"a".repeat(40),
            &["image.onnx", "text.onnx"],
        )?;
        assert!(new.join("text.onnx").is_file());
        assert!(root.join("blobs/shared").is_file());
        assert!(!root.join("blobs/unused").exists());
        assert!(!old.join("image.onnx").exists());
        Ok(())
    }
}
