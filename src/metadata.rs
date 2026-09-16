//! On-demand inspector data. Gallery scans never need to read these extra fields.
use std::fs;

use nom_exif::{ExifTag, read_exif};

use crate::assets::{Asset, MediaKind, SourceFingerprint};

#[derive(Debug)]
pub struct FileMetadata {
    pub source_state: SourceState,
    pub attributes: Vec<&'static str>,
    pub exif: Vec<MetadataField>,
    pub error: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum SourceState {
    Current,
    Changed,
    Missing,
    Unavailable,
}

#[derive(Debug)]
pub struct MetadataField {
    pub label: &'static str,
    pub value: String,
}

struct SourceCheck {
    state: SourceState,
    metadata: Option<fs::Metadata>,
    error: Option<String>,
}

fn check_source(
    expected: SourceFingerprint,
    metadata: std::io::Result<fs::Metadata>,
) -> SourceCheck {
    let metadata = match metadata {
        Ok(metadata) => metadata,
        Err(error) => {
            let missing = error.kind() == std::io::ErrorKind::NotFound;
            return SourceCheck {
                state: if missing {
                    SourceState::Missing
                } else {
                    SourceState::Unavailable
                },
                metadata: None,
                error: (!missing).then(|| error.to_string()),
            };
        }
    };
    match SourceFingerprint::from_metadata(&metadata) {
        Ok(fingerprint) => SourceCheck {
            state: if fingerprint == expected {
                SourceState::Current
            } else {
                SourceState::Changed
            },
            metadata: Some(metadata),
            error: None,
        },
        Err(error) => SourceCheck {
            state: SourceState::Unavailable,
            metadata: None,
            error: Some(error.to_string()),
        },
    }
}

fn apply_source_recheck(result: &mut FileMetadata, source: SourceCheck) {
    result.source_state = source.state;
    if result.source_state != SourceState::Current {
        result.exif.clear();
        // EXIF and its errors describe a source we can no longer verify. Prefer the stat error,
        // if any, so disappearance stays Missing and an unreadable source stays Unavailable.
        result.error = source.error;
    }
}

pub fn inspect(asset: &Asset) -> FileMetadata {
    let source = check_source(asset.fingerprint, fs::metadata(&asset.path));
    let mut result = FileMetadata {
        source_state: source.state,
        attributes: Vec::new(),
        exif: Vec::new(),
        error: source.error,
    };
    let Some(metadata) = source.metadata else {
        return result;
    };
    if metadata.permissions().readonly() {
        result.attributes.push("Read-only");
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        let flags = metadata.file_attributes();
        for (mask, label) in [
            (0x2, "Hidden"),
            (0x4, "System"),
            (0x20, "Archive"),
            (0x400, "Reparse point"),
            (0x800, "Compressed"),
            (0x1000, "Offline"),
            (0x4000, "Encrypted"),
        ] {
            if flags & mask != 0 {
                result.attributes.push(label);
            }
        }
    }
    // Do not combine EXIF from a changed source with dimensions/dates from the catalog.
    if result.source_state != SourceState::Current || asset.media_kind != MediaKind::Image {
        return result;
    }
    match read_exif(asset.path.as_std_path()) {
        Ok(exif) => {
            for (tag, label) in [
                (ExifTag::Make, "Camera make"),
                (ExifTag::Model, "Camera model"),
                (ExifTag::LensModel, "Lens"),
                (ExifTag::DateTimeOriginal, "Date taken (EXIF)"),
                (ExifTag::ExposureTime, "Exposure time"),
                (ExifTag::FNumber, "Aperture"),
                (ExifTag::ISOSpeedRatings, "ISO"),
                (ExifTag::FocalLength, "Focal length"),
                (ExifTag::Orientation, "Orientation"),
                (ExifTag::Copyright, "Copyright"),
            ] {
                if let Some(value) = exif.get(tag) {
                    let value = value
                        .to_string()
                        .trim_matches('\0')
                        .trim()
                        .chars()
                        .take(4096)
                        .collect::<String>();
                    if !value.is_empty() {
                        result.exif.push(MetadataField { label, value });
                    }
                }
            }
        }
        Err(nom_exif::Error::ExifNotFound | nom_exif::Error::UnsupportedFormat) => {}
        Err(error) => result.error = Some(error.to_string()),
    }
    // A source can be replaced while its EXIF is being read.
    apply_source_recheck(
        &mut result,
        check_source(asset.fingerprint, fs::metadata(&asset.path)),
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::AssetCatalog;
    use camino::Utf8PathBuf;

    #[test]
    fn recheck_keeps_current_exif_and_discards_replaced_source_exif() -> anyhow::Result<()> {
        let temp = tempfile::TempDir::new()?;
        let path = temp.path().join("source");
        fs::write(&path, "original")?;
        let expected = SourceFingerprint::from_metadata(&fs::metadata(&path)?)?;
        let mut result = FileMetadata {
            source_state: SourceState::Current,
            attributes: Vec::new(),
            exif: vec![MetadataField {
                label: "Camera make",
                value: "camera".to_owned(),
            }],
            error: Some("EXIF warning".to_owned()),
        };
        apply_source_recheck(&mut result, check_source(expected, fs::metadata(&path)));
        assert_eq!(result.source_state, SourceState::Current);
        assert_eq!(result.exif.len(), 1);
        assert_eq!(result.error.as_deref(), Some("EXIF warning"));

        fs::write(&path, "replacement has a different size")?;
        apply_source_recheck(&mut result, check_source(expected, fs::metadata(&path)));
        assert_eq!(result.source_state, SourceState::Changed);
        assert!(result.exif.is_empty());
        assert!(result.error.is_none());
        Ok(())
    }

    #[test]
    fn recheck_preserves_missing_and_unavailable_instead_of_reporting_changed() {
        let expected = SourceFingerprint {
            modified_ns: 1,
            size: 1,
        };
        for (kind, state, error) in [
            (std::io::ErrorKind::NotFound, SourceState::Missing, None),
            (
                std::io::ErrorKind::PermissionDenied,
                SourceState::Unavailable,
                Some("permission denied"),
            ),
        ] {
            let mut result = FileMetadata {
                source_state: SourceState::Current,
                attributes: Vec::new(),
                exif: vec![MetadataField {
                    label: "Camera make",
                    value: "old camera".to_owned(),
                }],
                error: Some("old EXIF error".to_owned()),
            };
            apply_source_recheck(
                &mut result,
                check_source(
                    expected,
                    Err(std::io::Error::new(kind, "permission denied")),
                ),
            );
            assert_eq!(result.source_state, state);
            assert!(result.exif.is_empty());
            assert_eq!(result.error.as_deref(), error);
        }
    }

    #[test]
    fn inspector_distinguishes_missing_changed_and_current_sources() -> anyhow::Result<()> {
        let temp = tempfile::TempDir::new()?;
        let root = Utf8PathBuf::try_from(temp.path().to_path_buf())?;
        let path = root.join("image.png");
        image::RgbImage::new(2, 3).save(&path)?;
        let catalog = AssetCatalog::new(&root.join("assets.db"))?;
        let asset = catalog.upsert(&path, &fs::metadata(&path)?)?;
        let info = inspect(&asset);
        assert_eq!(info.source_state, SourceState::Current);
        assert!(info.exif.is_empty());
        assert!(
            info.error.is_none(),
            "a PNG without EXIF is not a probe failure"
        );
        fs::write(&path, "changed source")?;
        assert_eq!(inspect(&asset).source_state, SourceState::Changed);
        fs::remove_file(&path)?;
        assert_eq!(inspect(&asset).source_state, SourceState::Missing);
        Ok(())
    }
}
