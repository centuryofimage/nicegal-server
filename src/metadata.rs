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

pub fn inspect(asset: &Asset) -> FileMetadata {
    let mut result = FileMetadata {
        source_state: SourceState::Unavailable,
        attributes: Vec::new(),
        exif: Vec::new(),
        error: None,
    };
    let metadata = match fs::metadata(&asset.path) {
        Ok(metadata) => metadata,
        Err(error) => {
            if error.kind() == std::io::ErrorKind::NotFound {
                result.source_state = SourceState::Missing;
            } else {
                result.error = Some(error.to_string());
            }
            return result;
        }
    };
    result.source_state = match SourceFingerprint::from_metadata(&metadata) {
        Ok(fingerprint) if fingerprint == asset.fingerprint => SourceState::Current,
        Ok(_) => SourceState::Changed,
        Err(error) => {
            result.error = Some(error.to_string());
            return result;
        }
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
    if fs::metadata(&asset.path)
        .ok()
        .and_then(|m| SourceFingerprint::from_metadata(&m).ok())
        != Some(asset.fingerprint)
    {
        result.source_state = SourceState::Changed;
        result.exif.clear();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::AssetCatalog;
    use camino::Utf8PathBuf;

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
