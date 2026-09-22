//! Request snapshots are never persisted. Only small model-specific vectors survive requests.
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use base64::{Engine, engine::general_purpose::STANDARD};
use nicegal_core::embedding::ImageEmbedder;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::error::ApiError;

pub(super) const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;
pub(super) const MAX_TOTAL_BYTES: usize = 32 * 1024 * 1024;
const MAX_PIXELS: usize = 40_000_000;
const CACHE_ENTRIES: usize = 32;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct ExternalImageRequest {
    bytes_base64: String,
}

impl ExternalImageRequest {
    pub(super) fn decode(self) -> Result<Vec<u8>, ApiError> {
        if self.bytes_base64.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
            return Err(ApiError::bad_request("external image exceeds 16 MiB"));
        }
        let bytes = STANDARD
            .decode(self.bytes_base64)
            .map_err(|_| ApiError::bad_request("external image must contain valid base64"))?;
        if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
            return Err(ApiError::bad_request(
                "external image must contain 1 byte to 16 MiB",
            ));
        }
        nicegal_core::imaging::Format::detect(&bytes).map_err(|_| {
            ApiError::bad_request("external image must be JPEG, PNG, GIF, WebP, or BMP")
        })?;
        let size = imagesize::blob_size(&bytes)
            .map_err(|_| ApiError::bad_request("external image dimensions could not be read"))?;
        if size.width == 0
            || size.height == 0
            || size
                .width
                .checked_mul(size.height)
                .is_none_or(|pixels| pixels > MAX_PIXELS)
        {
            return Err(ApiError::bad_request(
                "external image exceeds 40 megapixels or has invalid dimensions",
            ));
        }
        Ok(bytes)
    }
}

type CacheKey = (&'static str, [u8; 32]);
#[derive(Default)]
struct VectorCache(VecDeque<(CacheKey, Vec<f32>)>);

impl VectorCache {
    fn get(&mut self, key: &(&'static str, [u8; 32])) -> Option<Vec<f32>> {
        let index = self.0.iter().position(|entry| &entry.0 == key)?;
        let entry = self.0.remove(index)?;
        let vector = entry.1.clone();
        self.0.push_back(entry);
        Some(vector)
    }

    fn insert(&mut self, key: (&'static str, [u8; 32]), vector: Vec<f32>) {
        self.0.retain(|entry| entry.0 != key);
        while self.0.len() >= CACHE_ENTRIES {
            self.0.pop_front();
        }
        self.0.push_back((key, vector));
    }
}

pub(super) fn embed(bytes: &[u8], model: &ImageEmbedder) -> Result<Vec<f32>, ApiError> {
    static CACHE: OnceLock<Mutex<VectorCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(Mutex::default);
    let key = (model.model().id(), Sha256::digest(bytes).into());
    // Serialize misses as well, avoiding repeated inference for simultaneous identical searches.
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(vector) = cache.get(&key) {
        return Ok(vector);
    }
    let raster = model.decode_image(bytes).map_err(|_| {
        ApiError::bad_request("external image could not be decoded; choose another file")
    })?;
    if u64::from(raster.width()) * u64::from(raster.height()) > MAX_PIXELS as u64 {
        return Err(ApiError::bad_request(
            "external image exceeds 40 megapixels",
        ));
    }
    let vector = model.embed_raster(raster)?;
    cache.insert(key, vector.clone());
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the CLIP image encoder cached by explicit setup"]
    fn cached_encoder_embeds_external_snapshots() {
        super::super::tests::initialize_test_runtime();
        let model =
            ImageEmbedder::load_cached(&nicegal_core::embedding::ImageEmbedderOptions::default())
                .unwrap()
                .expect("cached image encoder");
        let png = nicegal_core::imaging::encode_png(1, 1, &[0, 0, 0, 255]).unwrap();
        let vector = embed(&png, &model).unwrap();
        assert_eq!(vector.len(), model.dimensions());
        assert!(vector.iter().all(|value| value.is_finite()));
        assert_eq!(vector, embed(&png, &model).unwrap());
    }

    #[test]
    fn cache_is_bounded_and_model_specific() {
        let mut cache = VectorCache::default();
        for i in 0..=CACHE_ENTRIES {
            cache.insert(("a", [i as u8; 32]), vec![i as f32]);
        }
        assert_eq!(cache.0.len(), CACHE_ENTRIES);
        assert!(cache.get(&("a", [0; 32])).is_none());
        assert!(cache.get(&("b", [1; 32])).is_none());
        assert_eq!(cache.get(&("a", [1; 32])), Some(vec![1.0]));
        cache.insert(("a", [33; 32]), vec![33.0]);
        assert!(cache.get(&("a", [2; 32])).is_none());
    }

    #[test]
    fn rejects_invalid_payloads_before_inference() {
        for bytes_base64 in [
            String::new(),
            "!!!".into(),
            STANDARD.encode(b"not an image"),
        ] {
            assert!(ExternalImageRequest { bytes_base64 }.decode().is_err());
        }
        let png = nicegal_core::imaging::encode_png(1, 1, &[0, 0, 0, 255]).unwrap();
        assert_eq!(
            ExternalImageRequest {
                bytes_base64: STANDARD.encode(&png)
            }
            .decode()
            .unwrap(),
            png
        );
    }
}
