use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use nicegal_core::assets::SourceFingerprint;
use nicegal_core::embedding::PatchFeatures;
use parking_lot::Mutex;

const BYTE_BUDGET: usize = 256 * 1024 * 1024;
type Key = (String, i64, Option<i64>, SourceFingerprint);

#[derive(Default)]
struct Cache {
    model: String,
    entries: HashMap<Key, (Arc<PatchFeatures>, usize, u64)>,
    bytes: usize,
    tick: u64,
}

impl Cache {
    fn select_model(&mut self, model: &str) {
        if self.model != model {
            self.model = model.to_owned();
            self.entries.clear();
            self.bytes = 0;
            self.tick = 0;
        }
    }

    fn get(
        &mut self,
        model: &str,
        asset_id: i64,
        timestamp_ms: Option<i64>,
        fingerprint: SourceFingerprint,
    ) -> Option<Arc<PatchFeatures>> {
        self.select_model(model);
        self.tick += 1;
        let (features, _, used) =
            self.entries
                .get_mut(&(model.to_owned(), asset_id, timestamp_ms, fingerprint))?;
        *used = self.tick;
        Some(Arc::clone(features))
    }

    fn insert(
        &mut self,
        model: &str,
        asset_id: i64,
        timestamp_ms: Option<i64>,
        fingerprint: SourceFingerprint,
        features: PatchFeatures,
    ) -> Arc<PatchFeatures> {
        self.select_model(model);
        self.tick += 1;
        let bytes = (features.patches.len() + features.embedding.len()) * size_of::<f32>();
        let features = Arc::new(features);
        if bytes > BYTE_BUDGET {
            return features;
        }
        // A changed source must not leave an old version occupying the budget.
        self.entries.retain(|key, (_, size, _)| {
            let keep = key.1 != asset_id || key.3 == fingerprint;
            if !keep {
                self.bytes -= *size;
            }
            keep
        });
        let key = (model.to_owned(), asset_id, timestamp_ms, fingerprint);
        if let Some((_, old_size, _)) = self
            .entries
            .insert(key, (Arc::clone(&features), bytes, self.tick))
        {
            self.bytes -= old_size;
        }
        self.bytes += bytes;
        while self.bytes > BYTE_BUDGET {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, (_, _, used))| used)
                .map(|(key, _)| key.clone())
                .expect("nonempty cache");
            if let Some((_, size, _)) = self.entries.remove(&oldest) {
                self.bytes -= size;
            }
        }
        features
    }
}

static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
fn cache() -> &'static Mutex<Cache> {
    CACHE.get_or_init(|| Mutex::new(Cache::default()))
}

pub(super) fn get(
    model: &str,
    asset_id: i64,
    timestamp_ms: Option<i64>,
    fingerprint: SourceFingerprint,
) -> Option<Arc<PatchFeatures>> {
    cache()
        .lock()
        .get(model, asset_id, timestamp_ms, fingerprint)
}

pub(super) fn insert(
    model: &str,
    asset_id: i64,
    timestamp_ms: Option<i64>,
    fingerprint: SourceFingerprint,
    features: PatchFeatures,
) -> Arc<PatchFeatures> {
    cache()
        .lock()
        .insert(model, asset_id, timestamp_ms, fingerprint, features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_video_frames_keep_distinct_patch_features() {
        let mut cache = Cache::default();
        let fingerprint = SourceFingerprint {
            modified_ns: 1,
            size: 10,
        };
        let features = |value| PatchFeatures {
            rows: 1,
            columns: 1,
            dimensions: 1,
            patches: vec![value],
            embedding: vec![value],
            region: [0.0, 0.0, 1.0, 1.0],
            method: "test",
        };
        cache.insert("model", 7, Some(100), fingerprint, features(0.1));
        cache.insert("model", 7, Some(200), fingerprint, features(0.9));
        assert_eq!(
            cache
                .get("model", 7, Some(100), fingerprint)
                .unwrap()
                .patches,
            [0.1]
        );
        assert_eq!(
            cache
                .get("model", 7, Some(200), fingerprint)
                .unwrap()
                .patches,
            [0.9]
        );
        assert!(cache.get("model", 7, None, fingerprint).is_none());
    }
}
