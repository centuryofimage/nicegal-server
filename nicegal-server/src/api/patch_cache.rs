use std::collections::HashMap;
use std::sync::Arc;

use nicegal_core::assets::SourceFingerprint;
use nicegal_core::embedding::PatchFeatures;
use parking_lot::Mutex;

const BYTE_BUDGET: usize = 256 * 1024 * 1024;
type Key = (i64, Option<i64>, SourceFingerprint);

/// Patch features of recently inspected assets, for the process's one image model.
#[derive(Default)]
pub(crate) struct PatchCache(Mutex<Cache>);

#[derive(Default)]
struct Cache {
    entries: HashMap<Key, (Arc<PatchFeatures>, usize, u64)>,
    bytes: usize,
    tick: u64,
}

impl Cache {
    fn get(
        &mut self,
        asset_id: i64,
        timestamp_ms: Option<i64>,
        fingerprint: SourceFingerprint,
    ) -> Option<Arc<PatchFeatures>> {
        self.tick += 1;
        let (features, _, used) = self
            .entries
            .get_mut(&(asset_id, timestamp_ms, fingerprint))?;
        *used = self.tick;
        Some(Arc::clone(features))
    }

    fn insert(
        &mut self,
        asset_id: i64,
        timestamp_ms: Option<i64>,
        fingerprint: SourceFingerprint,
        features: PatchFeatures,
    ) -> Arc<PatchFeatures> {
        self.tick += 1;
        let bytes = (features.patches.len() + features.embedding.len()) * size_of::<f32>();
        let features = Arc::new(features);
        if bytes > BYTE_BUDGET {
            return features;
        }
        // A changed source must not leave an old version occupying the budget.
        self.entries.retain(|key, (_, size, _)| {
            let keep = key.0 != asset_id || key.2 == fingerprint;
            if !keep {
                self.bytes -= *size;
            }
            keep
        });
        let key = (asset_id, timestamp_ms, fingerprint);
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
                .map(|(key, _)| *key)
                .expect("nonempty cache");
            if let Some((_, size, _)) = self.entries.remove(&oldest) {
                self.bytes -= size;
            }
        }
        features
    }
}

impl PatchCache {
    pub(super) fn get(
        &self,
        asset_id: i64,
        timestamp_ms: Option<i64>,
        fingerprint: SourceFingerprint,
    ) -> Option<Arc<PatchFeatures>> {
        self.0.lock().get(asset_id, timestamp_ms, fingerprint)
    }

    pub(super) fn insert(
        &self,
        asset_id: i64,
        timestamp_ms: Option<i64>,
        fingerprint: SourceFingerprint,
        features: PatchFeatures,
    ) -> Arc<PatchFeatures> {
        self.0
            .lock()
            .insert(asset_id, timestamp_ms, fingerprint, features)
    }
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
        cache.insert(7, Some(100), fingerprint, features(0.1));
        cache.insert(7, Some(200), fingerprint, features(0.9));
        assert_eq!(cache.get(7, Some(100), fingerprint).unwrap().patches, [0.1]);
        assert_eq!(cache.get(7, Some(200), fingerprint).unwrap().patches, [0.9]);
        assert!(cache.get(7, None, fingerprint).is_none());
    }
}
