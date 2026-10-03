use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// How long an idle entry is retained before the next sweep drops it.
pub(super) const ENTRY_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);
/// Most entries one map holds. New keys beyond it are refused rather than evicting others.
pub(super) const MAX_ENTRIES: usize = 4096;

pub(super) trait Retained {
    fn touched_at(&self) -> Instant;

    /// Pinned entries are never aged out, because their owner releases them explicitly.
    fn pinned(&self) -> bool {
        false
    }
}

/// Client-keyed state that is swept lazily, by whoever touches the map next.
pub(super) struct TtlMap<V> {
    entries: BTreeMap<String, V>,
}

impl<V> Default for TtlMap<V> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<V: Retained> TtlMap<V> {
    pub(super) fn sweep(&mut self) {
        self.entries
            .retain(|_, value| value.pinned() || value.touched_at().elapsed() < ENTRY_RETENTION);
    }

    pub(super) fn has_room_for(&self, key: &str) -> bool {
        self.entries.contains_key(key) || self.entries.len() < MAX_ENTRIES
    }

    pub(super) fn get(&self, key: &str) -> Option<&V> {
        self.entries.get(key)
    }

    pub(super) fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.entries.get_mut(key)
    }

    pub(super) fn insert(&mut self, key: String, value: V) {
        self.entries.insert(key, value);
    }

    pub(super) fn get_or_insert_with(&mut self, key: &str, create: impl FnOnce() -> V) -> &mut V {
        self.entries.entry(key.to_owned()).or_insert_with(create)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.entries.iter()
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Entry {
        touched: Instant,
        pinned: bool,
    }

    impl Retained for Entry {
        fn touched_at(&self) -> Instant {
            self.touched
        }

        fn pinned(&self) -> bool {
            self.pinned
        }
    }

    #[test]
    fn sweep_drops_only_aged_unpinned_entries_and_the_cap_admits_existing_keys() {
        let old = Instant::now() - ENTRY_RETENTION - Duration::from_secs(1);
        let mut map = TtlMap::default();
        map.insert(
            "aged".into(),
            Entry {
                touched: old,
                pinned: false,
            },
        );
        map.insert(
            "pinned".into(),
            Entry {
                touched: old,
                pinned: true,
            },
        );
        map.insert(
            "fresh".into(),
            Entry {
                touched: Instant::now(),
                pinned: false,
            },
        );
        map.sweep();
        assert!(map.get("aged").is_none());
        assert!(map.get("pinned").is_some());
        assert!(map.get("fresh").is_some());

        for index in 0..MAX_ENTRIES {
            map.insert(
                format!("k{index}"),
                Entry {
                    touched: Instant::now(),
                    pinned: false,
                },
            );
        }
        assert!(map.has_room_for("fresh"));
        assert!(!map.has_room_for("new"));
    }
}
