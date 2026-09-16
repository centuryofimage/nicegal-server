use crate::api::jobs::cancel_if;
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::assets::{Asset, AssetCatalog, Timeline};
use nicegal_core::index::{IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta};
use nicegal_core::thumbs::{GENERATOR_VERSION, SIZE_BUCKETS, ThumbnailService};
use serde::Deserialize;

use super::super::error::ApiError;
use super::super::roots;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    root: PathBuf,
    #[serde(default = "default_backfill_buckets")]
    buckets: Vec<u16>,
    #[serde(default)]
    force: bool,
    #[serde(default)]
    sweep_stale: bool,
    #[serde(default)]
    timeline: TimelineRequest,
    range: Option<TimelineRange>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
enum TimelineRequest {
    #[default]
    Modified,
    Capture,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TimelineRange {
    from_ns: Option<String>,
    to_ns: Option<String>,
}

pub(crate) struct Spec {
    root: PathBuf,
    buckets: Vec<u16>,
    force: bool,
    sweep_stale: bool,
    timeline: Timeline,
    from_ns: Option<i64>,
    to_ns: Option<i64>,
}

pub(crate) fn prepare(mut request: Request) -> Result<Spec, ApiError> {
    let root = roots::resolve_root("thumbnail", &request.root)?;
    request.buckets.sort_unstable();
    request.buckets.dedup();
    if request.buckets.is_empty() {
        return Err(ApiError::bad_request(
            "at least one thumbnail bucket is required",
        ));
    }
    if let Some(bucket) = request
        .buckets
        .iter()
        .find(|bucket| !SIZE_BUCKETS.contains(bucket))
    {
        return Err(ApiError::bad_request(format!(
            "unsupported thumbnail bucket: {bucket}"
        )));
    }
    let (from_ns, to_ns) = request
        .range
        .map(|range| -> Result<_, ApiError> {
            Ok((
                parse_ns("fromNs", range.from_ns)?,
                parse_ns("toNs", range.to_ns)?,
            ))
        })
        .transpose()?
        .unwrap_or_default();
    if matches!((from_ns, to_ns), (Some(from), Some(to)) if from >= to) {
        return Err(ApiError::bad_request(
            "timeline range fromNs must be less than toNs",
        ));
    }
    if request.sweep_stale
        && (from_ns.is_some() || to_ns.is_some() || request.buckets != SIZE_BUCKETS)
    {
        return Err(ApiError::bad_request(
            "sweepStale requires an unbounded backfill of every thumbnail bucket",
        ));
    }
    Ok(Spec {
        root,
        buckets: request.buckets,
        force: request.force,
        sweep_stale: request.sweep_stale,
        timeline: match request.timeline {
            TimelineRequest::Modified => Timeline::Modified,
            TimelineRequest::Capture => Timeline::Capture,
        },
        from_ns,
        to_ns,
    })
}

fn parse_ns(field: &str, value: Option<String>) -> Result<Option<i64>, ApiError> {
    value
        .map(|value| {
            value.parse::<i64>().map_err(|_| {
                ApiError::bad_request(format!(
                    "timeline range {field} must be a signed 64-bit decimal string"
                ))
            })
        })
        .transpose()
}

pub(crate) fn run(
    spec: Spec,
    asset_database: &PathBuf,
    thumbnails: &ThumbnailService,
    observer: &dyn IndexObserver,
) -> anyhow::Result<()> {
    let assets = selected_assets(&AssetCatalog::new(asset_database)?, &spec)?;
    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::Thumbnails));
    observer.on_event(IndexEvent::Discovered {
        count: assets.len(),
    });
    observer.on_event(IndexEvent::DiscoveryComplete {
        total: assets.len(),
    });
    let mut failures = 0_usize;
    for outcome in thumbnails.generate_observed(
        &assets,
        &spec.buckets,
        GENERATOR_VERSION,
        spec.force,
        || observer.is_cancelled(),
        |asset, active| {
            observer.on_event(IndexEvent::ActiveAsset {
                path: asset.path.clone(),
                active,
            });
        },
    ) {
        cancel_if(observer.is_cancelled())?;
        let asset = outcome.asset;
        match outcome.result {
            Ok(summary) => observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                phase_completed: 1,
                processed: 1,
                thumbnails_generated: summary.generated,
                ..IndexProgressDelta::default()
            })),
            Err(error) => {
                failures += 1;
                observer.on_event(IndexEvent::Error {
                    path: Some(asset.path),
                    message: format!("thumbnail generation failed: {error:#}"),
                });
                observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                    phase_completed: 1,
                    processed: 1,
                    thumbnail_failures: 1,
                    ..IndexProgressDelta::default()
                }));
            }
        }
    }
    cancel_if(observer.is_cancelled())?;
    if spec.sweep_stale {
        if failures > 0 {
            anyhow::bail!(
                "refusing to sweep stale thumbnail generators after {failures} backfill failures"
            );
        }
        thumbnails.sweep_old_generators_for_assets(
            GENERATOR_VERSION,
            assets.iter().map(|asset| asset.asset_id).collect(),
        )?;
    }
    Ok(())
}

fn selected_assets(catalog: &AssetCatalog, spec: &Spec) -> anyhow::Result<Vec<Asset>> {
    // Scope by the canonical root before filtering the selected timeline, so a broad range can
    // never make this maintenance job operate on a sibling library.
    Ok(catalog
        .under_root(&spec.root)?
        .into_iter()
        .filter(|asset| {
            let timestamp = match spec.timeline {
                Timeline::Modified => asset.fingerprint.modified_ns,
                Timeline::Capture => asset.exif_taken_ns.unwrap_or(asset.fingerprint.modified_ns),
            };
            spec.from_ns.is_none_or(|from| timestamp >= from)
                && spec.to_ns.is_none_or(|to| timestamp < to)
        })
        .collect())
}

fn default_backfill_buckets() -> Vec<u16> {
    vec![1024]
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use tempfile::TempDir;

    fn existing_root() -> PathBuf {
        PathBuf::try_from(std::env::current_dir().unwrap()).unwrap()
    }

    #[test]
    fn parses_lossless_half_open_timeline_range() {
        let root = existing_root();
        let request: Request = serde_json::from_value(serde_json::json!({
            "root": root.as_str(),
            "timeline": "capture",
            "range": {
                "fromNs": "1704067200000000000",
                "toNs": "1735689600000000000"
            }
        }))
        .unwrap();
        let spec = prepare(request).unwrap();
        assert_eq!(spec.timeline, Timeline::Capture);
        assert_eq!(spec.from_ns, Some(1_704_067_200_000_000_000));
        assert_eq!(spec.to_ns, Some(1_735_689_600_000_000_000));
    }

    #[test]
    fn rejects_reversed_or_imprecise_numeric_ranges() {
        let root = existing_root();
        let reversed: Request = serde_json::from_value(serde_json::json!({
            "root": root.as_str(),
            "range": { "fromNs": "2", "toNs": "1" }
        }))
        .unwrap();
        assert!(prepare(reversed).is_err());

        assert!(
            serde_json::from_value::<Request>(serde_json::json!({
                "root": root.as_str(),
                "range": { "fromNs": 1704067200000000000_i64 }
            }))
            .is_err()
        );
    }

    #[test]
    fn thumbnail_root_is_required_and_request_fields_are_strict() {
        assert!(serde_json::from_value::<Request>(serde_json::json!({})).is_err());

        let root = existing_root();
        assert!(
            serde_json::from_value::<Request>(serde_json::json!({
                "root": root.as_str(),
                "unexpected": true
            }))
            .is_err()
        );
    }

    #[test]
    fn thumbnail_selection_excludes_same_prefix_sibling_root() -> anyhow::Result<()> {
        let temporary = TempDir::new()?;
        let root = PathBuf::try_from(temporary.path().join("photos"))?;
        let sibling = PathBuf::try_from(temporary.path().join("photos-old"))?;
        fs::create_dir(&root)?;
        fs::create_dir(&sibling)?;
        let inside = root.join("inside.png");
        let outside = sibling.join("outside.png");
        fs::write(&inside, [])?;
        fs::write(&outside, [])?;

        let catalog_path = PathBuf::try_from(temporary.path().join("assets.db"))?;
        let catalog = AssetCatalog::new(&catalog_path)?;
        let included = catalog.upsert(&inside, &fs::metadata(&inside)?)?;
        let excluded = catalog.upsert(&outside, &fs::metadata(&outside)?)?;
        let spec = prepare(Request {
            root,
            buckets: default_backfill_buckets(),
            force: false,
            sweep_stale: false,
            timeline: TimelineRequest::Modified,
            range: None,
        })
        .expect("temporary root should prepare for thumbnail selection");

        let selected = selected_assets(&catalog, &spec)?;
        assert_eq!(
            selected
                .iter()
                .map(|asset| asset.asset_id)
                .collect::<Vec<_>>(),
            vec![included.asset_id]
        );
        assert_ne!(included.asset_id, excluded.asset_id);
        Ok(())
    }
}
