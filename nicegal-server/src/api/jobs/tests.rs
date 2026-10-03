use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use camino::Utf8PathBuf as PathBuf;
use nicegal_core::hub::{DownloadObserver, ModelSource};
use nicegal_core::index::{IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta};
use nicegal_core::thumbs::ThumbnailService;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::WatchStream;

use crate::api::models::{ImageModel as ImageEmbedder, ImageQueryModel, TextModel as TextEmbedder};
use crate::api::{Databases, ocr_models};

use super::job::*;
use super::manager::*;
use super::request::*;
use super::routes::*;

#[test]
fn library_job_snapshots_identify_their_library() {
    let job = Job::new_for(42, JobKind::LibraryScan, Some(7));
    let snapshot = serde_json::to_value(job.response()).unwrap();
    assert_eq!(snapshot["libraryId"], 7);
    assert_eq!(snapshot["status"], "queued");
    assert!(
        serde_json::to_value(Job::new(43, JobKind::ModelPrepare).response())
            .unwrap()
            .get("libraryId")
            .is_none()
    );
}

#[test]
fn model_preparation_wire_contract_and_progress() {
    let request: JobRequest = serde_json::from_value(serde_json::json!({
        "type": "modelPrepare", "params": {}
    }))
    .unwrap();
    let spec = request.prepare().unwrap();
    assert!(matches!(spec, JobSpec::ModelPrepare));
    let job = Job::new(1, spec.kind());
    assert!(job.begin());
    job.preparing_models(3);
    job.models_loaded(1);
    let response = serde_json::to_value(job.response()).unwrap();
    assert_eq!(response["type"], "modelPrepare");
    assert_eq!(response["phase"], "loadingModels");
    assert_eq!(response["progress"]["total"], 3);
    assert_eq!(response["progress"]["phaseCompleted"], 1);
    assert_eq!(response["progress"]["modelsLoaded"], 1);
    assert!(
        serde_json::from_value::<JobRequest>(serde_json::json!({
            "type": "modelPrepare", "params": {"root": "unexpected"}
        }))
        .is_err()
    );
}

#[test]
fn job_observer_tracks_progress_and_cancellation() {
    let job = Job::new(7, JobKind::OcrModelLoad);
    assert!(job.begin());
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Scanning));
    job.on_event(IndexEvent::Discovered { count: 12 });
    job.on_event(IndexEvent::DiscoveryComplete { total: 12 });
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        processed: 4,
        cataloged: 4,
        indexed: 2,
        skipped: 1,
        failed: 1,
        deleted: 0,
        ..IndexProgressDelta::default()
    }));
    job.on_event(IndexEvent::Error {
        path: Some(PathBuf::from("C:/gallery/broken.png")),
        message: "OCR failed: invalid image".to_owned(),
    });
    job.request_cancel();

    let response = job.response();
    assert_eq!(response.job_id, wire_job_id(7));
    assert_eq!(response.kind, JobKind::OcrModelLoad);
    assert_eq!(response.status, JobStatus::Cancelling);
    assert_eq!(response.phase, JobPhase::Scanning);
    assert_eq!(response.progress.discovered, 12);
    assert_eq!(response.progress.total, Some(12));
    assert_eq!(response.progress.processed, 4);
    assert_eq!(response.errors.len(), 1);
    assert_eq!(
        response.errors[0].path.as_deref(),
        Some(camino::Utf8Path::new("C:/gallery/broken.png"))
    );
    assert_eq!(response.errors[0].message, "OCR failed: invalid image");
    assert!(job.is_cancelled());
}

#[test]
fn file_downloads_preserve_model_steps_and_retry_byte_totals() {
    let job = Job::new(88, JobKind::ModelPrepare);
    job.begin();
    job.preparing_models(3);
    job.models_loaded(1);
    let source = ModelSource::local(std::path::Path::new("image.onnx")).unwrap();
    job.progress(&source, 25, 100);
    let response = serde_json::to_value(job.response()).unwrap();
    assert_eq!(response["phase"], "loadingModels");
    assert_eq!(response["progress"]["phaseCompleted"], 1);
    assert_eq!(response["progress"]["total"], 3);
    assert_eq!(response["progress"]["download"]["filename"], "image.onnx");
    assert_eq!(response["progress"]["download"]["downloadedBytes"], 25);
    DownloadObserver::finish(&job);
    assert!(job.response().progress.download.is_none());
    assert_eq!(job.response().progress.phase_completed, 1);

    let ocr = Job::new(89, JobKind::OcrModelLoad);
    ocr.ocr_download_progress(&source, 50, 100, 0, 0);
    // Retrying this file replaces its contribution instead of double-counting it.
    ocr.ocr_download_progress(&source, 0, 100, 50, 100);
    ocr.ocr_download_progress(&source, 100, 100, 0, 100);
    // A second file adds to the legacy job totals but has its own current-file counters.
    ocr.ocr_download_progress(&source, 20, 40, 0, 0);
    let progress = ocr.response().progress;
    assert_eq!(progress.downloaded_bytes, 120);
    assert_eq!(progress.download_total_bytes, 140);
    assert_eq!(progress.download.unwrap().downloaded_bytes, 20);
    ocr.fail("network error");
    assert!(ocr.response().progress.download.is_none());
}

#[test]
fn phase_progress_is_scoped_to_the_current_phase() {
    let job = Job::new(8, JobKind::LibraryScan);
    assert!(job.begin());
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Scanning));
    job.on_event(IndexEvent::Discovered { count: 5 });
    job.on_event(IndexEvent::DiscoveryComplete { total: 5 });
    assert_eq!(job.response().progress.total, Some(5));
    assert_eq!(job.response().progress.phase_completed, 5);

    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Cataloging));
    let after_change = job.response();
    assert_eq!(after_change.progress.discovered, 0);
    assert_eq!(after_change.progress.total, None);
    assert_eq!(after_change.progress.phase_completed, 0);

    job.on_event(IndexEvent::DiscoveryComplete { total: 5 });
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: 2,
        cataloged: 2,
        ..IndexProgressDelta::default()
    }));
    let response = job.response();
    assert_eq!(response.progress.total, Some(5));
    assert_eq!(response.progress.phase_completed, 2);
    assert_eq!(response.progress.cataloged, 2);
}

#[test]
fn throughput_resets_when_indexing_phase_changes() {
    let job = Job::new(8, JobKind::LibraryScan);
    assert!(job.begin());

    for phase in [
        IndexPhase::Cataloging,
        IndexPhase::Thumbnails,
        IndexPhase::Ocr,
        IndexPhase::Cleanup,
        IndexPhase::ImageEmbedding,
        IndexPhase::TextEmbedding,
    ] {
        job.on_event(IndexEvent::PhaseChanged(phase));
        let response = job.response();
        assert_eq!(response.progress.items_per_second, None);
        assert_eq!(response.progress.phase_completed, 0);
        assert!(job.data().phase_started_at.unwrap().elapsed() < Duration::from_secs(1));

        // A controlled elapsed time makes each phase's expected rate independent of
        // machine speed and proves that earlier phases' work is not counted again.
        job.data().phase_started_at = Some(Instant::now() - Duration::from_secs(10));
        job.on_event(IndexEvent::Progress(IndexProgressDelta {
            phase_completed: 20,
            images_inferred: if phase == IndexPhase::ImageEmbedding {
                20
            } else {
                0
            },
            ..IndexProgressDelta::default()
        }));
        let rate = job.response().progress.items_per_second.unwrap();
        assert!((1.9..=2.0).contains(&rate), "unexpected rate: {rate}");
    }
}

#[test]
fn resumed_ocr_throughput_excludes_skips_but_preserves_progress() {
    let job = Job::new(10, JobKind::LibraryScan);
    assert!(job.begin());
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Cataloging));
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: 100,
        skipped: 100,
        ..IndexProgressDelta::default()
    }));
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Ocr));
    job.on_event(IndexEvent::DiscoveryComplete { total: 10_020 });
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: 10_000,
        processed: 10_000,
        skipped: 10_000,
        ..IndexProgressDelta::default()
    }));
    assert_eq!(job.response().progress.items_per_second, None);

    job.data().phase_started_at = Some(Instant::now() - Duration::from_secs(10));
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: 20,
        processed: 20,
        ..IndexProgressDelta::default()
    }));
    let progress = job.response().progress;
    assert_eq!(progress.phase_completed, 10_020);
    assert_eq!(progress.total, Some(10_020));
    assert_eq!(progress.skipped, 10_100);
    assert!((1.9..=2.0).contains(&progress.items_per_second.unwrap()));

    // Saving the completed OCR batch must not count the same work twice.
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        indexed: 20,
        ..IndexProgressDelta::default()
    }));
    assert!((1.9..=2.0).contains(&job.response().progress.items_per_second.unwrap()));
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
    assert_eq!(job.response().progress.items_per_second, None);
}

#[test]
fn image_rate_counts_inferred_frames_not_files() {
    let job = Job::new(1, JobKind::LibraryScan);
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
    job.on_event(IndexEvent::DiscoveryComplete { total: 2 });
    job.data().phase_started_at = Some(Instant::now() - Duration::from_secs(10));
    // One still and three video frames are inferred before either file is saved.
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        images_inferred: 4,
        ..IndexProgressDelta::default()
    }));
    let progress = job.response().progress;
    assert_eq!(progress.phase_completed, 0);
    assert!((0.39..=0.4).contains(&progress.items_per_second.unwrap()));
    // Saving the files advances completion without counting the inference again.
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        phase_completed: 2,
        embedded: 1,
        skipped: 1,
        ..IndexProgressDelta::default()
    }));
    let progress = job.response().progress;
    assert_eq!(progress.phase_completed, 2);
    assert_eq!(progress.total, Some(2));
    assert!((0.39..=0.4).contains(&progress.items_per_second.unwrap()));

    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Cataloging));
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
    assert_eq!(job.response().progress.items_per_second, None);
    job.data().phase_started_at = Some(Instant::now() - Duration::from_secs(10));
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        images_inferred: 8,
        ..IndexProgressDelta::default()
    }));
    assert!((0.79..=0.8).contains(&job.response().progress.items_per_second.unwrap()));
}

#[test]
fn job_identifiers_belong_to_this_backend_instance() {
    assert_eq!(parse_job_id(&wire_job_id(42)).unwrap(), 42);
    for invalid in ["0", "-1", "not-a-job"] {
        assert!(parse_job_id(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn typed_job_envelope_rejects_unknown_fields() {
    let valid = serde_json::json!({
        "type": "ocrModelLoad",
        "params": {
            "detection": {
                "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx"
            },
            "recognition": {
                "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx"
            }
        }
    });
    assert!(serde_json::from_value::<JobRequest>(valid).is_ok());

    let library_scan = serde_json::json!({
        "type": "libraryScan",
        "params": { "libraryId": 1, "pendingOnly": true }
    });
    assert!(serde_json::from_value::<JobRequest>(library_scan).is_ok());
    for retired in ["libraryIndex", "catalogSync"] {
        let request = serde_json::json!({ "type": retired, "params": { "root": "C:/gallery" } });
        assert!(
            serde_json::from_value::<JobRequest>(request).is_err(),
            "{retired}"
        );
    }

    let thumbnail_generate = serde_json::json!({
        "type": "thumbnailGenerate",
        "params": {
            "libraryId": 1
        }
    });
    assert!(serde_json::from_value::<JobRequest>(thumbnail_generate).is_ok());

    let library_purge = serde_json::json!({
        "type": "libraryPurge",
        "params": {
            "libraryId": 1
        }
    });
    assert!(serde_json::from_value::<JobRequest>(library_purge).is_ok());

    let unknown = serde_json::json!({
        "type": "ocrModelLoad",
        "params": {
            "detection": {
                "modelId": "PaddlePaddle/PP-OCRv6_small_det_onnx"
            },
            "recognition": {
                "modelId": "PaddlePaddle/PP-OCRv6_small_rec_onnx"
            }
        },
        "unexpected": true
    });
    assert!(serde_json::from_value::<JobRequest>(unknown).is_err());
}

#[test]
fn folder_progress_attributes_catalog_events_to_the_current_folder() {
    let job = Job::new(11, JobKind::LibraryScan);
    assert!(job.begin());
    job.set_folders(vec!["/a".into(), "/b".into()]);
    job.enter_folder(0);
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Scanning));
    job.on_event(IndexEvent::Discovered { count: 3 });
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        cataloged: 2,
        failed: 1,
        ..IndexProgressDelta::default()
    }));
    job.finish_folder(0, FolderState::Scanned, None);
    job.enter_folder(1);
    job.on_event(IndexEvent::Discovered { count: 5 });
    job.finish_folder(1, FolderState::Unavailable, Some("offline".to_owned()));
    job.leave_folder();
    // Later phases are library-wide and leave folder counts alone.
    job.on_event(IndexEvent::Progress(IndexProgressDelta {
        cataloged: 9,
        ..IndexProgressDelta::default()
    }));

    let folders = serde_json::to_value(job.response()).unwrap()["folders"].clone();
    assert_eq!(
        folders,
        serde_json::json!([
            {"path": "/a", "state": "scanned", "discovered": 3, "cataloged": 2, "failed": 1, "error": null},
            {"path": "/b", "state": "unavailable", "discovered": 5, "cataloged": 0, "failed": 0, "error": "offline"}
        ])
    );
}

#[test]
fn library_purge_job_kind_serializes_as_the_typed_request_name() {
    assert_eq!(
        serde_json::to_value(JobKind::LibraryPurge).unwrap(),
        serde_json::json!("libraryPurge")
    );
}

#[tokio::test]
async fn event_stream_emits_initial_and_terminal_snapshots_then_closes() {
    let job = Job::new(9, JobKind::ThumbnailGenerate);
    let mut stream = JobEventStream {
        inner: WatchStream::new(job.subscribe()),
        finished: false,
    };
    assert!(stream.next().await.is_some());

    assert!(job.begin());
    assert!(stream.next().await.is_some());
    job.complete(false);
    assert!(stream.next().await.is_some());
    assert!(stream.next().await.is_none());
}

// An inert active job keeps scheduling tests deterministic without model loading or I/O work.
fn hold_active_job(manager: &JobManager, kind: JobKind) -> Arc<Job> {
    let id = manager.next_id.fetch_add(1, Ordering::Relaxed);
    let job = Arc::new(Job::new(id, kind));
    assert!(job.begin());
    let mut registry = manager.registry();
    registry.active = Some(id);
    registry.jobs.insert(id, Arc::clone(&job));
    job
}

#[test]
fn failed_view_scheduling_leaves_no_committed_view_behind() {
    let temp = tempfile::TempDir::new().unwrap();
    let manager = test_manager(&temp);
    hold_active_job(&manager, JobKind::ModelPrepare);
    manager.shutting_down.store(true, Ordering::Release);
    assert!(manager.set_view("desktop".into(), 1, Some(7)).is_err());
    assert!(manager.registry().views.get("desktop").is_none());
    assert!(!manager.registry().visited.contains(&7));

    manager.shutting_down.store(false, Ordering::Release);
    manager.set_view("desktop".into(), 1, Some(7)).unwrap();
    assert!(
        manager.registry().queued.len() == 1,
        "the retry with the same generation schedules the scan"
    );
}

#[test]
fn thumbnail_resume_record_parses_as_a_job_request() {
    let request: JobRequest = serde_json::from_value(serde_json::json!({
        "type": "thumbnailGenerate",
        "params": { "libraryId": 3, "buckets": [256, 1024], "force": true }
    }))
    .unwrap();
    let JobSpec::ThumbnailGenerate(spec) = request.prepare().unwrap() else {
        panic!("expected a thumbnail job");
    };
    let record = serde_json::to_vec(&spec.resume_request()).unwrap();
    let resumed: JobRequest = serde_json::from_slice(&record).unwrap();
    let JobSpec::ThumbnailGenerate(resumed) = resumed.prepare().unwrap() else {
        panic!("expected a thumbnail job");
    };
    assert_eq!(resumed.library_id(), 3);
    assert_eq!(
        serde_json::to_value(resumed.resume_request()).unwrap(),
        serde_json::to_value(spec.resume_request()).unwrap()
    );
}

#[test]
fn library_views_share_work_and_ignore_late_updates_after_release() {
    let temp = tempfile::TempDir::new().unwrap();
    let manager = test_manager(&temp);
    hold_active_job(&manager, JobKind::ModelPrepare);
    manager.schedule_viewed_scan(7).unwrap();
    assert!(
        manager.registry().queued.is_empty(),
        "unviewed libraries stay idle"
    );
    manager.set_view("desktop".into(), 1, Some(7)).unwrap();
    let first = manager.registry().queued[0].0;
    manager.set_view("phone".into(), 1, Some(7)).unwrap();
    assert_eq!(manager.registry().queued.len(), 1);
    manager.set_view("desktop".into(), 2, Some(8)).unwrap();
    assert_eq!(manager.registry().queued.len(), 2);
    assert_eq!(
        manager.get(first).unwrap().response().status,
        JobStatus::Queued
    );
    manager.set_view("phone".into(), u64::MAX, None).unwrap();
    assert_eq!(
        manager.get(first).unwrap().response().status,
        JobStatus::Cancelled
    );
    assert_eq!(manager.registry().queued.len(), 1);
    manager.set_view("phone".into(), 2, Some(7)).unwrap();
    manager.set_view("desktop".into(), 1, Some(7)).unwrap();
    assert_eq!(
        manager.registry().queued.len(),
        1,
        "late updates cannot resurrect closed views"
    );
    manager.set_view("desktop".into(), 3, Some(7)).unwrap();
    let registry = manager.registry();
    assert_eq!(registry.queued.len(), 1);
    assert_ne!(
        registry.queued[0].0, first,
        "returning must not merge into a cancelled scan"
    );
}

#[tokio::test]
async fn thumbnail_recovery_rejects_destructive_records_and_bad_records_do_not_block_startup() {
    let temp = tempfile::TempDir::new().unwrap();
    let manager = test_manager(&temp);
    for record in [
        "not json",
        r#"{"type":"libraryPurge","params":{"libraryId":1,"removeLibrary":true}}"#,
    ] {
        std::fs::write(manager.resume_path(), record).unwrap();
        assert!(manager.resume_thumbnail_job().is_err());
        manager.recover().await.unwrap();
        assert!(manager.list().jobs.is_empty());
    }
}

#[test]
fn thumbnail_resume_record_survives_old_cancellation_and_shutdown_but_not_explicit_cancel() {
    let temp = tempfile::TempDir::new().unwrap();
    let manager = test_manager(&temp);
    let old = hold_active_job(&manager, JobKind::ThumbnailGenerate);
    old.complete(false);
    let current = hold_active_job(&manager, JobKind::ThumbnailGenerate);
    std::fs::write(manager.resume_path(), b"pending work").unwrap();
    manager.cancel(old.id).unwrap();
    assert!(manager.resume_path().exists());
    manager.cancel(current.id).unwrap();
    assert!(!manager.resume_path().exists());
    std::fs::write(manager.resume_path(), b"pending work").unwrap();
    manager.cancel_all();
    assert!(manager.resume_path().exists());
}

#[tokio::test]
async fn interrupted_thumbnails_resume_without_a_frontend_and_completed_work_is_not_replayed() {
    let temp = tempfile::TempDir::new().unwrap();
    let manager = test_manager(&temp);
    let library = nicegal_core::assets::AssetCatalog::new(&manager.databases.assets)
        .unwrap()
        .create_library(
            &nicegal_core::libraries::LibraryDefinition {
                include: vec![PathBuf::try_from(temp.path().to_path_buf()).unwrap()],
                exclude: Vec::new(),
                options: Default::default(),
            },
            None,
        )
        .unwrap();
    let nicegal_core::libraries::Created::New(library) = library else {
        panic!("new library expected")
    };
    std::fs::write(
        manager.resume_path(),
        serde_json::json!({
            "type": "thumbnailGenerate", "params": { "libraryId": library.id }
        })
        .to_string(),
    )
    .unwrap();
    manager.recover().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if manager.registry().active.is_none() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("recovered empty-library job should complete");
    let jobs = manager.list().jobs;
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].status, JobStatus::Completed, "{jobs:?}");
    assert!(!manager.resume_path().exists());
    let restarted = test_manager(&temp);
    restarted.recover().await.unwrap();
    assert!(restarted.list().jobs.is_empty());
}

fn test_manager(runtime_config_dir: &tempfile::TempDir) -> Arc<JobManager> {
    crate::api::tests::initialize_test_runtime();
    let current_dir = PathBuf::try_from(runtime_config_dir.path().to_path_buf()).unwrap();
    let thumbnail_path = current_dir.join("unused-thumbnails.db");
    let runtime = Arc::new(
        crate::api::RuntimeSettings::load(
            PathBuf::try_from(runtime_config_dir.path().join("runtime.json")).unwrap(),
            None,
        )
        .unwrap(),
    );
    Arc::new(JobManager::new(
        Arc::new(Databases {
            assets: current_dir.join("unused-assets.db"),
            images: current_dir.join("unused-images.db"),
            ocr: current_dir.join("unused-ocr.db"),
            thumbnails: thumbnail_path.clone(),
        }),
        Arc::new(ThumbnailService::new(&thumbnail_path).unwrap()),
        Arc::new(TextEmbedder::deferred(
            nicegal_core::embedding::TextEmbedderOptions::default(),
        )),
        Arc::new(ImageEmbedder::deferred(
            nicegal_core::embedding::ImageEmbedderOptions::default(),
        )),
        Arc::new(ImageQueryModel::deferred(
            nicegal_core::embedding::ImageQueryEmbedderOptions::default(),
        )),
        Arc::new(ocr_models::ModelStore::new(
            nicegal_core::runtime::ExecutionProvider::Cpu,
        )),
        runtime,
    ))
}

#[tokio::test]
async fn manager_rejects_concurrent_jobs_and_cancels_queued_work() {
    let runtime_config_dir = tempfile::TempDir::new().unwrap();
    let manager = test_manager(&runtime_config_dir);
    let request: JobRequest = serde_json::from_value(serde_json::json!({
        "type": "ocrModelLoad",
        "params": {
            "detection": { "modelId": "owner/detection" },
            "recognition": { "modelId": "owner/recognition" }
        }
    }))
    .unwrap();
    let job = manager.start(request.prepare().unwrap()).unwrap();

    let second: JobRequest = serde_json::from_value(serde_json::json!({
        "type": "ocrModelLoad",
        "params": {
            "detection": { "modelId": "owner/detection" },
            "recognition": { "modelId": "owner/recognition" }
        }
    }))
    .unwrap();
    assert!(manager.start(second.prepare().unwrap()).is_err());

    let scan = |params: serde_json::Value| {
        let request: JobRequest =
            serde_json::from_value(serde_json::json!({ "type": "libraryScan", "params": params }))
                .unwrap();
        manager.start(request.prepare().unwrap()).unwrap()
    };
    let queued = scan(serde_json::json!({ "libraryId": 7, "pendingOnly": true }));
    assert_eq!(queued.response().status, JobStatus::Queued);
    assert_eq!(queued.response().library_id, Some(7));
    let merged = scan(serde_json::json!({ "libraryId": 7, "retryFailed": true }));
    assert_eq!(
        merged.id, queued.id,
        "a request for the queued library merges"
    );
    assert_eq!(manager.registry().queued.len(), 1);
    let replacement = scan(serde_json::json!({ "libraryId": 8 }));
    assert_ne!(replacement.id, queued.id);
    assert_eq!(
        queued.response().status,
        JobStatus::Queued,
        "another library keeps its place"
    );
    assert_eq!(replacement.response().status, JobStatus::Queued);
    assert_eq!(manager.registry().queued.len(), 2);

    let mut updates = job.subscribe();
    manager.cancel_all();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !updates.borrow().status.is_terminal() {
            updates.changed().await.unwrap();
        }
    })
    .await
    .expect("cancelled job should reach a terminal state");
    assert_eq!(job.response().status, JobStatus::Cancelled);
}
