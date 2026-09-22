//! Model-specific image-vector storage and ingestion.
//!
//! Each model owns a database file, so incompatible coordinate spaces never share a table. The
//! database tracks source fingerprints directly from the asset catalog; OCR success or failure has
//! no bearing on image-vector coverage.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use camino::{Utf8Path as Path, Utf8PathBuf as PathBuf};
use crossbeam_channel::{Receiver, Sender, bounded};
use image::RgbImage;
#[cfg(test)]
use rusqlite::Connection;
use tracing::{debug, info, instrument};

#[cfg(test)]
use crate::assets::SourceFingerprint;
use crate::assets::{Asset, AssetCatalog, MediaKind};
#[cfg(test)]
use crate::db::SearchFilters;
use crate::embedding::ImageEmbedder;
use crate::index::{
    IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta, aborted, send_unless_aborted,
};

mod storage;
use storage::StoredImageEmbedding;
pub use storage::{ImageIndexDb, ImageIndexReadSnapshot, ImageVectorHit, ImageVectorSearchOptions};

const MAX_DECODE_WORKERS: usize = 4;

/// Flags for an incremental image-indexing pass.
#[derive(Debug, Clone, Default)]
pub struct ImageIndexOptions {
    pub force: bool,
    pub retry_failed: bool,
    pub limit: Option<usize>,
    /// Required when indexing video samples, whose thumbnails accompany search results.
    pub thumbnail_database: Option<PathBuf>,
}

/// Incrementally embed cataloged images beneath `root`.
///
/// Decode workers overlap source I/O with a single batched inference lane. ONNX Runtime owns all
/// model execution parallelism; no session replicas are created.
#[instrument(
    name = "image_index",
    skip_all,
    fields(root = %root, model = %embedder.model(), sources = tracing::field::Empty)
)]
pub fn index_images_observed(
    catalog: &AssetCatalog,
    db: &mut ImageIndexDb,
    embedder: &ImageEmbedder,
    root: &Path,
    options: ImageIndexOptions,
    observer: &dyn IndexObserver,
) -> Result<bool> {
    let assets = catalog.under_root(root)?;
    index_catalog_images(catalog, db, embedder, &assets, options, observer)
}

/// Embed a supplied catalog snapshot, preserving its order when selecting pending work.
#[instrument(
    name = "image_index",
    skip_all,
    fields(root = %root, model = %embedder.model(), sources = tracing::field::Empty)
)]
pub fn index_catalog_images_observed(
    catalog: &AssetCatalog,
    db: &mut ImageIndexDb,
    embedder: &ImageEmbedder,
    root: &Path,
    assets: &[Asset],
    options: ImageIndexOptions,
    observer: &dyn IndexObserver,
) -> Result<bool> {
    index_catalog_images(catalog, db, embedder, assets, options, observer)
}

fn index_catalog_images(
    catalog: &AssetCatalog,
    db: &mut ImageIndexDb,
    embedder: &ImageEmbedder,
    assets: &[Asset],
    options: ImageIndexOptions,
    observer: &dyn IndexObserver,
) -> Result<bool> {
    let ImageIndexOptions {
        force,
        retry_failed,
        limit,
        thumbnail_database,
    } = options;
    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
    let images = assets
        .iter()
        .filter(|asset| is_embedding_image(asset))
        .cloned()
        .collect::<Vec<_>>();
    let mut thumbnails = None;
    let current = if force {
        // Retain the old complete set until its replacement is ready.
        HashSet::new()
    } else {
        let fingerprints = images
            .iter()
            .filter(|asset| asset.media_kind == MediaKind::Image)
            .map(|asset| (asset.asset_id, asset.fingerprint))
            .collect::<Vec<_>>();
        let mut current = db.current_asset_ids(&fingerprints)?;
        let videos = images
            .iter()
            .filter(|asset| asset.media_kind == MediaKind::Video)
            .map(|asset| (asset.asset_id, asset.fingerprint))
            .collect::<Vec<_>>();
        let video_current = db.current_video_asset_ids(&videos)?;
        if !video_current.is_empty() {
            let cache = crate::thumbs::ThumbnailDb::new(
                thumbnail_database
                    .as_deref()
                    .context("video indexing requires a thumbnail database")?,
            )?;
            current.extend(videos_with_complete_thumbnails(
                db,
                &cache,
                &images,
                &video_current,
            )?);
            thumbnails = Some(cache);
        }
        current
    };
    let decode_failed = if force || retry_failed {
        HashSet::new()
    } else {
        let fingerprints = images
            .iter()
            .map(|asset| (asset.asset_id, asset.fingerprint))
            .collect::<Vec<_>>();
        catalog.current_decode_failure_asset_ids(&fingerprints)?
    };
    let mut sources = images
        .into_iter()
        .filter(|asset| {
            !current.contains(&asset.asset_id) && !decode_failed.contains(&asset.asset_id)
        })
        .collect::<Vec<_>>();
    if let Some(limit) = limit {
        sources.truncate(limit);
    }
    tracing::Span::current().record("sources", sources.len());
    observer.on_event(IndexEvent::Discovered {
        count: sources.len(),
    });
    observer.on_event(IndexEvent::DiscoveryComplete {
        total: sources.len(),
    });
    if sources.is_empty() {
        return Ok(observer.is_cancelled());
    }

    if thumbnails.is_none()
        && sources
            .iter()
            .any(|asset| asset.media_kind == MediaKind::Video)
    {
        thumbnails = Some(crate::thumbs::ThumbnailDb::new(
            thumbnail_database
                .as_deref()
                .context("video indexing requires a thumbnail database")?,
        )?);
    }

    BoundedImagePipeline {
        catalog,
        db,
        embedder,
        observer,
        thumbnails: thumbnails.as_ref(),
    }
    .run(sources)
}

fn is_embedding_image(asset: &Asset) -> bool {
    asset.media_kind == MediaKind::Video
        || (asset.media_kind == MediaKind::Image
            && matches!(
                asset.media_format.as_str(),
                "png" | "jpeg" | "gif" | "webp" | "bmp"
            ))
}

fn videos_with_complete_thumbnails(
    db: &ImageIndexDb,
    cache: &crate::thumbs::ThumbnailDb,
    assets: &[Asset],
    vector_current: &HashSet<i64>,
) -> Result<HashSet<i64>> {
    let mut complete = HashSet::new();
    for asset in assets
        .iter()
        .filter(|asset| vector_current.contains(&asset.asset_id))
    {
        let timestamps = db.video_sample_timestamps(asset.asset_id)?;
        if cache.has_current_video_samples(asset, &timestamps)? {
            complete.insert(asset.asset_id);
        }
    }
    Ok(complete)
}

struct BoundedImagePipeline<'a> {
    catalog: &'a AssetCatalog,
    db: &'a mut ImageIndexDb,
    embedder: &'a ImageEmbedder,
    observer: &'a dyn IndexObserver,
    thumbnails: Option<&'a crate::thumbs::ThumbnailDb>,
}

impl BoundedImagePipeline<'_> {
    fn run(self, sources: Vec<Asset>) -> Result<bool> {
        let batch_size = self.embedder.max_batch_size().min(sources.len());
        let decode_workers = decode_worker_count().min(sources.len());
        // One normalized batch may wait while one batch is inferred. Decode workers use the
        // model-owned FastEmbed preprocessor, then discard their full-resolution source image.
        let (outcome_sender, outcome_receiver) = bounded::<DecodeOutcome>(batch_size);
        let (abort_sender, abort_receiver) = bounded::<()>(1);
        let next = AtomicUsize::new(0);
        let mut pending = Vec::with_capacity(batch_size);
        let mut fatal = None;
        let mut write_error = None;
        let mut cancelled = false;

        std::thread::scope(|scope| {
            let mut workers = Vec::with_capacity(decode_workers);
            let mut abort_sender = Some(abort_sender);
            for _ in 0..decode_workers {
                let outcomes = outcome_sender.clone();
                let abort = abort_receiver.clone();
                let sources = &sources;
                let next = &next;
                workers.push(scope.spawn(move || {
                    guard_decode_worker(&outcomes, &abort, || {
                        decode_sources(
                            sources,
                            next,
                            self.embedder,
                            self.observer,
                            &outcomes,
                            &abort,
                        );
                    });
                }));
            }
            drop(outcome_sender);

            while let Ok(outcome) = outcome_receiver.recv() {
                if !cancelled && self.observer.is_cancelled() {
                    cancelled = true;
                    pending.clear();
                    abort_sender.take();
                }
                let stopping = cancelled || write_error.is_some() || fatal.is_some();
                match outcome {
                    DecodeOutcome::Success(decoded) if !stopping => {
                        pending.push(decoded);
                        if pending.iter().map(|item| item.pixels.len()).sum::<usize>() >= batch_size
                            && let Err(error) = flush_batch(
                                self.db,
                                self.thumbnails,
                                self.embedder,
                                self.observer,
                                &mut pending,
                            )
                        {
                            write_error = Some(error);
                            abort_sender.take();
                        }
                    }
                    DecodeOutcome::Failure {
                        asset,
                        message,
                        cache_decode_failure,
                    } if !stopping => {
                        self.observer.on_event(IndexEvent::ActiveAsset {
                            path: asset.path.clone(),
                            active: false,
                        });
                        let recorded = if cache_decode_failure {
                            self.catalog.record_decode_failure(&asset)
                        } else {
                            Ok(())
                        };
                        match recorded {
                            Ok(()) => report_failure(self.observer, asset.path, message),
                            Err(error) => {
                                write_error = Some(error);
                                abort_sender.take();
                            }
                        }
                    }
                    DecodeOutcome::Fatal(message) => {
                        fatal.get_or_insert(message);
                        abort_sender.take();
                    }
                    DecodeOutcome::Success(_) | DecodeOutcome::Failure { .. } => {}
                }
            }

            drop(abort_sender);
            for worker in workers {
                if worker.join().is_err() {
                    fatal.get_or_insert_with(|| "an image decode worker panicked".to_owned());
                }
            }
        });

        if let Some(error) = write_error {
            return Err(error);
        }
        if let Some(message) = fatal {
            bail!("{message}");
        }
        cancelled |= self.observer.is_cancelled();
        if !cancelled {
            flush_batch(
                self.db,
                self.thumbnails,
                self.embedder,
                self.observer,
                &mut pending,
            )?;
        }
        info!(
            decode_workers,
            batch_size,
            cancelled = self.observer.is_cancelled(),
            "image embedding indexing complete"
        );
        Ok(cancelled)
    }
}

struct DecodedImage {
    asset: Asset,
    pixels: Vec<ndarray::Array3<f32>>,
    samples: Vec<crate::video::VideoSample>,
}

enum DecodeOutcome {
    Success(DecodedImage),
    Failure {
        asset: Asset,
        message: String,
        cache_decode_failure: bool,
    },
    Fatal(String),
}

fn decode_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .saturating_sub(1)
        .clamp(1, MAX_DECODE_WORKERS)
}

fn decode_sources(
    sources: &[Asset],
    next: &AtomicUsize,
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
    outcomes: &Sender<DecodeOutcome>,
    abort: &Receiver<()>,
) {
    loop {
        if observer.is_cancelled() || aborted(abort) {
            break;
        }
        let index = next.fetch_add(1, Ordering::Relaxed);
        let Some(asset) = sources.get(index) else {
            break;
        };
        observer.on_event(IndexEvent::ActiveAsset {
            path: asset.path.clone(),
            active: true,
        });
        let preparation = if asset.media_kind == MediaKind::Video {
            prepare_video(asset, embedder, || {
                observer.is_cancelled() || aborted(abort)
            })
        } else {
            prepare_image(&asset.path, embedder.model().is_deepghs(), |image| {
                embedder.preprocess_image(image)
            })
            .map(|pixels| DecodedImage {
                asset: asset.clone(),
                pixels: vec![pixels],
                samples: Vec::new(),
            })
        };
        if observer.is_cancelled() || aborted(abort) {
            break;
        }
        let outcome = match preparation {
            Ok(decoded) => DecodeOutcome::Success(decoded),
            Err(PreparationError::Decode(error))
                if crate::index::is_missing_source_error(&error) =>
            {
                crate::index::report_disappeared_source(observer, &asset.path);
                continue;
            }
            Err(PreparationError::Decode(error)) => DecodeOutcome::Failure {
                asset: asset.clone(),
                message: format!("decoding image for embedding failed: {error:#}"),
                // Video timeouts and codec limitations must remain retryable, and must
                // never poison the shared still-image/OCR decode-failure cache.
                cache_decode_failure: asset.media_kind == MediaKind::Image,
            },
            Err(PreparationError::Preprocess(error)) => DecodeOutcome::Failure {
                asset: asset.clone(),
                message: format!("preprocessing image for embedding failed: {error:#}"),
                cache_decode_failure: false,
            },
        };
        if !send_unless_aborted(outcomes, outcome, abort) {
            break;
        }
    }
}

#[instrument(
    name = "decode_image",
    level = "debug",
    skip_all,
    fields(path = %path)
)]
fn decode_image(path: &Path, white_background: bool) -> Result<RgbImage> {
    let data = std::fs::read(path).with_context(|| format!("opening image: {path}"))?;
    let raster = crate::imaging::decode_accurate(&data)
        .with_context(|| format!("decoding image: {path}"))?;
    let (width, height) = (raster.width(), raster.height());
    let pixels = if white_background {
        raster.flatten_rgb([255, 255, 255])
    } else {
        raster.into_rgb_bytes()
    };
    RgbImage::from_raw(width, height, pixels).context("assembling decoded image buffer")
}

enum PreparationError {
    Decode(anyhow::Error),
    Preprocess(anyhow::Error),
}

fn prepare_image(
    path: &Path,
    white_background: bool,
    preprocess: impl FnOnce(RgbImage) -> Result<ndarray::Array3<f32>>,
) -> Result<ndarray::Array3<f32>, PreparationError> {
    let image = decode_image(path, white_background).map_err(PreparationError::Decode)?;
    // A model-specific transform failure says nothing about whether OCR can decode the file.
    preprocess(image).map_err(PreparationError::Preprocess)
}

fn prepare_video(
    asset: &Asset,
    embedder: &ImageEmbedder,
    cancelled: impl Fn() -> bool,
) -> Result<DecodedImage, PreparationError> {
    let samples = crate::video::samples(&asset.path, crate::video::SAMPLE_MAX_EDGE, cancelled)
        .map_err(PreparationError::Decode)?;
    let pixels = samples
        .iter()
        .map(|sample| {
            let image = RgbImage::from_raw(
                sample.raster.width(),
                sample.raster.height(),
                sample.raster.pixels().to_vec(),
            )
            .context("assembling video sample for inference")?;
            embedder.preprocess_image(image)
        })
        .collect::<Result<Vec<_>>>()
        .map_err(PreparationError::Preprocess)?;
    Ok(DecodedImage {
        asset: asset.clone(),
        pixels,
        samples,
    })
}

fn flush_batch(
    db: &mut ImageIndexDb,
    thumbnails: Option<&crate::thumbs::ThumbnailDb>,
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
    pending: &mut Vec<DecodedImage>,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let mut batch = std::mem::take(pending);
    let counts = batch
        .iter()
        .map(|decoded| decoded.pixels.len())
        .collect::<Vec<_>>();
    let pixels = batch
        .iter_mut()
        .flat_map(|decoded| std::mem::take(&mut decoded.pixels))
        .collect::<Vec<_>>();
    // Never exceed the model's batch limit even when a video yields several inputs.
    let vectors = match embed_sample_batches(embedder, observer, pixels) {
        Ok(vectors) => vectors,
        Err(error) => {
            let message = format!("image embedding inference failed: {error:#}");
            for decoded in &batch {
                let asset = &decoded.asset;
                observer.on_event(IndexEvent::ActiveAsset {
                    path: asset.path.clone(),
                    active: false,
                });
                report_failure(observer, asset.path.clone(), message.clone());
            }
            return Err(error).context("running the image embedding batch");
        }
    };
    let attempted = batch.len();
    let active_paths = batch
        .iter()
        .map(|decoded| decoded.asset.path.clone())
        .collect::<Vec<_>>();
    let mut vectors = vectors.into_iter();
    let mut images = Vec::new();
    let mut stored = 0;
    let mut reported_failures = 0;
    for (decoded, count) in batch.into_iter().zip(counts) {
        let asset = decoded.asset;
        let sample_vectors = vectors.by_ref().take(count).collect::<Vec<_>>();
        if observer.is_cancelled() {
            break;
        }
        let current = std::fs::metadata(&asset.path)
            .ok()
            .and_then(|metadata| crate::assets::SourceFingerprint::from_metadata(&metadata).ok());
        if current != Some(asset.fingerprint) {
            reported_failures += 1;
            report_failure(
                observer,
                asset.path,
                "source changed during visual indexing; retry on next scan".to_owned(),
            );
            continue;
        }
        if asset.media_kind == MediaKind::Video {
            thumbnails
                .context("video indexing requires thumbnail storage")?
                .store_video_samples(&asset, &decoded.samples)?;
            let samples = decoded
                .samples
                .into_iter()
                .map(|sample| sample.timestamp_ms)
                .zip(sample_vectors)
                .collect();
            db.save_video_embeddings(&asset, samples)?;
            stored += 1;
        } else {
            let vector = sample_vectors
                .into_iter()
                .next()
                .context("image inference returned no vector")?;
            images.push(StoredImageEmbedding {
                asset_id: asset.asset_id,
                path: asset.path,
                fingerprint: asset.fingerprint,
                vector,
            });
        }
    }
    if !observer.is_cancelled() {
        stored += db.save_embeddings(images)?;
    }
    observer.on_event(IndexEvent::Progress(IndexProgressDelta {
        processed: attempted - reported_failures,
        phase_completed: attempted - reported_failures,
        embedded: stored,
        skipped: attempted - stored - reported_failures,
        ..IndexProgressDelta::default()
    }));
    for path in active_paths {
        observer.on_event(IndexEvent::ActiveAsset {
            path,
            active: false,
        });
    }
    debug!(stored, "saved an image embedding batch");
    Ok(())
}

fn embed_sample_batches(
    embedder: &ImageEmbedder,
    observer: &dyn IndexObserver,
    pixels: Vec<ndarray::Array3<f32>>,
) -> Result<Vec<Vec<f32>>> {
    let mut pixels = pixels.into_iter();
    let mut vectors = Vec::new();
    loop {
        if observer.is_cancelled() {
            return Ok(Vec::new());
        }
        let chunk = pixels
            .by_ref()
            .take(embedder.max_batch_size())
            .collect::<Vec<_>>();
        if chunk.is_empty() {
            break;
        }
        vectors.extend(embedder.embed_preprocessed_images(chunk)?);
    }
    Ok(vectors)
}

fn report_failure(observer: &dyn IndexObserver, path: PathBuf, message: String) {
    crate::index::report_item_failure(observer, &path, message);
}

fn guard_decode_worker(
    outcomes: &Sender<DecodeOutcome>,
    abort: &Receiver<()>,
    body: impl FnOnce(),
) {
    let Some(message) = crate::index::worker_panic(body) else {
        return;
    };
    send_unless_aborted(outcomes, DecodeOutcome::Fatal(message), abort);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn coverage_excludes_stale_vectors_videos_and_other_roots() -> Result<()> {
        let temp = TempDir::new()?;
        let root = PathBuf::from("C:/gallery");
        let rows = vec![
            (1, root.join("current.png"), 10, None, 100),
            (2, root.join("changed.png"), 20, None, 100),
            (3, root.join("missing.png"), 10, None, 100),
            (
                4,
                PathBuf::from("C:/gallery-other/outside.png"),
                10,
                None,
                100,
            ),
            (5, root.join("video.mp4"), 10, None, 100),
            (6, root.join("resized.png"), 10, None, 200),
        ];
        let catalog = catalog_with_rows(&temp, &rows)?;
        Connection::open(&catalog)?.execute(
            "UPDATE assets SET media_kind = 'video' WHERE asset_id = 5",
            [],
        )?;
        let vectors = [1, 2, 4, 6].map(|id| {
            (
                id,
                rows[(id - 1) as usize].1.clone(),
                10,
                100,
                vec![1.0, 0.0],
            )
        });
        let index = image_index_with_vectors(&temp, 2, &vectors)?;
        let db = ImageIndexDb::new_read_only(&index, 2, &catalog)?;
        assert_eq!(db.coverage(&SearchFilters::new(&root))?, (5, 1));
        assert_eq!(
            db.coverage(&SearchFilters::new(Path::new("C:/empty")))?,
            (0, 0)
        );
        Ok(())
    }

    #[test]
    fn preprocessing_errors_do_not_classify_valid_images_as_decode_failures() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("valid.jpg"))?;
        std::fs::write(
            &path,
            crate::imaging::test_support::jpeg_with_orientation(1)?,
        )?;
        assert!(matches!(
            prepare_image(&path, false, |_| anyhow::bail!("model transform failed")),
            Err(PreparationError::Preprocess(_))
        ));
        std::fs::write(&path, b"invalid image")?;
        assert!(matches!(
            prepare_image(&path, false, |_| panic!(
                "invalid pixels cannot reach the preprocessor"
            )),
            Err(PreparationError::Decode(_))
        ));
        Ok(())
    }

    use crate::assets::Timeline;
    use crate::db::TimeRange;

    /// One catalog row: id, path, `source_modified_ns`, `exif_taken_ns`, size.
    type CatalogRow = (i64, PathBuf, i64, Option<i64>, u64);

    /// Create an asset catalog with the given rows, exactly as the indexing pipeline would have
    /// cataloged them. The catalog's upsert API fingerprints real files, which a vector search
    /// fixture does not need; the row shape it stores is what search joins against.
    fn catalog_with_rows(temp: &TempDir, rows: &[CatalogRow]) -> Result<PathBuf> {
        let path = PathBuf::try_from(temp.path().join("assets.db"))?;
        drop(AssetCatalog::new(&path)?);
        let conn = Connection::open(&path)?;
        for (asset_id, path, modified_ns, exif_taken_ns, size) in rows {
            conn.execute(
                "INSERT INTO assets(asset_id, path, source_modified_ns, exif_taken_ns, \
                     source_size, media_kind, media_format, is_animated) \
                 VALUES (?1, ?2, ?3, ?4, ?5, 'image', 'png', 0)",
                (
                    asset_id,
                    path.as_str(),
                    modified_ns,
                    exif_taken_ns,
                    *size as i64,
                ),
            )?;
        }
        Ok(path)
    }

    /// An image index whose state records the fingerprint each vector was computed from.
    fn image_index_with_vectors(
        temp: &TempDir,
        dimensions: usize,
        vectors: &[(i64, PathBuf, i64, u64, Vec<f32>)],
    ) -> Result<PathBuf> {
        let path = PathBuf::try_from(temp.path().join("images.db"))?;
        let mut db = ImageIndexDb::new(&path, dimensions)?;
        db.save_embeddings(
            vectors
                .iter()
                .map(
                    |(asset_id, path, modified_ns, size, vector)| StoredImageEmbedding {
                        asset_id: *asset_id,
                        path: path.clone(),
                        fingerprint: SourceFingerprint {
                            modified_ns: *modified_ns,
                            size: *size,
                        },
                        vector: vector.clone(),
                    },
                )
                .collect(),
        )?;
        Ok(path)
    }

    #[test]
    fn model_database_tracks_fingerprints_and_replaces_vectors() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("clip.db"))?;
        let mut db = ImageIndexDb::new(&path, 3)?;
        let fingerprint = SourceFingerprint {
            modified_ns: 10,
            size: 20,
        };
        db.save_embeddings(vec![StoredImageEmbedding {
            asset_id: 1,
            path: PathBuf::from("C:/gallery/a.png"),
            fingerprint,
            vector: vec![1.0, 0.0, 0.0],
        }])?;
        assert_eq!(db.vector_count()?, 1);
        assert_eq!(
            db.current_asset_ids(&[(1, fingerprint)])?,
            HashSet::from([1])
        );

        db.save_embeddings(vec![StoredImageEmbedding {
            asset_id: 1,
            path: PathBuf::from("C:/gallery/a.png"),
            fingerprint: SourceFingerprint {
                modified_ns: 11,
                size: 20,
            },
            vector: vec![0.0, 1.0, 0.0],
        }])?;
        assert_eq!(db.vector_count()?, 1);
        assert!(db.current_asset_ids(&[(1, fingerprint)])?.is_empty());
        Ok(())
    }

    #[test]
    fn deleting_assets_removes_clip_vectors_and_fingerprint_state() -> Result<()> {
        let temp = TempDir::new()?;
        let path = image_index_with_vectors(
            &temp,
            3,
            &[
                (
                    1,
                    PathBuf::from("C:/gallery/deleted.png"),
                    10,
                    20,
                    vec![1.0, 0.0, 0.0],
                ),
                (
                    2,
                    PathBuf::from("C:/gallery/retained.png"),
                    10,
                    20,
                    vec![0.0, 1.0, 0.0],
                ),
            ],
        )?;
        let mut db = ImageIndexDb::new(&path, 3)?;
        assert_eq!(db.delete_assets(&[1])?, 1);
        assert_eq!(db.vector_count()?, 1);
        assert!(
            db.current_asset_ids(&[(
                1,
                SourceFingerprint {
                    modified_ns: 10,
                    size: 20
                }
            )])?
            .is_empty()
        );
        let vectors: i64 = db.conn.query_row(
            "SELECT count(*) FROM image_embeddings WHERE asset_id = 1",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(vectors, 0);
        assert_eq!(db.delete_assets(&[1])?, 0);
        Ok(())
    }

    #[test]
    fn database_rejects_a_different_vector_width() -> Result<()> {
        let temp = TempDir::new()?;
        let path = PathBuf::try_from(temp.path().join("clip.db"))?;
        drop(ImageIndexDb::new(&path, 3)?);
        assert!(ImageIndexDb::new(&path, 4).is_err());
        Ok(())
    }

    /// The shared fixture: five cataloged images, of which one lives outside the searched root,
    /// one inside the excluded subtree, and one whose recorded fingerprint no longer matches the
    /// catalog's row for it.
    fn search_fixture(temp: &TempDir) -> Result<(PathBuf, PathBuf, PathBuf)> {
        let root = PathBuf::try_from(temp.path().join("gallery"))?;
        let outside = PathBuf::try_from(temp.path().join("elsewhere"))?;
        let catalog = catalog_with_rows(
            temp,
            &[
                (1, root.join("a.png"), 100, None, 10),
                (2, root.join("b.png"), 200, Some(50), 20),
                (3, root.join("skip").join("c.png"), 300, None, 30),
                (4, outside.join("d.png"), 400, None, 40),
                (5, root.join("e.png"), 500, None, 50),
            ],
        )?;
        // Asset 5's vector was computed from a fingerprint the catalog no longer has, so search
        // must drop it like the OCR store drops a re-OCR'd row's vector.
        let images = image_index_with_vectors(
            temp,
            3,
            &[
                (1, root.join("a.png"), 100, 10, vec![1.0, 0.0, 0.0]),
                (2, root.join("b.png"), 200, 20, vec![0.0, 1.0, 0.0]),
                (
                    3,
                    root.join("skip").join("c.png"),
                    300,
                    30,
                    vec![0.0, 1.0, 0.0],
                ),
                (4, outside.join("d.png"), 400, 40, vec![1.0, 0.0, 0.0]),
                (5, root.join("e.png"), 999, 50, vec![1.0, 0.0, 0.0]),
            ],
        )?;
        Ok((catalog, images, root))
    }

    #[test]
    fn image_search_orders_neighbours_and_drops_stale_and_outside_rows() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            100,
            &ImageVectorSearchOptions::default(),
        )?;

        // Asset 4 is outside the root and asset 5 is stale; both drop out of the total. The two
        // orthogonal neighbours tie on distance and break on recency.
        assert_eq!(total, 3);
        let ids = hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 3, 2]);
        assert!(hits[0].distance < 1e-6, "{:?}", hits[0].distance);
        Ok(())
    }

    #[test]
    fn current_query_vector_requires_a_matching_catalog_fingerprint() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, _) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;

        assert_eq!(reader.current_vector(1)?, Some(vec![1.0, 0.0, 0.0]));
        // Asset 5 is still in both tables but its catalog fingerprint changed after embedding.
        assert_eq!(reader.current_vector(5)?, None);
        assert_eq!(reader.current_vector(999)?, None);
        Ok(())
    }

    #[test]
    fn a_component_and_its_image_search_share_one_catalog_snapshot() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let mut snapshot = reader.begin_read_snapshot()?;

        // This first read establishes the image index + attached catalog view for the component.
        assert_eq!(snapshot.current_vector(1)?, Some(vec![1.0, 0.0, 0.0]));
        let writer = Connection::open(&catalog)?;
        writer.execute(
            "UPDATE assets SET source_modified_ns = 101 WHERE asset_id = 1",
            [],
        )?;

        // The request consistently sees the old-current pair rather than accepting the old
        // component then mixing it with a newly stale result query.
        assert_eq!(snapshot.current_vector(1)?, Some(vec![1.0, 0.0, 0.0]));
        let (total, hits) = snapshot.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            100,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert_eq!(hits[0].asset_id, 1);
        snapshot.commit()?;

        // A later request establishes a new view and rejects the now-stale component.
        assert_eq!(reader.current_vector(1)?, None);
        Ok(())
    }

    #[test]
    fn image_search_honours_the_exclude_the_ceiling_the_limit_and_the_timeline() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let query = [1.0, 0.0, 0.0];

        let excluded = root.join("skip");
        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root).with_exclude(Some(excluded.as_str())),
            100,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2, "the excluded subtree does not answer");
        assert_eq!(
            hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1, 2]
        );

        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root),
            100,
            &ImageVectorSearchOptions {
                max_distance: Some(0.5),
            },
        )?;
        assert_eq!(total, 1, "the ceiling drops the orthogonal neighbours");
        assert_eq!(
            hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1]
        );

        // `limit` caps the returned rows without changing how many matched.
        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root),
            2,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert_eq!(hits.len(), 2);

        // Capture is `COALESCE(exif_taken_ns, source_modified_ns)`, so the after bound drops
        // asset 2 (taken at 50) and keeps assets 1 and 3.
        let time = TimeRange {
            timeline: Timeline::Capture,
            after_ns: Some(60),
            before_ns: None,
        };
        let (total, hits) = reader.search_vectors(
            &query,
            &SearchFilters::new(&root).with_time(Some(time)),
            100,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 2);
        assert_eq!(
            hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1, 3]
        );
        Ok(())
    }

    #[test]
    fn image_search_preserves_totals_when_no_hits_are_returned() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        let (total, hits) = reader.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            0,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        assert!(hits.is_empty());

        let (total, hits) = reader.search_vectors(
            &[-1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            10,
            &ImageVectorSearchOptions {
                max_distance: Some(0.0),
            },
        )?;
        assert_eq!(total, 0);
        assert!(hits.is_empty());
        Ok(())
    }

    #[test]
    fn image_search_refuses_a_query_vector_of_the_wrong_width() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        assert!(
            reader
                .search_vectors(
                    &[1.0, 0.0],
                    &SearchFilters::new(&root),
                    10,
                    &ImageVectorSearchOptions::default(),
                )
                .is_err(),
            "a vector from another width is not comparable"
        );
        Ok(())
    }

    #[test]
    fn a_read_only_search_connection_cannot_write_the_attached_catalog() -> Result<()> {
        let temp = TempDir::new()?;
        let (catalog, images, root) = search_fixture(&temp)?;
        let reader = ImageIndexDb::new_read_only(&images, 3, &catalog)?;
        // Read-only flags propagate to attached databases; if they ever stop doing so, this
        // fails open and the reader could mutate the catalog.
        assert!(
            reader
                .conn
                .execute(
                    "INSERT INTO catalog.assets(asset_id, path, source_modified_ns, \
                         source_size, media_kind, media_format, is_animated) \
                     VALUES (99, 'x', 1, 1, 'image', 'png', 0)",
                    [],
                )
                .is_err(),
            "the attached catalog must be read-only"
        );
        let (total, _) = reader.search_vectors(
            &[1.0, 0.0, 0.0],
            &SearchFilters::new(&root),
            10,
            &ImageVectorSearchOptions::default(),
        )?;
        assert_eq!(total, 3);
        Ok(())
    }
}
