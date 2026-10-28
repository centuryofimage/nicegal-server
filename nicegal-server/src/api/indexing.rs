//! The index stages a library scan runs after cataloging. The image and text embedding jobs run
//! their stage alone.
//!
//! Every stage has the same shape: enter its phase, fetch its query encoder's files (OCR has
//! none), select the pending work, and load the indexing model only when there is some. Loading a
//! model is a detour through `loadingModels` that the job returns from by itself, so a stage sets
//! its phase once. A library with nothing a stage could index skips that stage entirely.

use nicegal_core::assets::{Asset, AssetCatalog};
use nicegal_core::db::DB;
use nicegal_core::image_index::{ImageIndexOptions, is_embedding_candidate, select_catalog_images};
use nicegal_core::index::{
    IndexEvent, IndexObserver, IndexPhase, OcrIndexOptions, recognize_selection, select_ocr_sources,
};
use nicegal_core::scope::PathScope;
use nicegal_core::thumbs::ThumbnailService;

use super::jobs::Job;
use super::models::Models;
use super::text_embeddings::job::{self as text, TextEmbedOptions};
use super::{Databases, image_embeddings};

/// The shared services stages draw on; owned by the job manager.
pub(super) struct Services<'a> {
    pub(super) databases: &'a Databases,
    pub(super) thumbnails: &'a ThumbnailService,
    pub(super) models: &'a Models,
}

/// Give every image, and every video when `options.index_videos` is set, among `assets` a current
/// image vector. `assets` are a library's cataloged files.
pub(super) fn embed_images(
    services: &Services<'_>,
    assets: &[Asset],
    options: &ImageIndexOptions,
    job: &Job,
) -> anyhow::Result<()> {
    // A library without images downloads nothing, including for an explicit backfill.
    if !assets
        .iter()
        .any(|asset| is_embedding_candidate(asset, options.index_videos))
    {
        return Ok(());
    }
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::ImageEmbedding));
    // Searches never download, so the stage fetches the paired query encoder's files even when
    // every image is indexed. Building it is left to the first search that needs it.
    services.models.ensure_image_query_available(job)?;
    let databases = services.databases;
    let sources = select_catalog_images(
        &AssetCatalog::new_read_only(&databases.assets)?,
        &databases.open_images_read_only(services.models.image.dimensions())?,
        Some(&databases.thumbnails),
        assets,
        options,
    )?;
    if sources.is_empty() {
        return Ok(());
    }
    let model = services.models.image.prepare(job)?;
    job.check_cancelled()?;
    image_embeddings::run(services, &model, sources, job)
}

/// Recognize the text of every image among `assets`, a library's cataloged files, that lacks a
/// current OCR result.
pub(super) fn recognize_text(
    services: &Services<'_>,
    assets: &[Asset],
    options: &OcrIndexOptions,
    job: &Job,
) -> anyhow::Result<()> {
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::Ocr));
    let catalog = AssetCatalog::new(&services.databases.assets)?;
    let mut ocr = DB::new(&services.databases.ocr)?;
    let selection = select_ocr_sources(&catalog, &ocr, assets, options)?;
    if selection.is_empty() {
        return Ok(());
    }
    job.check_cancelled()?;
    let models = services.models.ocr.prepare(job)?;
    job.check_cancelled()?;
    recognize_selection(
        &catalog,
        &mut ocr,
        &mut models.lock(),
        selection,
        options,
        job,
    )?;
    job.check_cancelled()
}

/// Give the OCR text under `scope` current vectors for text search.
pub(super) fn embed_text(
    services: &Services<'_>,
    scope: &PathScope,
    options: &TextEmbedOptions,
    job: &Job,
) -> anyhow::Result<()> {
    job.on_event(IndexEvent::PhaseChanged(IndexPhase::TextEmbedding));
    let models = services.models;
    models.text_query.ensure_available(job)?;
    let backlog = text::select(services.databases, scope, options)?;
    if backlog.is_empty() {
        return Ok(());
    }
    let model = models.text_index.prepare(job)?;
    job.check_cancelled()?;
    text::run(services.databases, &model, backlog, job)
}
