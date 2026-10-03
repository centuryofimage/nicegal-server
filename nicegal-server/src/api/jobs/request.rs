//! Job request envelopes and the validated specs they prepare.
use serde::Deserialize;

use super::job::JobKind;
use crate::api::error::ApiError;
use crate::api::{
    image_embeddings, library_scan, ocr_models, prune_jobs, text_embeddings, thumbnails,
};

#[derive(Debug, Deserialize)]
#[serde(
    tag = "type",
    content = "params",
    rename_all = "camelCase",
    deny_unknown_fields
)]
pub(super) enum JobRequest {
    ModelPrepare(EmptyParams),
    OcrModelLoad(ocr_models::job::Request),
    LibraryScan(library_scan::Request),
    ThumbnailGenerate(thumbnails::job::Request),
    TextEmbed(text_embeddings::job::Request),
    ImageEmbed(image_embeddings::Request),
    PruneMissing(prune_jobs::Request),
    LibraryPurge(prune_jobs::LibraryPurgeRequest),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EmptyParams {}

pub(in crate::api) enum JobSpec {
    ModelPrepare,
    OcrModelLoad(ocr_models::job::Spec),
    LibraryScan(library_scan::Spec),
    ThumbnailGenerate(thumbnails::job::Spec),
    TextEmbed(text_embeddings::job::Spec),
    ImageEmbed(image_embeddings::Spec),
    PruneMissing(prune_jobs::Spec),
    LibraryPurge(prune_jobs::LibraryPurgeSpec),
}

impl JobRequest {
    pub(super) fn prepare(self) -> Result<JobSpec, ApiError> {
        match self {
            Self::ModelPrepare(_) => Ok(JobSpec::ModelPrepare),
            Self::OcrModelLoad(request) => {
                Ok(JobSpec::OcrModelLoad(ocr_models::job::prepare(request)?))
            }
            Self::LibraryScan(request) => Ok(JobSpec::LibraryScan(library_scan::prepare(request)?)),
            Self::ThumbnailGenerate(request) => Ok(JobSpec::ThumbnailGenerate(
                thumbnails::job::prepare(request)?,
            )),
            Self::TextEmbed(request) => {
                Ok(JobSpec::TextEmbed(text_embeddings::job::prepare(request)?))
            }
            Self::ImageEmbed(request) => {
                Ok(JobSpec::ImageEmbed(image_embeddings::prepare(request)?))
            }
            Self::PruneMissing(request) => Ok(JobSpec::PruneMissing(prune_jobs::prepare(request)?)),
            Self::LibraryPurge(request) => Ok(JobSpec::LibraryPurge(
                prune_jobs::prepare_library_purge(request)?,
            )),
        }
    }
}

impl JobSpec {
    /// The library the job works on, which must exist when the job is accepted.
    pub(super) fn library_id(&self) -> Option<i64> {
        match self {
            Self::LibraryScan(spec) => Some(spec.library_id()),
            Self::ThumbnailGenerate(spec) => Some(spec.library_id()),
            Self::TextEmbed(spec) => spec.library_id(),
            Self::ImageEmbed(spec) => spec.library_id(),
            Self::PruneMissing(spec) => Some(spec.library_id()),
            Self::LibraryPurge(spec) => Some(spec.library_id()),
            Self::ModelPrepare | Self::OcrModelLoad(_) => None,
        }
    }

    /// Only library scans wait behind a running job; every other kind is refused while one runs.
    pub(super) fn queues_behind_active(&self) -> bool {
        matches!(self, Self::LibraryScan(_))
    }

    pub(super) fn kind(&self) -> JobKind {
        match self {
            Self::ModelPrepare => JobKind::ModelPrepare,
            Self::OcrModelLoad(_) => JobKind::OcrModelLoad,
            Self::LibraryScan(_) => JobKind::LibraryScan,
            Self::ThumbnailGenerate(_) => JobKind::ThumbnailGenerate,
            Self::TextEmbed(_) => JobKind::TextEmbed,
            Self::ImageEmbed(_) => JobKind::ImageEmbed,
            Self::PruneMissing(_) => JobKind::PruneMissing,
            Self::LibraryPurge(_) => JobKind::LibraryPurge,
        }
    }
}
