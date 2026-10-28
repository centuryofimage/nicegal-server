//! The backfill job that gives indexed OCR text its vectors.
//!
//! Runs independently or after OCR indexing without rescanning images.

use crate::api::jobs::cancel_if;
use anyhow::Context;
use nicegal_core::db::{DB, SearchFilters, TextEmbedding, TextEmbeddingSpace};
use nicegal_core::embedding::TextEmbedder;
use nicegal_core::index::{IndexEvent, IndexObserver, IndexProgressDelta};
use nicegal_core::scope::PathScope;
use serde::Deserialize;

use nicegal_core::assets::AssetCatalog;

use super::super::error::ApiError;
use super::super::{Databases, libraries};

/// Request-level bound; the embedder's own batch limit may be smaller.
const MAX_BATCH_SIZE: usize = 512;
/// Maximum OCR text bytes considered for one embedding input.
const MAX_CONTENT_BYTES: usize = 8192;
/// This job fills the OCR-text space. CLIP image vectors live in the separate image index, not in
/// this space.
const SPACE: TextEmbeddingSpace = TextEmbeddingSpace::OcrText;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    library_id: i64,
    /// Re-embed text that already has a current vector. Used after a model change that kept the
    /// same identifier, which the store cannot detect on its own.
    #[serde(default)]
    force: bool,
    /// Rows per embedder call. Omit to use the backend limit.
    batch_size: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct Spec {
    library_id: i64,
    options: TextEmbedOptions,
}

/// How a text embedding pass runs.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TextEmbedOptions {
    /// Re-embed text that already has a current vector.
    pub(crate) force: bool,
    /// Rows per embedder call; the embedder's own limit when absent or smaller.
    pub(crate) batch_size: Option<usize>,
    /// Debug guardrail: embed at most this many rows.
    pub(crate) limit: Option<usize>,
}

impl Spec {
    pub(crate) fn library_id(&self) -> i64 {
        self.library_id
    }

    /// The library's scope and how to embed it, read when the job runs.
    pub(crate) fn resolve(
        &self,
        databases: &Databases,
    ) -> anyhow::Result<(PathScope, TextEmbedOptions)> {
        let catalog = AssetCatalog::new_read_only(&databases.assets)?;
        let scope = libraries::stored(&catalog, self.library_id)?.scope();
        Ok((scope, self.options))
    }
}

pub(crate) fn prepare(request: Request) -> Result<Spec, ApiError> {
    // Parameters are checked before the filesystem is touched, so a bad batch size is reported as
    // itself rather than as whatever the root happens to be wrong about.
    if let Some(batch_size) = request.batch_size
        && (batch_size == 0 || batch_size > MAX_BATCH_SIZE)
    {
        return Err(ApiError::bad_request(format!(
            "text embedding batchSize must be between 1 and {MAX_BATCH_SIZE}"
        )));
    }
    Ok(Spec {
        library_id: request.library_id,
        options: TextEmbedOptions {
            force: request.force,
            batch_size: request.batch_size,
            limit: None,
        },
    })
}

/// The OCR text a pass embeds, from [`select`].
pub(crate) struct Backlog {
    filters: SearchFilters,
    /// Rows the pass will embed.
    count: usize,
    options: TextEmbedOptions,
}

impl Backlog {
    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Count the text under `scope` that needs a vector, before the model loads. This writes: it
/// declares this build's model, which creates the vector table on a fresh database and discards
/// vectors from another model. Those are useless to queries either way, and once discarded all
/// of their rows count as pending.
#[tracing::instrument(name = "select_text", level = "info", skip_all, fields(pending = tracing::field::Empty))]
pub(crate) fn select(
    databases: &Databases,
    scope: &PathScope,
    options: &TextEmbedOptions,
) -> anyhow::Result<Backlog> {
    let mut ocr = DB::new(&databases.ocr)?;
    ocr.set_text_embedding_model(
        SPACE,
        TextEmbedder::MODEL_ID,
        TextEmbedder::DIMENSIONS,
        true,
    )?;
    let filters = SearchFilters::new(scope.clone());
    let coverage = ocr.text_embedding_coverage(SPACE, &filters)?;
    let count = if options.force {
        coverage.with_text
    } else {
        coverage.pending
    };
    let count = options.limit.map_or(count, |limit| count.min(limit));
    tracing::Span::current().record("pending", count);
    Ok(Backlog {
        filters,
        count,
        options: *options,
    })
}

/// Embed `backlog` with a loaded `embedder`. An embedder error fails the pass: the rows stay
/// pending, so the next scan retries them.
pub(crate) fn run(
    databases: &Databases,
    embedder: &TextEmbedder,
    backlog: Backlog,
    observer: &dyn IndexObserver,
) -> anyhow::Result<()> {
    // The write connection is opened up front but takes no write lock by doing so: in WAL mode
    // SQLite acquires it at the first statement of a write transaction and releases it at commit.
    // The only write transactions here are one per saved batch, so the embedder's forward pass
    // (the slow part) always runs with no lock held and searches keep reading throughout.
    let mut ocr = DB::new(&databases.ocr)?;
    let Backlog {
        filters,
        count,
        options,
    } = backlog;
    if options.force {
        ocr.clear_text_embeddings(SPACE)?;
    }
    let batch_size = options
        .batch_size
        .unwrap_or_else(|| embedder.max_batch_size())
        .min(embedder.max_batch_size());

    observer.on_event(IndexEvent::Discovered { count });
    observer.on_event(IndexEvent::DiscoveryComplete { total: count });

    let mut remaining = options.limit;
    loop {
        cancel_if(observer.is_cancelled())?;
        let next_batch_size = remaining.map_or(batch_size, |limit| batch_size.min(limit));
        if next_batch_size == 0 {
            return Ok(());
        }
        let pending =
            ocr.pending_text_embeddings(SPACE, &filters, next_batch_size, MAX_CONTENT_BYTES)?;
        if pending.is_empty() {
            return Ok(());
        }

        let texts: Vec<&str> = pending.iter().map(|row| row.content.as_str()).collect();
        let vectors = embedder
            .embed_documents(&texts)
            .context("embedding OCR text failed")?;

        let items: Vec<TextEmbedding> = pending
            .iter()
            .zip(vectors)
            .map(|(row, vector)| TextEmbedding {
                asset_id: row.asset_id,
                vector,
            })
            .collect();
        let attempted = items.len();
        let stored = ocr.save_text_embeddings(SPACE, TextEmbedder::MODEL_ID, items)?;
        if let Some(limit) = &mut remaining {
            *limit -= attempted;
        }
        observer.on_event(IndexEvent::Progress(IndexProgressDelta {
            processed: attempted,
            phase_completed: attempted,
            embedded: stored,
            skipped: attempted - stored,
            ..IndexProgressDelta::default()
        }));
        if stored == 0 {
            // Every row in the batch lost its OCR row to a concurrent prune. Re-querying would
            // return the same empty-handed batch forever.
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> Result<Request, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn the_library_is_required_and_unknown_fields_are_rejected() {
        assert!(request(serde_json::json!({})).is_err());
        assert!(request(serde_json::json!({"libraryId": 1, "nope": 1})).is_err());
        assert!(request(serde_json::json!({"libraryId": 1})).is_ok());
    }

    #[test]
    fn the_batch_size_is_bounded() {
        for size in [0, MAX_BATCH_SIZE + 1] {
            let parsed = request(serde_json::json!({"libraryId": 1, "batchSize": size})).unwrap();
            let error = prepare(parsed).expect_err("an unbounded batch is rejected");
            assert!(error.message.contains("batchSize"), "{error:?}");
        }
    }
}
