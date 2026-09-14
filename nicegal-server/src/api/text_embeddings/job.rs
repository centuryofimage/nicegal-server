//! The backfill job that gives indexed OCR text its vectors.
//!
//! The standalone backfill can also be the final phase of an OCR index job. A model swap can still
//! run this job on its own without re-scanning images.

use camino::Utf8PathBuf as PathBuf;
use nicegal_core::db::{DB, SearchFilters, TextEmbedding, TextEmbeddingSpace};
use nicegal_core::embedding::TextEmbedder;
use nicegal_core::index::{IndexEvent, IndexObserver, IndexPhase, IndexProgressDelta};
use serde::Deserialize;

use super::super::error::ApiError;
use super::super::roots;

/// The upper bound the *request* may ask for. The effective batch is additionally clamped to
/// [`TextEmbedder::max_batch_size`], which is where the real limit lives: the backend sizes its
/// buffers from it, and a batch is the unit it parallelises across its own threads.
const MAX_BATCH_SIZE: usize = 512;
/// OCR text past this point is almost always repeated page furniture, and every model truncates
/// far below it anyway. Capping here keeps a pathological scan from dominating a batch.
const MAX_CONTENT_BYTES: usize = 8192;
/// This job fills the OCR-text space. CLIP image vectors live in the separate image index, not in
/// this space.
const SPACE: TextEmbeddingSpace = TextEmbeddingSpace::OcrText;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    root: PathBuf,
    /// Re-embed text that already has a current vector. Used after a model change that kept the
    /// same identifier, which the store cannot detect on its own.
    #[serde(default)]
    force: bool,
    /// Rows handed to the embedder at once. Omit to use whatever the backend prefers, which is the
    /// right answer unless you are deliberately trading throughput for cancellation latency.
    batch_size: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct Spec {
    root: PathBuf,
    force: bool,
    batch_size: Option<usize>,
    debug_limit: Option<usize>,
}

impl Spec {
    /// Build the incremental default used after an OCR index run. The root was already resolved
    /// when its index-job request was accepted.
    pub(crate) fn pending_for(root: PathBuf, debug_limit: Option<usize>) -> Self {
        Self {
            root,
            force: false,
            batch_size: None,
            debug_limit,
        }
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
    let root = roots::resolve_root("text embeddings", &request.root)?;
    Ok(Spec {
        root,
        force: request.force,
        batch_size: request.batch_size,
        debug_limit: None,
    })
}

pub(crate) fn run(
    spec: Spec,
    ocr_database: &PathBuf,
    embedder: &TextEmbedder,
    observer: &dyn IndexObserver,
) -> anyhow::Result<bool> {
    // The write connection is opened up front but takes no write lock by doing so: in WAL mode
    // SQLite acquires it at the first statement of a write transaction and releases it at commit.
    // The only transactions here are the two short ones below and one per saved batch, so the
    // embedder's forward pass — the slow part — always runs with no lock held and searches keep
    // reading throughout.
    let mut ocr = DB::new(ocr_database)?;
    // Declaring the model is what creates the vector table on a fresh database, and what discards
    // vectors that a model change has invalidated.
    ocr.set_text_embedding_model(SPACE, embedder.model().id(), embedder.dimensions(), true)?;
    if spec.force {
        ocr.clear_text_embeddings(SPACE)?;
    }

    let batch_size = spec
        .batch_size
        .unwrap_or_else(|| embedder.max_batch_size())
        .min(embedder.max_batch_size());

    observer.on_event(IndexEvent::PhaseChanged(IndexPhase::TextEmbedding));
    let filters = SearchFilters::new(&spec.root);
    let backlog = ocr.text_embedding_coverage(SPACE, &filters)?.pending();
    let backlog = spec.debug_limit.map_or(backlog, |limit| backlog.min(limit));
    observer.on_event(IndexEvent::Discovered { count: backlog });
    observer.on_event(IndexEvent::DiscoveryComplete { total: backlog });

    let mut remaining = spec.debug_limit;
    loop {
        if observer.is_cancelled() {
            return Ok(true);
        }
        let next_batch_size = remaining.map_or(batch_size, |limit| batch_size.min(limit));
        if next_batch_size == 0 {
            return Ok(false);
        }
        let pending =
            ocr.pending_text_embeddings(SPACE, &filters, next_batch_size, MAX_CONTENT_BYTES)?;
        if pending.is_empty() {
            return Ok(false);
        }

        let texts: Vec<&str> = pending.iter().map(|row| row.content.as_str()).collect();
        let vectors = match embedder.embed_documents(&texts) {
            Ok(vectors) => vectors,
            Err(error) => {
                // One unembeddable batch must not strand the rest of the backlog, but the rows are
                // reported rather than silently skipped: they stay pending, so a retry re-tries
                // them and an operator can see why coverage never reaches 100%.
                for row in &pending {
                    observer.on_event(IndexEvent::Error {
                        path: None,
                        message: format!("text embedding asset {} failed: {error:#}", row.asset_id),
                    });
                }
                observer.on_event(IndexEvent::Progress(IndexProgressDelta {
                    processed: pending.len(),
                    phase_completed: pending.len(),
                    failed: pending.len(),
                    ..IndexProgressDelta::default()
                }));
                return Ok(false);
            }
        };

        let items: Vec<TextEmbedding> = pending
            .iter()
            .zip(vectors)
            .map(|(row, vector)| TextEmbedding {
                asset_id: row.asset_id,
                vector,
            })
            .collect();
        let attempted = items.len();
        let stored = ocr.save_text_embeddings(SPACE, embedder.model().id(), items)?;
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
            return Ok(false);
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
    fn the_root_is_required_and_unknown_fields_are_rejected() {
        assert!(request(serde_json::json!({})).is_err());
        assert!(request(serde_json::json!({"root": "C:/gallery", "nope": 1})).is_err());
        assert!(request(serde_json::json!({"root": "C:/gallery"})).is_ok());
    }

    #[test]
    fn the_batch_size_is_bounded() {
        for size in [0, MAX_BATCH_SIZE + 1] {
            let parsed =
                request(serde_json::json!({"root": "C:/gallery", "batchSize": size})).unwrap();
            let error = prepare(parsed).expect_err("an unbounded batch is rejected");
            assert!(error.message.contains("batchSize"), "{error:?}");
        }
    }
}
