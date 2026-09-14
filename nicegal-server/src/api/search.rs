//! Search: one vector-first route over OCR text and indexed images, in one request.
//!
//! # Why the modes combine into one request rather than one request each
//!
//! Clients search several ways at once, and the alternative — a parallel `GET` per mode — was
//! rejected for four reasons, in descending order of how much they matter:
//!
//! 1. **One OCR snapshot.** [`DB::read_snapshot`] runs every OCR mode against a single WAL read
//!    transaction. `type=image` has an independent image-index connection because its data lives
//!    in another database file; see below. Parallel requests each get their own, so an index job
//!    committing between them lets a client fuse two different versions of the library into one
//!    ranking.
//! 2. **Fusion needs both lists.** Reciprocal rank fusion is defined over the ranked lists, so it
//!    has to happen where both exist. Server-side it is one implementation over result sets that
//!    never leave the process; client-side every client reimplements it over as many as 250,000
//!    ids per mode, shipped across the socket only to be discarded.
//! 3. **The query is embedded once per engine.** A request that searches the same words
//!    semantically, literally, and visually pays for no duplicate forward pass within one engine.
//! 4. **One root resolution and one connection per store.** Each handler opens its own SQLite
//!    connections and re-registers their scalar functions; N modes over N requests pays that N
//!    times.
//!
//! `GET /v1/search` stays exactly as it was for the single-mode case, including its `total` and
//! `results` shape, and gains `type=vector` and `type=image`. `POST /v1/search` is the combined
//! form. Vector queries are the reason the combined form is a POST at all: an embedding request
//! carries a distance ceiling and per-mode weights that do not belong in a query string.
//!
//! `type=image` answers from the CLIP image index rather than the OCR store: a different engine
//! and a different database, joined to the request only by the shared filters and the combined
//! response. It cannot share the OCR snapshot — the two stores are separate files — so it runs in
//! the same blocking task on its own read-only connection, which is the closest one-request
//! equivalent: the image index changes only through indexing jobs, so the two reads a request
//! makes agree unless an index job commits mid-request.

use std::collections::HashMap;

use axum::Json;
use axum::extract::State;
use axum::routing::{MethodRouter, get};
use camino::Utf8PathBuf as PathBuf;
use nicegal_core::assets::Timeline;
use nicegal_core::cancellation::SearchCancellation;
use nicegal_core::db::{
    DB, SNIPPET_CLOSE, SNIPPET_OPEN, SearchFilters, SearchType, TextEmbeddingSpace,
    TextVectorSearchOptions, TimeRange, query_syntax_message,
};
use nicegal_core::highlight::{self, Highlight};
use nicegal_core::image_index::{ImageIndexDb, ImageVectorSearchOptions};
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::external_image::{self, ExternalImageRequest};
use super::extract::{ApiJson, ApiQuery};
use super::{AppState, roots, run_blocking};

const DEFAULT_LIMIT: usize = 100_000;
const MAX_LIMIT: usize = 250_000;
// Long enough for any query a person types or pastes, short enough that a runaway client cannot
// push a megabyte of text through the FTS5 parser or the embedder.
const MAX_QUERY_BYTES: usize = 4096;
/// More modes than this in one request is a client bug, not a search.
const MAX_QUERIES: usize = 6;
/// A composite image query is intentionally small: each component may trigger model inference or
/// a database lookup, and a human search needs only a handful of positive and negative examples.
const MAX_IMAGE_QUERY_COMPONENTS: usize = 16;
/// Keep weighted CLIP arithmetic numerically meaningful and bounded. Larger intent is expressed
/// by changing relative component weights, not by pushing an unbounded float through the API.
const MAX_IMAGE_QUERY_COMPONENT_WEIGHT: f64 = 100.0;
/// The constant from the reciprocal-rank-fusion paper. It damps the top of each list so one mode's
/// first hit cannot outweigh agreement between the others.
const DEFAULT_RRF_K: f64 = 60.0;

/// The async handler owns this guard; the blocking worker owns only the token.
/// Dropping the handler (for example on a disconnected HTTP request) interrupts its readers.
struct CancelSearchOnDrop(Option<SearchCancellation>);

impl Drop for CancelSearchOnDrop {
    fn drop(&mut self) {
        if let Some(cancellation) = &self.0 {
            cancellation.cancel();
            tracing::debug!("cancelled abandoned search");
        }
    }
}

async fn run_search<T: Send + 'static>(
    task: impl FnOnce(SearchCancellation) -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    let cancellation = SearchCancellation::default();
    let mut guard = CancelSearchOnDrop(Some(cancellation.clone()));
    let result = run_blocking(move || {
        cancellation.check()?;
        task(cancellation)
    })
    .await;
    guard.0 = None;
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
enum SearchTypeRequest {
    /// Nearest neighbours of the query text's OCR-text embedding. The primary mode.
    Vector,
    /// Nearest neighbours of the query text's *image* embedding, searched over the CLIP image
    /// index. A different engine from `vector`: the query is embedded by the image model's
    /// paired text encoder, and the neighbours are pictures, not OCR text.
    Image,
    Simple,
    Match,
    Glob,
    Regex,
}

impl SearchTypeRequest {
    /// The modes that rank by cosine distance and therefore accept `maxDistance`.
    fn uses_distance(self) -> bool {
        matches!(self, Self::Vector | Self::Image)
    }
}

// ---------------------------------------------------------------------------------------------
// GET /v1/search — one mode, unchanged wire shape
// ---------------------------------------------------------------------------------------------

// The existing field names are all single words, so `camelCase` only affects `maxDistance`; the
// documented v1 query string is unchanged.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SearchRequest {
    q: String,
    #[serde(rename = "type", default = "default_search_type")]
    kind: SearchTypeRequest,
    root: PathBuf,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Cosine distance ceiling, honoured by `type=vector` and `type=image`.
    max_distance: Option<f64>,
    #[serde(flatten)]
    time: TimeRequest,
}

/// The time filter every search mode accepts, defined once and flattened into each request shape.
///
/// `timeline` has no default on purpose: "when it was taken" and "when the file last changed" are
/// different questions, and picking one silently produces a result set the caller cannot interpret.
/// It is therefore required whenever a bound is given, and pointless without one.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimeRequest {
    /// Inclusive lower bound, Unix nanoseconds as a decimal string.
    after: Option<String>,
    /// Exclusive upper bound, Unix nanoseconds as a decimal string.
    before: Option<String>,
    timeline: Option<TimelineRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
enum TimelineRequest {
    /// `COALESCE(exif_taken_ns, source_modified_ns)` — when the photo was taken.
    Capture,
    /// `source_modified_ns` — when the file last changed.
    Modified,
}

impl TimeRequest {
    /// Whether the caller named any time field at all, which is what distinguishes a per-query
    /// override from a query that simply inherits the request-level filter.
    fn is_set(&self) -> bool {
        self.after.is_some() || self.before.is_some() || self.timeline.is_some()
    }

    /// `None` when no bound was given, which searches every instant.
    fn resolve(&self, what: &str) -> Result<Option<TimeRange>, ApiError> {
        let after_ns = parse_instant(what, "after", self.after.as_deref())?;
        let before_ns = parse_instant(what, "before", self.before.as_deref())?;
        if after_ns.is_none() && before_ns.is_none() {
            return Ok(None);
        }
        let Some(timeline) = self.timeline else {
            return Err(ApiError::bad_request(format!(
                "{what}: timeline is required with before or after, and must be capture or modified"
            )));
        };
        if matches!((after_ns, before_ns), (Some(after), Some(before)) if after >= before) {
            return Err(ApiError::bad_request(format!(
                "{what}: after must be less than before"
            )));
        }
        Ok(Some(TimeRange {
            timeline: match timeline {
                TimelineRequest::Capture => Timeline::Capture,
                TimelineRequest::Modified => Timeline::Modified,
            },
            after_ns,
            before_ns,
        }))
    }
}

/// Nanosecond instants cross the wire as decimal strings for the same reason the rest of this API
/// does it: a JSON number loses the low digits of a Unix nanosecond timestamp in JavaScript.
fn parse_instant(what: &str, field: &str, value: Option<&str>) -> Result<Option<i64>, ApiError> {
    value
        .map(|value| {
            value.parse::<i64>().map_err(|_| {
                ApiError::bad_request(format!(
                    "{what}: {field} must be a signed 64-bit decimal string of Unix nanoseconds"
                ))
            })
        })
        .transpose()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResponse {
    total: usize,
    results: Vec<SearchHit>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchHit {
    asset_id: i64,
    snippet: String,
    /// Position in this mode's own ranking, from 1. Present so a client can fuse two responses
    /// itself if it ever wants to, without re-deriving ranks from array order.
    rank: usize,
    /// Cosine distance from the query embedding: 0 identical, 1 orthogonal, 2 opposite. Absent for
    /// the text modes, which have no distance.
    #[serde(skip_serializing_if = "Option::is_none")]
    distance: Option<f64>,
    /// This mode's own relevance score: FTS5 `bm25()` for `simple`/`match` (lower is better) or
    /// matching word-term occurrence count for `glob` (higher is better). Not comparable across
    /// modes — a client that needs one ordering across modes should use `rank` or the fused
    /// response, not this. Absent for the distance-ranked `vector` and `image` modes.
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    /// Where in `snippet` this query's words appear to match, for a client that wants to underline
    /// them. Omitted for image hits, which deliberately have no OCR snippet to underline.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    highlights: Vec<HighlightResponse>,
}

/// One estimated match inside a hit's `snippet`.
///
/// The bounds are character offsets rather than byte offsets, because the client slicing them is
/// JavaScript; see [`nicegal_core::highlight`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct HighlightResponse {
    start: usize,
    end: usize,
    /// `exact`, `stem`, `prefix`, or `fuzzy`, in descending confidence.
    kind: &'static str,
}

impl From<Highlight> for HighlightResponse {
    fn from(highlight: Highlight) -> Self {
        Self {
            start: highlight.start,
            end: highlight.end,
            kind: highlight.kind.as_str(),
        }
    }
}

/// Whether SQLite has already marked this mode's snippet around the terms it matched.
fn marks_its_own_snippet(kind: SearchType) -> bool {
    matches!(kind, SearchType::Simple | SearchType::Match)
}

/// Fills in each OCR hit's `highlights` from its own snippet.
///
/// Two modes reach the same field by different routes, and the difference is worth keeping in
/// mind: FTS5 *knows* which terms it matched and says so with markers, while vector search returns
/// a ranking and no reason for it at all, so its spans are estimated after the fact by matching the
/// query's words against the OCR text that came back. Both only ever decorate a hit — neither
/// filters or reorders one — and `kind` is what tells a client which of the two it is holding.
///
/// This runs inside the read snapshot, where the snippet is produced.
fn add_highlights(query: &str, kind: Option<SearchType>, hits: &mut [SearchHit]) {
    for hit in hits {
        let highlights = if kind.is_some_and(marks_its_own_snippet) {
            // Trading the marked-up snippet for the plain one is the whole point: a client that
            // renders spans should never also have to strip delimiters out of the text.
            let (snippet, highlights) =
                highlight::from_marked(&hit.snippet, SNIPPET_OPEN, SNIPPET_CLOSE);
            hit.snippet = snippet;
            highlights
        } else {
            highlight::highlights(query, &hit.snippet)
        };
        hit.highlights = highlights
            .into_iter()
            .map(HighlightResponse::from)
            .collect();
    }
}

// ---------------------------------------------------------------------------------------------
// POST /v1/search — several modes, one response
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MultiSearchRequest {
    root: PathBuf,
    queries: Vec<QueryRequest>,
    /// A directory under `root` to leave out, as an absolute path.
    exclude: Option<PathBuf>,
    #[serde(default = "default_limit")]
    limit: usize,
    /// Applies to every query that does not set its own.
    #[serde(flatten)]
    time: TimeRequest,
    /// Omit to get the per-mode lists only.
    fuse: Option<FuseRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct QueryRequest {
    /// Names this mode's block in the response and its weight in `fuse`. Defaults to the mode
    /// name, which is unambiguous until a request runs the same mode twice.
    key: Option<String>,
    #[serde(rename = "type", default = "default_search_type")]
    kind: SearchTypeRequest,
    /// The legacy positive text component. Required by every OCR mode; optional for image mode
    /// because `imageQuery.components` can contain assets and/or signed text on its own.
    q: Option<String>,
    /// A composable CLIP query. This is intentionally POST-only: repeated, structured components
    /// do not fit safely in the single-mode route's query string.
    image_query: Option<ImageQueryRequest>,
    /// Overrides the request-level `limit` for this mode alone.
    limit: Option<usize>,
    max_distance: Option<f64>,
    /// Multiplies this mode's contribution to the fused score. 1 is neutral.
    #[serde(default = "default_weight")]
    weight: f64,
    /// Overrides the request-level time filter for this mode alone. A mode that sets any of the
    /// three replaces the whole request-level filter rather than merging with it, so a partial
    /// override cannot silently inherit a timeline it did not ask for.
    #[serde(flatten)]
    time: TimeRequest,
}

/// Structured components whose normalized CLIP vectors are weighted and summed into one image
/// query. Paths and binary vectors can become additional component sources without changing this
/// composition contract.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImageQueryRequest {
    components: Vec<ImageQueryComponentRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ImageQueryComponentRequest {
    external_image: Option<ExternalImageRequest>,
    /// An existing asset with a current CLIP vector in the active image index.
    asset_id: Option<i64>,
    /// Text embedded by the active image model's paired CLIP text encoder.
    text: Option<String>,
    /// Positive components pull results toward their meaning; negative components push results
    /// toward the opposite direction. A value's magnitude is its relative influence.
    #[serde(default = "default_weight")]
    weight: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FuseRequest {
    #[serde(default)]
    method: FuseMethod,
    /// Reciprocal-rank-fusion damping. Larger flattens the contribution of a mode's top hits.
    #[serde(default = "default_rrf_k")]
    k: f64,
    /// How many fused hits to return. Defaults to the request-level `limit`.
    limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
enum FuseMethod {
    #[default]
    Rrf,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MultiSearchResponse {
    /// The embedding model the stored vectors belong to, so a client can tell "no vector hits"
    /// from "nothing has been embedded yet". Null before the first backfill.
    model: Option<EmbeddingModelResponse>,
    /// The image engine's model, whose stored CLIP vectors `type=image` queries are compared
    /// against. Always present: the image index belongs to the process from startup, and its
    /// model changes only through a restart.
    image_model: EmbeddingModelResponse,
    queries: Vec<QueryResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fused: Option<FusedResponse>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingModelResponse {
    model: String,
    dimensions: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct QueryResponse {
    key: String,
    #[serde(rename = "type")]
    kind: SearchTypeRequest,
    /// Matches before `limit` was applied, exactly as the single-mode route reports it.
    total: usize,
    results: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FusedResponse {
    /// Distinct assets any mode returned, before the fused limit.
    total: usize,
    results: Vec<FusedHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FusedHit {
    asset_id: i64,
    score: f64,
    rank: usize,
    /// The keys of the modes that returned this asset, in request order. A hit several modes agree
    /// on is the signal the fused ranking exists to surface.
    sources: Vec<String>,
    snippet: String,
    /// The highlights of the mode that contributed `snippet`, so the fused list underlines the
    /// same words the per-mode list does.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    highlights: Vec<HighlightResponse>,
}

pub(super) fn route() -> MethodRouter<AppState> {
    get(search).post(multi_search)
}

async fn search(
    State(state): State<AppState>,
    ApiQuery(request): ApiQuery<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let query = validate_query(&request.q)?.to_owned();
    let limit = validate_limit("limit", request.limit)?;
    let max_distance = validate_max_distance(request.max_distance)?;
    if max_distance.is_some() && !request.kind.uses_distance() {
        return Err(ApiError::bad_request(
            "maxDistance applies only to the vector modes, type=vector and type=image",
        ));
    }
    let time = request.time.resolve("search")?;
    let kind = search_type(request.kind)?;
    let plan = QueryPlan {
        key: String::new(),
        input: match kind {
            PlanKind::Image => QueryInput::Image(ImageQuery::from_text(query)),
            _ => QueryInput::Text(query),
        },
        kind,
        limit,
        max_distance,
        weight: 1.0,
        time,
    };
    let requested_root = request.root;
    let databases = state.databases;
    let embedder = state.embedder;
    let image_query_embedder = state.image_query_embedder;

    let (total, hits) = run_search(move |cancellation| {
        // A root that exists but has never been indexed is not an error: an empty result set is
        // the honest answer, and the gallery tells the two apart from its own catalog.
        let root = roots::resolve_root("search", &requested_root)?;
        let (total, mut hits) = if matches!(plan.kind, PlanKind::Image) {
            // Embedding after the root check on purpose: a bad root is the caller's mistake and
            // should not pay a forward pass to be told so.
            let images = databases.open_images_read_only(image_query_embedder.dimensions())?;
            images.set_search_cancellation(&cancellation)?;
            let mut snapshot = images.begin_read_snapshot()?;
            let mut resolver = ImageQueryResolver::new(&image_query_embedder, &snapshot);
            let vector = resolver.resolve(plan.image_query(), &cancellation)?;
            cancellation.check()?;
            let result = plan.run_image(&snapshot, Some(vector.as_slice()), &root, None)?;
            snapshot.commit()?;
            result
        } else {
            let mut db = databases.open_ocr_read_only()?;
            db.set_search_cancellation(&cancellation)?;
            let vector = match plan.kind {
                PlanKind::OcrVector => Some(embedder.ready_or_cached()?.embed_query(plan.text())?),
                _ => None,
            };
            cancellation.check()?;
            db.read_snapshot(|db| plan.run(db, &root, None, vector.as_deref()))
                .map_err(classify)?
        };
        stamp_ranks(&mut hits);
        Ok((total, hits))
    })
    .await?;

    Ok(Json(SearchResponse {
        total,
        results: hits,
    }))
}

async fn multi_search(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<MultiSearchRequest>,
) -> Result<Json<MultiSearchResponse>, ApiError> {
    let default_limit = validate_limit("limit", request.limit)?;
    if request.queries.is_empty() {
        return Err(ApiError::bad_request("at least one query is required"));
    }
    if request.queries.len() > MAX_QUERIES {
        return Err(ApiError::bad_request(format!(
            "a search may combine at most {MAX_QUERIES} queries"
        )));
    }

    let request_time = request.time.resolve("search")?;
    let mut plans = Vec::with_capacity(request.queries.len());
    let mut seen = Vec::with_capacity(request.queries.len());
    for query in request.queries {
        let key = query.key.unwrap_or_else(|| {
            serde_json::to_value(query.kind).map_or_else(
                |_| "query".to_owned(),
                |value| value.as_str().unwrap_or("query").to_owned(),
            )
        });
        if key.trim().is_empty() {
            return Err(ApiError::bad_request("a query key must not be empty"));
        }
        if seen.contains(&key) {
            return Err(ApiError::bad_request(format!(
                "duplicate query key {key}; give each query its own key"
            )));
        }
        seen.push(key.clone());

        let max_distance = validate_max_distance(query.max_distance)?;
        if max_distance.is_some() && !query.kind.uses_distance() {
            return Err(ApiError::bad_request(format!(
                "query {key}: maxDistance applies only to the vector modes, vector and image"
            )));
        }
        if !query.weight.is_finite() || query.weight <= 0.0 {
            return Err(ApiError::bad_request(format!(
                "query {key}: weight must be a positive number"
            )));
        }
        // A query naming any time field owns its whole filter; otherwise it inherits the
        // request-level one intact.
        let time = if query.time.is_set() {
            query.time.resolve(&format!("query {key}"))?
        } else {
            request_time
        };
        let kind = search_type(query.kind)?;
        let input = query_input(kind, query.q, query.image_query, &key)?;
        plans.push(QueryPlan {
            kind,
            input,
            limit: match query.limit {
                Some(limit) => validate_limit("query limit", limit)?,
                None => default_limit,
            },
            max_distance,
            weight: query.weight,
            time,
            key,
        });
    }

    let fusion = request
        .fuse
        .map(|fuse| -> Result<Fusion, ApiError> {
            if !fuse.k.is_finite() || fuse.k <= 0.0 {
                return Err(ApiError::bad_request("fuse.k must be a positive number"));
            }
            Ok(Fusion {
                method: fuse.method,
                k: fuse.k,
                limit: match fuse.limit {
                    Some(limit) => validate_limit("fuse limit", limit)?,
                    None => default_limit,
                },
            })
        })
        .transpose()?;

    let requested_root = request.root;
    let requested_exclude = request.exclude;
    let databases = state.databases;
    let embedder = state.embedder;
    let image_query_embedder = state.image_query_embedder;
    let image_dimensions = image_query_embedder.dimensions();
    let image_embedder = state.image_embedder;
    let external_bytes: usize = plans
        .iter()
        .filter_map(|plan| match &plan.input {
            QueryInput::Image(query) => Some(
                query
                    .components
                    .iter()
                    .map(|component| match &component.source {
                        ImageQuerySource::ExternalImage(bytes) => bytes.len(),
                        _ => 0,
                    })
                    .sum::<usize>(),
            ),
            _ => None,
        })
        .sum();
    if external_bytes > external_image::MAX_TOTAL_BYTES {
        return Err(ApiError::bad_request(
            "external images exceed 32 MiB in one request",
        ));
    }

    run_search(move |cancellation| {
        let root = roots::resolve_root("search", &requested_root)?;
        let exclude = requested_exclude
            .map(|exclude| roots::resolve_root("exclude", &exclude))
            .transpose()?;
        let exclude = exclude.as_deref().map(camino::Utf8Path::as_str);
        let mut db = databases.open_ocr_read_only()?;
        db.set_search_cancellation(&cancellation)?;
        // The image index is a separate store, opened only when a query plans to use it.
        let has_image_queries = plans
            .iter()
            .any(|plan| matches!(plan.kind, PlanKind::Image));
        let images = if has_image_queries {
            let images = databases.open_images_read_only(image_dimensions)?;
            images.set_search_cancellation(&cancellation)?;
            Some(images)
        } else {
            None
        };
        // Resolve every asset component and run every image search in the same image-index
        // snapshot. Without this, an asset could be current when its vector is read and become
        // stale just before the result query correctly filters stale rows.
        let mut image_snapshot = images
            .as_ref()
            .map(ImageIndexDb::begin_read_snapshot)
            .transpose()?;

        // One forward pass covers every OCR-vector query with the same text. Composite image
        // queries have their own resolver: it similarly caches repeated text and asset
        // components, then normalizes each weighted sum before it reaches the image index.
        let mut ocr_embeddings: HashMap<&str, Vec<f32>> = HashMap::new();
        for plan in &plans {
            cancellation.check()?;
            match plan.kind {
                PlanKind::OcrVector if !ocr_embeddings.contains_key(plan.text()) => {
                    ocr_embeddings.insert(
                        plan.text(),
                        embedder.ready_or_cached()?.embed_query(plan.text())?,
                    );
                }
                _ => {}
            }
        }
        let mut image_vectors: HashMap<&str, Vec<f32>> = HashMap::new();
        if let Some(snapshot) = image_snapshot.as_ref() {
            let mut resolver = ImageQueryResolver::new(&image_query_embedder, snapshot);
            resolver.image_embedder = Some(&image_embedder);
            for plan in &plans {
                if matches!(plan.kind, PlanKind::Image) {
                    image_vectors.insert(
                        plan.key.as_str(),
                        resolver.resolve(plan.image_query(), &cancellation)?,
                    );
                }
            }
        }

        cancellation.check()?;
        let model = db
            .text_embedding_model(TextEmbeddingSpace::OcrText)?
            .map(|model| EmbeddingModelResponse {
                model: model.model,
                dimensions: model.dimensions,
            });
        let image_model = EmbeddingModelResponse {
            model: image_query_embedder.model().id().to_owned(),
            dimensions: image_query_embedder.dimensions(),
        };

        let mut queries = db
            .read_snapshot(|db| {
                let mut answered = Vec::with_capacity(plans.len());
                for plan in &plans {
                    cancellation.check()?;
                    let (total, results) = match plan.kind {
                        PlanKind::Image => {
                            let images = image_snapshot
                                .as_ref()
                                .expect("an image query opens the image snapshot first");
                            let vector = image_vectors.get(plan.key.as_str()).map(Vec::as_slice);
                            plan.run_image(images, vector, &root, exclude)?
                        }
                        _ => {
                            let vector = ocr_embeddings.get(plan.text()).map(Vec::as_slice);
                            plan.run(db, &root, exclude, vector)?
                        }
                    };
                    answered.push(QueryResponse {
                        key: plan.key.clone(),
                        kind: plan.request_kind(),
                        total,
                        results,
                    });
                }
                Ok(answered)
            })
            .map_err(classify)?;
        if let Some(snapshot) = image_snapshot.as_mut() {
            snapshot.commit()?;
        }
        for query in &mut queries {
            stamp_ranks(&mut query.results);
        }

        cancellation.check()?;
        let fused = fusion.map(|fusion| fusion.apply(&plans, &queries));
        Ok(Json(MultiSearchResponse {
            model,
            image_model,
            queries,
            fused,
        }))
    })
    .await
}

/// Which engine a validated query runs against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanKind {
    /// Nearest OCR-text vectors: the query embedded by the OCR engine's own text model and
    /// searched over the OCR store's vector space.
    OcrVector,
    /// Nearest image vectors: the query embedded by the image engine's paired text encoder and
    /// searched over the CLIP image index. A separate engine, a separate database.
    Image,
    /// A literal text mode over the OCR store.
    Text(SearchType),
}

/// A source vector and its signed influence in one composite CLIP image query.
#[derive(Debug)]
struct ImageQueryComponent {
    source: ImageQuerySource,
    weight: f64,
}

#[derive(Debug)]
enum ImageQuerySource {
    ExternalImage(Vec<u8>),
    AssetId(i64),
    Text(String),
}

#[derive(Debug)]
struct ImageQuery {
    components: Vec<ImageQueryComponent>,
}

impl ImageQuery {
    fn from_text(text: String) -> Self {
        Self {
            components: vec![ImageQueryComponent {
                source: ImageQuerySource::Text(text),
                weight: 1.0,
            }],
        }
    }
}

/// A plan's input is deliberately typed by its engine: OCR modes only ever see text, whereas the
/// image engine composes CLIP text and image-vector components before one vector search.
#[derive(Debug)]
enum QueryInput {
    Text(String),
    Image(ImageQuery),
}

/// One mode's validated query, ready to run against its stores.
struct QueryPlan {
    key: String,
    kind: PlanKind,
    input: QueryInput,
    limit: usize,
    max_distance: Option<f64>,
    weight: f64,
    time: Option<TimeRange>,
}

impl QueryPlan {
    fn text(&self) -> &str {
        match &self.input {
            QueryInput::Text(text) => text,
            QueryInput::Image(_) => {
                unreachable!("only OCR plans have a single text query")
            }
        }
    }

    fn image_query(&self) -> &ImageQuery {
        match &self.input {
            QueryInput::Image(query) => query,
            QueryInput::Text(_) => {
                unreachable!("only image plans have composite image queries")
            }
        }
    }

    fn request_kind(&self) -> SearchTypeRequest {
        match self.kind {
            PlanKind::OcrVector => SearchTypeRequest::Vector,
            PlanKind::Image => SearchTypeRequest::Image,
            PlanKind::Text(SearchType::Simple) => SearchTypeRequest::Simple,
            PlanKind::Text(SearchType::Match) => SearchTypeRequest::Match,
            PlanKind::Text(SearchType::Glob) => SearchTypeRequest::Glob,
            #[cfg(feature = "regex")]
            PlanKind::Text(SearchType::Regex) => SearchTypeRequest::Regex,
        }
    }

    /// Run inside a caller-owned snapshot, which is why this uses `search_count` and `search`
    /// rather than `search_with_count`: the latter opens a transaction of its own. Covers the
    /// OCR-store modes; image plans route to [`Self::run_image`] instead.
    fn run(
        &self,
        db: &mut DB,
        root: &camino::Utf8Path,
        exclude: Option<&str>,
        vector: Option<&[f32]>,
    ) -> anyhow::Result<(usize, Vec<SearchHit>)> {
        // Every mode narrows the same way. The root, the exclusion, and the time range are built
        // once here and handed to whichever mode runs, rather than each mode re-deriving them.
        let filters = SearchFilters::new(root)
            .with_exclude(exclude)
            .with_time(self.time);
        match self.kind {
            PlanKind::Text(kind) => {
                let text = self.text();
                let total = db.search_count(vec![text], &filters, kind)?;
                let results = db.search(vec![text], &filters, self.limit, kind)?;
                let mut hits: Vec<SearchHit> = results
                    .into_iter()
                    .map(|hit| SearchHit {
                        asset_id: hit.asset_id,
                        snippet: hit.contents,
                        rank: 0,
                        distance: None,
                        score: Some(hit.score),
                        highlights: Vec::new(),
                    })
                    .collect();
                add_highlights(text, Some(kind), &mut hits);
                Ok((total, hits))
            }
            PlanKind::OcrVector => {
                let text = self.text();
                let vector = vector.expect("a vector query is embedded before it is run");
                let options = TextVectorSearchOptions {
                    max_distance: self.max_distance,
                    ..TextVectorSearchOptions::default()
                };
                let (total, results) = db.search_text_vectors(
                    vector,
                    TextEmbeddingSpace::OcrText,
                    &filters,
                    self.limit,
                    options,
                )?;
                let mut hits: Vec<SearchHit> = results
                    .into_iter()
                    .map(|hit| SearchHit {
                        asset_id: hit.asset_id,
                        snippet: hit.contents,
                        rank: 0,
                        distance: Some(hit.distance),
                        score: None,
                        highlights: Vec::new(),
                    })
                    .collect();
                add_highlights(text, None, &mut hits);
                Ok((total, hits))
            }
            PlanKind::Image => unreachable!("image plans run against the image index"),
        }
    }

    /// Run against the CLIP image index on its own connection, outside the OCR snapshot: the two
    /// stores are separate databases, so this is the closest one-request equivalent of sharing
    /// one. Same filters, same `total`-before-`limit` contract as every other mode.
    fn run_image(
        &self,
        images: &ImageIndexDb,
        vector: Option<&[f32]>,
        root: &camino::Utf8Path,
        exclude: Option<&str>,
    ) -> anyhow::Result<(usize, Vec<SearchHit>)> {
        let vector = vector.expect("an image query is embedded before it is run");
        let filters = SearchFilters::new(root)
            .with_exclude(exclude)
            .with_time(self.time);
        let options = ImageVectorSearchOptions {
            max_distance: self.max_distance,
        };
        let (total, results) = images.search_vectors(vector, &filters, self.limit, &options)?;
        // The image mode has no text of its own: the hit is the picture, so `snippet` is empty
        // and there is nothing to estimate matches in.
        let hits = results
            .into_iter()
            .map(|hit| SearchHit {
                asset_id: hit.asset_id,
                snippet: String::new(),
                rank: 0,
                distance: Some(hit.distance),
                score: None,
                highlights: Vec::new(),
            })
            .collect();
        Ok((total, hits))
    }
}

/// Resolves a request's image-query components without repeating inference or index reads for
/// identical text and asset IDs. A resolver belongs to one request and one image-model connection;
/// its cache is deliberately short-lived so index updates are visible to the next request.
struct ImageQueryResolver<'a> {
    image_embedder: Option<&'a super::models::ImageModel>,
    external_vector: Vec<f32>,
    embedder: &'a super::models::ImageQueryModel,
    images: &'a ImageIndexDb,
    text_vectors: HashMap<String, Vec<f32>>,
    asset_vectors: HashMap<i64, Vec<f32>>,
}

impl<'a> ImageQueryResolver<'a> {
    fn new(embedder: &'a super::models::ImageQueryModel, images: &'a ImageIndexDb) -> Self {
        Self {
            embedder,
            image_embedder: None,
            external_vector: Vec::new(),
            images,
            text_vectors: HashMap::new(),
            asset_vectors: HashMap::new(),
        }
    }

    /// Build one unit-length direction from normalized, signed CLIP components:
    ///
    /// `normalize(weight₁ × normalize(component₁) + …)`.
    ///
    /// A negative weight is directional arithmetic, not a boolean exclusion. For example,
    /// `+beach - people` finds the CLIP direction that is beach-like and people-unlike; it does
    /// not prove a result contains no person.
    fn resolve(
        &mut self,
        query: &ImageQuery,
        cancellation: &SearchCancellation,
    ) -> Result<Vec<f32>, ApiError> {
        let mut combined = vec![0.0_f64; self.embedder.dimensions()];
        for (index, component) in query.components.iter().enumerate() {
            cancellation.check()?;
            let vector = self
                .component_vector(&component.source)
                .map_err(|mut error| {
                    error.message = format!("imageQuery component {index}: {}", error.message);
                    error.component_index = Some(index);
                    error
                })?;
            add_normalized_component(&mut combined, vector, component.weight)?;
        }
        normalize_combined_image_vector(combined)
    }

    fn component_vector(&mut self, source: &ImageQuerySource) -> Result<&[f32], ApiError> {
        match source {
            ImageQuerySource::ExternalImage(bytes) => {
                let model = self
                    .image_embedder
                    .expect("POST image resolver has an image model")
                    .ready_or_cached()?;
                if model.model() != self.embedder.model() {
                    return Err(ApiError::bad_request(
                        "external image model is incompatible with the active search model",
                    ));
                }
                self.external_vector = external_image::embed(bytes, &model)?;
                Ok(&self.external_vector)
            }
            ImageQuerySource::Text(text) => {
                if !self.embedder.model().supports_text_queries() {
                    return Err(ApiError::bad_request(
                        "This model supports image examples only. Remove text descriptions from visual search.",
                    ));
                }
                if !self.text_vectors.contains_key(text) {
                    self.text_vectors.insert(
                        text.clone(),
                        self.embedder.ready_or_cached()?.embed_query(text)?,
                    );
                }
                Ok(self
                    .text_vectors
                    .get(text)
                    .expect("the just-inserted image text vector exists"))
            }
            ImageQuerySource::AssetId(asset_id) => {
                if !self.asset_vectors.contains_key(asset_id) {
                    let vector = self.images.current_vector(*asset_id)?.ok_or_else(|| {
                        ApiError::bad_request(format!(
                            "image query asset {asset_id} has no current image embedding"
                        ))
                    })?;
                    self.asset_vectors.insert(*asset_id, vector);
                }
                Ok(self
                    .asset_vectors
                    .get(asset_id)
                    .expect("the just-inserted image asset vector exists"))
            }
        }
    }
}

/// Add `weight × normalize(vector)` to `combined`, rejecting malformed model/index output before
/// it can turn a cosine search into NaNs.
fn add_normalized_component(
    combined: &mut [f64],
    vector: &[f32],
    weight: f64,
) -> Result<(), ApiError> {
    if vector.len() != combined.len() {
        return Err(ApiError::internal(anyhow::anyhow!(
            "image query component has {} dimensions but the active image model requires {}",
            vector.len(),
            combined.len()
        )));
    }
    let squared_norm = vector.iter().try_fold(0.0_f64, |sum, value| {
        let value = f64::from(*value);
        if value.is_finite() {
            Ok(sum + value * value)
        } else {
            Err(ApiError::internal(anyhow::anyhow!(
                "image query component contains a non-finite vector value"
            )))
        }
    })?;
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(ApiError::bad_request(
            "image query component has a zero-length vector",
        ));
    }
    for (sum, value) in combined.iter_mut().zip(vector) {
        *sum += weight * f64::from(*value) / norm;
    }
    Ok(())
}

/// Make the signed component sum a valid cosine-query direction. Exact cancellation is a caller
/// error: a zero vector cannot express similarity in any direction.
fn normalize_combined_image_vector(combined: Vec<f64>) -> Result<Vec<f32>, ApiError> {
    let squared_norm = combined.iter().map(|value| value * value).sum::<f64>();
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(ApiError::bad_request(
            "image query components cancel out; use non-cancelling signed weights",
        ));
    }
    Ok(combined
        .into_iter()
        .map(|value| (value / norm) as f32)
        .collect())
}

/// Stamps every hit's `rank` from its position in `hits`, from 1. Called once per mode, after
/// retrieval and before the hits are returned to the caller — so callers (including
/// [`Fusion::apply`]) always see a `rank` that matches the order actually returned. Retrieval
/// already produced that order (cosine distance ascending for vector search, `bm25()`/match count
/// for the text modes); this only stamps it, it does not reorder.
///
/// A cross-encoder reranking pass used to run here. It was removed: vector-mode hits were already
/// cosine-sorted, so reranking them only spent a second-plus of ONNX inference to reorder a list
/// that was already in the right order — and the text modes' gain from it was marginal for this
/// corpus's short, keyword-heavy OCR snippets. See git history (`src/rerank/`) if it's ever worth
/// resurrecting for a different corpus shape.
fn stamp_ranks(hits: &mut [SearchHit]) {
    for (index, hit) in hits.iter_mut().enumerate() {
        hit.rank = index + 1;
    }
}

struct Fusion {
    method: FuseMethod,
    k: f64,
    limit: usize,
}

/// One asset's running fused score, while [`Fusion::apply`] is still accumulating modes into it.
struct FusedEntry {
    score: f64,
    sources: Vec<String>,
    snippet: String,
    highlights: Vec<HighlightResponse>,
}

impl Fusion {
    /// Reciprocal rank fusion: an asset scores `weight / (k + rank)` in every mode that returned
    /// it, summed across modes.
    ///
    /// Ranks are used rather than the underlying scores on purpose. FTS5 rank and cosine distance
    /// are not on a common scale and cannot be made comparable by normalising them, so combining
    /// the positions is the only combination that means anything.
    fn apply(&self, plans: &[QueryPlan], queries: &[QueryResponse]) -> FusedResponse {
        let FuseMethod::Rrf = self.method;
        let weights: HashMap<&str, f64> = plans
            .iter()
            .map(|plan| (plan.key.as_str(), plan.weight))
            .collect();

        // The snippet and its highlights are taken from the first mode that returned the asset and
        // then travel together: spans are offsets into one particular snippet, so pairing them
        // with a different mode's excerpt would point at the wrong words.
        let mut scores: HashMap<i64, FusedEntry> = HashMap::new();
        for query in queries {
            let weight = weights.get(query.key.as_str()).copied().unwrap_or(1.0);
            for hit in &query.results {
                let entry = scores.entry(hit.asset_id).or_insert_with(|| FusedEntry {
                    score: 0.0,
                    sources: Vec::new(),
                    snippet: hit.snippet.clone(),
                    highlights: hit.highlights.clone(),
                });
                entry.score += weight / (self.k + hit.rank as f64);
                entry.sources.push(query.key.clone());
            }
        }

        let total = scores.len();
        let mut ranked: Vec<(i64, FusedEntry)> = scores.into_iter().collect();
        // Descending score, then ascending id so equal scores are ordered deterministically
        // rather than by hash iteration order.
        ranked.sort_by(|left, right| {
            right
                .1
                .score
                .partial_cmp(&left.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(self.limit);

        FusedResponse {
            total,
            results: ranked
                .into_iter()
                .enumerate()
                .map(|(index, (asset_id, entry))| FusedHit {
                    asset_id,
                    score: entry.score,
                    rank: index + 1,
                    sources: entry.sources,
                    snippet: entry.snippet,
                    highlights: entry.highlights,
                })
                .collect(),
        }
    }
}

/// SQLite's own text is the only description of what is wrong with a query, and the UI shows it
/// verbatim. Nothing else reaching here is the caller's fault.
fn classify(error: anyhow::Error) -> ApiError {
    match query_syntax_message(&error) {
        Some(message) => ApiError::query_syntax(message),
        None => ApiError::internal(error),
    }
}

fn validate_query(query: &str) -> Result<&str, ApiError> {
    if query.trim().is_empty() {
        return Err(ApiError::bad_request("search query must not be empty"));
    }
    if query.len() > MAX_QUERY_BYTES {
        return Err(ApiError::bad_request(format!(
            "search query must be at most {MAX_QUERY_BYTES} bytes"
        )));
    }
    Ok(query)
}

fn validate_limit(name: &str, limit: usize) -> Result<usize, ApiError> {
    if limit == 0 || limit > MAX_LIMIT {
        return Err(ApiError::bad_request(format!(
            "search {name} must be between 1 and {MAX_LIMIT}"
        )));
    }
    Ok(limit)
}

/// Cosine distance runs from 0 to 2; anything outside that is a client that has confused distance
/// with similarity, which is worth saying rather than silently matching everything.
fn validate_max_distance(max_distance: Option<f64>) -> Result<Option<f64>, ApiError> {
    match max_distance {
        Some(value) if !value.is_finite() || !(0.0..=2.0).contains(&value) => Err(
            ApiError::bad_request("maxDistance must be a cosine distance between 0 and 2"),
        ),
        other => Ok(other),
    }
}

/// Validate a query according to its engine and turn its wire form into the representation the
/// execution plan owns. Legacy `q` stays a positive image-text component, so existing image
/// callers retain their exact meaning while POST can add signed components around it.
fn query_input(
    kind: PlanKind,
    q: Option<String>,
    image_query: Option<ImageQueryRequest>,
    key: &str,
) -> Result<QueryInput, ApiError> {
    if !matches!(kind, PlanKind::Image) {
        if image_query.is_some() {
            return Err(ApiError::bad_request(format!(
                "query {key}: imageQuery applies only to type=image"
            )));
        }
        return Ok(QueryInput::Text(
            validate_query(q.as_deref().unwrap_or_default())?.to_owned(),
        ));
    }

    let mut components = match image_query {
        Some(image_query) => {
            if image_query.components.is_empty() {
                return Err(ApiError::bad_request(format!(
                    "query {key}: imageQuery.components must not be empty"
                )));
            }
            if image_query.components.len() > MAX_IMAGE_QUERY_COMPONENTS {
                return Err(ApiError::bad_request(format!(
                    "query {key}: imageQuery accepts at most {MAX_IMAGE_QUERY_COMPONENTS} components"
                )));
            }
            image_query
                .components
                .into_iter()
                .enumerate()
                .map(|(index, component)| {
                    image_query_component(component, key, index).map_err(|mut error| {
                        error.component_index = Some(index);
                        error
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        None => Vec::new(),
    };

    // An absent or blank legacy `q` adds no text component, which makes asset-only image search
    // possible without a magic placeholder string. Nonblank `q` remains strict and bounded.
    if let Some(text) = q.filter(|text| !text.trim().is_empty()) {
        if components.len() == MAX_IMAGE_QUERY_COMPONENTS {
            return Err(ApiError::bad_request(format!(
                "query {key}: image queries accept at most {MAX_IMAGE_QUERY_COMPONENTS} components"
            )));
        }
        components.push(ImageQueryComponent {
            source: ImageQuerySource::Text(validate_query(&text)?.to_owned()),
            weight: 1.0,
        });
    }
    if components.is_empty() {
        return Err(ApiError::bad_request(format!(
            "query {key}: type=image needs q or imageQuery.components"
        )));
    }
    Ok(QueryInput::Image(ImageQuery { components }))
}

fn image_query_component(
    component: ImageQueryComponentRequest,
    key: &str,
    index: usize,
) -> Result<ImageQueryComponent, ApiError> {
    if !component.weight.is_finite()
        || component.weight == 0.0
        || component.weight.abs() > MAX_IMAGE_QUERY_COMPONENT_WEIGHT
    {
        return Err(ApiError::bad_request(format!(
            "query {key}: imageQuery component {index} weight must be finite, nonzero, and at most {MAX_IMAGE_QUERY_COMPONENT_WEIGHT} in magnitude"
        )));
    }
    let source = match (component.asset_id, component.text, component.external_image) {
        (None, None, Some(image)) => {
            ImageQuerySource::ExternalImage(image.decode().map_err(|mut error| {
                error.message = format!(
                    "query {key}: imageQuery component {index}: {}",
                    error.message
                );
                error
            })?)
        }
        (Some(asset_id), None, None) => {
            if asset_id <= 0 {
                return Err(ApiError::bad_request(format!(
                    "query {key}: imageQuery component {index} assetId must be greater than zero"
                )));
            }
            ImageQuerySource::AssetId(asset_id)
        }
        (None, Some(text), None) => ImageQuerySource::Text(validate_query(&text)?.to_owned()),
        _ => {
            return Err(ApiError::bad_request(format!(
                "query {key}: imageQuery component {index} needs exactly one of assetId, text, or externalImage"
            )));
        }
    };
    Ok(ImageQueryComponent {
        source,
        weight: component.weight,
    })
}

/// Which engine and which literal mode a requested type runs.
fn search_type(kind: SearchTypeRequest) -> Result<PlanKind, ApiError> {
    Ok(match kind {
        SearchTypeRequest::Vector => PlanKind::OcrVector,
        SearchTypeRequest::Image => PlanKind::Image,
        SearchTypeRequest::Simple => PlanKind::Text(SearchType::Simple),
        SearchTypeRequest::Match => PlanKind::Text(SearchType::Match),
        SearchTypeRequest::Glob => PlanKind::Text(SearchType::Glob),
        SearchTypeRequest::Regex => {
            #[cfg(feature = "regex")]
            {
                PlanKind::Text(SearchType::Regex)
            }
            #[cfg(not(feature = "regex"))]
            {
                return Err(ApiError::bad_request(
                    "regex search is unavailable in this server build",
                ));
            }
        }
    })
}

fn default_search_type() -> SearchTypeRequest {
    SearchTypeRequest::Vector
}

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

fn default_weight() -> f64 {
    1.0
}

fn default_rrf_k() -> f64 {
    DEFAULT_RRF_K
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::super::error::ErrorCode;
    use super::*;

    #[test]
    fn image_only_queries_do_not_prepare_a_text_encoder() {
        use nicegal_core::embedding::{ImageEmbeddingModel, ImageQueryEmbedderOptions};
        let temp = tempfile::tempdir().unwrap();
        let path = camino::Utf8PathBuf::try_from(temp.path().join("images.db")).unwrap();
        drop(ImageIndexDb::new(&path, 768).unwrap());
        let catalog = camino::Utf8PathBuf::try_from(temp.path().join("assets.db")).unwrap();
        drop(nicegal_core::assets::AssetCatalog::new(&catalog).unwrap());
        let images = ImageIndexDb::new_read_only(&path, 768, &catalog).unwrap();
        let query_model =
            super::super::models::ImageQueryModel::deferred(ImageQueryEmbedderOptions {
                model: ImageEmbeddingModel::DinoV3B16,
                ..Default::default()
            });
        let mut resolver = ImageQueryResolver::new(&query_model, &images);
        let error = resolver
            .resolve(
                &ImageQuery::from_text("a cat".into()),
                &SearchCancellation::default(),
            )
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest);
        assert_eq!(error.component_index, Some(0));
        assert!(error.message.contains("image examples only"));
        // A missing indexed example is an index error, never a request to load a text model.
        let error = resolver
            .component_vector(&ImageQuerySource::AssetId(42))
            .unwrap_err();
        assert!(error.message.contains("no current image embedding"));
        assert!(query_model.ready().is_err());
    }

    #[tokio::test]
    async fn dropping_search_cancels_its_blocking_worker() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let request = tokio::spawn(run_search(move |cancellation| {
            started_tx.send(()).unwrap();
            resume_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            finished_tx.send(cancellation.check().is_err()).unwrap();
            Ok(())
        }));
        started_rx.await.unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        resume_tx.send(()).unwrap();
        assert!(finished_rx.await.unwrap());
    }

    #[tokio::test]
    async fn completing_search_does_not_cancel_its_token() {
        let cancellation = run_search(Ok).await.unwrap();
        assert!(cancellation.check().is_ok());
    }

    fn plan(key: &str, weight: f64) -> QueryPlan {
        QueryPlan {
            key: key.to_owned(),
            kind: PlanKind::Text(SearchType::Simple),
            input: QueryInput::Text("x".to_owned()),
            limit: 10,
            max_distance: None,
            weight,
            time: None,
        }
    }

    fn answered(key: &str, ids: &[i64]) -> QueryResponse {
        QueryResponse {
            key: key.to_owned(),
            kind: SearchTypeRequest::Simple,
            total: ids.len(),
            results: ids
                .iter()
                .enumerate()
                .map(|(index, asset_id)| SearchHit {
                    asset_id: *asset_id,
                    snippet: format!("snippet {asset_id}"),
                    rank: index + 1,
                    distance: None,
                    score: None,
                    highlights: Vec::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn query_string_defaults_match_the_documented_contract() {
        let request: SearchRequest =
            serde_urlencoded::from_str("q=receipt&root=C:/gallery").expect("minimal query parses");
        assert_eq!(request.q, "receipt");
        assert_eq!(request.limit, DEFAULT_LIMIT);
        // Vector search is the default mode; the text modes are opt-in.
        assert_eq!(request.kind, SearchTypeRequest::Vector);
        assert_eq!(request.max_distance, None);
    }

    #[test]
    fn the_image_mode_parses_and_is_a_distance_mode() {
        let request: SearchRequest =
            serde_urlencoded::from_str("q=sunset&type=image&root=C:/gallery")
                .expect("type=image parses");
        assert_eq!(request.kind, SearchTypeRequest::Image);
        assert!(request.kind.uses_distance());
        // `vector` and `image` are two engines with a distance each; the text modes have none.
        assert!(SearchTypeRequest::Vector.uses_distance());
        for kind in [
            SearchTypeRequest::Simple,
            SearchTypeRequest::Match,
            SearchTypeRequest::Glob,
        ] {
            assert!(!kind.uses_distance());
        }
        assert_eq!(
            search_type(SearchTypeRequest::Image).unwrap(),
            PlanKind::Image
        );
    }

    #[test]
    fn image_components_support_signed_asset_and_text_arithmetic() {
        let input = query_input(
            PlanKind::Image,
            Some("legacy positive text".to_owned()),
            Some(ImageQueryRequest {
                components: vec![
                    ImageQueryComponentRequest {
                        asset_id: Some(101),
                        external_image: None,
                        text: None,
                        weight: 1.0,
                    },
                    ImageQueryComponentRequest {
                        asset_id: Some(202),
                        external_image: None,
                        text: None,
                        weight: -1.0,
                    },
                    ImageQueryComponentRequest {
                        asset_id: None,
                        external_image: None,
                        text: Some("sports car".to_owned()),
                        weight: -2.0,
                    },
                ],
            }),
            "visual",
        )
        .expect("signed image components are valid");
        let QueryInput::Image(query) = input else {
            panic!("an image plan has an image query");
        };
        assert_eq!(query.components.len(), 4);
        assert!(matches!(
            query.components[0].source,
            ImageQuerySource::AssetId(101)
        ));
        assert_eq!(query.components[1].weight, -1.0);
        assert!(matches!(
            query.components[2].source,
            ImageQuerySource::Text(ref text) if text == "sports car"
        ));
        assert!(matches!(
            query.components[3].source,
            ImageQuerySource::Text(ref text) if text == "legacy positive text"
        ));
        assert_eq!(query.components[3].weight, 1.0);
    }

    #[test]
    fn asset_only_image_query_needs_no_placeholder_text() {
        let input = query_input(
            PlanKind::Image,
            None,
            Some(ImageQueryRequest {
                components: vec![ImageQueryComponentRequest {
                    asset_id: Some(101),
                    external_image: None,
                    text: None,
                    weight: 1.0,
                }],
            }),
            "visual",
        )
        .expect("an indexed asset is enough for an image query");
        assert!(matches!(input, QueryInput::Image(_)));
    }

    #[test]
    fn malformed_image_components_are_rejected_before_inference() {
        let invalid: ImageQueryRequest = serde_json::from_value(serde_json::json!({
            "components": [{"externalImage": {"bytesBase64": "!"}}]
        }))
        .unwrap();
        let error = query_input(PlanKind::Image, None, Some(invalid), "visual").unwrap_err();
        assert_eq!(error.component_index, Some(0));
        assert!(error.message.contains("valid base64"));

        assert!(
            query_input(
                PlanKind::Image,
                None,
                Some(ImageQueryRequest {
                    components: Vec::new(),
                }),
                "visual",
            )
            .is_err(),
            "an image query needs a component"
        );

        assert!(
            query_input(
                PlanKind::Image,
                None,
                Some(ImageQueryRequest {
                    components: vec![ImageQueryComponentRequest {
                        asset_id: Some(1),
                        external_image: None,
                        text: Some("beach".to_owned()),
                        weight: 1.0,
                    }],
                }),
                "visual",
            )
            .is_err(),
            "one component has one source"
        );

        assert!(
            query_input(
                PlanKind::Text(SearchType::Simple),
                Some("beach".to_owned()),
                Some(ImageQueryRequest {
                    components: vec![ImageQueryComponentRequest {
                        asset_id: Some(1),
                        external_image: None,
                        text: None,
                        weight: 1.0,
                    }],
                }),
                "literal",
            )
            .is_err(),
            "image components do not leak into OCR modes"
        );
    }

    #[test]
    fn cancelling_image_directions_are_rejected() {
        assert!(
            normalize_combined_image_vector(vec![0.0, 0.0]).is_err(),
            "a zero direction cannot be searched"
        );
    }

    #[test]
    fn signed_image_components_are_normalized_before_their_sum() {
        let mut combined = vec![0.0, 0.0];
        add_normalized_component(&mut combined, &[3.0, 0.0], 1.0).unwrap();
        add_normalized_component(&mut combined, &[0.0, 4.0], -1.0).unwrap();
        let query = normalize_combined_image_vector(combined).unwrap();
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((query[0] - expected).abs() < 1e-6, "{query:?}");
        assert!((query[1] + expected).abs() < 1e-6, "{query:?}");
    }

    #[test]
    fn unknown_query_parameters_are_rejected() {
        assert!(
            serde_urlencoded::from_str::<SearchRequest>("q=a&root=C:/gallery&bogus=1").is_err()
        );
        // `deny_unknown_fields` and `flatten` are documented as not composing in serde's own
        // derive, so the combination is pinned here rather than assumed: the time fields must
        // reach `TimeRequest` while an unknown one is still rejected.
        assert!(
            serde_urlencoded::from_str::<SearchRequest>(
                "q=a&root=C:/gallery&timeline=capture&after=1&bogus=1"
            )
            .is_err()
        );
    }

    #[test]
    fn the_time_filter_parses_from_the_query_string() {
        let request: SearchRequest = serde_urlencoded::from_str(
            "q=a&root=C:/gallery&timeline=capture&after=1717243200123456789",
        )
        .expect("the documented time parameters parse");
        let range = request
            .time
            .resolve("search")
            .expect("a bound with a timeline is complete")
            .expect("a bound was given");
        assert_eq!(range.timeline, Timeline::Capture);
        // Full nanosecond precision, which is why these cross the wire as decimal strings.
        assert_eq!(range.after_ns, Some(1_717_243_200_123_456_789));
        assert_eq!(range.before_ns, None);

        // No bound at all is an unbounded search, not an error, even with a timeline named.
        let unbounded: SearchRequest =
            serde_urlencoded::from_str("q=a&root=C:/gallery&timeline=modified").unwrap();
        assert_eq!(unbounded.time.resolve("search").unwrap(), None);
    }

    #[test]
    fn rejected_parameters_say_which_one_and_why() {
        let empty = validate_query("   ").expect_err("blank queries are rejected");
        assert_eq!(empty.status, StatusCode::BAD_REQUEST);
        assert_eq!(empty.code, ErrorCode::InvalidRequest);
        assert!(empty.message.contains("must not be empty"), "{empty:?}");

        let long = "x".repeat(MAX_QUERY_BYTES + 1);
        let oversized = validate_query(&long).expect_err("oversized queries are rejected");
        assert!(oversized.message.contains("4096 bytes"), "{oversized:?}");

        for limit in [0, MAX_LIMIT + 1] {
            let error = validate_limit("limit", limit).expect_err("limit is bounded");
            assert_eq!(error.code, ErrorCode::InvalidRequest);
            assert!(error.message.contains("between 1 and"), "{error:?}");
        }
        assert_eq!(validate_limit("limit", 1).unwrap(), 1);
        assert_eq!(validate_limit("limit", MAX_LIMIT).unwrap(), MAX_LIMIT);
    }

    #[test]
    fn a_distance_ceiling_outside_the_cosine_range_is_rejected() {
        for value in [-0.1, 2.1, f64::NAN, f64::INFINITY] {
            let error = validate_max_distance(Some(value)).expect_err("{value} is not a distance");
            assert!(error.message.contains("between 0 and 2"), "{error:?}");
        }
        assert_eq!(validate_max_distance(Some(0.7)).unwrap(), Some(0.7));
        assert_eq!(validate_max_distance(None).unwrap(), None);
    }

    #[test]
    fn fusion_rewards_agreement_between_modes() {
        // 7 is second in both modes; 1 and 2 each top one mode and are absent from the other.
        // RRF is designed so the agreed-on result outranks both leaders.
        let plans = [plan("vector", 1.0), plan("text", 1.0)];
        let queries = [answered("vector", &[1, 7]), answered("text", &[2, 7])];
        let fused = Fusion {
            method: FuseMethod::Rrf,
            k: 1.0,
            limit: 10,
        }
        .apply(&plans, &queries);

        assert_eq!(fused.total, 3);
        assert_eq!(
            fused
                .results
                .iter()
                .map(|hit| hit.asset_id)
                .collect::<Vec<_>>(),
            vec![7, 1, 2]
        );
        assert_eq!(fused.results[0].sources, vec!["vector", "text"]);
        assert_eq!(fused.results[0].rank, 1);
        assert_eq!(fused.results[1].sources, vec!["vector"]);
    }

    #[test]
    fn a_weight_shifts_the_fused_order_without_changing_the_lists() {
        let queries = [answered("vector", &[1, 7]), answered("text", &[2, 7])];
        // Trusting the vector mode three times as much promotes its leader past the agreed hit.
        let plans = [plan("vector", 3.0), plan("text", 1.0)];
        let fused = Fusion {
            method: FuseMethod::Rrf,
            k: 1.0,
            limit: 10,
        }
        .apply(&plans, &queries);
        assert_eq!(fused.results[0].asset_id, 1);
        assert_eq!(fused.total, 3, "weighting does not add or drop assets");
    }

    #[test]
    fn the_fused_limit_caps_the_list_without_hiding_how_many_matched() {
        let plans = [plan("vector", 1.0)];
        let queries = [answered("vector", &[1, 2, 3, 4])];
        let fused = Fusion {
            method: FuseMethod::Rrf,
            k: DEFAULT_RRF_K,
            limit: 2,
        }
        .apply(&plans, &queries);
        assert_eq!(fused.total, 4);
        assert_eq!(fused.results.len(), 2);
        assert_eq!(fused.results[1].rank, 2);
    }

    #[test]
    fn a_combined_request_parses_its_documented_shape() {
        let request: MultiSearchRequest = serde_json::from_value(serde_json::json!({
            "root": "C:/gallery",
            "limit": 50,
            "exclude": "C:/gallery/tmp",
            "queries": [
                {"key": "semantic", "type": "vector", "q": "coffee receipt", "maxDistance": 1.2, "weight": 2.0},
                {"key": "visual", "type": "image", "maxDistance": 0.8,
                 "imageQuery": {"components": [
                   {"assetId": 101, "weight": 1},
                   {"text": "a latte on a marble table", "weight": -0.5}
                 ]}},
                {"key": "literal", "type": "simple", "q": "coffee"}
            ],
            "fuse": {"method": "rrf", "k": 60, "limit": 20}
        }))
        .expect("the documented body parses");
        assert_eq!(request.queries.len(), 3);
        assert_eq!(request.queries[0].max_distance, Some(1.2));
        assert_eq!(request.queries[0].weight, 2.0);
        assert_eq!(request.queries[1].kind, SearchTypeRequest::Image);
        assert_eq!(request.queries[1].max_distance, Some(0.8));
        assert!(request.queries[1].q.is_none());
        assert_eq!(
            request.queries[1]
                .image_query
                .as_ref()
                .unwrap()
                .components
                .len(),
            2
        );
        // An unspecified weight is neutral, and an unspecified fuse block means no fused list.
        assert_eq!(request.queries[2].weight, 1.0);
        assert_eq!(request.fuse.as_ref().unwrap().limit, Some(20));

        assert!(
            serde_json::from_value::<MultiSearchRequest>(serde_json::json!({
                "root": "C:/gallery",
                "queries": [{"q": "a", "bogus": 1}]
            }))
            .is_err(),
            "unknown query fields are rejected like everywhere else"
        );
    }

    fn hit(asset_id: i64, rank: usize) -> SearchHit {
        SearchHit {
            asset_id,
            snippet: format!("snippet {asset_id}"),
            rank,
            distance: None,
            score: None,
            highlights: Vec::new(),
        }
    }

    #[test]
    fn stamp_ranks_derives_rank_from_position_rather_than_trusting_the_caller() {
        // Ranks start deliberately wrong, to prove `stamp_ranks` derives them from position
        // rather than trusting whatever the caller set beforehand.
        let mut hits = [hit(1, 9), hit(2, 9), hit(3, 9)];
        stamp_ranks(&mut hits);
        assert_eq!(
            hits.iter().map(|hit| hit.rank).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            hits.iter().map(|hit| hit.asset_id).collect::<Vec<_>>(),
            vec![1, 2, 3],
            "stamp_ranks does not reorder hits, only labels their existing order"
        );
    }
}
