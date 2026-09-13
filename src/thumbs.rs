use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use camino::Utf8Path as Path;
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use tracing::{Span, debug_span, field, trace_span};

use crate::assets::{Asset, MediaKind, SourceFingerprint};
use crate::imaging;
use crate::poster;
use crate::schema::{check_schema_read_only, open_schema};

const SCHEMA_VERSION: i32 = 3;
const SCHEMA_LABEL: &str = "thumbnail database";
pub const SIZE_BUCKETS: [u16; 4] = [128, 256, 512, 1024];
pub const EAGER_SIZE_BUCKETS: [u16; 3] = [128, 256, 512];
pub const GENERATOR_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailEncoding {
    Jpeg,
    Png,
    Webp,
}

impl ThumbnailEncoding {
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Webp => "image/webp",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "image/jpeg" => Ok(Self::Jpeg),
            "image/png" => Ok(Self::Png),
            "image/webp" => Ok(Self::Webp),
            _ => bail!("unsupported static thumbnail encoding: {value}"),
        }
    }
}

/// A static derived artifact. Animated originals are never stored in this database.
#[derive(Debug, PartialEq, Eq)]
pub struct Thumbnail {
    pub size_bucket: u16,
    pub generator_version: u32,
    pub width: u32,
    pub height: u32,
    pub encoding: ThumbnailEncoding,
    pub data: Vec<u8>,
}

/// An owned, validated-on-write thumbnail awaiting durable persistence. Decoding creates these
/// values without borrowing either an image decoder or a SQLite connection, so callers can run
/// expensive source work independently from the one thumbnail writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedThumbnail {
    pub asset_id: i64,
    pub size_bucket: u16,
    pub generator_version: u32,
    pub fingerprint: SourceFingerprint,
    pub width: u32,
    pub height: u32,
    pub encoding: ThumbnailEncoding,
    pub data: Vec<u8>,
}

pub struct ThumbnailDb {
    conn: Connection,
}

impl ThumbnailDb {
    pub fn new(path: &Path) -> Result<Self> {
        let span = debug_span!("thumbnail_db_open", path = %path, read_only = false);
        let _entered = span.enter();
        let conn = Connection::open(path)
            .with_context(|| format!("opening thumbnail database: {path}"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.pragma_update(None, "auto_vacuum", "FULL")?;
        conn.pragma_update(None, "journal_mode", "wal")?;
        conn.pragma_update(None, "synchronous", "normal")?;
        open_schema(
            &conn,
            SCHEMA_LABEL,
            SCHEMA_VERSION,
            include_str!("thumbs_create.sql"),
        )?;
        Ok(Self { conn })
    }

    /// Open a query-only connection for serving stored variants. Unlike [`ThumbnailDb::new`] this
    /// never creates the schema, so a reader cannot bring an empty database into existence.
    pub fn new_read_only(path: &Path) -> Result<Self> {
        let span = debug_span!("thumbnail_db_open", path = %path, read_only = true);
        let _entered = span.enter();
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening thumbnail database read-only: {path}"))?;
        conn.busy_timeout(Duration::from_secs(5))?;
        check_schema_read_only(&conn, SCHEMA_LABEL, SCHEMA_VERSION)?;
        Ok(Self { conn })
    }

    /// Store or replace exactly one size and generator variant.
    #[allow(clippy::too_many_arguments)]
    pub fn put(
        &self,
        asset_id: i64,
        size_bucket: u16,
        generator_version: u32,
        fingerprint: SourceFingerprint,
        width: u32,
        height: u32,
        encoding: ThumbnailEncoding,
        data: &[u8],
    ) -> Result<()> {
        self.store_batch(&[DecodedThumbnail {
            asset_id,
            size_bucket,
            generator_version,
            fingerprint,
            width,
            height,
            encoding,
            data: data.to_vec(),
        }])
    }

    /// Persist a bounded decoded batch atomically. Every row is validated before the transaction
    /// begins, so malformed input cannot leave an earlier row from the same batch committed.
    pub fn store_batch(&self, thumbnails: &[DecodedThumbnail]) -> Result<()> {
        for thumbnail in thumbnails {
            validate_key(
                thumbnail.asset_id,
                thumbnail.size_bucket,
                thumbnail.generator_version,
            )?;
            validate_static_thumbnail(
                thumbnail.size_bucket,
                thumbnail.width,
                thumbnail.height,
                thumbnail.encoding,
                &thumbnail.data,
            )?;
            i64::try_from(thumbnail.fingerprint.size)
                .context("source byte size exceeds SQLite's integer range")?;
        }
        if thumbnails.is_empty() {
            return Ok(());
        }

        let span = debug_span!(
            "thumbnail_store_batch",
            variants = thumbnails.len(),
            data_bytes = thumbnails
                .iter()
                .map(|thumbnail| thumbnail.data.len())
                .sum::<usize>(),
        );
        let _entered = span.enter();
        let transaction = self.conn.unchecked_transaction()?;
        let mut statement = transaction.prepare_cached(
            "INSERT INTO thumbnails (asset_id, size_bucket, generator_version, source_modified_ns, source_size, width, height, encoding, data) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) \
             ON CONFLICT(asset_id, size_bucket, generator_version) DO UPDATE SET source_modified_ns=excluded.source_modified_ns, source_size=excluded.source_size, width=excluded.width, height=excluded.height, encoding=excluded.encoding, data=excluded.data",
        )?;
        for thumbnail in thumbnails {
            statement.execute((
                thumbnail.asset_id,
                thumbnail.size_bucket,
                thumbnail.generator_version,
                thumbnail.fingerprint.modified_ns,
                i64::try_from(thumbnail.fingerprint.size)?,
                thumbnail.width,
                thumbnail.height,
                thumbnail.encoding.content_type(),
                &thumbnail.data,
            ))?;
        }
        drop(statement);
        transaction.commit().context("committing thumbnail batch")?;
        Ok(())
    }

    pub fn has_current(
        &self,
        asset_id: i64,
        size_bucket: u16,
        generator_version: u32,
        fingerprint: SourceFingerprint,
    ) -> Result<bool> {
        validate_key(asset_id, size_bucket, generator_version)?;
        let span = trace_span!(
            "thumbnail_cache_lookup",
            asset_id,
            size_bucket,
            generator_version,
            current = field::Empty,
        );
        let _entered = span.enter();
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let current = self.conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM thumbnails WHERE asset_id = ?1 AND size_bucket = ?2 AND generator_version = ?3 AND source_modified_ns = ?4 AND source_size = ?5)",
                (
                    asset_id,
                    size_bucket,
                    generator_version,
                    fingerprint.modified_ns,
                    source_size,
                ),
                |row| row.get(0),
            )
            .context("checking current thumbnail variant")?;
        span.record("current", current);
        Ok(current)
    }

    /// Remove variants from older generators after the current generator has been backfilled.
    pub fn sweep_old_generators(&self, generator_version: u32) -> Result<usize> {
        if generator_version == 0 {
            bail!("generator version must be greater than zero");
        }
        self.conn
            .execute(
                "DELETE FROM thumbnails WHERE generator_version <> ?1",
                [generator_version],
            )
            .context("sweeping stale thumbnail generator versions")
    }

    /// Remove variants from older generators for the specified catalog assets only.
    pub fn sweep_old_generators_for_assets(
        &self,
        generator_version: u32,
        asset_ids: impl IntoIterator<Item = i64>,
    ) -> Result<usize> {
        if generator_version == 0 {
            bail!("generator version must be greater than zero");
        }
        let transaction = self.conn.unchecked_transaction()?;
        let mut statement = transaction.prepare_cached(
            "DELETE FROM thumbnails WHERE asset_id = ?1 AND generator_version <> ?2",
        )?;
        let mut deleted = 0;
        for asset_id in asset_ids {
            if asset_id <= 0 {
                bail!("asset identifier must be greater than zero");
            }
            deleted += statement.execute((asset_id, generator_version))?;
        }
        drop(statement);
        transaction
            .commit()
            .context("sweeping scoped stale thumbnail generators")?;
        Ok(deleted)
    }

    /// Select the smallest current bucket that satisfies the requested physical size.
    /// If none is large enough, return the largest current smaller bucket.
    pub fn get(
        &self,
        asset_id: i64,
        requested_physical_size: u32,
        generator_version: u32,
        fingerprint: SourceFingerprint,
    ) -> Result<Option<Thumbnail>> {
        if asset_id <= 0 {
            bail!("asset identifier must be greater than zero");
        }
        if requested_physical_size == 0 {
            bail!("requested physical size must be greater than zero");
        }
        if generator_version == 0 {
            bail!("generator version must be greater than zero");
        }
        let span = trace_span!(
            "thumbnail_cache_get",
            asset_id,
            requested_physical_size,
            generator_version,
            hit = field::Empty,
            size_bucket = field::Empty,
            data_bytes = field::Empty,
        );
        let _entered = span.enter();
        let requested = i64::from(requested_physical_size);
        let source_size = i64::try_from(fingerprint.size)
            .context("source byte size exceeds SQLite's integer range")?;
        let thumbnail = self.conn
            .query_row(
                "SELECT size_bucket, generator_version, width, height, encoding, data \
                 FROM thumbnails \
                 WHERE asset_id = ?1 AND generator_version = ?2 AND source_modified_ns = ?3 AND source_size = ?4 \
                 ORDER BY CASE WHEN size_bucket >= ?5 THEN 0 ELSE 1 END, \
                          CASE WHEN size_bucket >= ?5 THEN size_bucket END ASC, \
                          CASE WHEN size_bucket < ?5 THEN size_bucket END DESC \
                 LIMIT 1",
                (
                    asset_id,
                    generator_version,
                    fingerprint.modified_ns,
                    source_size,
                    requested,
                ),
                |row| {
                    let encoding: String = row.get(4)?;
                    Ok((
                        row.get::<_, u16>(0)?,
                        row.get::<_, u32>(1)?,
                        row.get::<_, u32>(2)?,
                        row.get::<_, u32>(3)?,
                        encoding,
                        row.get::<_, Vec<u8>>(5)?,
                    ))
                },
            )
            .optional()?
            .map(|(size_bucket, generator_version, width, height, encoding, data)| -> Result<Thumbnail> {
                Ok(Thumbnail {
                    size_bucket,
                    generator_version,
                    width,
                    height,
                    encoding: ThumbnailEncoding::parse(&encoding)?,
                    data,
                })
            })
            .transpose()
            .context("loading thumbnail variant")?;
        if let Some(thumbnail) = &thumbnail {
            span.record("hit", true);
            span.record("size_bucket", thumbnail.size_bucket);
            span.record("data_bytes", thumbnail.data.len());
        } else {
            span.record("hit", false);
        }
        Ok(thumbnail)
    }

    /// Delete every size and generator variant belonging to an asset.
    pub fn delete_asset(&self, asset_id: i64) -> Result<usize> {
        self.delete_assets(&[asset_id])
    }

    pub fn delete_assets(&self, asset_ids: &[i64]) -> Result<usize> {
        if asset_ids.iter().any(|asset_id| *asset_id <= 0) {
            bail!("asset identifiers must be greater than zero");
        }
        if asset_ids.is_empty() {
            return Ok(0);
        }
        let transaction = self.conn.unchecked_transaction()?;
        let deleted = {
            let mut statement =
                transaction.prepare("DELETE FROM thumbnails WHERE asset_id = ?1")?;
            asset_ids.iter().try_fold(0usize, |deleted, asset_id| {
                statement.execute([asset_id]).map(|count| deleted + count)
            })?
        };
        transaction
            .commit()
            .context("committing asset thumbnail deletions")?;
        Ok(deleted)
    }
}

/// Decode all requested variants for one image in one source pass. Persistence is deliberately a
/// separate stage: callers must hand the returned values to [`ThumbnailDb::store_batch`] before
/// treating the variants as complete.
pub fn decode_asset_variants(
    asset: &Asset,
    buckets: &[u16],
    generator_version: u32,
) -> Result<Vec<DecodedThumbnail>> {
    if asset.media_kind != MediaKind::Image {
        bail!("asset {} is not an image", asset.asset_id);
    }
    for bucket in buckets {
        validate_key(asset.asset_id, *bucket, generator_version)?;
    }
    let span = debug_span!(
        "thumbnail_source_decode",
        asset_id = asset.asset_id,
        path = %asset.path,
        media_format = %asset.media_format,
        source_bytes = asset.fingerprint.size,
        source_width = ?asset.width,
        source_height = ?asset.height,
        is_gif = asset.media_format == "gif",
        variants = buckets.len(),
        buckets = ?buckets,
    );
    let _entered = span.enter();
    let posters = poster::image_buckets(&asset.path, buckets)?;
    Ok(buckets
        .iter()
        .copied()
        .zip(posters)
        .map(|(size_bucket, poster)| DecodedThumbnail {
            asset_id: asset.asset_id,
            size_bucket,
            generator_version,
            fingerprint: asset.fingerprint,
            width: poster.width,
            height: poster.height,
            encoding: poster.encoding,
            data: poster.data,
        })
        .collect())
}

pub fn validate_static_thumbnail(
    size_bucket: u16,
    width: u32,
    height: u32,
    encoding: ThumbnailEncoding,
    data: &[u8],
) -> Result<()> {
    if !SIZE_BUCKETS.contains(&size_bucket) {
        bail!("thumbnail size bucket must be one of 128, 256, 512, or 1024");
    }
    if width == 0 || height == 0 {
        bail!("thumbnail dimensions must be greater than zero");
    }
    if data.is_empty() {
        bail!("thumbnail data must not be empty");
    }
    let detected = imaging::Format::detect(data).context("detecting thumbnail encoding")?;
    let expected = expected_format(encoding);
    if detected != expected {
        bail!(
            "thumbnail bytes are {detected:?}, not the declared static {} encoding",
            encoding.content_type()
        );
    }
    let (actual_width, actual_height) = thumbnail_dimensions(data, size_bucket)?;
    if (actual_width, actual_height) != (width, height) {
        bail!(
            "thumbnail dimensions are {actual_width}x{actual_height}, not the declared {width}x{height}"
        );
    }
    Ok(())
}

fn expected_format(encoding: ThumbnailEncoding) -> imaging::Format {
    match encoding {
        ThumbnailEncoding::Jpeg => imaging::Format::Jpeg,
        ThumbnailEncoding::Png => imaging::Format::Png,
        ThumbnailEncoding::Webp => imaging::Format::Webp,
    }
}

/// Reject an oversized declared size before decoding (a cheap header-only check), then fully
/// decode to confirm the bytes are genuinely well-formed and read their real dimensions.
fn thumbnail_dimensions(data: &[u8], size_bucket: u16) -> Result<(u32, u32)> {
    // A hint that fails to parse (e.g. genuinely truncated data) just falls through to the real
    // decode below, which reports it properly; this check only short-circuits the oversized case.
    if let Ok(hint) = imagesize::blob_size(data)
        && (hint.width > usize::from(size_bucket) || hint.height > usize::from(size_bucket))
    {
        bail!("thumbnail dimensions exceed size bucket {size_bucket}");
    }
    let decoded = imaging::decode(data).context("decoding static thumbnail bytes")?;
    Ok((decoded.width(), decoded.height()))
}

fn validate_key(asset_id: i64, size_bucket: u16, generator_version: u32) -> Result<()> {
    if asset_id <= 0 {
        bail!("asset identifier must be greater than zero");
    }
    if !SIZE_BUCKETS.contains(&size_bucket) {
        bail!("thumbnail size bucket must be one of 128, 256, 512, or 1024");
    }
    if generator_version == 0 {
        bail!("generator version must be greater than zero");
    }
    Ok(())
}

const WRITER_QUEUE_CAPACITY: usize = 64;
const GENERATION_ASSET_CHUNK_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerateSummary {
    pub generated: usize,
    pub skipped: usize,
}

#[derive(Debug)]
pub struct AssetGeneration {
    pub asset: Asset,
    pub result: Result<GenerateSummary>,
}

/// Identifies one decoded thumbnail variant: the unit that single-flight dedup and the writer
/// actor both key on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VariantKey {
    asset_id: i64,
    size_bucket: u16,
    generator_version: u32,
    fingerprint: SourceFingerprint,
}

/// How a [`Flight`] ended. `Failed` carries a shareable string because every waiter clones it.
#[derive(Clone)]
enum FlightResult {
    Stored,
    Cached,
    Failed(Arc<str>),
}

/// A single-flight slot: the first caller to request a [`VariantKey`] becomes its owner and does
/// the decode work, while every other caller for the same key waits here instead of duplicating it.
struct Flight {
    result: Mutex<Option<FlightResult>>,
    complete: Condvar,
}

impl Flight {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            complete: Condvar::new(),
        }
    }

    fn finish(&self, result: FlightResult) {
        let mut state = self
            .result
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *state = Some(result);
        self.complete.notify_all();
    }

    fn wait(&self) -> FlightResult {
        let started = Instant::now();
        let mut state = self
            .result
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        while state.is_none() {
            state = self
                .complete
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        tracing::debug!(
            wait_ms = started.elapsed().as_millis() as u64,
            "thumbnail flight waiter completed"
        );
        state
            .as_ref()
            .expect("flight completion was checked")
            .clone()
    }
}

/// `flights` is the single-flight registry; `writer`/`writer_thread` front the one thread allowed
/// to touch the writable SQLite connection; `pool` runs decodes off that thread.
struct ServiceInner {
    flights: Mutex<HashMap<VariantKey, Arc<Flight>>>,
    writer: mpsc::SyncSender<WriterCommand>,
    writer_thread: Mutex<Option<thread::JoinHandle<()>>>,
    pool: rayon::ThreadPool,
    #[cfg(test)]
    decode_count: AtomicUsize,
}

/// The process-wide thumbnail coordinator. It owns a bounded decode pool, a short-held
/// single-flight registry, and the sole writable thumbnail database connection.
#[derive(Clone)]
pub struct ThumbnailService {
    inner: Arc<ServiceInner>,
}

/// One request for the writer thread. Each variant carries its own reply channel so a caller
/// blocks only on its own response, never on the rest of the queue.
enum WriterCommand {
    Current {
        keys: Vec<VariantKey>,
        reply: mpsc::Sender<Result<Vec<bool>, Arc<str>>>,
    },
    Store {
        variants: Vec<DecodedThumbnail>,
        reply: mpsc::Sender<Result<(), Arc<str>>>,
    },
    Put {
        variant: DecodedThumbnail,
        reply: mpsc::Sender<Result<(), Arc<str>>>,
    },
    Delete {
        asset_ids: Vec<i64>,
        reply: mpsc::Sender<Result<usize, Arc<str>>>,
    },
    Sweep {
        generator_version: u32,
        asset_ids: Vec<i64>,
        reply: mpsc::Sender<Result<usize, Arc<str>>>,
    },
    Shutdown {
        reply: mpsc::Sender<()>,
    },
}

/// One asset this [`ThumbnailService::generate`] call is waiting on — whether it owns the decode
/// work for that asset or only joined flights another concurrent caller already started.
struct Waiter {
    asset: Asset,
    flights: Vec<Arc<Flight>>,
    /// Buckets skipped before any flight existed (non-image assets never get one).
    skipped: usize,
}

/// An asset for which this call became the owner of at least one flight, and therefore must
/// check, decode, and persist its missing buckets.
struct Claimed {
    asset: Asset,
    keys: Vec<VariantKey>,
    flights: Vec<Arc<Flight>>,
    /// Parallel to `keys`: whether the stored variant is already current. Filled in by
    /// [`ThumbnailService::mark_current`].
    current: Vec<bool>,
}

impl Drop for ServiceInner {
    fn drop(&mut self) {
        let (reply, response) = mpsc::channel();
        let _ = self.writer.send(WriterCommand::Shutdown { reply });
        let _ = response.recv();
        if let Some(thread) = self
            .writer_thread
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            let _ = thread.join();
        }
    }
}

impl ThumbnailService {
    pub fn new(path: &Path) -> Result<Self> {
        // Open before spawning so startup reports schema errors synchronously.
        drop(ThumbnailDb::new(path)?);
        let (sender, receiver) = mpsc::sync_channel(WRITER_QUEUE_CAPACITY);
        let path = path.to_owned();
        let writer_thread = thread::Builder::new()
            .name("thumbnail-writer".to_owned())
            .spawn(move || thumbnail_writer(path, receiver))
            .context("starting thumbnail writer thread")?;
        let threads = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 8);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("thumbnail-decode-{index}"))
            .build()
            .context("building thumbnail decode pool")?;
        Ok(Self {
            inner: Arc::new(ServiceInner {
                flights: Mutex::new(HashMap::new()),
                writer: sender,
                writer_thread: Mutex::new(Some(writer_thread)),
                pool,
                #[cfg(test)]
                decode_count: AtomicUsize::new(0),
            }),
        })
    }

    /// Durably ensure requested variants. An owner is elected before cache lookup; this closes
    /// the lookup/claim race while registry synchronization remains limited to map operations.
    ///
    /// Each chunk moves through four stages: [`Self::claim_chunk`] elects an owner per variant,
    /// [`Self::mark_current`] asks the writer which owned variants are already up to date,
    /// [`Self::decode_claimed`] decodes only the stale ones, and [`Self::persist_decoded`] writes
    /// and completes their flights. Every waiter — owner or joiner — then blocks on its flights.
    pub fn generate<F>(
        &self,
        assets: &[Asset],
        buckets: &[u16],
        generator_version: u32,
        force: bool,
        is_cancelled: F,
    ) -> Vec<AssetGeneration>
    where
        F: Fn() -> bool,
    {
        self.generate_observed(
            assets,
            buckets,
            generator_version,
            force,
            is_cancelled,
            |_, _| {},
        )
    }

    /// As [`Self::generate`], while reporting the assets whose source bytes are actively being
    /// decoded. The callback can be invoked concurrently by the decode pool.
    pub fn generate_observed<F, G>(
        &self,
        assets: &[Asset],
        buckets: &[u16],
        generator_version: u32,
        force: bool,
        is_cancelled: F,
        active: G,
    ) -> Vec<AssetGeneration>
    where
        F: Fn() -> bool,
        G: Fn(&Asset, bool) + Send + Sync,
    {
        let mut generations = Vec::with_capacity(assets.len());
        for chunk in assets.chunks(GENERATION_ASSET_CHUNK_SIZE) {
            if is_cancelled() {
                break;
            }
            let (waiters, mut claimed) = self.claim_chunk(chunk, buckets, generator_version);
            match self.mark_current(&mut claimed, force) {
                // The cache check itself failed (writer stopped, DB error): every flight this
                // call owns must still be woken, or its joiners would block forever.
                Err(error) => {
                    for work in claimed {
                        self.fail_all(
                            work.keys,
                            work.flights,
                            anyhow::Error::msg(error.to_string()),
                        );
                    }
                }
                Ok(()) => {
                    let decoded = self.decode_claimed(claimed, &active);
                    self.persist_decoded(decoded);
                }
            }
            generations.extend(waiters.into_iter().map(Self::finish_waiter));
        }
        generations
    }

    /// Elect an owner for each requested `(asset, bucket)` pair. Electing before any cache lookup
    /// closes the lookup/claim race: whoever wins `claim` is guaranteed to be the one who checks
    /// and, if needed, decodes that variant.
    fn claim_chunk(
        &self,
        chunk: &[Asset],
        buckets: &[u16],
        generator_version: u32,
    ) -> (Vec<Waiter>, Vec<Claimed>) {
        let mut waiters = Vec::with_capacity(chunk.len());
        let mut claimed = Vec::new();
        for asset in chunk {
            // Non-image assets (video, audio) never had thumbnail flights to begin with.
            if asset.media_kind != MediaKind::Image {
                waiters.push(Waiter {
                    asset: asset.clone(),
                    flights: Vec::new(),
                    skipped: buckets.len(),
                });
                continue;
            }
            let mut flights = Vec::with_capacity(buckets.len());
            let mut owner_keys = Vec::new();
            let mut owner_flights = Vec::new();
            for &size_bucket in buckets {
                let key = VariantKey {
                    asset_id: asset.asset_id,
                    size_bucket,
                    generator_version,
                    fingerprint: asset.fingerprint,
                };
                let (flight, is_owner) = self.claim(key);
                flights.push(Arc::clone(&flight));
                if is_owner {
                    owner_keys.push(key);
                    owner_flights.push(flight);
                } else {
                    tracing::debug!(
                        asset_id = asset.asset_id,
                        size_bucket,
                        "thumbnail flight joined"
                    );
                }
            }
            if !owner_keys.is_empty() {
                claimed.push(Claimed {
                    asset: asset.clone(),
                    keys: owner_keys,
                    flights: owner_flights,
                    current: Vec::new(),
                });
            }
            waiters.push(Waiter {
                asset: asset.clone(),
                flights,
                skipped: 0,
            });
        }
        (waiters, claimed)
    }

    /// Ask the writer which claimed variants are already current, and fill in `claimed[_].current`
    /// so [`Self::decode_claimed`] only decodes what actually changed. `force` skips the query.
    fn mark_current(&self, claimed: &mut [Claimed], force: bool) -> Result<(), Arc<str>> {
        // One flat request covers every owned key across the whole chunk in a single round trip
        // to the writer thread; the offsets below split the flat reply back out per asset.
        let keys: Vec<_> = claimed
            .iter()
            .flat_map(|work| work.keys.iter().copied())
            .collect();
        let current: Vec<bool> = if force {
            vec![false; keys.len()]
        } else {
            self.request(|reply| WriterCommand::Current { keys, reply })
                .context("checking thumbnail cache")
                .map_err(|error| Arc::<str>::from(format!("{error:#}")))?
        };
        let mut offset = 0;
        for work in claimed.iter_mut() {
            let end = offset + work.keys.len();
            work.current = current[offset..end].to_vec();
            offset = end;
        }
        Ok(())
    }

    /// Decode every claimed asset's stale buckets in parallel on the decode pool. An asset whose
    /// buckets are all already current skips the source pass entirely.
    fn decode_claimed<G>(
        &self,
        claimed: Vec<Claimed>,
        active: &G,
    ) -> Vec<(Claimed, Result<Vec<DecodedThumbnail>>)>
    where
        G: Fn(&Asset, bool) + Send + Sync,
    {
        let parent = Span::current();
        self.inner.pool.install(|| {
            claimed
                .into_par_iter()
                .map(|work| {
                    let _entered = parent.enter();
                    let buckets: Vec<_> = work
                        .keys
                        .iter()
                        .zip(&work.current)
                        .filter_map(|(key, current)| (!*current).then_some(key.size_bucket))
                        .collect();
                    let variants = if buckets.is_empty() {
                        Ok(Vec::new())
                    } else {
                        active(&work.asset, true);
                        #[cfg(test)]
                        self.inner.decode_count.fetch_add(1, Ordering::Relaxed);
                        tracing::debug!(
                            asset_id = work.asset.asset_id,
                            variants = buckets.len(),
                            "thumbnail source decode"
                        );
                        let result = decode_asset_variants(
                            &work.asset,
                            &buckets,
                            work.keys[0].generator_version,
                        );
                        active(&work.asset, false);
                        result
                    };
                    (work, variants)
                })
                .collect::<Vec<_>>()
        })
    }

    /// Persist every successfully decoded variant in one writer batch, then complete each
    /// asset's flights. An asset whose decode failed fails its flights immediately instead of
    /// entering the batch, so one bad source image cannot hold up its siblings' writes.
    fn persist_decoded(&self, decoded: Vec<(Claimed, Result<Vec<DecodedThumbnail>>)>) {
        let mut persisted = Vec::with_capacity(decoded.len());
        let mut variants = Vec::new();
        for (work, result) in decoded {
            match result {
                Ok(decoded) => {
                    variants.extend(decoded);
                    persisted.push(work);
                }
                Err(error) => self.fail_all(work.keys, work.flights, error),
            }
        }
        let stored: Result<(), Arc<str>> = if variants.is_empty() {
            Ok(())
        } else {
            self.request(|reply| WriterCommand::Store { variants, reply })
                .map_err(|error| Arc::<str>::from(format!("{error:#}")))
        };
        // A variant that was already current is `Cached` regardless of whether this batch write
        // succeeded; only the newly-decoded variants share the batch's outcome.
        for work in persisted {
            for ((key, flight), current) in
                work.keys.into_iter().zip(work.flights).zip(work.current)
            {
                let result = match (&stored, current) {
                    (_, true) => FlightResult::Cached,
                    (Ok(()), false) => FlightResult::Stored,
                    (Err(error), false) => FlightResult::Failed(Arc::clone(error)),
                };
                self.complete(key, flight, result);
            }
        }
    }

    /// Block on every flight a waiter depends on and fold the outcomes into one summary. A single
    /// failed flight fails the whole asset, since a partial set of variants isn't usable.
    fn finish_waiter(waiter: Waiter) -> AssetGeneration {
        let mut generated = 0;
        let mut skipped = waiter.skipped;
        for flight in waiter.flights {
            match flight.wait() {
                FlightResult::Stored => generated += 1,
                FlightResult::Cached => skipped += 1,
                FlightResult::Failed(error) => {
                    return AssetGeneration {
                        asset: waiter.asset,
                        result: Err(anyhow::Error::msg(error.to_string())),
                    };
                }
            }
        }
        AssetGeneration {
            asset: waiter.asset,
            result: Ok(GenerateSummary { generated, skipped }),
        }
    }

    pub fn put(&self, variant: DecodedThumbnail) -> Result<()> {
        self.request(|reply| WriterCommand::Put { variant, reply })
    }

    pub fn delete_asset(&self, asset_id: i64) -> Result<usize> {
        self.delete_assets(vec![asset_id])
    }

    pub fn delete_assets(&self, asset_ids: Vec<i64>) -> Result<usize> {
        self.request(|reply| WriterCommand::Delete { asset_ids, reply })
    }

    pub fn sweep_old_generators_for_assets(
        &self,
        generator_version: u32,
        asset_ids: Vec<i64>,
    ) -> Result<usize> {
        self.request(|reply| WriterCommand::Sweep {
            generator_version,
            asset_ids,
            reply,
        })
    }

    /// Join the flight for `key`, creating and registering it as the owner if none exists yet.
    /// The bool reports ownership: `true` means the caller must drive this variant to completion.
    fn claim(&self, key: VariantKey) -> (Arc<Flight>, bool) {
        let mut registry = self
            .inner
            .flights
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(flight) = registry.get(&key) {
            return (Arc::clone(flight), false);
        }
        let flight = Arc::new(Flight::new());
        registry.insert(key, Arc::clone(&flight));
        tracing::debug!(
            asset_id = key.asset_id,
            size_bucket = key.size_bucket,
            "thumbnail flight owner"
        );
        (flight, true)
    }

    /// Wake every waiter on `flight` and, if it is still the registered owner for `key`, evict it
    /// so a future request starts a fresh flight instead of joining a finished one.
    fn complete(&self, key: VariantKey, flight: Arc<Flight>, result: FlightResult) {
        flight.finish(result);
        let mut registry = self
            .inner
            .flights
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if registry
            .get(&key)
            .is_some_and(|registered| Arc::ptr_eq(registered, &flight))
        {
            registry.remove(&key);
        }
    }

    /// Complete every listed flight as `Failed` with the same shared error message.
    fn fail_all(&self, keys: Vec<VariantKey>, flights: Vec<Arc<Flight>>, error: anyhow::Error) {
        let error: Arc<str> = format!("{error:#}").into();
        for (key, flight) in keys.into_iter().zip(flights) {
            self.complete(key, flight, FlightResult::Failed(Arc::clone(&error)));
        }
    }

    /// Send one command to the writer thread and block for its reply. Every public mutation and
    /// lookup on [`ThumbnailService`] funnels through here, which is what keeps SQLite access
    /// confined to the single writer thread.
    fn request<T>(
        &self,
        command: impl FnOnce(mpsc::Sender<Result<T, Arc<str>>>) -> WriterCommand,
    ) -> Result<T> {
        let (reply, response) = mpsc::channel();
        self.inner
            .writer
            .send(command(reply))
            .map_err(|_| anyhow::anyhow!("thumbnail writer stopped"))?;
        response
            .recv()
            .map_err(|_| anyhow::anyhow!("thumbnail writer stopped"))?
            .map_err(|error| anyhow::Error::msg(error.to_string()))
    }

    #[cfg(test)]
    fn decode_count(&self) -> usize {
        self.inner.decode_count.load(Ordering::Relaxed)
    }
}

/// The single writer actor. It owns the only writable connection and drains `receiver` until
/// `Shutdown`, so every database mutation in the process is serialized through this one thread.
fn thumbnail_writer(path: camino::Utf8PathBuf, receiver: mpsc::Receiver<WriterCommand>) {
    // The connection opens lazily and retries on every command until it succeeds. A one-shot
    // open here would otherwise wedge the service for the rest of the process's life if it lost
    // a race against something transient (a drive not yet mounted, a momentary file lock),
    // since this thread would never get another chance to reopen on its own.
    let mut database: Option<ThumbnailDb> = None;
    for command in receiver {
        match command {
            WriterCommand::Current { keys, reply } => {
                let _ = reply.send(respond(&mut database, &path, |database| {
                    keys.iter()
                        .map(|key| {
                            database.has_current(
                                key.asset_id,
                                key.size_bucket,
                                key.generator_version,
                                key.fingerprint,
                            )
                        })
                        .collect()
                }));
            }
            WriterCommand::Store { variants, reply } => {
                let _ = reply.send(respond(&mut database, &path, |database| {
                    database.store_batch(&variants)
                }));
            }
            WriterCommand::Put { variant, reply } => {
                let _ = reply.send(respond(&mut database, &path, |database| {
                    database.store_batch(&[variant])
                }));
            }
            WriterCommand::Delete { asset_ids, reply } => {
                let _ = reply.send(respond(&mut database, &path, |database| {
                    database.delete_assets(&asset_ids)
                }));
            }
            WriterCommand::Sweep {
                generator_version,
                asset_ids,
                reply,
            } => {
                let _ = reply.send(respond(&mut database, &path, |database| {
                    database.sweep_old_generators_for_assets(generator_version, asset_ids)
                }));
            }
            WriterCommand::Shutdown { reply } => {
                let _ = reply.send(());
                break;
            }
        }
    }
}

/// Ensure the writer's connection is open — retrying a previously failed open, since whatever
/// caused it (e.g. an unmounted drive) may have cleared by the time the next command arrives —
/// then run one database operation against it, collapsing either error into the `Arc<str>` every
/// reply channel expects.
fn respond<T>(
    database: &mut Option<ThumbnailDb>,
    path: &Path,
    operation: impl FnOnce(&ThumbnailDb) -> Result<T>,
) -> Result<T, Arc<str>> {
    let database = match database {
        Some(database) => database,
        None => database.insert(
            ThumbnailDb::new(path).map_err(|error| Arc::<str>::from(format!("{error:#}")))?,
        ),
    };
    operation(database).map_err(|error| Arc::<str>::from(format!("{error:#}")))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;
    use crate::assets::AssetCatalog;
    use camino::Utf8PathBuf as PathBuf;
    use tempfile::TempDir;

    const CURRENT: SourceFingerprint = SourceFingerprint {
        modified_ns: 123_456_789,
        size: 4000,
    };
    fn test_db(temp: &TempDir) -> Result<ThumbnailDb> {
        ThumbnailDb::new(&PathBuf::try_from(temp.path().join("thumbnails.db"))?)
    }

    fn png() -> Result<Vec<u8>> {
        imaging::encode_png(1, 1, &[0, 0, 0, 0])
    }

    fn put(
        db: &ThumbnailDb,
        bucket: u16,
        version: u32,
        fingerprint: SourceFingerprint,
    ) -> Result<()> {
        let data = png()?;
        db.put(
            42,
            bucket,
            version,
            fingerprint,
            1,
            1,
            ThumbnailEncoding::Png,
            &data,
        )
    }

    #[test]
    fn one_asset_holds_multiple_buckets() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        put(&db, 128, 1, CURRENT)?;
        put(&db, 512, 1, CURRENT)?;

        assert_eq!(db.get(42, 100, 1, CURRENT)?.unwrap().size_bucket, 128);
        assert_eq!(db.get(42, 300, 1, CURRENT)?.unwrap().size_bucket, 512);
        Ok(())
    }

    #[test]
    fn lookup_uses_smallest_adequate_then_largest_smaller() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        for bucket in [128, 256, 512] {
            put(&db, bucket, 1, CURRENT)?;
        }

        assert_eq!(db.get(42, 129, 1, CURRENT)?.unwrap().size_bucket, 256);
        assert_eq!(db.get(42, 900, 1, CURRENT)?.unwrap().size_bucket, 512);
        Ok(())
    }

    #[test]
    fn source_change_invalidates_variants() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        put(&db, 256, 1, CURRENT)?;

        assert_eq!(
            db.get(
                42,
                256,
                1,
                SourceFingerprint {
                    modified_ns: CURRENT.modified_ns + 1,
                    ..CURRENT
                },
            )?,
            None
        );
        assert_eq!(
            db.get(
                42,
                256,
                1,
                SourceFingerprint {
                    size: CURRENT.size + 1,
                    ..CURRENT
                }
            )?,
            None
        );
        Ok(())
    }

    #[test]
    fn generator_change_invalidates_variants() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        put(&db, 256, 1, CURRENT)?;

        assert_eq!(db.get(42, 256, 2, CURRENT)?, None);
        Ok(())
    }

    #[test]
    fn sweeping_old_generators_can_be_scoped_to_catalog_assets() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        put(&db, 256, 1, CURRENT)?;
        db.put(7, 256, 1, CURRENT, 1, 1, ThumbnailEncoding::Png, &png()?)?;
        db.put(7, 256, 2, CURRENT, 1, 1, ThumbnailEncoding::Png, &png()?)?;

        assert_eq!(db.sweep_old_generators_for_assets(2, [7])?, 1);
        assert_eq!(db.get(7, 256, 1, CURRENT)?, None);
        assert!(db.get(7, 256, 2, CURRENT)?.is_some());
        assert!(db.get(42, 256, 1, CURRENT)?.is_some());
        Ok(())
    }

    #[test]
    fn deleting_asset_removes_every_variant() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        put(&db, 128, 1, CURRENT)?;
        put(&db, 256, 1, CURRENT)?;
        put(&db, 256, 2, CURRENT)?;
        db.put(7, 128, 1, CURRENT, 1, 1, ThumbnailEncoding::Png, &png()?)?;

        assert_eq!(db.delete_assets(&[42, 7])?, 4);
        assert_eq!(db.get(42, 128, 1, CURRENT)?, None);
        assert_eq!(db.get(42, 256, 2, CURRENT)?, None);
        assert_eq!(db.get(7, 128, 1, CURRENT)?, None);
        Ok(())
    }

    #[test]
    fn rejects_malformed_or_incorrectly_described_thumbnail_data() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        let data = png()?;

        let error = db
            .put(42, 128, 1, CURRENT, 2, 1, ThumbnailEncoding::Png, &data)
            .unwrap_err();
        assert!(error.to_string().contains("not the declared 2x1"));

        let error = db
            .put(42, 128, 1, CURRENT, 1, 1, ThumbnailEncoding::Jpeg, &data)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not the declared static image/jpeg")
        );

        let error = db
            .put(
                42,
                128,
                1,
                CURRENT,
                1,
                1,
                ThumbnailEncoding::Png,
                &data[..16],
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("decoding static thumbnail bytes")
        );
        Ok(())
    }

    #[test]
    fn batch_rejects_invalid_rows_without_committing_valid_predecessors() -> Result<()> {
        let temp = TempDir::new()?;
        let db = test_db(&temp)?;
        let data = png()?;
        let valid = DecodedThumbnail {
            asset_id: 42,
            size_bucket: 128,
            generator_version: 1,
            fingerprint: CURRENT,
            width: 1,
            height: 1,
            encoding: ThumbnailEncoding::Png,
            data: data.clone(),
        };
        let invalid = DecodedThumbnail {
            size_bucket: 256,
            width: 2,
            data,
            ..valid.clone()
        };
        assert!(db.store_batch(&[valid, invalid]).is_err());
        assert!(!db.has_current(42, 128, 1, CURRENT)?);
        Ok(())
    }

    #[test]
    fn decodes_owned_variants_then_persists_them_as_one_batch() -> Result<()> {
        let temp = TempDir::new()?;
        let source = PathBuf::try_from(temp.path().join("source.png"))?;
        std::fs::write(
            &source,
            imaging::encode_png(20, 10, &vec![0u8; 20 * 10 * 4])?,
        )?;
        let catalog = AssetCatalog::new(&PathBuf::try_from(temp.path().join("assets.db"))?)?;
        let asset = catalog.upsert(&source, &std::fs::metadata(&source)?)?;
        let db = test_db(&temp)?;

        let variants = decode_asset_variants(&asset, &[128, 256], GENERATOR_VERSION)?;
        assert_eq!(variants.len(), 2);
        db.store_batch(&variants)?;
        assert!(db.has_current(asset.asset_id, 128, GENERATOR_VERSION, asset.fingerprint)?);
        assert!(db.has_current(asset.asset_id, 256, GENERATOR_VERSION, asset.fingerprint)?);
        Ok(())
    }

    fn service_asset(temp: &TempDir, name: &str) -> Result<Asset> {
        let source = PathBuf::try_from(temp.path().join(name))?;
        std::fs::write(
            &source,
            imaging::encode_png(20, 10, &vec![0u8; 20 * 10 * 4])?,
        )?;
        let catalog = AssetCatalog::new(&PathBuf::try_from(temp.path().join("assets.db"))?)?;
        catalog.upsert(&source, &std::fs::metadata(&source)?)
    }

    fn service(temp: &TempDir) -> Result<ThumbnailService> {
        ThumbnailService::new(&PathBuf::try_from(temp.path().join("thumbnails.db"))?)
    }

    #[test]
    fn concurrent_identical_variants_share_one_owner() -> Result<()> {
        let temp = TempDir::new()?;
        let service = service(&temp)?;
        let asset = service_asset(&temp, "source.png")?;
        let barrier = Arc::new(Barrier::new(3));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let service = service.clone();
                let asset = asset.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    service.generate(&[asset], &[256], GENERATOR_VERSION, false, || false)
                })
            })
            .collect();
        barrier.wait();
        for worker in workers {
            assert!(
                worker.join().expect("thumbnail worker should not panic")[0]
                    .result
                    .is_ok()
            );
        }
        assert_eq!(service.decode_count(), 1);
        Ok(())
    }

    #[test]
    fn disjoint_variants_remain_independent_and_fingerprints_do_not_join() -> Result<()> {
        let temp = TempDir::new()?;
        let service = service(&temp)?;
        let first = service_asset(&temp, "first.png")?;
        let second = service_asset(&temp, "second.png")?;
        let barrier = Arc::new(Barrier::new(3));
        let workers: Vec<_> = [first.clone(), second]
            .into_iter()
            .map(|asset| {
                let service = service.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    service.generate(&[asset], &[256], GENERATOR_VERSION, false, || false)
                })
            })
            .collect();
        barrier.wait();
        for worker in workers {
            assert!(
                worker.join().expect("thumbnail worker should not panic")[0]
                    .result
                    .is_ok()
            );
        }
        let changed = Asset {
            fingerprint: SourceFingerprint {
                modified_ns: first.fingerprint.modified_ns + 1,
                ..first.fingerprint
            },
            ..first
        };
        assert!(
            service.generate(&[changed], &[256], GENERATOR_VERSION, false, || false)[0]
                .result
                .is_ok()
        );
        assert_eq!(service.decode_count(), 3);
        Ok(())
    }

    #[test]
    fn failed_flight_wakes_waiters_and_can_retry() -> Result<()> {
        let temp = TempDir::new()?;
        let service = service(&temp)?;
        let asset = service_asset(&temp, "source.png")?;
        let key = VariantKey {
            asset_id: asset.asset_id,
            size_bucket: 256,
            generator_version: GENERATOR_VERSION,
            fingerprint: asset.fingerprint,
        };
        let (owner, owner_claimed) = service.claim(key);
        let (waiter, waiter_claimed) = service.claim(key);
        assert!(owner_claimed);
        assert!(!waiter_claimed);
        service.fail_all(vec![key], vec![owner], anyhow::anyhow!("decode failed"));
        assert!(matches!(waiter.wait(), FlightResult::Failed(_)));
        let (_, retry_claimed) = service.claim(key);
        assert!(retry_claimed);
        Ok(())
    }
}
