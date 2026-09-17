use std::collections::HashMap;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak, mpsc};
use std::thread;
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use camino::Utf8Path as Path;
use rayon::prelude::*;
use tracing::{Span, debug_span};

use crate::assets::{Asset, MediaKind, SourceFingerprint};
use crate::imaging;
use crate::poster;

pub const SIZE_BUCKETS: [u16; 4] = [128, 256, 512, 1024];
pub const EAGER_SIZE_BUCKETS: [u16; 3] = [128, 256, 512];
pub const GENERATOR_VERSION: u32 = 3;

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

/// An owned thumbnail awaiting validation and durable persistence.
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

mod storage;
pub use storage::ThumbnailDb;

/// Decode all requested variants for one image in one source pass. The caller persists them.
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

/// Key used by single-flight deduplication and persistence.
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

struct ServiceInner {
    flights: Mutex<HashMap<VariantKey, Arc<Flight>>>,
    writer: mpsc::SyncSender<WriterCommand>,
    writer_thread: Mutex<Option<thread::JoinHandle<()>>>,
    pool: rayon::ThreadPool,
    #[cfg(test)]
    decode_count: AtomicUsize,
}

/// Coordinates a bounded decode pool, single-flight registry, and one database writer.
#[derive(Clone)]
pub struct ThumbnailService {
    inner: Arc<ServiceInner>,
}

/// One request for the writer thread, with a command-specific reply channel.
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
    Maintain {
        catalog: camino::Utf8PathBuf,
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
    variants: Vec<ClaimedVariant>,
}

/// Ownership travels with each variant through decode and persistence. Unwinding anywhere in
/// that pipeline must wake joiners and release the key for a later retry.
struct ClaimedVariant {
    key: VariantKey,
    flight: Arc<Flight>,
    service: Weak<ServiceInner>,
    current: bool,
    completed: bool,
}

impl ClaimedVariant {
    fn complete(mut self, result: FlightResult) {
        self.resolve(result);
    }

    fn resolve(&mut self, result: FlightResult) {
        if let Some(service) = self.service.upgrade() {
            let mut registry = service
                .flights
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            self.flight.finish(result);
            if registry
                .get(&self.key)
                .is_some_and(|registered| Arc::ptr_eq(registered, &self.flight))
            {
                registry.remove(&self.key);
            }
        } else {
            self.flight.finish(result);
        }
        self.completed = true;
    }
}

impl Drop for ClaimedVariant {
    fn drop(&mut self) {
        if !self.completed {
            self.resolve(FlightResult::Failed(
                "thumbnail generation was interrupted".into(),
            ));
        }
    }
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

    /// Durably ensure requested variants. Claiming before cache lookup prevents duplicate owners.
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
                        Self::fail_all(work.variants, Arc::clone(&error));
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

    /// Elect one owner for each requested `(asset, bucket)` pair before cache lookup.
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
            let mut variants = Vec::new();
            for &size_bucket in buckets {
                let key = VariantKey {
                    asset_id: asset.asset_id,
                    size_bucket,
                    generator_version,
                    fingerprint: asset.fingerprint,
                };
                let (flight, owner) = self.claim(key);
                flights.push(flight);
                if let Some(owner) = owner {
                    variants.push(owner);
                } else {
                    tracing::debug!(
                        asset_id = asset.asset_id,
                        size_bucket,
                        "thumbnail flight joined"
                    );
                }
            }
            if !variants.is_empty() {
                claimed.push(Claimed {
                    asset: asset.clone(),
                    variants,
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

    /// Ask the writer which claimed variants are already current, and record each result
    /// so [`Self::decode_claimed`] only decodes what actually changed. `force` skips the query.
    fn mark_current(&self, claimed: &mut [Claimed], force: bool) -> Result<(), Arc<str>> {
        // One flat request covers every owned key across the whole chunk in a single round trip
        // to the writer thread; both traversals preserve asset and bucket order.
        let keys: Vec<_> = claimed
            .iter()
            .flat_map(|work| work.variants.iter().map(|variant| variant.key))
            .collect();
        let current: Vec<bool> = if force {
            vec![false; keys.len()]
        } else {
            self.request(|reply| WriterCommand::Current { keys, reply })
                .context("checking thumbnail cache")
                .map_err(|error| Arc::<str>::from(format!("{error:#}")))?
        };
        for (variant, current) in claimed
            .iter_mut()
            .flat_map(|work| &mut work.variants)
            .zip(current)
        {
            variant.current = current;
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
                        .variants
                        .iter()
                        .filter_map(|variant| (!variant.current).then_some(variant.key.size_bucket))
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
                            work.variants[0].key.generator_version,
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
                Err(error) => Self::fail_all(work.variants, format!("{error:#}").into()),
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
            for variant in work.variants {
                let result = match (&stored, variant.current) {
                    (_, true) => FlightResult::Cached,
                    (Ok(()), false) => FlightResult::Stored,
                    (Err(error), false) => FlightResult::Failed(Arc::clone(error)),
                };
                variant.complete(result);
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

    /// Serialize orphan pruning and SQLite maintenance with all other thumbnail mutations.
    pub fn prune_orphans_and_maintain(&self, catalog: camino::Utf8PathBuf) -> Result<usize> {
        self.request(|reply| WriterCommand::Maintain { catalog, reply })
    }

    /// Join the flight for `key`, creating and registering it as the owner if none exists yet.
    /// A new flight returns an ownership guard; a joined flight has no guard.
    fn claim(&self, key: VariantKey) -> (Arc<Flight>, Option<ClaimedVariant>) {
        let mut registry = self
            .inner
            .flights
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(flight) = registry.get(&key) {
            return (Arc::clone(flight), None);
        }
        let flight = Arc::new(Flight::new());
        registry.insert(key, Arc::clone(&flight));
        let owner = ClaimedVariant {
            key,
            flight: Arc::clone(&flight),
            service: Arc::downgrade(&self.inner),
            current: false,
            completed: false,
        };
        drop(registry);
        tracing::debug!(
            asset_id = key.asset_id,
            size_bucket = key.size_bucket,
            "thumbnail flight owner"
        );
        (flight, Some(owner))
    }

    /// Complete every listed flight as `Failed` with the same shared error message.
    fn fail_all(variants: Vec<ClaimedVariant>, error: Arc<str>) {
        for variant in variants {
            variant.complete(FlightResult::Failed(Arc::clone(&error)));
        }
    }

    /// Send one command to the writer thread and block for its reply.
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

/// Own the writable connection and serialize database mutations until shutdown.
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
            WriterCommand::Maintain { catalog, reply } => {
                let _ = reply.send(respond(&mut database, &path, |database| {
                    let deleted = database.prune_orphans(&catalog)?;
                    database.maintain()?;
                    Ok(deleted)
                }));
            }
            WriterCommand::Shutdown { reply } => {
                let _ = reply.send(());
                break;
            }
        }
    }
}

/// Open or reuse the writer connection, retrying a failed open on each command.
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
        db.put(DecodedThumbnail {
            asset_id: 42,
            size_bucket: bucket,
            generator_version: version,
            fingerprint,
            width: 1,
            height: 1,
            encoding: ThumbnailEncoding::Png,
            data: data.to_vec(),
        })
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
        db.put(DecodedThumbnail {
            asset_id: 7,
            size_bucket: 256,
            generator_version: 1,
            fingerprint: CURRENT,
            width: 1,
            height: 1,
            encoding: ThumbnailEncoding::Png,
            data: png()?,
        })?;
        db.put(DecodedThumbnail {
            asset_id: 7,
            size_bucket: 256,
            generator_version: 2,
            fingerprint: CURRENT,
            width: 1,
            height: 1,
            encoding: ThumbnailEncoding::Png,
            data: png()?,
        })?;

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
        db.put(DecodedThumbnail {
            asset_id: 7,
            size_bucket: 128,
            generator_version: 1,
            fingerprint: CURRENT,
            width: 1,
            height: 1,
            encoding: ThumbnailEncoding::Png,
            data: png()?,
        })?;

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
            .put(DecodedThumbnail {
                asset_id: 42,
                size_bucket: 128,
                generator_version: 1,
                fingerprint: CURRENT,
                width: 2,
                height: 1,
                encoding: ThumbnailEncoding::Png,
                data: data.to_vec(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("not the declared 2x1"));

        let error = db
            .put(DecodedThumbnail {
                asset_id: 42,
                size_bucket: 128,
                generator_version: 1,
                fingerprint: CURRENT,
                width: 1,
                height: 1,
                encoding: ThumbnailEncoding::Jpeg,
                data: data.to_vec(),
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not the declared static image/jpeg")
        );

        let error = db
            .put(DecodedThumbnail {
                asset_id: 42,
                size_bucket: 128,
                generator_version: 1,
                fingerprint: CURRENT,
                width: 1,
                height: 1,
                encoding: ThumbnailEncoding::Png,
                data: data[..16].to_vec(),
            })
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
        let (_, owner_claimed) = service.claim(key);
        let (waiter, waiter_claimed) = service.claim(key);
        assert!(waiter_claimed.is_none());
        ThumbnailService::fail_all(vec![owner_claimed.unwrap()], "decode failed".into());
        assert!(matches!(waiter.wait(), FlightResult::Failed(_)));
        let (_, retry_claimed) = service.claim(key);
        assert!(retry_claimed.is_some());
        Ok(())
    }

    fn callback_panic_releases_owned_variants(panic_on_active: bool) -> Result<()> {
        let temp = TempDir::new()?;
        let service = service(&temp)?;
        let asset = service_asset(&temp, "source.png")?;
        let (entered, callback_entered) = mpsc::channel();
        let (release, callback_release) = mpsc::channel();
        let callback_release = Mutex::new(callback_release);
        let owner = {
            let service = service.clone();
            let asset = asset.clone();
            thread::spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    service.generate_observed(
                        &[asset],
                        &[128, 256],
                        GENERATOR_VERSION,
                        false,
                        || false,
                        |_, active| {
                            if active == panic_on_active {
                                entered.send(()).unwrap();
                                callback_release.lock().unwrap().recv().unwrap();
                                panic!("observer failed");
                            }
                        },
                    )
                }))
                .is_err()
            })
        };
        callback_entered.recv_timeout(Duration::from_secs(10))?;
        // Join while the owner is inside the callback, before allowing it to unwind.
        let (waiters, claimed) =
            service.claim_chunk(std::slice::from_ref(&asset), &[128, 256], GENERATOR_VERSION);
        assert!(claimed.is_empty());
        let (finished, results) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let failures: Vec<_> = waiters
                .into_iter()
                .flat_map(|waiter| waiter.flights)
                .map(|flight| matches!(flight.wait(), FlightResult::Failed(_)))
                .collect();
            let _ = finished.send(failures);
        });
        release.send(())?;
        assert!(owner.join().expect("panic was caught in owner"));
        assert_eq!(results.recv_timeout(Duration::from_secs(10))?, [true, true]);
        waiter.join().expect("waiter should not panic");
        assert!(service.inner.flights.lock().unwrap().is_empty());

        let retry = service.generate(
            std::slice::from_ref(&asset),
            &[128, 256],
            GENERATOR_VERSION,
            false,
            || false,
        );
        assert_eq!(retry[0].result.as_ref().unwrap().generated, 2);
        let cached = service.generate(&[asset], &[128, 256], GENERATOR_VERSION, false, || false);
        assert_eq!(cached[0].result.as_ref().unwrap().skipped, 2);
        Ok(())
    }

    #[test]
    fn callback_panic_before_decode_wakes_joiners_and_allows_retry() -> Result<()> {
        callback_panic_releases_owned_variants(true)
    }

    #[test]
    fn callback_panic_after_decode_wakes_joiners_and_allows_retry() -> Result<()> {
        callback_panic_releases_owned_variants(false)
    }
}
