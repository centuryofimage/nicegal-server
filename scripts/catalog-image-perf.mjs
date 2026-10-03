import assert from 'node:assert/strict'
import { createHash, randomBytes } from 'node:crypto'
import { mkdir, mkdtemp, readFile, realpath, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { parseArgs } from 'node:util'
import { createLibrary, spawnServer, readReadyMessage, sseSnapshots, stopChild, withTimeout } from './server-harness.mjs'

const { values } = parseArgs({ options: {
  before: { type: 'string' }, after: { type: 'string' },
  corpus: { type: 'string', default: join(homedir(), 'Pictures') },
  limit: { type: 'string', default: '2000' }, rounds: { type: 'string', default: '2' },
  output: { type: 'string', default: 'target/catalog-image-perf' },
  model: { type: 'string', default: 'facebook/metaclip-2-worldwide-b32' },
  provider: { type: 'string', default: 'directml' },
  'images-only': { type: 'boolean', default: false },
  'trace-stages': { type: 'boolean', default: false },
  help: { type: 'boolean', default: false }
} })
if (values.help) {
  console.log('Usage: node scripts/catalog-image-perf.mjs --before EXE --after EXE [--corpus DIR] [--limit 2000] [--rounds 2] [--model ID] [--provider directml|webgpu|cpu] [--images-only] [--trace-stages] [--output DIR]\nUses fresh databases and verifies the selected provider. Retains results and logs in a unique output directory.')
  process.exit(0)
}
assert.ok(values.before && values.after, '--before and --after executables are required')
const limit = positiveInteger(values.limit)
const rounds = positiveInteger(values.rounds)
const corpus = await realpath(values.corpus)
const executables = { before: await realpath(values.before), after: await realpath(values.after) }
await mkdir(resolve(values.output), { recursive: true })
const output = await mkdtemp(join(resolve(values.output), 'run-'))
const model = values.model
const provider = values.provider
assert.ok(['directml', 'webgpu', 'cpu'].includes(provider), 'unsupported benchmark provider')
const report = { corpus, limit, model, imagesOnly: values['images-only'], provider, executables, runs: [] }
console.log(`Results: ${output}`)
for (let round = 0; round < rounds; round += 1) {
  for (const variant of round % 2 ? ['after', 'before'] : ['before', 'after']) {
    const stateDirectory = join(output, `${round + 1}-${variant}`)
    await mkdir(stateDirectory)
    const run = await benchmark(variant, round + 1, stateDirectory)
    report.runs.push(run)
    await writeFile(join(output, 'results.json'), JSON.stringify(report, null, 2) + '\n')
    assert.equal(run.sourceHash, report.runs[0].sourceHash, 'versions must scan identical source paths and revisions')
    console.log(JSON.stringify({ variant, round: round + 1, sources: run.sourceCount,
      catalogMs: run.catalog.phaseMs.cataloging, modelLoadMs: run.image.phaseMs.loadingModels,
      imageScanMs: run.image.phaseMs.imageEmbedding, embedded: run.image.progress.embedded,
      failed: run.image.progress.failed, maxCatalogGapMs: run.catalog.maxCatalogGapMs }))
  }
}

function positiveInteger(value) {
  const parsed = Number(value)
  assert.ok(Number.isSafeInteger(parsed) && parsed > 0, `expected a positive integer: ${value}`)
  return parsed
}

async function benchmark(variant, round, stateDirectory) {
  const token = randomBytes(32).toString('hex')
  const child = spawnServer({ executable: executables[variant], repository: process.cwd(), stateDirectory, token,
    env: { NICEGAL_EXECUTION_PROVIDER: provider, NICEGAL_IMAGE_MODEL: model,
      RUST_LOG: values['trace-stages']
        ? 'warn,nicegal_core::index=info,nicegal_core::image_index=debug,nicegal_core::embedding=debug,nicegal_core::imaging=debug,nicegal_core::runtime=info,nom_exif=off'
        : 'warn,nicegal_core::index=info,nicegal_core::image_index=info,nicegal_core::embedding=info,nicegal_core::runtime=info,nom_exif=off' } })
  let stderr = ''
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-32000) })
  try {
    const { endpoint } = await withTimeout(readReadyMessage(child, () => stderr), 60000, 'server readiness')
    const libraryId = await createLibrary(endpoint, token, { include: [corpus], ocr: false, image: false,
      videos: !values['images-only'] })
    // A library with no search indexes only catalogs, so this measures cataloging alone.
    const catalog = await measureJob(endpoint, token, 'libraryScan',
      { libraryId, debugLimit: values['images-only'] ? undefined : limit }, `${variant}/${round}`)
    const db = new DatabaseSync(join(stateDirectory, 'assets.db'), { readOnly: true })
    let sources
    try {
      const statement = db.prepare('SELECT path, source_modified_ns, source_size FROM assets ORDER BY path')
      statement.setReadBigInts(true)
      sources = statement.all()
    } finally { db.close() }
    const sourceHash = createHash('sha256').update(JSON.stringify(sources, (_, value) => typeof value === 'bigint' ? value.toString() : value)).digest('hex')
    // Image selection is capped independently from cataloging. Image-only runs catalog the
    // root so videos encountered during walking cannot reduce the requested image sample.
    const image = await measureJob(endpoint, token, 'imageEmbed', { libraryId, debugLimit: limit }, `${variant}/${round}`)
    if (values['images-only']) assert.equal(image.progress.embedded, limit, 'expected the requested number of image assets')
    await stopChild(child)
    const trace = (await readFile(join(stateDirectory, 'nicegal-server.log'), 'utf8')).trim().split('\n').map(line => JSON.parse(line))
    const catalogTrace = trace.find(event => event.span?.name === 'catalog' && event.fields?.message === 'close')
    const imageTrace = trace.find(event => event.span?.name === 'image_index' && event.fields?.message === 'close')
    const providers = [...new Set(trace.filter(event => event.fields?.message === 'loaded the image embedding model').map(event => event.fields.execution_provider))]
    assert.deepEqual(providers, [provider], 'benchmark must actually run image inference on the requested provider')
    return { variant, round, sourceCount: sources.length, sourceHash, providers, catalog, image, catalogTrace, imageTrace,
      stageTimings: summarizeStages(trace),
      loadedModel: trace.filter(event => event.fields?.message === 'loaded the image embedding model')
        .map(event => ({ batchSize: event.fields.max_batch, provider: event.fields.execution_provider,
          dimensions: event.fields.dimensions })) }
  } catch (error) {
    await writeFile(join(stateDirectory, 'failure.txt'), `${error.stack}\n${stderr}`)
    throw error
  } finally { await stopChild(child) }
}

function durationMs(text) {
  const match = /^([\d.]+)(ns|µs|us|ms|s)$/.exec(text ?? '')
  return match ? Number(match[1]) * { ns: 1e-6, 'µs': 1e-3, us: 1e-3, ms: 1, s: 1000 }[match[2]] : 0
}

function summarizeStages(trace) {
  const stages = {}
  for (const event of trace) {
    if (event.fields?.message !== 'close' || !event.span?.name) continue
    const name = event.span.name
    if (!['decode_image', 'preprocess_image', 'embed_preprocessed_images', 'save_image_embeddings',
      'image_index', 'video_preprocess'].includes(name)) continue
    const item = stages[name] ??= { count: 0, totalMs: 0, maxMs: 0, samplesMs: [], batchCounts: {} }
    const elapsed = durationMs(event.fields['time.busy']) + durationMs(event.fields['time.idle'])
    item.count += 1
    item.totalMs += elapsed
    item.maxMs = Math.max(item.maxMs, elapsed)
    item.samplesMs.push(elapsed)
    if (event.span.batch !== undefined) {
      item.batchCounts[event.span.batch] = (item.batchCounts[event.span.batch] ?? 0) + 1
    }
  }
  for (const item of Object.values(stages)) {
    item.samplesMs.sort((a, b) => a - b)
    item.medianMs = item.samplesMs[Math.floor(item.count / 2)]
    item.p95Ms = item.samplesMs[Math.min(item.count - 1, Math.floor(item.count * .95))]
    delete item.samplesMs
  }
  return stages
}

async function measureJob(endpoint, token, type, params, label) {
  const started = performance.now()
  const signal = AbortSignal.timeout(30 * 60 * 1000)
  const headers = { authorization: `Bearer ${token}`, 'content-type': 'application/json' }
  const created = await fetch(endpoint + '/v1/jobs', { method: 'POST', headers, body: JSON.stringify({ type, params }), signal })
  assert.equal(created.status, 202, await created.clone().text())
  let job = await created.json()
  const response = await fetch(endpoint + `/v1/jobs/${job.jobId}/events`, { headers, signal })
  assert.equal(response.status, 200)
  const phases = {}
  let phase, phaseStarted = started, lastPrint = started, previousCatalog = 0, lastCatalogTime
  const catalogDeltas = [], catalogGaps = []
  for await (const snapshot of sseSnapshots(response)) {
    job = snapshot
    const now = performance.now()
    if (phase !== job.phase) {
      if (phase) phases[phase] = (phases[phase] ?? 0) + now - phaseStarted
      phase = job.phase
      phaseStarted = now
      console.log(`${label} ${type}: ${phase}`)
    }
    if (job.phase === 'cataloging' && job.progress.cataloged > previousCatalog) {
      catalogDeltas.push(job.progress.cataloged - previousCatalog)
      if (lastCatalogTime !== undefined) catalogGaps.push(now - lastCatalogTime)
      previousCatalog = job.progress.cataloged
      lastCatalogTime = now
    }
    if (now - lastPrint > 15000) {
      console.log(`${label} ${type}: ${job.progress.phaseCompleted}/${job.progress.total}`)
      lastPrint = now
    }
  }
  assert.equal(job.status, 'completed', JSON.stringify(job))
  return { elapsedMs: performance.now() - started, phaseMs: phases, progress: job.progress, errors: job.errors,
    catalogProgressEvents: catalogDeltas.length, catalogDeltas: [...new Set(catalogDeltas)], maxCatalogGapMs: Math.max(0, ...catalogGaps) }
}
