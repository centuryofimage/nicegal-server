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
  help: { type: 'boolean', default: false }
} })
if (values.help) {
  console.log('Usage: node scripts/catalog-image-perf.mjs --before EXE --after EXE [--corpus DIR] [--limit 2000] [--rounds 2] [--output DIR]\nUses fresh databases, DirectML, and MetaCLIP2 B/32. Retains results and logs in a unique output directory.')
  process.exit(0)
}
assert.ok(values.before && values.after, '--before and --after executables are required')
const limit = positiveInteger(values.limit)
const rounds = positiveInteger(values.rounds)
const corpus = await realpath(values.corpus)
const executables = { before: await realpath(values.before), after: await realpath(values.after) }
await mkdir(resolve(values.output), { recursive: true })
const output = await mkdtemp(join(resolve(values.output), 'run-'))
const model = 'facebook/metaclip-2-worldwide-b32'
const report = { corpus, limit, model, provider: 'directml', executables, runs: [] }
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
    env: { NICEGAL_EXECUTION_PROVIDER: 'directml', NICEGAL_IMAGE_MODEL: model,
      RUST_LOG: 'warn,nicegal_core::index=info,nicegal_core::image_index=info,nicegal_core::embedding=info,nicegal_core::runtime=info,nom_exif=off' } })
  let stderr = ''
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-32000) })
  try {
    const { endpoint } = await withTimeout(readReadyMessage(child, () => stderr), 60000, 'server readiness')
    const libraryId = await createLibrary(endpoint, token, { include: [corpus], ocr: false, image: false })
    // A library with no search indexes only catalogs, so this measures cataloging alone.
    const catalog = await measureJob(endpoint, token, 'libraryScan', { libraryId, debugLimit: limit }, `${variant}/${round}`)
    const db = new DatabaseSync(join(stateDirectory, 'assets.db'), { readOnly: true })
    let sources
    try {
      const statement = db.prepare('SELECT path, source_modified_ns, source_size FROM assets ORDER BY path')
      statement.setReadBigInts(true)
      sources = statement.all()
    } finally { db.close() }
    const sourceHash = createHash('sha256').update(JSON.stringify(sources, (_, value) => typeof value === 'bigint' ? value.toString() : value)).digest('hex')
    const image = await measureJob(endpoint, token, 'imageEmbed', { libraryId, debugLimit: limit }, `${variant}/${round}`)
    await stopChild(child)
    const trace = (await readFile(join(stateDirectory, 'nicegal-server.log'), 'utf8')).trim().split('\n').map(line => JSON.parse(line))
    const catalogTrace = trace.find(event => event.span?.name === 'catalog' && event.fields?.message === 'close')
    const imageTrace = trace.find(event => event.span?.name === 'image_index' && event.fields?.message === 'close')
    const providers = [...new Set(trace.filter(event => event.fields?.message === 'loaded the image embedding model').map(event => event.fields.execution_provider))]
    assert.deepEqual(providers, ['directml'], 'benchmark must actually run image inference on DirectML')
    return { variant, round, sourceCount: sources.length, sourceHash, providers, catalog, image, catalogTrace, imageTrace }
  } catch (error) {
    await writeFile(join(stateDirectory, 'failure.txt'), `${error.stack}\n${stderr}`)
    throw error
  } finally { await stopChild(child) }
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
