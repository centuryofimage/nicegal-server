import assert from 'node:assert/strict'
import { randomBytes } from 'node:crypto'
import { once } from 'node:events'
import { copyFile, mkdir, mkdtemp, realpath, rm, unlink } from 'node:fs/promises'
import { DatabaseSync } from 'node:sqlite'
import { createInterface } from 'node:readline'
import { spawn } from 'node:child_process'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repository = join(scriptDirectory, '..')
const executableName = process.platform === 'win32' ? 'nicegal-server.exe' : 'nicegal-server'
const executable = process.argv[2] ?? join(repository, 'target', 'debug', executableName)
const sourcePath = await realpath(
  process.argv[3] ?? join(repository, '..', 'testdata', 'pink', 'benice.gif')
)
const textSourcePath = await realpath(
  join(repository, '..', 'testdata', 'pink', 'SmartSelect_20201224-134223_Firefox Beta.jpg')
)
const temporaryDirectory = await mkdtemp(join(tmpdir(), `nicegal-server-rpc-smoke-${process.pid}-`))
const assetDatabasePath = join(temporaryDirectory, 'assets.db')
const thumbnailDatabasePath = join(temporaryDirectory, 'thumbnails.db')
const ocrDatabasePath = join(temporaryDirectory, 'index.db')
const indexRoot = join(temporaryDirectory, 'images')
const textIndexRoot = join(temporaryDirectory, 'ocr-text')
const indexedSourcePath = join(indexRoot, 'benice.gif')
const indexedTextPath = join(textIndexRoot, 'text.jpg')
const token = randomBytes(32).toString('hex')
// Static 1x1 PNG poster. Never copy the original animated GIF into the thumbnail store.
const thumbnailBytes = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACdFJOUwAAdpPNOAAAAAJiS0dEAAHdihOkAAAAB3RJTUUH6ggRBDo1181b+QAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyNi0wOC0xN1QwNDo1ODo1MyswMDowMNaigBgAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjYtMDgtMTdUMDQ6NTg6NTMrMDA6MDCn/zikAAAAKHRFWHRkYXRlOnRpbWVzdGFtcAAyMDI2LTA4LTE3VDA0OjU4OjUzKzAwOjAw8OoZewAAAApJREFUCNdjYAAAAAIAAeIhvDMAAAAASUVORK5CYII=',
  'base64'
)
await mkdir(indexRoot)
await mkdir(textIndexRoot)
await copyFile(sourcePath, indexedSourcePath)
await copyFile(textSourcePath, indexedTextPath)

const child = spawn(
  executable,
  [
    '--asset-database',
    assetDatabasePath,
    '--ocr-database',
    ocrDatabasePath,
    '--thumbnail-database',
    thumbnailDatabasePath
  ],
  {
    cwd: repository,
    env: {
      ...process.env,
      NICEGAL_RPC_TOKEN: token
    },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true
  }
)
let stderr = ''
child.stderr.setEncoding('utf8')
child.stderr.on('data', (chunk) => {
  stderr += chunk
})

try {
  const ready = await withTimeout(readReadyMessage(child), 10_000, 'server readiness')
  assert.equal(ready.apiVersion, 1)
  assert.match(ready.endpoint, /^http:\/\/127\.0\.0\.1:\d+$/)

  const unauthorized = await fetch(`${ready.endpoint}/v1/health`)
  assert.equal(unauthorized.status, 401)

  const health = await fetch(`${ready.endpoint}/v1/health`, {
    headers: { authorization: `Bearer ${token}` }
  })
  assert.equal(health.status, 200)
  assert.deepEqual(await health.json(), { apiVersion: 1 })

  const unloadedIndex = await fetch(`${ready.endpoint}/v1/jobs`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      type: 'ocrIndex',
      params: { root: indexRoot }
    })
  })
  await assertStatus(unloadedIndex, 409)
  assert.equal((await unloadedIndex.json()).error.code, 'ocr_models_not_loaded')

  const modelRequest = {
    detection: { modelId: 'PaddlePaddle/PP-OCRv6_small_det_onnx' },
    recognition: { modelId: 'PaddlePaddle/PP-OCRv6_small_rec_onnx' }
  }
  const modelLoaded = await runTypedJob(ready.endpoint, token, 'ocrModelLoad', modelRequest)
  assert.equal(modelLoaded.type, 'ocrModelLoad')
  assert.equal(modelLoaded.status, 'completed')
  assert.equal(modelLoaded.phase, 'finished')
  assert.equal(modelLoaded.progress.processed, 2)
  assert.equal(modelLoaded.progress.modelsLoaded, 2)
  assert.ok(modelLoaded.progress.downloadedBytes <= modelLoaded.progress.downloadTotalBytes)
  assert.deepEqual(modelLoaded.errors, [])

  const cached = await runTypedJob(ready.endpoint, token, 'ocrModelLoad', modelRequest)
  assert.equal(cached.status, 'completed')
  assert.equal(cached.progress.processed, 2)
  assert.equal(cached.progress.downloadedBytes, 0)
  assert.equal(cached.progress.downloadTotalBytes, 0)
  assert.equal(cached.progress.modelsLoaded, 2)
  assert.deepEqual(cached.errors, [])

  const modelStatus = await fetch(`${ready.endpoint}/v1/ocr/models`, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(modelStatus, 200)
  assert.deepEqual(await modelStatus.json(), {
    loaded: {
      detection: {
        modelId: modelRequest.detection.modelId,
        filename: 'inference.onnx'
      },
      recognition: {
        modelId: modelRequest.recognition.modelId,
        filename: 'inference.onnx'
      },
      executionProvider: 'cpu'
    }
  })

  const recognized = await runTypedJob(ready.endpoint, token, 'ocrIndex', {
    root: textIndexRoot
  })
  assert.equal(recognized.type, 'ocrIndex')
  assert.equal(recognized.status, 'completed')
  assert.equal(recognized.phase, 'finished')
  assert.equal(recognized.progress.discovered, 1)
  assert.equal(recognized.progress.total, 1)
  assert.equal(recognized.progress.cataloged, 1)
  assert.equal(recognized.progress.processed, 1)
  assert.equal(recognized.progress.indexed, 1)
  assert.equal(recognized.progress.failed, 0)
  assert.deepEqual(recognized.errors, [])

  const indexed = await runTypedJob(ready.endpoint, token, 'ocrIndex', {
    root: indexRoot
  })
  assert.equal(indexed.type, 'ocrIndex')
  assert.equal(indexed.status, 'completed')
  assert.equal(indexed.progress.discovered, 1)
  assert.equal(indexed.progress.cataloged, 1)
  assert.equal(indexed.progress.processed, 1)
  assert.equal(indexed.progress.indexed, 1)
  assert.equal(indexed.progress.failed, 0)
  assert.deepEqual(indexed.errors, [])

  // The remainder of this smoke test exercises deterministic search text. The row itself must
  // come from the real scan/catalog/decode/detect/recognize pipeline before its content is fixed.
  const ocr = new DatabaseSync(ocrDatabasePath)
  const recognizedRow = ocr.prepare(
    'SELECT asset_id, width, height, content FROM ocr_results WHERE source_path = ?'
  ).get(indexedTextPath)
  assert.ok(recognizedRow)
  assert.ok(recognizedRow.width > 0)
  assert.ok(recognizedRow.height > 0)
  assert.equal(typeof recognizedRow.content, 'string')
  assert.ok(
    recognizedRow.content.trim().length > 0,
    'PaddleOCR should recognize text in the smoke fixture'
  )
  const indexedRow = ocr.prepare(
    'SELECT asset_id FROM ocr_results WHERE source_path = ?'
  ).get(indexedSourcePath)
  assert.ok(indexedRow)
  ocr.prepare('UPDATE ocr_results SET content = ? WHERE asset_id = ?').run(
    'pink image',
    indexedRow.asset_id
  )
  ocr.close()

  const jobs = await fetch(`${ready.endpoint}/v1/jobs`, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(jobs, 200)
  const jobList = await jobs.json()
  assert.equal(jobList.activeJobId, null)
  assert.equal(jobList.jobs.length, 4)
  assert.equal(jobList.jobs[0].jobId, indexed.jobId)

  const cancelledCompletedJob = await fetch(
    `${ready.endpoint}/v1/jobs/${cached.jobId}`,
    {
      method: 'DELETE',
      headers: { authorization: `Bearer ${token}` }
    }
  )
  await assertStatus(cancelledCompletedJob, 200)
  assert.equal((await cancelledCompletedJob.json()).status, 'completed')

  const missingJob = await fetch(`${ready.endpoint}/v1/jobs/999999`, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(missingJob, 404)
  assert.equal((await missingJob.json()).error.code, 'job_not_found')
  const assetUrl = new URL('/v1/assets', ready.endpoint)
  assetUrl.searchParams.set('path', indexedSourcePath)
  const assetResponse = await fetch(assetUrl, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(assetResponse, 200)
  const asset = await assetResponse.json()
  assert.equal(asset.path.replaceAll('\\', '/'), indexedSourcePath.replaceAll('\\', '/'))
  assert.equal(asset.mediaKind, 'image')
  assert.equal(asset.mediaFormat, 'gif')
  assert.equal(asset.isAnimated, true)
  const sourceModifiedNs = BigInt(asset.sourceModifiedNs)
  assert.ok(sourceModifiedNs > 0n)
  assert.ok(asset.sourceSize > 0)
  const assetId = asset.assetId
  assert.ok(Number.isSafeInteger(assetId) && assetId > 0)

  const ensuredResponse = await fetch(`${ready.endpoint}/v1/thumbnails`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/json'
    },
    body: JSON.stringify({ assetIds: [assetId], requiredSize: 100 })
  })
  await assertStatus(ensuredResponse, 200)
  assert.deepEqual(await ensuredResponse.json(), {
    assetIds: [assetId],
    requiredSize: 100,
    sizeBucket: 128,
    generatorVersion: 1
  })
  const ensuredLookup = new URL('/v1/thumbnails', ready.endpoint)
  ensuredLookup.searchParams.set('assetId', assetId.toString())
  ensuredLookup.searchParams.set('requestedSize', '100')
  ensuredLookup.searchParams.set('generatorVersion', '1')
  const ensuredThumbnail = await fetch(ensuredLookup, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(ensuredThumbnail, 200)
  assert.equal(ensuredThumbnail.headers.get('x-nicegal-server-size-bucket'), '128')

  const searchUrl = new URL('/v1/search', ready.endpoint)
  searchUrl.searchParams.set('q', '*')
  searchUrl.searchParams.set('type', 'glob')
  searchUrl.searchParams.set('root', indexRoot)
  searchUrl.searchParams.set('limit', '100')
  const searchResponse = await fetch(searchUrl, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(searchResponse, 200)
  const search = await searchResponse.json()
  assert.equal(search.total, 1)
  assert.deepEqual(search.results.map((result) => result.assetId), [assetId])

  // Vector search: coverage is visible before anything is embedded, the backfill fills it, and
  // the default search mode then answers from the stored vectors.
  const coverageUrl = new URL('/v1/text-embeddings', ready.endpoint)
  coverageUrl.searchParams.set('root', indexRoot)
  const beforeResponse = await fetch(coverageUrl, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(beforeResponse, 200)
  const before = await beforeResponse.json()
  assert.equal(before.stored, null, 'nothing is embedded before the first backfill')
  assert.equal(before.indexed, 1)
  assert.equal(before.pending, 1)
  assert.ok(before.embedder.dimensions > 0, JSON.stringify(before))

  // The default mode is vector, and an unembedded library is empty rather than an error.
  const unembedded = await fetch(searchRequestUrl(ready.endpoint, { q: 'pink', root: indexRoot }), {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(unembedded, 200)
  assert.equal((await unembedded.json()).total, 0)

  const embedCreated = await fetch(`${ready.endpoint}/v1/text-embeddings/generate`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify({ root: indexRoot })
  })
  await assertStatus(embedCreated, 202)
  const embedJob = await waitForJob(ready.endpoint, token, await embedCreated.json())
  assert.equal(embedJob.type, 'embed')
  assert.equal(embedJob.status, 'completed')
  assert.equal(embedJob.progress.embedded, 1)
  assert.deepEqual(embedJob.errors, [])

  const afterResponse = await fetch(coverageUrl, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(afterResponse, 200)
  const after = await afterResponse.json()
  assert.equal(after.embedded, 1)
  assert.equal(after.pending, 0)
  assert.equal(after.stored.model, after.embedder.model)

  const vectorResponse = await fetch(searchRequestUrl(ready.endpoint, { q: 'pink', root: indexRoot }), {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(vectorResponse, 200)
  const vector = await vectorResponse.json()
  assert.equal(vector.total, 1)
  assert.deepEqual(vector.results.map((result) => result.assetId), [assetId])
  assert.equal(vector.results[0].rank, 1)
  assert.equal(typeof vector.results[0].distance, 'number')

  // Combined search: every mode answers from one snapshot, and the fused list names its sources.
  const combinedResponse = await fetch(`${ready.endpoint}/v1/search`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      root: indexRoot,
      queries: [
        { key: 'semantic', type: 'vector', q: 'pink' },
        { key: 'files', type: 'glob', q: '*' }
      ],
      fuse: { method: 'rrf' }
    })
  })
  await assertStatus(combinedResponse, 200)
  const combined = await combinedResponse.json()
  assert.deepEqual(combined.queries.map((query) => query.key), ['semantic', 'files'])
  assert.equal(combined.queries[0].total, 1)
  assert.equal(combined.queries[1].total, 1)
  assert.equal(combined.fused.total, 1)
  assert.deepEqual(combined.fused.results[0].sources, ['semantic', 'files'])
  assert.equal(combined.fused.results[0].assetId, assetId)

  // A query SQLite cannot parse fails the whole combined request rather than half-ranking it.
  const combinedSyntax = await fetch(`${ready.endpoint}/v1/search`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      root: indexRoot,
      queries: [{ type: 'match', q: 'AND' }]
    })
  })
  await assertStatus(combinedSyntax, 400)
  assert.equal((await combinedSyntax.json()).error.code, 'query_syntax')

  // Time filtering applies to every mode, and the timeline is never chosen for the caller.
  const modifiedNs = asset.sourceModifiedNs
  const justBefore = (BigInt(modifiedNs) - 1n).toString()
  const justAfter = (BigInt(modifiedNs) + 1n).toString()

  // `glob` with `*` matches whatever text this image happens to contain, and `vector` ranks
  // neighbours regardless of it, so neither depends on knowing the OCR output. The full mode
  // matrix, `simple` and `match` included, is covered by the db unit tests against fixed text.
  for (const [type, q] of [['vector', 'pink'], ['glob', '*']]) {
    const inside = await fetch(
      searchRequestUrl(ready.endpoint, {
        q, type, root: indexRoot, timeline: 'modified', after: modifiedNs, before: justAfter
      }),
      { headers: { authorization: `Bearer ${token}` } }
    )
    await assertStatus(inside, 200)
    assert.equal((await inside.json()).total, 1, `${type} lost the row inside its own range`)

    const outside = await fetch(
      searchRequestUrl(ready.endpoint, {
        q, type, root: indexRoot, timeline: 'modified', after: justAfter
      }),
      { headers: { authorization: `Bearer ${token}` } }
    )
    await assertStatus(outside, 200)
    assert.equal((await outside.json()).total, 0, `${type} ignored the range`)
  }

  // `after` is inclusive and `before` is exclusive, so adjacent ranges tile exactly once.
  const atUpperBound = await fetch(
    searchRequestUrl(ready.endpoint, {
      q: '*', type: 'glob', root: indexRoot, timeline: 'modified', before: modifiedNs
    }),
    { headers: { authorization: `Bearer ${token}` } }
  )
  await assertStatus(atUpperBound, 200)
  assert.equal((await atUpperBound.json()).total, 0, 'before is exclusive')

  const atLowerBound = await fetch(
    searchRequestUrl(ready.endpoint, {
      q: '*', type: 'glob', root: indexRoot, timeline: 'modified', after: justBefore
    }),
    { headers: { authorization: `Bearer ${token}` } }
  )
  await assertStatus(atLowerBound, 200)
  assert.equal((await atLowerBound.json()).total, 1, 'after is inclusive')

  // A bound without a timeline is refused rather than guessed at.
  const missingTimeline = await searchError(ready.endpoint, token, {
    q: 'pink',
    type: 'simple',
    root: indexRoot,
    after: '0'
  })
  assert.equal(missingTimeline.status, 400)
  assert.equal(missingTimeline.body.error.code, 'invalid_request')
  assert.match(missingTimeline.body.error.message, /timeline is required/)

  const reversedRange = await searchError(ready.endpoint, token, {
    q: 'pink',
    type: 'simple',
    root: indexRoot,
    timeline: 'capture',
    after: '9',
    before: '2'
  })
  assert.equal(reversedRange.status, 400)
  assert.equal(reversedRange.body.error.code, 'invalid_request')

  // The combined form takes a request-level range that each query can replace outright.
  const combinedRange = await fetch(`${ready.endpoint}/v1/search`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      root: indexRoot,
      timeline: 'modified',
      after: justAfter,
      queries: [
        { key: 'inherits', type: 'glob', q: '*' },
        { key: 'overrides', type: 'glob', q: '*', timeline: 'modified', after: modifiedNs }
      ]
    })
  })
  await assertStatus(combinedRange, 200)
  const ranged = await combinedRange.json()
  assert.equal(ranged.queries[0].total, 0)
  assert.equal(ranged.queries[1].total, 1)

  // Error contract: a failure the user can fix must be a 4xx carrying a displayable cause,
  // never the generic 500 the renderer used to have to guess at.
  const syntaxError = await searchError(ready.endpoint, token, {
    q: 'ocr:"unterminated',
    type: 'simple',
    root: indexRoot
  })
  assert.equal(syntaxError.status, 400)
  assert.equal(syntaxError.body.error.code, 'query_syntax')
  assert.match(syntaxError.body.error.message, /unterminated string/)

  const ftsError = await searchError(ready.endpoint, token, {
    q: 'AND',
    type: 'match',
    root: indexRoot
  })
  assert.equal(ftsError.status, 400)
  assert.equal(ftsError.body.error.code, 'query_syntax')
  assert.match(ftsError.body.error.message, /fts5: syntax error/)

  const missingRoot = await searchError(ready.endpoint, token, {
    q: 'pink',
    root: join(indexRoot, 'definitely-missing')
  })
  assert.equal(missingRoot.status, 400)
  assert.equal(missingRoot.body.error.code, 'invalid_root')

  const relativeRoot = await searchError(ready.endpoint, token, { q: 'pink', root: 'relative' })
  assert.equal(relativeRoot.status, 400)
  assert.equal(relativeRoot.body.error.code, 'invalid_root')

  const missingParameter = await searchError(ready.endpoint, token, { q: 'pink' })
  assert.equal(missingParameter.status, 400)
  assert.equal(missingParameter.body.error.code, 'invalid_request')

  // An indexed root with no matches is an empty result set, not an error. This is a property of
  // the text modes: vector search ranks neighbours by distance and has no notion of "no match", so
  // it answers with the nearest rows until `maxDistance` says otherwise.
  const noMatches = await fetch(
    searchRequestUrl(ready.endpoint, { q: 'zzzznotpresent', type: 'simple', root: indexRoot }),
    { headers: { authorization: `Bearer ${token}` } }
  )
  await assertStatus(noMatches, 200)
  assert.equal((await noMatches.json()).total, 0)

  const looseNeighbours = await fetch(
    searchRequestUrl(ready.endpoint, { q: 'zzzznotpresent', root: indexRoot }),
    { headers: { authorization: `Bearer ${token}` } }
  )
  await assertStatus(looseNeighbours, 200)
  assert.equal((await looseNeighbours.json()).total, 1, 'vector search returns neighbours, not matches')

  // A distance ceiling is how a caller asks for "close enough" rather than "closest".
  const tightNeighbours = await fetch(
    searchRequestUrl(ready.endpoint, { q: 'pink', root: indexRoot, maxDistance: '0' }),
    { headers: { authorization: `Bearer ${token}` } }
  )
  await assertStatus(tightNeighbours, 200)
  assert.ok((await tightNeighbours.json()).total <= 1)

  const distanceOnText = await searchError(ready.endpoint, token, {
    q: 'pink',
    type: 'glob',
    root: indexRoot,
    maxDistance: '0.5'
  })
  assert.equal(distanceOnText.status, 400)
  assert.equal(distanceOnText.body.error.code, 'invalid_request')

  // Unmatched routes, wrong methods, and malformed bodies use the same envelope.
  const unknownRoute = await fetch(`${ready.endpoint}/v1/does-not-exist`, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(unknownRoute, 404)
  assert.equal((await unknownRoute.json()).error.code, 'not_found')

  const wrongMethod = await fetch(`${ready.endpoint}/v1/health`, {
    method: 'DELETE',
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(wrongMethod, 405)
  assert.equal((await wrongMethod.json()).error.code, 'method_not_allowed')

  const malformedBody = await fetch(`${ready.endpoint}/v1/jobs`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: '{not json'
  })
  await assertStatus(malformedBody, 400)
  assert.equal((await malformedBody.json()).error.code, 'invalid_request')

  const backfillCreated = await fetch(`${ready.endpoint}/v1/thumbnails/generate`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      root: indexRoot,
      timeline: 'modified',
      range: {
        fromNs: sourceModifiedNs.toString(),
        toNs: (sourceModifiedNs + 1n).toString()
      }
    })
  })
  await assertStatus(backfillCreated, 202)
  const backfill = await waitForJob(ready.endpoint, token, await backfillCreated.json())
  assert.equal(backfill.type, 'thumbnailGenerate')
  assert.equal(backfill.status, 'completed')
  assert.equal(backfill.progress.thumbnailsGenerated, 1)
  assert.deepEqual(backfill.errors, [])

  const thumbnailUrl = new URL('/v1/thumbnails', ready.endpoint)
  thumbnailUrl.searchParams.set('assetId', assetId.toString())
  thumbnailUrl.searchParams.set('sizeBucket', '128')
  thumbnailUrl.searchParams.set('generatorVersion', '1')
  thumbnailUrl.searchParams.set('width', '1')
  thumbnailUrl.searchParams.set('height', '1')
  thumbnailUrl.searchParams.set('encoding', 'image/png')

  const malformedThumbnail = await fetch(thumbnailUrl, {
    method: 'PUT',
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/octet-stream'
    },
    body: thumbnailBytes.subarray(0, 16)
  })
  await assertStatus(malformedThumbnail, 400)
  assert.equal((await malformedThumbnail.json()).error.code, 'invalid_request')

  const stored = await fetch(thumbnailUrl, {
    method: 'PUT',
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/octet-stream'
    },
    body: thumbnailBytes
  })
  await assertStatus(stored, 200)
  assert.deepEqual(await stored.json(), { assetId, sizeBucket: 128, generatorVersion: 1 })

  for (const parameter of ['sizeBucket', 'width', 'height', 'encoding']) {
    thumbnailUrl.searchParams.delete(parameter)
  }
  thumbnailUrl.searchParams.set('requestedSize', '100')
  const loaded = await fetch(thumbnailUrl, {
    headers: { authorization: `Bearer ${token}` }
  })
  await assertStatus(loaded, 200)
  assert.equal(loaded.headers.get('content-type'), 'image/png')
  assert.equal(loaded.headers.get('x-nicegal-server-asset-id'), assetId.toString())
  assert.equal(loaded.headers.get('x-nicegal-server-size-bucket'), '128')
  assert.equal(loaded.headers.get('x-nicegal-server-generator-version'), '1')
  assert.equal(loaded.headers.get('x-nicegal-server-thumbnail-width'), '1')
  assert.equal(loaded.headers.get('x-nicegal-server-thumbnail-height'), '1')
  assert.deepEqual(Buffer.from(await loaded.arrayBuffer()), thumbnailBytes)

  await unlink(indexedSourcePath)
  const dryPrune = await runTypedJob(ready.endpoint, token, 'pruneMissing', { root: indexRoot })
  assert.equal(dryPrune.progress.pruneCandidates, 1)
  assert.equal(dryPrune.progress.deleted, 0)
  assert.deepEqual(dryPrune.errors, [])

  const pruned = await runTypedJob(ready.endpoint, token, 'pruneMissing', {
    root: indexRoot,
    dryRun: false
  })
  assert.equal(pruned.progress.pruneCandidates, 1)
  assert.equal(pruned.progress.deleted, 1)
  assert.deepEqual(pruned.errors, [])

  const emptyPrune = await runTypedJob(ready.endpoint, token, 'pruneMissing', { root: indexRoot })
  assert.equal(emptyPrune.progress.total, 0)

  child.stdin.end()
  const [exitCode, signal] = await withTimeout(once(child, 'exit'), 5_000, 'server shutdown')
  assert.equal(signal, null)
  assert.equal(exitCode, 0, stderr)
  assert.match(stderr, /paddle_ocr_load/)
  assert.ok(
    (stderr.match(/onnx_compile/g) ?? []).length >= 4,
    `expected detector and recognizer compilation traces for both loads\n${stderr}`
  )

  console.log(
    `RPC smoke passed: indexed with PaddleOCR and round-tripped ${thumbnailBytes.length} thumbnail bytes`
  )
} finally {
  await stopChild(child)
  await rm(temporaryDirectory, { force: true, recursive: true, maxRetries: 10, retryDelay: 50 })
}

function searchRequestUrl(endpoint, parameters) {
  const url = new URL('/v1/search', endpoint)
  for (const [key, value] of Object.entries(parameters)) {
    url.searchParams.set(key, value)
  }
  return url
}

async function searchError(endpoint, token, parameters) {
  const response = await fetch(searchRequestUrl(endpoint, parameters), {
    headers: { authorization: `Bearer ${token}` }
  })
  const text = await response.text()
  let body
  try {
    body = JSON.parse(text)
  } catch (error) {
    assert.fail(`error responses must be JSON, got ${JSON.stringify(text)}: ${error.message}`)
  }
  return { status: response.status, body }
}

async function assertStatus(response, expected) {
  if (response.status === expected) return
  assert.equal(response.status, expected, `${await response.text()}\n${stderr}`)
}

async function runTypedJob(endpoint, token, type, params) {
  const created = await fetch(`${endpoint}/v1/jobs`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/json'
    },
    body: JSON.stringify({ type, params })
  })
  await assertStatus(created, 202)
  return waitForJob(endpoint, token, await created.json())
}

async function waitForJob(endpoint, token, initialJob) {
  let job = initialJob
  const terminal = new Set(['cancelled', 'completed', 'failed'])
  assert.match(job.jobId, /^[1-9]\d*$/)
  if (!terminal.has(job.status)) {
    const response = await fetch(`${endpoint}/v1/jobs/${job.jobId}/events`, {
      headers: { authorization: `Bearer ${token}` }
    })
    await assertStatus(response, 200)
    for await (const snapshot of sseSnapshots(response)) {
      job = snapshot
    }
  }
  assert.ok(terminal.has(job.status), `index job did not finish:\n${JSON.stringify(job)}\n${stderr}`)
  assert.notEqual(job.status, 'failed', job.error)
  return job
}

async function* sseSnapshots(response) {
  let buffer = ''
  for await (const chunk of response.body.pipeThrough(new TextDecoderStream())) {
    buffer += chunk.replaceAll('\r\n', '\n')
    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const data = block
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n')
      if (data) yield JSON.parse(data)
    }
  }
}

async function stopChild(process) {
  if (process.exitCode !== null || process.signalCode !== null) return

  const gracefulExit = once(process, 'exit')
  if (!process.stdin.writableEnded) process.stdin.end()
  try {
    await withTimeout(gracefulExit, 5_000, 'server cleanup')
    return
  } catch {
    // Fall through to forced termination after the graceful deadline.
  }

  if (process.exitCode === null && process.signalCode === null) {
    const forcedExit = once(process, 'exit')
    process.kill()
    await withTimeout(forcedExit, 5_000, 'forced server cleanup')
  }
}

function readReadyMessage(process) {
  return new Promise((resolve, reject) => {
    const lines = createInterface({ input: process.stdout })
    const onError = (error) => {
      lines.close()
      reject(error)
    }
    const onExit = (code, signal) => {
      lines.close()
      reject(new Error(`server exited before readiness: code=${code} signal=${signal}\n${stderr}`))
    }

    process.once('error', onError)
    process.once('exit', onExit)
    lines.once('line', (line) => {
      process.off('error', onError)
      process.off('exit', onExit)
      lines.close()
      try {
        resolve(JSON.parse(line))
      } catch (error) {
        reject(new Error(`invalid readiness message: ${line}`, { cause: error }))
      }
    })
  })
}

function withTimeout(promise, milliseconds, operation) {
  let timer
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${operation} timed out after ${milliseconds}ms`)),
      milliseconds
    )
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer))
}
