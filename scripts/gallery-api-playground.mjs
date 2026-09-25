import { spawnServer, createAndWaitForJob, createLibrary, readReadyMessage, stopChild, withTimeout } from './server-harness.mjs'
import assert from 'node:assert/strict'
import { randomBytes } from 'node:crypto'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// Manual walkthrough of a library over a real folder: create it, scan it, read it back, rescan,
// then drop its exclusions and scan only what that revealed.
//
// Usage:
//   node scripts/gallery-api-playground.mjs [SERVER_EXE] --root=GALLERY [--exclude=FOLDER ...]
//     [--ocr] [--no-image] [--keep]
//
// Relative --exclude folders resolve against --root. Without --root, the script only loads the
// OCR pair and prints the model status.

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repository = join(scriptDirectory, '..')
const executableName = process.platform === 'win32' ? 'nicegal-server.exe' : 'nicegal-server'
const arguments_ = process.argv.slice(2)
const keep = arguments_.includes('--keep')
const option = (name) => arguments_.filter((argument) => argument.startsWith(`--${name}=`))
  .map((argument) => argument.slice(name.length + 3))
const root = option('root')[0] && resolve(option('root')[0])
const exclude = option('exclude').map((folder) => resolve(root ?? '.', folder))
const ocr = arguments_.includes('--ocr')
const image = !arguments_.includes('--no-image')
const executable = arguments_.find((argument) => !argument.startsWith('--'))
  ?? join(repository, 'target', 'debug', executableName)
const stateDirectory = await mkdtemp(join(tmpdir(), `nicegal-server-playground-${process.pid}-`))
const token = randomBytes(32).toString('hex')
const ocrModels = {
  detection: { modelId: 'PaddlePaddle/PP-OCRv6_small_det_onnx' },
  recognition: { modelId: 'PaddlePaddle/PP-OCRv6_small_rec_onnx' }
}

console.log(`Temporary state: ${stateDirectory}`)
const child = spawnServer({ executable, repository, stateDirectory, token })
let stderr = ''
child.stderr.setEncoding('utf8')
child.stderr.on('data', (chunk) => {
  stderr += chunk
})

try {
  const ready = await withTimeout(readReadyMessage(child, () => stderr), 10_000, 'server readiness')
  const endpoint = ready.endpoint
  console.log(`RPC endpoint: ${endpoint}`)

  if (!root) {
    const job = await createAndWaitForJob(endpoint, token, 'ocrModelLoad', ocrModels)
    assert.equal(job.status, 'completed')
    console.log(JSON.stringify(await get(endpoint, '/v1/ocr/models'), null, 2))
    console.log('\nPass --root=GALLERY to walk through a library scan.')
  } else {
    const libraryId = await createLibrary(endpoint, token, { include: [root], exclude, ocr, image })
    console.log(`\nLibrary ${libraryId}: ${root}`)
    for (const folder of exclude) console.log(`  excluding ${folder}`)
    const indexes = [ocr && 'text recognition', image && 'image search'].filter(Boolean)
    console.log(`  indexes: ${indexes.join(', ') || 'none'}`)

    await scan(endpoint, 'first scan', { libraryId, ocrModels })
    const count = await get(endpoint, `/v1/catalog/count?libraryId=${libraryId}`)
    const gallery = await get(endpoint, `/v1/catalog?libraryId=${libraryId}`)
    console.log(`\nCataloged in the library: ${count}`)
    const leaked = gallery.filter((asset) => exclude.some((folder) => isInside(asset.path, folder)))
    assert.deepEqual(leaked, [], 'excluded folders must not reach the catalog')
    console.log(`  none under an excluded folder (${gallery.length} rows checked)`)
    if (image) {
      const search = await get(endpoint, `/v1/search?q=a%20cat%20in%20a%20box&type=image&limit=3&libraryId=${libraryId}`)
      const top = search.results.map((hit) => nameOf(gallery, hit.assetId)).join(', ')
      console.log(`  image search "a cat in a box": ${search.total} hits; top: ${top}`)
    }

    const again = await scan(endpoint, 'rescan with nothing changed', { libraryId, ocrModels })
    assert.equal(again.progress.modelsLoaded, 0, 'nothing pending, so no model loads')
    assert.equal(again.progress.indexed + again.progress.embedded, 0, 'nothing is re-indexed')

    if (exclude.length > 0) {
      const library = await get(endpoint, `/v1/libraries/${libraryId}`)
      const edited = await send(endpoint, 'PUT', `/v1/libraries/${libraryId}`, {
        include: library.include.map((folder) => folder.path), exclude: [], ocr, image
      })
      const pending = edited.include.filter((folder) => folder.scanPending).map((folder) => folder.path)
      console.log(`\nRemoved the exclusions; pending: ${pending.join(', ')}`)
      await scan(endpoint, 'pending-only scan after removing the exclusions', {
        libraryId, ocrModels, pendingOnly: true
      })
      const revealed = await get(endpoint, `/v1/catalog/count?libraryId=${libraryId}`)
      console.log(`\nCataloged in the library: ${count} -> ${revealed}`)
      assert.ok(revealed > count, 'removing an exclusion reveals its files')
    }
    console.log('\nLibrary after scanning:')
    console.log(JSON.stringify(await get(endpoint, `/v1/libraries/${libraryId}`), null, 2))
  }
} finally {
  await stopChild(child)
  if (keep) {
    console.log(`Kept state at ${stateDirectory}`)
  } else {
    await rm(stateDirectory, { recursive: true, force: true })
  }
}

async function scan(endpoint, label, params) {
  const started = performance.now()
  const job = await createAndWaitForJob(endpoint, token, 'libraryScan', params, () => stderr)
  const seconds = ((performance.now() - started) / 1000).toFixed(1)
  console.log(`\n${label}: ${job.status} in ${seconds}s`)
  for (const folder of job.folders) {
    const error = folder.error ? `  (${folder.error})` : ''
    console.log(`  ${folder.state.padEnd(11)} ${folder.path}  discovered ${folder.discovered}, cataloged ${folder.cataloged}, failed ${folder.failed}${error}`)
  }
  const { progress } = job
  console.log(`  models loaded ${progress.modelsLoaded}, OCR indexed ${progress.indexed}, embedded ${progress.embedded}, pruned ${progress.deleted}, item errors ${job.errors.length}`)
  for (const error of job.errors.slice(0, 5)) console.log(`    ${error.path ?? ''}: ${error.message}`)
  assert.equal(job.status, 'completed')
  return job
}

async function get(endpoint, path) {
  const response = await fetch(`${endpoint}${path}`, { headers: { authorization: `Bearer ${token}` } })
  assert.equal(response.status, 200, `${path}: ${await response.clone().text()}`)
  return response.json()
}

async function send(endpoint, method, path, body) {
  const response = await fetch(`${endpoint}${path}`, {
    method,
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body)
  })
  assert.equal(response.status, 200, `${path}: ${await response.clone().text()}`)
  return response.json()
}

function isInside(path, folder) {
  const normalize = (value) => value.replaceAll('\\', '/').replace(/\/+$/, '')
  return normalize(path).startsWith(`${normalize(folder)}/`)
}

function nameOf(gallery, assetId) {
  return gallery.find((asset) => asset.id === String(assetId))?.displayName ?? `#${assetId}`
}
