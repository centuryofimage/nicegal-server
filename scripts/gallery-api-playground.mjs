import { spawnServer, createAndWaitForJob, readReadyMessage, stopChild, withTimeout } from './server-harness.mjs'
import assert from 'node:assert/strict'
import { randomBytes } from 'node:crypto'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// Standalone playground for PaddleOCR model preparation and optional indexing.
//
// Usage:
//   node scripts/gallery-api-playground.mjs [SERVER_EXE] [--root=ABSOLUTE_GALLERY] [--keep]

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repository = join(scriptDirectory, '..')
const executableName = process.platform === 'win32' ? 'nicegal-server.exe' : 'nicegal-server'
const arguments_ = process.argv.slice(2)
const keep = arguments_.includes('--keep')
const indexRoot = arguments_
  .find((argument) => argument.startsWith('--root='))
  ?.slice('--root='.length)
const executable = arguments_.find((argument) => !argument.startsWith('--'))
  ?? join(repository, 'target', 'debug', executableName)
const stateDirectory = await mkdtemp(join(tmpdir(), `nicegal-server-model-playground-${process.pid}-`))
const token = randomBytes(32).toString('hex')

console.log(`Temporary state: ${stateDirectory}`)

const child = spawnServer({ executable, repository, stateDirectory, token })

let stderr = ''
child.stderr.setEncoding('utf8')
child.stderr.on('data', (chunk) => {
  stderr += chunk
})

try {
  const ready = await withTimeout(readReadyMessage(child, () => stderr), 10_000, 'server readiness')
  console.log(`RPC endpoint: ${ready.endpoint}`)

  const params = {
    detection: { modelId: 'PaddlePaddle/PP-OCRv6_small_det_onnx' },
    recognition: { modelId: 'PaddlePaddle/PP-OCRv6_small_rec_onnx' }
  }
  const job = await createAndWaitForJob(ready.endpoint, token, 'ocrModelLoad', params)
  console.log(JSON.stringify(job, null, 2))
  assert.equal(job.status, 'completed')
  assert.equal(job.progress.processed, 2)
  assert.equal(job.progress.modelsLoaded, 2)

  const status = await fetch(`${ready.endpoint}/v1/ocr/models`, {
    headers: authorization(token)
  })
  await assertStatus(status, 200)
  console.log('\nLoaded model status:')
  console.log(JSON.stringify(await status.json(), null, 2))

  if (indexRoot) {
    const root = resolve(indexRoot)
    const indexJob = await createAndWaitForJob(ready.endpoint, token, 'libraryIndex', { root })
    assert.equal(indexJob.status, 'completed')
    console.log(`\nIndexed ${root}:`)
    console.log(JSON.stringify(indexJob, null, 2))
  } else {
    console.log('\nPass --root=ABSOLUTE_GALLERY to run the bounded library index pipeline.')
  }
} finally {
  await stopChild(child)
  if (keep) {
    console.log(`Kept state at ${stateDirectory}`)
  } else {
    await rm(stateDirectory, { recursive: true, force: true })
  }
}

function authorization(token) {
  return { authorization: `Bearer ${token}` }
}

async function assertStatus(response, expected) {
  if (response.status === expected) return
  assert.equal(response.status, expected, `${await response.text()}\n${stderr}`)
}
