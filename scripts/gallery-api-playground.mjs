import assert from 'node:assert/strict'
import { randomBytes } from 'node:crypto'
import { once } from 'node:events'
import { mkdtemp, rm } from 'node:fs/promises'
import { createInterface } from 'node:readline'
import { spawn } from 'node:child_process'
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

const child = spawn(
  executable,
  [
    '--asset-database', join(stateDirectory, 'assets.db'),
    '--ocr-database', join(stateDirectory, 'index.db'),
    '--thumbnail-database', join(stateDirectory, 'thumbnails.db')
  ],
  {
    cwd: repository,
    env: { ...process.env, NICEGAL_RPC_TOKEN: token },
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
    const indexJob = await createAndWaitForJob(ready.endpoint, token, 'ocrIndex', { root })
    assert.equal(indexJob.status, 'completed')
    console.log(`\nIndexed ${root}:`)
    console.log(JSON.stringify(indexJob, null, 2))
  } else {
    console.log('\nPass --root=ABSOLUTE_GALLERY to run the bounded OCR index pipeline.')
  }
} finally {
  await stopChild(child)
  if (keep) {
    console.log(`Kept state at ${stateDirectory}`)
  } else {
    await rm(stateDirectory, { recursive: true, force: true })
  }
}

async function createAndWaitForJob(endpoint, token, type, params) {
  const created = await fetch(`${endpoint}/v1/jobs`, {
    method: 'POST',
    headers: { ...authorization(token), 'content-type': 'application/json' },
    body: JSON.stringify({ type, params })
  })
  await assertStatus(created, 202)
  let job = await created.json()
  if (!['cancelled', 'completed', 'failed'].includes(job.status)) {
    const events = await fetch(`${endpoint}/v1/jobs/${job.jobId}/events`, {
      headers: authorization(token)
    })
    await assertStatus(events, 200)
    for await (const snapshot of sseSnapshots(events)) job = snapshot
  }
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

function authorization(token) {
  return { authorization: `Bearer ${token}` }
}

async function assertStatus(response, expected) {
  if (response.status === expected) return
  assert.equal(response.status, expected, `${await response.text()}\n${stderr}`)
}

async function stopChild(process) {
  if (process.exitCode !== null || process.signalCode !== null) return
  const gracefulExit = once(process, 'exit')
  if (!process.stdin.writableEnded) process.stdin.end()
  try {
    await withTimeout(gracefulExit, 10_000, 'server shutdown')
  } catch {
    if (process.exitCode === null && process.signalCode === null) {
      const forcedExit = once(process, 'exit')
      process.kill()
      await withTimeout(forcedExit, 5_000, 'forced server shutdown')
    }
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
