import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { once } from 'node:events'
import { createWriteStream } from 'node:fs'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { createInterface } from 'node:readline'
import { DatabaseSync } from 'node:sqlite'
import { finished } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'

const HELP = `Usage: node scripts/thumbnail-perf.mjs [SERVER_EXE] [options]

Creates a fresh catalog for a corpus, then measures staggered, overlapping synchronous
POST /v1/thumbnails ensure requests. Server stderr is both displayed and written to trace.log.

Options:
  --requests N          Number of ensure requests in each burst (default: 12)
  --batch-size N        Asset IDs per ensure request (default: 8)
  --interval-ms N       Delay between launches; requests are never awaited before the next launch (default: 20)
  --required-size N     Required thumbnail size, from 1 through 1024 (default: 256)
  --corpus PATH         Corpus root (default: ../testdata/catcopy)
  --executable PATH     Server executable (default: target/release/nicegal-server)
  --keep                Keep the temporary databases and trace.log
  --warm                Run a separately labelled warm burst after the cold burst
  --help                Show this help

SERVER_EXE is an alternative positional executable override.`

const options = parseOptions(process.argv.slice(2))
if (options.help) {
  console.log(HELP)
  process.exit(0)
}

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repository = join(scriptDirectory, '..')
const executableName = process.platform === 'win32' ? 'nicegal-server.exe' : 'nicegal-server'
const corpus = await realpath(options.corpus ?? join(repository, '..', 'testdata', 'catcopy'))
const executable = await realpath(
  options.executable ?? join(repository, 'target', 'release', executableName)
)
const rustLog = process.env.RUST_LOG ??
  'nicegal_core=trace,nicegal_server=trace,tower_http=info,hyper=warn,h2=warn,tower=warn,rustls=warn'
const stateDirectory = await mkdtemp(join(tmpdir(), `nicegal-server-thumbnail-perf-${process.pid}-`))
const assetDatabasePath = join(stateDirectory, 'assets.db')
const ocrDatabasePath = join(stateDirectory, 'index.db')
const thumbnailDatabasePath = join(stateDirectory, 'thumbnails.db')
const tracePath = join(stateDirectory, 'trace.log')
const token = randomBytes(32).toString('hex')
const trace = createWriteStream(tracePath, { flags: 'a' })
let traceError
trace.on('error', (error) => {
  traceError = error
})

let child
let primaryError
try {
  console.log(`Temporary state: ${stateDirectory}`)
  console.log(`Trace log: ${tracePath}`)
  console.log(`Corpus: ${corpus}`)
  console.log(`Executable: ${executable}`)

  child = spawn(
    executable,
    [
      '--asset-database', assetDatabasePath,
      '--ocr-database', ocrDatabasePath,
      '--thumbnail-database', thumbnailDatabasePath
    ],
    {
      cwd: repository,
      env: {
        ...process.env,
        NICEGAL_RPC_TOKEN: token,
        // Trace is required because this harness is specifically for attributing request latency.
        RUST_LOG: rustLog
      },
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true
    }
  )

  let stderrTail = ''
  child.stderr.setEncoding('utf8')
  child.stderr.on('data', (chunk) => {
    trace.write(chunk)
    process.stderr.write(chunk)
    stderrTail = `${stderrTail}${chunk}`.slice(-32_768)
  })

  const ready = await withTimeout(readReadyMessage(child, () => stderrTail), 30_000, 'server readiness')
  if (ready.apiVersion !== 1 || !/^http:\/\/127\.0\.0\.1:\d+$/.test(ready.endpoint)) {
    throw new Error(`unexpected server readiness response: ${JSON.stringify(ready)}`)
  }

  const catalog = await createAndWaitForJob(
    ready.endpoint,
    token,
    'catalogSync',
    { root: corpus },
    () => stderrTail
  )
  if (catalog.status !== 'completed') {
    throw new Error(`catalogSync ended as ${catalog.status}: ${JSON.stringify(catalog)}`)
  }

  const assetIds = readImageAssetIds(assetDatabasePath)
  if (assetIds.length === 0) {
    throw new Error(`catalogSync completed without image assets under ${corpus}`)
  }

  const effectiveBatchSize = Math.min(options.batchSize, assetIds.length)
  const windows = buildOverlappingWindows(assetIds, options.requests, effectiveBatchSize)
  printConfiguration({ corpus, executable, assetIds, options, effectiveBatchSize })

  const cold = await runBurst('cold', ready.endpoint, token, windows)
  printBurstResults('cold', cold)

  if (options.warm) {
    console.log('\nWarm burst (separate from and excluded from cold statistics):')
    const warm = await runBurst('warm', ready.endpoint, token, windows)
    printBurstResults('warm', warm)
  }
} catch (error) {
  primaryError = error
  throw error
} finally {
  let cleanupError
  if (child) {
    try {
      await stopChild(child)
    } catch (error) {
      cleanupError = error
    }
  }

  try {
    trace.end()
    await finished(trace)
    if (traceError) throw traceError
  } catch (error) {
    cleanupError ??= error
  }

  if (options.keep) {
    console.log(`Retained temporary artifacts and trace log: ${stateDirectory}`)
  } else {
    try {
      await rm(stateDirectory, { force: true, recursive: true, maxRetries: 10, retryDelay: 50 })
      console.log(`Removed temporary artifacts; rerun with --keep to retain trace log: ${tracePath}`)
    } catch (error) {
      cleanupError ??= error
    }
  }

  if (!primaryError && cleanupError) throw cleanupError
}

function parseOptions(arguments_) {
  const options = {
    requests: 12,
    batchSize: 8,
    intervalMs: 20,
    requiredSize: 256,
    keep: false,
    warm: false,
    help: false
  }
  const aliases = new Map([
    ['request-count', 'requests'],
    ['interval', 'interval-ms']
  ])
  const valueOptions = new Map([
    ['requests', 'requests'],
    ['batch-size', 'batchSize'],
    ['interval-ms', 'intervalMs'],
    ['required-size', 'requiredSize'],
    ['corpus', 'corpus'],
    ['executable', 'executable']
  ])
  const positionals = []
  const provided = new Set()

  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index]
    if (argument === '--help') {
      options.help = true
      continue
    }
    if (argument === '--keep' || argument === '--warm') {
      options[argument.slice(2)] = true
      continue
    }
    if (!argument.startsWith('--')) {
      positionals.push(argument)
      continue
    }

    const equals = argument.indexOf('=')
    const rawName = argument.slice(2, equals === -1 ? undefined : equals)
    const name = aliases.get(rawName) ?? rawName
    const option = valueOptions.get(name)
    if (!option) throw new Error(`unknown option: ${argument}\n\n${HELP}`)
    const value = equals === -1 ? arguments_[++index] : argument.slice(equals + 1)
    if (!value || value.startsWith('--')) throw new Error(`missing value for --${rawName}`)
    if (provided.has(option)) {
      throw new Error(`--${rawName} was provided more than once`)
    }
    options[option] = value
    provided.add(option)
  }

  if (positionals.length > 1) throw new Error(`expected at most one positional executable, got: ${positionals.join(' ')}`)
  if (positionals.length === 1) {
    if (options.executable) throw new Error('provide the executable either positionally or with --executable, not both')
    options.executable = positionals[0]
  }

  options.requests = parsePositiveInteger(options.requests, '--requests')
  options.batchSize = parsePositiveInteger(options.batchSize, '--batch-size')
  options.intervalMs = parseNonNegativeInteger(options.intervalMs, '--interval-ms')
  options.requiredSize = parsePositiveInteger(options.requiredSize, '--required-size')
  if (options.requiredSize > 1024) throw new Error('--required-size must be between 1 and 1024')
  return options
}

function parsePositiveInteger(value, option) {
  const number = Number(value)
  if (!Number.isSafeInteger(number) || number < 1) throw new Error(`${option} must be a positive integer`)
  return number
}

function parseNonNegativeInteger(value, option) {
  const number = Number(value)
  if (!Number.isSafeInteger(number) || number < 0) throw new Error(`${option} must be a non-negative integer`)
  return number
}

function readImageAssetIds(assetDatabasePath) {
  const database = new DatabaseSync(assetDatabasePath, { readOnly: true })
  try {
    return database
      .prepare("SELECT asset_id FROM assets WHERE media_kind = 'image' ORDER BY asset_id")
      .all()
      .map((row) => row.asset_id)
  } finally {
    database.close()
  }
}

function buildOverlappingWindows(assetIds, requestCount, batchSize) {
  const overlap = Math.min(Math.floor(batchSize / 2), batchSize - 1)
  // A half-corpus stride bounces between two narrow regions. Use a golden-ratio jump and adjust
  // it until it is coprime with the corpus size so short bursts sample the whole catalog and long
  // bursts eventually visit every starting position.
  let jump = Math.max(1, Math.floor(assetIds.length * 0.6180339887498948))
  while (greatestCommonDivisor(jump, assetIds.length) !== 1) jump += 1
  const windows = []

  for (let request = 0; request < requestCount; request += 1) {
    const previous = windows.at(-1) ?? []
    const window = previous.slice(Math.max(0, previous.length - overlap))
    const start = (request * jump) % assetIds.length
    for (let offset = 0; window.length < batchSize && offset < assetIds.length; offset += 1) {
      const assetId = assetIds[(start + offset) % assetIds.length]
      if (!window.includes(assetId)) window.push(assetId)
    }
    windows.push(window)
  }
  return windows
}

function greatestCommonDivisor(left, right) {
  while (right !== 0) [left, right] = [right, left % right]
  return left
}

function printConfiguration({ corpus, executable, assetIds, options, effectiveBatchSize }) {
  console.log('\nThumbnail ensure performance configuration:')
  console.log(`  corpus: ${corpus}`)
  console.log(`  executable: ${executable}`)
  console.log(`  RUST_LOG: ${rustLog}`)
  console.log(`  catalog image assets: ${assetIds.length}`)
  console.log(`  cold requests: ${options.requests}`)
  console.log(`  batch size: ${effectiveBatchSize}${effectiveBatchSize !== options.batchSize ? ` (capped from ${options.batchSize} by corpus)` : ''}`)
  console.log(`  launch interval: ${options.intervalMs}ms`)
  console.log(`  required size: ${options.requiredSize}`)
}

async function runBurst(label, endpoint, token, windows) {
  const pending = []
  for (let index = 0; index < windows.length; index += 1) {
    pending.push(ensureThumbnails(label, index, endpoint, token, windows[index]))
    if (index + 1 < windows.length) await delay(options.intervalMs)
  }
  return Promise.all(pending)
}

async function ensureThumbnails(label, index, endpoint, token, assetIds) {
  const started = performance.now()
  try {
    const response = await fetch(`${endpoint}/v1/thumbnails`, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${token}`,
        'content-type': 'application/json'
      },
      body: JSON.stringify({ assetIds, requiredSize: options.requiredSize })
    })
    const body = await response.text()
    const milliseconds = performance.now() - started
    if (response.status !== 200) {
      throw new Error(`HTTP ${response.status}: ${body}`)
    }
    return { index, assetIds, milliseconds }
  } catch (error) {
    const milliseconds = performance.now() - started
    throw new Error(
      `${label} ensure request ${index + 1} failed after ${milliseconds.toFixed(2)}ms: ${error.message}`,
      { cause: error }
    )
  }
}

function printBurstResults(label, results) {
  console.log(`\n${label} request timings:`)
  for (const result of results) {
    console.log(
      `  ${String(result.index + 1).padStart(2, '0')}: ${result.milliseconds.toFixed(2)}ms  assetIds=[${result.assetIds.join(', ')}]`
    )
  }
  const values = results.map((result) => result.milliseconds).sort((left, right) => left - right)
  console.log(
    `  aggregate: min=${values[0].toFixed(2)}ms p50=${percentile(values, 50).toFixed(2)}ms p90=${percentile(values, 90).toFixed(2)}ms p95=${percentile(values, 95).toFixed(2)}ms max=${values.at(-1).toFixed(2)}ms`
  )
}

function percentile(sorted, percent) {
  const position = ((sorted.length - 1) * percent) / 100
  const lower = Math.floor(position)
  const upper = Math.ceil(position)
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
}

async function createAndWaitForJob(endpoint, token, type, params, stderrTail) {
  const created = await fetch(`${endpoint}/v1/jobs`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${token}`,
      'content-type': 'application/json'
    },
    body: JSON.stringify({ type, params })
  })
  const createdBody = await created.text()
  if (created.status !== 202) {
    throw new Error(`creating ${type} returned HTTP ${created.status}: ${createdBody}\n${stderrTail()}`)
  }

  let job
  try {
    job = JSON.parse(createdBody)
  } catch (error) {
    throw new Error(`creating ${type} returned invalid JSON: ${createdBody}`, { cause: error })
  }
  const terminal = new Set(['cancelled', 'completed', 'failed'])
  if (!terminal.has(job.status)) {
    const events = await fetch(`${endpoint}/v1/jobs/${job.jobId}/events`, {
      headers: { authorization: `Bearer ${token}` }
    })
    if (events.status !== 200) {
      throw new Error(`opening ${type} event stream returned HTTP ${events.status}: ${await events.text()}`)
    }
    for await (const snapshot of sseSnapshots(events)) job = snapshot
  }
  if (!terminal.has(job.status)) throw new Error(`${type} event stream ended without a terminal job state`)
  if (job.status !== 'completed') {
    throw new Error(`${type} failed: ${JSON.stringify(job)}\n${stderrTail()}`)
  }
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
    await withTimeout(gracefulExit, 10_000, 'server shutdown')
    return
  } catch {
    // The process is ours, so a direct signal cannot affect unrelated processes.
  }

  if (process.exitCode === null && process.signalCode === null) {
    const forcedExit = once(process, 'exit')
    process.kill()
    await withTimeout(forcedExit, 5_000, 'forced server shutdown')
  }
}

function readReadyMessage(process, stderrTail) {
  return new Promise((resolve, reject) => {
    const lines = createInterface({ input: process.stdout })
    const onError = (error) => {
      lines.close()
      reject(error)
    }
    const onExit = (code, signal) => {
      lines.close()
      reject(new Error(`server exited before readiness: code=${code} signal=${signal}\n${stderrTail()}`))
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

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
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
