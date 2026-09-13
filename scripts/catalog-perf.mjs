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

const HELP = `Usage: node scripts/catalog-perf.mjs [SERVER_EXE] [options]

Starts the server with fresh databases and measures one catalogSync job against a real corpus.
Server stderr is displayed and retained as trace.log with --keep.

Options:
  --corpus PATH       Corpus root (default: C:/Users/bepvt/Pictures/laptop)
  --limit N           Debug limit passed to catalogSync (default: 1000)
  --full              Catalog the complete corpus without a debug limit
  --executable PATH   Server executable (default: target/release/nicegal-server)
  --keep              Keep the temporary databases and trace.log
  --help              Show this help

SERVER_EXE is an alternative positional executable override.`

const options = parseOptions(process.argv.slice(2))
if (options.help) {
  console.log(HELP)
  process.exit(0)
}

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repository = join(scriptDirectory, '..')
const executableName = process.platform === 'win32' ? 'nicegal-server.exe' : 'nicegal-server'
const corpus = await realpath(options.corpus ?? 'C:/Users/bepvt/Pictures/laptop')
const executable = await realpath(
  options.executable ?? join(repository, 'target', 'release', executableName)
)
const stateDirectory = await mkdtemp(join(tmpdir(), `nicegal-server-catalog-perf-${process.pid}-`))
const assetDatabasePath = join(stateDirectory, 'assets.db')
const tracePath = join(stateDirectory, 'trace.log')
const token = randomBytes(32).toString('hex')
const trace = createWriteStream(tracePath, { flags: 'a' })
const rustLog = process.env.RUST_LOG ??
  'nicegal_core::index=trace,nicegal_server=info,hyper=warn,h2=warn,tower=warn,rustls=warn'
let traceText = ''
let child
let primaryError

try {
  console.log(`Temporary state: ${stateDirectory}`)
  console.log(`Trace log: ${tracePath}`)
  console.log(`Corpus: ${corpus}`)
  console.log(`Debug limit: ${options.full ? 'none' : options.limit}`)
  console.log(`Executable: ${executable}`)

  child = spawn(
    executable,
    [
      '--asset-database', assetDatabasePath,
      '--ocr-database', join(stateDirectory, 'index.db'),
      '--thumbnail-database', join(stateDirectory, 'thumbnails.db')
    ],
    {
      cwd: repository,
      env: { ...process.env, NICEGAL_RPC_TOKEN: token, RUST_LOG: rustLog },
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true
    }
  )

  let stderrTail = ''
  child.stderr.setEncoding('utf8')
  child.stderr.on('data', (chunk) => {
    traceText += chunk
    trace.write(chunk)
    process.stderr.write(chunk)
    stderrTail = `${stderrTail}${chunk}`.slice(-32_768)
  })

  const ready = await withTimeout(readReadyMessage(child, () => stderrTail), 120_000, 'server readiness')
  if (ready.apiVersion !== 1 || !/^http:\/\/127\.0\.0\.1:\d+$/.test(ready.endpoint)) {
    throw new Error(`unexpected server readiness response: ${JSON.stringify(ready)}`)
  }

  const started = performance.now()
  const catalog = await createAndWaitForJob(
    ready.endpoint,
    token,
    'catalogSync',
    options.full ? { root: corpus } : { root: corpus, scan: { debugLimit: options.limit } },
    () => stderrTail
  )
  const elapsedMilliseconds = performance.now() - started
  if (catalog.status !== 'completed') {
    throw new Error(`catalogSync ended as ${catalog.status}: ${JSON.stringify(catalog)}`)
  }

  const database = new DatabaseSync(assetDatabasePath, { readOnly: true })
  let row
  try {
    row = database.prepare(
      'SELECT COUNT(*) AS assets, SUM(source_size) AS bytes, COUNT(DISTINCT media_format) AS formats FROM assets'
    ).get()
  } finally {
    database.close()
  }

  console.log('\nCatalog benchmark:')
  console.log(`  wall: ${elapsedMilliseconds.toFixed(2)}ms`)
  console.log(`  cataloged rows: ${row.assets}`)
  console.log(`  source bytes inspected: ${Number(row.bytes ?? 0).toLocaleString()}`)
  console.log(`  distinct formats: ${row.formats}`)
  printTraceAttribution(traceText)
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
  } catch (error) {
    cleanupError ??= error
  }

  if (options.keep) {
    console.log(`Retained temporary artifacts and trace log: ${stateDirectory}`)
  } else {
    try {
      await rm(stateDirectory, { force: true, recursive: true, maxRetries: 10, retryDelay: 50 })
      console.log('Removed temporary artifacts; rerun with --keep to retain the trace and databases')
    } catch (error) {
      cleanupError ??= error
    }
  }

  if (!primaryError && cleanupError) throw cleanupError
}

function parseOptions(arguments_) {
  const options = { limit: 1000, full: false, keep: false, help: false }
  const positionals = []
  const provided = new Set()
  const valueOptions = new Map([
    ['corpus', 'corpus'],
    ['limit', 'limit'],
    ['executable', 'executable']
  ])

  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index]
    if (argument === '--help') {
      options.help = true
      continue
    }
    if (argument === '--keep') {
      options.keep = true
      continue
    }
    if (argument === '--full') {
      options.full = true
      continue
    }
    if (!argument.startsWith('--')) {
      positionals.push(argument)
      continue
    }
    const equals = argument.indexOf('=')
    const name = argument.slice(2, equals === -1 ? undefined : equals)
    const option = valueOptions.get(name)
    if (!option) throw new Error(`unknown option: ${argument}\n\n${HELP}`)
    const value = equals === -1 ? arguments_[++index] : argument.slice(equals + 1)
    if (!value || value.startsWith('--')) throw new Error(`missing value for --${name}`)
    if (provided.has(option)) throw new Error(`--${name} was provided more than once`)
    options[option] = value
    provided.add(option)
  }

  if (positionals.length > 1) throw new Error(`expected at most one positional executable, got: ${positionals.join(' ')}`)
  if (positionals.length === 1) {
    if (options.executable) throw new Error('provide the executable either positionally or with --executable, not both')
    options.executable = positionals[0]
  }
  options.limit = parsePositiveInteger(options.limit, '--limit')
  if (options.full && provided.has('limit')) {
    throw new Error('--full and --limit cannot be used together')
  }
  return options
}

function parsePositiveInteger(value, option) {
  const number = Number(value)
  if (!Number.isSafeInteger(number) || number < 1) throw new Error(`${option} must be a positive integer`)
  return number
}

function printTraceAttribution(text) {
  const line = text.split(/\r?\n/).findLast((candidate) =>
    candidate.includes(':catalog:') && candidate.includes('metadata_us=') && candidate.includes('time.busy=')
  )
  if (!line) {
    console.log('  trace attribution: catalog close span not found; inspect trace.log with --keep')
    return
  }

  const names = [
    'metadata_us',
    'canonicalize_us',
    'fingerprint_us',
    'lookup_us',
    'dimensions_us',
    'exif_us',
    'animation_us',
    'transaction_begin_us',
    'row_upsert_us',
    'revision_update_us',
    'commit_us'
  ]
  const values = Object.fromEntries(names.map((name) => {
    const match = line.match(new RegExp(`${name}=(\\d+)`))
    return [name, match ? Number(match[1]) : 0]
  }))
  const measured = Object.values(values).reduce((sum, value) => sum + value, 0)
  console.log('  traced catalog stages:')
  for (const name of names) {
    const percent = measured === 0 ? 0 : (values[name] * 100) / measured
    console.log(`    ${name.replace('_us', '').padEnd(12)} ${(values[name] / 1000).toFixed(2).padStart(10)}ms  ${percent.toFixed(1).padStart(5)}%`)
  }
  console.log(`    measured sum ${(measured / 1000).toFixed(2).padStart(10)}ms`)
}

async function createAndWaitForJob(endpoint, token, type, params, stderrTail) {
  const created = await fetch(`${endpoint}/v1/jobs`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify({ type, params })
  })
  const createdBody = await created.text()
  if (created.status !== 202) {
    throw new Error(`creating ${type} returned HTTP ${created.status}: ${createdBody}\n${stderrTail()}`)
  }

  let job = JSON.parse(createdBody)
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
  if (job.status !== 'completed') throw new Error(`${type} failed: ${JSON.stringify(job)}\n${stderrTail()}`)
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
      const data = block.split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n')
      if (data) yield JSON.parse(data)
    }
  }
}

function readReadyMessage(process, stderrTail) {
  return new Promise((resolve, reject) => {
    const lines = createInterface({ input: process.stdout })
    const onError = (error) => { lines.close(); reject(error) }
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
      try { resolve(JSON.parse(line)) } catch (error) {
        reject(new Error(`invalid readiness message: ${line}`, { cause: error }))
      }
    })
  })
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

function withTimeout(promise, milliseconds, operation) {
  let timer
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${operation} timed out after ${milliseconds}ms`)), milliseconds)
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer))
}
