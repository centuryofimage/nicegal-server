import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { mkdir, mkdtemp, readFile, realpath, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createLibrary, spawnServer, readReadyMessage, sseSnapshots, stopChild, withTimeout } from './server-harness.mjs'

const repository = join(dirname(fileURLToPath(import.meta.url)), '..')
const options = Object.fromEntries(process.argv.slice(2).map(argument => {
  const match = /^--([^=]+)=(.*)$/.exec(argument)
  assert.ok(match, `expected --name=value: ${argument}`)
  return [match[1], match[2]]
}))
const corpus = await realpath(options.corpus ?? join(repository, '..', 'testdata'))
const executable = await realpath(options.executable ?? join(repository, 'target', 'release', 'nicegal-server.exe'))
const outputRoot = resolve(options.output ?? join(repository, 'target', 'video-index-perf'))
const image = options.image !== 'false'
const limit = options.limit === undefined ? undefined : Number(options.limit)
if (limit !== undefined) assert.ok(Number.isSafeInteger(limit) && limit > 0, 'expected --limit to be a positive integer')
const rustLog = process.env.RUST_LOG ??
  'nicegal_core=trace,nicegal_server=trace,tower_http=info,hyper=warn,h2=warn,tower=warn,rustls=warn'
await mkdir(outputRoot, { recursive: true })
const stateDirectory = await mkdtemp(join(outputRoot, 'run-'))
const token = randomBytes(32).toString('hex')
const child = spawnServer({ executable, repository, stateDirectory, token, env: { RUST_LOG: rustLog } })
let stderr = ''
child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-32_768) })
console.log(`Profile: ${stateDirectory}`)
console.log(`Corpus: ${corpus}`)
console.log(`RUST_LOG: ${rustLog}`)

try {
  const { endpoint } = await withTimeout(readReadyMessage(child, () => stderr), 60_000, 'server readiness')
  const started = performance.now()
  const cpuBefore = cpuSeconds(child.pid)
  const headers = { authorization: `Bearer ${token}`, 'content-type': 'application/json' }
  const libraryId = await createLibrary(endpoint, token, { include: [corpus], ocr: false, image })
  const created = await fetch(`${endpoint}/v1/jobs`, {
    method: 'POST', headers,
    body: JSON.stringify({ type: 'libraryScan', params: { libraryId, debugLimit: limit } })
  })
  assert.equal(created.status, 202, await created.clone().text())
  let job = await created.json()
  const events = await fetch(`${endpoint}/v1/jobs/${job.jobId}/events`, { headers })
  assert.equal(events.status, 200, await events.clone().text())
  let phase = job.phase
  for await (const snapshot of sseSnapshots(events)) {
    job = snapshot
    if (job.phase !== phase) {
      phase = job.phase
      console.log(`Phase: ${phase}`)
    }
  }
  const elapsedMs = performance.now() - started
  const cpuMs = (cpuSeconds(child.pid) - cpuBefore) * 1000
  await stopChild(child)
  const eventsInLog = (await readFile(join(stateDirectory, 'nicegal-server.log'), 'utf8'))
    .trim().split('\n').filter(Boolean).map(line => JSON.parse(line))
  const spans = {}
  for (const event of eventsInLog) {
    if (event.fields?.message !== 'close' || !event.span?.name) continue
    const name = event.span.name
    const entry = spans[name] ??= { count: 0, totalMs: 0, maxMs: 0, totalBusyMs: 0 }
    const elapsed = durationMs(event.fields['time.busy'])
    const lifetime = elapsed + durationMs(event.fields['time.idle'])
    entry.count += 1
    entry.totalMs += lifetime
    entry.totalBusyMs += elapsed
    entry.maxMs = Math.max(entry.maxMs, lifetime)
  }
  const phaseTraceMs = Object.fromEntries(['scan', 'catalog', 'image_index']
    .filter(name => spans[name])
    .map(name => [name, spans[name].totalMs]))
  const report = { corpus, executable, image, limit, rustLog, elapsedMs, cpuMs,
    averageCores: cpuMs / elapsedMs, phaseTraceMs, progress: job.progress, status: job.status,
    errors: job.errors, spans }
  await writeFile(join(stateDirectory, 'profile.json'), JSON.stringify(report, null, 2) + '\n')
  console.log(JSON.stringify({ elapsedMs, cpuMs, averageCores: report.averageCores,
    phaseTraceMs, progress: job.progress, status: job.status }, null, 2))
  assert.equal(job.status, 'completed', JSON.stringify(job))
} catch (error) {
  await writeFile(join(stateDirectory, 'failure.txt'), `${error.stack}\n${stderr}`)
  throw error
} finally {
  await stopChild(child)
}

function cpuSeconds(pid) {
  if (process.platform !== 'win32') return NaN
  return Number(execFileSync('powershell.exe', ['-NoProfile', '-Command', `(Get-Process -Id ${pid}).CPU`],
    { encoding: 'utf8' }).trim())
}

function durationMs(text) {
  const match = /^([\d.]+)(ns|µs|us|ms|s)$/.exec(text)
  if (!match) return 0
  return Number(match[1]) * { ns: 0.000001, 'µs': 0.001, us: 0.001, ms: 1, s: 1000 }[match[2]]
}
