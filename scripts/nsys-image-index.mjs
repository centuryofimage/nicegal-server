import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, realpath, rm } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { createAndWaitForJob, createLibrary, readReadyMessage, spawnServer, stopChild, withTimeout } from './server-harness.mjs'

const defaultNsys = 'C:\\Program Files\\NVIDIA Corporation\\Nsight Systems 2026.5.1\\target-windows-x64\\nsys.exe'
const { values } = parseArgs({ options: {
  executable: { type: 'string', default: 'target/release/nicegal-server.exe' },
  corpus: { type: 'string', default: '../testdata/catcopy' },
  model: { type: 'string', default: 'google/embeddinggemma-2@70' },
  provider: { type: 'string', default: 'directml' },
  'local-models': { type: 'string' },
  output: { type: 'string', default: 'target/nsys-image-index' },
  nsys: { type: 'string', default: process.env.NSYS ?? defaultNsys },
  'no-capture': { type: 'boolean', default: false },
  inner: { type: 'boolean', default: false },
  help: { type: 'boolean', default: false }
} })
if (values.help) {
  console.log(`Usage: node scripts/nsys-image-index.mjs [--corpus DIR] [--model ID] [--local-models JSON] [--output DIR] [--nsys PATH] [--no-capture]
Indexes one corpus's images with a fresh database under Nsight Systems and writes
<output>/image-index.nsys-rep and .sqlite. Build the server with --features nvtx first so tracing
spans appear as NVTX ranges. CPU sampling, context switches and GPU metrics need an elevated shell.
--no-capture runs the same indexing without nsys. Summarize with scripts/analyze-nsys-image-index.py.`)
  process.exit(0)
}

const output = resolve(values.output)
if (values.inner || values['no-capture']) {
  await indexOnce()
} else {
  // nsys does not forward stdin to its target, and the server exits when stdin closes, so nsys
  // profiles this script and the script starts the server. Injection follows child processes.
  if (!existsSync(values.nsys)) throw new Error(`nsys not found: ${values.nsys} (pass --nsys or set NSYS)`)
  await mkdir(output, { recursive: true })
  const forwarded = process.argv.slice(2).filter(argument => argument !== '--no-capture')
  const child = spawn(values.nsys, ['profile', '--trace=dx12,nvtx', '--dx12-gpu-workload=individual',
    '--sample=process-tree', '--cpuctxsw=process-tree', '--gpu-metrics-devices=all', '--gpu-metrics-frequency=10000',
    '--stats=true', '--force-overwrite=true', '-o', join(output, 'image-index'),
    process.execPath, fileURLToPath(import.meta.url), '--inner', ...forwarded], { stdio: 'inherit' })
  const code = await new Promise(resolve => child.once('exit', resolve))
  process.exit(code ?? 1)
}

async function indexOnce() {
  const state = join(output, 'state')
  await rm(state, { recursive: true, force: true })
  await mkdir(state, { recursive: true })
  const corpus = await realpath(values.corpus)
  const token = randomBytes(32).toString('hex')
  const env = { NICEGAL_EXECUTION_PROVIDER: values.provider, NICEGAL_IMAGE_MODEL: values.model,
    RUST_LOG: 'warn,nicegal_core=debug,fastembed=debug,nom_exif=off' }
  if (values['local-models']) env.NICEGAL_LOCAL_MODELS = await realpath(values['local-models'])
  const child = spawnServer({ executable: await realpath(values.executable), repository: process.cwd(),
    stateDirectory: state, token, env })
  let stderr = ''
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-32000) })
  try {
    const { endpoint } = await withTimeout(readReadyMessage(child, () => stderr), 60000, 'server readiness')
    const libraryId = await createLibrary(endpoint, token, { include: [corpus], ocr: false, image: false, videos: false })
    await createAndWaitForJob(endpoint, token, 'libraryScan', { libraryId }, () => stderr)
    const started = performance.now()
    const job = await createAndWaitForJob(endpoint, token, 'imageEmbed', { libraryId }, () => stderr)
    console.log(JSON.stringify({ imageEmbedMs: Math.round(performance.now() - started), embedded: job.progress.embedded,
      failed: job.progress.failed, itemsPerSecond: job.progress.itemsPerSecond }))
  } finally {
    await stopChild(child)
  }
}
