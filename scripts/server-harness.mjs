import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { join } from 'node:path'
import { createInterface } from 'node:readline'

const terminalStatuses = new Set(['cancelled', 'completed', 'failed'])

export function spawnServer({ executable, repository, stateDirectory, token, env = {} }) {
  return spawn(executable, [
    '--asset-database', join(stateDirectory, 'assets.db'),
    '--ocr-database', join(stateDirectory, 'index.db'),
    '--thumbnail-database', join(stateDirectory, 'thumbnails.db'),
    '--runtime-config', join(stateDirectory, 'runtime.json')
  ], {
    cwd: repository,
    env: { ...process.env, ...env, NICEGAL_RPC_TOKEN: token },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true
  })
}

export async function createAndWaitForJob(endpoint, token, type, params, stderrTail = () => '') {
  const response = await fetch(`${endpoint}/v1/jobs`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify({ type, params })
  })
  if (response.status !== 202) {
    throw new Error(`creating ${type} returned HTTP ${response.status}: ${await response.text()}\n${stderrTail()}`)
  }
  return waitForJob(endpoint, token, await response.json(), stderrTail)
}

export async function waitForJob(endpoint, token, job, stderrTail = () => '') {
  if (!/^[1-9]\d*$/.test(job.jobId)) throw new Error(`invalid job ID: ${job.jobId}`)
  if (!terminalStatuses.has(job.status)) {
    const response = await fetch(`${endpoint}/v1/jobs/${job.jobId}/events`, {
      headers: { authorization: `Bearer ${token}` }
    })
    if (response.status !== 200) {
      throw new Error(`opening job events returned HTTP ${response.status}: ${await response.text()}`)
    }
    for await (const snapshot of sseSnapshots(response)) job = snapshot
  }
  if (!terminalStatuses.has(job.status)) throw new Error('job event stream ended without a terminal state')
  if (job.status === 'failed') throw new Error(`job failed: ${JSON.stringify(job)}\n${stderrTail()}`)
  return job
}

export async function* sseSnapshots(response) {
  let buffer = ''
  for await (const chunk of response.body.pipeThrough(new TextDecoderStream())) {
    buffer += chunk
    let boundary
    // Match after concatenation: CRLF can straddle network chunks.
    while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
      const block = buffer.slice(0, boundary.index)
      buffer = buffer.slice(boundary.index + boundary[0].length)
      const data = block.split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).replace(/^ /, ''))
        .join('\n')
      if (data) yield JSON.parse(data)
    }
  }
}

export function readReadyMessage(child, stderrTail = () => '') {
  return new Promise((resolve, reject) => {
    const lines = createInterface({ input: child.stdout })
    const cleanup = () => {
      child.off('error', onError)
      child.off('exit', onExit)
      lines.close()
    }
    const onError = (error) => { cleanup(); reject(error) }
    const onExit = (code, signal) => onError(new Error(
      `server exited before readiness: code=${code} signal=${signal}\n${stderrTail()}`
    ))
    child.once('error', onError)
    child.once('exit', onExit)
    lines.once('line', (line) => {
      cleanup()
      try { resolve(JSON.parse(line)) } catch (error) {
        reject(new Error(`invalid readiness message: ${line}`, { cause: error }))
      }
    })
  })
}

export async function stopChild(child) {
  if (child.exitCode !== null || child.signalCode !== null) return
  const exit = once(child, 'exit')
  if (!child.stdin.writableEnded) child.stdin.end()
  try {
    await withTimeout(exit, 10_000, 'server shutdown')
  } catch {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill()
      await withTimeout(exit, 5_000, 'forced server shutdown')
    }
  }
}

export function withTimeout(promise, milliseconds, operation) {
  let timer
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${operation} timed out after ${milliseconds}ms`)), milliseconds)
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer))
}
