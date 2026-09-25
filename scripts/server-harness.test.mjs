import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { createServer } from 'node:http'
import test from 'node:test'
import { createAndWaitForJob, readReadyMessage, sseSnapshots, stopChild, withTimeout } from './server-harness.mjs'

test('SSE snapshots tolerate split CRLF and UTF-8 sequences', async () => {
  const bytes = new TextEncoder().encode(': ping\r\n\r\ndata: {"text":"中"}\r\n\r\ndata: {"status":\n' +
    'data: "completed"}\n\n')
  const body = new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte))
      controller.close()
    }
  })
  const snapshots = []
  for await (const snapshot of sseSnapshots(new Response(body))) snapshots.push(snapshot)
  assert.deepEqual(snapshots, [{ text: '中' }, { status: 'completed' }])
})

test('job helpers require a terminal event and report failed jobs', async () => {
  let status = 'running'
  const server = createServer((request, response) => {
    assert.equal(request.headers.authorization, 'Bearer test-token')
    if (request.method === 'POST') {
      request.resume()
      response.writeHead(202, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ jobId: '1', status: 'queued' }))
    } else {
      response.writeHead(200, { 'content-type': 'text/event-stream' })
      response.end(`data: ${JSON.stringify({ jobId: '1', status })}\n\n`)
    }
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const endpoint = `http://127.0.0.1:${server.address().port}`
  try {
    await assert.rejects(createAndWaitForJob(endpoint, 'test-token', 'libraryScan', {}), /without a terminal state/)
    status = 'failed'
    await assert.rejects(createAndWaitForJob(endpoint, 'test-token', 'libraryScan', {}), /job failed/)
    status = 'completed'
    assert.equal((await createAndWaitForJob(endpoint, 'test-token', 'libraryScan', {})).status, status)
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()))
  }
})

test('readiness removes listeners and shutdown closes the owned process', async () => {
  const child = spawn(process.execPath, ['-e',
    'console.log(JSON.stringify({apiVersion:1})); process.stdin.resume(); process.stdin.on("end", () => process.exit(0))'
  ], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true })
  try {
    assert.deepEqual(await withTimeout(readReadyMessage(child), 5_000, 'readiness'), { apiVersion: 1 })
    assert.equal(child.listenerCount('error'), 0)
    assert.equal(child.listenerCount('exit'), 0)
  } finally {
    await stopChild(child)
  }
  assert.equal(child.exitCode, 0)
})
