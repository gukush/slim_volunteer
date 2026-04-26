#!/usr/bin/env node
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

function parseArgs(argv) {
  const out = {};
  for (const token of argv.slice(2)) {
    const m = token.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2];
    else if (token.startsWith('--')) out[token.slice(2)] = true;
  }
  return out;
}

async function api(baseURL, pathname, opts = {}) {
  const url = new URL(pathname, baseURL).toString();
  const r = await fetch(url, opts);
  const ct = r.headers.get('content-type') || '';
  const isJSON = ct.includes('application/json');
  if (!r.ok) {
    const body = isJSON ? await r.json().catch(() => ({})) : await r.text().catch(() => '');
    throw new Error(`HTTP ${r.status} ${r.statusText} @ ${url}: ${isJSON ? JSON.stringify(body) : body}`);
  }
  if (opts.responseType === 'arraybuffer') return await r.arrayBuffer();
  if (opts.responseType === 'text') return await r.text();
  if (isJSON) return await r.json();
  return await r.arrayBuffer();
}

function nonceBufferLE(nonce) {
  const value = BigInt(nonce);
  const out = Buffer.alloc(8);
  out.writeUInt32LE(Number(value & 0xffffffffn) >>> 0, 0);
  out.writeUInt32LE(Number((value >> 32n) & 0xffffffffn) >>> 0, 4);
  return out;
}

function hashFor(prefix, nonce) {
  return crypto.createHash('sha256').update(Buffer.from(prefix, 'utf8')).update(nonceBufferLE(nonce)).digest('hex');
}

async function waitForCompletion(baseURL, taskId, { intervalMs = 1000, timeoutMs = 120000 } = {}) {
  const start = Date.now();
  while (true) {
    const status = await api(baseURL, `/tasks/${taskId}`);
    if (status.status === 'completed') return status;
    if (status.status === 'error' || status.status === 'canceled') {
      throw new Error(`Task ${taskId} ended with status=${status.status}`);
    }
    const total = status.totalChunks || '?';
    process.stdout.write(`\rStatus=${status.status} completedChunks=${status.completedChunks || 0}/${total}   `);
    if (Date.now() - start > timeoutMs) throw new Error('Timeout waiting for completion');
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const baseURL = args.baseURL || process.env.BASE_URL || 'https://localhost:3000';
  const prefix = args.prefix ?? 'abc';
  const expectedNonce = BigInt(args.expectedNonce ?? 42);
  const startNonce = BigInt(args.startNonce ?? 0);
  const totalNonces = BigInt(args.totalNonces ?? 512);
  const chunkSize = Number(args.chunkSize ?? 128);
  const outDir = args.outDir || path.join(process.cwd(), `hash-preimage-results-${Date.now()}`);
  const targetHash = args.targetHash || hashFor(prefix, expectedNonce);

  fs.mkdirSync(outDir, { recursive: true });
  console.log('Target hash:', targetHash);

  const createBody = {
    strategyId: 'hash-preimage',
    label: `hash-preimage prefix=${prefix} target=${targetHash.slice(0, 16)}`,
    input: {
      prefix,
      targetHash,
      startNonce: startNonce.toString(),
      totalNonces: totalNonces.toString(),
      chunkSize,
    },
    config: {
      framework: 'webgpu',
      debugWgslHash: Boolean(args.debugWgslHash),
    },
  };

  const created = await api(baseURL, '/tasks', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(createBody),
  });
  const taskId = created.id || created.taskId || created.task?.id;
  if (!taskId) throw new Error('Could not obtain task id from response: ' + JSON.stringify(created));
  console.log('Created task', taskId);

  await api(baseURL, `/tasks/${taskId}/start`, { method: 'POST' });
  console.log('Started task', taskId);
  await waitForCompletion(baseURL, taskId, { intervalMs: 1000 });
  console.log('\nTask completed');

  const summary = await api(baseURL, `/tasks/${taskId}/output?name=output.json`);
  fs.writeFileSync(path.join(outDir, 'output.json'), JSON.stringify(summary, null, 2));
  console.log('Summary:', JSON.stringify(summary, null, 2));

  if (!summary.found) throw new Error('Expected preimage was not found');
  if (BigInt(summary.match.nonce) !== expectedNonce) {
    throw new Error(`Expected nonce ${expectedNonce}, got ${summary.match.nonce}`);
  }
  if (summary.match.hash !== targetHash) {
    throw new Error(`Expected hash ${targetHash}, got ${summary.match.hash}`);
  }
  console.log('Verified hash preimage result. Artifacts saved to', outDir);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
