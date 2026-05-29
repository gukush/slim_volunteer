#!/usr/bin/env node
import { Agent } from 'https';
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

const httpsAgent = new Agent({ rejectUnauthorized: false });

async function fetchWithAgent(url, options = {}) {
  if (url.startsWith('https://localhost')) {
    const https = await import('https');
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const reqOptions = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: options.method || 'GET',
        headers: options.headers || {},
        rejectUnauthorized: false,
      };
      const req = https.request(reqOptions, (res) => {
        const data = [];
        res.on('data', (chunk) => data.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(data);
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            statusText: res.statusMessage,
            json: async () => JSON.parse(buffer.toString()),
            text: async () => buffer.toString(),
            arrayBuffer: async () => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
          });
        });
      });
      req.on('error', reject);
      if (options.body) req.write(options.body);
      req.end();
    });
  }
  return fetch(url, { agent: httpsAgent, ...options });
}

async function api(baseURL, pathname, opts = {}) {
  const url = new URL(pathname, baseURL).toString();
  const response = await fetchWithAgent(url, opts);
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText} @ ${url}: ${text}`);
  }
  return text ? JSON.parse(text) : {};
}

async function waitForCompletion(baseURL, taskId, { intervalMs = 1000, timeoutMs = 300000 } = {}) {
  const started = Date.now();
  while (true) {
    const status = await api(baseURL, `/tasks/${taskId}`);
    if (status.status === 'completed') return status;
    if (status.status === 'error' || status.status === 'canceled') {
      throw new Error(`Task ${taskId} ended with status=${status.status}`);
    }
    const total = status.totalChunks || '?';
    const completed = status.completedChunks || 0;
    process.stdout.write(`\rStatus=${status.status} completedChunks=${completed}/${total}   `);
    if (timeoutMs > 0 && Date.now() - started > timeoutMs) {
      throw new Error('Timeout waiting for GridPack-2D RL task completion');
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const host = args.host || args.baseURL || process.env.BASE_URL || 'https://localhost:3000';
  const Krep = Number(args.Krep || 1);
  const timeoutMs = Number(args.timeoutMs || 300000);
  const intervalMs = Number(args.intervalMs || 1000);
  const outDir = args.outDir || path.join(process.cwd(), `gridpack-es-results-${Date.now()}`);

  const gridW = Number(args.gridW || 16);
  const gridH = Number(args.gridH || 16);
  const numBlocks = Number(args.numBlocks || 6);
  const numProblems = Number(args.numProblems || 8);
  const rolloutsPerThread = Number(args.rolloutsPerThread || 128);
  const numThreads = Number(args.numThreads || 1024);
  const maxAttempts = Number(args.maxAttempts || 50);
  const allowRotation = Number(args.allowRotation !== undefined ? args.allowRotation : 1);
  const sigma = Number(args.sigma || 0.01);
  const lr = Number(args.lr || 0.001);
  const mlpHiddenDim = Number(args.mlpHiddenDim || 32);
  const actionRegions = Number(args.actionRegions || 8);
  const policyPath = args.policyPath || null;

  const payload = {
    strategyId: 'gridpack-2d-rl',
    K: Krep,
    label: `gridpack-es-${Date.now()}`,
    input: {
      gridW,
      gridH,
      numBlocks,
      numProblems,
      rolloutsPerThread,
      numThreads,
      maxAttempts,
      allowRotation,
      sigma,
      lr,
      mlpHiddenDim,
      actionRegions,
      ...(policyPath ? { policyPath } : {}),
    },
    config: {
      framework: 'webgpu',
      mlpHiddenDim,
      actionRegions,
    },
  };

  fs.mkdirSync(outDir, { recursive: true });
  console.log('Creating GridPack-2D RL ES task');
  console.log('Payload:', JSON.stringify(payload, null, 2));

  const created = await api(host, '/tasks', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const taskId = created.id || created.taskId || created.task?.id;
  if (!taskId) throw new Error('Could not obtain task id from response: ' + JSON.stringify(created));

  console.log('Created task', taskId);
  await api(host, `/tasks/${taskId}/start`, { method: 'POST' });
  console.log('Started task', taskId);

  const taskStart = Date.now();
  await waitForCompletion(host, taskId, { intervalMs, timeoutMs });
  const totalMs = Date.now() - taskStart;
  console.log(`\nTask completed in ${(totalMs / 1000).toFixed(3)}s`);

  const summary = await api(host, `/tasks/${taskId}/output?name=output.json`);
  fs.writeFileSync(path.join(outDir, 'output.json'), JSON.stringify(summary, null, 2));
  console.log('Summary:', JSON.stringify(summary, null, 2));

  if (summary.validPairs === 0) {
    throw new Error(`GridPack-2D ES found no valid base+perturbed pairs`);
  }
  console.log(`GridPack-2D ES passed: ${summary.validPairs}/${summary.totalPairs} pairs, delta=${summary.meanDelta?.toFixed(6) ?? 'N/A'}`);
  console.log('Artifacts saved to', outDir);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
