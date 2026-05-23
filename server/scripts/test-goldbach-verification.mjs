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

async function waitForCompletion(baseURL, taskId, { intervalMs = 1000, timeoutMs = 120000 } = {}) {
  const started = Date.now();
  while (true) {
    const status = await api(baseURL, `/tasks/${taskId}`);
    if (status.status === 'completed') return status;
    if (status.status === 'error' || status.status === 'canceled') {
      throw new Error(`Task ${taskId} ended with status=${status.status}`);
    }
    const total = status.totalChunks || '?';
    process.stdout.write(`\rStatus=${status.status} completedChunks=${status.completedChunks || 0}/${total}   `);
    if (timeoutMs > 0 && Date.now() - started > timeoutMs) {
      throw new Error('Timeout waiting for Goldbach task completion');
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

function parseNumbers(value) {
  if (!value) return null;
  return String(value).split(',').map((x) => {
    const n = Number(x.trim());
    if (!Number.isInteger(n) || n < 0 || n > 0xffffffff) {
      throw new Error(`Invalid uint32 number: ${x}`);
    }
    return n;
  });
}

async function main() {
  const args = parseArgs(process.argv);
  const host = args.host || args.baseURL || process.env.BASE_URL || 'https://localhost:3000';
  const framework = args.framework || 'webgpu';
  const chunkSize = Number(args.chunkSize || 1024);
  const Krep = Number(args.Krep || 1);
  const timeoutMs = Number(args.timeoutMs || 120000);
  const intervalMs = Number(args.intervalMs || 1000);
  const outDir = args.outDir || path.join(process.cwd(), `goldbach-results-${Date.now()}`);

  if (framework.toLowerCase() !== 'webgpu') {
    throw new Error(`goldbach-verification currently supports framework=webgpu, got: ${framework}`);
  }

  const input = { chunkSize };
  const numbers = parseNumbers(args.numbers);
  if (numbers) {
    input.numbers = numbers;
  } else {
    input.start = Number(args.start || 4);
    input.end = Number(args.end || 100000);
  }

  const payload = {
    strategyId: 'goldbach-verification',
    K: Krep,
    label: `goldbach-verification-${framework}`,
    input,
    config: {
      framework: 'webgpu',
      chunkSize,
    },
  };

  fs.mkdirSync(outDir, { recursive: true });
  console.log('Creating Goldbach verification task');
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

  if (!summary.valid) {
    throw new Error(`Goldbach verification failed: ${JSON.stringify(summary.failed)}`);
  }
  console.log('Goldbach verification passed. Artifacts saved to', outDir);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
