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
      const req = https.request({
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: options.method || 'GET',
        headers: options.headers || {},
        rejectUnauthorized: false,
      }, (res) => {
        const data = [];
        res.on('data', (chunk) => data.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(data);
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            statusText: res.statusMessage,
            text: async () => buffer.toString(),
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

async function waitForCompletion(baseURL, taskId, { intervalMs, timeoutMs }) {
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
      throw new Error('Timeout waiting for torus-grid-rl task completion');
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const host = args.host || args.baseURL || process.env.BASE_URL || 'https://localhost:3000';
  const totalTrajectories = Number(args.totalTrajectories || 8192);
  const chunkTrajectories = Number(args.chunkTrajectories || args.chunkSize || 4096);
  const workgroupSize = Number(args.workgroupSize || 128);
  const maxSteps = Number(args.maxSteps || 128);
  const Krep = Number(args.Krep || 1);
  const timeoutMs = Number(args.timeoutMs || 120000);
  const intervalMs = Number(args.intervalMs || 1000);
  const outDir = args.outDir || path.join(process.cwd(), `torus-grid-rl-results-${Date.now()}`);

  const payload = {
    strategyId: 'torus-grid-rl',
    K: Krep,
    label: 'torus-grid-rl-webgpu',
    input: {
      totalTrajectories,
      chunkTrajectories,
      workgroupSize,
      maxSteps,
    },
    config: {
      framework: 'webgpu',
      totalTrajectories,
      chunkTrajectories,
      workgroupSize,
      maxSteps,
    },
  };

  fs.mkdirSync(outDir, { recursive: true });
  console.log('Creating Torus Grid RL task');
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
  const taskDuration = Date.now() - taskStart;
  console.log(`\nTask completed in ${taskDuration} ms.`);

  const summary = await api(host, `/tasks/${taskId}/output?name=output.json`);
  fs.writeFileSync(path.join(outDir, 'output.json'), JSON.stringify(summary, null, 2));
  console.log('Summary:', JSON.stringify(summary, null, 2));
  console.log('Artifacts saved to', outDir);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
