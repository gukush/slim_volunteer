#!/usr/bin/env node

import fs from 'fs';
import path from 'path';

const args = Object.fromEntries(process.argv.slice(2).map((s) => {
  const m = s.match(/^--([^=]+)=(.*)$/);
  return m ? [m[1], m[2]] : [s.replace(/^--/, ''), true];
}));

const host = args.host || 'https://localhost:3000';
const N = args.N || null;
const batchFile = args.batchFile || null;
const limit = args.limit ? Number(args.limit) : null;
const B1 = Number(args.B1 ?? 100000);
const startBase = Number(args.startBase ?? 2);
const totalBases = Number(args.totalBases ?? 1);
const chunkSize = Number(args.chunkSize ?? 128);
const Krep = Number(args.Krep ?? args.K ?? 1);
const timeoutMs = Number(args.timeoutMs ?? 300000);
const intervalMs = Number(args.intervalMs ?? 1000);
const outDir = args.outDir || `/app/pollard-pminus1-results-${Date.now()}`;

async function api(pathname, options = {}) {
  const url = new URL(pathname, host).toString();
  const response = await fetch(url, {
    ...options,
    headers: {
      ...(options.body ? { 'content-type': 'application/json' } : null),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}: ${await response.text()}`);
  }
  const ct = response.headers.get('content-type') || '';
  if (options.responseType === 'arraybuffer') return response.arrayBuffer();
  return ct.includes('application/json') ? response.json() : response.text();
}

async function waitForTask(taskId) {
  const start = Date.now();
  while (true) {
    const status = await api(`/tasks/${taskId}`);
    if (status.status === 'completed') return status;
    if (status.status === 'error' || status.status === 'canceled') {
      throw new Error(`Task ${taskId} ended with status ${status.status}`);
    }
    process.stdout.write(`\rStatus=${status.status} completedChunks=${status.completedChunks || 0}/${status.totalChunks ?? '?'}   `);
    if (Date.now() - start > timeoutMs) throw new Error('Timeout waiting for Pollard p-1 task');
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

async function main() {
  fs.mkdirSync(outDir, { recursive: true });

  const input = { B1, startBase, totalBases, chunkSize };
  const config = { framework: 'webgpu', B1, startBase, totalBases, chunkSize };

  if (batchFile) {
    input.batchFile = batchFile;
    config.batchFile = batchFile;
    if (limit !== null && limit > 0) {
      input.limit = limit;
      config.limit = limit;
    }
    console.log(`Using batchFile: ${batchFile}${limit ? ` (limit=${limit})` : ''}`);
  } else {
    input.N = N || '8051';
    console.log(`Using single N: ${input.N}`);
  }

  const payload = {
    strategyId: 'pollard-pminus1',
    K: Krep,
    label: `pollard-pminus1-${Date.now()}`,
    input,
    config,
  };
  console.log('Creating Pollard p-1 task');
  console.log('Payload:', JSON.stringify(payload, null, 2));

  const created = await api('/tasks', { method: 'POST', body: JSON.stringify(payload) });
  const taskId = created.id || created.taskId || created.task?.id;
  if (!taskId) throw new Error(`Could not determine task id: ${JSON.stringify(created)}`);
  console.log('Created task', taskId);

  await api(`/tasks/${taskId}/start`, { method: 'POST' });
  console.log('Started task', taskId);

  await waitForTask(taskId);
  console.log('\nTask completed');

  const summary = await api(`/tasks/${taskId}/output?name=output.json`);
  fs.writeFileSync(path.join(outDir, 'output.json'), JSON.stringify(summary, null, 2));
  console.log('Summary:', JSON.stringify(summary, null, 2));
  console.log(`Pollard p-1 artifacts saved to ${outDir}`);
}

main().catch((error) => {
  console.error('Error:', error.stack || error.message);
  process.exit(1);
});
