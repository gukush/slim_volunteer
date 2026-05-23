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
  const disableWatchdog = Boolean(args.disableWatchdog || false);
  const Krep = Number(args.Krep ?? args.K ?? 1);
const timeoutMs = args.timeoutMs !== undefined ? Number(args.timeoutMs) : 0;
const intervalMs = Number(args.intervalMs ?? 1000);
const skipOutput = args.skipOutput || args.noOutput || false;
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
    if (timeoutMs > 0 && Date.now() - start > timeoutMs) throw new Error('Timeout waiting for Pollard p-1 task');
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

async function main() {
  fs.mkdirSync(outDir, { recursive: true });

  const input = { B1, startBase, totalBases, chunkSize, disableWatchdog };
  const config = { framework: 'webgpu', B1, startBase, totalBases, chunkSize, disableWatchdog };

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

  const taskStart = Date.now();
  await waitForTask(taskId);
  const totalMs = Date.now() - taskStart;
  console.log(`\nTask completed in ${(totalMs / 1000).toFixed(3)}s`);

  if (skipOutput) {
    console.log(`--skipOutput set, skipping output.json fetch. Task ${taskId} done.`);
    return;
  }

  // The task status is 'completed' but the server may still be finalizing.
  // Give it a generous window, then retry the API a few times.
  const maxRetries = 5;
  const retryDelayMs = 2000;
  let summary;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
    try {
      summary = await api(`/tasks/${taskId}/output?name=output.json`);
      console.log(`Fetched output.json on attempt ${attempt}`);
      break;
    } catch (e) {
      if (attempt === maxRetries) {
        console.error(`Failed to fetch output.json after ${maxRetries} attempts.`);
        console.error(`Task ${taskId} completed but output.json is missing from the API.`);
        console.error(`Last error: ${e.message}`);
        throw e;
      }
      console.log(`Attempt ${attempt} failed (${e.message}), retrying...`);
    }
  }
  fs.writeFileSync(path.join(outDir, 'output.json'), JSON.stringify(summary, null, 2));

  // Print found factors
  if (summary.results) {
    const foundLines = [];
    for (const r of summary.results) {
      if (r.found && r.factors && r.factors.length > 0) {
        for (const f of r.factors) {
          if (f.source === 'pollard-pminus1') {
            foundLines.push(`FOUND factor: N=${r.N} factor=${f.factorHex || f.factor} base=${f.base} chunk=${f.chunkIndex}`);
          }
        }
      }
    }

    const totalFound = foundLines.length;
    if (totalFound > 40) {
      for (let i = 0; i < 20; i++) console.log(foundLines[i]);
      console.log(`... (${totalFound - 40} more factors omitted) ...`);
      for (let i = totalFound - 20; i < totalFound; i++) console.log(foundLines[i]);
    } else {
      for (const line of foundLines) console.log(line);
    }
    console.log(`Total Pollard p-1 factors found: ${totalFound}`);
  }

  console.log('Summary:', JSON.stringify(summary, null, 2));
  console.log(`Pollard p-1 artifacts saved to ${outDir}`);
}

main().catch((error) => {
  console.error('Error:', error.stack || error.message);
  process.exit(1);
});
