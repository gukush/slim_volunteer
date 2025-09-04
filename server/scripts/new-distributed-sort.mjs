#!/usr/bin/env node
// Enhanced test script for large distributed-sort with external merging

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import https from 'node:https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s => {
  const m = s.match(/^--([^=]+)=(.*)$/);
  return m ? [m[1], m[2]] : [s.replace(/^--/, ''), true];
}));

// Simple sleep
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const httpsAgent = new https.Agent({ keepAlive: true, maxSockets: 64 });
const host = args.host || 'https://localhost:3000';
const framework = args.framework || 'webgpu';
const count = parseInt(args.count || '10000000', 10); // Default 10M integers (40MB)
const chunkSize = parseInt(args.chunkSize || '65536', 10); // Integers per chunk
const memoryLimitMB = parseInt(args.memoryLimit || '100', 10); // Memory limit for external sort
const ascending = args.descending ? false : true;
const Krep = parseInt(args.Krep || '1', 10); // Replication factor
const validateResult = !args.skipValidation; // Skip validation for very large datasets

function formatSize(bytes) {
  const mb = bytes / 1024 / 1024;
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${mb.toFixed(1)} MB`;
}

function generateRandomIntegers(count) {
  console.log(`Generating ${count.toLocaleString()} random integers (${formatSize(count * 4)})`);
  const data = new Uint32Array(count);
  const batchSize = 1000000; // Generate in 1M batches to avoid blocking

  for (let start = 0; start < count; start += batchSize) {
    const end = Math.min(start + batchSize, count);
    for (let i = start; i < end; i++) {
      data[i] = Math.floor(Math.random() * 0xFFFFFFFF);
    }

    // Progress indicator for large datasets
    if (end % (batchSize * 10) === 0 || end === count) {
      const pct = (end / count * 100).toFixed(1);
      process.stdout.write(`\rGenerating data: ${pct}%`);
    }
  }
  console.log(); // New line after progress
  return data;
}

function isSorted(data, ascending = true, sampleSize = 1000000) {
  // For very large datasets, sample validation to avoid long verification times
  const checkSize = Math.min(data.length, sampleSize);
  const step = Math.max(1, Math.floor(data.length / checkSize));

  console.log(`Validating sort order (sampling every ${step} elements)`);

  for (let i = step; i < data.length; i += step) {
    const prev = data[i - step];
    const curr = data[i];

    if (ascending) {
      if (curr < prev) {
        console.error(`Sort validation failed at positions ${i-step},${i}: ${prev} > ${curr}`);
        return false;
      }
    } else {
      if (curr > prev) {
        console.error(`Sort validation failed at positions ${i-step},${i}: ${prev} < ${curr}`);
        return false;
      }
    }
  }
  return true;
}

function createProgressTracker(startTime) {
  let lastUpdate = 0;
  return (current, total, status) => {
    const now = Date.now();
    if (now - lastUpdate > 2000) { // Update every 2 seconds
      const pct = total ? (current / total * 100).toFixed(1) : '?';
      const elapsed = ((now - startTime) / 1000).toFixed(0);
      process.stdout.write(`\r[${elapsed}s] ${status} ${current.toLocaleString()}/${total?.toLocaleString() || '?'} (${pct}%)   `);
      lastUpdate = now;
    }
  };
}

async function robustFetch(url, opts = {}, retryCfg = {}) {
  const {
    retries = 6,           // ~6 attempts
    baseDelay = 400,       // base backoff in ms
    maxDelay = 5000,       // cap backoff
    timeoutMs = 15000      // per-request timeout
  } = retryCfg;

  let attempt = 0, lastErr;
  while (attempt <= retries) {
    const controller = new AbortController();
    const to = setTimeout(() => controller.abort(new Error('Fetch timeout')), timeoutMs);
    try {
      const res = await fetch(url, { agent: httpsAgent, ...opts, signal: controller.signal });
      clearTimeout(to);
      // Retry 5xx and 429/408/502/503/504
      if (!res.ok && [408, 429, 500, 502, 503, 504].includes(res.status)) {
        throw Object.assign(new Error(`HTTP ${res.status}`), { code: `HTTP_${res.status}` });
      }
      return res;
    } catch (err) {
      clearTimeout(to);
      lastErr = err;
      const code = err?.code || err?.cause?.code;
      const retryable =
        ['ECONNRESET', 'ETIMEDOUT', 'EAI_AGAIN', 'UND_ERR_SOCKET',
         'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT'].includes(code)
        || err?.name === 'AbortError'
        || /timeout/i.test(err?.message || '')
        || /network|fetch failed/i.test(err?.message || '');
      if (!retryable || attempt === retries) break;
      const backoff = Math.min(maxDelay, baseDelay * 2 ** attempt) * (0.7 + Math.random() * 0.6);
      await sleep(backoff);
      attempt++;
    }
  }
  throw lastErr;
}

async function main() {
  const dataSize = formatSize(count * 4);
  const memoryEfficiency = ((count * 4 / 1024 / 1024) / memoryLimitMB).toFixed(1);

  console.log(`Testing external distributed sort:`);
  console.log(`  Dataset: ${count.toLocaleString()} integers (${dataSize})`);
  console.log(`  Memory limit: ${memoryLimitMB} MB`);
  console.log(`  Memory efficiency: ${memoryEfficiency}x dataset size`);
  console.log(`  Chunk size: ${chunkSize.toLocaleString()} integers`);
  console.log(`  Sort direction: ${ascending ? 'ascending' : 'descending'}`);
  console.log();

  // 1) Generate test data
  const inputData = generateRandomIntegers(count);
  const inputFile = path.join(__dirname, 'large_sort_input.bin');

  console.log('Writing input file...');
  fs.writeFileSync(inputFile, Buffer.from(inputData.buffer));

  // Create reference for small datasets only
  let reference = null;
  if (validateResult && count <= 1000000) {
    console.log('Creating reference sorted array for validation...');
    reference = new Uint32Array(inputData);
    reference.sort((a, b) => ascending ? a - b : b - a);
  }

  // 2) Submit task with external merge configuration
  console.log('Submitting distributed sort task...');

  const uploadToken = `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  const fd = new FormData();
  fd.append('strategyId', 'distributed-sort');
  fd.append('K', String(Krep));
  fd.append('label', 'large-distributed-sort-test');
  fd.append('config', JSON.stringify({
    uploadToken: uploadToken,
    chunkSize,
    ascending,
    framework,
    memoryThresholdMB: memoryLimitMB,
    maxRunsBeforeMerge: 8,
    runMergeSize: 4
  }));
  fd.append('large_sort_input.bin', new Blob([fs.readFileSync(inputFile)]), `${uploadToken}_large_sort_input.bin`);

  let resp = await fetch(`${host}/tasks`, { method: 'POST', body: fd });
  if (!resp.ok) {
    console.error('Create task failed', await resp.text());
    process.exit(1);
  }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log(`Created task: ${taskId}`);

  // 3) Start task
  resp = await fetch(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) {
    console.error('Start failed', await resp.text());
    process.exit(1);
  }

  // 4) Poll for completion with progress tracking
  let status;
  const startTime = Date.now();
  const tracker = createProgressTracker(startTime);

  console.log('Monitoring task progress...');

  while (true) {
    await new Promise(r => setTimeout(r, 2000)); // Poll every 2 seconds

    const s = await fetch(`${host}/tasks/${taskId}`);
    const j = await s.json();
    status = j.status;

    tracker(j.completedChunks || 0, j.totalChunks, status);

    if (status === 'completed' || status === 'error' || status === 'canceled') break;
  }
  console.log(); // New line after progress

  if (status !== 'completed') {
    console.error('Task did not complete:', status);
    process.exit(2);
  }

  const sortTime = (Date.now() - startTime) / 1000;
  const throughput = Math.round(count / sortTime);

  console.log(`Sort completed in ${sortTime.toFixed(1)}s (${throughput.toLocaleString()} integers/sec)`);

  // 5) Download and validate results
  console.log('Downloading results...');
  const out = await fetch(`${host}/tasks/${taskId}/output`);
  if (!out.ok) {
    console.error('Download failed', await out.text());
    process.exit(3);
  }

  const resultBuffer = new Uint8Array(await out.arrayBuffer());
  const resultData = new Uint32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

  console.log(`Downloaded ${resultData.length.toLocaleString()} sorted integers (${formatSize(resultData.byteLength)})`);

  // Validate results
  if (resultData.length !== inputData.length) {
    console.error(`Length mismatch: expected ${inputData.length.toLocaleString()}, got ${resultData.length.toLocaleString()}`);
    process.exit(4);
  }

  if (validateResult) {
    console.log('Validating sort order...');
    if (!isSorted(resultData, ascending)) {
      console.error('Output is not properly sorted!');
      process.exit(5);
    }

    // For small datasets, verify elements match reference
    if (reference) {
      console.log('Validating element correctness...');
      let elementsMismatch = false;
      const checkCount = Math.min(100000, count); // Sample validation for large datasets
      const step = Math.max(1, Math.floor(count / checkCount));

      for (let i = 0; i < count; i += step) {
        if (reference[i] !== resultData[i]) {
          console.error(`Element mismatch at position ${i}: expected ${reference[i]}, got ${resultData[i]}`);
          elementsMismatch = true;
          break;
        }
      }

      if (elementsMismatch) {
        console.log('✗ FAIL - Elements do not match reference');
        process.exit(6);
      }
    }
  }

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
  const finalThroughput = Math.round(count / parseFloat(totalTime));
  const memoryEfficiencyActual = memoryLimitMB / (count * 4 / 1024 / 1024);

  console.log('\n✅ SUCCESS - External distributed sort completed');
  console.log(`📊 Performance Summary:`);
  console.log(`   Dataset: ${count.toLocaleString()} integers (${dataSize})`);
  console.log(`   Total time: ${totalTime}s`);
  console.log(`   Throughput: ${finalThroughput.toLocaleString()} integers/sec`);
  console.log(`   Memory efficiency: ${(memoryEfficiencyActual * 100).toFixed(1)}% of dataset size`);
  console.log(`   Validation: ${validateResult ? 'PASSED' : 'SKIPPED'}`);

  // Cleanup
  try {
    fs.unlinkSync(inputFile);
    console.log('Cleaned up temporary input file');
  } catch (e) {
    // Ignore cleanup errors
  }

  process.exit(0);
}

main().catch(e => {
  console.error('Test failed:', e);
  process.exit(99);
});