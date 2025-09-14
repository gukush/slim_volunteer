#!/usr/bin/env node
// Test script for distributed-sort strategy

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s => {
  const m = s.match(/^--([^=]+)=(.*)$/);
  return m ? [m[1], m[2]] : [s.replace(/^--/, ''), true];
}));

const host = args.host || 'https://localhost:3000';
const framework = args.framework || 'webgpu';
const count = parseInt(args.count || '100000', 10); // Number of integers to sort
const chunkSize = parseInt(args.chunkSize || '65536', 10); // Integers per chunk
const ascending = args.descending ? false : true; // Sort direction
const Krep = parseInt(args.Krep || '1', 10); // Replication factor
const maxElements = args.maxElements ? parseInt(args.maxElements, 10) : undefined; // Optional limit on elements to process

function generateRandomIntegers(count) {
  const data = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    data[i] = Math.floor(Math.random() * 0xFFFFFFFF);
  }
  return data;
}

function isSorted(data, ascending = true) {
  for (let i = 1; i < data.length; i++) {
    if (ascending) {
      if (data[i] < data[i - 1]) return false;
    } else {
      if (data[i] > data[i - 1]) return false;
    }
  }
  return true;
}

async function main() {
  console.log(`Testing distributed sort with ${count} integers, chunkSize=${chunkSize}, ascending=${ascending}${maxElements ? `, maxElements=${maxElements}` : ''}`);

  // 1) Generate test data
  const inputData = generateRandomIntegers(count);
  const inputFile = path.join(__dirname, 'sort_input.bin');
  fs.writeFileSync(inputFile, Buffer.from(inputData.buffer));

  console.log(`Generated ${count} random integers`);

  // Create reference sorted array for validation
  const reference = new Uint32Array(inputData);
  reference.sort((a, b) => ascending ? a - b : b - a);

  // 2) Submit task
  const fd = new FormData();
  fd.append('strategyId', 'distributed-sort');
  fd.append('K', String(Krep));
  fd.append('label', 'distributed-sort-test');
  const config = {
    chunkSize,
    ascending,
    framework
  };
  
  if (maxElements) {
    config.maxElements = maxElements;
  }
  
  fd.append('config', JSON.stringify(config));
  fd.append('sort_input.bin', new Blob([fs.readFileSync(inputFile)]), 'sort_input.bin');

  let resp = await fetch(`${host}/tasks`, { method: 'POST', body: fd });
  if (!resp.ok) {
    console.error('Create task failed', await resp.text());
    process.exit(1);
  }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log('Created task', taskId);

  // 3) Start task
  resp = await fetch(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) {
    console.error('Start failed', await resp.text());
    process.exit(1);
  }

  // 4) Poll for completion
  let status;
  const startTime = Date.now();
  while (true) {
    await new Promise(r => setTimeout(r, 1000));
    const s = await fetch(`${host}/tasks/${taskId}`);
    const j = await s.json();
    status = j.status;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    process.stdout.write(`\r[${elapsed}s] status=${status} ${j.completedChunks || 0}/${j.totalChunks || '?'}   `);
    if (status === 'completed' || status === 'error' || status === 'canceled') break;
  }
  console.log();

  if (status !== 'completed') {
    console.error('Task did not complete:', status);
    process.exit(2);
  }

  // 5) Download and validate results
  const out = await fetch(`${host}/tasks/${taskId}/output`);
  if (!out.ok) {
    console.error('Download failed', await out.text());
    process.exit(3);
  }

  const resultBuffer = new Uint8Array(await out.arrayBuffer());
  const resultData = new Uint32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

  console.log(`Downloaded ${resultData.length} sorted integers`);

  // Validate results
  if (resultData.length !== inputData.length) {
    console.error(`Length mismatch: expected ${inputData.length}, got ${resultData.length}`);
    process.exit(4);
  }

  if (!isSorted(resultData, ascending)) {
    console.error('Output is not properly sorted!');

    // Find first unsorted position for debugging
    for (let i = 1; i < resultData.length; i++) {
      const curr = resultData[i];
      const prev = resultData[i - 1];
      const wrongOrder = ascending ? curr < prev : curr > prev;
      if (wrongOrder) {
        console.error(`First unsorted position: [${i-1}]=${prev}, [${i}]=${curr}`);
        break;
      }
    }
    process.exit(5);
  }

  // Verify we have the same elements (sorted reference vs our result)
  let elementsMismatch = false;
  for (let i = 0; i < count; i++) {
    if (reference[i] !== resultData[i]) {
      console.error(`Element mismatch at position ${i}: expected ${reference[i]}, got ${resultData[i]}`);
      elementsMismatch = true;
      break;
    }
  }
  if (elementsMismatch) {
    console.log('❌ FAIL - Elements do not match reference');
    process.exit(6);
  } else {
    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
    const throughput = Math.round(count / parseFloat(totalTime));
    console.log(`✅ PASS - Successfully sorted ${count} integers in ${totalTime}s (${throughput} int/s)`);
    process.exit(0);
  }
}

main().catch(e => {
  console.error(e);
  process.exit(99);
});
