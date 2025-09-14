#!/usr/bin/env node
// Test script for native-distributed-sort strategy (CUDA implementation)
// Based on test-distributed-sort.mjs with native client support and cached file functionality

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
const framework = 'native-cuda'; // Force native CUDA for this test
const count = parseInt(args.count || '100000', 10); // Number of integers to sort
const chunkSize = parseInt(args.chunkSize || '65536', 10); // Integers per chunk
const ascending = args.descending ? false : true; // Sort direction
const Krep = parseInt(args.Krep || '1', 10); // Replication factor
const maxElements = args.maxElements ? parseInt(args.maxElements, 10) : undefined; // Optional limit on elements to process
const useCache = !args.noCache; // Use cached files by default
const cacheDir = args.cacheDir || path.join(__dirname, '..', 'test-cache'); // Cache directory
const skipUpload = args.skipUpload; // Skip upload if cache file exists

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

// Create cache directory if it doesn't exist
function ensureCacheDir() {
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
    console.log(`Created cache directory: ${cacheDir}`);
  }
}

// Generate cache filename based on parameters
function getCacheFileName(count, seed = null) {
  const seedStr = seed ? `_seed${seed}` : '';
  return `sort_input_${count}${seedStr}.bin`;
}

// Try to load from cache, generate if not found
function getOrGenerateInputData(count) {
  if (!useCache) {
    console.log('Cache disabled, generating fresh data');
    return generateRandomIntegers(count);
  }

  ensureCacheDir();
  const cacheFile = path.join(cacheDir, getCacheFileName(count));

  if (fs.existsSync(cacheFile)) {
    console.log(`Loading cached input data from: ${cacheFile}`);
    try {
      const buffer = fs.readFileSync(cacheFile);
      if (buffer.length === count * 4) {
        return new Uint32Array(buffer.buffer, buffer.byteOffset, count);
      } else {
        console.warn(`Cache file size mismatch, regenerating. Expected: ${count * 4}, got: ${buffer.length}`);
      }
    } catch (error) {
      console.warn(`Failed to read cache file: ${error.message}, regenerating`);
    }
  }

  console.log('Generating new input data and caching...');
  const data = generateRandomIntegers(count);

  try {
    fs.writeFileSync(cacheFile, Buffer.from(data.buffer));
    console.log(`Cached input data to: ${cacheFile}`);
  } catch (error) {
    console.warn(`Failed to cache input data: ${error.message}`);
  }

  return data;
}

async function robustFetch(url, opts = {}, retryCfg = {}) {
  const maxRetries = retryCfg.maxRetries || 3;
  const baseDelay = retryCfg.baseDelay || 1000;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, opts);
      return response;
    } catch (error) {
      if (attempt === maxRetries) throw error;

      const delay = baseDelay * Math.pow(2, attempt);
      console.log(`Fetch attempt ${attempt + 1} failed, retrying in ${delay}ms: ${error.message}`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

async function main() {
  console.log(`Testing native CUDA distributed sort with ${count} integers, chunkSize=${chunkSize}, ascending=${ascending}${maxElements ? `, maxElements=${maxElements}` : ''}`);
  console.log(`Framework: ${framework}, Cache: ${useCache ? 'enabled' : 'disabled'}`);

  // 1) Get or generate test data
  const inputData = getOrGenerateInputData(count);
  const inputFile = path.join(__dirname, 'native_sort_input.bin');

  // Write to temporary file for upload
  fs.writeFileSync(inputFile, Buffer.from(inputData.buffer));
  console.log(`Input data prepared: ${count} integers`);

  // Create reference sorted array for validation
  const reference = new Uint32Array(inputData);
  reference.sort((a, b) => ascending ? a - b : b - a);

  // 2) Check if we should skip upload due to cached server-side file
  if (skipUpload) {
    console.log('Skipping file upload (--skipUpload specified)');
  }

  // 3) Submit task
  const fd = new FormData();
  fd.append('strategyId', 'native-distributed-sort');
  fd.append('K', String(Krep));
  fd.append('label', `native-cuda-sort-test-${count}`);

  const config = {
    chunkSize,
    ascending,
    framework: framework,
    // Add native-specific configuration if needed
    validateChunks: true, // Enable validation for testing
  };

  if (maxElements) {
    config.maxElements = maxElements;
  }

  fd.append('config', JSON.stringify(config));

  // Only append file if not skipping upload
  if (!skipUpload) {
    fd.append('large_sort_input.bin', new Blob([fs.readFileSync(inputFile)]), 'large_sort_input.bin');
    console.log('Uploading input file...');
  }

  let resp = await robustFetch(`${host}/tasks`, { method: 'POST', body: fd });
  if (!resp.ok) {
    const errorText = await resp.text();
    console.error('Create task failed:', errorText);
    process.exit(1);
  }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log('Created task:', taskId);

  // 4) Start task
  resp = await robustFetch(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) {
    const errorText = await resp.text();
    console.error('Start failed:', errorText);
    process.exit(1);
  }
  console.log('Task started');

  // 5) Poll for completion
  let status;
  const startTime = Date.now();
  let lastChunkCount = 0;

  while (true) {
    await new Promise(r => setTimeout(r, 2000)); // Poll every 2 seconds for native tasks

    try {
      const s = await robustFetch(`${host}/tasks/${taskId}`);
      const j = await s.json();
      status = j.status;
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const completed = j.completedChunks || 0;
      const total = j.totalChunks || '?';

      // Show progress updates
      if (completed !== lastChunkCount) {
        console.log(`[${elapsed}s] Progress: ${completed}/${total} chunks completed`);
        lastChunkCount = completed;
      }

      process.stdout.write(`\r[${elapsed}s] status=${status} ${completed}/${total}   `);

      if (status === 'completed' || status === 'error' || status === 'canceled') break;
    } catch (error) {
      console.warn(`Polling error: ${error.message}`);
    }
  }
  console.log();

  if (status !== 'completed') {
    console.error('Task did not complete successfully:', status);

    // Try to get more details about the error
    try {
      const details = await robustFetch(`${host}/tasks/${taskId}`);
      const detailsJson = await details.json();
      if (detailsJson.error) {
        console.error('Error details:', detailsJson.error);
      }
    } catch {}

    process.exit(2);
  }

  // 6) Download and validate results
  console.log('Downloading results...');
  const out = await robustFetch(`${host}/tasks/${taskId}/output`);
  if (!out.ok) {
    const errorText = await out.text();
    console.error('Download failed:', errorText);
    process.exit(3);
  }

  const resultBuffer = new Uint8Array(await out.arrayBuffer());
  const resultData = new Uint32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

  console.log(`Downloaded ${resultData.length} sorted integers`);

  // 7) Validate results
  if (resultData.length !== inputData.length) {
    console.error(`❌ Length mismatch: expected ${inputData.length}, got ${resultData.length}`);
    process.exit(4);
  }

  if (!isSorted(resultData, ascending)) {
    console.error('❌ Output is not properly sorted!');

    // Find first unsorted position for debugging
    for (let i = 1; i < resultData.length; i++) {
      const curr = resultData[i];
      const prev = resultData[i - 1];
      const wrongOrder = ascending ? curr < prev : curr > prev;
      if (wrongOrder) {
        console.error(`First unsorted position: [${i-1}]=${prev}, [${i}]=${curr}`);
        // Show context around the error
        const start = Math.max(0, i - 3);
        const end = Math.min(resultData.length, i + 3);
        console.error(`Context: ${Array.from(resultData.slice(start, end))}`);
        break;
      }
    }
    process.exit(5);
  }

  // Verify we have the same elements (sorted reference vs our result)
  let elementsMismatch = false;
  for (let i = 0; i < count; i++) {
    if (reference[i] !== resultData[i]) {
      console.error(`❌ Element mismatch at position ${i}: expected ${reference[i]}, got ${resultData[i]}`);
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
    console.log(`Framework: ${framework}, Chunks: ${chunkSize}, Direction: ${ascending ? 'ascending' : 'descending'}`);

    // Cleanup temporary file
    try {
      fs.unlinkSync(inputFile);
    } catch {}

    process.exit(0);
  }
}

// Show usage if help requested
if (args.help || args.h) {
  console.log(`
Usage: test-native-distributed-sort.mjs [options]

Options:
  --host=URL            Server URL (default: https://localhost:3000)
  --count=N             Number of integers to sort (default: 100000)
  --chunkSize=N         Integers per chunk (default: 65536)
  --descending          Sort in descending order (default: ascending)
  --Krep=N              Replication factor (default: 1)
  --maxElements=N       Limit processing to N elements
  --noCache             Disable input data caching
  --cacheDir=PATH       Cache directory (default: ../test-cache)
  --skipUpload          Skip file upload (assumes server has cached file)
  --help, -h            Show this help

Example:
  # Basic test with 1M integers
  ./test-native-distributed-sort.mjs --count=1000000

  # Test with cached data and no upload
  ./test-native-distributed-sort.mjs --count=500000 --skipUpload

  # Test descending sort with small chunks
  ./test-native-distributed-sort.mjs --count=100000 --chunkSize=16384 --descending
`);
  process.exit(0);
}

main().catch(e => {
  console.error('❌ Test failed with error:', e.message);
  console.error(e.stack);
  process.exit(99);
});

