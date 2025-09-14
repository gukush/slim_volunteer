#!/usr/bin/env node
// Test script for exe-distributed-sort strategy (native CUDA binary execution)
// Based on test-distributed-sort.mjs and test-exe-block-matmul.mjs patterns

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Agent } from 'https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s => {
  const m = s.match(/^--([^=]+)=(.*)$/);
  return m ? [m[1], m[2]] : [s.replace(/^--/, ''), true];
}));

// Configuration
const host = args.host || 'https://localhost:3000';
const backend = args.backend || 'cuda'; // 'cuda' | 'opencl'
const binary = args.binary; // Path to native binary executable
const program = args.program || 'exe_distributed_sort'; // Program name hint for native client
const count = parseInt(args.count || '100000', 10); // Number of integers to sort
const chunkSize = parseInt(args.chunkSize || '65536', 10); // Integers per chunk
const ascending = args.descending ? false : true; // Sort direction
const Krep = parseInt(args.Krep || '1', 10); // Replication factor
const maxElements = args.maxElements ? parseInt(args.maxElements, 10) : undefined; // Optional limit
const useCache = !args.noCache; // Use cached files by default
const cacheDir = args.cacheDir || path.join(__dirname, '..', 'test-cache'); // Cache directory
const skipUpload = args.skipUpload; // Skip upload if cache file exists

// Validate backend
if (!['cuda', 'opencl'].includes(backend)) {
  console.error(`Invalid backend: ${backend}. Must be one of: cuda, opencl`);
  process.exit(1);
}

// Default binary paths if not specified
const defaultBinaries = {
  cuda: 'server/binaries/exe_distributed_sort',
  opencl: 'server/binaries/exe_distributed_sort_opencl'
};

const binaryPath = binary || defaultBinaries[backend];

console.log('=== Native Binary Distributed Sort Test ===');
console.log('Configuration:');
console.log(`  Backend: ${backend}`);
console.log(`  Binary: ${binaryPath}`);
console.log(`  Program: ${program}`);
console.log(`  Integers to sort: ${count}`);
console.log(`  Chunk size: ${chunkSize} integers`);
console.log(`  Sort direction: ${ascending ? 'ascending' : 'descending'}`);
console.log(`  K-replication: ${Krep}`);
if (maxElements) console.log(`  Max elements: ${maxElements}`);
console.log(`  Cache: ${useCache ? 'enabled' : 'disabled'}`);
console.log(`  Host: ${host}`);
console.log('');

// Integer generation and validation functions
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
  return `exe_sort_input_${count}${seedStr}.bin`;
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

// Helper to ignore self-signed certificates for localhost
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function robustFetch(url, opts = {}, retryCfg = {}) {
  const maxRetries = retryCfg.maxRetries || 3;
  const baseDelay = retryCfg.baseDelay || 1000;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const fetchOptions = host.startsWith('https://localhost')
        ? { agent: httpsAgent, ...opts }
        : opts;
      
      const response = await fetch(url, fetchOptions);
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
  console.log('Step 1: Generating test data...');
  const inputData = getOrGenerateInputData(count);
  const inputFile = path.join(__dirname, 'exe_sort_input.bin');

  fs.writeFileSync(inputFile, Buffer.from(inputData.buffer));
  console.log(`Input data prepared: ${count} integers`);

  console.log('\nStep 2: Computing CPU reference result...');
  const startCPU = Date.now();
  const reference = new Uint32Array(inputData);
  reference.sort((a, b) => ascending ? a - b : b - a);
  const cpuTime = Date.now() - startCPU;
  console.log(`  CPU sort took ${cpuTime}ms`);

  console.log('\nStep 3: Creating task with native binary strategy...');

  // Build configuration for native execution
  const config = {
    chunkSize,
    ascending,
    backend,        // cuda or opencl
    framework: backend,
    validateChunks: true, // Enable validation for testing
  };

  // Add binary path if specified
  if (binaryPath) {
    config.binary = binaryPath;
  }

  // Add program name for native client
  if (program) {
    config.program = program;
  }

  if (maxElements) {
    config.maxElements = maxElements;
  }

  // Create FormData for multipart upload
  const fd = new FormData();
  fd.append('strategyId', 'exe-distributed-sort'); // Strategy ID from exe_distributed_sort.js
  fd.append('K', String(Krep));
  fd.append('label', `exe-sort-test-${backend}`);
  fd.append('config', JSON.stringify(config));

  // Add input file only if not skipping upload
  if (!skipUpload) {
    const fileBlob = new Blob([fs.readFileSync(inputFile)], { type: 'application/octet-stream' });
    fd.append('large_sort_input.bin', fileBlob, 'large_sort_input.bin');
    console.log('  Uploading input file...');
  } else {
    console.log('  Skipping file upload (--skipUpload specified)');
  }

  let resp = await robustFetch(`${host}/tasks`, {
    method: 'POST',
    body: fd
  });

  if (!resp.ok) {
    console.error('Create task failed:', await resp.text());
    process.exit(1);
  }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log(`  Task created: ${taskId}`);
  console.log(`  Strategy: exe-distributed-sort`);
  console.log(`  Framework: exe`);

  console.log('\nStep 4: Starting task (will send binary to native client)...');
  resp = await robustFetch(`${host}/tasks/${taskId}/start`, {
    method: 'POST'
  });

  if (!resp.ok) {
    console.error('Start failed:', await resp.text());
    process.exit(1);
  }
  console.log('  Task started, binary and workload sent to native client');

  console.log('\nStep 5: Waiting for native execution to complete...');
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
    console.error(`\nTask failed with status: ${status}`);

    // Try to get error details
    try {
      const details = await robustFetch(`${host}/tasks/${taskId}`);
      const detailsJson = await details.json();
      if (detailsJson.error) {
        console.error(`Error details: ${detailsJson.error}`);
      }
    } catch {}

    process.exit(2);
  }

  console.log('  Native execution completed successfully!');

  console.log('\nStep 6: Downloading and validating results...');
  const out = await robustFetch(`${host}/tasks/${taskId}/output`);
  if (!out.ok) {
    console.error('Download failed:', await out.text());
    process.exit(3);
  }

  const resultBuffer = new Uint8Array(await out.arrayBuffer());
  const resultData = new Uint32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

  console.log(`  Downloaded result: ${resultData.length} integers`);

  console.log('\nStep 7: Validating results...');
  
  // Check length
  if (resultData.length !== inputData.length) {
    console.error(`⚠️ Length mismatch: expected ${inputData.length}, got ${resultData.length}`);
    process.exit(4);
  }

  // Check if sorted
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
    console.log('❌ VALIDATION FAILED - Elements do not match reference');
    process.exit(6);
  } else {
    const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
    const throughput = Math.round(count / parseFloat(totalTime));
    console.log(`✅ VALIDATION PASSED`);
    console.log(`Successfully sorted ${count} integers in ${totalTime}s (${throughput} int/s)`);
    console.log(`Backend: ${backend}, Chunks: ${chunkSize}, Direction: ${ascending ? 'ascending' : 'descending'}`);

    // Performance comparison
    const speedup = cpuTime / (parseFloat(totalTime) * 1000);
    console.log(`Performance: ${speedup.toFixed(2)}x vs single-threaded CPU`);

    // Clean up temporary file
    try {
      fs.unlinkSync(inputFile);
    } catch {}

    process.exit(0);
  }
}

// Show usage if help requested
if (args.help || args.h) {
  console.log(`
Usage: test-exe-distributed-sort.mjs [options]

Options:
  --host=URL            Server URL (default: https://localhost:3000)
  --backend=NAME        Backend type: cuda, opencl (default: cuda)
  --binary=PATH         Path to native binary executable
  --program=NAME        Program name hint for native client (default: exe_distributed_sort)
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
  # Basic test with CUDA backend
  ./test-exe-distributed-sort.mjs --backend=cuda --count=1000000

  # Test with OpenCL backend and custom binary
  ./test-exe-distributed-sort.mjs --backend=opencl --binary=./my_sort_binary

  # Test with cached data and no upload
  ./test-exe-distributed-sort.mjs --count=500000 --skipUpload

  # Test descending sort with small chunks
  ./test-exe-distributed-sort.mjs --count=100000 --chunkSize=16384 --descending
`);
  process.exit(0);
}

main().catch(e => {
  console.error('❌ Test failed with error:', e.message);
  if (args.verbose) {
    console.error(e.stack);
  }
  process.exit(99);
});
