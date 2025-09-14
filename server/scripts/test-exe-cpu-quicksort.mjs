#!/usr/bin/env node
// Verification test for exe-cpu-quicksort (native binary execution) strategy.
// This tests the native binary upload and execution path via WebSocket connection.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Agent } from 'https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

// Configuration
const host = args.host || 'https://localhost:3000';
const binary = args.binary; // Path to native binary executable
const program = args.program || 'cpu-quicksort'; // Program name hint for native client
const N = parseInt(args.N||'1000000',10); // Number of integers to sort
const chunkSize = parseInt(args.chunkSize || args.C || '65536', 10); // 64K integers per chunk
const ascending = args.ascending !== 'false'; // Sort direction
const maxElements = parseInt(args.maxElements || '0', 10); // Limit elements (0 = no limit)

// Default binary path if not specified
const defaultBinary = 'server/executors/binary/cpu-quicksort';
const binaryPath = binary || defaultBinary;

// Check if binary exists (optional - server will check too)
if (binaryPath && !binaryPath.startsWith('server/')) {
  // Only check if it's a local file path, not a server-relative path
  if (!fs.existsSync(binaryPath)) {
    console.warn(`Warning: Binary not found at ${binaryPath}`);
    console.warn(`Server will look for it relative to its working directory`);
  }
}

console.log('=== Native CPU Quicksort Binary Execution Test ===');
console.log('Configuration:');
console.log(`  Binary: ${binaryPath}`);
console.log(`  Program: ${program}`);
console.log(`  Number of integers: ${N}`);
console.log(`  Chunk size: ${chunkSize} elements (${(chunkSize*4/1024/1024).toFixed(2)} MB)`);
console.log(`  Ascending: ${ascending}`);
console.log(`  Max elements: ${maxElements || 'unlimited'}`);
console.log(`  Host: ${host}`);
console.log('');

// Integer generation and sorting functions
function generateRandomIntegers(count) {
  const data = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    data[i] = Math.floor(Math.random() * 0xFFFFFFFF);
  }
  return data;
}

function sortIntegersCPU(data, ascending = true) {
  const sorted = new Uint32Array(data);
  if (ascending) {
    sorted.sort((a, b) => a - b);
  } else {
    sorted.sort((a, b) => b - a);
  }
  return sorted;
}

// Helper to ignore self-signed certificates for localhost
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function main(){
  console.log('Step 1: Generating test data...');
  const testData = generateRandomIntegers(N);
  const tmpDir = '/tmp';
  const tmpInput = path.join(tmpDir, `input_${Date.now()}.bin`);
  fs.writeFileSync(tmpInput, Buffer.from(testData.buffer));
  console.log(`  Created input.bin: ${N} integers (${(N*4/1024/1024).toFixed(2)} MB)`);

  console.log('\nStep 2: Computing CPU reference result...');
  const startCPU = Date.now();
  const refResult = sortIntegersCPU(testData, ascending);
  const cpuTime = Date.now() - startCPU;
  console.log(`  CPU reference sort took ${cpuTime}ms`);

  console.log('\nStep 3: Creating task with native binary strategy...');

  // Build configuration for native execution
  const config = {
    chunk_size: chunkSize,
    ascending: ascending,
    maxElements: maxElements || undefined,
    framework: 'exe', // Updated to match exe framework
  };

  // Add binary path if specified
  if (binaryPath) {
    config.binary = binaryPath;
  }

  // Add program name for native client
  if (program) {
    config.program = program;
  }

  // Create FormData for multipart upload
  const fd = new FormData();
  fd.append('strategyId', 'exe-cpu-quicksort'); // Strategy ID from exe-cpu-quicksort.js
  fd.append('K', '1'); // Single replication for CPU sort
  fd.append('label', `cpu-quicksort-test`);
  fd.append('config', JSON.stringify(config));

  // Add input file
  const fileInput = new Blob([fs.readFileSync(tmpInput)], { type: 'application/octet-stream' });
  fd.append('input.bin', fileInput, 'input.bin');

  const fetchOptions = host.startsWith('https://localhost')
    ? { agent: httpsAgent }
    : {};

  let resp = await fetch(`${host}/tasks`, {
    method: 'POST',
    body: fd,
    ...fetchOptions
  });

  if(!resp.ok){
    console.error('Create task failed:', await resp.text());
    process.exit(1);
  }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log(`  Task created: ${taskId}`);
  console.log(`  Strategy: exe-cpu-quicksort`);
  console.log(`  Framework: binary`);

  console.log('\nStep 4: Starting task (will send binary to native client)...');
  resp = await fetch(`${host}/tasks/${taskId}/start`, {
    method: 'POST',
    ...fetchOptions
  });

  if(!resp.ok){
    console.error('Start failed:', await resp.text());
    process.exit(1);
  }
  console.log('  Task started, binary and workload sent to native client');

  console.log('\nStep 5: Waiting for native execution to complete...');
  let status;
  let lastUpdate = Date.now();
  let totalChunks = '?';

  while(true){
    await new Promise(r=>setTimeout(r, 1000));
    const s = await fetch(`${host}/tasks/${taskId}`, fetchOptions);
    const j = await s.json();
    status = j.status;

    if (j.totalChunks) totalChunks = j.totalChunks;

    const elapsed = Math.floor((Date.now() - lastUpdate) / 1000);
    process.stdout.write(`\r  Status: ${status} | Progress: ${j.completedChunks||0}/${totalChunks} chunks | Time: ${elapsed}s   `);

    if(status==='completed' || status==='error' || status==='canceled') break;
  }
  console.log('');

  if(status!=='completed'){
    console.error(`\nTask failed with status: ${status}`);

    // Try to get error details
    const taskInfo = await fetch(`${host}/tasks/${taskId}`, fetchOptions);
    const details = await taskInfo.json();
    if (details.error) {
      console.error(`Error details: ${details.error}`);
    }

    process.exit(2);
  }

  console.log('  Native execution completed successfully!');

  console.log('\nStep 6: Downloading and validating results...');
  const out = await fetch(`${host}/tasks/${taskId}/output`, fetchOptions);
  if(!out.ok){
    console.error('Download failed:', await out.text());
    process.exit(3);
  }

  const buf = new Uint8Array(await out.arrayBuffer());
  const resultData = new Uint32Array(buf.buffer, buf.byteOffset, buf.byteLength/4);
  console.log(`  Downloaded result: ${resultData.length} integers (${(resultData.length*4/1024/1024).toFixed(2)} MB)`);

  console.log('\nStep 7: Validating sort correctness...');
  let isSorted = true;
  let errors = 0;
  const maxErrors = 10;

  for (let i = 1; i < resultData.length; i++) {
    const valid = ascending ? (resultData[i-1] <= resultData[i]) : (resultData[i-1] >= resultData[i]);
    if (!valid) {
      isSorted = false;
      if (errors < maxErrors) {
        console.log(`  Sort error at position ${i}: ${resultData[i-1]} ${ascending ? '>' : '<'} ${resultData[i]}`);
        errors++;
      }
    }
  }

  if (!isSorted) {
    console.log(`\n❌ SORT VALIDATION FAILED`);
    console.log(`  Found ${errors} errors (showing first ${maxErrors})`);
    process.exit(4);
  }

  console.log('\nStep 8: Validating numerical accuracy against reference...');
  const absTol = parseFloat(args.absTol || '0');
  const relTol = parseFloat(args.relTol || '0');
  let worst = { i:-1, a:0, b:0, abs:0, rel:0 };
  let ok = true;
  let maxAbs = 0, maxRel = 0;
  let numErrors = 0;

  const compareLength = Math.min(resultData.length, refResult.length);
  for (let i=0; i<compareLength; i++){
    const a = resultData[i], b = refResult[i];
    const abs = Math.abs(a-b);
    const rel = abs / Math.max(Math.abs(b), 1e-6);  // stabilized rel
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    if (abs > worst.abs) worst = { i, a, b, abs, rel };
    // pass if abs is small OR proportional to ref
    if (abs > absTol + relTol * Math.abs(b)) {
      ok = false;
      numErrors++;
    }
  }

  console.log(`  Max absolute error: ${maxAbs.toExponential(3)}`);
  console.log(`  Max relative error: ${maxRel.toExponential(3)}`);
  console.log(`  Tolerance: abs=${absTol}, rel=${relTol}`);

  if (!ok) {
    const row = Math.floor(worst.i / 1000); // Assuming some row width for display
    const col = worst.i % 1000;
    console.log(`\n  Worst error at position ${worst.i}:`);
    console.log(`    Result: ${worst.a}`);
    console.log(`    Reference: ${worst.b}`);
    console.log(`    Absolute error: ${worst.abs.toExponential(3)}`);
    console.log(`    Relative error: ${worst.rel.toExponential(3)}`);
    console.log(`  Total errors: ${numErrors}/${compareLength} elements`);
    console.log('\n❌ VALIDATION FAILED');
    process.exit(4);
  } else {
    console.log('\n✅ VALIDATION PASSED');
    console.log(`All ${compareLength} elements match reference!`);

    // Performance estimate
    const elementsPerSecond = (resultData.length / (cpuTime / 1000)).toFixed(0);
    console.log(`\nPerformance estimate:`);
    console.log(`  Elements sorted: ${resultData.length.toLocaleString()}`);
    console.log(`  CPU time: ${cpuTime}ms`);
    console.log(`  Rate: ${elementsPerSecond.toLocaleString()} elements/second`);

    // Clean up temp files
    fs.unlinkSync(tmpInput);

    process.exit(0);
  }
}

main().catch(e=>{
  console.error('\n❌ Fatal error:', e.message);
  if (args.verbose) {
    console.error(e.stack);
  }
  process.exit(99);
});
