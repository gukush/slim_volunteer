#!/usr/bin/env node
// Verification test for native-block-matmul-flex (native binary execution) strategy.
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
const backend = args.backend || 'opencl'; // 'opencl' | 'cuda' | 'vulkan'
const binary = args.binary; // Path to native binary executable
const program = args.program || 'block_matmul_native'; // Program name hint for native client
const N = parseInt(args.N||'64',10), K = parseInt(args.K||'64',10), M = parseInt(args.M||'64',10);
const TS = parseInt(args.tileSize||'32',10);
const KTS = parseInt(args.kTileSize||'256',10);
const Krep = parseInt(args.Krep||'1',10);
const chunkSize = parseInt(args.chunkSize || args.C || '8388608', 10); // 8MB default

// Validate backend
if (!['opencl', 'cuda', 'vulkan'].includes(backend)) {
  console.error(`Invalid backend: ${backend}. Must be one of: opencl, cuda, vulkan`);
  process.exit(1);
}

// Default binary paths if not specified
const defaultBinaries = {
  opencl: 'server/scripts/native/ocl_block_matmul_chunked',
  cuda: 'server/scripts/native/cuda_block_matmul',
  vulkan: 'server/scripts/native/vk_block_matmul'
};

const binaryPath = binary || defaultBinaries[backend];

// Check if binary exists (optional - server will check too)
if (binaryPath && !binaryPath.startsWith('server/')) {
  // Only check if it's a local file path, not a server-relative path
  if (!fs.existsSync(binaryPath)) {
    console.warn(`Warning: Binary not found at ${binaryPath}`);
    console.warn(`Server will look for it relative to its working directory`);
  }
}

console.log('=== Native Binary Execution Test ===');
console.log('Configuration:');
console.log(`  Backend: ${backend}`);
console.log(`  Binary: ${binaryPath}`);
console.log(`  Program: ${program}`);
console.log(`  Matrix dimensions: ${N}x${K} * ${K}x${M} = ${N}x${M}`);
console.log(`  Tile size: ${TS}x${TS}, K-tile: ${KTS}`);
console.log(`  Chunk size: ${chunkSize} elements (${(chunkSize*4/1024/1024).toFixed(2)} MB)`);
console.log(`  K-replication: ${Krep}`);
console.log(`  Host: ${host}`);
console.log('');

// Matrix generation and multiplication functions
function randMat(r,c){
  const a = new Float32Array(r*c);
  for(let i=0;i<a.length;i++) a[i] = (Math.random()*2-1);
  return a;
}

function matmulCPUf32(A,B,rows,kk,cols){
  const C = new Float32Array(rows*cols);
  const f = Math.fround;
  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      let acc = f(0);
      for (let k=0;k<kk;k++){
        acc = f( acc + f( f(A[r*kk+k]) * f(B[k*cols+c]) ) );
      }
      C[r*cols+c] = acc;
    }
  }
  return C;
}

// Helper to ignore self-signed certificates for localhost
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function main(){
  console.log('Step 1: Generating test matrices...');
  const A = randMat(N,K);
  const B = randMat(K,M);
  const tmpDir = '/tmp';
  const tmpA = path.join(tmpDir, `A_${Date.now()}.bin`);
  const tmpB = path.join(tmpDir, `B_${Date.now()}.bin`);
  fs.writeFileSync(tmpA, Buffer.from(A.buffer));
  fs.writeFileSync(tmpB, Buffer.from(B.buffer));
  console.log(`  Created A.bin: ${N}x${K} (${(N*K*4/1024/1024).toFixed(2)} MB)`);
  console.log(`  Created B.bin: ${K}x${M} (${(K*M*4/1024/1024).toFixed(2)} MB)`);

  console.log('\nStep 2: Computing CPU reference result...');
  const startCPU = Date.now();
  const Cref = matmulCPUf32(A,B,N,K,M);
  const cpuTime = Date.now() - startCPU;
  console.log(`  CPU computation took ${cpuTime}ms`);

  console.log('\nStep 3: Creating task with native binary strategy...');

  // Build configuration for native execution
  const config = {
    N, K, M,
    backend,        // opencl, cuda, or vulkan
    tileSize: TS,
    kTileSize: KTS,
    chunk_size: chunkSize,
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
  fd.append('strategyId', 'exe-block-matmul-flex'); // Strategy ID from exe-block-matmul-flex.js
  fd.append('K', String(Krep));
  fd.append('label', `native-binary-test`);
  fd.append('config', JSON.stringify(config));

  // Add matrix files
  const fileA = new Blob([fs.readFileSync(tmpA)], { type: 'application/octet-stream' });
  const fileB = new Blob([fs.readFileSync(tmpB)], { type: 'application/octet-stream' });
  fd.append('A.bin', fileA, 'A.bin');
  fd.append('B.bin', fileB, 'B.bin');

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
  console.log(`  Strategy: exe-block-matmul-flex`);
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
  const Cgpu = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength/4);
  console.log(`  Downloaded result: ${N}x${M} (${(N*M*4/1024/1024).toFixed(2)} MB)`);

  console.log('\nStep 7: Validating numerical accuracy...');
  const absTol = parseFloat(args.absTol || '1e-5');
  const relTol = parseFloat(args.relTol || '1e-3');
  let worst = { i:-1, a:0, b:0, abs:0, rel:0 };
  let ok = true;
  let maxAbs = 0, maxRel = 0;
  let numErrors = 0;

  for (let i=0;i<Cgpu.length;i++){
    const a = Cgpu[i], b = Cref[i];
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
    const row = Math.floor(worst.i / M);
    const col = worst.i % M;
    console.log(`\n  Worst error at [${row},${col}]:`);
    console.log(`    GPU result: ${worst.a}`);
    console.log(`    CPU result: ${worst.b}`);
    console.log(`    Absolute error: ${worst.abs.toExponential(3)}`);
    console.log(`    Relative error: ${worst.rel.toExponential(3)}`);
    console.log(`  Total errors: ${numErrors}/${Cgpu.length} elements`);
    console.log('\n VALIDATION FAILED');
    process.exit(4);
  } else {
    console.log('\n VALIDATION PASSED');
    console.log(`All ${Cgpu.length} elements within tolerance!`);

    // Performance estimate (rough)
    const flops = 2.0 * N * M * K; // 2 ops per multiply-add
    const gflops = flops / 1e9;
    console.log(`\nPerformance estimate:`);
    console.log(`  Total FLOPs: ${gflops.toFixed(2)} GFLOPs`);
    console.log(`  CPU time: ${cpuTime}ms (${(gflops/(cpuTime/1000)).toFixed(2)} GFLOPS)`);

    // Clean up temp files
    fs.unlinkSync(tmpA);
    fs.unlinkSync(tmpB);

    process.exit(0);
  }
}

main().catch(e=>{
  console.error('\n Fatal error:', e.message);
  if (args.verbose) {
    console.error(e.stack);
  }
  process.exit(99);
});