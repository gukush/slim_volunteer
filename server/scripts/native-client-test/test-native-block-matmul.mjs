#!/usr/bin/env node
// Test script for native block matrix multiplication using min_native_client2
// Tests OpenCL, Vulkan, and CUDA frameworks with LuaJIT host script

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
const framework = args.framework || 'cuda'; // 'cuda' | 'opencl' | 'vulkan'
const N = parseInt(args.N||'64',10), K = parseInt(args.K||'64',10), M = parseInt(args.M||'64',10);
const TS = parseInt(args.tileSize||'32',10);
const KTS = parseInt(args.kTileSize||'256',10);
const Krep = parseInt(args.Krep||'1',10);
const chunkSize = parseInt(args.chunkSize || args.C || '8388608', 10); // 8MB default

// Validate framework
if (!['cuda', 'opencl', 'vulkan'].includes(framework)) {
  console.error(`Invalid framework: ${framework}. Must be one of: cuda, opencl, vulkan`);
  process.exit(1);
}

console.log('=== Native Block Matrix Multiplication Test ===');
console.log(`Framework: ${framework}`);
console.log(`Matrix dimensions: ${N}x${K} * ${K}x${M} = ${N}x${M}`);
console.log(`Tile size: ${TS}, K tile size: ${KTS}`);
console.log(`Chunk size: ${chunkSize} elements`);

// Helper functions
function randMat(r,c){
  const a = new Float32Array(r*c);
  for(let i=0;i<a.length;i++) a[i] = (Math.random()*2-1);
  return a;
}

function matmulCPUf32(A,B,rows,kk,cols){
  const C = new Float32Array(rows*cols);
  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      let acc = 0;
      for (let k=0;k<kk;k++){
        acc += A[r*kk+k] * B[k*cols+c];
      }
      C[r*cols+c] = acc;
    }
  }
  return C;
}

// Helper to ignore self-signed certificates for localhost
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function main(){
  console.log('\nStep 1: Generating test matrices...');
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

  console.log('\nStep 3: Loading Lua host script...');
  const luaScriptPath = path.join(__dirname, '../../executors/host_block_matmul.lua');
  if (!fs.existsSync(luaScriptPath)) {
    console.error(`Lua script not found: ${luaScriptPath}`);
    process.exit(1);
  }
  const luaScript = fs.readFileSync(luaScriptPath, 'utf8');
  console.log(`  Loaded Lua script (${luaScript.length} bytes)`);

  console.log('\nStep 4: Creating task with native framework strategy...');

  // Build configuration for native execution
  const config = {
    N, K, M,
    framework,        // cuda, opencl, or vulkan
    tileSize: TS,
    kTileSize: KTS,
    chunk_size: chunkSize,
    backend: framework, // Must match expected framework
  };

  // Create FormData for multipart upload
  const fd = new FormData();
  fd.append('strategyId', 'native-block-matmul-flex'); // Strategy ID from exe-block-matmul-flex.js
  fd.append('K', String(Krep));
  fd.append('label', `native-${framework}-test`);
  fd.append('config', JSON.stringify(config));

  // Add matrix files
  const fileA = new Blob([fs.readFileSync(tmpA)], { type: 'application/octet-stream' });
  const fileB = new Blob([fs.readFileSync(tmpB)], { type: 'application/octet-stream' });
  fd.append('A.bin', fileA, 'A.bin');
  fd.append('B.bin', fileB, 'B.bin');

  // Add Lua host script as artifact
  const luaBlob = new Blob([luaScript], { type: 'text/plain' });
  fd.append('artifacts', luaBlob, 'host.lua');

  const fetchOptions = host.startsWith('https://localhost')
    ? { agent: httpsAgent }
    : {};

  let resp = await fetch(`${host}/tasks`, {
    method: 'POST',
    body: fd,
    ...fetchOptions
  });

  if(!resp.ok){
    console.error('Create task failed', await resp.text());
    process.exit(1);
  }
  const desc = await resp.json();
  const taskId = desc.id;
  console.log(`  Created task ${taskId}`);

  // Step 5: Start task
  console.log('\nStep 5: Starting task...');
  resp = await fetch(`${host}/tasks/${taskId}/start`, {
    method: 'POST',
    ...fetchOptions
  });
  if(!resp.ok){
    console.error('Start failed', await resp.text());
    process.exit(1);
  }
  console.log('  Task started');

  // Step 6: Monitor progress
  console.log('\nStep 6: Monitoring task progress...');
  let status;
  let startTime = Date.now();
  while(true){
    await new Promise(r=>setTimeout(r, 1000));
    resp = await fetch(`${host}/tasks/${taskId}`, fetchOptions);
    if(!resp.ok){
      console.error('Status check failed', await resp.text());
      process.exit(1);
    }
    status = await resp.json();
    const elapsed = Date.now() - startTime;
    console.log(`  [${elapsed}ms] Status: ${status.status}, Progress: ${status.progress || 0}%`);

    if(status.status === 'completed' || status.status === 'failed') break;
  }

  if(status.status === 'failed'){
    console.error('❌ Task failed:', status.error);
    process.exit(1);
  }

  console.log('\nStep 7: Downloading and verifying result...');
  const resultPath = status.resultPath;
  if(!resultPath){
    console.error('No result path in status');
    process.exit(1);
  }

  resp = await fetch(`${host}/tasks/${taskId}/result`, fetchOptions);
  if(!resp.ok){
    console.error('Download failed', await resp.text());
    process.exit(1);
  }

  const resultBuffer = await resp.arrayBuffer();
  const Cresult = new Float32Array(resultBuffer);
  console.log(`  Downloaded result: ${Cresult.length} elements`);

  // Verify result
  console.log('\nStep 8: Verifying result...');
  let maxError = 0;
  let totalError = 0;
  for(let i=0;i<Cresult.length;i++){
    const diff = Math.abs(Cresult[i] - Cref[i]);
    maxError = Math.max(maxError, diff);
    totalError += diff;
  }
  const avgError = totalError / Cresult.length;
  const relError = maxError / (Math.max(...Cref) - Math.min(...Cref));

  console.log(`  Max absolute error: ${maxError.toExponential(3)}`);
  console.log(`  Average absolute error: ${avgError.toExponential(3)}`);
  console.log(`  Relative error: ${(relError*100).toFixed(6)}%`);

  if(relError < 1e-4){
    console.log('✅ Result verification passed!');
  } else {
    console.log('❌ Result verification failed - errors too large');
    process.exit(1);
  }

  // Performance summary
  const totalTime = Date.now() - startTime;
  const gflops = (2 * N * K * M) / (totalTime / 1000) / 1e9;
  console.log('\n=== Performance Summary ===');
  console.log(`Total time: ${totalTime}ms`);
  console.log(`CPU time: ${cpuTime}ms`);
  console.log(`GPU time: ${totalTime - cpuTime}ms`);
  console.log(`Performance: ${gflops.toFixed(3)} GFLOPS`);

  // Cleanup
  try {
    fs.unlinkSync(tmpA);
    fs.unlinkSync(tmpB);
  } catch(e) {
    // ignore cleanup errors
  }

  console.log('\n✅ Test completed successfully!');
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
