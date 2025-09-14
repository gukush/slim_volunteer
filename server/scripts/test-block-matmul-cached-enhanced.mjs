#!/usr/bin/env node
// Enhanced cached block matmul script with multi-framework support
// Supports: webgpu, webgl2, cpp-wasm, native (opencl/cuda via lua), exe (opencl/cuda/vulkan via binaries)

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

const host = args.host || 'https://localhost:3000';
const framework = args.framework || 'webgpu'; // 'webgpu' | 'webgl2' | 'cpp-wasm' | 'native' | 'exe'
const backend = args.backend || 'opencl'; // For native/exe: 'opencl' | 'cuda' | 'vulkan'
const N = parseInt(args.N||'64',10), K = parseInt(args.K||'64',10), M = parseInt(args.M||'64',10);
const TS = parseInt(args.tileSize||'32',10);
const Krep = parseInt(args.Krep||'1',10);
const validate = args.validate === true || args.validate === 'true' || args.validate === '1';
const datatype = args.datatype || 'f32'; // 'f32', 'f16', 'int8'
const chunkSize = parseInt(args.chunkSize||'8388608',10); // For exe strategies
const fileA = args.fileA || 'A.bin';
const fileB = args.fileB || 'B.bin';
const cleanupOutput = args.cleanupOutput === true || args.cleanupOutput === 'true' || args.cleanupOutput === '1'; // Remove output files after task completion

// CPU reference implementation for validation
function randMat(r,c){ const a = new Float32Array(r*c); for(let i=0;i<a.length;i++) a[i] = (Math.random()*2-1); return a; }
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
import { Agent } from 'https';
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function fetchWithAgent(url, options = {}) {
  if (url.startsWith('https://localhost')) {
    // For self-signed certificates, we need to use a custom agent
    const https = await import('https');
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const reqOptions = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: options.method || 'GET',
        headers: options.headers || {},
        rejectUnauthorized: false
      };

      const req = https.request(reqOptions, (res) => {
        let data = [];
        res.on('data', chunk => data.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(data);
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            json: async () => JSON.parse(buffer.toString()),
            text: async () => buffer.toString(),
            arrayBuffer: async () => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
          });
        });
      });

      req.on('error', reject);
      if (options.body) {
        req.write(options.body);
      }
      req.end();
    });
  }
  return fetch(url, options);
}

// Strategy routing based on framework
function getStrategyAndConfig(framework, backend, N, K, M, TS, datatype, chunkSize) {
  const frameworkLower = framework.toLowerCase();
  const backendLower = backend.toLowerCase();

  switch (frameworkLower) {
    case 'webgpu':
      // Use block-matmul-flex for WebGPU with datatype support
      return {
        strategyId: 'block-matmul-flex',
        config: { N, K, M, tileSize: TS, framework: 'webgpu', datatype }
      };

    case 'webgl2':
      // Use block-matmul-flex for WebGL2
      return {
        strategyId: 'block-matmul-flex',
        config: { N, K, M, tileSize: TS, framework: 'webgl2', datatype }
      };

    case 'cpp-wasm':
      // Use block-matmul-flex for CPP-WASM
      return {
        strategyId: 'block-matmul-flex',
        config: { N, K, M, tileSize: TS, framework: 'cpp-wasm', datatype }
      };

    case 'native':
      // Use native-block-matmul for LuaJIT-based native execution
      if (!['opencl', 'cuda'].includes(backendLower)) {
        throw new Error(`Unsupported backend for native framework: ${backend}. Use 'opencl' or 'cuda'`);
      }
      return {
        strategyId: 'native-block-matmul',
        config: { N, K, M, tileSize: TS, framework: `native-${backendLower}` }
      };

    case 'exe':
      // Use exe-block-matmul-flex for native binary execution
      if (!['opencl', 'cuda', 'vulkan'].includes(backendLower)) {
        throw new Error(`Unsupported backend for exe framework: ${backend}. Use 'opencl', 'cuda', or 'vulkan'`);
      }
      return {
        strategyId: 'exe-block-matmul-flex',
        config: { N, K, M, tileSize: TS, backend: backendLower, chunk_size: chunkSize }
      };

    default:
      throw new Error(`Unsupported framework: ${framework}. Use 'webgpu', 'webgl2', 'cpp-wasm', 'native', or 'exe'`);
  }
}

async function runBlockMatmul() {
  console.log(`🚀 Running Block Matmul with ${framework.toUpperCase()}...`);
  if (['native', 'exe'].includes(framework.toLowerCase())) {
    console.log(`🎯 Backend: ${backend.toUpperCase()}`);
  }
  console.log(`📊 Dimensions: A(${N}x${K}) × B(${K}x${M}) = C(${N}x${M})`);
  console.log(`📁 Cached files: ${fileA}, ${fileB}`);

  // Load cached files for validation if needed
  let referenceA, referenceB, referenceC;
  if (validate) {
    const uploadsDir = path.join(__dirname, '..', 'storage', 'uploads');
    const fileAPath = path.join(uploadsDir, fileA);
    const fileBPath = path.join(uploadsDir, fileB);

    if (!fs.existsSync(fileAPath) || !fs.existsSync(fileBPath)) {
      console.error(`❌ Cached files not found:`);
      console.error(`  - ${fileAPath}: ${fs.existsSync(fileAPath) ? '✅' : '❌'}`);
      console.error(`  - ${fileBPath}: ${fs.existsSync(fileBPath) ? '✅' : '❌'}`);
      console.error('Available files in uploads directory:');
      try {
        const files = fs.readdirSync(uploadsDir);
        files.forEach(f => {
          const fullPath = path.join(uploadsDir, f);
          const stats = fs.statSync(fullPath);
          console.error(`  - ${f} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);
        });
      } catch (e) {
        console.error('  (Could not list uploads directory)');
      }
      process.exit(1);
    }

    console.log('🔍 Loading cached files for validation...');
    const fileABuffer = fs.readFileSync(fileAPath);
    const fileBBuffer = fs.readFileSync(fileBPath);

    referenceA = new Float32Array(fileABuffer.buffer, fileABuffer.byteOffset, fileABuffer.byteLength / 4);
    referenceB = new Float32Array(fileBBuffer.buffer, fileBBuffer.byteOffset, fileBBuffer.byteLength / 4);

    console.log(`📊 Loaded A: ${referenceA.length} elements (${(fileABuffer.length / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`📊 Loaded B: ${referenceB.length} elements (${(fileBBuffer.length / 1024 / 1024).toFixed(2)} MB)`);

    // Generate reference result
    console.log('🧮 Computing reference result...');
    referenceC = matmulCPUf32(referenceA, referenceB, N, K, M);
    console.log('✅ Reference computation complete');
  }

  // Get strategy and config based on framework
  const { strategyId, config } = getStrategyAndConfig(framework, backend, N, K, M, TS, datatype, chunkSize);

  // Add cleanup flag to config if requested
  if (cleanupOutput) {
    config.cleanupOutputFiles = true;
  }

  console.log(`🎯 Using strategy: ${strategyId}`);
  console.log(`⚙️  Config:`, JSON.stringify(config, null, 2));
  if (cleanupOutput) {
    console.log(`🧹 Output files will be cleaned up after task completion`);
  }

  // Create task using cached file paths
  const taskPayload = {
    strategyId,
    K: Krep,
    label: `block-matmul-cached-${framework}`,
    config,
    cachedFilePaths: [fileA, fileB]
  };

  let resp = await fetchWithAgent(`${host}/tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(taskPayload)
  });

  if (!resp.ok) {
    console.error('❌ Create task failed:', await resp.text());
    process.exit(1);
  }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log(`✅ Created task ${taskId}`);

  // Start task
  resp = await fetchWithAgent(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) {
    console.error('❌ Start failed:', await resp.text());
    process.exit(1);
  }

  // Poll for completion
  let status;
  const startTime = Date.now();
  console.log('⏳ Monitoring task progress...');
  while (true) {
    await new Promise(r => setTimeout(r, 1000));
    const s = await fetchWithAgent(`${host}/tasks/${taskId}`);
    const j = await s.json();
    status = j.status;

    const chunks = j.completedChunks || 0;
    const total = j.totalChunks || '?';
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    process.stdout.write(`\r📊 [${elapsed}s] status=${status} ${chunks}/${total} chunks   `);

    if (status === 'completed' || status === 'error' || status === 'canceled') break;
  }
  console.log();

  if (status !== 'completed') {
    console.error(`❌ Task did not complete: ${status}`);
    process.exit(2);
  }

  // Download results
  console.log('📥 Downloading results...');
  const out = await fetchWithAgent(`${host}/tasks/${taskId}/output`);
  if (!out.ok) {
    console.error('❌ Download failed:', await resp.text());
    process.exit(3);
  }

  const resultBuffer = new Uint8Array(await out.arrayBuffer());
  const resultData = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

  console.log(`✅ Successfully computed ${resultData.length} elements`);

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
  const gflops = (2 * N * K * M) / (parseFloat(totalTime) * 1e9);
  console.log(`⏱️  Time: ${totalTime}s, Performance: ${gflops.toFixed(2)} GFLOPS`);

  // Validate results if requested
  if (validate && referenceC) {
    console.log('🔍 Validating results...');

    if (resultData.length !== referenceC.length) {
      console.error(`❌ Length mismatch: expected ${referenceC.length}, got ${resultData.length}`);
      process.exit(4);
    }

    let maxError = 0;
    let errorCount = 0;
    for (let i = 0; i < resultData.length; i++) {
      const error = Math.abs(resultData[i] - referenceC[i]);
      if (error > 1e-4) errorCount++;
      maxError = Math.max(maxError, error);
    }

    console.log(`📊 Validation: ${errorCount}/${resultData.length} elements differ`);
    console.log(`📊 Max error: ${maxError.toExponential(3)}`);

    if (maxError > 1e-3) {
      console.error('❌ FAIL - Results differ significantly from reference');
      process.exit(5);
    } else {
      console.log('✅ PASS - Results match reference within tolerance');
    }
  }

  return true;
}

async function main() {
  console.log('🎯 Enhanced Cached Block Matmul Test');
  console.log('=====================================');
  console.log(`Framework: ${framework.toUpperCase()}`);
  if (['native', 'exe'].includes(framework.toLowerCase())) {
    console.log(`Backend: ${backend.toUpperCase()}`);
  }
  console.log(`Host: ${host}`);
  console.log(`Dimensions: ${N}x${K} × ${K}x${M} = ${N}x${M}`);
  console.log(`Cached files: ${fileA}, ${fileB}`);
  console.log('');

  try {
    await runBlockMatmul();
    console.log('\n🎉 Block matmul test completed successfully!');
  } catch (error) {
    console.error('💥 Error:', error);
    process.exit(99);
  }
}

main();
