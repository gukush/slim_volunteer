#!/usr/bin/env node
// Enhanced ECM Stage1 script with multi-framework support
// Supports: webgpu, native-cuda, exe-cuda
// Based on patterns from test-distributed-sort-cached-enhanced.mjs

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
const framework = args.framework || 'webgpu'; // 'webgpu' | 'native-cuda' | 'exe-cuda'
const backend = args.backend || 'cuda'; // For native/exe strategies
const N = args.N || '0x00d6b8f2c8e4a1b7'; // Default demo semiprime
const B1 = parseInt(args.B1 || '50000', 10);
const B2 = parseInt(args.B2 || String(B1 * 20), 10);
const curves = parseInt(args.curves || '256', 10);
const chunkSize = parseInt(args.chunkSize || '64', 10);
const threads = parseInt(args.threads || '1', 10);
const seed = parseInt(args.seed || String(Date.now() % 2147483647), 10);
const gcdMode = parseInt(args.gcdMode || '1', 10);
const targetWindowBits = parseInt(args.targetWindowBits || '256', 10);
const Krep = parseInt(args.Krep || '1', 10);
const validate = args.validate === true || args.validate === 'true' || args.validate === '1';
const binaryPath = args.binary; // Optional binary path for exe strategy

// Helper to ignore self-signed certificates
import { Agent } from 'https';
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function fetchWithAgent(url, options = {}) {
  if (url.startsWith('https://localhost')) {
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
function getStrategyAndConfig(framework, backend, N, B1, B2, curves, chunkSize, threads, seed, gcdMode, targetWindowBits, binaryPath) {
  const frameworkLower = framework.toLowerCase();

  switch (frameworkLower) {
    case 'webgpu':
      return {
        strategyId: 'ecm-stage1',
        config: { gcdMode },
        inputArgs: { N, B1, chunk_size: chunkSize, total_curves: curves }
      };

    case 'native-cuda':
      return {
        strategyId: 'native-ecm-stage1',
        config: {
          framework: 'native-cuda',
          backend: 'cuda',
          gcdMode
        },
        inputArgs: { N, B1, chunk_size: chunkSize, total_curves: curves }
      };

    case 'exe-cuda':
      return {
        strategyId: 'exe-ecm-stage1',
        config: {
          framework: 'exe',
          backend: backend,
          program: 'ecm_stage1_cuda',
          ...(binaryPath ? { binary: binaryPath } : null)
        },
        inputArgs: { N, curves, threads, B1, B2, seed, targetWindowBits }
      };

    default:
      throw new Error(`Unsupported framework: ${framework}. Use 'webgpu', 'native-cuda', or 'exe-cuda'`);
  }
}

// Factor verification utilities
const LIMBS = 8;
const STRIDE_WORDS = LIMBS + 1;

function toBigInt(x) {
  if (typeof x === 'bigint') return x;
  if (typeof x === 'number') return BigInt(x);
  if (typeof x === 'string') {
    const s = x.trim().toLowerCase();
    return s.startsWith('0x') ? BigInt(s) : BigInt(s);
  }
  throw new Error('Invalid BigInt: ' + x);
}

function limbsToBigInt(u32, off = 0) {
  let v = 0n;
  for (let i = LIMBS - 1; i >= 0; i--) {
    v = (v << 32n) | BigInt(u32[off + i] >>> 0);
  }
  return v;
}

function parseClientResultBuffer(buf) {
  const u32 = new Uint32Array(buf);
  if (u32.length < 8) throw new Error('Result buffer too small');

  const magic = u32[0] >>> 0;
  if (magic !== 0x45434d31) throw new Error('Bad magic (ECM1 expected)');

  const pp_count = u32[4] >>> 0;
  const n = u32[5] >>> 0;

  const HEADER_WORDS = 8;
  const CONST_WORDS = LIMBS * 3 + 8;
  const CURVE_IN_WORDS_PER = LIMBS * 2;
  const CURVE_OUT_WORDS_PER = STRIDE_WORDS;

  const outStart = HEADER_WORDS + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER;
  const need = outStart + n * CURVE_OUT_WORDS_PER;
  if (u32.length < need) throw new Error(`Result buffer incomplete: have ${u32.length} words need ${need}`);

  const records = [];
  for (let i = 0; i < n; i++) {
    const off = outStart + i * CURVE_OUT_WORDS_PER;
    const limbs = u32.subarray(off, off + LIMBS);
    const status = u32[off + LIMBS] >>> 0;
    records.push({ limbs: new Uint32Array(limbs), status });
  }
  return { n, records };
}

function verifyFactors(records, N) {
  const found = [];
  for (let i = 0; i < records.length; i++) {
    const { limbs, status } = records[i];
    if (status === 2) {
      const f = limbsToBigInt(limbs, 0);
      if (f > 1n) {
        found.push({
          index: i,
          factor: '0x' + f.toString(16),
          ok: (N % f) === 0n
        });
      }
    }
  }
  return found;
}

async function runECMStage1() {
  console.log(`🚀 Running ECM Stage1 with ${framework.toUpperCase()}...`);
  console.log(`⚙️  N=${N}, B1=${B1}, curves=${curves}, chunkSize=${chunkSize}`);
  if (framework === 'exe-cuda') {
    console.log(`🎯 Backend: ${backend}, threads: ${threads}, seed: ${seed}`);
  }

  // Get strategy and config based on framework
  const { strategyId, config, inputArgs } = getStrategyAndConfig(
    framework, backend, N, B1, B2, curves, chunkSize, threads, seed, gcdMode, targetWindowBits, binaryPath
  );

  console.log(`🎯 Using strategy: ${strategyId}`);
  console.log(`⚙️  Config:`, JSON.stringify(config, null, 2));
  console.log(`📊 Input args:`, JSON.stringify(inputArgs, null, 2));

  // Create task
  const taskPayload = {
    strategyId,
    label: `ecm-stage1-${framework}-${Date.now()}`,
    config,
    inputArgs
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

  // Try to fetch summary first (JSON)
  let summary;
  try {
    const summaryResp = await fetchWithAgent(`${host}/tasks/${taskId}/output?name=output.summary.json`);
    if (summaryResp.ok) {
      summary = await summaryResp.json();
      console.log('✅ Downloaded summary JSON');
    }
  } catch (e) {
    console.log('⚠️  No summary JSON available');
  }

  // Download binary output
  let binBuf;
  try {
    const binResp = await fetchWithAgent(`${host}/tasks/${taskId}/output?name=output.bin`);
    if (binResp.ok) {
      binBuf = new Uint8Array(await binResp.arrayBuffer());
      console.log('✅ Downloaded binary output');
    }
  } catch (e) {
    // Try main output endpoint
    try {
      const mainResp = await fetchWithAgent(`${host}/tasks/${taskId}/output`);
      if (mainResp.ok) {
        binBuf = new Uint8Array(await mainResp.arrayBuffer());
        console.log('✅ Downloaded main output');
      }
    } catch (e2) {
      console.log('⚠️  No binary output available');
    }
  }

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`⏱️  Total time: ${totalTime}s`);

  // Process results
  if (summary) {
    console.log('\n📋 Summary:');
    console.log(`   Total curves: ${summary.totalCurves || curves}`);
    console.log(`   Chunks processed: ${summary.chunksProcessed || 'unknown'}`);

    if (summary.factorSummary?.primeFactors?.length > 0) {
      console.log('🎯 Prime factors found:');
      summary.factorSummary.primeFactors.forEach(f => {
        console.log(`   - ${f.factorHex} (multiplicity: ${f.multiplicity}, hits: ${f.hits})`);
      });
    }

    if (summary.factorSummary?.ecmCandidates?.length > 0) {
      console.log('🔍 ECM candidates:');
      summary.factorSummary.ecmCandidates.forEach(f => {
        console.log(`   - ${f.factorHex} (hits: ${f.hits})`);
      });
    }

    if (summary.factorSummary?.cofactor) {
      const cofactor = toBigInt(summary.factorSummary.cofactor);
      console.log(`📊 Cofactor: ${cofactor === 1n ? '1 (fully factored)' : summary.factorSummary.cofactorHex}`);
    }
  }

  // Parse binary output if available
  if (binBuf && validate) {
    console.log('\n🔍 Validating binary results...');
    try {
      const u32 = new Uint32Array(binBuf.buffer, binBuf.byteOffset, Math.floor(binBuf.byteLength / 4));
      const total = Math.floor(u32.length / STRIDE_WORDS);
      const records = [];

      for (let i = 0; i < total; i++) {
        const off = i * STRIDE_WORDS;
        const limbs = u32.subarray(off, off + LIMBS);
        const status = u32[off + LIMBS] >>> 0;
        records.push({ limbs: new Uint32Array(limbs), status });
      }

      const found = verifyFactors(records, toBigInt(N));
      console.log(`📊 Parsed ${total} curve records from binary output`);

      if (found.length === 0) {
        console.log('ℹ️  No non-trivial factors reported in binary output');
      } else {
        console.log('🎯 Factors found in binary output:');
        found.forEach(f => {
          console.log(`   Curve #${f.index}: factor=${f.factor} dividesN=${f.ok}`);
        });
      }
    } catch (error) {
      console.log('⚠️  Could not parse binary output:', error.message);
    }
  }

  return true;
}

async function main() {
  console.log('🎯 Enhanced ECM Stage1 Multi-Framework Test');
  console.log('==========================================');
  console.log(`Framework: ${framework.toUpperCase()}`);
  console.log(`Host: ${host}`);
  console.log(`Target N: ${N}`);
  console.log(`B1: ${B1}, B2: ${B2}`);
  console.log(`Curves: ${curves}, Chunk size: ${chunkSize}`);
  console.log(`GCD mode: ${gcdMode}`);
  console.log('');

  try {
    await runECMStage1();
    console.log('\n🎉 ECM Stage1 test completed successfully!');
  } catch (error) {
    console.error('💥 Error:', error);
    process.exit(99);
  }
}

// Show usage if help requested
if (args.help || args.h) {
  console.log(`
Usage: test-ecm-stage1-enhanced.mjs [options]

Multi-Framework Options:
  --framework=NAME       Framework: webgpu, native-cuda, exe-cuda (default: webgpu)
  --backend=NAME         Backend for native/exe: cuda, opencl (default: cuda)

ECM Parameters:
  --N=HEX                Number to factor (hex string, default: demo semiprime)
  --B1=N                 B1 bound (default: 50000)
  --B2=N                 B2 bound (default: B1*20)
  --curves=N             Number of curves (default: 256)
  --chunkSize=N          Curves per chunk (default: 64)
  --threads=N            Threads for exe strategy (default: 1)
  --seed=N               RNG seed for exe strategy (default: random)
  --gcdMode=N            GCD mode: 0=off, 1=on (default: 1)
  --targetWindowBits=N   Target window bits (default: 256)

Execution Options:
  --host=URL             Server URL (default: https://localhost:3000)
  --Krep=N               Replication factor (default: 1)
  --validate             Validate binary output results
  --binary=PATH          Path to binary for exe strategy
  --help, -h             Show this help

Examples:
  # WebGPU framework
  ./test-ecm-stage1-enhanced.mjs --framework=webgpu --N=0x1234567890abcdef --curves=512

  # Native CUDA framework
  ./test-ecm-stage1-enhanced.mjs --framework=native-cuda --N=0x1234567890abcdef

  # Exe CUDA framework with custom binary
  ./test-ecm-stage1-enhanced.mjs --framework=exe-cuda --binary=./bin/ecm_stage1_cuda --threads=4

  # Large test with validation
  ./test-ecm-stage1-enhanced.mjs --framework=webgpu --N=0x1234567890abcdef --curves=1024 --validate
`);
  process.exit(0);
}

main();
