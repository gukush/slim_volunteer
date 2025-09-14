// ECM Stage 1 (Native CUDA) — strategy wrapper that reuses your webgpu chunker/assembler.
// Uses Lua host script to orchestrate native CUDA execution.

import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// Reuse chunker/assembler from your existing WebGPU strategy file to avoid duplication.
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
} from './ecm-stage1.js';

export const id = 'native-ecm-stage1';
export const name = 'ECM Stage 1 (Native CUDA)';
export const framework = 'native-cuda';

// Helper functions to resolve file paths and create artifacts
function tryRead(p) {
  try { return fs.readFileSync(p); } catch { return null; }
}

function findFirstExisting(paths) {
  for (const p of paths) {
    const b = tryRead(p);
    if (b) return { path: p, bytes: b };
  }
  return null;
}

function resolveCandidates(rel) {
  const cwd = process.cwd();
  const here = path.dirname(new URL(import.meta.url).pathname);
  return [
    path.join(cwd, rel),
    path.join(cwd, 'strategies', rel),
    path.join(cwd, 'kernels', rel),
    path.join(here, rel),
    path.join(here, '..', rel),
    path.join(here, '..', 'kernels', rel),
  ];
}

function b64(buf) { return Buffer.isBuffer(buf) ? buf.toString('base64') : Buffer.from(buf).toString('base64'); }

function makeArtifact({ type = 'text', name, program, backend, exec = false, bytes }) {
  return { type, name, program, backend, exec, bytes };
}

// This tells the server what to ship to clients.
export function getClientExecutorInfo(config, inputArgs) {
  const framework = String(config?.framework || 'native-cuda').toLowerCase();

  // Always try to include host.lua so native client can route + compile at runtime.
  const hostCandidates = resolveCandidates('executors/host-ecm-stage1.lua');
  const host = findFirstExisting(hostCandidates);

  // Optional kernels (these are just hints; Lua host can also generate or use its own)
  const ku = findFirstExisting(resolveCandidates('kernels/cuda/ecm_stage1_cuda.cu'));
  const kl = findFirstExisting(resolveCandidates('kernels/opencl/ecm_stage1_opencl.cl'));

  const artifacts = [];
  if (host) artifacts.push(makeArtifact({
    type: 'lua', name: 'host.lua', program: 'host', backend: 'host', bytes: b64(host.bytes)
  }));
  if (ku) artifacts.push(makeArtifact({
    type: 'text', name: path.basename(ku.path), program: 'ecm_stage1', backend: 'cuda', bytes: b64(ku.bytes)
  }));
  if (kl) artifacts.push(makeArtifact({
    type: 'text', name: path.basename(kl.path), program: 'ecm_stage1', backend: 'opencl', bytes: b64(kl.bytes)
  }));

  return {
    framework: 'cuda', // Native client will use CUDA executor via Lua
    artifacts: artifacts,
    // Output is still the same packed Uint32 layout you already consume.
    schema: { output: 'Uint32Array' },
  };
}

// Helper functions for trial division
function parseBig(value) {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'number') return BigInt(value);
  if (typeof value === 'string' && value.trim().length) {
    const s = value.trim().toLowerCase();
    return s.startsWith('0x') ? BigInt(s) : BigInt(s);
  }
  throw new Error('Cannot parse BigInt from: ' + String(value));
}

function trialDivideSmallPrimes(Nbig, maxPrime = 7919n) {
  const factors = new Map(); // BigInt -> multiplicity
  let N = parseBig(Nbig);

  // Only check for factor 2 (most common case)
  const p = 2n;
  if (N % p === 0n) {
    let k = 0n;
    while (N % p === 0n) { N /= p; k++; }
    factors.set(p, Number(k));
  }
  if (N === 1n) return { reducedN: N, smallFactors: factors };

  return { reducedN: N, smallFactors: factors };
}

// Copy WebGPU chunker logic verbatim and modify just the payload format for native client
export function buildChunker({ taskId, taskDir, K, config, inputArgs, inputFiles }) {
  const N_hex = inputArgs.N;
  const N0 = BigInt(N_hex);
  const B1 = Number(inputArgs.B1);
  const totalCurves = Number(inputArgs.total_curves);
  const chunkSize = Number(inputArgs.chunk_size);
  const totalChunks = Math.ceil(totalCurves / chunkSize);

  // Copy the exact same logic from ecm-stage1.js
  function computeMontgomeryConstants(N) {
    // R = 2^256, mont_one = R mod N, R2 = R^2 mod N
    const R = 1n << 256n;
    const R2 = (R * R) % N;
    const mont_one = R % N;

    function modInverse(a, m) {
      function extGcd(a, b) {
        if (a === 0n) return [b, 0n, 1n];
        const [gcd, x1, y1] = extGcd(b % a, a);
        const x = y1 - (b / a) * x1;
        const y = x1;
        return [gcd, x, y];
      }
      const [gcd, x] = extGcd(a % m, m);
      if (gcd !== 1n) throw new Error('Modular inverse does not exist');
      return (x % m + m) % m;
    }

    const n0_low = N & 0xFFFFFFFFn;
    const n0inv = modInverse(n0_low, 1n << 32n);
    const n0inv32 = Number((-n0inv) & 0xFFFFFFFFn);

    return {
      N: bigIntToLimbs(N),
      R2: bigIntToLimbs(R2),
      mont_one: bigIntToLimbs(mont_one),
      n0inv32: n0inv32 >>> 0
    };
  }

  function bigIntToLimbs(n) {
    const limbs = new Uint32Array(8);
    for (let i = 0; i < 8; i++) {
      limbs[i] = Number((n >> (32n * BigInt(i))) & 0xFFFFFFFFn);
    }
    return limbs;
  }

  function isPrime(n) {
    if (n < 2) return false;
    if (n === 2) return true;
    if (n % 2 === 0) return false;
    for (let i = 3; i * i <= n; i += 2) {
      if (n % i === 0) return false;
    }
    return true;
  }

  // Generate prime powers up to B1
  const primePowers = [];
  for (let p = 2; p <= B1; p++) {
    if (isPrime(p)) {
      let power = p;
      while (power <= B1) {
        primePowers.push(power);
        power *= p;
      }
    }
  }
  const pp_count = primePowers.length;

  // Constants for buffer layout
  const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + padding(3)
  const STATE_WORDS_PER_CURVE = 8 + 8 + 8 + 2; // X + Z + A24 + (sigma, curve_ok)

  // Pre-pass: strip tiny primes (especially factor 2)
  const { reducedN: Nred, smallFactors } = trialDivideSmallPrimes(N0);
  if (Nred === 1n) {
    console.log('[native-ecm-stage1] All factors were tiny; skipping GPU stage.');
    return { async *stream() {} };
  }

  // Compute Montgomery constants on reduced N (now guaranteed to be odd)
  const constants = computeMontgomeryConstants(Nred);

  return {
    async *stream() {
      for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
        const startCurve = chunkIndex * chunkSize;
        const endCurve = Math.min(startCurve + chunkSize, totalCurves);
        const curvesInChunk = endCurve - startCurve;

        // Generate random seed for this chunk
        const taskSeed64 = BigInt(Date.now()) + BigInt(chunkIndex) * 1000000007n;
        const seed_lo = Number(taskSeed64 & 0xFFFFFFFFn);
        const seed_hi = Number((taskSeed64 >> 32n) & 0xFFFFFFFFn);

        // Calculate buffer size
        const HEADER_WORDS_V3 = 12;
        const CONST_WORDS = 8*3 + 4;
        const totalWords = HEADER_WORDS_V3 + CONST_WORDS + pp_count + curvesInChunk * (CURVE_OUT_WORDS_PER + STATE_WORDS_PER_CURVE);
        const buffer = new Uint32Array(totalWords);

        let offset = 0;
        // Header
        buffer[offset++] = 0;
        buffer[offset++] = pp_count;
        buffer[offset++] = curvesInChunk;
        buffer[offset++] = seed_lo;
        buffer[offset++] = seed_hi;
        buffer[offset++] = startCurve;    // base_curve
        buffer[offset++] = (config?.gcdMode ? 1 : 0);
        buffer[offset++] = 0;             // pp_start (will be set by client)
        buffer[offset++] = 0;             // pp_len (will be set by client)

        // Constants for Nred
        buffer.set(constants.N, offset);         offset += 8;
        buffer.set(constants.R2, offset);        offset += 8;
        buffer.set(constants.mont_one, offset);  offset += 8;
        buffer[offset++] = constants.n0inv32;
        buffer[offset++] = 0; buffer[offset++] = 0; buffer[offset++] = 0;

        // Prime powers
        buffer.set(primePowers, offset);         offset += pp_count;

        // Reserve output space
        offset += curvesInChunk * CURVE_OUT_WORDS_PER;

        // Reserve state space (initialized to zeros)
        offset += curvesInChunk * STATE_WORDS_PER_CURVE;

        // Convert ArrayBuffer to base64 for native client (Lua script expects base64)
        const dataBase64 = Buffer.from(buffer.buffer).toString('base64');

        // Create payload in format expected by Lua script
        const payload = {
          data: dataBase64,  // base64 string instead of ArrayBuffer
          dims: { n: curvesInChunk, pp_count, total_words: totalWords }
        };

        const meta = {
          chunkIndex, startCurve, endCurve,
          n: curvesInChunk, pp_count, total_words: totalWords,
          N: N_hex,
          reducedN: '0x' + N0.toString(16),
          B1,
          rngSeed: '0x' + taskSeed64.toString(16),
          version: 3  // Mark as version 3
        };

        yield {
          id: uuidv4(),
          payload,
          meta,
          tCreate: Date.now()
        };
      }
    }
  };
}

// Reuse assembler from WebGPU version
export const buildAssembler = buildAssemblerWeb;
