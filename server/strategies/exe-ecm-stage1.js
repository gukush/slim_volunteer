// server/strategies/exe-ecm-stage1.js
//
// ECM Stage 1 (exe/CUDA) – thin shim around the WebGPU strategy,
// repackaged for the BinaryExecutor (stdin -> stdout).
//
// Reuses the WebGPU chunker/assembler; only the transport changes.

import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// Reuse the webgpu ECM strategy's chunker/assembler so buffer layout stays identical.
import { buildChunker as buildChunkerWeb, buildAssembler as buildAssemblerWeb } from './ecm-stage1.js';

export const id = 'exe-ecm-stage1';
export const name = 'ECM Stage 1 (exe/CUDA)';
export const framework = 'exe';

/* ------------------------- Small primes (≤ 7919) ------------------------- */
const SMALL_PRIMES = [2]; // Add more for real use

function trialDivideSmallPrimes(Nbig, maxPrime = 7919n) {
  const factors = new Map(); // BigInt -> multiplicity
  let N = parseBig(Nbig);

  for (const pn of SMALL_PRIMES) {
    const p = BigInt(pn);
    if (p > maxPrime) break;
    if (N % p === 0n) {
      let k = 0n;
      while (N % p === 0n) { N /= p; k++; }
      factors.set(p, Number(k));
    }
    if (N === 1n) break;
  }
  return { reducedN: N, smallFactors: factors };
}

/* ------------------------------ BigInt utils ----------------------------- */

function hexToBigInt(hex) {
  if (typeof hex === 'string' && hex.startsWith('0x')) return BigInt(hex);
  return BigInt(hex);
}

function bigIntToLimbs(n, limbCount = 8) {
  const limbs = new Uint32Array(limbCount);
  let temp = n;
  for (let i = 0; i < limbCount; i++) {
    limbs[i] = Number(temp & 0xFFFFFFFFn);
    temp >>= 32n;
  }
  return limbs;
}

function generatePrimePowers(B1) {
  // Highest prime powers ≤ B1
  const primes = [];
  const sieve = new Array(B1 + 1).fill(true);
  sieve[0] = sieve[1] = false;

  for (let p = 2; p <= B1; p++) {
    if (!sieve[p]) continue;
    // mark multiples
    if (p * p <= B1) {
      for (let i = p * p; i <= B1; i += p) sieve[i] = false;
    }
    // highest power of p ≤ B1
    let pk = p;
    while (pk * p <= B1) pk *= p;
    primes.push(pk);
  }

  return new Uint32Array(primes);
}

// ---- BigInt + limbs helpers ----
function gcdBig(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) { const t = a % b; a = b; b = t; }
  return a;
}

function limbsToBigLE(u32, start, nWords) {
  let x = 0n;
  for (let i = nWords - 1; i >= 0; --i) {
    x = (x << 32n) + BigInt(u32[start + i] >>> 0);
  }
  return x;
}

function parseBig(value) {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'number') return BigInt(value);
  if (typeof value === 'string' && value.trim().length) {
    const s = value.trim().toLowerCase();
    return s.startsWith('0x') ? BigInt(s) : BigInt(s);
  }
  throw new Error('Cannot parse BigInt from: ' + String(value));
}

/* ------------------ Montgomery constants for WGSL kernel ----------------- */

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

/* ------------------------------ Factor bag -------------------------------- */

class FactorBag {
  constructor(originalN) {
    this.N0 = parseBig(originalN);
    this.remaining = this.N0;
    this.hits = new Map(); // key=str(n) -> { n: BigInt, hits: number }
    this.mult = new Map(); // key=str(n) -> multiplicity (number)
  }

  addCandidate(raw) {
    let f = parseBig(raw);
    if (f <= 1n) return;
    f = gcdBig(f, this.N0);
    if (f <= 1n || f === this.N0) return;
    const key = f.toString();
    const rec = this.hits.get(key) || { n: f, hits: 0 };
    rec.hits++;
    this.hits.set(key, rec);
  }

  // Attempt to divide remaining by each unique candidate to find true multiplicity
  finalize() {
    const uniq = [...this.hits.values()].sort((a, b) => (a.n < b.n ? 1 : -1));
    for (const { n: f } of uniq) {
      let k = 0;
      while (this.remaining % f === 0n) {
        this.remaining /= f;
        k++;
      }
      if (k > 0) this.mult.set(f.toString(), k);
    }
    return {
      uniqueFactors: uniq.map(x => x.n),
      hits: this.hits,
      multiplicity: this.mult,
      cofactor: this.remaining,
    };
  }
}

/* --------------------------- Binary view helper --------------------------- */

function asU32View(bin){
  if (bin instanceof ArrayBuffer){
    if ((bin.byteLength % 4) !== 0) throw new Error('Result byteLength not multiple of 4');
    return new Uint32Array(bin);
  }
  if (ArrayBuffer.isView(bin)){
    const { buffer, byteOffset, byteLength } = bin;
    if ((byteLength % 4) !== 0) throw new Error('Result view length not multiple of 4');
    if ((byteOffset % 4) === 0){
      return new Uint32Array(buffer, byteOffset, Math.floor(byteLength / 4));
    }
    const copy = new Uint8Array(byteLength);
    copy.set(new Uint8Array(buffer, byteOffset, byteLength));
    return new Uint32Array(copy.buffer);
  }
  if (typeof Buffer !== 'undefined' && Buffer.isBuffer(bin)){
    const buf = bin;
    if ((buf.byteLength % 4) !== 0) throw new Error('Result Buffer length not multiple of 4');
    if ((buf.byteOffset % 4) === 0){
      return new Uint32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
    }
    const copy = Buffer.from(buf);
    return new Uint32Array(copy.buffer, copy.byteOffset, Math.floor(copy.buffer.byteLength / 4));
  }
  const coerced = Buffer.from(bin);
  if ((coerced.byteLength % 4) !== 0) throw new Error('Result coerced length not multiple of 4');
  return new Uint32Array(coerced.buffer, coerced.byteOffset, Math.floor(coerced.byteLength / 4));
}

// Default program names per backend (override via config.program)
const defaultPrograms = {
  cuda: 'ecm_stage1_cuda',   // look up in PATH on the native client unless provided via artifact
  // opencl: 'ecm_stage1_opencl', // (optional) if you later add a CL exe
};

export function getClientExecutorInfo({ config = {}, inputArgs = {} } = {}) {
  const backend = (config.backend || 'cuda').toLowerCase();
  const program = config.program || defaultPrograms[backend] || 'ecm_stage1_cuda';

  // If the caller provides a compiled binary path, ship it as an artifact to the native client.
  // Otherwise, the native client will resolve `program` in PATH.
  const artifacts = [];
  if (config.binary) {
    const binPath = path.resolve(config.binary);
    const bytes = fs.readFileSync(binPath);
    artifacts.push({
      name: program,          // BinaryExecutor will cache by this program name
      type: 'binary',
      exec: true,
      bytes: bytes.toString('base64'),
    });
  }

  return {
    id,
    name,
    framework,
    backend,
    program,
    artifacts,

    // (Optional) schematic; helpful for debugging
    schema: {
      order: ['stdin', 'stdout'],
      inputs: [{ name: 'io', type: 'u32', note: 'Full ECM Stage 1 IO buffer (header + consts + pp + output + state)' }],
      outputs: [{ name: 'io', type: 'u32', size: '== input size' }],
    },
  };
}

// Build chunks using the *same* IO buffer the WebGPU path expects,
// but convert each chunk to the exe transport using common protocol.
export function buildChunker({ taskId, taskDir, K, config, inputArgs, inputFiles }) {
  console.log('DEBUG exe-ecm-stage1 buildChunker args:', JSON.stringify({ taskId, taskDir, K, config, inputArgs, inputFiles }, null, 2));
  console.log('DEBUG exe-ecm-stage1 inputArgs:', JSON.stringify(inputArgs, null, 2));
  const webChunker = buildChunkerWeb({ taskId, taskDir, K, config, inputArgs, inputFiles });

  return {
    async *stream() {
      for await (const chunk of webChunker.stream()) {
        // `chunk.payload.data` is the full IO ArrayBuffer produced by the web ECM strategy.
        const ioBuffer = chunk?.payload?.data;
        if (!ioBuffer || !(ioBuffer instanceof ArrayBuffer)) {
          throw new Error(`[${id}] Expected web ECM chunk to have payload.data as ArrayBuffer`);
        }

        // For exe: write entire IO buffer to stdin, expect same-sized stdout.
        const outSize = ioBuffer.byteLength;

        yield {
          id: chunk.id,
          payload: {
            action: 'exec',
            // TaskManager will base64-encode ArrayBuffers for native clients automatically.
            buffers: [ioBuffer],          // stdin
            outputs: [outSize],           // expected stdout size
            meta: {
              program: (config?.program) || defaultPrograms[(config?.backend || 'cuda').toLowerCase()] || 'ecm_stage1_cuda',
              backend: (config?.backend || 'cuda').toLowerCase(),
              framework: 'exe',
              // Carry through any useful metadata for logging/diagnostics
              ...chunk.meta,
            },
          },
          meta: chunk.meta,              // keep the original meta (useful in assembler)
        };
      }
    }
  };
}

// Reuse the exact same assembler the web strategy uses;
// it already knows how to parse the returned IO buffer and produce summary + artifacts.
export const buildAssembler = buildAssemblerWeb;

// Kill-switch support: Calculate total chunks deterministically
export function getTotalChunks(config, inputArgs) {
  const N_hex = inputArgs.N;
  const B1 = inputArgs.B1 || 50000;
  const chunk_size = inputArgs.chunk_size || 256;
  const total_curves = inputArgs.total_curves || 1024;

  if (!N_hex) {
    logger.warn('ECM getTotalChunks: No N parameter provided, using default estimate');
    return Math.ceil(total_curves / chunk_size);
  }

  try {
    const N0 = hexToBigInt(N_hex);

    // Pre-pass: strip tiny primes (same logic as buildChunker)
    const { reducedN: Nred } = trialDivideSmallPrimes(N0);

    const totalChunks = (Nred === 1n) ? 0 : Math.ceil(total_curves / chunk_size);

    logger.info(`exe-ecm-stage1 getTotalChunks: N=${N0.toString(16)}, B1=${B1}, curves=${total_curves}, chunk_size=${chunk_size} -> ${totalChunks} chunks`);
    return totalChunks;
  } catch (e) {
    logger.warn(`exe-ecm-stage1 getTotalChunks: Error calculating chunks, using fallback:`, e.message);
    return Math.ceil(total_curves / chunk_size);
  }
}