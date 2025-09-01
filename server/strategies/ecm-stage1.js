import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'ecm-stage1';
export const name = 'ECM Stage 1 (WebGPU)';
export const framework = 'webgpu';

export function getClientExecutorInfo(config, inputArgs){
  return {
    framework,
    path: 'executors/webgpu-ecm-stage1.client.js',
    kernels: ['kernels/ecm_stage1_webgpu_compute.wgsl'],
    schema: { output: 'Uint32Array' },
  };
}

// Helper functions for big integer operations
function hexToBigInt(hex) {
  if (typeof hex === 'string' && hex.startsWith('0x')) {
    return BigInt(hex);
  }
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
  // Generate list of prime powers <= B1
  const primes = [];
  const sieve = new Array(B1 + 1).fill(true);
  sieve[0] = sieve[1] = false;

  for (let p = 2; p <= B1; p++) {
    if (sieve[p]) {
      // Add highest power of p that's <= B1
      let pk = p;
      while (pk * p <= B1) {
        pk *= p;
      }
      primes.push(pk);

      // Mark multiples as non-prime
      for (let i = p * p; i <= B1; i += p) {
        sieve[i] = false;
      }
    }
  }

  return new Uint32Array(primes);
}

function computeMontgomeryConstants(N) {
  // For WebGPU implementation, we need:
  // - R = 2^256 (implicit)
  // - R2 = (2^256)^2 mod N
  // - mont_one = 2^256 mod N
  // - n0inv32 = -N^(-1) mod 2^32

  const R = 1n << 256n; // 2^256
  const R2 = (R * R) % N;
  const mont_one = R % N;

  // Compute n0inv32 = -N^(-1) mod 2^32 using extended GCD
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
    n0inv32: n0inv32 >>> 0 // Ensure unsigned 32-bit
  };
}

// --- modular helpers (BigInt) ---
function mod(a, n){ a %= n; return a < 0n ? a + n : a; }

// Extended GCD (iterative; safe in Node)
function egcd(a, b){
  a = mod(a, b); // ensure 0 <= a < b when possible
  let old_r = b, r = a;
  let old_s = 1n, s = 0n;
  let old_t = 0n, t = 1n;
  while (r !== 0n) {
    const q = old_r / r;
    [old_r, r] = [r, old_r - q*r];
    [old_s, s] = [s, old_s - q*s];
    [old_t, t] = [t, old_t - q*t];
  }
  // old_r = gcd(a,b), and old_s*a + old_t*b = gcd
  return [old_r, old_s, old_t];
}

// Multiplicative inverse mod n; returns null if non-invertible
function modInv(a, n){
  a = mod(a, n);
  if (a === 0n) return null;
  const [g, x] = egcd(a, n);
  if (g !== 1n) return null;            // no inverse ⇒ shares factor with n
  return mod(x, n);
}

// --- Suyama parametrization for (A24, X1) on a Montgomery curve ---
// This REPLACES your current generateRandomCurve
function generateRandomCurve(N, seed) {
  // small LCG for reproducible σ
  let rng = BigInt(seed) & 0x7fffffffn;
  function rand32(){ rng = (rng * 1103515245n + 12345n) & 0x7fffffffn; return rng; }
  function nextSigma(){
    let s = 0n;
    for (let i=0;i<3;i++) s = (s << 21n) | rand32();
    return mod(s, N);
  }

  const inv4 = modInv(4n, N);  // should exist for odd N
  for (let tries = 0; tries < 32; tries++){
    const sigma = nextSigma();
    // Reject bad sigmas: 0, ±1, ±2
    if (sigma===0n) continue;
    if (sigma===1n || sigma===N-1n) continue;
    if (sigma===2n || sigma===N-2n) continue;

    const u = mod(sigma*sigma - 5n, N);
    const v = mod(4n*sigma, N);

    const invu = modInv(u, N);
    const invv = modInv(v, N);
    if (!invu || !invv || !inv4) continue; // unlucky (or a factor) → retry

    // A = ((v−u)^3 * (3u+v)) / (4 u^3 v) − 2   (mod N)
    const vm_u   = mod(v - u, N);
    const num    = mod(vm_u*vm_u % N * vm_u % N * mod(3n*u + v, N), N);
    const den    = mod(4n * (u*u % N * u % N) % N * v, N);
    const invDen = modInv(den, N);
    if (!invDen) continue;

    const A      = mod(num * invDen - 2n, N);
    const A24    = mod((A + 2n) * inv4, N);

    // X1 = (u^3 / v^3) = u^3 * (v^-3)
    const u3     = mod(u*u % N * u % N, N);
    const vInv3  = modInv(v, N);
    if (!vInv3) continue;
    const X1     = mod(u3 * (vInv3*vInv3 % N * vInv3 % N), N);

    return {
      A24: bigIntToLimbs(A24),
      X1:  bigIntToLimbs(X1)
    };
  }
  // If we failed many times (extremely unlikely), perturb seed and retry
  return generateRandomCurve(N, Number((BigInt(seed) + 777n) % 0x7fffffffn));
}


export function buildChunker({ taskId, taskDir, K, config, inputArgs, inputFiles }){
  // Parse ECM parameters from inputArgs
  const N_hex = inputArgs.N;
  const B1 = inputArgs.B1 || 50000;
  const chunk_size = inputArgs.chunk_size || 256;  // curves per chunk
  const total_curves = inputArgs.total_curves || 1024;

  if (!N_hex) {
    throw new Error('ECM requires parameter N (number to factor)');
  }

  const N = hexToBigInt(N_hex);
  logger.info(`ECM Stage 1: N=${N.toString(16)}, B1=${B1}, curves=${total_curves}, chunk_size=${chunk_size}`);

  // Pre-compute constants and prime powers
  const constants = computeMontgomeryConstants(N);
  const primePowers = generatePrimePowers(B1);

  const totalChunks = Math.ceil(total_curves / chunk_size);

  return {
    async *stream(){
      for(let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++){
        const startCurve = chunkIndex * chunk_size;
        const endCurve = Math.min(startCurve + chunk_size, total_curves);
        const curvesInChunk = endCurve - startCurve;

        // Generate curve parameters for this chunk
        const curves = [];
        for(let i = 0; i < curvesInChunk; i++){
          const curveIndex = startCurve + i;
          curves.push(generateRandomCurve(N, curveIndex + 12345)); // deterministic seed
        }

        // Pack data for WebGPU kernel
        // Layout: header + constants + primePowers + curveInputs + curveOutputs
        const HEADER_WORDS = 8;
        const CONST_WORDS = 8*3 + 4; // N(8) + R2(8) + mont_one(8) + n0inv32(1) + pad(3)
        const pp_count = primePowers.length;
        const CURVE_IN_WORDS_PER = 8*2; // A24(8) + X1(8)
        const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + pad(3)

        const totalWords = HEADER_WORDS + CONST_WORDS + pp_count +
                          curvesInChunk * CURVE_IN_WORDS_PER +
                          curvesInChunk * CURVE_OUT_WORDS_PER;

        const buffer = new Uint32Array(totalWords);
        let offset = 0;

        // Header: magic, version, reserved, reserved, pp_count, n_curves, reserved, reserved
        buffer[offset++] = 0x45434D31; // "ECM1" magic
        buffer[offset++] = 1; // version
        buffer[offset++] = 0; // reserved
        buffer[offset++] = 0; // reserved
        buffer[offset++] = pp_count;
        buffer[offset++] = curvesInChunk;
        buffer[offset++] = 0; // reserved
        buffer[offset++] = 0; // reserved

        // Constants
        buffer.set(constants.N, offset); offset += 8;
        buffer.set(constants.R2, offset); offset += 8;
        buffer.set(constants.mont_one, offset); offset += 8;
        buffer[offset++] = constants.n0inv32;
        buffer[offset++] = 0; // padding
        buffer[offset++] = 0; // padding
        buffer[offset++] = 0; // padding

        // Prime powers
        buffer.set(primePowers, offset); offset += pp_count;

        // Curve inputs
        for(const curve of curves){
          buffer.set(curve.A24, offset); offset += 8;
          buffer.set(curve.X1, offset); offset += 8;
        }

        // Reserve space for outputs (will be filled by GPU)
        offset += curvesInChunk * CURVE_OUT_WORDS_PER;

        const payload = {
          data: buffer.buffer.slice(0),
          dims: {
            n: curvesInChunk,
            pp_count,
            total_words: totalWords
          }
        };
        const meta = {
          chunkIndex,
          startCurve,
          endCurve,
          n: curvesInChunk,      // <— add
          pp_count,              // <— add
          total_words: totalWords,
          N: N_hex,
          B1
        };

        yield {
          id: uuidv4(),
          payload,
          meta,
          tCreate: Date.now()
        };
      }
      logger.info(`ECM chunker finished: ${totalChunks} chunks, ${total_curves} total curves`);
    }
  };
}

function asU32View(bin){
  // 1) Pure ArrayBuffer
  if (bin instanceof ArrayBuffer){
    if ((bin.byteLength % 4) !== 0) throw new Error('Result byteLength not multiple of 4');
    return new Uint32Array(bin);
  }
  // 2) Any typed array / DataView
  if (ArrayBuffer.isView(bin)){
    const { buffer, byteOffset, byteLength } = bin;
    if ((byteLength % 4) !== 0) throw new Error('Result view length not multiple of 4');
    // If aligned to 4 bytes, we can make a zero-copy view
    if ((byteOffset % 4) === 0){
      return new Uint32Array(buffer, byteOffset, Math.floor(byteLength / 4));
    }
    // Unaligned: copy into an aligned buffer
    const copy = new Uint8Array(byteLength);
    copy.set(new Uint8Array(buffer, byteOffset, byteLength));
    return new Uint32Array(copy.buffer);
  }
  // 3) Node Buffer (what Socket.IO gives us)
  if (typeof Buffer !== 'undefined' && Buffer.isBuffer(bin)){
    const buf = bin;
    if ((buf.byteLength % 4) !== 0) throw new Error('Result Buffer length not multiple of 4');
    if ((buf.byteOffset % 4) === 0){
      return new Uint32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
    }
    // Unaligned Buffer: realign by copying
    const copy = Buffer.from(buf);
    return new Uint32Array(copy.buffer, copy.byteOffset, Math.floor(copy.byteLength / 4));
  }

  // 4) Last resort: coerce to Buffer then view
  const coerced = Buffer.from(bin);
  if ((coerced.byteLength % 4) !== 0) throw new Error('Result coerced length not multiple of 4');
  return new Uint32Array(coerced.buffer, coerced.byteOffset, Math.floor(coerced.byteLength / 4));
}

export function buildAssembler({ taskId, taskDir, config, inputArgs }){
  const results = [];
  const factorsFound = [];
  const N = hexToBigInt(inputArgs.N);

  // Output paths
  const summaryPath = path.join(taskDir, 'output.summary.json');
  const binaryPath = path.join(taskDir, 'output.bin');

  return {
    integrate({ chunkId, result, meta }){
      // Parse results from WebGPU output
      const u32 = asU32View(result);
        if (u32[0] !== 0x45434D31 /* "ECM1" */){
          logger.warn('Unexpected header magic in chunk', chunkId, 'got', '0x'+u32[0].toString(16));
        }
      // Debug: Check if the buffer looks initialized
      let nonZeroCount = 0;
      for(let i = 0; i < Math.min(100, u32.length); i++) {
        if(u32[i] !== 0) nonZeroCount++;
      }
      console.log(`Buffer non-zero ratio: ${nonZeroCount}/100`);
      const { n, pp_count } = meta;
      logger.debug(`Integrate ${chunkId}: n=${n}, pp_count=${pp_count}`);
      // Find output section in the buffer
      const HEADER_WORDS = 8;
      const CONST_WORDS = 8*3 + 4;
      const CURVE_IN_WORDS_PER = 8*2;
      const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + pad(3)

      const outStart = HEADER_WORDS + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER;
      console.log(`Parsing results starting at word ${outStart}, buffer length: ${u32.length}`);
      const chunkResults = [];
      for(let i = 0; i < n; i++){
        const offset = outStart + i * CURVE_OUT_WORDS_PER;
        const resultLimbs = u32.slice(offset, offset + 8);
        const status = u32[offset + 8];

        // Convert limbs to BigInt
        let factor = 0n;
        for(let j = 7; j >= 0; j--){
          factor = (factor << 32n) | BigInt(resultLimbs[j] >>> 0);
        }

        // Debug log
        if (i < 3 || status === 2) {
          console.log(`Curve ${meta.startCurve + i}: status=${status}, factor=${factor.toString(16)}`);
        }

        const curveIndex = meta.startCurve + i;
        chunkResults.push({ curveIndex, factor, status });

        // Check for non-trivial factors (status === 2 means factor found)
        if(status === 2 && factor > 1n && factor < N){
          const isValid = (N % factor) === 0n;
          console.log(`Factor check: ${factor.toString(16)} divides ${N.toString(16)}? ${isValid}`);
          factorsFound.push({
            curveIndex,
            factor: factor.toString(),
            factorHex: '0x' + factor.toString(16),
            status,
            valid: isValid
          });
        }
      }

      results.push({
        chunkId,
        meta: { ...meta, curvesProcessed: n },
        curves: chunkResults
      });
    },

    finalize(){
      // Create binary output (concatenated curve results)
      const STRIDE_WORDS = 8 + 1; // result(8) + status(1) - no padding in final output
      const totalCurves = results.reduce((sum, r) => sum + r.curves.length, 0);
      const binaryOutput = new Uint32Array(totalCurves * STRIDE_WORDS);

      let writeOffset = 0;
      for(const chunk of results){
        for(const curve of chunk.curves){
          // Write result limbs
          const factor = BigInt(curve.factor);
          for(let i = 0; i < 8; i++){
            binaryOutput[writeOffset++] = Number((factor >> (32n * BigInt(i))) & 0xFFFFFFFFn);
          }
          // Write status
          binaryOutput[writeOffset++] = curve.status;
        }
      }

      // Write binary file
      fs.writeFileSync(binaryPath, Buffer.from(binaryOutput.buffer));

      // Create summary
      const summary = {
        taskId,
        inputN: inputArgs.N,
        B1: inputArgs.B1,
        totalCurves,
        chunksProcessed: results.length,
        factorsFound,
        completedAt: new Date().toISOString()
      };

      fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

      logger.info(`ECM assembly complete: ${totalCurves} curves, ${factorsFound.length} factors found`);

      return {
        summaryPath,
        binaryPath,
        totalCurves,
        factorsFound: factorsFound.length
      };
    }
  };
}