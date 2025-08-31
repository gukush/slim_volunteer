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

function generateRandomCurve(N, seed) {
  // Generate pseudo-random curve parameters (A24, X1) for ECM
  // Using a simple LCG for reproducible randomness
  let rng = seed;
  function rand32() {
    rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF;
    return rng;
  }

  function randBigInt() {
    let result = 0n;
    for (let i = 0; i < 8; i++) {
      result |= BigInt(rand32()) << (32n * BigInt(i));
    }
    return result % N;
  }

  // Generate random A24 and X1 coordinates
  const A24 = randBigInt();
  const X1 = randBigInt();

  return {
    A24: bigIntToLimbs(A24),
    X1: bigIntToLimbs(X1)
  };
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
      const u32 = new Uint32Array(result);
      const { n, pp_count } = meta;

      // Find output section in the buffer
      const HEADER_WORDS = 8;
      const CONST_WORDS = 8*3 + 4;
      const CURVE_IN_WORDS_PER = 8*2;
      const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + pad(3)

      const outStart = HEADER_WORDS + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER;

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

        const curveIndex = meta.startCurve + i;
        chunkResults.push({ curveIndex, factor, status });

        // Check for non-trivial factors (status === 2)
        if(status === 2 && factor > 1n && factor < N){
          const isValid = (N % factor) === 0n;
          factorsFound.push({
            curveIndex,
            factor: factor.toString(),
            factorHex: '0x' + factor.toString(16),
            status,
            valid: isValid
          });
          if(isValid){
            logger.info(`ECM found factor: curve ${curveIndex}, factor=${factor.toString(16)}`);
          }
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