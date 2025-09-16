// server/strategies/ecm-stage1.js - Version 3 with Resume Support

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'ecm-stage1';
export const name = 'ECM Stage 1 (WebGPU)';
export const framework = 'webgpu';

export function getClientExecutorInfo(config, inputArgs){
  // Browser clients only support WebGPU for ECM Stage 1
  return {
    framework,
    path: 'executors/webgpu-ecm-stage1.client.js',
    kernels: ['kernels/webgpu/ecm_stage1_webgpu_compute.wgsl'],
    schema: { output: 'Uint32Array' },
  };
}

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

/* ------------------------------- Chunker ---------------------------------- */

export function buildChunker({ taskId, taskDir, K, config, inputArgs, inputFiles }){
  const N_hex = inputArgs.N;
  const B1 = inputArgs.B1 || 50000;
  const chunk_size = inputArgs.chunk_size || 256;
  const total_curves = inputArgs.total_curves || 1024;

  if (!N_hex) throw new Error('ECM requires parameter N (number to factor)');

  const N0 = hexToBigInt(N_hex);
  logger.info(`ECM Stage 1: N=${N0.toString(16)}, B1=${B1}, curves=${total_curves}, chunk_size=${chunk_size}`);

  // Pre-pass: strip tiny primes
  const { reducedN: Nred, smallFactors } = trialDivideSmallPrimes(N0);
  if (Nred !== N0) {
    logger.info(`Trial division removed small factors; N reduced from 0x${N0.toString(16)} to 0x${Nred.toString(16)}`);
  }
  if (Nred === 1n) {
    logger.info('All factors were tiny; skipping GPU stage.');
  }

  // Constants and prime powers for the GPU **use reduced N**
  const constants = computeMontgomeryConstants(Nred);
  const primePowers = generatePrimePowers(B1);

  const totalChunks = (Nred === 1n) ? 0 : Math.ceil(total_curves / chunk_size);

  return {
    async *stream(){
      for(let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++){
        const startCurve = chunkIndex * chunk_size;
        const endCurve = Math.min(startCurve + chunk_size, total_curves);
        const curvesInChunk = endCurve - startCurve;

        // --- header sizes (VERSION 3) ---
        const HEADER_WORDS_V3 = 12;
        const CONST_WORDS = 8*3 + 4;
        const CURVE_OUT_WORDS_PER = 8 + 1 + 3;
        const STATE_WORDS_PER_CURVE = 8 + 8 + 8 + 2; // X + Z + A24 + (sigma, curve_ok)

        // RNG seed (deterministic per chunk)
        const taskSeed64 = BigInt.asUintN(64, (BigInt('0x' + taskId.replace(/-/g,'')) ^ 0x9e3779b97f4a7c15n) + BigInt(chunkIndex));
        const seed_lo = Number(taskSeed64 & 0xffffffffn) >>> 0;
        const seed_hi = Number((taskSeed64 >> 32n) & 0xffffffffn) >>> 0;

        const pp_count = primePowers.length;

        // Calculate total buffer size including state storage
        const totalWords = HEADER_WORDS_V3 + CONST_WORDS + pp_count
                         + curvesInChunk * CURVE_OUT_WORDS_PER
                         + curvesInChunk * STATE_WORDS_PER_CURVE;

        const buffer = new Uint32Array(totalWords);
        let offset = 0;

        // Header v3
        buffer[offset++] = 0x45434D31;    // "ECM1"
        buffer[offset++] = 3;             // version = 3 (resumable)
        buffer[offset++] = 0;
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

        const payload = {
          data: buffer.buffer.slice(0),
          dims: { n: curvesInChunk, pp_count, total_words: totalWords }
        };
        const meta = {
          chunkIndex, startCurve, endCurve,
          n: curvesInChunk, pp_count, total_words: totalWords,
          N: N_hex,
          reducedN: '0x' + Nred.toString(16),
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
      logger.info(`ECM chunker finished: ${totalChunks} chunks, ${total_curves} total curves`);
    }
  };
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
    return new Uint32Array(copy.buffer, copy.byteOffset, Math.floor(copy.byteLength / 4));
  }
  const coerced = Buffer.from(bin);
  if ((coerced.byteLength % 4) !== 0) throw new Error('Result coerced length not multiple of 4');
  return new Uint32Array(coerced.buffer, coerced.byteOffset, Math.floor(coerced.byteLength / 4));
}

/* ------------------------------- Assembler -------------------------------- */

export function buildAssembler({ taskId, taskDir, config, inputArgs }){
  const results = [];
  const rawCurveFinds = [];
  const N0 = hexToBigInt(inputArgs.N);

  // Output paths
  const summaryPath = path.join(taskDir, 'output.summary.json');
  const binaryPath = path.join(taskDir, 'output.bin');

  // Dedup aggregator across chunks
  const bag = new FactorBag(N0);

  return {
    integrate({ chunkId, result, meta }){
      const u32 = asU32View(result);

      if (u32[0] !== 0x45434D31){
        logger.warn('Unexpected header magic in chunk', chunkId, 'got', '0x'+u32[0].toString(16));
      }

      let nonZeroCount = 0;
      for(let i = 0; i < Math.min(100, u32.length); i++) if(u32[i] !== 0) nonZeroCount++;
      console.log(`Buffer non-zero ratio: ${nonZeroCount}/100`);

      const { n, pp_count } = meta;
      logger.debug(`Integrate ${chunkId}: n=${n}, pp_count=${pp_count}`);

      // Version-aware assembler
      const version = u32[1] >>> 0;
      const HEADER_WORDS_V1 = 8;
      const HEADER_WORDS_V2 = 12;
      const HEADER_WORDS_V3 = 12;
      const CONST_WORDS = 8*3 + 4;
      const CURVE_IN_WORDS_PER = 8*2;         // only used for v1
      const CURVE_OUT_WORDS_PER = 8 + 1 + 3;

      let outStart;
      if (version >= 2) {
        // Version 2 and 3 have same output location
        outStart = HEADER_WORDS_V3 + CONST_WORDS + pp_count;
      } else {
        outStart = HEADER_WORDS_V1 + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER;
      }
      console.log(`Parsing results starting at word ${outStart}, buffer length: ${u32.length}, version: ${version}`);

      const chunkResults = [];
      let factorsFoundInChunk = 0;

      for(let i = 0; i < n; i++){
        const base = outStart + i * CURVE_OUT_WORDS_PER;
        const factor = limbsToBigLE(u32, base, 8);
        const status = u32[base + 8] >>> 0;
        const curveIndex = meta.startCurve + i;

        // Log interesting results
        if (status === 2 || (i < 3 && meta.chunkIndex === 0)) {
          console.log(`Curve ${curveIndex}: status=${status}, factor=${factor.toString(16)}`);
        }

        chunkResults.push({ curveIndex, factor, status });

        // Record raw per-curve find
        if (status === 2 && factor > 1n && factor < N0){
          factorsFoundInChunk++;
          const isValid = (N0 % factor) === 0n;
          rawCurveFinds.push({
            curveIndex,
            factor: factor.toString(),
            factorHex: '0x' + factor.toString(16),
            status,
            valid: isValid
          });
          // Add to dedup bag
          bag.addCandidate(factor);
        }
      }

      if (factorsFoundInChunk > 0) {
        logger.info(`🎯 Found ${factorsFoundInChunk} factor(s) in chunk ${chunkId}`);
      }

      results.push({
        chunkId,
        meta: { ...meta, curvesProcessed: n },
        curves: chunkResults
      });
    },

    finalize(){
      // Create binary output (concatenated curve results)
      const STRIDE_WORDS = 8 + 1; // result(8) + status(1)
      const totalCurves = results.reduce((sum, r) => sum + r.curves.length, 0);
      const binaryOutput = new Uint32Array(totalCurves * STRIDE_WORDS);

      let writeOffset = 0;
      for(const chunk of results){
        for(const curve of chunk.curves){
          const factor = BigInt(curve.factor);
          for(let i = 0; i < 8; i++){
            binaryOutput[writeOffset++] = Number((factor >> (32n * BigInt(i))) & 0xFFFFFFFFn);
          }
          binaryOutput[writeOffset++] = curve.status >>> 0;
        }
      }
      fs.writeFileSync(binaryPath, Buffer.from(binaryOutput.buffer));

      // Combine with small-prime prepass (re-run here for robustness/reporting)
      const { reducedN, smallFactors } = trialDivideSmallPrimes(N0);

      // Seed bag multiplicities with small primes (remove them from remaining)
      bag.remaining = N0;
      for (const [p, k] of smallFactors.entries()){
        let r = bag.remaining;
        for (let i = 0; i < k; i++) r /= p;
        bag.remaining = r;
        bag.mult.set(p.toString(), k);
        const rec = bag.hits.get(p.toString()) || { n: p, hits: 0 };
        rec.hits += k;
        bag.hits.set(p.toString(), rec);
      }

      // Resolve ECM candidates
      const finalBag = bag.finalize();

      // Pretty summary lines
      const primeLines = [];
      for (const [pStr, k] of [...bag.mult].sort((a,b)=> (BigInt(a[0]) > BigInt(b[0]) ? 1 : -1))) {
        const p = BigInt(pStr);
        const hits = bag.hits.get(pStr)?.hits ?? 0;
        primeLines.push(`${p}  (hits=${hits}, multiplicity=${k})`);
      }

      const candidateLines = [];
      for (const [key, rec] of bag.hits) {
        if (!bag.mult.has(key)) {
          candidateLines.push(`${rec.n}  (hits=${rec.hits}, multiplicity=0)`);
        }
      }

      const cofactor = finalBag.cofactor;
      const cofactorLine = (cofactor > 1n)
        ? `cofactor=${cofactor} (0x${cofactor.toString(16)})`
        : 'fully factored (cofactor=1)';

      const factorSummaryText =
        [...primeLines, ...candidateLines, cofactorLine].join('\n');

      logger.info('ECM summary:\n' + factorSummaryText);

      // JSON-friendly summaries
      const primeFactorsJson = [...bag.mult]
        .sort((a,b)=> (BigInt(a[0]) > BigInt(b[0]) ? 1 : -1))
        .map(([pStr, k]) => ({
          factor: pStr,
          factorHex: '0x' + BigInt(pStr).toString(16),
          multiplicity: k,
          hits: bag.hits.get(pStr)?.hits ?? 0
        }));

      const ecmCandidatesJson = [...bag.hits]
        .filter(([key]) => !bag.mult.has(key))
        .map(([key, rec]) => ({
          factor: rec.n.toString(),
          factorHex: '0x' + rec.n.toString(16),
          hits: rec.hits,
          multiplicity: 0
        }));

      const smallPrimeFactorsJson = [...smallFactors.entries()].map(([p,k]) => ({
        factor: p.toString(),
        factorHex: '0x' + p.toString(16),
        multiplicity: k
      }));

      // Original per-curve list is kept for debugging
      const summary = {
        taskId,
        inputN: inputArgs.N,
        B1: inputArgs.B1,
        totalCurves,
        chunksProcessed: results.length,
        factorsFound: rawCurveFinds,
        factorSummary: {
          primeFactors: primeFactorsJson,
          ecmCandidates: ecmCandidatesJson,
          cofactor: cofactor.toString(),
          cofactorHex: '0x' + cofactor.toString(16),
          text: factorSummaryText
        },
        smallPrimePrepass: {
          factors: smallPrimeFactorsJson,
          reducedN: reducedN.toString(),
          reducedNHex: '0x' + reducedN.toString(16)
        },
        completedAt: new Date().toISOString()
      };

      fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
      logger.info(`ECM assembly complete: ${totalCurves} curves, ${rawCurveFinds.length} raw factor hits`);

      return {
        summaryPath,
        binaryPath,
        totalCurves,
        factorsFound: rawCurveFinds.length
      };
    },
    cleanup(){
      // ECM assembler doesn't maintain file descriptors, just clear memory
      try {
        results.length = 0;
        rawCurveFinds.length = 0;
        bag.hits.clear();
        bag.mult.clear();
        logger.info(`ECM-stage1 assembler cleanup: cleared memory structures`);
      } catch (e) {
        logger.warn(`ECM-stage1 assembler cleanup failed:`, e.message);
      }
    }
  };
}

// Kill-switch support: Calculate total chunks deterministically
export function getTotalChunks(config, inputArgs) {
  const { N, B1 } = config;
  const { chunk_size = 1000 } = config;

  // Use the same logic as in buildChunker
  const Nred = BigInt(N);
  const B1Big = BigInt(B1);
  const total_curves = Number(B1Big * B1Big / (2n * Nred));
  const totalChunks = (Nred === 1n) ? 0 : Math.ceil(total_curves / chunk_size);

  logger.info(`ECM-stage1 getTotalChunks: N=${N}, B1=${B1}, chunk_size=${chunk_size} -> ${totalChunks} chunks`);
  return totalChunks;
}