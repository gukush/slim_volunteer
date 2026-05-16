import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'pollard-pminus1';
export const name = 'Pollard p-1 Factorization (WebGPU)';

const MAGIC = 0x314d5031; // "1MP1" little-endian marker for PM1.

export function getClientExecutorInfo(config) {
  const framework = (config?.framework || 'webgpu').toLowerCase();
  if (framework !== 'webgpu') {
    throw new Error('Unsupported framework for pollard-pminus1: ' + framework);
  }
  return {
    framework: 'webgpu',
    path: 'executors/webgpu-pollard-pminus1.client.js',
    kernels: [
      'kernels/webgpu/pollard_pminus1_batched.wgsl',
      'kernels/webgpu/pollard_pminus1.wgsl',
    ],
    schema: { output: 'Uint32Array' },
  };
}

function parseBig(value, name = 'value') {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer`);
    return BigInt(value);
  }
  if (typeof value === 'string' && value.trim()) {
    const s = value.trim().toLowerCase();
    return s.startsWith('0x') ? BigInt(s) : BigInt(s);
  }
  throw new Error(`Cannot parse ${name}`);
}

function bigIntToLimbs(n, limbCount = 8) {
  const limbs = new Uint32Array(limbCount);
  let x = n;
  for (let i = 0; i < limbCount; i++) {
    limbs[i] = Number(x & 0xffffffffn) >>> 0;
    x >>= 32n;
  }
  return limbs;
}

function limbsToBigLE(u32, start, nWords = 8) {
  let x = 0n;
  for (let i = nWords - 1; i >= 0; i--) {
    x = (x << 32n) + BigInt(u32[start + i] >>> 0);
  }
  return x;
}

function gcdBig(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

function modInverse(a, m) {
  let t = 0n;
  let newT = 1n;
  let r = m;
  let newR = ((a % m) + m) % m;
  while (newR !== 0n) {
    const q = r / newR;
    [t, newT] = [newT, t - q * newT];
    [r, newR] = [newR, r - q * newR];
  }
  if (r !== 1n) throw new Error('Modular inverse does not exist');
  return t < 0n ? t + m : t;
}

function computeMontgomeryConstants(N) {
  const R = 1n << 256n;
  const R2 = (R * R) % N;
  const montOne = R % N;
  const n0 = N & 0xffffffffn;
  const n0inv = modInverse(n0, 1n << 32n);
  return {
    N: bigIntToLimbs(N),
    R2: bigIntToLimbs(R2),
    montOne: bigIntToLimbs(montOne),
    n0inv32: Number((-n0inv) & 0xffffffffn) >>> 0,
  };
}

function generatePrimePowers(B1) {
  const limit = Number(B1);
  if (!Number.isInteger(limit) || limit < 2 || limit > 0xffffffff) {
    throw new Error('B1 must be an integer in [2, 2^32-1]');
  }
  const sieve = new Uint8Array(limit + 1);
  const powers = [];
  for (let p = 2; p <= limit; p++) {
    if (sieve[p]) continue;
    if (p <= Math.floor(limit / p)) {
      for (let m = p * p; m <= limit; m += p) sieve[m] = 1;
    }
    let pk = p;
    while (pk <= Math.floor(limit / p)) pk *= p;
    powers.push(pk >>> 0);
  }
  return Uint32Array.from(powers);
}

function readResultU32(result) {
  if (result instanceof ArrayBuffer) return new Uint32Array(result);
  if (ArrayBuffer.isView(result)) return new Uint32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (Buffer.isBuffer(result)) return new Uint32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
    const u8 = Uint8Array.from(result.data);
    return new Uint32Array(u8.buffer, u8.byteOffset, Math.floor(u8.byteLength / 4));
  }
  throw new Error('Unsupported pollard-pminus1 result buffer type');
}

function normalizeInput(config, inputArgs) {
  const B1 = Number(inputArgs.B1 ?? config.B1 ?? 10000);
  const startBase = Number(inputArgs.startBase ?? config.startBase ?? 2);
  const totalBases = Number(inputArgs.totalBases ?? config.totalBases ?? 64);
  const chunkSize = Number(inputArgs.chunkSize ?? config.chunkSize ?? 64);
  for (const [name, value] of [['startBase', startBase], ['totalBases', totalBases], ['chunkSize', chunkSize]]) {
    if (!Number.isInteger(value) || value <= 0 || value > 0xffffffff) {
      throw new Error(`${name} must be an integer in [1, 2^32-1]`);
    }
  }
  if (startBase < 2) throw new Error('startBase must be >= 2');
  if (startBase + totalBases - 1 > 0xffffffff) throw new Error('base range must fit in uint32');

  let ns0 = [];
  const batchRaw = inputArgs.batch ?? config.batch;
  if (batchRaw !== undefined && batchRaw !== null) {
    const arr = Array.isArray(batchRaw) ? batchRaw : String(batchRaw).split(',').map((x) => x.trim()).filter(Boolean);
    ns0 = arr.map((s, i) => parseBig(s, `batch[${i}]`));
  } else {
    const N0 = parseBig(inputArgs.N ?? config.N, 'N');
    ns0 = [N0];
  }
  for (let i = 0; i < ns0.length; i++) {
    if (ns0[i] < 4n) throw new Error(`N[${i}] must be >= 4`);
    if (ns0[i] >= (1n << 256n)) throw new Error(`N[${i}] must fit in 256 bits`);
  }
  return { ns0, B1, startBase, totalBases, chunkSize };
}

function buildPayload({ ns, B1, baseStart, nBases }) {
  const primePowers = generatePrimePowers(B1);
  const HEADER_WORDS = 8;
  const CONST_WORDS = 8 * 3 + 4;
  const OUT_WORDS_PER_BASE = 12;
  const numNs = ns.length;
  const totalWords = HEADER_WORDS + numNs * CONST_WORDS + primePowers.length + numNs * nBases * OUT_WORDS_PER_BASE;
  const buffer = new Uint32Array(totalWords);
  let offset = 0;
  buffer[offset++] = MAGIC;
  buffer[offset++] = 2; // batched version
  buffer[offset++] = primePowers.length;
  buffer[offset++] = nBases;
  buffer[offset++] = baseStart >>> 0;
  buffer[offset++] = numNs;
  buffer[offset++] = 0;
  buffer[offset++] = 0;
  for (const N of ns) {
    const constants = computeMontgomeryConstants(N);
    buffer.set(constants.N, offset); offset += 8;
    buffer.set(constants.R2, offset); offset += 8;
    buffer.set(constants.montOne, offset); offset += 8;
    buffer[offset++] = constants.n0inv32;
    buffer[offset++] = 0; buffer[offset++] = 0; buffer[offset++] = 0;
  }
  buffer.set(primePowers, offset);
  return {
    data: buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
    nBases,
    numNs,
    ppCount: primePowers.length,
    totalWords,
  };
}

export function buildChunker({ taskId, taskDir, K, config, inputArgs }) {
  const { ns0, B1, startBase, totalBases, chunkSize } = normalizeInput(config, inputArgs);
  const ns = [];
  const evenFactors = [];
  for (let i = 0; i < ns0.length; i++) {
    const N0 = ns0[i];
    const factor2 = N0 % 2n === 0n ? 2n : null;
    const N = factor2 ? N0 / 2n : N0;
    if (N % 2n === 0n) throw new Error(`N[${i}]: after removing one factor 2, reduced N is still even; use trial division first`);
    ns.push(N);
    evenFactors.push(factor2 ? { index: i, N0, factor: '2', factorHex: '0x2', source: 'even-prepass' } : null);
  }
  const totalChunks = Math.ceil(totalBases / chunkSize);
  logger.info(`Pollard p-1: batch=${ns0.length} numbers, B1=${B1}, bases=${totalBases}, chunkSize=${chunkSize}`);

  return {
    async *stream() {
      for (let offset = 0, chunkIndex = 0; offset < totalBases; offset += chunkSize, chunkIndex++) {
        const nBases = Math.min(chunkSize, totalBases - offset);
        const baseStart = startBase + offset;
        const payload = buildPayload({ ns, B1, baseStart, nBases });
        yield {
          id: uuidv4(),
          payload,
          meta: {
            chunkIndex,
            baseStart,
            nBases,
            numNs: ns.length,
            ppCount: payload.ppCount,
            ns: ns0.map((n) => '0x' + n.toString(16)),
            B1,
          },
          tCreate: Date.now(),
        };
      }
      logger.info(`Pollard p-1 chunker done: ${totalChunks} chunks`);
    },
  };
}

export function buildAssembler({ taskId, taskDir, config, inputArgs }) {
  const { ns0, B1, startBase, totalBases, chunkSize } = normalizeInput(config, inputArgs);
  const outPath = path.join(taskDir, 'output.json');
  const perN = ns0.map((N0) => {
    const found = [];
    const evenFactor = N0 % 2n === 0n ? {
      factor: '2',
      factorHex: '0x2',
      base: null,
      chunkIndex: null,
      source: 'even-prepass',
    } : null;
    if (evenFactor) found.push(evenFactor);
    return { N0, found };
  });
  let chunksProcessed = 0;
  let basesProcessed = 0;

  return {
    integrate({ result, meta }) {
      chunksProcessed++;
      basesProcessed += Number(meta.nBases || 0);
      const u32 = readResultU32(result);
      if (u32[0] !== MAGIC) throw new Error(`Unexpected Pollard p-1 result magic 0x${(u32[0] >>> 0).toString(16)}`);
      const ppCount = u32[2] >>> 0;
      const nBases = u32[3] >>> 0;
      const numNs = u32[5] >>> 0;
      const CONST_WORDS = 8 * 3 + 4;
      const outStart = 8 + numNs * CONST_WORDS + ppCount;

      for (let n = 0; n < numNs; n++) {
        const { N0, found } = perN[n];
        const reducedN = N0 % 2n === 0n ? N0 / 2n : N0;
        const nOutStart = outStart + n * nBases * 12;
        for (let i = 0; i < nBases; i++) {
          const base = nOutStart + i * 12;
          const factor = limbsToBigLE(u32, base, 8);
          const status = u32[base + 8] >>> 0;
          const baseValue = u32[base + 9] >>> 0;
          if (status !== 2) continue;
          const g = gcdBig(factor, N0);
          if (g <= 1n || g >= N0 || N0 % g !== 0n) {
            throw new Error(`Client reported invalid Pollard p-1 factor ${factor} for N[${n}]`);
          }
          if (!found.some((x) => x.factor === g.toString())) {
            found.push({
              factor: g.toString(),
              factorHex: '0x' + g.toString(16),
              base: baseValue,
              chunkIndex: meta.chunkIndex,
              source: 'pollard-pminus1',
            });
          }
        }
      }
    },
    finalize() {
      const summary = {
        taskId,
        batchSize: ns0.length,
        results: perN.map(({ N0, found }) => ({
          N: '0x' + N0.toString(16),
          found: found.length > 0,
          factors: found,
        })),
        B1,
        startBase,
        totalBases,
        chunkSize,
        chunksProcessed,
        basesProcessed,
        completedAt: new Date().toISOString(),
      };
      fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));
      const anyFound = perN.some((x) => x.found.length > 0);
      return { outPath, found: anyFound, results: summary.results };
    },
  };
}

export function getTotalChunks(config, inputArgs) {
  const { ns0, totalBases, chunkSize } = normalizeInput(config || {}, inputArgs || {});
  const allEven = ns0.every((N0) => N0 % 2n === 0n);
  if (allEven) return 0;
  return Math.ceil(totalBases / chunkSize);
}
