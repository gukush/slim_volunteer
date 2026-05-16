import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'goldbach-verification';
export const name = 'Goldbach Conjecture Verification (WebGPU)';

const SMALL_PRIMES = buildSmallPrimes(65535);

export function getClientExecutorInfo(config) {
  const framework = (config?.framework || 'webgpu').toLowerCase();
  if (framework !== 'webgpu') {
    throw new Error('Unsupported framework for goldbach-verification: ' + framework);
  }
  return {
    framework: 'webgpu',
    path: 'executors/webgpu-goldbach-verification.client.js',
    kernels: ['kernels/webgpu/goldbach_verify.wgsl'],
    schema: { output: 'Uint32Array' },
  };
}

function parseU32(value, name) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 0 || n > 0xffffffff) {
    throw new Error(`${name} must be a uint32 integer`);
  }
  return n >>> 0;
}

function normalizeNumbers(config, inputArgs) {
  const rawNumbers = inputArgs.numbers ?? config.numbers;
  if (rawNumbers !== undefined && rawNumbers !== null && rawNumbers !== '') {
    const values = Array.isArray(rawNumbers)
      ? rawNumbers
      : String(rawNumbers).split(',').map((x) => x.trim()).filter(Boolean);
    if (values.length === 0) throw new Error('numbers must not be empty');
    return values.map((value, i) => parseU32(value, `numbers[${i}]`));
  }

  const start = parseU32(inputArgs.start ?? config.start ?? 4, 'start');
  const end = parseU32(inputArgs.end ?? config.end, 'end');
  if (end < start) throw new Error('end must be >= start');

  const numbers = [];
  let current = start % 2 === 0 ? start : start + 1;
  if (current < 4) current = 4;
  for (; current <= end; current += 2) {
    numbers.push(current >>> 0);
  }
  if (numbers.length === 0) throw new Error('range produced no even numbers >= 4');
  return numbers;
}

function buildSmallPrimes(limit) {
  const composite = new Uint8Array(limit + 1);
  const primes = [];
  for (let n = 2; n <= limit; n++) {
    if (composite[n]) continue;
    primes.push(n);
    if (n <= Math.floor(limit / n)) {
      for (let m = n * n; m <= limit; m += n) {
        composite[m] = 1;
      }
    }
  }
  return Uint32Array.from(primes);
}

function isPrime(n, smallPrimes = SMALL_PRIMES) {
  if (n < 2) return false;
  for (const p of smallPrimes) {
    if (p > Math.floor(n / p)) break;
    if (n % p === 0) return n === p;
  }
  return true;
}

function findGoldbachWitness(n, smallPrimes = SMALL_PRIMES) {
  if (n === 4) return 2;
  for (const p of smallPrimes) {
    if (p === 2) continue;
    if (p > Math.floor(n / 2)) break;
    if (isPrime(n - p, smallPrimes)) return p;
  }
  return 0;
}

function readResultU32(result) {
  if (result instanceof ArrayBuffer) return new Uint32Array(result);
  if (ArrayBuffer.isView(result)) return new Uint32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (Buffer.isBuffer(result)) return new Uint32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
    const u8 = Uint8Array.from(result.data);
    return new Uint32Array(u8.buffer, u8.byteOffset, Math.floor(u8.byteLength / 4));
  }
  throw new Error('Unsupported goldbach-verification result buffer type');
}

function minMax(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { min, max };
}

export function buildChunker({ taskId, taskDir, K, config, inputArgs }) {
  const numbers = normalizeNumbers(config, inputArgs);
  const chunkSize = Number(inputArgs.chunkSize ?? config.chunkSize ?? 1024);
  if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
    throw new Error('chunkSize must be a positive integer');
  }

  const totalChunks = Math.ceil(numbers.length / chunkSize);
  logger.info(`Goldbach verification: ${numbers.length} numbers, ${totalChunks} chunks, chunkSize=${chunkSize}`);

  return {
    async *stream() {
      for (let offset = 0, chunkIndex = 0; offset < numbers.length; offset += chunkSize, chunkIndex++) {
        const sublist = numbers.slice(offset, offset + chunkSize);
        const typed = Uint32Array.from(sublist);
        yield {
          id: uuidv4(),
          payload: {
            numbers: typed.buffer.slice(typed.byteOffset, typed.byteOffset + typed.byteLength),
            count: typed.length,
          },
          meta: {
            chunkIndex,
            offset,
            count: typed.length,
            minNumber: sublist[0],
            maxNumber: sublist[sublist.length - 1],
          },
          tCreate: Date.now(),
        };
      }
      logger.info('Goldbach verification chunker done');
    },
  };
}

export function buildAssembler({ taskId, taskDir, config, inputArgs }) {
  const numbers = normalizeNumbers(config, inputArgs);
  const outPath = path.join(taskDir, 'output.json');
  let checkedNumbers = 0;
  let failed = null;

  return {
    integrate({ result, meta }) {
      checkedNumbers += Number(meta.count || 0);
      const words = readResultU32(result);
      if (words.length < 6 || words[0] !== 1) return;

      const localIndex = words[1] >>> 0;
      const n = words[2] >>> 0;
      const reason = words[4] >>> 0;
      const globalIndex = Number(meta.offset || 0) + localIndex;
      const expected = numbers[globalIndex];
      if (expected !== n) {
        throw new Error(`Client reported Goldbach failure for unexpected number ${n}; expected ${expected}`);
      }

      const witness = findGoldbachWitness(n);
      const invalidInput = n <= 2 || n % 2 !== 0;
      if (!invalidInput && witness !== 0) {
        throw new Error(`Client reported invalid Goldbach failure for ${n}; server found witness ${witness}+${n - witness}`);
      }
      if (!failed) {
        failed = {
          number: n,
          globalIndex,
          chunkIndex: meta.chunkIndex,
          reason: reason === 1 ? 'invalid-input' : 'no-prime-pair',
        };
      }
    },
    finalize() {
      const { min, max } = minMax(numbers);
      const summary = {
        valid: !failed,
        failed,
        checkedNumbers,
        totalNumbers: numbers.length,
        minNumber: min,
        maxNumber: max,
      };
      fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));
      return { outPath, valid: summary.valid, checkedNumbers };
    },
  };
}
