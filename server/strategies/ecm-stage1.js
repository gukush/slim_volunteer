// server/strategies/ecm-stage1.js

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

/* ------------------------- Small primes (≤ 7919) ------------------------- */
/* Removed `1`, kept first ~1000 primes. Stored as Number for compactness.  */
const SMALL_PRIMES = [
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
  157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
  239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
  331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
  421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
  509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
  613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
  709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
  821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
  919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
  1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097,
  1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201,
  1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
  1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409,
  1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487,
  1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579,
  1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667,
  1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777,
  1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
  1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993,
  1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083,
  2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179,
  2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287,
  2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381,
  2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473,
  2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609,
  2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693,
  2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789,
  2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887,
  2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001,
  3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119,
  3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229,
  3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331,
  3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457,
  3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541,
  3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637,
  3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739,
  3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853,
  3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947,
  3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073,
  4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177,
  4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273,
  4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409,
  4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517,
  4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639,
  4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733,
  4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871,
  4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969,
  4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077,
  5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189,
  5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309,
  5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431,
  5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521,
  5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651,
  5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743,
  5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851,
  5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981,
  5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091,
  6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211,
  6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311,
  6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397,
  6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553,
  6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673,
  6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781,
  6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883,
  6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991,
  6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121,
  7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237,
  7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369,
  7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507,
  7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589,
  7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699,
  7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829,
  7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919
];

function trialDivideSmallPrimes(Nbig, maxPrime = 7919n) {
  const factors = new Map(); // BigInt -> multiplicity
  let N = parseBig(Nbig);

  for (const pn of SMALL_PRIMES) {
    const p = BigInt(pn);
    if (p > maxPrime) break;
    if (N % p === 0n) {
      let k = 0n;
      while (N % p === 0n) { N /= p; k++; }
      factors.set(p, Number(k)); // store multiplicity as Number for JSON friendliness
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

/* -------------------------- Curve generation (ECM) ------------------------ */

function mod(a, n){ a %= n; return a < 0n ? a + n : a; }

function egcd(a, b){
  a = mod(a, b);
  let old_r = b, r = a;
  let old_s = 1n, s = 0n;
  let old_t = 0n, t = 1n;
  while (r !== 0n) {
    const q = old_r / r;
    [old_r, r] = [r, old_r - q*r];
    [old_s, s] = [s, old_s - q*s];
    [old_t, t] = [t, old_t - q*t];
  }
  return [old_r, old_s, old_t];
}

function modInv(a, n){
  a = mod(a, n);
  if (a === 0n) return null;
  const [g, x] = egcd(a, n);
  if (g !== 1n) return null;
  return mod(x, n);
}

// Suyama parametrization for (A24, X1) on a Montgomery curve
function generateRandomCurve(N, seed) {
  let rng = BigInt(seed) & 0x7fffffffn;
  function rand32(){ rng = (rng * 1103515245n + 12345n) & 0x7fffffffn; return rng; }
  function nextSigma(){
    let s = 0n;
    for (let i=0;i<3;i++) s = (s << 21n) | rand32();
    return mod(s, N);
  }

  const inv4 = modInv(4n, N);
  for (let tries = 0; tries < 32; tries++){
    const sigma = nextSigma();
    if (sigma===0n) continue;
    if (sigma===1n || sigma===N-1n) continue;
    if (sigma===2n || sigma===N-2n) continue;

    const u = mod(sigma*sigma - 5n, N);
    const v = mod(4n*sigma, N);

    const invu = modInv(u, N);
    const invv = modInv(v, N);
    if (!invu || !invv || !inv4) continue;

    const vm_u   = mod(v - u, N);
    const num    = mod(vm_u*vm_u % N * vm_u % N * mod(3n*u + v, N), N);
    const den    = mod(4n * (u*u % N * u % N) % N * v, N);
    const invDen = modInv(den, N);
    if (!invDen) continue;

    const A      = mod(num * invDen - 2n, N);
    const A24    = mod((A + 2n) * inv4, N);

    const u3     = mod(u*u % N * u % N, N);
    const vInv   = modInv(v, N);
    if (!vInv) continue;
    const vInv3  = mod(vInv*vInv % N * vInv % N, N);
    const X1     = mod(u3 * vInv3, N);

    return { A24: bigIntToLimbs(A24), X1: bigIntToLimbs(X1) };
  }
  return generateRandomCurve(N, Number((BigInt(seed) + 777n) % 0x7fffffffn));
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

        // Deterministic curve params against Nred
        const curves = [];
        for(let i = 0; i < curvesInChunk; i++){
          const curveIndex = startCurve + i;
          curves.push(generateRandomCurve(Nred, curveIndex + 12345));
        }

        // Layout: header + constants + primePowers + curveInputs + curveOutputs
        const HEADER_WORDS = 8;
        const CONST_WORDS = 8*3 + 4;             // N(8) + R2(8) + mont_one(8) + n0inv32 + pad(3)
        const pp_count = primePowers.length;
        const CURVE_IN_WORDS_PER = 8*2;          // A24(8) + X1(8)
        const CURVE_OUT_WORDS_PER = 8 + 1 + 3;   // result(8) + status + pad(3)

        const totalWords = HEADER_WORDS + CONST_WORDS + pp_count +
                          curvesInChunk * CURVE_IN_WORDS_PER +
                          curvesInChunk * CURVE_OUT_WORDS_PER;

        const buffer = new Uint32Array(totalWords);
        let offset = 0;

        // Header: magic, version, reserved, reserved, pp_count, n_curves, reserved, reserved
        buffer[offset++] = 0x45434D31; // "ECM1"
        buffer[offset++] = 1;
        buffer[offset++] = 0;
        buffer[offset++] = 0;
        buffer[offset++] = pp_count;
        buffer[offset++] = curvesInChunk;
        buffer[offset++] = 0;
        buffer[offset++] = 0;

        // Constants for Nred
        buffer.set(constants.N, offset);         offset += 8;
        buffer.set(constants.R2, offset);        offset += 8;
        buffer.set(constants.mont_one, offset);  offset += 8;
        buffer[offset++] = constants.n0inv32;
        buffer[offset++] = 0; // pad
        buffer[offset++] = 0; // pad
        buffer[offset++] = 0; // pad

        // Prime powers
        buffer.set(primePowers, offset);         offset += pp_count;

        // Curve inputs
        for(const curve of curves){
          buffer.set(curve.A24, offset);         offset += 8;
          buffer.set(curve.X1,  offset);         offset += 8;
        }

        // Reserve output space
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
          n: curvesInChunk,
          pp_count,
          total_words: totalWords,
          N: N_hex,                               // original N (for reporting)
          reducedN: '0x' + Nred.toString(16),     // reduced N used by GPU
          B1,
          smallPrimeFactors: [...smallFactors.entries()].map(([p,k]) => [ '0x'+p.toString(16), k ])
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
  const rawCurveFinds = [];                   // per-curve log (unchanged)
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

      const HEADER_WORDS = 8;
      const CONST_WORDS = 8*3 + 4;
      const CURVE_IN_WORDS_PER = 8*2;
      const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status + pad(3)

      const outStart = HEADER_WORDS + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER;
      console.log(`Parsing results starting at word ${outStart}, buffer length: ${u32.length}`);

      const chunkResults = [];
      for(let i = 0; i < n; i++){
        const base = outStart + i * CURVE_OUT_WORDS_PER;
        const factor = limbsToBigLE(u32, base, 8);
        const status = u32[base + 8] >>> 0;
        const curveIndex = meta.startCurve + i;

        if (i < 3 || status === 2) {
          console.log(`Curve ${curveIndex}: status=${status}, factor=${factor.toString(16)}`);
        }

        chunkResults.push({ curveIndex, factor, status });

        // Record raw per-curve find
        if (status === 2 && factor > 1n && factor < N0){
          const isValid = (N0 % factor) === 0n;
          rawCurveFinds.push({
            curveIndex,
            factor: factor.toString(),
            factorHex: '0x' + factor.toString(16),
            status,
            valid: isValid
          });
          // Add to dedup bag regardless of validity flag (bag normalizes with gcd)
          bag.addCandidate(factor);
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
        // Treat trial-division as k "hits" for the small factors (optional)
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
        factorsFound: rawCurveFinds,                 // raw per-curve records
        factorSummary: {
          primeFactors: primeFactorsJson,           // deduped with multiplicity
          ecmCandidates: ecmCandidatesJson,         // deduped hits that didn't divide remaining
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
    }
  };
}
