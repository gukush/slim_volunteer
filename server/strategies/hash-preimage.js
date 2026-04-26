import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'hash-preimage';
export const name = 'Hash Pre-image Search (WebGPU SHA-256)';

const UINT64_MOD = 1n << 64n;

export function getClientExecutorInfo(config) {
  const framework = (config?.framework || 'webgpu').toLowerCase();
  if (framework !== 'webgpu') {
    throw new Error('Unsupported framework for hash-preimage: ' + framework);
  }
  return {
    framework: 'webgpu',
    path: 'executors/webgpu-hash-preimage.client.js',
    kernels: ['kernels/webgpu/hash_preimage_sha256.wgsl'],
    schema: { output: 'Uint32Array' },
  };
}

function parseU64(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  if (typeof value === 'bigint') return value;
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error('Nonce values must be non-negative safe integers or strings');
    return BigInt(value);
  }
  const text = String(value).trim().toLowerCase();
  if (!text) return fallback;
  return text.startsWith('0x') ? BigInt(text) : BigInt(text);
}

function normalizeTargetHash(value) {
  if (value === undefined || value === null || value === '') return null;
  const hex = String(value).trim().toLowerCase().replace(/^0x/, '');
  if (!/^[0-9a-f]{64}$/.test(hex)) {
    throw new Error('targetHash must be a 32-byte SHA-256 hex string');
  }
  return hex;
}

function prefixToBuffer(config, inputArgs) {
  const prefixHex = inputArgs.prefixHex ?? config.prefixHex;
  if (prefixHex !== undefined && prefixHex !== null && prefixHex !== '') {
    const hex = String(prefixHex).trim().toLowerCase().replace(/^0x/, '');
    if (hex.length % 2 !== 0 || !/^[0-9a-f]*$/.test(hex)) {
      throw new Error('prefixHex must contain an even number of hex characters');
    }
    return Buffer.from(hex, 'hex');
  }
  const prefix = inputArgs.prefix ?? config.prefix ?? '';
  return Buffer.from(String(prefix), 'utf8');
}

function nonceToParts(nonce) {
  return {
    lo: Number(nonce & 0xFFFFFFFFn) >>> 0,
    hi: Number((nonce >> 32n) & 0xFFFFFFFFn) >>> 0,
  };
}

function nonceBufferLE(nonce) {
  const out = Buffer.alloc(8);
  out.writeUInt32LE(Number(nonce & 0xFFFFFFFFn) >>> 0, 0);
  out.writeUInt32LE(Number((nonce >> 32n) & 0xFFFFFFFFn) >>> 0, 4);
  return out;
}

function countLeadingZeroBits(buf) {
  let bits = 0;
  for (const byte of buf) {
    if (byte === 0) {
      bits += 8;
      continue;
    }
    for (let bit = 7; bit >= 0; bit--) {
      if (((byte >> bit) & 1) === 0) bits++;
      else return bits;
    }
  }
  return bits;
}

function verifyCandidate({ nonce, prefix, targetHash, leadingZeroBits }) {
  const digest = crypto.createHash('sha256').update(prefix).update(nonceBufferLE(nonce)).digest();
  const digestHex = digest.toString('hex');
  const targetOk = targetHash ? digestHex === targetHash : true;
  const zerosOk = leadingZeroBits > 0 ? countLeadingZeroBits(digest) >= leadingZeroBits : true;
  return { ok: targetOk && zerosOk, digestHex };
}

function readResultU32(result) {
  if (result instanceof ArrayBuffer) return new Uint32Array(result);
  if (ArrayBuffer.isView(result)) return new Uint32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (Buffer.isBuffer(result)) return new Uint32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
    const u8 = Uint8Array.from(result.data);
    return new Uint32Array(u8.buffer, u8.byteOffset, Math.floor(u8.byteLength / 4));
  }
  throw new Error('Unsupported hash-preimage result buffer type');
}

export function buildChunker({ taskId, taskDir, K, config, inputArgs }) {
  const targetHash = normalizeTargetHash(inputArgs.targetHash ?? config.targetHash);
  const leadingZeroBits = Number(inputArgs.leadingZeroBits ?? config.leadingZeroBits ?? 0);
  if (!targetHash && leadingZeroBits <= 0) {
    throw new Error('hash-preimage requires targetHash or leadingZeroBits');
  }
  if (!Number.isInteger(leadingZeroBits) || leadingZeroBits < 0 || leadingZeroBits > 256) {
    throw new Error('leadingZeroBits must be an integer in [0, 256]');
  }

  const prefix = prefixToBuffer(config, inputArgs);
  if (prefix.length > 47) {
    throw new Error('prefix must be at most 47 bytes; this WebGPU kernel hashes one SHA-256 block containing prefix + uint64_le(nonce)');
  }

  const startNonce = parseU64(inputArgs.startNonce ?? config.startNonce, 0n);
  let endNonce = parseU64(inputArgs.endNonce ?? config.endNonce, null);
  const totalNonces = parseU64(inputArgs.totalNonces ?? config.totalNonces, null);
  if (endNonce === null) {
    endNonce = startNonce + (totalNonces ?? BigInt(config.chunkSize || 1_000_000));
  }
  if (startNonce < 0n || endNonce < 0n || startNonce >= UINT64_MOD || endNonce > UINT64_MOD || endNonce <= startNonce) {
    throw new Error('Nonce range must satisfy 0 <= startNonce < endNonce <= 2^64');
  }

  const chunkSize = Number(inputArgs.chunkSize ?? config.chunkSize ?? 1_000_000);
  if (!Number.isInteger(chunkSize) || chunkSize <= 0 || chunkSize > 0xFFFFFFFF) {
    throw new Error('chunkSize must be an integer in [1, 2^32-1]');
  }

  const totalChunks = Number((endNonce - startNonce + BigInt(chunkSize) - 1n) / BigInt(chunkSize));
  logger.info(`Hash preimage: ${totalChunks} chunks, prefixBytes=${prefix.length}, range=[${startNonce}, ${endNonce}), target=${targetHash || 'none'}, leadingZeroBits=${leadingZeroBits}`);

  return {
    async *stream() {
      let chunkIndex = 0;
      for (let nonce = startNonce; nonce < endNonce; nonce += BigInt(chunkSize)) {
        const next = nonce + BigInt(chunkSize);
        const count = Number((next > endNonce ? endNonce : next) - nonce);
        const { lo: startLo, hi: startHi } = nonceToParts(nonce);
        yield {
          id: uuidv4(),
          payload: {
            prefix: prefix.buffer.slice(prefix.byteOffset, prefix.byteOffset + prefix.byteLength),
            targetHash,
            leadingZeroBits,
            startLo,
            startHi,
            count,
          },
          meta: {
            chunkIndex,
            startNonce: nonce.toString(),
            count,
            targetHash,
            leadingZeroBits,
          },
          tCreate: Date.now(),
        };
        chunkIndex++;
      }
      logger.info('Hash preimage chunker done');
    },
  };
}

export function buildAssembler({ taskId, taskDir, config, inputArgs }) {
  const targetHash = normalizeTargetHash(inputArgs.targetHash ?? config.targetHash);
  const leadingZeroBits = Number(inputArgs.leadingZeroBits ?? config.leadingZeroBits ?? 0);
  const prefix = prefixToBuffer(config, inputArgs);
  const outPath = path.join(taskDir, 'output.json');
  let found = null;
  let searchedChunks = 0;
  let searchedNonces = 0n;

  return {
    integrate({ result, meta }) {
      searchedChunks++;
      searchedNonces += BigInt(meta.count || 0);
      const words = readResultU32(result);
      if (words.length < 11 || words[0] !== 1) return;

      const nonce = (BigInt(words[2] >>> 0) << 32n) + BigInt(words[1] >>> 0);
      const verified = verifyCandidate({ nonce, prefix, targetHash, leadingZeroBits });
      if (!verified.ok) {
        throw new Error(`Client reported invalid hash preimage nonce ${nonce}`);
      }
      if (!found) {
        found = {
          nonce: nonce.toString(),
          nonceHex: '0x' + nonce.toString(16),
          hash: verified.digestHex,
          chunkIndex: meta.chunkIndex,
        };
      }
    },
    finalize() {
      const summary = {
        found: Boolean(found),
        match: found,
        searchedChunks,
        searchedNonces: searchedNonces.toString(),
        targetHash,
        leadingZeroBits,
      };
      fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));
      return { outPath, found: summary.found, match: summary.match };
    },
  };
}
