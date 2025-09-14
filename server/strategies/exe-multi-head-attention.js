// File: strategies/exe-multi-head-attention.js
// AOT native binary variant of Multi-Head Attention.
// Sends one binary executable (provided in config) to native clients.
// Each chunk payload is a compact byte list in the order:
//   UNIFORMS (i32 seq_len,d_k,d_v) → Q → K → V
//
// Config example:
//   {
//     seq_len: 2048,
//     d_model: 768,
//     num_heads: 12,
//     backend: "cuda",
//     binary: "binaries/native_cuda_multi_head_attention", // path on server
//     program: "native_cuda_multi_head_attention"          // optional program name
//   }

import fs from 'fs';
import path from 'path';
import * as baseMHA from './multi-head-attention.js';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id   = 'exe-multi-head-attention';
export const name = 'Multi-Head Attention (Native AOT Binary)';

function getArtifacts(config = {}) {
  const backend = String(config.backend || 'cuda').toLowerCase();
  if (!['cuda'].includes(backend)) {
    throw new Error(`exe-multi-head-attention only supports backend=cuda (got: ${backend})`);
  }

  // Default location similar to exe-block-matmul-flex
  const defaultBinary = 'binaries/native_cuda_multi_head_attention';
  const rel = config.binary || defaultBinary;
  const abs = path.isAbsolute(rel) ? rel : path.join(process.cwd(), rel);

  const bytes = fs.readFileSync(abs).toString('base64');
  const artifactName = config.program || path.basename(rel);

  return [{
    type: 'binary',
    name: artifactName,
    program: artifactName,
    backend,
    bytes,
    exec: true
  }];
}

export function getClientExecutorInfo(config = {}) {
  return {
    framework: 'exe',
    kernels: [],
    schema: {
      order: ['UNIFORMS','INPUTS','OUTPUTS'],
      uniforms: [
        { name: 'seq_len', type: 'i32' },
        { name: 'd_k',     type: 'i32' },
        { name: 'd_v',     type: 'i32' },
      ],
      inputs: [
        { name: 'Q', type: 'f32' },
        { name: 'K', type: 'f32' },
        { name: 'V', type: 'f32' },
      ],
      outputs: [
        { name: 'O', type: 'f32' },
      ]
    },
    artifacts: getArtifacts(config)
  };
}

// Wrap base chunker to repack payload using common protocol for 'exe' path.
export function buildChunker(args) {
  const inner = baseMHA.buildChunker(args);

  return {
    async *stream() {
      for await (const ch of inner.stream()) {
        const { payload, meta } = ch;
        const { seq_len, d_k, d_v } = payload.dims || meta || {};

        // uniforms -> Int32Array -> Uint8Array -> JSON array
        const u8U = new Uint8Array(new Int32Array([seq_len, d_k, d_v]).buffer);
        const u8Q = new Uint8Array(payload.q);
        const u8K = new Uint8Array(payload.k);
        const u8V = new Uint8Array(payload.v);

        const exePayload = {
          action: 'exec',
          framework: 'exe',
          buffers: [
            Array.from(u8U),
            Array.from(u8Q),
            Array.from(u8K),
            Array.from(u8V),
          ],
          outputs: [ { byteLength: seq_len * d_v * 4 } ]
        };

        yield {
          id: ch.id || uuidv4(),
          payload: exePayload,
          meta,
          tCreate: ch.tCreate || Date.now()
        };
      }
    }
  };
}


// Reuse base assembler; coerce result like native-block-matmul.
export function buildAssembler(args) {
  const base = baseMHA.buildAssembler(args);

  function coerceResult(r) {
    if (typeof r === 'string') return Buffer.from(r, 'base64');
    if (r && r.type === 'Buffer' && Array.isArray(r.data)) {
      return Buffer.from(r.data);
    }
    return r;
  }

  return {
    integrate({ result, meta }) {
      return base.integrate({ result: coerceResult(result), meta });
    },
    finish() {
      return base.finish();
    }
  };
}
