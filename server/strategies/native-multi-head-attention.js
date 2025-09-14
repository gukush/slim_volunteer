// File: strategies/native-multi-head-attention.js
// Runtime-compiled CUDA (via Lua host) variant of Multi-Head Attention.
// Reuses the WebGPU strategy's chunking/assembly logic and adapts
// each chunk payload to the native (Lua host) schema.
//
// Inputs expected (uploaded with the task):
//   - Q.bin, K.bin, V.bin    (Float32, row-major: [seq_len, d_model])
//
// Config:
//   {
//     seq_len: number,
//     d_model: number,
//     num_heads: number,
//     framework?: "native-cuda" | "cuda"   // optional; default "native-cuda"
//   }
//
// Native artifacts sent to clients:
//   - executors/cuda-multi-head-attention.host.lua  (Lua host)
//   - kernels/multi_head_attention.cu               (CUDA kernel source)

import fs from 'fs';
import path from 'path';
import { logger } from '../lib/logger.js';
import { v4 as uuidv4 } from 'uuid';
import * as baseMHA from './multi-head-attention.js';

export const id   = 'native-multi-head-attention';
export const name = 'Multi-Head Attention (Native CUDA via LuaJIT)';

// ---------- helpers ----------
function b64(buf) {
  return Buffer.isBuffer(buf) ? buf.toString('base64') : Buffer.from(buf).toString('base64');
}
function readProjectFile(rel) {
  const abs = path.isAbsolute(rel) ? rel : path.join(process.cwd(), rel);
  return fs.readFileSync(abs);
}

// ---------- client/executor info ----------
export function getClientExecutorInfo(config = {}) {
  // We keep "cuda" as the low-level framework (like native-block-matmul).
  const framework = 'cuda';

  // Prepare artifacts: Lua host + CUDA kernel text
  const artifacts = [];

  // Lua host
  const hostPath = 'executors/cuda-multi-head-attention.host.lua';
  const hostBytes = readProjectFile(hostPath);
  artifacts.push({
    type: 'lua',
    name: 'host.lua',
    program: 'host',
    backend: 'host',
    exec: false,
    bytes: b64(hostBytes)
  });

  // CUDA kernel
  const cuPath = 'kernels/cuda/multi_head_attention.cu';
  const cuBytes = readProjectFile(cuPath);
  artifacts.push({
    type: 'text',
    name: path.basename(cuPath),
    program: 'multi_head_attention',
    backend: 'cuda',
    exec: false,
    bytes: b64(cuBytes)
  });

  // Provide a schema to make native runtimes self-describing
  const schema = {
    action: 'compile_and_run',
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
    ],
  };

  return {
    framework,
    kernels: [path.basename(cuPath)],
    schema,
    artifacts
  };
}

// ---------- chunker ----------
// We wrap the WebGPU chunker and remap its payload to native LUA-host schema.
export function buildChunker(args) {
  const inner = baseMHA.buildChunker(args);

  return {
    async *stream() {
      for await (const ch of inner.stream()) {
        const { payload, meta } = ch;
        const { seq_len, d_k, d_v } = payload.dims || meta || {};

        // raw -> base64 for native payload
        const toB64 = (ab) => b64(Buffer.from(new Uint8Array(ab)));

        const nativePayload = {
          action: 'compile_and_run',
          framework: 'cuda',
          entry: 'execute_task',
          uniforms: [ seq_len, d_k, d_v ],
          inputs: [
            { b64: toB64(payload.q) },
            { b64: toB64(payload.k) },
            { b64: toB64(payload.v) },
          ],
          outputSizes: [ seq_len * d_v * 4 ],
          // explicit launch dims for CUDA runtime
          grid:  { 0: Math.max(1, seq_len), 1: Math.max(1, d_v), 2: 1 },
          block: { 0: 256, 1: 1, 2: 1 },
        };

        yield {
          id: ch.id || uuidv4(),
          payload: nativePayload,
          meta,
          tCreate: ch.tCreate || Date.now()
        };
      }
    }
  };
}

// ---------- assembler ----------
// Reuse base assembler but accept base64/Buffer/etc from native clients.
export function buildAssembler(args) {
  const base = baseMHA.buildAssembler(args);

  function coerceResult(r) {
    if (typeof r === 'string') return Buffer.from(r, 'base64');
    if (r && r.type === 'Buffer' && Array.isArray(r.data)) {
      return Buffer.from(r.data);
    }
    return r; // Buffer | ArrayBuffer | Uint8Array already fine
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
