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

        // Create protocol-compliant payload
        const protocolPayload = createProtocolPayload({
          framework: 'CUDA',
          dataType: 'FLOAT32',
          inputs: [
            {
              data: payload.q,
              dataType: 'FLOAT32',
              dimensions: [seq_len, d_k, 1, 1] // Q matrix
            },
            {
              data: payload.k,
              dataType: 'FLOAT32',
              dimensions: [seq_len, d_k, 1, 1] // K matrix
            },
            {
              data: payload.v,
              dataType: 'FLOAT32',
              dimensions: [seq_len, d_v, 1, 1] // V matrix
            }
          ],
          outputs: [
            {
              dataType: 'FLOAT32',
              dimensions: [seq_len, d_v, 1, 1] // Output matrix
            }
          ],
          metadata: JSON.stringify({
            seq_len,
            d_k,
            d_v,
            backend: 'cuda',
            program: 'native_cuda_multi_head_attention'
          })
        });

        const exePayload = {
          action: 'execute_binary_stream',
          binary: 'native_cuda_multi_head_attention',
          args: ['--stdin'], // Use stdin/stdout mode
          stdin: b64(protocolPayload), // Protocol-compliant data
          stdoutSize: seq_len * d_v * 4 // Expected output size in bytes
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

// Helper function to create protocol-compliant payload
function createProtocolPayload({ framework, dataType, inputs, outputs, metadata = '' }) {
  const buffer = Buffer.alloc(0);

  // Protocol header
  const header = Buffer.alloc(32); // ProtocolHeader size
  header.writeUInt32LE(0x4558454D, 0); // magic "EXEM"
  header.writeUInt32LE(1, 4); // version
  header.writeUInt32LE(getFrameworkCode(framework), 8); // framework
  header.writeUInt32LE(getDataTypeCode(dataType), 12); // dataType
  header.writeUInt32LE(inputs.length, 16); // num_inputs
  header.writeUInt32LE(outputs.length, 20); // num_outputs
  header.writeUInt32LE(Buffer.byteLength(metadata, 'utf8'), 24); // metadata_size
  header.writeUInt32LE(0, 28); // reserved

  let result = Buffer.concat([Buffer.from(header), Buffer.from(metadata, 'utf8')]);

  // Add input buffers
  for (const input of inputs) {
    const desc = Buffer.alloc(32); // BufferDescriptor size
    desc.writeUInt32LE(input.data.byteLength, 0); // size
    desc.writeUInt32LE(getDataTypeCode(input.dataType), 4); // dataType
    desc.writeUInt32LE(input.dimensions[0] || 0, 8);
    desc.writeUInt32LE(input.dimensions[1] || 0, 12);
    desc.writeUInt32LE(input.dimensions[2] || 0, 16);
    desc.writeUInt32LE(input.dimensions[3] || 0, 20);
    desc.writeUInt32LE(0, 24); // reserved
    desc.writeUInt32LE(0, 28); // reserved
    desc.writeUInt32LE(0, 32); // reserved

    result = Buffer.concat([result, desc, Buffer.from(input.data)]);
  }

  return result;
}

function getFrameworkCode(framework) {
  const codes = { 'CPU': 0, 'CUDA': 1, 'OPENCL': 2, 'VULKAN': 3, 'WEBGPU': 4 };
  return codes[framework] || 0;
}

function getDataTypeCode(dataType) {
  const codes = { 'FLOAT32': 0, 'FLOAT16': 1, 'INT32': 2, 'INT16': 3, 'INT8': 4, 'UINT32': 5, 'UINT16': 6, 'UINT8': 7 };
  return codes[dataType] || 0;
}

function b64(buffer) {
  return Buffer.from(buffer).toString('base64');
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
