// server/strategies/exe-ecm-stage1.js
//
// ECM Stage 1 (exe/CUDA) – thin shim around the WebGPU strategy,
// repackaged for the BinaryExecutor (stdin -> stdout).
//
// Reuses the WebGPU chunker/assembler; only the transport changes.

import path from 'path';
import fs from 'fs';

// Reuse the webgpu ECM strategy’s chunker/assembler so buffer layout stays identical.
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
  // you may also reuse any helpers exported by your webgpu ECM strategy if needed
} from './ecm-stage1.js';

export const id = 'exe-ecm-stage1';
export const name = 'ECM Stage 1 (exe/CUDA)';
export const framework = 'exe';

// Default program names per backend (override via config.program)
const defaultPrograms = {
  cuda: 'ecm_stage1_cuda',   // look up in PATH on the native client unless provided via artifact
  // opencl: 'ecm_stage1_opencl', // (optional) if you later add a CL exe
};

export function getClientExecutorInfo({ config = {}, inputArgs = {} } = {}) {
  const backend = (config.backend || 'cuda').toLowerCase();
  const program = config.program || defaultPrograms[backend] || 'ecm_stage1_cuda';

  // If the caller provides a compiled binary path, ship it as an artifact to the native client.
  // Otherwise, the native client will resolve `program` in PATH.
  const artifacts = [];
  if (config.binary) {
    const binPath = path.resolve(config.binary);
    const bytes = fs.readFileSync(binPath);
    artifacts.push({
      name: program,          // BinaryExecutor will cache by this program name
      type: 'binary',
      exec: true,
      bytes: bytes.toString('base64'),
    });
  }

  return {
    id,
    name,
    framework,
    backend,
    program,
    artifacts,

    // (Optional) schematic; helpful for debugging
    schema: {
      order: ['stdin', 'stdout'],
      inputs: [{ name: 'io', type: 'u32', note: 'Full ECM Stage 1 IO buffer (header + consts + pp + output + state)' }],
      outputs: [{ name: 'io', type: 'u32', size: '== input size' }],
    },
  };
}

// Build chunks using the *same* IO buffer the WebGPU path expects,
// but convert each chunk to the exe transport using common protocol.
export function buildChunker(args) {
  const webChunker = buildChunkerWeb(args);

  function* generator() {
    for (const chunk of webChunker) {
      // `chunk.payload.data` is the full IO ArrayBuffer produced by the web ECM strategy.
      const ioBuffer = chunk?.payload?.data;
      if (!ioBuffer || !(ioBuffer instanceof ArrayBuffer)) {
        throw new Error(`[${id}] Expected web ECM chunk to have payload.data as ArrayBuffer`);
      }

      // Create protocol-compliant payload
      const protocolPayload = createProtocolPayload({
        framework: 'CUDA',
        dataType: 'UINT32',
        inputs: [
          {
            data: ioBuffer,
            dataType: 'UINT32',
            dimensions: [ioBuffer.byteLength / 4, 1, 1, 1] // 1D array of uint32s
          }
        ],
        outputs: [
          {
            dataType: 'UINT32',
            dimensions: [ioBuffer.byteLength / 4, 1, 1, 1] // Same size output
          }
        ],
        metadata: JSON.stringify({
          program: (args?.config?.program) || defaultPrograms[(args?.config?.backend || 'cuda').toLowerCase()] || 'ecm_stage1_cuda',
          backend: (args?.config?.backend || 'cuda').toLowerCase(),
          framework: 'exe',
          ...chunk.meta
        })
      });

      yield {
        id: chunk.id,
        payload: {
          action: 'execute_binary_stream',
          binary: (args?.config?.program) || defaultPrograms[(args?.config?.backend || 'cuda').toLowerCase()] || 'ecm_stage1_cuda',
          args: ['--stdin'], // Use stdin/stdout mode
          stdin: b64(protocolPayload), // Protocol-compliant data
          stdoutSize: ioBuffer.byteLength // Expected output size in bytes
        },
        meta: chunk.meta,              // keep the original meta (useful in assembler)
      };
    }
  }

  return generator();
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

// Reuse the exact same assembler the web strategy uses;
// it already knows how to parse the returned IO buffer and produce summary + artifacts.
export const buildAssembler = buildAssemblerWeb;
