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
  const { taskId, taskDir, K, config, inputArgs, inputFiles } = args;
  console.log('DEBUG exe-ecm-stage1 buildChunker args:', JSON.stringify(args, null, 2));
  console.log('DEBUG exe-ecm-stage1 inputArgs:', JSON.stringify(inputArgs, null, 2));
  const webChunker = buildChunkerWeb({ taskId, taskDir, K, config, inputArgs, inputFiles });

  function* generator() {
    for (const chunk of webChunker) {
      // `chunk.payload.data` is the full IO ArrayBuffer produced by the web ECM strategy.
      const ioBuffer = chunk?.payload?.data;
      if (!ioBuffer || !(ioBuffer instanceof ArrayBuffer)) {
        throw new Error(`[${id}] Expected web ECM chunk to have payload.data as ArrayBuffer`);
      }

      // For exe: write entire IO buffer to stdin, expect same-sized stdout.
      const outSize = ioBuffer.byteLength;

      yield {
        id: chunk.id,
        payload: {
          action: 'exec',
          // TaskManager will base64-encode ArrayBuffers for native clients automatically.
          buffers: [ioBuffer],          // stdin
          outputs: [outSize],           // expected stdout size
          meta: {
            program: (args?.config?.program) || defaultPrograms[(args?.config?.backend || 'cuda').toLowerCase()] || 'ecm_stage1_cuda',
            backend: (args?.config?.backend || 'cuda').toLowerCase(),
            framework: 'exe',
            // Carry through any useful metadata for logging/diagnostics
            ...chunk.meta,
          },
        },
        meta: chunk.meta,              // keep the original meta (useful in assembler)
      };
    }
  }

  return {
    async *stream() {
      for (const chunk of generator()) {
        yield chunk;
      }
    }
  };
}


// Reuse the exact same assembler the web strategy uses;
// it already knows how to parse the returned IO buffer and produce summary + artifacts.
export const buildAssembler = buildAssemblerWeb;
