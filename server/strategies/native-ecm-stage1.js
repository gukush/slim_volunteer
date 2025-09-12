// ECM Stage 1 (Native CUDA) — strategy wrapper that reuses your webgpu chunker/assembler.
// Ship a Lua host + CUDA kernel to native clients. Framework advertised: 'cuda'.

import path from 'path';
import fs from 'fs';

// Reuse chunker/assembler from your existing WebGPU strategy file to avoid duplication.
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
} from './ecm-stage1.js';

export const id = 'native-ecm-stage1';
export const name = 'ECM Stage 1 (Native CUDA)';
export const framework = 'native-cuda';

// This tells the server what to ship to clients.
// Your infra likely turns the "kernels" list into artifacts and inlines host.lua.
export function getClientExecutorInfo(config, inputArgs) {
  return {
    framework: 'cuda', // <— key change vs opencl
    // The server should include these as artifacts delivered with the workload.
    // - host.lua is used by the native client (lua host) to drive the CUDA executor.
    // - kernels include CUDA first (primary), with OpenCL as a courtesy fallback.
    kernels: [
      'kernels/cuda/ecm_stage1_cuda.cu',
      'kernels/opencl/ecm_stage1_opencl.cl',
    ],
    hostLua: 'executors/host-ecm-stage1.lua', // see file below
    // Output is still the same packed Uint32 layout you already consume.
    schema: { output: 'Uint32Array' },
  };
}

// Just reuse your proven chunker & assembler.
export const buildChunker = buildChunkerWeb;
export const buildAssembler = buildAssemblerWeb;
