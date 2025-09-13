// ECM Stage 1 (Native CUDA) — strategy wrapper that reuses your webgpu chunker/assembler.
// Converts WebGPU payload format to native CUDA format for the native client.

import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// Reuse assembler from your existing WebGPU strategy file to avoid duplication.
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
} from './ecm-stage1.js';

export const id = 'native-ecm-stage1';
export const name = 'ECM Stage 1 (Native CUDA)';
export const framework = 'native-cuda';

// This tells the server what to ship to clients.
export function getClientExecutorInfo(config, inputArgs) {
  return {
    framework: 'cuda', // Native client will use CUDA executor
    // The server should include these as artifacts delivered with the workload.
    kernels: [
      'kernels/cuda/ecm_stage1_cuda.cu',
      'kernels/opencl/ecm_stage1_opencl.cl',
    ],
    // Output is still the same packed Uint32 layout you already consume.
    schema: { output: 'Uint32Array' },
  };
}

// Custom chunker that converts WebGPU payload format to native CUDA format
export function buildChunker({ taskId, taskDir, K, config, inputArgs, inputFiles }) {
  // Use the WebGPU chunker to generate the base chunks
  const webgpuChunker = buildChunkerWeb({ taskId, taskDir, K, config, inputArgs, inputFiles });
  
  return {
    async *stream() {
      for await (const chunk of webgpuChunker.stream()) {
        // Convert WebGPU payload format to native CUDA format
        const { data, dims } = chunk.payload;
        const { n, pp_count, total_words } = dims;
        
        // Convert ArrayBuffer to base64 for native client
        const dataBase64 = Buffer.from(data).toString('base64');
        
        // Calculate output size (same as WebGPU version)
        const CURVE_OUT_WORDS_PER = 8 + 1; // result(8) + status(1)
        const outputSize = n * CURVE_OUT_WORDS_PER * 4; // 4 bytes per Uint32
        
        // Create native CUDA payload format
        const nativePayload = {
          action: 'compile_and_run',
          framework: 'cuda',
          entry: 'main',
          source: '', // Will be filled by the native client from kernels
          inputs: [{ data: dataBase64 }],
          outputSizes: [outputSize],
          uniforms: [n, pp_count, total_words],
          grid: [Math.ceil(n / 256), 1, 1], // 256 threads per block
          block: [256, 1, 1]
        };
        
        // Create new chunk with native payload format
        const nativeChunk = {
          ...chunk,
          payload: nativePayload
        };
        
        yield nativeChunk;
      }
    }
  };
}

// Reuse assembler from WebGPU version
export const buildAssembler = buildAssemblerWeb;
