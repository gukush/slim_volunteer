// native-distributed-sort.js
// Native CUDA implementation of distributed sorting strategy
// Reuses chunker/assembler from WebGPU strategy, but with CUDA execution

import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// Reuse chunker/assembler from the existing WebGPU strategy to avoid duplication
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
} from './distributed-sort.js';

export const id = 'native-distributed-sort';
export const name = 'Distributed Integer Sort (Native CUDA)';
export const framework = 'native-cuda';

// Helper functions to resolve file paths and create artifacts
function tryRead(p) {
  try { return fs.readFileSync(p); } catch { return null; }
}

function findFirstExisting(paths) {
  for (const p of paths) {
    const b = tryRead(p);
    if (b) return { path: p, bytes: b };
  }
  return null;
}

function resolveCandidates(rel) {
  const cwd = process.cwd();
  const here = path.dirname(new URL(import.meta.url).pathname);
  return [
    path.join(cwd, rel),
    path.join(cwd, 'strategies', rel),
    path.join(cwd, 'kernels', rel),
    path.join(here, rel),
    path.join(here, '..', rel),
    path.join(here, '..', 'kernels', rel),
  ];
}

function b64(buf) {
  return Buffer.isBuffer(buf) ? buf.toString('base64') : Buffer.from(buf).toString('base64');
}

function makeArtifact({ type = 'text', name, program, backend, exec = false, bytes }) {
  return { type, name, program, backend, exec, bytes };
}

// This tells the server what to ship to native clients
export function getClientExecutorInfo(config, inputArgs) {
  const framework = String(config?.framework || 'native-cuda').toLowerCase();

  // Find the Lua host script
  const hostCandidates = resolveCandidates('executors/host-distributed-sort.lua');
  const host = findFirstExisting(hostCandidates);

  // Find the CUDA kernel
  const cudaCandidates = resolveCandidates('kernels/cuda/bitonic_sort_cuda.cu');
  const cuda = findFirstExisting(cudaCandidates);

  // Optional OpenCL fallback
  const openclCandidates = resolveCandidates('kernels/opencl/bitonic_sort_opencl.cl');
  const opencl = findFirstExisting(openclCandidates);

  const artifacts = [];

  if (host) {
    artifacts.push(makeArtifact({
      type: 'lua',
      name: 'host.lua',
      program: 'host',
      backend: 'host',
      bytes: b64(host.bytes)
    }));
  }

  if (cuda) {
    artifacts.push(makeArtifact({
      type: 'text',
      name: path.basename(cuda.path),
      program: 'bitonic_sort',
      backend: 'cuda',
      bytes: b64(cuda.bytes)
    }));
  }

  if (opencl) {
    artifacts.push(makeArtifact({
      type: 'text',
      name: path.basename(opencl.path),
      program: 'bitonic_sort',
      backend: 'opencl',
      bytes: b64(opencl.bytes)
    }));
  }

  return {
    framework: 'cuda', // Native client will use CUDA executor via Lua
    artifacts: artifacts,
    schema: { output: 'Uint32Array' }, // Output is sorted integer array
  };
}

// Round up to next power of 2 for efficient bitonic sort
function nextPowerOf2(n) {
  if (n <= 0) return 1;
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

// Custom chunker that creates the proper payload format for native client
export function buildChunker({ taskId, taskDir, K, config, inputArgs, inputFiles }) {
  // Use the same file discovery logic as the WebGPU version
  const webChunker = buildChunkerWeb({ taskId, taskDir, K, config, inputArgs, inputFiles });

  return {
    async *stream() {
      for await (const chunk of webChunker.stream()) {
        // Transform the payload for native client consumption
        const { payload, meta } = chunk;

        // Native client expects base64-encoded data instead of ArrayBuffer
        const dataArrayBuffer = payload.data;
        const dataBuffer = Buffer.from(dataArrayBuffer);
        const dataBase64 = dataBuffer.toString('base64');

        // Create native-compatible payload
        const nativePayload = {
          data: dataBase64,  // base64 string instead of ArrayBuffer
          originalSize: payload.originalSize,
          paddedSize: payload.paddedSize,
          ascending: payload.ascending
        };

        const nativeMeta = {
          ...meta,
          framework: 'native-cuda',
          // Add any additional metadata needed for native processing
        };

        yield {
          id: chunk.id,
          payload: nativePayload,
          meta: nativeMeta,
          tCreate: chunk.tCreate
        };
      }
    }
  };
}

// Reuse the assembler from WebGPU version since the output format is the same
export const buildAssembler = buildAssemblerWeb;
