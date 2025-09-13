// ECM Stage 1 (Native CUDA) — strategy wrapper that reuses your webgpu chunker/assembler.
// Uses Lua host script to orchestrate native CUDA execution.

import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// Reuse chunker/assembler from your existing WebGPU strategy file to avoid duplication.
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
} from './ecm-stage1.js';

export const id = 'native-ecm-stage1';
export const name = 'ECM Stage 1 (Native CUDA)';
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

function b64(buf) { return Buffer.isBuffer(buf) ? buf.toString('base64') : Buffer.from(buf).toString('base64'); }

function makeArtifact({ type = 'text', name, program, backend, exec = false, bytes }) {
  return { type, name, program, backend, exec, bytes };
}

// This tells the server what to ship to clients.
export function getClientExecutorInfo(config, inputArgs) {
  const framework = String(config?.framework || 'native-cuda').toLowerCase();

  // Always try to include host.lua so native client can route + compile at runtime.
  const hostCandidates = resolveCandidates('executors/host-ecm-stage1.lua');
  const host = findFirstExisting(hostCandidates);

  // Optional kernels (these are just hints; Lua host can also generate or use its own)
  const ku = findFirstExisting(resolveCandidates('kernels/cuda/ecm_stage1_cuda.cu'));
  const kl = findFirstExisting(resolveCandidates('kernels/opencl/ecm_stage1_opencl.cl'));

  const artifacts = [];
  if (host) artifacts.push(makeArtifact({
    type: 'lua', name: 'host.lua', program: 'host', backend: 'host', bytes: b64(host.bytes)
  }));
  if (ku) artifacts.push(makeArtifact({
    type: 'text', name: path.basename(ku.path), program: 'ecm_stage1', backend: 'cuda', bytes: b64(ku.bytes)
  }));
  if (kl) artifacts.push(makeArtifact({
    type: 'text', name: path.basename(kl.path), program: 'ecm_stage1', backend: 'opencl', bytes: b64(kl.bytes)
  }));

  return {
    framework: 'cuda', // Native client will use CUDA executor via Lua
    artifacts: artifacts,
    // Output is still the same packed Uint32 layout you already consume.
    schema: { output: 'Uint32Array' },
  };
}

// Just reuse your proven chunker & assembler from WebGPU version.
// The Lua host script will handle the WebGPU payload format and orchestrate CUDA execution.
export const buildChunker = buildChunkerWeb;
export const buildAssembler = buildAssemblerWeb;
