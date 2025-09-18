// server/strategies/exe-ecm-stage1.js
//
// ECM Stage 1 (exe/CUDA) – thin shim around the WebGPU strategy,
// repackaged for the BinaryExecutor (stdin -> stdout).
//
// Reuses the WebGPU chunker/assembler; only the transport changes.

import path from 'path';
import fs from 'fs';
import { logger } from '../lib/logger.js';

// Reuse the webgpu ECM strategy's chunker/assembler so buffer layout stays identical.
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
} from './ecm-stage1.js';

export const id = 'exe-ecm-stage1';
export const name = 'ECM Stage 1 (exe/CUDA)';
export const framework = 'exe';

// Default binary paths per backend
const defaultBinaries = {
  cuda: '/app/binaries/exe_cuda_ecm_stage1',
  opencl: '/app/binaries/exe_opencl_ecm_stage1', // Future support
};

export function getArtifacts(config) {
  const backend = (config?.backend || 'cuda').toLowerCase();
  const artifacts = [];

  // Determine binary path - prioritize config.binary, then defaults
  const binaryPath = config.binary || defaultBinaries[backend];
  if (!binaryPath) {
    logger.warn(`[${id}] No binary configured for backend: ${backend}`);
    return [];
  }

  try {
    // Handle both absolute and relative paths
    const abs = path.isAbsolute(binaryPath) ? binaryPath : path.resolve(binaryPath);

    if (!fs.existsSync(abs)) {
      throw new Error(`Binary not found: ${abs}`);
    }

    const bytes = fs.readFileSync(abs);
    const artifactName = config.program || path.basename(binaryPath);

    artifacts.push({
      type: 'binary',
      name: artifactName,
      program: artifactName,
      backend,
      bytes: bytes.toString('base64'),
      exec: true
    });

    logger.info(`[${id}] Added binary artifact: ${abs} as ${artifactName} (${bytes.length} bytes)`);
  } catch (error) {
    logger.error(`[${id}] Failed to read binary ${binaryPath}:`, error.message);
    throw new Error(`Binary not found for ${backend}: ${binaryPath}`);
  }

  return artifacts;
}

export function getClientExecutorInfo(config = {}, inputArgs = {}) {
  const backend = (config.backend || 'cuda').toLowerCase();

  if (!['cuda', 'opencl'].includes(backend)) {
    throw new Error(`Unsupported backend: ${backend}. Must be 'cuda' or 'opencl'`);
  }

  // Get artifacts with proper error handling
  let artifacts = [];
  try {
    artifacts = getArtifacts(config);
  } catch (error) {
    logger.error(`[${id}] Failed to get artifacts:`, error.message);
    // Don't throw here - let the task fail later if the binary is actually needed
  }

  return {
    id,
    name,
    framework,
    backend,
    program: config.program || path.basename(defaultBinaries[backend] || 'exe_cuda_ecm_stage1'),
    artifacts,

    // Schema for debugging
    schema: {
      order: ['stdin', 'stdout'],
      inputs: [{ name: 'io', type: 'u32', note: 'Full ECM Stage 1 IO buffer (header + consts + pp + output + state)' }],
      outputs: [{ name: 'io', type: 'u32', size: 'header + consts + pp + outputs only (no state)' }],
    },
  };
}

export function buildChunker(args) {
  const { taskId, taskDir, K, config, inputArgs, inputFiles } = args;

  logger.info(`[${id}] buildChunker called with config:`, JSON.stringify(config, null, 2));
  logger.info(`[${id}] buildChunker inputArgs:`, JSON.stringify(inputArgs, null, 2));

  const webChunker = buildChunkerWeb({ taskId, taskDir, K, config, inputArgs, inputFiles });

  // Get the binary name for program reference
  const backend = (config?.backend || 'cuda').toLowerCase();
  const binaryPath = config.binary || defaultBinaries[backend];
  const programName = config.program || path.basename(binaryPath || 'exe_cuda_ecm_stage1');

  return {
    async *stream() {
      for await (const chunk of webChunker.stream()) {
        // Extract the IO buffer from the web ECM chunk
        const ioBuffer = chunk?.payload?.data;
        if (!ioBuffer || !(ioBuffer instanceof ArrayBuffer)) {
          throw new Error(`[${id}] Expected web ECM chunk to have payload.data as ArrayBuffer`);
        }

        // For exe: write entire IO buffer to stdin, expect same-sized stdout
        const outSize = ioBuffer.byteLength;

        yield {
          id: chunk.id,
          payload: {
            action: 'exec',
            framework: 'exe',
            // Send raw binary data as Uint8Array for proper binary handling
            buffers: [new Uint8Array(ioBuffer)],
            outputs: [{ byteLength: outSize }],
            meta: {
              program: programName,
              backend: backend,
              framework: 'exe',
              // Carry through any useful metadata for logging/diagnostics
              ...chunk.meta,
            },
          },
          meta: {
            ...chunk.meta,
            program: programName,
            backend: backend,
          },
        };
      }
    }
  };
}

// Reuse the exact same assembler the web strategy uses
export const buildAssembler = buildAssemblerWeb;

// Helper function to get total chunks (for kill-switch)
export function getTotalChunks(config, inputArgs) {
  const totalCurves = inputArgs?.total_curves || 0;
  const chunkSize = inputArgs?.chunk_size || 1;
  if (totalCurves && chunkSize) {
    return Math.ceil(totalCurves / chunkSize);
  }
  return null;
}