// strategies/multi-head-attention.js
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// ---------- Public strategy identifiers ----------
export const id = 'multi-head-attention';
export const name = 'Multi-Head Attention (WebGPU)';

// ---------- Executor selection ----------
export function getClientExecutorInfo(config) {
  const framework = (config?.framework || 'webgpu').toLowerCase();

  if (framework === 'webgpu') {
    return {
      framework: 'webgpu',
      path: 'executors/webgpu-multi-head-attention.client.js',
      kernels: ['kernels/multi_head_attention.wgsl']
    };
  }

  throw new Error('Unsupported framework for multi-head-attention: ' + framework);
}

// ---------- Helpers ----------
function toF32(x) {
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  if (ArrayBuffer.isView(x)) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  if (typeof Buffer !== 'undefined' && x instanceof Buffer) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  throw new Error('Unsupported buffer type for Float32 view');
}

function readMatrix(fd, rows, cols, elementSize = 4) {
  const totalBytes = rows * cols * elementSize;
  const buffer = Buffer.alloc(totalBytes);
  fs.readSync(fd, buffer, 0, totalBytes, 0);
  return buffer;
}

function pickInputFiles(files, expectedNames) {
  const result = {};

  for (const expectedName of expectedNames) {
    // Try exact match first
    let file = files.find(f => {
      const name = f.originalName || (f.path ? path.basename(f.path) : '');
      return name.toLowerCase() === expectedName.toLowerCase();
    });

    // Try pattern match
    if (!file) {
      const pattern = new RegExp(`${expectedName.replace('.', '\\.')}$`, 'i');
      file = files.find(f => {
        const name = f.originalName || (f.path ? path.basename(f.path) : '');
        return pattern.test(name);
      });
    }

    if (!file) {
      throw new Error(`Required input file not found: ${expectedName}`);
    }

    result[expectedName] = file.path;
  }

  return result;
}

function readHeadSlice(fd, seq_len, d_model, d_k, head) {
  // Basic validation
  if (!Number.isInteger(fd)) throw new Error('readHeadSlice: invalid file descriptor');
  if (!Number.isInteger(seq_len) || !Number.isInteger(d_model) ||
      !Number.isInteger(d_k) || !Number.isInteger(head)) {
    throw new Error('readHeadSlice: seq_len, d_model, d_k, head must be integers');
  }
  if (seq_len <= 0 || d_model <= 0 || d_k <= 0) {
    throw new Error('readHeadSlice: seq_len, d_model, d_k must be > 0');
  }
  if (d_model % d_k !== 0) {
    throw new Error(`readHeadSlice: d_model (${d_model}) must be divisible by d_k (${d_k})`);
  }
  const num_heads = Math.floor(d_model / d_k);
  if (head < 0 || head >= num_heads) {
    throw new Error(`readHeadSlice: head index out of range (got ${head}, total ${num_heads})`);
  }

  const BYTES_PER_F32 = 4;
  const rowBytesTotal = d_model * BYTES_PER_F32;
  const rowBytesSlice = d_k * BYTES_PER_F32;
  const headOffsetBytes = head * rowBytesSlice;

  // Pre-allocate one contiguous ArrayBuffer for [seq_len, d_k]
  const totalBytes = seq_len * rowBytesSlice;
  const outAB = new ArrayBuffer(totalBytes);
  const outBuf = Buffer.from(outAB); // shares memory with outAB

  // Read per row using absolute positions (doesn't disturb file offset)
  for (let row = 0; row < seq_len; row++) {
    const srcPos = row * rowBytesTotal + headOffsetBytes;
    const dstPos = row * rowBytesSlice;

    // Ensure we fully read the desired slice (handle short reads)
    let readSoFar = 0;
    while (readSoFar < rowBytesSlice) {
      const n = fs.readSync(
        fd,
        outBuf,
        dstPos + readSoFar,
        rowBytesSlice - readSoFar,
        srcPos + readSoFar
      );
      if (n === 0) {
        throw new Error(
          `readHeadSlice: unexpected EOF @ row ${row}, head ${head}, ` +
          `wanted ${rowBytesSlice} bytes, got ${readSoFar}`
        );
      }
      readSoFar += n;
    }
  }

  // Return a Float32Array view over the filled ArrayBuffer
  return new Float32Array(outAB);
}

// ---------- Chunker (one chunk per attention head) ----------
export function buildChunker({ taskId, taskDir, K, config, inputFiles }) {
  logger.info('Building multi-head attention chunker...');

  try {
    const { seq_len, d_model, num_heads } = config;

    if (!seq_len || !d_model || !num_heads) {
      throw new Error('Config must specify seq_len, d_model, and num_heads');
    }

    const d_k = Math.floor(d_model / num_heads);
    const d_v = d_k; // Typically d_v = d_k

    if (d_k * num_heads !== d_model) {
      throw new Error(`d_model (${d_model}) must be divisible by num_heads (${num_heads})`);
    }

    logger.info(`MHA Config: seq_len=${seq_len}, d_model=${d_model}, num_heads=${num_heads}, d_k=${d_k}`);

    // Find input files with robust fallback like block-matmul-flex
    const inputPaths = pickInputFiles(inputFiles, ['Q.bin', 'K.bin', 'V.bin']);
    logger.info('Input files found:', inputPaths);

    // Verify file sizes match expected dimensions
    const expectedSize = seq_len * d_model * 4; // float32
    for (const [name, filePath] of Object.entries(inputPaths)) {
      const stat = fs.statSync(filePath);
      if (stat.size !== expectedSize) {
        throw new Error(`${name} size mismatch: expected ${expectedSize} bytes, got ${stat.size} bytes`);
      }
      logger.info(`${name}: ${stat.size} bytes (${seq_len} x ${d_model} float32)`);
    }

    // Open file descriptors
    let fdQ, fdK, fdV;
    try {
      fdQ = fs.openSync(inputPaths['Q.bin'], 'r');
      fdK = fs.openSync(inputPaths['K.bin'], 'r');
      fdV = fs.openSync(inputPaths['V.bin'], 'r');
      logger.info('File descriptors opened successfully');
    } catch (e) {
      throw new Error(`Failed to open input files: ${e.message}`);
    }

    return {
      async *stream() {
        logger.info(`Starting chunk generation for ${num_heads} heads...`);

        try {
          for (let head = 0; head < num_heads; head++) {
            logger.debug(`Generating chunk for head ${head}/${num_heads - 1}`);

            try {
              // Read the slice for this head from each matrix using safer approach
              const headQ = readHeadSlice(fdQ, seq_len, d_model, d_k, head);
              const headK = readHeadSlice(fdK, seq_len, d_model, d_k, head);
              const headV = readHeadSlice(fdV, seq_len, d_model, d_k, head);

              logger.debug(`Head ${head}: read ${headQ.length + headK.length + headV.length} total bytes`);

              const payload = {
                q: headQ.buffer.slice(headQ.byteOffset, headQ.byteOffset + headQ.byteLength),
                k: headK.buffer.slice(headK.byteOffset, headK.byteOffset + headK.byteLength),
                v: headV.buffer.slice(headV.byteOffset, headV.byteOffset + headV.byteLength),
                dims: { seq_len, d_k, d_v }
              };

              const meta = {
                head,
                seq_len,
                d_k,
                d_v,
                outputSize: seq_len * d_v * 4
              };

              yield {
                id: uuidv4(),
                payload,
                meta,
                tCreate: Date.now()
              };

            } catch (e) {
              logger.error(`Failed to generate chunk for head ${head}:`, e.message);
              throw new Error(`Chunk generation failed for head ${head}: ${e.message}`);
            }
          }

          logger.info('Multi-head attention chunker completed successfully');

        } finally {
          // Always close file descriptors
          try {
            if (fdQ !== undefined) fs.closeSync(fdQ);
            if (fdK !== undefined) fs.closeSync(fdK);
            if (fdV !== undefined) fs.closeSync(fdV);
            logger.info('File descriptors closed');
          } catch (e) {
            logger.error('Error closing file descriptors:', e.message);
          }
        }
      }
    };

  } catch (e) {
    logger.error('Failed to build multi-head attention chunker:', e.message);
    throw e;
  }
}

// ---------- Assembler (concatenate head outputs) ----------
export function buildAssembler({ taskId, taskDir, config }) {
  const { seq_len, d_model, num_heads } = config;
  const d_v = Math.floor(d_model / num_heads);

  const outPath = path.join(taskDir, 'attention_output.bin');
  const fdOut = fs.openSync(outPath, 'w');

  // Pre-allocate output file
  const totalOutputSize = seq_len * d_model * 4; // seq_len × (num_heads * d_v)
  fs.ftruncateSync(fdOut, totalOutputSize);

  const headResults = new Map(); // head -> Float32Array

  return {
    integrate({ result, meta }) {
      const { head, seq_len, d_v } = meta;
      const headOutput = toF32(result);

      if (headOutput.length !== seq_len * d_v) {
        throw new Error(`Head ${head} output size mismatch: expected ${seq_len * d_v}, got ${headOutput.length}`);
      }

      headResults.set(head, headOutput);
      logger.debug(`Integrated head ${head} result`);

      // Write this head's output to the appropriate slice of the output file
      const headBytes = seq_len * d_v * 4;
      for (let seq = 0; seq < seq_len; seq++) {
        const srcOffset = seq * d_v;
        const fileOffset = seq * d_model * 4 + head * d_v * 4;
        const sliceBytes = d_v * 4;

        const buffer = Buffer.from(
          headOutput.buffer,
          headOutput.byteOffset + srcOffset * 4,
          sliceBytes
        );

        fs.writeSync(fdOut, buffer, 0, sliceBytes, fileOffset);
      }
    },

    finalize() {
      fs.closeSync(fdOut);

      const receivedHeads = Array.from(headResults.keys()).sort((a, b) => a - b);
      const expectedHeads = Array.from({ length: num_heads }, (_, i) => i);

      logger.info(`Assembly complete. Received heads: [${receivedHeads.join(', ')}]`);

      if (receivedHeads.length !== expectedHeads.length) {
        logger.warn(`Expected ${expectedHeads.length} heads, got ${receivedHeads.length}`);
      }

      return {
        outPath,
        totalHeads: num_heads,
        receivedHeads: receivedHeads.length,
        outputShape: [seq_len, d_model]
      };
    },

    cleanup() {
      try {
        fs.closeSync(fdOut);
        headResults.clear();
        logger.info(`Multi-head-attention assembler cleanup: closed output file descriptor and cleared memory`);
      } catch (e) {
        logger.warn(`Multi-head-attention assembler cleanup failed:`, e.message);
      }
    }
  };
}

// Kill-switch support: Calculate total chunks deterministically
export function getTotalChunks(config, inputArgs) {
  const { num_heads } = config;

  // Multi-head attention has exactly one chunk per head
  const totalChunks = num_heads;

  logger.info(`Multi-head-attention getTotalChunks: num_heads=${num_heads} -> ${totalChunks} chunks`);
  return totalChunks;
}