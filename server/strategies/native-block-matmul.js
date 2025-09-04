// strategies/native-block-matmul.js
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'native-block-matmul';
export const name = 'Native Block Matrix Multiplication (CPU/GPU Binary Execution)';

export function getClientExecutorInfo(config) {
  const framework = (config?.framework || 'cuda').toLowerCase();

  // For native clients, we return the kernel/binary info they need
  switch (framework) {
    case 'cuda':
      return {
        framework: 'cuda',
        path: 'executors/native-bridge.client.js', // Not used by native clients
        kernels: ['kernels/block_matrix_multiply_cuda_kernel.cu'],
        schema: {
          action: 'compile_and_run',
          inputs: ['float32[]', 'float32[]', 'int32[]'], // A, B, dims
          outputs: ['float32[]'] // result
        }
      };
    case 'opencl':
      return {
        framework: 'opencl',
        path: 'executors/native-bridge.client.js',
        kernels: ['kernels/block_matrix_multiply_opencl_kernel.cl'],
        schema: {
          action: 'compile_and_run',
          inputs: ['float32[]', 'float32[]', 'int32[]'],
          outputs: ['float32[]']
        }
      };
    case 'vulkan':
      return {
        framework: 'vulkan',
        path: 'executors/native-bridge.client.js',
        kernels: ['kernels/block_matrix_multiply_vulkan_compute.glsl'],
        schema: {
          action: 'compile_and_run',
          inputs: ['float32[]', 'float32[]', 'int32[]'],
          outputs: ['float32[]']
        }
      };
    case 'cpu':
      return {
        framework: 'cpu',
        path: 'executors/native-bridge.client.js',
        kernels: [], // CPU doesn't need kernels
        schema: {
          action: 'cpu_matmul',
          inputs: ['float32[]', 'float32[]', 'int32[]'],
          outputs: ['float32[]']
        }
      };
    default:
      throw new Error(`Unsupported framework for native execution: ${framework}`);
  }
}

// Helper functions (reuse from block-matmul-flex.js)
function toF32(x) {
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  if (ArrayBuffer.isView(x)) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  if (typeof Buffer !== 'undefined' && x instanceof Buffer) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  if (x && x.type === 'Buffer' && Array.isArray(x.data)) {
    const u8 = Uint8Array.from(x.data);
    return new Float32Array(u8.buffer);
  }
  throw new Error('Unsupported buffer type for Float32 view');
}

function readWindow(fd, rowStart, rowCount, colStart, colCount, rowLen, elementSize = 4) {
  const out = Buffer.alloc(rowCount * colCount * elementSize);
  const rowBytes = colCount * elementSize;
  for (let r = 0; r < rowCount; r++) {
    const srcOff = ((rowStart + r) * rowLen + colStart) * elementSize;
    const dstOff = r * rowBytes;
    fs.readSync(fd, out, dstOff, rowBytes, srcOff);
  }
  return out;
}

async function readWindowAsync(filePath, rowStart, rowCount, colStart, colCount, rowLen, elementSize = 4) {
  const out = Buffer.alloc(rowCount * colCount * elementSize);
  const rowBytes = colCount * elementSize;

  // Use streaming instead of synchronous reads
  const fileHandle = await fs.open(filePath, 'r');

  try {
    for (let r = 0; r < rowCount; r++) {
      const srcOff = ((rowStart + r) * rowLen + colStart) * elementSize;
      const dstOff = r * rowBytes;

      // Async read with small buffer
      const { buffer } = await fileHandle.read(out, dstOff, rowBytes, srcOff);
    }
  } finally {
    await fileHandle.close();
  }

  return out;
}

// Updated chunker with async operations and better memory management
export function buildChunker({ taskId, taskDir, K, config, inputFiles }) {
  const Afile = inputFiles.find(f => /A\.bin$/i.test(f.originalName))?.path || inputFiles[0]?.path;
  const Bfile = inputFiles.find(f => /B\.bin$/i.test(f.originalName))?.path || inputFiles[1]?.path;
  if (!Afile || !Bfile) throw new Error('Need A.bin and B.bin');

  const { N, K: KK, M } = config;

  // Use much larger chunks for massive matrices
  const C = Number(config.chunk_size ?? config.C ?? 134217728); // Default 128MB
  let baseRows, baseCols, kSpan;

  if (config.tileSize || config.kTileSize) {
    const ts = Math.max(1, Number(config.tileSize || 512)); // Increased default
    const ks = Math.max(1, Number(config.kTileSize || Math.min(KK, ts)));
    baseRows = ts; baseCols = ts; kSpan = Math.min(ks, KK);
  } else {
    const pick = pickTileParams({ N, M, K: KK, C });
    baseRows = pick.rows; baseCols = pick.cols; kSpan = pick.kTileSize;
  }

  const nIB = Math.ceil(N / baseRows);
  const nJB = Math.ceil(M / baseCols);
  const totalChunks = nIB * nJB * Math.ceil(KK / kSpan);

  logger.info(`Chunker: ${nIB}×${nJB}×${Math.ceil(KK/kSpan)} = ${totalChunks.toLocaleString()} chunks`);

  // Prevent excessive chunk counts
  if (totalChunks > 500000) {
    throw new Error(`Too many chunks: ${totalChunks.toLocaleString()}. Increase chunk_size to at least ${Math.ceil(C * totalChunks / 500000)} bytes`);
  }

  return {
    async *stream() {
      let chunkCount = 0;

      for (let ib = 0; ib < nIB; ib++) {
        const rNow = Math.min(baseRows, N - ib * baseRows);

        for (let jb = 0; jb < nJB; jb++) {
          const cNow = Math.min(baseCols, M - jb * baseCols);

          for (let kb = 0; kb < KK; kb += kSpan) {
            const kNow = Math.min(kSpan, KK - kb);

            try {
              // ASYNC file reading - won't block event loop
              const [Ablock, Bblock] = await Promise.all([
                readWindowAsync(Afile, ib * baseRows, rNow, kb, kNow, KK, 4),
                readWindowAsync(Bfile, kb, kNow, jb * baseCols, cNow, M, 4)
              ]);

              const payload = {
                a: Ablock.buffer.slice(Ablock.byteOffset, Ablock.byteOffset + Ablock.byteLength),
                b: Bblock.buffer.slice(Bblock.byteOffset, Bblock.byteOffset + Bblock.byteLength),
                dims: { rows: rNow, K: kNow, cols: cNow },
              };

              const meta = {
                ib, jb, kb, kSpan: kNow, rows: rNow, cols: cNow, baseRows, baseCols,
                // Native execution parameters
                outputSizes: [rNow * cNow * 4],
                uniforms: [rNow, kNow, cNow]
              };

              // Add framework-specific parameters
              const fw = (config.framework || '').toLowerCase();
              if (fw === 'native-cuda') {
                meta.grid = [Math.ceil(cNow/16), Math.ceil(rNow/16), 1];
                meta.block = [16, 16, 1];
              } else if (fw === 'native-opencl') {
                meta.global = [cNow, rNow, 1];
                meta.local = [16, 16, 1];
              } else if (fw === 'native-vulkan') {
                meta.groups = [Math.ceil(cNow/16), Math.ceil(rNow/16), 1];
              }

              yield {
                id: `${taskId}_${ib}_${jb}_${kb}`,
                payload,
                meta,
                tCreate: Date.now()
              };

              chunkCount++;

              // Progress logging for large tasks
              if (chunkCount % 1000 === 0) {
                logger.info(`Generated ${chunkCount.toLocaleString()}/${totalChunks.toLocaleString()} chunks`);
              }

              // Yield control periodically to prevent event loop blocking
              if (chunkCount % 100 === 0) {
                await new Promise(resolve => setImmediate(resolve));
              }

            } catch (error) {
              logger.error(`Chunk generation failed at (${ib},${jb},${kb}):`, error.message);
              throw error;
            }
          }
        }
      }

      logger.info(`block-matmul-flex chunker completed: ${chunkCount.toLocaleString()} chunks`);
    }
  };
}


function pickTileParams({ N, M, K, C, outFrac = 1/3, align = 32 }) {
  const Cn = Math.max(3, Math.floor(Number(C) || 0));
  if (!Cn || !Number.isFinite(Cn)) {
    return { rows: Math.min(N, 256), cols: Math.min(M, 256), kTileSize: Math.min(K, 256) };
  }
  const C_out = Math.max(1, Math.floor(outFrac * Cn));
  const C_in = Math.max(1, Cn - C_out);

  let cols = Math.min(M, Math.max(1, Math.floor(Math.sqrt(C_out))));
  if (align > 1) cols -= (cols % align);
  if (cols < 1) cols = Math.min(M, 1);

  let rows = Math.min(N, Math.max(1, Math.floor(C_out / Math.max(1, cols))));
  if (align > 1) rows -= (rows % align);
  if (rows < 1) rows = Math.min(N, 1);

  let kSpan = Math.floor(C_in / Math.max(1, (rows + cols)));
  if (align > 1) kSpan -= (kSpan % align);
  if (kSpan < 1) kSpan = 1;
  kSpan = Math.min(K, kSpan);

  return { rows, cols, kTileSize: kSpan };
}
/*
export function buildChunker({ taskId, taskDir, K, config, inputFiles }) {
  const Afile = inputFiles.find(f => /A\.bin$/i.test(f.originalName))?.path || inputFiles[0]?.path;
  const Bfile = inputFiles.find(f => /B\.bin$/i.test(f.originalName))?.path || inputFiles[1]?.path;
  if (!Afile || !Bfile) throw new Error('Need A.bin and B.bin input files');

  const { N, K: KK, M } = config;
  const framework = (config.framework || 'cuda').toLowerCase();

  // Choose tile sizes
  const C = Number(config.chunk_size ?? config.C);
  let baseRows, baseCols, kSpan;
  if (config.tileSize || config.kTileSize) {
    const ts = Math.max(1, Number(config.tileSize || 256));
    const ks = Math.max(1, Number(config.kTileSize || Math.min(KK, ts)));
    baseRows = ts; baseCols = ts; kSpan = Math.min(ks, KK);
  } else {
    const pick = pickTileParams({ N, M, K: KK, C: C || 8*1024*1024 });
    baseRows = pick.rows; baseCols = pick.cols; kSpan = pick.kTileSize;
  }

  const fdA = fs.openSync(Afile, 'r');
  const fdB = fs.openSync(Bfile, 'r');
  const nIB = Math.ceil(N / baseRows);
  const nJB = Math.ceil(M / baseCols);

  return {
    async *stream() {
      for (let ib = 0; ib < nIB; ib++) {
        const rNow = Math.min(baseRows, N - ib * baseRows);
        for (let jb = 0; jb < nJB; jb++) {
          const cNow = Math.min(baseCols, M - jb * baseCols);
          for (let kb = 0; kb < KK; kb += kSpan) {
            const kNow = Math.min(kSpan, KK - kb);

            // Read matrix blocks
            const Ablock = readWindow(fdA, ib * baseRows, rNow, kb, kNow, KK, 4);
            const Bblock = readWindow(fdB, kb, kNow, jb * baseCols, cNow, M, 4);

            // Create dimensions array for native clients
            const dims = new Int32Array([rNow, kNow, cNow]);
            const dimsBuffer = Buffer.from(dims.buffer);

            // Prepare payload for native clients
            const payload = {
              // For native clients: binary data + dimensions
              action: framework === 'cpu' ? 'cpu_matmul' : 'compile_and_run',
              framework: framework,
              source: framework !== 'cpu' ? getKernelSource(framework) : undefined,
              inputs: [
                Array.from(new Uint8Array(Ablock)), // A matrix as byte array
                Array.from(new Uint8Array(Bblock)), // B matrix as byte array
                Array.from(new Uint8Array(dimsBuffer)) // dimensions as byte array
              ],
              outputSizes: [rNow * cNow * 4], // Expected output size in bytes
              // Keep original format for compatibility with existing executors
              a: Ablock.buffer.slice(Ablock.byteOffset, Ablock.byteOffset + Ablock.byteLength),
              b: Bblock.buffer.slice(Bblock.byteOffset, Bblock.byteOffset + Bblock.byteLength),
              dims: { rows: rNow, K: kNow, cols: cNow }
            };

            const meta = {
              ib, jb, kb,
              kSpan: kNow,
              rows: rNow,
              cols: cNow,
              baseRows,
              baseCols,
              framework,
              // Native execution parameters
              outputSizes: [rNow * cNow * 4],
              uniforms: [rNow, kNow, cNow]
            };

            // Add framework-specific execution parameters
            if (framework === 'cuda') {
              meta.grid = [Math.ceil(cNow/16), Math.ceil(rNow/16), 1];
              meta.block = [16, 16, 1];
            } else if (framework === 'opencl') {
              meta.global = [cNow, rNow, 1];
              meta.local = [16, 16, 1];
            } else if (framework === 'vulkan') {
              meta.groups = [Math.ceil(cNow/16), Math.ceil(rNow/16), 1];
            }

            yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
          }
        }
      }
      fs.closeSync(fdA);
      fs.closeSync(fdB);
      logger.info('native-block-matmul chunker done');
    }
  };
}
*/
function getKernelSource(framework) {
  const kernelPaths = {
    'cuda': 'kernels/block_matrix_multiply_cuda_kernel.cu',
    'opencl': 'kernels/block_matrix_multiply_opencl_kernel.cl',
    'vulkan': 'kernels/block_matrix_multiply_vulkan_compute.glsl'
  };

  const kernelPath = kernelPaths[framework];
  if (!kernelPath) return undefined;

  try {
    return fs.readFileSync(path.join(process.cwd(), kernelPath), 'utf-8');
  } catch (e) {
    logger.warn(`Could not read kernel source for ${framework}:`, e.message);
    return undefined;
  }
}

export function buildAssembler({ taskId, taskDir, config }) {
  const { N, M, K } = config;
  const outPath = path.join(taskDir, 'output.bin');

  const fdC = fs.openSync(outPath, 'w+');
  const totalBytes = Number(N) * Number(M) * 4;
  fs.ftruncateSync(fdC, totalBytes);

  const acc = new Map();
  const progressedK = new Map();
  const sizes = new Map();

  const key = (ib, jb) => `${ib},${jb}`;

  function writeTileToFile(tileF32, ib, jb, rows, cols, baseRows, baseCols) {
    const rowBytes = cols * 4;
    for (let r = 0; r < rows; r++) {
      const globalRow = ib * baseRows + r;
      const globalColStart = jb * baseCols;
      const fileIndex = globalRow * M + globalColStart;
      const fileOffsetBytes = fileIndex * 4;
      const buf = Buffer.from(tileF32.buffer, tileF32.byteOffset + r * cols * 4, rowBytes);
      fs.writeSync(fdC, buf, 0, rowBytes, fileOffsetBytes);
    }
  }

  return {
    integrate({ result, meta }) {
      const { ib, jb, kSpan, rows, cols, baseRows, baseCols } = meta;
      const k = key(ib, jb);
      if (!sizes.has(k)) sizes.set(k, { rows, cols, baseRows, baseCols });

      // Handle result from native clients (might be byte array)
      let part;
      if (Array.isArray(result)) {
        // Convert byte array back to Float32Array
        const uint8Array = new Uint8Array(result);
        part = new Float32Array(uint8Array.buffer);
      } else {
        part = toF32(result);
      }

      let tile = acc.get(k);
      if (!tile) {
        tile = new Float32Array(rows * cols);
        acc.set(k, tile);
        progressedK.set(k, 0);
      }

      // Accumulate partial result
      for (let i = 0; i < part.length; i++) tile[i] += part[i];

      const soFar = (progressedK.get(k) || 0) + kSpan;
      progressedK.set(k, soFar);

      if (soFar >= K) {
        const s = sizes.get(k);
        writeTileToFile(tile, ib, jb, s.rows, s.cols, s.baseRows, s.baseCols);
        acc.delete(k);
        progressedK.delete(k);
        sizes.delete(k);
      }
    },
    finalize() {
      // Flush remaining tiles
      for (const [k, tile] of acc) {
        const [ibS, jbS] = k.split(',').map(Number);
        const s = sizes.get(k);
        if (s) writeTileToFile(tile, ibS, jbS, s.rows, s.cols, s.baseRows, s.baseCols);
      }
      fs.closeSync(fdC);
      return { outPath, elements: N * M };
    }
  };
}