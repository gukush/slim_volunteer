// strategies/native-block-matmul.js (Fixed version)
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'native-block-matmul';
export const name = 'Native Block Matrix Multiplication (Binary Execution)';

export function getClientExecutorInfo(config) {
  const framework = (config?.framework || 'native-cuda').toLowerCase();

  switch (framework) {
    case 'native-cuda':
      return {
        framework: 'cuda',
        path: 'executors/native-bridge.client.js',
        kernels: ['kernels/block_matrix_multiply_cuda_kernel.cu'],
        schema: {
          action: 'compile_and_run',
          inputs: ['float32[]', 'float32[]', 'int32[]'],
          outputs: ['float32[]']
        }
      };
    case 'native-opencl':
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
    case 'native-vulkan':
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
        kernels: [],
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

// Helper functions (copied from block-matmul-flex.js)
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

// ASYNC row-major window reader (fixed from sync version)
async function readWindowAsync(filePath, rowStart, rowCount, colStart, colCount, rowLen, elementSize = 4) {
  const out = Buffer.alloc(rowCount * colCount * elementSize);
  const rowBytes = colCount * elementSize;

  const fileHandle = await fs.promises.open(filePath, 'r');
  try {
    for (let r = 0; r < rowCount; r++) {
      const srcOff = ((rowStart + r) * rowLen + colStart) * elementSize;
      const dstOff = r * rowBytes;
      await fileHandle.read(out, dstOff, rowBytes, srcOff);
    }
  } finally {
    await fileHandle.close();
  }
  return out;
}

function pickTileParams({ N, M, K, C, outFrac = 1/3, align = 32 }) {
  const Cn = Math.max(3, Math.floor(Number(C) || 0));
  if (!Cn || !Number.isFinite(Cn)) {
    return { rows: Math.min(N, 512), cols: Math.min(M, 512), kTileSize: Math.min(K, 512) };
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

function pickInputs(files, N, K, M) {
  if (!files || files.length < 2) throw new Error('No input files uploaded');

  const nameOf = f => f.originalName || (f.path ? path.basename(f.path) : '');
  const endsWithA = f => /(^|[_\-])A\.bin$/i.test(nameOf(f));
  const endsWithB = f => /(^|[_\-])B\.bin$/i.test(nameOf(f));

  let Af = files.find(endsWithA);
  let Bf = files.find(endsWithB);
  if (Af && Bf) return [Af.path, Bf.path];

  // Size-based fallback
  const withSize = files.map(f => ({ ...f, size: f.size ?? (f.path ? fs.statSync(f.path).size : 0) }));
  const targetA = BigInt(N) * BigInt(K) * 4n;
  const targetB = BigInt(K) * BigInt(M) * 4n;
  const closest = target => withSize.reduce((best, cur) => {
    const s = BigInt(cur.size);
    const d = s > target ? s - target : target - s;
    return (!best || d < best.diff) ? { diff: d, f: cur } : best;
  }, null)?.f;

  Af = Af || closest(targetA);
  Bf = Bf || closest(targetB);

  if (!Af || !Bf || Af.path === Bf.path) {
    if (files[0]?.path && files[1]?.path) return [files[0].path, files[1].path];
    throw new Error('Need two input files');
  }
  return [Af.path, Bf.path];
}

// Fixed chunker with proper async operations
export function buildChunker({ taskId, taskDir, K, config, inputFiles }) {
  const [Afile, Bfile] = pickInputs(inputFiles, config.N, config.K, config.M);
  const { N, K: KK, M } = config;

  // Use same tile parameter logic as block-matmul-flex.js
  const C = Number(config.chunk_size ?? config.C ?? 16777216); // 16MB default
  let baseRows, baseCols, kSpan;

  if (config.tileSize || config.kTileSize) {
    const ts = Math.max(1, Number(config.tileSize || 256));
    const ks = Math.max(1, Number(config.kTileSize || Math.min(KK, ts)));
    baseRows = ts; baseCols = ts; kSpan = Math.min(ks, KK);
  } else {
    const pick = pickTileParams({ N, M, K: KK, C });
    baseRows = pick.rows; baseCols = pick.cols; kSpan = pick.kTileSize;
  }

  const nIB = Math.ceil(N / baseRows);
  const nJB = Math.ceil(M / baseCols);
  const totalChunks = nIB * nJB * Math.ceil(KK / kSpan);

  logger.info(`Native chunker: ${nIB}×${nJB}×${Math.ceil(KK/kSpan)} = ${totalChunks.toLocaleString()} chunks`);

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
              // ASYNC file reading (fixed)
              const [Ablock, Bblock] = await Promise.all([
                readWindowAsync(Afile, ib * baseRows, rNow, kb, kNow, KK, 4),
                readWindowAsync(Bfile, kb, kNow, jb * baseCols, cNow, M, 4)
              ]);

              // Prepare data for native client (compatible with server_client.cpp format)
              const dims = new Int32Array([rNow, kNow, cNow]);

              // Convert to base64 for JSON transport (matching server_client.cpp expectations)
              const aBase64 = Ablock.toString('base64');
              const bBase64 = Bblock.toString('base64');
              const dimsBase64 = Buffer.from(dims.buffer).toString('base64');

              const framework = (config.framework || 'native-cuda').toLowerCase();

              const payload = {
                // Native client format (matching server_client.cpp)
                action: framework === 'cpu' ? 'cpu_matmul' : 'compile_and_run',
                framework: framework.replace('native-', ''),
                entry: 'main',
                inputs: [
                  { data: aBase64 },  // A matrix
                  { data: bBase64 },  // B matrix
                  { data: dimsBase64 } // dimensions
                ],
                outputSizes: [rNow * cNow * 4],
                uniforms: [rNow, kNow, cNow],

                // Keep compatibility fields for other executors
                a: Ablock.buffer.slice(Ablock.byteOffset, Ablock.byteOffset + Ablock.byteLength),
                b: Bblock.buffer.slice(Bblock.byteOffset, Bblock.byteOffset + Bblock.byteLength),
                dims: { rows: rNow, K: kNow, cols: cNow }
              };

              const meta = {
                ib, jb, kb, kSpan: kNow, rows: rNow, cols: cNow, baseRows, baseCols,
                framework,
                outputSizes: [rNow * cNow * 4],
                uniforms: [rNow, kNow, cNow]
              };

              // Add framework-specific execution parameters
              if (framework === 'native-cuda') {
                meta.grid = [Math.ceil(cNow/16), Math.ceil(rNow/16), 1];
                meta.block = [16, 16, 1];
                payload.grid = meta.grid;
                payload.block = meta.block;
              } else if (framework === 'native-opencl') {
                meta.global = [cNow, rNow, 1];
                meta.local = [16, 16, 1];
                payload.global = meta.global;
                payload.local = meta.local;
              } else if (framework === 'native-vulkan') {
                meta.groups = [Math.ceil(cNow/16), Math.ceil(rNow/16), 1];
                payload.groups = meta.groups;
              }

              yield {
                id: uuidv4(),
                payload,
                meta,
                tCreate: Date.now()
              };

              chunkCount++;

              // Progress logging
              if (chunkCount % 100 === 0) {
                logger.info(`Generated ${chunkCount.toLocaleString()}/${totalChunks.toLocaleString()} chunks`);
              }

              // Yield control to prevent event loop blocking
              if (chunkCount % 25 === 0) {
                await new Promise(resolve => setImmediate(resolve));
              }

            } catch (error) {
              logger.error(`Native chunk generation failed at (${ib},${jb},${kb}):`, error.message);
              throw error;
            }
          }
        }
      }

      logger.info(`Native chunker completed: ${chunkCount.toLocaleString()} chunks`);
    }
  };
}

// Fixed assembler (same as block-matmul-flex.js)
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

      // Handle result from native clients (base64 or binary)
      let part;
      if (typeof result === 'string') {
        // Base64 encoded result from native client
        const buffer = Buffer.from(result, 'base64');
        part = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
      } else if (Array.isArray(result)) {
        // Byte array from native client
        const uint8Array = new Uint8Array(result);
        part = new Float32Array(uint8Array.buffer);
      } else {
        // Direct binary data
        part = toF32(result);
      }

      let tile = acc.get(k);
      if (!tile) {
        tile = new Float32Array(rows * cols);
        acc.set(k, tile);
        progressedK.set(k, 0);
      }

      // Accumulate partial result
      for (let i = 0; i < Math.min(part.length, tile.length); i++) {
        tile[i] += part[i];
      }

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