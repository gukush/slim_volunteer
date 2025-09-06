// server/strategies/native-block-matmul-flex.js
// Equivalent to block-matmul-flex, but targets the native binary route (no browser executor).
// It prepares chunks as raw buffers for a native C++ client connected over ws-native.
// Buffer order is: UNIFORMS -> INPUTS -> OUTPUTS (placeholder sizes), as requested.

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'native-block-matmul-flex';
export const name = 'Block Matmul (native binary, chunked, streaming)';

// Native client: we just declare the framework and the buffer schema.
// No browser executor path is returned.
export function getClientExecutorInfo(config){
  const backend = (config?.backend || 'opencl').toLowerCase();
  if (!['opencl','cuda','vulkan'].includes(backend)) {
    throw new Error(`Unsupported native backend: ${backend}`);
  }
  // We still pass kernels for reference (e.g., OpenCL kernel source) in case the native runtime wants it,
  // but the native client is free to ignore this and use its compiled binary instead.
  const kernels = {
    opencl: ['kernels/block_matrix_multiply_opencl_kernel.cl'],
    cuda:   ['kernels/block_matrix_multiply_cuda_kernel.cu'],
    vulkan: ['kernels/block_matrix_multiply_vulkan_compute.glsl']
  }[backend];

  return {
    framework: backend,
    kernels,
    schema: {
      order: ['UNIFORMS','INPUTS','OUTPUTS'],
      uniforms: [ { name: 'rows', type: 'i32' }, { name: 'K', type: 'i32' }, { name: 'cols', type: 'i32' } ],
      // two inputs: A (rows x K), B (K x cols); one output: C (rows x cols)
      inputs:  [ { name: 'A', type: 'f32' }, { name: 'B', type: 'f32' } ],
      outputs: [ { name: 'C', type: 'f32' } ]
    }
  };
}

export function getArtifacts(config){
	const backend = (config?.backend || 'opencl').toLowerCase();
	const defaultBins = {
		opencl: 'scripts/native/ocl_block_matmul', // built from this file
		cuda: 'scripts/native/cuda_block_matmul', // if you add later
		vulkan: 'scripts/native/vk_block_matmul' // if you add later
	};
	const rel = config.binary || defaultBins[backend];
	const abs = path.isAbsolute(rel) ? rel : path.join(process.cwd(), 'server', rel);
	const bytes = fs.readFileSync(abs);
	return [{
		type: 'binary',
		name: path.basename(rel),
		program: config.program || 'block_matmul_native',
		backend,
		bytes,
		exec: true
	}, {
		type: 'input',
		name: 'A.bin',
		description: 'Matrix A (rows x K)',
		required: true
	}, {
		type: 'input',
		name: 'B.bin',
		description: 'Matrix B (K x cols)',
		required: true
	}];
}

function toF32(x){
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  if (ArrayBuffer.isView(x)) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength/4));
  if (typeof Buffer !== 'undefined' && x instanceof Buffer) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength/4));
  if (x && x.type === 'Buffer' && Array.isArray(x.data)) return new Float32Array(Uint8Array.from(x.data).buffer);
  throw new Error('Unsupported buffer type for Float32 view');
}

// Row-major window reader: returns Buffer of (rowCount x colCount) starting at (rowStart, colStart)
function readWindow(fd, rowStart, rowCount, colStart, colCount, rowLen, elementSize=4){
  const out = Buffer.alloc(rowCount * colCount * elementSize);
  const rowBytes = colCount * elementSize;
  for (let r = 0; r < rowCount; r++){
    const srcOff = ((rowStart + r) * rowLen + colStart) * elementSize;
    const dstOff = r * rowBytes;
    fs.readSync(fd, out, dstOff, rowBytes, srcOff);
  }
  return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
}

function pickTileParams({ N, M, K, C }){
  // choose a roughly square tile within the budget C (in elements)
  const kTile = Math.min(K, 256);
  const perElem = 4; // f32
  const budgetBytes = Math.max(1, Number(C) || 8*1024*1024);
  // memory per chunk ~ rows*kTile + kTile*cols + rows*cols + small uniforms
  // assume rows ~= cols = t
  // bytes ~ perElem*(t*kTile + kTile*t + t*t) = perElem*(2tk + t^2)
  const k = kTile;
  const t = Math.max(16, Math.min(1024, Math.floor(Math.sqrt(budgetBytes/perElem))));
  const rows = Math.min(N, t);
  const cols = Math.min(M, t);
  return { rows, cols, kTileSize: k };
}

function pickInputs(files, N, K, M){
  if (!files || files.length < 2) throw new Error('No input files uploaded');
  const nameOf = f => f.originalName || (f.path ? path.basename(f.path) : '');
  const endsWithA = f => /(^|[_\-])A\.bin$/i.test(nameOf(f));
  const endsWithB = f => /(^|[_\-])B\.bin$/i.test(nameOf(f));
  let Af = files.find(endsWithA);
  let Bf = files.find(endsWithB);
  if (Af && Bf) return [Af.path, Bf.path];

  // Size-based fallback: closest to expected sizes
  const withSize = files.map(f => ({ ...f, size: f.size ?? (f.path ? fs.statSync(f.path).size : 0) }));
  const targetA = BigInt(N) * BigInt(K) * 4n;
  const targetB = BigInt(K) * BigInt(M) * 4n;
  withSize.sort((x,y)=>Number((BigInt(x.size)-targetA)**2n - (BigInt(y.size)-targetA)**2n));
  Af = withSize[0];
  withSize.sort((x,y)=>Number((BigInt(x.size)-targetB)**2n - (BigInt(y.size)-targetB)**2n));
  Bf = withSize[0];
  return [Af.path, Bf.path];
}

export function buildChunker({ taskId, taskDir, K, config, inputFiles }){
  const { N, K: KK, M } = config;
  const [Afile, Bfile] = pickInputs(inputFiles, N, KK, M);
  if (!Afile || !Bfile) throw new Error('Need A.bin and B.bin');

  const C = Number(config.chunk_size ?? config.C);
  let baseRows, baseCols, kSpan;
  if (config.tileSize || config.kTileSize){
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
    async *stream(){
      for (let ib = 0; ib < nIB; ib++){
        const rNow = Math.min(baseRows, N - ib*baseRows);
        for (let jb = 0; jb < nJB; jb++){
          const cNow = Math.min(baseCols, M - jb*baseCols);

          // C tile accumulator hint for native side (optional)
          const outputBytes = rNow * cNow * 4;

          for (let kb = 0; kb < KK; kb += kSpan){
            const kNow = Math.min(kSpan, KK - kb);

            const Ablock = readWindow(fdA, ib*baseRows, rNow, kb, kNow, KK);
            const Bblock = readWindow(fdB, kb, kNow, jb*baseCols, cNow, M);

            const uniforms = new Int32Array([rNow, kNow, cNow]).buffer;

            // Payload prepared in strict order: UNIFORMS, then INPUTS, then OUTPUTS (placeholder)
            const payload = {
              buffers: [ uniforms, Ablock, Bblock ],
              outputs: [ { byteLength: outputBytes } ]
            };

            // Metadata for the native runtime
            const meta = {
              ib, jb, kb,
              rows: rNow, cols: cNow, kNow,
              baseRows, baseCols, kSpan,
              // Helpful hints for different backends
              outputSizes: [outputBytes],
              uniforms: [rNow, kNow, cNow],
              dispatch: {
                opencl: { global: [cNow, rNow, 1], local: [16,16,1] },
                cuda:   { grid:   [Math.ceil(cNow/16), Math.ceil(rNow/16), 1], block: [16,16,1] },
                vulkan: { groups: [Math.ceil(cNow/16), Math.ceil(rNow/16), 1] }
              }
            };

            // (Optional) indicate the program name to run on the native client
            // If omitted, the native client can choose its own default for this strategy/back-end.
            if (config.program) meta.program = String(config.program);

            yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
          }
        }
      }
      fs.closeSync(fdA);
      fs.closeSync(fdB);
      logger.info(`${id} chunker done`);
    }
  };
}

// Assembler: collect partial tiles and stream to C.bin
export function buildAssembler({ taskId, taskDir, K, config }){
  const { N, M } = config;
  const outPath = path.join(taskDir, 'C.bin');
  const fdC = fs.openSync(outPath, 'w');
  const acc = new Map();
  const sizes = new Map();

  function writeTileToFile(tile, ib, jb, rows, cols, baseRows, baseCols){
    const f32 = toF32(tile);
    for (let r = 0; r < rows; r++){
      const start = r * cols;
      const end = start + cols;
      const rowSlice = Buffer.from(f32.subarray(start, end).buffer);
      const fileOffset = ((ib * baseRows + r) * M + jb * baseCols) * 4;
      fs.writeSync(fdC, rowSlice, 0, rowSlice.length, fileOffset);
    }
  }

  return {
    onChunkResult({ chunkId, replica, status, result, checksum, timings, meta }){
      if (status !== 'ok'){
        logger.warn(`${id} chunk ${chunkId} replica ${replica} failed`);
        return false; // let K-replication handle it
      }
      const key = `${meta.ib},${meta.jb}`;
      const tile = acc.get(key) || new Float32Array(meta.baseRows * meta.baseCols);
      const f32 = toF32(result);
      // Accumulate into the tile at (kb-segment)
      for (let r = 0; r < meta.rows; r++){
        for (let c = 0; c < meta.cols; c++){
          let sum = 0.0;
          // Already summed K-slices on native side; this is a single slice result.
          // If native returns the full sum per (ib,jb), we just overwrite; allow either by summing.
          sum = tile[r*meta.baseCols + c] + f32[r*meta.cols + c];
          tile[r*meta.baseCols + c] = sum;
        }
      }
      acc.set(key, tile);
      sizes.set(key, { rows: meta.rows, cols: meta.cols, baseRows: meta.baseRows, baseCols: meta.baseCols });
      return true;
    },

    finalize(){
      for (const [k, tile] of acc){
        const [ibS, jbS] = k.split(',').map(Number);
        const s = sizes.get(k);
        if (s) writeTileToFile(tile, ibS, jbS, s.rows, s.cols, s.baseRows, s.baseCols);
      }
      fs.closeSync(fdC);
      return { outPath, elements: N*M };
    }
  };
}
