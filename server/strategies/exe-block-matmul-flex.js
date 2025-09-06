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
		opencl: 'scripts/native/ocl_block_matmul_chunked', // built from this file
		cuda: 'scripts/native/cuda_block_matmul', // if you add later
		vulkan: 'scripts/native/vk_block_matmul' // if you add later
	};
	const rel = config.binary || defaultBins[backend];
	const abs = path.isAbsolute(rel) ? rel : path.join(process.cwd(), 'server', rel);
	const bytes = fs.readFileSync(abs);

	// Debug output
	console.log(`[DEBUG] getArtifacts - config.program: ${config.program}`);
	console.log(`[DEBUG] getArtifacts - rel: ${rel}`);
	console.log(`[DEBUG] getArtifacts - path.basename(rel): ${path.basename(rel)}`);

	return [{
		type: 'binary',
		name: path.basename(rel),
		program: path.basename(rel), // Use the actual binary name as program name
		backend,
		bytes,
		exec: true
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

// Async version to prevent event loop blocking
async function readWindowAsync(fd, rowStart, rowCount, colStart, colCount, rowLen, elementSize=4){
  const out = Buffer.alloc(rowCount * colCount * elementSize);
  const rowBytes = colCount * elementSize;
  for (let r = 0; r < rowCount; r++){
    const srcOff = ((rowStart + r) * rowLen + colStart) * elementSize;
    const dstOff = r * rowBytes;
    await new Promise((resolve, reject) => {
      fs.read(fd, out, dstOff, rowBytes, srcOff, (err, bytesRead) => {
        if (err) reject(err);
        else resolve(bytesRead);
      });
    });
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

  // Get the binary name for program reference
  const backend = (config?.backend || 'opencl').toLowerCase();
  const defaultBins = {
    opencl: 'scripts/native/ocl_block_matmul_chunked',
    cuda: 'scripts/native/cuda_block_matmul',
    vulkan: 'scripts/native/vk_block_matmul'
  };
  const rel = config.binary || defaultBins[backend];
  const binaryName = config.program || path.basename(rel);

  // Debug output
  console.log(`[DEBUG] buildChunker - config.program: ${config.program}`);
  console.log(`[DEBUG] buildChunker - rel: ${rel}`);
  console.log(`[DEBUG] buildChunker - path.basename(rel): ${path.basename(rel)}`);
  console.log(`[DEBUG] buildChunker - binaryName: ${binaryName}`);

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
      let chunkCount = 0;
      const totalChunks = nIB * nJB * Math.ceil(KK / kSpan);

      for (let ib = 0; ib < nIB; ib++){
        const rNow = Math.min(baseRows, N - ib*baseRows);
        for (let jb = 0; jb < nJB; jb++){
          const cNow = Math.min(baseCols, M - jb*baseCols);

          // C tile accumulator hint for native side (optional)
          const outputBytes = rNow * cNow * 4;

          for (let kb = 0; kb < KK; kb += kSpan){
            const kNow = Math.min(kSpan, KK - kb);

            // Use async file reading to prevent event loop blocking
            const [Ablock, Bblock] = await Promise.all([
              readWindowAsync(fdA, ib*baseRows, rNow, kb, kNow, KK),
              readWindowAsync(fdB, kb, kNow, jb*baseCols, cNow, M)
            ]);

            // Convert uniforms to raw bytes (not base64 encoded)
            const uniforms = new Int32Array([rNow, kNow, cNow]);
            const uniformsBytes = new Uint8Array(uniforms.buffer);

            // Payload prepared in strict order: UNIFORMS, then INPUTS, then OUTPUTS (placeholder)
            // Send as raw binary data, not base64 encoded
            const payload = {
              buffers: [
                Array.from(uniformsBytes),  // Raw bytes for uniforms
                Array.from(new Uint8Array(Ablock)),  // Raw bytes for A
                Array.from(new Uint8Array(Bblock))   // Raw bytes for B
              ],
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
              backend: config.backend || 'opencl',
              dispatch: {
                opencl: { global: [cNow, rNow, 1], local: [16,16,1] },
                cuda:   { grid:   [Math.ceil(cNow/16), Math.ceil(rNow/16), 1], block: [16,16,1] },
                vulkan: { groups: [Math.ceil(cNow/16), Math.ceil(rNow/16), 1] }
              }
            };

            // Set the program name to the actual binary name
            meta.program = binaryName;

            yield { id: uuidv4(), payload, meta, tCreate: Date.now() };

            chunkCount++;

            // Yield control to event loop every 10 chunks to prevent blocking
            if (chunkCount % 10 === 0) {
              await new Promise(resolve => setImmediate(resolve));
            }

            // Log progress every 100 chunks
            if (chunkCount % 100 === 0) {
              logger.info(`Generated ${chunkCount}/${totalChunks} chunks (${Math.round(chunkCount/totalChunks*100)}%)`);
            }
          }
        }
      }
      fs.closeSync(fdA);
      fs.closeSync(fdB);
      logger.info(`${id} chunker done - generated ${chunkCount} chunks`);
    }
  };
}

// Assembler: accumulate partial tiles per (ib,jb) and stream to C.bin
export function buildAssembler({ taskId, taskDir, K, config }) {
  const { N, M } = config;
  const outPath = path.join(taskDir, 'output.bin');

  // Preallocate output
  const fdC = fs.openSync(outPath, 'w+');
  fs.ftruncateSync(fdC, Number(N) * Number(M) * 4);

  // State
  const acc = new Map();          // "ib,jb" -> Float32Array(rows*cols), accumulated sum
  const progressedK = new Map();  // "ib,jb" -> total k covered so far
  const sizes = new Map();        // "ib,jb" -> { rows, cols, baseRows, baseCols }

  const key = (ib, jb) => `${ib},${jb}`;

  const toF32 = (buf) => {
    if (buf instanceof Float32Array) return buf;
    if (Buffer.isBuffer(buf)) return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    // ArrayBuffer or TypedArray-like
    return new Float32Array(buf);
  };

  function writeTileToFile(tileF32, ib, jb, rows, cols, baseRows, baseCols) {
    const rowBytes = cols * 4;
    for (let r = 0; r < rows; r++) {
      const globalRow = ib * baseRows + r;
      const globalColStart = jb * baseCols;
      const fileIndex = globalRow * M + globalColStart;
      const fileOffsetBytes = fileIndex * 4;
      const slice = Buffer.from(tileF32.buffer, tileF32.byteOffset + r * cols * 4, rowBytes);
      fs.writeSync(fdC, slice, 0, rowBytes, fileOffsetBytes);
    }
  }

  function onChunkResult({ chunkId, result, meta }) {
    // meta MUST include: ib, jb, kSpan, rows, cols, baseRows, baseCols
    const { ib, jb, kSpan, rows, cols, baseRows, baseCols } = meta;
    const k = key(ib, jb);
    if (!sizes.has(k)) sizes.set(k, { rows, cols, baseRows, baseCols });

    console.log(`[ASSEMBLER] Chunk ${chunkId}: result type=${typeof result}, isBuffer=${Buffer.isBuffer(result)}, isArrayBuffer=${result instanceof ArrayBuffer}, length=${result?.length || 'N/A'}`);

    const part = toF32(result);
    let tile = acc.get(k);
    if (!tile) {
      tile = new Float32Array(rows * cols);
      acc.set(k, tile);
      progressedK.set(k, 0);
    }

    // accumulate: tile += part
    for (let i = 0; i < part.length; i++) tile[i] += part[i];

    const soFar = (progressedK.get(k) || 0) + Number(kSpan || 0);
    progressedK.set(k, soFar);

    // When we've covered full K for this (ib,jb), flush the tile to disk
    const fullK = Number(config.K ?? K ?? 0);
    console.log(`[ASSEMBLER] Chunk ${chunkId}: ib=${ib}, jb=${jb}, kSpan=${kSpan}, soFar=${soFar}, fullK=${fullK}, resultLength=${part.length}`);

    if (fullK > 0 && soFar >= fullK) {
      console.log(`[ASSEMBLER] Writing tile for ${k}: ${sizes.get(k).rows}x${sizes.get(k).cols}`);
      const s = sizes.get(k);
      writeTileToFile(tile, ib, jb, s.rows, s.cols, s.baseRows, s.baseCols);
      acc.delete(k);
      progressedK.delete(k);
      sizes.delete(k);
    }
  }

  function finalize() {
    // Flush anything that didn’t reach full K (best-effort)
    for (const [k, tile] of acc) {
      const [ibS, jbS] = k.split(',').map(Number);
      const s = sizes.get(k);
      if (s) writeTileToFile(tile, ibS, jbS, s.rows, s.cols, s.baseRows, s.baseCols);
    }
    fs.closeSync(fdC);
    return { outPath, elements: Number(N) * Number(M) };
  }

  // IMPORTANT: shape expected by TaskManager
  return {
    integrate: ({ chunkId, result, meta }) => onChunkResult({ chunkId, result, meta }),
    finalize
  };
}