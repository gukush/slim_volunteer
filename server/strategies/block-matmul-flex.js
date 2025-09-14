// strategies/block-matmul-flex.js
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// ---------- Public strategy identifiers (compat with StrategyRegistry) ----------
export const id = 'block-matmul-flex';
export const name = 'Block Matmul (switchable framework, chunked, streaming)';

// ---------- Executor selection (kept compatible with your previous version) ----------
export function getClientExecutorInfo(config){
  const fw = (config?.framework || 'webgpu').toLowerCase();
  if (fw === 'cpp-wasm') {
    return { framework:'cpp-wasm', path:'executors/wasm-block-matmul.client.js', kernels:['kernels/cpp/block_matmul.js'] };
  }
  if (fw === 'webgpu') {
    const datatype = (config?.datatype || 'f32').toLowerCase();
    let kernelPath = 'kernels/webgpu/block_matmul.wgsl'; // default f32
    
    if (datatype === 'f16') {
      kernelPath = 'kernels/webgpu/block_matmul_fp16.wgsl';
    } else if (datatype === 'int8') {
      kernelPath = 'kernels/webgpu/block_matmul_int8.wgsl';
    } else if (datatype !== 'f32') {
      logger.warn(`Unknown datatype '${datatype}', using default f32`);
    }
    
    return { framework:'webgpu', path:'executors/webgpu-block-matmul.client.js', kernels:[kernelPath] };
  }
  if (fw === 'webgl2') {
    return { framework:'webgl2', path:'executors/webgl2-block-matmul.client.js', kernels:[
      'kernels/webgl/block_matrix_multiply_webgl_vertex.glsl',
      'kernels/webgl/block_matrix_multiply_webgl_fragment.glsl'
    ]};
  }
  throw new Error('Unsupported framework in config.framework: '+fw);
}

// ---------- Helpers ----------
function toF32(x){
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  if (ArrayBuffer.isView(x)) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  if (typeof Buffer !== 'undefined' && x instanceof Buffer) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  if (x && x.type === 'Buffer' && Array.isArray(x.data)) {
    const u8 = Uint8Array.from(x.data);
    return new Float32Array(u8.buffer);
  }
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
  return out;
}

// Derive rows, cols, kSpan from chunk_size (C) for good locality
function pickTileParams({ N, M, K, C, outFrac = 1/3, align = 32 }){
  const Cn = Math.max(3, Math.floor(Number(C) || 0));
  if (!Cn || !Number.isFinite(Cn)) {
    // fallback to a modest default if chunk_size missing
    return { rows: Math.min(N, 256), cols: Math.min(M, 256), kTileSize: Math.min(K, 256) };
  }
  const C_out = Math.max(1, Math.floor(outFrac * Cn)); // rows*cols <= C_out
  const C_in  = Math.max(1, Cn - C_out);               // rows*k + k*cols <= C_in

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

// ---------- Chunker (depth-tiling over K, payload bounded by chunk_size) ----------
export function buildChunker({ taskId, taskDir, K, config, inputFiles }){
    function pickInputs(files, N, K, M){
    if (!files || files.length < 2) throw new Error('No input files provided (either upload files or use cachedFilePaths)');

    const nameOf = f => f.originalName || (f.path ? path.basename(f.path) : '');
    const endsWithA = f => /(^|[_\-])A\.bin$/i.test(nameOf(f));
    const endsWithB = f => /(^|[_\-])B\.bin$/i.test(nameOf(f));

    let Af = files.find(endsWithA);
    let Bf = files.find(endsWithB);
    if (Af && Bf) return [Af.path, Bf.path];

    // Size-based fallback: choose closest to expected bytes N*K*4 and K*M*4
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
      // Final fallback: first two files
      if (files[0]?.path && files[1]?.path) return [files[0].path, files[1].path];
      throw new Error('Need two input files');
    }
    return [Af.path, Bf.path];
  }
  const [Afile, Bfile] = pickInputs(inputFiles, config.N, config.K, config.M);
  //const Afile = inputFiles.find(f=>/A\.bin$/i.test(f.originalName))?.path || inputFiles[0]?.path;
  //const Bfile = inputFiles.find(f=>/B\.bin$/i.test(f.originalName))?.path || inputFiles[1]?.path;
  if(!Afile || !Bfile) throw new Error('Need A.bin and B.bin');

  const { N, K:KK, M } = config;

  // Choose tile sizes:
  // if explicit tileSize/kTileSize provided -> honor them (backwards compat)
  // else derive from chunk_size (C in elements)
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
          for (let kb = 0; kb < KK; kb += kSpan){
            const kNow = Math.min(kSpan, KK - kb);

            // A slice: rows x kNow from (rowStart=ib*baseRows, colStart=kb)
            const Ablock = readWindow(fdA, ib*baseRows, rNow, kb, kNow, KK, 4);
            // B slice: kNow x cols from (rowStart=kb, colStart=jb*baseCols)
            const Bblock = readWindow(fdB, kb, kNow, jb*baseCols, cNow, M, 4);

            const payload = {
              a: Ablock.buffer.slice(Ablock.byteOffset, Ablock.byteOffset + Ablock.byteLength),
              b: Bblock.buffer.slice(Bblock.byteOffset, Bblock.byteOffset + Bblock.byteLength),
              // executor multiplies (rows x kNow)*(kNow x cols) -> (rows x cols)
              dims: { rows: rNow, K: kNow, cols: cNow },
            };

            const meta = { ib, jb, kb, kSpan: kNow, rows: rNow, cols: cNow, baseRows, baseCols };


            yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
          }
        }
      }
      fs.closeSync(fdA);
      fs.closeSync(fdB);
      logger.info('block-matmul-flex chunker done');
    }
  };
}

// ---------- Assembler (accumulate per (ib,jb) and stream to disk) ----------
export function buildAssembler({ taskId, taskDir, config }){
  const { N, M, K } = config;
  const outPath = path.join(taskDir, 'output.bin');

  const fdC = fs.openSync(outPath, 'w+');
  const totalBytes = Number(N) * Number(M) * 4;
  fs.ftruncateSync(fdC, totalBytes);

  const acc = new Map();          // key "ib,jb" -> Float32Array(rows*cols)
  const progressedK = new Map();  // key -> accumulated K so far
  const sizes = new Map();        // key -> { rows, cols, baseRows, baseCols }

  const key = (ib,jb)=>`${ib},${jb}`;

  function writeTileToFile(tileF32, ib, jb, rows, cols, baseRows, baseCols){
    const rowBytes = cols * 4;
    for (let r = 0; r < rows; r++){
      const globalRow = ib*baseRows + r;
      const globalColStart = jb*baseCols;
      const fileIndex = globalRow * M + globalColStart;
      const fileOffsetBytes = fileIndex * 4;
      const buf = Buffer.from(tileF32.buffer, tileF32.byteOffset + r*cols*4, rowBytes);
      fs.writeSync(fdC, buf, 0, rowBytes, fileOffsetBytes);
    }
  }

  return {
    integrate({ result, meta }){
      const { ib, jb, kSpan, rows, cols, baseRows, baseCols } = meta;
      const k = key(ib, jb);
      if (!sizes.has(k)) sizes.set(k, { rows, cols, baseRows, baseCols });

      const part = toF32(result);
      let tile = acc.get(k);
      if (!tile){
        tile = new Float32Array(rows * cols);
        acc.set(k, tile);
        progressedK.set(k, 0);
      }
      // accumulate partial: tile += part
      for (let i = 0; i < part.length; i++) tile[i] += part[i];

      const soFar = (progressedK.get(k) || 0) + kSpan;
      progressedK.set(k, soFar);

      if (soFar >= K){
        const s = sizes.get(k);
        writeTileToFile(tile, ib, jb, s.rows, s.cols, s.baseRows, s.baseCols);
        acc.delete(k); progressedK.delete(k); sizes.delete(k);
      }
    },
    finalize(){
      // Best-effort flush (should be empty if all chunks arrived)
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
