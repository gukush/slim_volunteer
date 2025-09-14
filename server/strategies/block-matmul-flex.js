// strategies/block-matmul-flex.js - DEBUG VERSION
// Replace your current block-matmul-flex.js with this version temporarily

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'block-matmul-flex';
export const name = 'Block Matmul (switchable framework, chunked, streaming) - DEBUG';

export function getClientExecutorInfo(config){
  const fw = (config?.framework || 'webgpu').toLowerCase();
  if (fw === 'cpp-wasm') {
    return { framework:'cpp-wasm', path:'executors/wasm-block-matmul.client.js', kernels:['kernels/cpp/block_matmul.js'] };
  }
  if (fw === 'webgpu') {
    const datatype = (config?.datatype || 'f32').toLowerCase();
    let kernelPath = 'kernels/webgpu/block_matmul.wgsl';

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

// DEBUG VERSION: Add extensive logging to readWindow
function readWindow(fd, rowStart, rowCount, colStart, colCount, rowLen, elementSize=4){
  const startTime = Date.now();
  const totalBytes = rowCount * colCount * elementSize;

  console.log(`[DEBUG] readWindow: rows ${rowStart}-${rowStart+rowCount-1} (${rowCount} rows), cols ${colStart}-${colStart+colCount-1} (${colCount} cols), rowLen=${rowLen}`);
  console.log(`[DEBUG] readWindow: allocating ${totalBytes} bytes`);

  try {
    const out = Buffer.alloc(totalBytes);
    const rowBytes = colCount * elementSize;

    console.log(`[DEBUG] readWindow: starting ${rowCount} row reads, ${rowBytes} bytes per row`);

    for (let r = 0; r < rowCount; r++){
      const srcOff = ((rowStart + r) * rowLen + colStart) * elementSize;
      const dstOff = r * rowBytes;

      // Log every 100th row and first/last rows
      if (r === 0 || r === rowCount - 1 || r % 100 === 0) {
        console.log(`[DEBUG] readWindow: row ${r}/${rowCount}, srcOff=${srcOff}, dstOff=${dstOff}, reading ${rowBytes} bytes`);
      }

      try {
        fs.readSync(fd, out, dstOff, rowBytes, srcOff);
      } catch (error) {
        console.error(`[DEBUG] readWindow: ERROR at row ${r}, srcOff=${srcOff}:`, error.message);
        throw error;
      }

      // Check for hang - warn if a single row takes too long
      if (r % 10 === 0) {
        const elapsed = Date.now() - startTime;
        if (elapsed > 5000) { // More than 5 seconds
          console.warn(`[DEBUG] readWindow: SLOW - ${elapsed}ms elapsed, only ${r}/${rowCount} rows read`);
        }
      }
    }

    const elapsed = Date.now() - startTime;
    console.log(`[DEBUG] readWindow: completed in ${elapsed}ms, ${totalBytes} bytes read`);

    return out;
  } catch (error) {
    const elapsed = Date.now() - startTime;
    console.error(`[DEBUG] readWindow: FAILED after ${elapsed}ms:`, error.message);
    throw error;
  }
}

function pickTileParams({ N, M, K, C, outFrac = 1/3, align = 32 }){
  console.log(`[DEBUG] pickTileParams: N=${N}, M=${M}, K=${K}, C=${C}`);

  const Cn = Math.max(3, Math.floor(Number(C) || 0));
  if (!Cn || !Number.isFinite(Cn)) {
    console.log(`[DEBUG] pickTileParams: invalid C, using fallback`);
    return { rows: Math.min(N, 256), cols: Math.min(M, 256), kTileSize: Math.min(K, 256) };
  }

  const C_out = Math.max(1, Math.floor(outFrac * Cn));
  const C_in  = Math.max(1, Cn - C_out);

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

  console.log(`[DEBUG] pickTileParams: result = rows=${rows}, cols=${cols}, kTileSize=${kSpan}`);
  return { rows, cols, kTileSize: kSpan };
}

export function buildChunker({ taskId, taskDir, K, config, inputFiles }){
  console.log(`[DEBUG] buildChunker: starting with config:`, config);
  console.log(`[DEBUG] buildChunker: inputFiles:`, inputFiles.map(f => ({ path: f.path, size: f.size })));

  function pickInputs(files, N, K, M){
    console.log(`[DEBUG] pickInputs: looking for A.bin and B.bin in ${files.length} files`);

    if (!files || files.length < 2) throw new Error('No input files provided');

    const nameOf = f => f.originalName || (f.path ? path.basename(f.path) : '');
    const endsWithA = f => /(^|[_\-])A\.bin$/i.test(nameOf(f));
    const endsWithB = f => /(^|[_\-])B\.bin$/i.test(nameOf(f));

    let Af = files.find(endsWithA);
    let Bf = files.find(endsWithB);

    console.log(`[DEBUG] pickInputs: found A file:`, Af ? nameOf(Af) : 'none');
    console.log(`[DEBUG] pickInputs: found B file:`, Bf ? nameOf(Bf) : 'none');

    if (Af && Bf) return [Af.path, Bf.path];

    // Size-based fallback
    const withSize = files.map(f => ({ ...f, size: f.size ?? (f.path ? fs.statSync(f.path).size : 0) }));
    const targetA = BigInt(N) * BigInt(K) * 4n;
    const targetB = BigInt(K) * BigInt(M) * 4n;

    console.log(`[DEBUG] pickInputs: target sizes - A: ${targetA} bytes, B: ${targetB} bytes`);

    const closest = target => withSize.reduce((best, cur) => {
      const s = BigInt(cur.size);
      const d = s > target ? s - target : target - s;
      return (!best || d < best.diff) ? { diff: d, f: cur } : best;
    }, null)?.f;

    Af = Af || closest(targetA);
    Bf = Bf || closest(targetB);

    if (!Af || !Bf || Af.path === Bf.path) {
      if (files[0]?.path && files[1]?.path) {
        console.log(`[DEBUG] pickInputs: using first two files as fallback`);
        return [files[0].path, files[1].path];
      }
      throw new Error('Need two input files');
    }

    console.log(`[DEBUG] pickInputs: final selection - A: ${Af.path}, B: ${Bf.path}`);
    return [Af.path, Bf.path];
  }

  const [Afile, Bfile] = pickInputs(inputFiles, config.N, config.K, config.M);
  if(!Afile || !Bfile) throw new Error('Need A.bin and B.bin');

  const { N, K:KK, M } = config;
  console.log(`[DEBUG] buildChunker: matrix dimensions N=${N}, K=${KK}, M=${M}`);

  // Check actual file sizes
  const AfileSize = fs.statSync(Afile).size;
  const BfileSize = fs.statSync(Bfile).size;
  console.log(`[DEBUG] buildChunker: file sizes - A: ${AfileSize} bytes, B: ${BfileSize} bytes`);

  // Calculate expected vs actual sizes
  const expectedASize = N * KK * 4;
  const expectedBSize = KK * M * 4;
  console.log(`[DEBUG] buildChunker: expected sizes - A: ${expectedASize} bytes, B: ${expectedBSize} bytes`);

  if (AfileSize !== expectedASize || BfileSize !== expectedBSize) {
    console.warn(`[DEBUG] buildChunker: SIZE MISMATCH WARNING!`);
    console.warn(`[DEBUG] A file: expected ${expectedASize}, actual ${AfileSize}`);
    console.warn(`[DEBUG] B file: expected ${expectedBSize}, actual ${BfileSize}`);
  }

  // Choose tile sizes
  const C = Number(config.chunk_size ?? config.C);
  let baseRows, baseCols, kSpan;

  console.log(`[DEBUG] buildChunker: tile size selection - tileSize=${config.tileSize}, kTileSize=${config.kTileSize}, chunk_size=${C}`);

  if (config.tileSize || config.kTileSize){
    const ts = Math.max(1, Number(config.tileSize || 256));
    const ks = Math.max(1, Number(config.kTileSize || Math.min(KK, ts)));
    baseRows = ts; baseCols = ts; kSpan = Math.min(ks, KK);
    console.log(`[DEBUG] buildChunker: using explicit tile sizes - baseRows=${baseRows}, baseCols=${baseCols}, kSpan=${kSpan}`);
  } else {
    const pick = pickTileParams({ N, M, K: KK, C: C || 8*1024*1024 });
    baseRows = pick.rows; baseCols = pick.cols; kSpan = pick.kTileSize;
    console.log(`[DEBUG] buildChunker: computed tile sizes - baseRows=${baseRows}, baseCols=${baseCols}, kSpan=${kSpan}`);
  }

  const nIB = Math.ceil(N / baseRows);
  const nJB = Math.ceil(M / baseCols);
  const nKB = Math.ceil(KK / kSpan);
  const totalChunks = nIB * nJB * nKB;

  console.log(`[DEBUG] buildChunker: grid dimensions - ${nIB} x ${nJB} x ${nKB} = ${totalChunks} total chunks`);
  console.log(`[DEBUG] buildChunker: estimated bytes per chunk: ${baseRows * baseCols * kSpan * 4 * 2} bytes (A+B)`);

  console.log(`[DEBUG] buildChunker: opening files...`);
  const fdA = fs.openSync(Afile, 'r');
  const fdB = fs.openSync(Bfile, 'r');
  console.log(`[DEBUG] buildChunker: files opened successfully - fdA=${fdA}, fdB=${fdB}`);

  return {
    async *stream(){
      console.log(`[DEBUG] chunker stream: starting chunk generation`);
      let chunkCount = 0;
      const startTime = Date.now();
      let lastLogTime = startTime;

      try {
        for (let ib = 0; ib < nIB; ib++){
          const rNow = Math.min(baseRows, N - ib*baseRows);
          console.log(`[DEBUG] chunker: processing ib=${ib}/${nIB}, rows=${rNow}`);

          for (let jb = 0; jb < nJB; jb++){
            const cNow = Math.min(baseCols, M - jb*baseCols);

            console.log(`[DEBUG] chunker: ib=${ib}/${nIB}, jb=${jb}/${nJB} (${((ib*nJB + jb)/(nIB*nJB)*100).toFixed(1)}%)`);

            for (let kb = 0; kb < KK; kb += kSpan){
              const kNow = Math.min(kSpan, KK - kb);

              const chunkStartTime = Date.now();
              console.log(`[DEBUG] chunk ${chunkCount}: START ib=${ib}, jb=${jb}, kb=${kb}, dims=${rNow}x${kNow}x${cNow}`);

              // A slice timing
              const AstartTime = Date.now();
              const Ablock = readWindow(fdA, ib*baseRows, rNow, kb, kNow, KK, 4);
              const AendTime = Date.now();
              console.log(`[DEBUG] chunk ${chunkCount}: A read ${AendTime - AstartTime}ms`);

              // B slice timing
              const BstartTime = Date.now();
              const Bblock = readWindow(fdB, kb, kNow, jb*baseCols, cNow, M, 4);
              const BendTime = Date.now();
              console.log(`[DEBUG] chunk ${chunkCount}: B read ${BendTime - BstartTime}ms`);

              // Payload creation timing
              const payloadStartTime = Date.now();
              const payload = {
                a: Ablock.buffer.slice(Ablock.byteOffset, Ablock.byteOffset + Ablock.byteLength),
                b: Bblock.buffer.slice(Bblock.byteOffset, Bblock.byteOffset + Bblock.byteLength),
                dims: { rows: rNow, K: kNow, cols: cNow },
              };
              const payloadEndTime = Date.now();
              console.log(`[DEBUG] chunk ${chunkCount}: payload creation ${payloadEndTime - payloadStartTime}ms`);

              const meta = { ib, jb, kb, kSpan: kNow, rows: rNow, cols: cNow, baseRows, baseCols };

              const chunkTotalTime = Date.now() - chunkStartTime;
              console.log(`[DEBUG] chunk ${chunkCount}: COMPLETE ${chunkTotalTime}ms total`);

              yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
              chunkCount++;

              // More frequent progress for debugging
              const now = Date.now();
              if (now - lastLogTime > 2000) { // Every 2 seconds
                const elapsed = (now - startTime) / 1000;
                const rate = chunkCount / elapsed;
                const eta = totalChunks > chunkCount ? (totalChunks - chunkCount) / rate : 0;
                console.log(`[DEBUG] PROGRESS: ${chunkCount}/${totalChunks} chunks, ${rate.toFixed(1)}/sec, ETA ${eta.toFixed(0)}s`);
                lastLogTime = now;
              }
            }
          }
        }
      } catch (error) {
        console.error(`[DEBUG] chunker: ERROR during chunk generation:`, error);
        throw error;
      } finally {
        console.log(`[DEBUG] chunker: closing files`);
        fs.closeSync(fdA);
        fs.closeSync(fdB);
      }

      const duration = (Date.now() - startTime) / 1000;
      console.log(`[DEBUG] chunker: completed ${chunkCount} chunks in ${duration.toFixed(1)}s`);
      logger.info('block-matmul-flex chunker done');
    }
  };
}

// Keep original assembler
export function buildAssembler({ taskId, taskDir, config }){
  const { N, M, K } = config;
  const outPath = path.join(taskDir, 'output.bin');

  const fdC = fs.openSync(outPath, 'w+');
  const totalBytes = Number(N) * Number(M) * 4;
  fs.ftruncateSync(fdC, totalBytes);

  const acc = new Map();
  const progressedK = new Map();
  const sizes = new Map();

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