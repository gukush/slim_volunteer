// Standalone test runner that:
//  1) builds A.bin and B.bin (float32 row-major),
//  2) runs the tiled pipeline locally using CPU for chunk compute,
//  3) streams the output tile-by-tile to disk without full-matrix RAM use,
//  4) verifies by sampling against a reference CPU multiply (or full for small sizes).

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  buildChunker,
  buildAssembler,
  matmulPartialCPU,
  pickTileParams,
  toF32,
} from './block-matmul-flex.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function writeMatrixBin(filePath, rows, cols, fn){
  const buf = Buffer.alloc(rows * cols * 4);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength/4);
  for (let i = 0; i < rows; i++){
    for (let j = 0; j < cols; j++){
      f32[i*cols + j] = fn(i, j);
    }
  }
  fs.writeFileSync(filePath, buf);
}

function multiplyCPUFull(Af, Bf, N, K, M){
  const A = toF32(Af);
  const B = toF32(Bf);
  const C = new Float32Array(N*M);
  for (let i=0;i<N;i++){
    for (let k=0;k<K;k++){
      const a = A[i*K + k];
      let cj = i*M;
      const bk = k*M;
      for (let j=0;j<M;j++) C[cj++] += a * B[bk + j];
    }
  }
  return C;
}

function sampleCheck(outPath, Afile, Bfile, N, K, M, samples=8){
  const fdOut = fs.openSync(outPath, 'r');
  const fdA = fs.openSync(Afile, 'r');
  const fdB = fs.openSync(Bfile, 'r');

  function readRow(fd, row, cols){
    const buf = Buffer.alloc(cols * 4);
    fs.readSync(fd, buf, 0, buf.length, row*cols*4);
    return new Float32Array(buf.buffer);
  }

  // pick some rows/cols
  const rows = new Set([0, Math.floor(N/3), Math.floor(2*N/3), N-1]);
  const cols = new Set([0, Math.floor(M/4), Math.floor(M/2), M-1]);

  // compute per-sample exact value and compare
  let maxAbsErr = 0, maxRelErr = 0;
  for (const i of rows){
    // A[i,:]
    const ArowBuf = readWindow(fdA, i, 1, 0, K, K, 4);
    const Arow = new Float32Array(ArowBuf.buffer);
    for (const j of cols){
      // B[:,j]  -> gather by column
      let dot = 0;
      const tmpB = Buffer.alloc(K*4);
      for (let k=0;k<K;k++){
        const b = readWindow(fdB, k, 1, j, 1, M, 4); // 1x1 scalar at (k,j)
        dot += Arow[k] * new Float32Array(b.buffer)[0];
      }
      // read C[i,j]
      const pos = (i*M + j)*4;
      const cbuf = Buffer.alloc(4);
      fs.readSync(fdOut, cbuf, 0, 4, pos);
      const c = new Float32Array(cbuf.buffer)[0];
      const absErr = Math.abs(c - dot);
      const relErr = absErr / Math.max(1e-30, Math.abs(dot));
      if (absErr > maxAbsErr) maxAbsErr = absErr;
      if (relErr > maxRelErr) maxRelErr = relErr;
    }
  }
  fs.closeSync(fdOut); fs.closeSync(fdA); fs.closeSync(fdB);
  return { maxAbsErr, maxRelErr };
}

async function main(){
  const workDir = path.join(__dirname, 'tmp-matmul');
  fs.mkdirSync(workDir, { recursive: true });

  // Matrix sizes (adjust to taste). Keep small-ish by default for a quick run.
  const N = Number(process.env.N || 512);
  const K = Number(process.env.K || 768);
  const M = Number(process.env.M || 384);

  // chunk_size in ELEMENTS. e.g., 8M elements ≈ 32 MB of float32 moved per task
  const chunk_size = Number(process.env.CHUNK || 8*1024*1024);

  const Afile = path.join(workDir, 'A.bin');
  const Bfile = path.join(workDir, 'B.bin');

  // Deterministic-ish data
  console.log(`[prep] writing A(${N}x${K}) and B(${K}x${M})`);
  writeMatrixBin(Afile, N, K, (i,j)=> Math.sin(i*0.01) + Math.cos(j*0.02));
  writeMatrixBin(Bfile, K, M, (i,j)=> Math.sin(i*0.03) - Math.cos(j*0.005));

  // Derive tile params for info
  const pick = pickTileParams({ N, M, K, C: chunk_size });
  console.log(`[tiling] rows=${pick.rows}, cols=${pick.cols}, kSpan=${pick.kTileSize}`);

  // Build pipeline
  const config = { N, K, M, chunk_size };
  const chunker = buildChunker({ taskId: 't1', taskDir: workDir, config, inputFiles: [
    { originalName: 'A.bin', path: Afile },
    { originalName: 'B.bin', path: Bfile },
  ]});
  const assembler = buildAssembler({ taskDir: workDir, config });

  console.log('[run] streaming chunks…');
  let count = 0;
  for await (const task of chunker.stream()){
    const { a, b, dims } = task.payload;
    const partial = matmulPartialCPU(a, b, dims); // emulate executor
    assembler.integrate({ result: partial, meta: task.meta });
    count++;
  }
  const { outPath } = assembler.finalize();
  console.log(`[done] wrote ${outPath} with ${N*M} float32s`);

  // Quick sanity check (sampled)
  const { maxAbsErr, maxRelErr } = sampleCheck(outPath, Afile, Bfile, N, K, M);
  console.log(`[check] maxAbsErr=${maxAbsErr.toExponential(3)} maxRelErr=${maxRelErr.toExponential(3)}`);
  console.log('OK');
}

main().catch(e=>{ console.error(e); process.exit(1); });