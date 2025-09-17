// strategies/native-block-matmul.js
// Runtime-dispatched block matmul for native clients (OpenCL / CUDA / Vulkan / CPU via LuaJIT host)
//
// Key design points:
// - Emits chunks with { payload: { action, framework, entry, inputs[], outputSizes[], uniforms[] } }.
// - Ships a Lua host script as an artifact (host.lua) that selects backend and compiles a kernel at runtime.
// - Optionally ships kernel sources as artifacts if present (CL/CU/GLSL/SPIR-V).
// - Assembler writes the final C.bin in row-major order and supports base64 or raw byte return.

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'native-block-matmul';
export const name = 'Native Block Matrix Multiplication (Native backends via LuaJIT)';

// --------- helpers to resolve optional artifacts (kernels + host.lua) ----------
function tryRead(p){
  try { return fs.readFileSync(p); } catch { return null; }
}

function findFirstExisting(paths){
  for(const p of paths){
    const b = tryRead(p);
    if(b) return { path: p, bytes: b };
  }
  return null;
}

// Resolve project root-ish paths: try CWD, strategies/, kernels/, etc.
function resolveCandidates(rel){
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

function b64(buf){ return Buffer.isBuffer(buf) ? buf.toString('base64') : Buffer.from(buf).toString('base64'); }

// Build a minimal artifact entry shaped for server.js -> wss.send('workload:new', { artifacts: [...] })
function makeArtifact({ type='text', name, program, backend, exec=false, bytes }){
  return { type, name, program, backend, exec, bytes };
}

// Return framework/backend info + optional artifacts (host.lua + kernels if available)
export function getClientExecutorInfo(config){
  const framework = String(config?.framework || 'native-opencl').toLowerCase();

  // Always try to include host.lua so native client can route + compile at runtime.
  const hostCandidates = resolveCandidates('executors/host_block_matmul.lua');
  const host = findFirstExisting(hostCandidates);

  // Optional kernels (these are just hints; Lua host can also generate or use its own)
  const ku = findFirstExisting(resolveCandidates('cuda/block_matrix_multiply_cuda_kernel.cu'));
  const kl = findFirstExisting(resolveCandidates('opencl/block_matrix_multiply.cl'));
  const kv = findFirstExisting(resolveCandidates('vulkan/block_matmul_vulkan_accelerated.glsl')) ||
             findFirstExisting(resolveCandidates('vulkan/block_matrix_multiply_vulkan_compute.glsl'));
  //const kspv = findFirstExisting(resolveCandidates('kernels/block_matrix_multiply_vulkan_compute.spv'));

  const artifacts = [];
  if (host) artifacts.push(makeArtifact({
    type: 'lua', name: 'host.lua', program: 'host', backend: 'host', bytes: b64(host.bytes)
  }));
  if (kl) artifacts.push(makeArtifact({
    type: 'text', name: path.basename(kl.path), program: 'block_matmul', backend: 'opencl', bytes: b64(kl.bytes)
  }));
  if (ku) artifacts.push(makeArtifact({
    type: 'text', name: path.basename(ku.path), program: 'block_matmul', backend: 'cuda', bytes: b64(ku.bytes)
  }));
  if (kv) artifacts.push(makeArtifact({
    type: 'text', name: path.basename(kv.path), program: 'block_matmul', backend: 'vulkan', bytes: b64(kv.bytes)
  }));
  //if (kspv) artifacts.push(makeArtifact({
  //  type: 'binary', name: path.basename(kspv.path), program: 'block_matmul', backend: 'vulkan', exec: true, bytes: kspv.bytes
  //}));
  const schema = {
    action: 'compile_and_run',
    order: ['UNIFORMS','INPUTS','OUTPUTS'],
    uniforms: [
      { name: 'rows', type: 'i32' },
      { name: 'K',    type: 'i32' },
      { name: 'cols', type: 'i32' },
    ],
    inputs: [
      { name: 'A', type: 'f32' }, // rows x K
      { name: 'B', type: 'f32' }, // K x cols
    ],
    outputs: [
      { name: 'C', type: 'f32' }, // rows x cols
    ],
  };
  switch (framework) {
    case 'native-cuda':
      return {
        framework: 'cuda',
        kernels: ku ? [path.basename(ku.path)] : [],
        schema,
        artifacts,
      };
    case 'native-opencl':
      return {
        framework: 'opencl',
        kernels: kl ? [path.basename(kl.path)] : [],
        schema,
        artifacts,
      };
    case 'native-vulkan':
      return {
        framework: 'vulkan',
        kernels: kv ? [ path.basename(kv.path) ] : [], //kspv || kv
        schema,
        artifacts,
      };
    case 'cpu':
      return {
        framework: 'cpu',
        kernels: [],
        schema,
        artifacts,
      };
    default:
      throw new Error(`Unsupported framework: ${framework}`);
  }
}

// ----------------- numeric helpers -----------------
function toF32(x){
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  if (ArrayBuffer.isView(x)) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength/4));
  if (Array.isArray(x)) return Float32Array.from(x);
  throw new Error('Unsupported buffer type for Float32 view');
}

async function readWindowAsync(filePath, rowStart, rowCount, colStart, colCount, rowLen, elementSize=4){
  const out = Buffer.alloc(rowCount * colCount * elementSize);
  const rowBytes = colCount * elementSize;
  const fh = await fs.promises.open(filePath, 'r');
  try{
    for(let r=0;r<rowCount;r++){
      const src = ((rowStart + r) * rowLen + colStart) * elementSize;
      const dst = r * rowBytes;
      await fh.read(out, dst, rowBytes, src);
    }
  } finally {
    await fh.close();
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

  let cols = Math.min(M, Math.max(1, Math.floor(C_out / Math.max(1, Math.floor(Math.sqrt(C_out))))));
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

function pickInputs(files, N, K, M){
  // prefer exact names A.bin/B.bin, else fallback to 2 largest .bin
  const bins = files.filter(f => f.originalName && /\.bin$/i.test(f.originalName));
  const byNameA = bins.find(f => /(^|\/)A\.bin$/i.test(f.originalName) || /(^|\/)a\.bin$/i.test(f.originalName));
  const byNameB = bins.find(f => /(^|\/)B\.bin$/i.test(f.originalName) || /(^|\/)b\.bin$/i.test(f.originalName));
  if (byNameA && byNameB) return [byNameA.path, byNameB.path];
  const sorted = bins.slice().sort((a,b)=>b.size-a.size);
  if (sorted.length >= 2) return [sorted[0].path, sorted[1].path];
  throw new Error('Need two input files (A.bin and B.bin)');
}

// Data type helpers for native strategy
function getDataTypeInfo(datatype) {
  const type = (datatype || 'f32').toLowerCase();
  switch (type) {
    case 'f32':
    case 'int32':
      return { elementSize: 4, isPacked: false, packFactor: 1 };
    case 'f16':
      return { elementSize: 2, isPacked: false, packFactor: 1 }; // Native fp16, no packing
    case 'int8':
      return { elementSize: 1, isPacked: true, packFactor: 4 };
    default:
      logger.warn(`Unknown datatype '${type}', using default f32`);
      return { elementSize: 4, isPacked: false, packFactor: 1 };
  }
}

function packData(buffer, datatype) {
  const typeInfo = getDataTypeInfo(datatype);
  if (!typeInfo.isPacked) return buffer;

  const view = new Uint8Array(buffer);

  if (typeInfo.elementSize === 1) { // int8 -> pack 4 values per 32-bit word
    const packedSize = Math.ceil(view.length / 4) * 4;
    const packed = new Uint8Array(packedSize);
    const output = new Uint32Array(packed.buffer, packed.byteOffset, packedSize / 4);

    for (let i = 0; i < view.length; i += 4) {
      const val1 = view[i] || 0;
      const val2 = view[i + 1] || 0;
      const val3 = view[i + 2] || 0;
      const val4 = view[i + 3] || 0;
      output[i / 4] = val1 | (val2 << 8) | (val3 << 16) | (val4 << 24);
    }

    return packed.buffer;
  }

  return buffer;
}

// ----------------- chunker -----------------
export function buildChunker({ taskId, taskDir, K, config, inputFiles }){
  const { N, K: KK, M } = config;
  const datatype = config.datatype || 'f32';
  const typeInfo = getDataTypeInfo(datatype);

  const [Afile, Bfile] = pickInputs(inputFiles, N, KK, M);

  const C = Number(config.chunk_size ?? config.C ?? 16*1024*1024); // ~16MB default
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
  logger.info(`Native chunker: datatype=${datatype}, elementSize=${typeInfo.elementSize}, isPacked=${typeInfo.isPacked}, ${nIB}×${nJB}×${Math.ceil(KK/kSpan)} = ${totalChunks.toLocaleString()} chunks`);

  return {
    async *stream(){
      let chunkCount = 0;
      for(let ib=0; ib<nIB; ib++){
        const rNow = Math.min(baseRows, N - ib * baseRows);
        for(let jb=0; jb<nJB; jb++){
          const cNow = Math.min(baseCols, M - jb * baseCols);
          for(let kb=0; kb<KK; kb += kSpan){
            const kNow = Math.min(kSpan, KK - kb);

            // read tiles with correct element size
            const [Ablock, Bblock] = await Promise.all([
              readWindowAsync(Afile, ib*baseRows, rNow, kb,   kNow, KK, typeInfo.elementSize),
              readWindowAsync(Bfile, kb,          kNow, jb*baseCols, cNow, M, typeInfo.elementSize),
            ]);

            // Pack data if needed for non-32bit types
            const aData = packData(Ablock.buffer.slice(Ablock.byteOffset, Ablock.byteOffset + Ablock.byteLength), datatype);
            const bData = packData(Bblock.buffer.slice(Bblock.byteOffset, Bblock.byteOffset + Bblock.byteLength), datatype);

            // For int8, we need to adjust dimensions to account for packing
            let dims;
            if (datatype === 'int8') {
              const groupsK = Math.ceil(kNow / 4); // 4 int8 values per 32-bit word
              dims = new Int32Array([rNow, kNow, cNow, groupsK]);
            } else {
              dims = new Int32Array([rNow, kNow, cNow, 0]); // pad to 16B
            }

            const aBase64 = Buffer.from(aData).toString('base64');
            const bBase64 = Buffer.from(bData).toString('base64');
            const dBase64 = Buffer.from(dims.buffer).toString('base64');
            const framework = String(config.framework || 'native-opencl').toLowerCase().replace(/^native-/, '');

            const payload = {
              action: framework === 'cpu' ? 'cpu_matmul' : 'compile_and_run',
              framework, // 'opencl' | 'cuda' | 'vulkan' | 'cpu'
              entry: 'execute_task',
              // For OpenCL/CUDA, pass only A and B. Dims go via uniforms.
              inputs: [{ data: aBase64 }, { data: bBase64 }],
              outputSizes: [rNow * cNow * 4],
              uniforms: [rNow, kNow, cNow],
            };

            // Add CUDA-specific launch dimensions
            if (framework === 'cuda') {
              const TILE = config.tileSize ?? 16;  // match your CUDA kernel TILE

              // CUDA has a maximum of 1024 threads per block
              // If TILE^2 > 1024, we need to adjust the block dimensions
              const maxThreadsPerBlock = 1024;
              let blockX, blockY;

              if (TILE * TILE <= maxThreadsPerBlock) {
                // Use square blocks if possible
                blockX = TILE;
                blockY = TILE;
              } else {
                // Adjust to fit within thread limit
                // Try to keep blocks as square as possible
                const maxDim = Math.floor(Math.sqrt(maxThreadsPerBlock));
                blockX = Math.min(TILE, maxDim);
                blockY = Math.min(TILE, Math.floor(maxThreadsPerBlock / blockX));
              }

              payload.block = [blockX, blockY, 1];
              payload.grid = [Math.ceil(cNow / blockX), Math.ceil(rNow / blockY), 1];
            }

            // If you later add a Vulkan path that *requires* a uniform buffer,
            // you can opt-in to append it here behind a flag:
            // if (framework === 'vulkan' && config.useUniformBuffer) {
            //   payload.inputs.push({ data: dBase64 });
            // }

            const meta = {
              ib, jb, kb,
              rows: rNow, cols: cNow, kSpan: kNow,
              baseRows, baseCols,
            };

            yield {  id: uuidv4(), payload, meta, tCreate: Date.now() };
            chunkCount++;
            if (chunkCount % 50 === 0) await new Promise(r=>setImmediate(r));
          }
        }
      }
      logger.info(`Native chunker completed: ${chunkCount.toLocaleString()} chunks`);
    }
  };
}

// ----------------- assembler -----------------
export function buildAssembler({ taskId, taskDir, config }){
  const { N, M } = config;
  const outPath = path.join(taskDir, `C_${uuidv4()}.bin`);
  const fdC = fs.openSync(outPath, 'w+');
  fs.ftruncateSync(fdC, N * M * 4);

  const acc = new Map();
  const sizes = new Map();
  const key = (i,j)=>`${i},${j}`;

  function writeTileToFile(tileF32, ib, jb, rows, cols, baseRows, baseCols){
    const rowBytes = cols * 4;
    for(let r=0;r<rows;r++){
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
      const { ib, jb, rows, cols, baseRows, baseCols } = meta;
      const k = key(ib, jb);
      if (!sizes.has(k)) sizes.set(k, { rows, cols, baseRows, baseCols });

      let part;
      if (typeof result === 'string') {
        const buffer = Buffer.from(result, 'base64');
        part = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength/4);
      } else if (typeof Buffer !== 'undefined' && Buffer.isBuffer && Buffer.isBuffer(result)) {
      // Node Buffer (what TaskManager passes after decoding base64)
      part = new Float32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
      } else if (Array.isArray(result)) {
        const u8 = new Uint8Array(result);
        part = new Float32Array(u8.buffer, u8.byteOffset, Math.floor(u8.byteLength/4));
      } else if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
        const u8 = Uint8Array.from(result.data);
        part = new Float32Array(u8.buffer);
      } else if (result instanceof ArrayBuffer) {
        part = new Float32Array(result);
      } else {
        throw new Error('Unknown result format from native client');
      }

      let tile = acc.get(k);
      if (!tile) {
        tile = new Float32Array(baseRows * baseCols);
        acc.set(k, tile);
      }

      for(let r=0;r<rows;r++){
        for(let c=0;c<cols;c++){
          const idx = r*cols + c;
          tile[r*baseCols + c] += part[idx];
        }
      }

      if (acc.size > 32) {
        for (const [kk, t] of acc) {
          if (kk !== k) {
            const s = sizes.get(kk);
            if (s) writeTileToFile(t, ...kk.split(',').map(Number), s.rows, s.cols, s.baseRows, s.baseCols);
            acc.delete(kk);
          }
        }
      }
    },
    finalize(){
      for (const [k, tile] of acc) {
        const [ibS, jbS] = k.split(',').map(Number);
        const s = sizes.get(k);
        if (s) writeTileToFile(tile, ibS, jbS, s.rows, s.cols, s.baseRows, s.baseCols);
      }
      fs.closeSync(fdC);
   try {
     const alias = path.join(taskDir, 'output.bin');
     try { fs.unlinkSync(alias); } catch {}
     try {
       fs.symlinkSync(outPath, alias);
     } catch {
       // containers or Windows without symlink perms → copy
       fs.copyFileSync(outPath, alias);
     }
   } catch {}
   return { outPath, elements: N*M };
    }
  };
}
