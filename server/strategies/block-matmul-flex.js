import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'block-matmul-flex';
export const name = 'Block Matmul (switchable framework)';

export function getClientExecutorInfo(config){
  const fw = (config?.framework || 'webgpu').toLowerCase();
  if (fw === 'webgpu') {
    return { framework:'webgpu', path:'executors/webgpu-block-matmul.client.js', kernels:['kernels/block_matmul.wgsl'] };
  }
  if (fw === 'webgl2') {
    return { framework:'webgl2', path:'executors/webgl2-block-matmul.client.js', kernels:[
      'kernels/block_matrix_multiply_webgl_vertex.glsl',
      'kernels/block_matrix_multiply_webgl_fragment.glsl'
    ]};
  }
  if (fw === 'native-cuda') {
    return { framework:'native-cuda', path:'executors/native-bridge.client.js', kernels:[
      'kernels/block_matrix_multiply_cuda_kernel.cu'
    ]};
  }
  if (fw === 'native-opencl') {
    return { framework:'native-opencl', path:'executors/native-bridge.client.js', kernels:[
      'kernels/block_matrix_multiply_opencl_kernel.cl'
    ]};
  }
  if (fw === 'native-vulkan') {
    return { framework:'native-vulkan', path:'executors/native-bridge.client.js', kernels:[
      'kernels/block_matrix_multiply_vulkan_compute.glsl'
    ]};
  }
  throw new Error('Unsupported framework in config.framework: '+fw);
}

function toF32(x){
  // Browser-style ArrayBuffer
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  // Any TypedArray / DataView
  if (ArrayBuffer.isView(x)) {
    return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  }
  // Node Buffer (most common with socket.io)
  if (typeof Buffer !== 'undefined' && x instanceof Buffer) {
    return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength / 4));
  }
  // Buffer serialized as plain object (rare, but safe to handle)
  if (x && x.type === 'Buffer' && Array.isArray(x.data)) {
    const u8 = Uint8Array.from(x.data);
    return new Float32Array(u8.buffer);
  }
  throw new Error('Unsupported buffer type for Float32 view');
}

function readRowsSlice(fd, rowStart, rowCount, rowLen, elementSize){
  const out = Buffer.alloc(rowCount * rowLen * elementSize);
  for(let r=0; r<rowCount; r++){
    const fileOffset = ((rowStart + r) * rowLen) * elementSize;
    const targetOffset = r * rowLen * elementSize;
    fs.readSync(fd, out, targetOffset, rowLen * elementSize, fileOffset);
  }
  return out;
}
function readColsSlice(fd, totalCols, colStart, colCount, totalRows, elementSize){
  const out = Buffer.alloc(totalRows * colCount * elementSize);
  for(let r=0; r<totalRows; r++){
    const rowBase = (r * totalCols + colStart) * elementSize;
    const target = (r * colCount) * elementSize;
    fs.readSync(fd, out, target, colCount * elementSize, rowBase);
  }
  return out;
}

export function buildChunker({ taskId, taskDir, K, config, inputFiles }){
  const Afile = inputFiles.find(f=>/A\.bin$/.test(f.originalName))?.path || inputFiles[0]?.path;
  const Bfile = inputFiles.find(f=>/B\.bin$/.test(f.originalName))?.path || inputFiles[1]?.path;
  if(!Afile || !Bfile) throw new Error('Need A.bin and B.bin');
  const { N, K:KK, M, tileSize:TS } = config;
  const fdA = fs.openSync(Afile, 'r');
  const fdB = fs.openSync(Bfile, 'r');
  const nIB = Math.ceil(N/TS);
  const nJB = Math.ceil(M/TS);
  const EL = 4;

  return {
    async *stream(){
      for(let ib=0; ib<nIB; ib++){
        const rows = Math.min(TS, N - ib*TS);
        const Ablock = readRowsSlice(fdA, ib*TS, rows, KK, EL);
        for(let jb=0; jb<nJB; jb++){
          const cols = Math.min(TS, M - jb*TS);
          const Bblock = readColsSlice(fdB, M, jb*TS, cols, KK, EL);
          const payload = {
            a: Ablock.buffer.slice(Ablock.byteOffset, Ablock.byteOffset + Ablock.byteLength),
            b: Bblock.buffer.slice(Bblock.byteOffset, Bblock.byteOffset + Bblock.byteLength),
            dims: { rows, K: KK, cols },
          };
          const meta = { ib, jb, rows, cols, TS };
          if ((config.framework||'').startsWith('native')) {
            meta.outputSizes = [rows * cols * 4];
            meta.uniforms = [rows, KK, cols];
            if (config.framework === 'native-cuda') {
              meta.grid = [Math.ceil(cols/16), Math.ceil(rows/16), 1];
              meta.block = [16,16,1];
            }
            if (config.framework === 'native-opencl') {
              meta.global = [cols, rows, 1];
              meta.local  = [16,16,1];
            }
            if (config.framework === 'native-vulkan') {
              meta.groups = [Math.ceil(cols/16), Math.ceil(rows/16), 1];
            }
          }
          yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
        }
      }
      fs.closeSync(fdA);
      fs.closeSync(fdB);
      logger.info('block-matmul-flex chunker done');
    }
  };
}

export function buildAssembler({ taskId, taskDir, config }){
  const { N, M, tileSize } = config;
  const C = new Float32Array(N*M);
  const outPath = path.join(taskDir, 'output.bin');
  return {
    integrate({ chunkId, result, meta }){
      const { ib, jb, rows, cols } = meta;
      const tile = new toF32(result);
      for(let r=0;r<rows;r++){
        const destBase = (ib*tileSize + r) * M + jb*tileSize;
        for(let c=0;c<cols;c++){
          C[destBase + c] = tile[r*cols + c];
        }
      }
    },
    finalize(){
      fs.writeFileSync(outPath, Buffer.from(C.buffer));
      return { outPath, elements: C.length };
    }
  };
}
