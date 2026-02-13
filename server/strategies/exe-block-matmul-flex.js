// server/strategies/native-block-matmul-flex.js
// Native binary route for block matmul (no browser executor)

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';
import { framework } from './block-matmul.js';

export const id = 'exe-block-matmul-flex';
export const name = 'Block Matmul (native binary, chunked, streaming)';

export function getClientExecutorInfo(config){
  const backend = (config?.backend || 'opencl').toLowerCase();
  if (!['opencl','cuda','vulkan'].includes(backend)) {
    throw new Error(`Unsupported native backend: ${backend}`);
  }

  return {
    framework: 'exe',
    kernels: [],
    schema: {
      order: ['UNIFORMS','INPUTS','OUTPUTS'],
      uniforms: [ { name: 'rows', type: 'i32' }, { name: 'K', type: 'i32' }, { name: 'cols', type: 'i32' } ],
      inputs:  [ { name: 'A', type: 'f32' }, { name: 'B', type: 'f32' } ],
      outputs: [ { name: 'C', type: 'f32' } ]
    },
    artifacts: getArtifacts(config)
  };
}

export function getArtifacts(config){
	const backend = (config?.backend || 'opencl').toLowerCase();
	const artifacts = [];

	const frameworkBinaries = {
		opencl: config.openclBinary || config.binary || '/app/binaries/ocl_block_matmul_chunked',
		cuda: config.cudaBinary || '/app/binaries/exe_block_matmul',
		vulkan: config.vulkanBinary || '/app/scripts/native/vk_block_matmul'
	};

	const binaryPath = frameworkBinaries[backend];
	if (!binaryPath) {
		return [];
	}

	try {
		const abs = path.isAbsolute(binaryPath) ? binaryPath : binaryPath;
		const bytes = fs.readFileSync(abs).toString('base64');
		const artifactName = config.program || path.basename(binaryPath);

		artifacts.push({
			type: 'binary',
			name: artifactName,
			program: artifactName,
			backend,
			bytes,
			exec: true
		});
	} catch (error) {
		console.error(`[DEBUG] getArtifacts - Failed to read binary ${binaryPath}:`, error.message);
		throw new Error(`Binary not found for ${backend}: ${binaryPath}`);
	}

	return artifacts;
}

function toF32(x){
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  if (ArrayBuffer.isView(x)) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength/4));
  if (typeof Buffer !== 'undefined' && x instanceof Buffer) return new Float32Array(x.buffer, x.byteOffset, Math.floor(x.byteLength/4));
  if (x && x.type === 'Buffer' && Array.isArray(x.data)) return new Float32Array(Uint8Array.from(x.data).buffer);
  throw new Error('Unsupported buffer type for Float32 view');
}

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

async function readWindowAsync(fd, rowStart, rowCount, colStart, colCount, rowLen, elementSize=4){
  const out = Buffer.alloc(rowCount * colCount * elementSize);
  const rowBytes = colCount * elementSize;
  for (let r = 0; r < rowCount; r++){
    const srcOff = ((rowStart + r) * rowLen + colStart) * elementSize;
    const dstOff = r * rowBytes;
    await new Promise((resolve, reject) => {
      fs.read(fd, out, dstOff, rowBytes, srcOff, (err, bytesRead) => {
        if (err) reject(err); else resolve(bytesRead);
      });
    });
  }
  return out;
}

function pickTileParams({ N, M, K, C }){
  const kTile = Math.min(K, 256);
  const perElem = 4;
  const budgetBytes = Math.max(1, Number(C) || 8*1024*1024);
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

  const withSize = files.map(f => ({ ...f, size: f.size ?? (f.path ? fs.statSync(f.path).size : 0) }));
  const targetA = BigInt(N) * BigInt(K) * 4n;
  const targetB = BigInt(K) * BigInt(M) * 4n;
  withSize.sort((x,y)=>Number((BigInt(x.size)-targetA)**2n - (BigInt(y.size)-targetA)**2n));
  Af = withSize[0];
  withSize.sort((x,y)=>Number((BigInt(x.size)-targetB)**2n - (BigInt(y.size)-targetB)**2n));
  Bf = withSize[0];
  return [Af.path, Bf.path];
}

function getDataTypeInfo(datatype) {
  const type = (datatype || 'f32').toLowerCase();
  switch (type) {
    case 'f32':
    case 'int32':
      return { elementSize: 4, isPacked: false, packFactor: 1 };
    case 'f16':
      return { elementSize: 2, isPacked: false, packFactor: 1 };
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

  if (typeInfo.elementSize === 1) {
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

export function buildChunker({ taskId, taskDir, K, config, inputFiles }){
  const { N, K: KK, M } = config;
  const datatype = config.datatype || 'f32';
  const typeInfo = getDataTypeInfo(datatype);

  const [Afile, Bfile] = pickInputs(inputFiles, N, KK, M);
  if (!Afile || !Bfile) throw new Error('Need A.bin and B.bin');

  const backend = (config?.backend || 'opencl').toLowerCase();
  const defaultBins = {
    opencl: 'scripts/native/ocl_block_matmul_chunked',
    cuda: 'exe_block_matmul',
    vulkan: 'scripts/native/vk_block_matmul'
  };
  const rel = config.binary || defaultBins[backend];
  const binaryName = config.program || path.basename(rel);

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

          const outputBytes = rNow * cNow * 4;

          for (let kb = 0; kb < KK; kb += kSpan){
            const kNow = Math.min(kSpan, KK - kb);

            const [Ablock, Bblock] = await Promise.all([
              readWindowAsync(fdA, ib*baseRows, rNow, kb, kNow, KK, typeInfo.elementSize),
              readWindowAsync(fdB, kb, kNow, jb*baseCols, cNow, M, typeInfo.elementSize)
            ]);

            const aData = packData(Ablock.buffer, datatype);
            const bData = packData(Bblock.buffer, datatype);

            let uniforms;
            if (datatype === 'int8') {
              const groupsK = Math.ceil(kNow / 4);
              uniforms = new Int32Array([rNow, kNow, cNow, groupsK]);
            } else {
              uniforms = new Int32Array([rNow, kNow, cNow, 0]);
            }
            const uniformsBytes = new Uint8Array(uniforms.buffer);

            const payload = {
              action: 'exec',
              framework: 'exe',
              buffers: [
                uniformsBytes.buffer.slice(uniformsBytes.byteOffset, uniformsBytes.byteOffset + uniformsBytes.byteLength),
                aData,
                bData
              ],
              outputs: [ { byteLength: outputBytes } ],
              datatype: datatype,
              isPacked: typeInfo.isPacked,
              packFactor: typeInfo.packFactor,
            };

            const meta = {
              ib, jb, kb,
              rows: rNow, cols: cNow, kNow,
              baseRows, baseCols, kSpan,
              outputSizes: [outputBytes],
              uniforms: [rNow, kNow, cNow],
              backend: config.backend || 'opencl',
              program: binaryName,
              dispatch: {
                opencl: { global: [cNow, rNow, 1], local: [16,16,1] },
                cuda:   { grid:   [Math.ceil(cNow/16), Math.ceil(rNow/16), 1], block: [16,16,1] },
                vulkan: { groups: [Math.ceil(cNow/16), Math.ceil(rNow/16), 1] }
              }
            };

            yield { id: uuidv4(), payload, meta, tCreate: Date.now() };

            chunkCount++;
            if (chunkCount % 10 === 0) await new Promise(resolve => setImmediate(resolve));
            if (chunkCount % 100 === 0) logger.info(`Generated ${chunkCount}/${totalChunks} chunks`);
          }
        }
      }
      fs.closeSync(fdA);
      fs.closeSync(fdB);
      logger.info(`${id} chunker done - generated ${chunkCount} chunks (focus)`);
    }
  };
}

export function buildAssembler({ taskId, taskDir, K, config }) {
  const { N, M } = config;
  const outPath = path.join(taskDir, 'output.bin');

  const fdC = fs.openSync(outPath, 'w+');
  fs.ftruncateSync(fdC, Number(N) * Number(M) * 4);

  const acc = new Map();
  const progressedK = new Map();
  const sizes = new Map();

  const key = (ib, jb) => `${ib},${jb}`;

  const toF32 = (buf) => {
    if (buf instanceof Float32Array) return buf;
    if (Buffer.isBuffer(buf)) return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
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
    const { ib, jb, kSpan, rows, cols, baseRows, baseCols } = meta;
    const k = key(ib, jb);
    if (!sizes.has(k)) sizes.set(k, { rows, cols, baseRows, baseCols });

    const part = toF32(result);
    let tile = acc.get(k);
    if (!tile) {
      tile = new Float32Array(rows * cols);
      acc.set(k, tile);
      progressedK.set(k, 0);
    }

    for (let i = 0; i < part.length; i++) tile[i] += part[i];

    const soFar = (progressedK.get(k) || 0) + Number(kSpan || 0);
    progressedK.set(k, soFar);

    const fullK = Number(config.K ?? K ?? 0);
    if (fullK > 0 && soFar >= fullK) {
      const s = sizes.get(k);
      writeTileToFile(tile, ib, jb, s.rows, s.cols, s.baseRows, s.baseCols);
      acc.delete(k);
      progressedK.delete(k);
      sizes.delete(k);
    }
  }

  function finalize() {
    for (const [k, tile] of acc) {
      const [ibS, jbS] = k.split(',').map(Number);
      const s = sizes.get(k);
      if (s) writeTileToFile(tile, ibS, jbS, s.rows, s.cols, s.baseRows, s.baseCols);
    }
    fs.closeSync(fdC);
    return { outPath, elements: Number(N) * Number(M) };
  }

  return { integrate: ({ chunkId, result, meta }) => onChunkResult({ chunkId, result, meta }), finalize };
}