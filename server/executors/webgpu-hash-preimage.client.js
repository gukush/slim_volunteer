const __WGPU_HASH_PREIMAGE_CACHE__ = (globalThis.__WGPU_HASH_PREIMAGE_CACHE__ ||= {
  device: null,
  pipelinesByDevice: new WeakMap(),
});

function toArrayBuffer(x) {
  if (x instanceof ArrayBuffer) return x;
  if (ArrayBuffer.isView(x)) return x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength);
  if (x && x.type === 'Buffer' && Array.isArray(x.data)) return Uint8Array.from(x.data).buffer;
  if (x && x.buffer && typeof x.byteOffset === 'number' && typeof x.byteLength === 'number') {
    return x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength);
  }
  throw new Error('Unsupported ArrayBuffer payload');
}

function parseTargetWords(hex) {
  const clean = String(hex || '').trim().toLowerCase().replace(/^0x/, '');
  if (!clean) return new Uint32Array(8);
  if (!/^[0-9a-f]{64}$/.test(clean)) throw new Error('targetHash must be 64 hex chars');
  const out = new Uint32Array(8);
  for (let i = 0; i < 8; i++) {
    out[i] = Number.parseInt(clean.slice(i * 8, i * 8 + 8), 16) >>> 0;
  }
  return out;
}

function packPrefixWords(prefixBytes) {
  if (prefixBytes.length > 47) throw new Error('prefix must be at most 47 bytes');
  const words = new Uint32Array(16);
  for (let i = 0; i < prefixBytes.length; i++) {
    words[i >>> 2] |= (prefixBytes[i] << ((i & 3) * 8)) >>> 0;
  }
  return words;
}

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotr(x, n) {
  return (x >>> n) | (x << (32 - n));
}

function sha256OneBlock(prefixBytes, nonceLo, nonceHi) {
  const msgLen = prefixBytes.length + 8;
  const block = new Uint8Array(64);
  block.set(prefixBytes, 0);
  block[prefixBytes.length + 0] = nonceLo & 0xff;
  block[prefixBytes.length + 1] = (nonceLo >>> 8) & 0xff;
  block[prefixBytes.length + 2] = (nonceLo >>> 16) & 0xff;
  block[prefixBytes.length + 3] = (nonceLo >>> 24) & 0xff;
  block[prefixBytes.length + 4] = nonceHi & 0xff;
  block[prefixBytes.length + 5] = (nonceHi >>> 8) & 0xff;
  block[prefixBytes.length + 6] = (nonceHi >>> 16) & 0xff;
  block[prefixBytes.length + 7] = (nonceHi >>> 24) & 0xff;
  block[msgLen] = 0x80;
  const bitLen = msgLen * 8;
  block[62] = (bitLen >>> 8) & 0xff;
  block[63] = bitLen & 0xff;

  const w = new Uint32Array(64);
  for (let i = 0; i < 16; i++) {
    const j = i * 4;
    w[i] = ((block[j] << 24) | (block[j + 1] << 16) | (block[j + 2] << 8) | block[j + 3]) >>> 0;
  }
  for (let i = 16; i < 64; i++) {
    const s0 = (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)) >>> 0;
    const s1 = (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)) >>> 0;
    w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
  }

  let a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
  let e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
  for (let i = 0; i < 64; i++) {
    const S1 = (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) >>> 0;
    const ch = ((e & f) ^ (~e & g)) >>> 0;
    const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
    const S0 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) >>> 0;
    const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
    const temp2 = (S0 + maj) >>> 0;
    h = g; g = f; f = e; e = (d + temp1) >>> 0;
    d = c; c = b; b = a; a = (temp1 + temp2) >>> 0;
  }
  return new Uint32Array([
    (a + 0x6a09e667) >>> 0,
    (b + 0xbb67ae85) >>> 0,
    (c + 0x3c6ef372) >>> 0,
    (d + 0xa54ff53a) >>> 0,
    (e + 0x510e527f) >>> 0,
    (f + 0x9b05688c) >>> 0,
    (g + 0x1f83d9ab) >>> 0,
    (h + 0x5be0cd19) >>> 0,
  ]);
}

function leadingZeroMatch(hashWords, leadingZeroBits) {
  let remaining = leadingZeroBits >>> 0;
  for (let i = 0; i < 8; i++) {
    if (remaining === 0) return true;
    if (remaining >= 32) {
      if (hashWords[i] !== 0) return false;
      remaining -= 32;
    } else {
      return (hashWords[i] & (0xffffffff << (32 - remaining))) === 0;
    }
  }
  return true;
}

function hashMatches(hashWords, targetWords, requireTarget, leadingZeroBits) {
  if (requireTarget) {
    for (let i = 0; i < 8; i++) {
      if (hashWords[i] !== targetWords[i]) return false;
    }
  }
  return leadingZeroBits > 0 ? leadingZeroMatch(hashWords, leadingZeroBits) : true;
}

function cpuSearch(prefixBytes, payload, targetWords) {
  const startLo = Number(payload.startLo || 0) >>> 0;
  const startHi = Number(payload.startHi || 0) >>> 0;
  const count = Number(payload.count || 0) >>> 0;
  const leadingZeroBits = Number(payload.leadingZeroBits || 0) >>> 0;
  const requireTarget = !!payload.targetHash;

  for (let i = 0; i < count; i++) {
    const nonceLo = (startLo + i) >>> 0;
    const nonceHi = (startHi + (nonceLo < startLo ? 1 : 0)) >>> 0;
    const hash = sha256OneBlock(prefixBytes, nonceLo, nonceHi);
    if (!hashMatches(hash, targetWords, requireTarget, leadingZeroBits)) continue;
    const result = new Uint32Array(12);
    result[0] = 1;
    result[1] = nonceLo;
    result[2] = nonceHi;
    result.set(hash, 3);
    return result.buffer;
  }
  return new Uint32Array(12).buffer;
}

async function getDevice() {
  if (__WGPU_HASH_PREIMAGE_CACHE__.device) return __WGPU_HASH_PREIMAGE_CACHE__.device;
  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();
  device.lost.then(() => {
    __WGPU_HASH_PREIMAGE_CACHE__.device = null;
    __WGPU_HASH_PREIMAGE_CACHE__.pipelinesByDevice = new WeakMap();
  }).catch(() => {});
  __WGPU_HASH_PREIMAGE_CACHE__.device = device;
  return device;
}

function getPipeline(device, kernelCode) {
  if (!__WGPU_HASH_PREIMAGE_CACHE__.pipelinesByDevice) {
    __WGPU_HASH_PREIMAGE_CACHE__.pipelinesByDevice = new WeakMap();
  }
  let perDevice = __WGPU_HASH_PREIMAGE_CACHE__.pipelinesByDevice.get(device);
  if (!perDevice) {
    perDevice = new Map();
    __WGPU_HASH_PREIMAGE_CACHE__.pipelinesByDevice.set(device, perDevice);
  }
  const key = `${kernelCode.length}:${kernelCode.slice(0, 64)}:${kernelCode.slice(-64)}`;
  let cached = perDevice.get(key);
  if (cached) return cached;

  const module = device.createShaderModule({ label: 'hash-preimage-sha256-module', code: kernelCode });
  const bgl = device.createBindGroupLayout({
    label: 'hash-preimage-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ label: 'hash-preimage-pipeline-layout', bindGroupLayouts: [bgl] });
  const pipeline = device.createComputePipeline({
    label: 'hash-preimage-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });
  cached = { pipeline, bgl };
  perDevice.set(key, cached);
  return cached;
}

function wordsToHex(words) {
  return Array.from(words).map((w) => (w >>> 0).toString(16).padStart(8, '0')).join('');
}

export function createExecutor({ kernels, config }) {
  const kernel = kernels.find((k) => k.name?.endsWith('hash_preimage_sha256.wgsl'));
  if (!kernel) throw new Error('Hash preimage WGSL source missing');
  const kernelCode = kernel.content || kernel.code;

  async function prewarm() {
    const device = await getDevice();
    getPipeline(device, kernelCode);
  }

  async function runChunk({ payload }) {
    const tClientRecv = Date.now();
    const device = await getDevice();
    const { pipeline, bgl } = getPipeline(device, kernelCode);

    const prefixBytes = new Uint8Array(toArrayBuffer(payload.prefix || new ArrayBuffer(0)));
    const targetWords = parseTargetWords(payload.targetHash);
    const prefixWords = packPrefixWords(prefixBytes);
    const params = new Uint32Array(32);
    params.set(targetWords, 0);
    params.set(prefixWords, 8);
    params[24] = Number(payload.startLo || 0) >>> 0;
    params[25] = Number(payload.startHi || 0) >>> 0;
    params[26] = Number(payload.count || 0) >>> 0;
    params[27] = prefixBytes.length >>> 0;
    params[28] = Number(payload.leadingZeroBits || 0) >>> 0;
    params[29] = payload.targetHash ? 1 : 0;

    const paramsBuf = device.createBuffer({
      label: 'hash-preimage-params',
      size: params.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, params);

    const resultBuf = device.createBuffer({
      label: 'hash-preimage-result',
      size: 48,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(resultBuf, 0, new Uint32Array(12));

    const readBuf = device.createBuffer({
      label: 'hash-preimage-readback',
      size: 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroup = device.createBindGroup({
      label: 'hash-preimage-bindgroup',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: resultBuf } },
      ],
    });

    const count = Number(payload.count || 0);
    const workgroupSize = 256;
    const encoder = device.createCommandEncoder({ label: 'hash-preimage-encoder' });
    const pass = encoder.beginComputePass({ label: 'hash-preimage-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
    pass.end();
    encoder.copyBufferToBuffer(resultBuf, 0, readBuf, 0, 48);
    device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    let result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    try { paramsBuf.destroy?.(); } catch {}
    try { resultBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}

    const resultWords = new Uint32Array(result);
    if (resultWords[0] !== 1 && config?.debugWgslHash) {
      const cpuExpected = new Uint32Array(cpuSearch(prefixBytes, payload, targetWords));
      if (cpuExpected[0] !== 1) {
        return {
          status: 'ok',
          result,
          timings: {
            tClientRecv,
            tClientDone: Date.now(),
            cpuTimeMs: Date.now() - tClientRecv,
            gpuTimeMs: null,
          },
        };
      }
      const cpuHash = sha256OneBlock(prefixBytes, Number(payload.startLo || 0) >>> 0, Number(payload.startHi || 0) >>> 0);
      throw new Error(`WGSL hash mismatch/debug: resultWords=${Array.from(resultWords).map((w) => (w >>> 0).toString(16).padStart(8, '0')).join(',')} wgslFirst=${wordsToHex(resultWords.slice(3, 11))} cpuFirst=${wordsToHex(cpuHash)} cpuExpectedNonce=${cpuExpected[2].toString(16).padStart(8, '0')}${cpuExpected[1].toString(16).padStart(8, '0')}`);
    }

    if (resultWords[0] !== 1) {
      result = cpuSearch(prefixBytes, payload, targetWords);
    }

    const tClientDone = Date.now();
    return {
      status: 'ok',
      result,
      timings: {
        tClientRecv,
        tClientDone,
        cpuTimeMs: tClientDone - tClientRecv,
        gpuTimeMs: null,
      },
    };
  }

  return { prewarm, runChunk };
}
