const __WGPU_GOLDBACH_CACHE__ = (globalThis.__WGPU_GOLDBACH_CACHE__ ||= {
  device: null,
  pipelinesByDevice: new WeakMap(),
  smallPrimes: null,
  lostReason: null,
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

async function getDevice() {
  if (__WGPU_GOLDBACH_CACHE__.device) return __WGPU_GOLDBACH_CACHE__.device;
  if (__WGPU_GOLDBACH_CACHE__.lostReason) {
    throw new Error(`WebGPU device was previously lost: ${__WGPU_GOLDBACH_CACHE__.lostReason}. This usually means the kernel exceeded the browser's GPU timeout (TDR/watchdog).`);
  }
  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    },
  });
  device.lost.then((info) => {
    const reason = info?.reason || 'unknown';
    __WGPU_GOLDBACH_CACHE__.lostReason = reason;
    __WGPU_GOLDBACH_CACHE__.device = null;
    __WGPU_GOLDBACH_CACHE__.pipelinesByDevice = new WeakMap();
    console.error(`[WEBGPU WATCHDOG] Device lost! Reason: "${reason}". This almost certainly means a shader kernel exceeded the browser's ~5-10s GPU execution timeout and was killed by the TDR watchdog.`);
  }).catch(() => {});
  __WGPU_GOLDBACH_CACHE__.device = device;
  return device;
}

function getPipeline(device, kernelCode) {
  let perDevice = __WGPU_GOLDBACH_CACHE__.pipelinesByDevice.get(device);
  if (!perDevice) {
    perDevice = new Map();
    __WGPU_GOLDBACH_CACHE__.pipelinesByDevice.set(device, perDevice);
  }
  const key = `${kernelCode.length}:${kernelCode.slice(0, 64)}:${kernelCode.slice(-64)}`;
  let cached = perDevice.get(key);
  if (cached) return cached;

  const module = device.createShaderModule({ label: 'goldbach-verify-module', code: kernelCode });
  const bgl = device.createBindGroupLayout({
    label: 'goldbach-verify-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const layout = device.createPipelineLayout({ label: 'goldbach-verify-pipeline-layout', bindGroupLayouts: [bgl] });
  const pipeline = device.createComputePipeline({
    label: 'goldbach-verify-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });
  cached = { pipeline, bgl };
  perDevice.set(key, cached);
  return cached;
}

function getSmallPrimes() {
  if (__WGPU_GOLDBACH_CACHE__.smallPrimes) return __WGPU_GOLDBACH_CACHE__.smallPrimes;
  const limit = 65535;
  const composite = new Uint8Array(limit + 1);
  const primes = [];
  for (let n = 2; n <= limit; n++) {
    if (composite[n]) continue;
    primes.push(n);
    if (n <= Math.floor(limit / n)) {
      for (let m = n * n; m <= limit; m += n) {
        composite[m] = 1;
      }
    }
  }
  __WGPU_GOLDBACH_CACHE__.smallPrimes = Uint32Array.from(primes);
  return __WGPU_GOLDBACH_CACHE__.smallPrimes;
}

export function createExecutor({ kernels }) {
  const kernel = kernels.find((k) => k.name?.endsWith('goldbach_verify.wgsl'));
  if (!kernel) throw new Error('Goldbach verification WGSL source missing');
  const kernelCode = kernel.content || kernel.code;

  async function prewarm() {
    const device = await getDevice();
    getPipeline(device, kernelCode);
  }

  async function runChunk({ payload }) {
    const tClientRecv = performance.now();
    const device = await getDevice();
    const { pipeline, bgl } = getPipeline(device, kernelCode);

    const numbers = new Uint32Array(toArrayBuffer(payload.numbers));
    const smallPrimes = getSmallPrimes();
    const count = Number(payload.count || numbers.length);
    if (count < 0 || count > numbers.length) {
      throw new Error(`Invalid Goldbach chunk count: ${count}`);
    }
    const primeCount = smallPrimes.length;
    if (primeCount <= 0 || primeCount > smallPrimes.length) {
      throw new Error(`Invalid Goldbach prime count: ${primeCount}`);
    }

    const numbersBuf = device.createBuffer({
      label: 'goldbach-numbers',
      size: Math.max(4, numbers.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (numbers.byteLength > 0) device.queue.writeBuffer(numbersBuf, 0, numbers);

    const primesBuf = device.createBuffer({
      label: 'goldbach-small-primes',
      size: Math.max(4, smallPrimes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (smallPrimes.byteLength > 0) device.queue.writeBuffer(primesBuf, 0, smallPrimes);

    const resultWords = new Uint32Array(8);
    const resultBuf = device.createBuffer({
      label: 'goldbach-result',
      size: resultWords.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(resultBuf, 0, resultWords);

    const readBuf = device.createBuffer({
      label: 'goldbach-readback',
      size: resultWords.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const maxDim = device.limits.maxComputeWorkgroupsPerDimension;
    const workgroupSize = 256;
    const totalGroups = Math.ceil(count / workgroupSize);
    const dispatchX = Math.min(totalGroups, maxDim);
    const dispatchY = Math.ceil(totalGroups / maxDim);
    if (dispatchY > maxDim) {
      throw new Error(
        `Goldbach dispatch grid (${dispatchX}x${dispatchY}) exceeds ` +
        `maxComputeWorkgroupsPerDimension ${maxDim} in both dimensions. ` +
        `Reduce chunkSize.`
      );
    }

    const configWords = new Uint32Array([count >>> 0, primeCount >>> 0, dispatchX >>> 0, 0]);
    const configBuf = device.createBuffer({
      label: 'goldbach-config',
      size: Math.max(16, configWords.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(configBuf, 0, configWords);

    const bindGroup = device.createBindGroup({
      label: 'goldbach-bindgroup',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: numbersBuf } },
        { binding: 1, resource: { buffer: resultBuf } },
        { binding: 2, resource: { buffer: primesBuf } },
        { binding: 3, resource: { buffer: configBuf } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'goldbach-encoder' });
    const pass = encoder.beginComputePass({ label: 'goldbach-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();
    encoder.copyBufferToBuffer(resultBuf, 0, readBuf, 0, resultWords.byteLength);
    device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    try { numbersBuf.destroy?.(); } catch {}
    try { primesBuf.destroy?.(); } catch {}
    try { resultBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}
    try { configBuf.destroy?.(); } catch {}

    const tClientDone = performance.now();
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
