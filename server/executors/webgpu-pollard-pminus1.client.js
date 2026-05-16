const __WGPU_PM1_CACHE__ = (globalThis.__WGPU_PM1_CACHE__ ||= {
  device: null,
  pipelinesByDevice: new WeakMap(),
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
  if (__WGPU_PM1_CACHE__.device) return __WGPU_PM1_CACHE__.device;
  if (__WGPU_PM1_CACHE__.lostReason) {
    throw new Error(`WebGPU device was previously lost: ${__WGPU_PM1_CACHE__.lostReason}. This usually means the kernel exceeded the browser's GPU timeout (TDR/watchdog).`);
  }
  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    const reason = info?.reason || 'unknown';
    __WGPU_PM1_CACHE__.lostReason = reason;
    __WGPU_PM1_CACHE__.device = null;
    __WGPU_PM1_CACHE__.pipelinesByDevice = new WeakMap();
    console.error(`[WEBGPU WATCHDOG] Device lost! Reason: "${reason}". This almost certainly means a shader kernel exceeded the browser's ~5-10s GPU execution timeout and was killed by the TDR watchdog.`);
  }).catch(() => {});
  __WGPU_PM1_CACHE__.device = device;
  return device;
}

function getPipeline(device, kernelCode) {
  let perDevice = __WGPU_PM1_CACHE__.pipelinesByDevice.get(device);
  if (!perDevice) {
    perDevice = new Map();
    __WGPU_PM1_CACHE__.pipelinesByDevice.set(device, perDevice);
  }
  const key = `${kernelCode.length}:${kernelCode.slice(0, 64)}:${kernelCode.slice(-64)}`;
  let cached = perDevice.get(key);
  if (cached) return cached;

  const module = device.createShaderModule({ label: 'pollard-pminus1-module', code: kernelCode });
  const bgl = device.createBindGroupLayout({
    label: 'pollard-pminus1-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ label: 'pollard-pminus1-pipeline-layout', bindGroupLayouts: [bgl] });
  const pipeline = device.createComputePipeline({
    label: 'pollard-pminus1-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });
  cached = { pipeline, bgl };
  perDevice.set(key, cached);
  return cached;
}

export function createExecutor({ kernels }) {
  const kernel = kernels.find((k) => k.name?.endsWith('pollard_pminus1_batched.wgsl'));
  if (!kernel) throw new Error('Pollard p-1 batched WGSL source missing');
  const kernelCode = kernel.content || kernel.code;

  async function prewarm() {
    const device = await getDevice();
    getPipeline(device, kernelCode);
  }

  async function runChunk({ payload }) {
    const tClientRecv = performance.now();
    const device = await getDevice();
    const { pipeline, bgl } = getPipeline(device, kernelCode);
    const input = new Uint32Array(toArrayBuffer(payload.data));
    const nBases = Number(payload.nBases || input[3] || 0) >>> 0;
    const numNs = Number(input[5] || 1) >>> 0;
    const totalThreads = numNs * nBases;
    if (nBases === 0) throw new Error('Pollard p-1 chunk has no bases');
    if (totalThreads === 0) throw new Error('Pollard p-1 chunk has no work');

    const ioBuf = device.createBuffer({
      label: 'pollard-pminus1-io',
      size: Math.max(4, input.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(ioBuf, 0, input);

    const readBuf = device.createBuffer({
      label: 'pollard-pminus1-readback',
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroup = device.createBindGroup({
      label: 'pollard-pminus1-bindgroup',
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: ioBuf } }],
    });

    const encoder = device.createCommandEncoder({ label: 'pollard-pminus1-encoder' });
    const pass = encoder.beginComputePass({ label: 'pollard-pminus1-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalThreads / 64));
    pass.end();
    encoder.copyBufferToBuffer(ioBuf, 0, readBuf, 0, input.byteLength);
    device.queue.submit([encoder.finish()]);

    const KERNEL_TIMEOUT_MS = 120000;
    try {
      const mapPromise = readBuf.mapAsync(GPUMapMode.READ);
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error(`Pollard p-1 kernel timed out after ${KERNEL_TIMEOUT_MS}ms for chunk with ${numNs} numbers`)), KERNEL_TIMEOUT_MS);
      });
      await Promise.race([mapPromise, timeoutPromise]);
    } catch (e) {
      if (__WGPU_PM1_CACHE__.lostReason) {
        throw new Error(`Pollard p-1 chunk failed because WebGPU device was lost (reason: "${__WGPU_PM1_CACHE__.lostReason}"). This is the browser watchdog/TDR killing the kernel for exceeding the GPU time limit.`);
      }
      throw e;
    }
    const result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    try { ioBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}

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
