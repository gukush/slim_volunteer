// executors/webgpu-distributed-sort.client.js

// -------------------- Global cache (per tab/worker) --------------------
const __WGPU_SORT_CACHE__ = (globalThis.__WGPU_SORT_CACHE__ ||= {
  device: null,
  deviceId: null,
  // Per-device pipeline caches
  pipelinesByDevice: new WeakMap(), // WeakMap<GPUDevice, Map<string, CachedPipeline>>
});

/**
 * Cached pipeline record
 * @typedef {{ pipeline: GPUComputePipeline, bgl: GPUBindGroupLayout, layout: GPUPipelineLayout, module: GPUShaderModule }} CachedPipeline
 */
// ----------------------------------------------------------------------

function nextPow2(x) {
  x = x >>> 0;
  if (x <= 1) return 1;
  return 1 << (32 - Math.clz32(x - 1));
}

// Normalize any incoming data into an ArrayBuffer
function toArrayBuffer(x) {
  if (!x) throw new Error('toArrayBuffer: empty');

  // Raw ArrayBuffer
  if (x instanceof ArrayBuffer) return x;

  // TypedArray / DataView
  if (ArrayBuffer.isView(x)) {
    return x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength);
  }

  // Node-style Buffer JSON from Socket.IO or similar
  if (x && x.type === 'Buffer' && Array.isArray(x.data)) {
    return new Uint8Array(x.data).buffer;
  }

  // Objects that expose a .buffer with byteOffset/byteLength (e.g., custom views)
  if (x && x.buffer && typeof x.byteOffset === 'number' && typeof x.byteLength === 'number') {
    return x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength);
  }

  throw new Error('toArrayBuffer: unsupported payload type');
}

async function getDevice() {
  if (__WGPU_SORT_CACHE__.device) return __WGPU_SORT_CACHE__.device;

  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');

  const device = await adapter.requestDevice();
  __WGPU_SORT_CACHE__.device = device;
  __WGPU_SORT_CACHE__.deviceId = Math.random().toString(36).slice(2, 11);

  console.log('Created WebGPU device:', __WGPU_SORT_CACHE__.deviceId);

  // If the device is lost, clear our caches so we can rebuild later
  device.lost.then((info) => {
    console.warn('WebGPU device lost:', info?.message || info);
    __WGPU_SORT_CACHE__.device = null;
    __WGPU_SORT_CACHE__.deviceId = null;
    __WGPU_SORT_CACHE__.pipelinesByDevice = new WeakMap();
  }).catch(()=>{ /* no-op */ });

  return device;
}

function getPipeline(dev, kernelCode) {
  // Get or create the per-device pipeline map
  let perDev = __WGPU_SORT_CACHE__.pipelinesByDevice.get(dev);
  if (!perDev) {
    perDev = new Map();
    __WGPU_SORT_CACHE__.pipelinesByDevice.set(dev, perDev);
  }

  const key = 'bitonic:' + (kernelCode?.length || 0);
  /** @type {CachedPipeline|undefined} */
  let cached = perDev.get(key);
  if (cached && cached.pipeline) return cached;

  const module = dev.createShaderModule({ code: kernelCode });

  const bgl = dev.createBindGroupLayout({
    label: 'bitonic-sort-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const layout = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const pipeline = dev.createComputePipeline({
    label: 'bitonic-sort-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });

  cached = { pipeline, bgl, layout, module };
  perDev.set(key, cached);
  console.log('Creating new pipeline for device:', __WGPU_SORT_CACHE__.deviceId);
  return cached;
}

export function createExecutor({ kernels, config, inputArgs }) {
  const kernel = kernels.find((k) => k.name?.endsWith('bitonic_sort.wgsl'));
  if (!kernel) throw new Error('Bitonic sort kernel source missing');

  const kernelCode = kernel.content || kernel.code;
  if (!kernelCode) throw new Error('Bitonic sort WGSL code is empty');

  async function prewarm() {
    const dev = await getDevice();
    getPipeline(dev, kernelCode);
  }

  async function runChunk({ payload, meta }) {
    const tClientRecv = performance.now();

    // Accept either raw ArrayBuffer or the payload object { data, originalSize, paddedSize, ascending }
    const srcData = (payload && payload.data != null) ? payload.data : payload;
    const payloadBuf = toArrayBuffer(srcData);

    // Extract sizes and flags
    const originalSize = (payload && payload.originalSize) ?? (meta?.originalSize) ?? (payloadBuf.byteLength >>> 2);
    const paddedSize  = (payload && payload.paddedSize)  ?? (meta?.paddedSize)  ?? nextPow2(originalSize);
    const ascending   = (payload && 'ascending' in payload) ? !!payload.ascending : !!(meta?.ascending ?? true);

    const dev = await getDevice();
    const { pipeline, bgl } = getPipeline(dev, kernelCode);

    // Debug line to mirror your logs
    console.log(`Processing chunk: ${originalSize} integers (padded to ${paddedSize}) with device ${__WGPU_SORT_CACHE__.deviceId}`);

    // --- Create per-run resources ---
    // Storage buffer sized to the provided (already padded) data
    const dataSize = payloadBuf.byteLength;
    const dataBuf = dev.createBuffer({
      label: 'bitonic-sort-data',
      size: dataSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Upload the entire (padded) payload
    dev.queue.writeBuffer(dataBuf, 0, new Uint8Array(payloadBuf));

    // Per-run uniform buffer (16 bytes)
    const uniformBuf = dev.createBuffer({
      label: 'bitonic-sort-uniforms',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Bind group (rebuilt per run due to unique data buffer)
    const bindGroup = dev.createBindGroup({
      label: 'bitonic-sort-bindgroup',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: dataBuf } },
        { binding: 1, resource: { buffer: uniformBuf } },
      ],
    });

    // --- Encode passes ---
    const encoder = dev.createCommandEncoder({ label: 'sort-encoder' });
    const pass = encoder.beginComputePass({ label: 'sort-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    const workgroupSize = 256;
    const numGroups = Math.ceil(paddedSize / workgroupSize);

    // Bitonic sort staging: k is the global stage, j is the sub-stage
    for (let k = 2; k <= paddedSize; k <<= 1) {
      for (let j = k >> 1; j > 0; j >>= 1) {
        const params = new Uint32Array([paddedSize, k, j, ascending ? 1 : 0]);
        dev.queue.writeBuffer(uniformBuf, 0, params);
        pass.dispatchWorkgroups(numGroups);
      }
    }

    pass.end();

    // --- Readback only the original (unpadded) portion ---
    const readBytes = originalSize * 4;
    const readBuf = dev.createBuffer({
      label: 'read-buffer',
      size: readBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(dataBuf, 0, readBuf, 0, readBytes);
    dev.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    // Cleanup transient buffers (optional)
    try { dataBuf.destroy?.(); } catch {}
    try { uniformBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}

    const tClientDone = performance.now();

    return {
      status: 'ok',
      result,
      timings: { tClientRecv, tClientDone },
    };
  }

  return { runChunk, prewarm };
}
