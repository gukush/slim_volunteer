// executors/webgpu-ecm-stage1.client.js

// -------------------- Global cache (per tab/worker) --------------------
const __WGPU_ECM_CACHE__ = (globalThis.__WGPU_ECM_CACHE__ ||= {
  device: null,
  deviceId: null,
  // Per-device pipeline caches
  pipelinesByDevice: new WeakMap(), // WeakMap<GPUDevice, Map<string, CachedPipeline>>
});

/**
 * Cached pipeline record
 * @typedef {{ pipeline: GPUComputePipeline, bindGroupLayout: GPUBindGroupLayout, pipelineLayout: GPUPipelineLayout, module: GPUShaderModule }} CachedPipeline
 */

// Lightweight debug helpers
const DBG = (...args) => console.log('[ecm-client]', ...args);
const ERR = (...args) => console.error('[ecm-client]', ...args);
const u32sum = (u32) => {
  let x = 0 >>> 0;
  for (let i = 0; i < u32.length; i++) x ^= (u32[i] >>> 0);
  return ('00000000' + x.toString(16)).slice(-8);
};
// ----------------------------------------------------------------------

export async function getDevice() {
  DBG('getDevice()');
  if (__WGPU_ECM_CACHE__.device) return __WGPU_ECM_CACHE__.device;

  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');

  const device = await adapter.requestDevice();
  __WGPU_ECM_CACHE__.device = device;
  __WGPU_ECM_CACHE__.deviceId = Math.random().toString(36).slice(2, 11);

  console.log('Created WebGPU device for ECM:', __WGPU_ECM_CACHE__.deviceId);

  // If the device is lost, clear our caches so we can rebuild later
  device.lost.then((info) => {
    console.warn('WebGPU device lost (ECM):', info?.message || info);
    __WGPU_ECM_CACHE__.device = null;
    __WGPU_ECM_CACHE__.deviceId = null;
    __WGPU_ECM_CACHE__.pipelinesByDevice = new WeakMap();
  }).catch(()=>{ /* no-op */ });

  return device;
}

export function getPipeline(dev, kernelCode) {
  // Get or create the per-device pipeline map
  let perDev = __WGPU_ECM_CACHE__.pipelinesByDevice.get(dev);
  if (!perDev) {
    perDev = new Map();
    __WGPU_ECM_CACHE__.pipelinesByDevice.set(dev, perDev);
  }

  // Key by code length; OK because you said path/name stays the same but contents changed
  const key = 'ecm-stage1-v2:' + (kernelCode?.length || 0);
  /** @type {CachedPipeline|undefined} */
  let cached = perDev.get(key);
  if (cached && cached.pipeline) return cached;

  const module = dev.createShaderModule({
    label: 'ecm-stage1-module-v2',
    code: kernelCode
  });

  // Updated bind group layout for version 2 (single storage buffer):
  //  @binding(0) IO buffer (storage) - contains header + constants + primes + outputs
  const bindGroupLayout = dev.createBindGroupLayout({
    label: 'ecm-stage1-v2-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]
  });

  const pipelineLayout = dev.createPipelineLayout({
    label: 'ecm-stage1-v2-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout]
  });

  const pipeline = dev.createComputePipeline({
    label: 'ecm-stage1-v2-pipeline',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' }
  });

  cached = { pipeline, bindGroupLayout, pipelineLayout, module };
  perDev.set(key, cached);
  console.log('Creating new ECM v2 pipeline for device:', __WGPU_ECM_CACHE__.deviceId);
  return cached;
}

export function createExecutor({ kernels, config }){
  // Keep the path/name EXACTLY like your example:
  const kernel = kernels.find(k => k.name.endsWith('ecm_stage1_webgpu_compute.wgsl'));
  if(!kernel) throw new Error('ECM kernel missing');

  const kernelCode = kernel.content || kernel.code;
  if (!kernelCode) throw new Error('ECM WGSL code is empty');

  async function prewarm() {
    const dev = await getDevice();
    getPipeline(dev, kernelCode);
  }

  async function runChunk({ payload, meta }){
    const tClientRecv = Date.now();

    const dev = await getDevice();
    const { pipeline, bindGroupLayout } = getPipeline(dev, kernelCode);

    const { data, dims } = payload;
    const { n, pp_count, total_words } = dims;

    console.log(`Processing ECM chunk v2: ${n} curves, ${pp_count} prime powers with device ${__WGPU_ECM_CACHE__.deviceId}`);

    // Parse the incoming buffer structure (Version 2)
    const u32 = new Uint32Array(data);
    const version = u32[1] >>> 0;

    DBG(`Buffer version: ${version}, expected: 2`);
    if (version !== 2) {
      ERR(`Unexpected buffer version: ${version}, expected 2`);
    }

    const HEADER_WORDS_V2 = 12;
    const CONST_WORDS = 8*3 + 4; // N(8) + R2(8) + mont_one(8) + n0inv32(1) + pad(3)
    const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + pad(3)

    // Version 2 layout: no curve input data
    const calcTotalWords = HEADER_WORDS_V2 + CONST_WORDS + pp_count + n * CURVE_OUT_WORDS_PER;
    if (typeof total_words === 'number' && total_words !== calcTotalWords) {
      ERR('Layout mismatch v2: total_words vs calculated', { total_words, calcTotalWords, n, pp_count });
    }

    const outputOffset = HEADER_WORDS_V2 + CONST_WORDS + pp_count;
    DBG('v2 layout', { HEADER_WORDS_V2, CONST_WORDS, CURVE_OUT_WORDS_PER, outputOffset, total_words });

    const checksumIn = u32sum(u32);

    // --- Create WebGPU buffer (single storage buffer approach) ---

    // Create a single storage buffer that contains the entire data
    const ioBuffer = dev.createBuffer({
      label: 'ecm-io-buffer-v2',
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    // Upload the entire buffer
    dev.queue.writeBuffer(ioBuffer, 0, data);

    // Create bind group with single storage buffer
    const bindGroup = dev.createBindGroup({
      label: 'ecm-bindgroup-v2',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: ioBuffer } },
      ]
    });

    // Dispatch compute shader — 1 curve per thread
    const WG_SIZE = 64; // Updated to match WGSL @workgroup_size(64)
    const numWorkgroups = Math.ceil(n / WG_SIZE);

    const encoder = dev.createCommandEncoder({ label: 'ecm-encoder-v2' });
    const pass = encoder.beginComputePass({ label: 'ecm-pass-v2' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(numWorkgroups, 1, 1);
    pass.end();

    // Copy output back to readable buffer
    const readBuffer = dev.createBuffer({
      label: 'ecm-read-buffer-v2',
      size: data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    encoder.copyBufferToBuffer(ioBuffer, 0, readBuffer, 0, data.byteLength);

    // Submit and wait
    dev.queue.submit([encoder.finish()]);
    await dev.queue.onSubmittedWorkDone().catch(e=>ERR('onSubmittedWorkDone error', e));
    DBG('submitted & GPU done');

    await readBuffer.mapAsync(GPUMapMode.READ);
    const outputData = readBuffer.getMappedRange();

    // The output data already contains the full buffer with updated results
    const fullResult = new Uint32Array(outputData);

    console.log('Output section starts at word', outputOffset);
    for(let i = 0; i < Math.min(5, n); i++) {
      const curveOffset = outputOffset + i * CURVE_OUT_WORDS_PER;
      const status = fullResult[curveOffset + 8];
      console.log(`Curve ${i}: status=${status}, result limbs:`,
        Array.from(fullResult.slice(curveOffset, curveOffset + 8)));
    }

    const checksumOut = u32sum(fullResult);
    DBG('runChunk(): finished', { checksumIn, checksumOut, total_words, outputOffset });

    const result = fullResult.buffer.slice(0);
    readBuffer.unmap();

    // Cleanup buffers
    try { ioBuffer.destroy?.(); } catch {}
    try { readBuffer.destroy?.(); } catch {}

    const tClientDone = Date.now();
    return {
      status: 'ok',
      result,
      timings: { tClientRecv, tClientDone }
    };
  }

  return { runChunk, prewarm };
}