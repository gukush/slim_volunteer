// executors/webgpu-ecm-stage1.client.js - Version 3 with Resume Support

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

  // Key by code length (or hash if you prefer)
  const key = 'ecm-stage1-v3:' + (kernelCode?.length || 0);
  /** @type {CachedPipeline|undefined} */
  let cached = perDev.get(key);
  if (cached && cached.pipeline) return cached;

  const module = dev.createShaderModule({
    label: 'ecm-stage1-module-v3',
    code: kernelCode
  });

  // Bind group layout for version 3 (single storage buffer)
  const bindGroupLayout = dev.createBindGroupLayout({
    label: 'ecm-stage1-v3-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]
  });

  const pipelineLayout = dev.createPipelineLayout({
    label: 'ecm-stage1-v3-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout]
  });

  const pipeline = dev.createComputePipeline({
    label: 'ecm-stage1-v3-pipeline',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' }
  });

  cached = { pipeline, bindGroupLayout, pipelineLayout, module };
  perDev.set(key, cached);
  console.log('Creating new ECM v3 pipeline for device:', __WGPU_ECM_CACHE__.deviceId);
  return cached;
}

export function createExecutor({ kernels, config }){
  // Find the kernel - should be the v3 kernel now
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

    console.log(`Processing ECM chunk v3: ${n} curves, ${pp_count} prime powers with device ${__WGPU_ECM_CACHE__.deviceId}`);

    // Parse the incoming buffer structure (Version 3 expected)
    let u32 = new Uint32Array(data);
    const version = u32[1] >>> 0;

    if (version < 3) {
      console.warn(`Buffer version ${version}, expected 3 - upgrading header`);
      // Upgrade to version 3 by adding pp_start and pp_len fields
      const newU32 = new Uint32Array(u32.length);
      newU32.set(u32);
      newU32[1] = 3; // Update version
      newU32[10] = 0; // pp_start = 0
      newU32[11] = 0; // pp_len will be set in loop
      u32 = newU32;
    }

    const HEADER_WORDS_V3 = 12;
    const CONST_WORDS = 8*3 + 4; // N(8) + R2(8) + mont_one(8) + n0inv32(1) + pad(3)
    const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + pad(3)
    const STATE_WORDS_PER_CURVE = 8 + 8 + 8 + 2; // X(8) + Z(8) + A24(8) + (sigma, curve_ok)

    const outputOffset = HEADER_WORDS_V3 + CONST_WORDS + pp_count;
    const stateOffset = outputOffset + n * CURVE_OUT_WORDS_PER;

    // Calculate total buffer size with state storage
    const totalBufferWords = stateOffset + n * STATE_WORDS_PER_CURVE;
    const bufferSize = totalBufferWords * 4;

    DBG('v3 layout', {
      HEADER_WORDS_V3,
      CONST_WORDS,
      CURVE_OUT_WORDS_PER,
      STATE_WORDS_PER_CURVE,
      outputOffset,
      stateOffset,
      bufferSize
    });

    // --- RESUMABLE COMPUTATION LOOP ---

    // Tuning parameters
    const TARGET_MS = 50;  // Target time per GPU submit (well below watchdog)
    let pp_len = Math.min(1500, pp_count); // Start conservative
    let pp_start = 0;

    // Create GPU buffer with space for state
    const ioBuffer = dev.createBuffer({
      label: 'ecm-io-buffer-v3',
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    // Extend u32 array if needed for state storage
    if (u32.length < totalBufferWords) {
      const extendedU32 = new Uint32Array(totalBufferWords);
      extendedU32.set(u32);
      u32 = extendedU32;
    }

    // Initial upload
    dev.queue.writeBuffer(ioBuffer, 0, u32.buffer, 0, bufferSize);

    // Create bind group once (reuse across iterations)
    const bindGroup = dev.createBindGroup({
      label: 'ecm-bindgroup-v3',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: ioBuffer } },
      ]
    });

    console.log(`Starting ECM Stage 1 with B1 ≈ ${pp_count} prime powers`);
    const overallStart = performance.now();

    // Main resume loop
    while (pp_start < pp_count) {
      // Update header with current window
      u32[10] = pp_start; // pp_start field
      u32[11] = Math.min(pp_len, pp_count - pp_start); // pp_len field

      // Write updated header to GPU
      dev.queue.writeBuffer(ioBuffer, 0, u32.buffer, 0, 48); // Just update header (12 words * 4 bytes)

      // Dispatch compute shader
      const WG_SIZE = 64;
      const numWorkgroups = Math.ceil(n / WG_SIZE);

      const encoder = dev.createCommandEncoder({ label: `ecm-encoder-v3-pass-${pp_start}` });
      const pass = encoder.beginComputePass({ label: `ecm-pass-v3-${pp_start}` });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(numWorkgroups, 1, 1);
      pass.end();

      // Time this submit
      const t0 = performance.now();
      dev.queue.submit([encoder.finish()]);
      await dev.queue.onSubmittedWorkDone().catch(e => {
        ERR('GPU submit error:', e);
        throw e;
      });
      const dt = performance.now() - t0;

      const processed = u32[11];
      console.log(`Pass: pp[${pp_start}:${pp_start + processed}] completed in ${dt.toFixed(1)}ms`);

      // Update position
      pp_start += processed;

      // Adaptive tuning to stay near target time
      if (dt < TARGET_MS / 2 && pp_len < pp_count / 10) {
        pp_len = Math.min(pp_len * 2, 8000);
        DBG(`Increasing pp_len to ${pp_len}`);
      } else if (dt > TARGET_MS * 1.5) {
        pp_len = Math.max(Math.floor(pp_len * TARGET_MS / dt), 100);
        DBG(`Decreasing pp_len to ${pp_len}`);
      }

      // Progress reporting
      if (pp_start % 10000 === 0 || pp_start === pp_count) {
        const pct = (100 * pp_start / pp_count).toFixed(1);
        const elapsed = ((performance.now() - overallStart) / 1000).toFixed(1);
        console.log(`ECM Progress: ${pp_start}/${pp_count} (${pct}%) - ${elapsed}s elapsed`);
      }
    }

    const totalTime = ((performance.now() - overallStart) / 1000).toFixed(2);
    console.log(`✅ ECM Stage 1 complete: ${pp_count} prime powers in ${totalTime}s`);

    // Final readback
    const readBuffer = dev.createBuffer({
      label: 'ecm-read-buffer-v3',
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const finalEncoder = dev.createCommandEncoder({ label: 'ecm-final-encoder-v3' });
    finalEncoder.copyBufferToBuffer(ioBuffer, 0, readBuffer, 0, bufferSize);
    dev.queue.submit([finalEncoder.finish()]);
    await dev.queue.onSubmittedWorkDone();

    await readBuffer.mapAsync(GPUMapMode.READ);
    const outputData = readBuffer.getMappedRange();
    const fullResult = new Uint32Array(outputData);

    // Log first few results for debugging
    console.log('First 3 curve results:');
    for(let i = 0; i < Math.min(3, n); i++) {
      const curveOffset = outputOffset + i * CURVE_OUT_WORDS_PER;
      const status = fullResult[curveOffset + 8];
      const resultLimbs = Array.from(fullResult.slice(curveOffset, curveOffset + 8));

      // Convert result to hex for easier reading (if non-zero)
      let resultHex = '0x';
      for(let j = 7; j >= 0; j--) {
        resultHex += resultLimbs[j].toString(16).padStart(8, '0');
      }
      if (resultHex === '0x0000000000000000000000000000000000000000000000000000000000000000') {
        resultHex = '0x0';
      }

      console.log(`  Curve ${i}: status=${status}, result=${resultHex.slice(0, 20)}...`);
    }

    const checksumOut = u32sum(fullResult);
    DBG('runChunk(): finished', { checksumOut, totalTime });

    // Return the complete buffer (header + constants + primes + results)
    const resultSize = outputOffset + n * CURVE_OUT_WORDS_PER;
    const result = fullResult.slice(0, resultSize).buffer.slice(0);

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