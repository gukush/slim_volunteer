// executors/webgpu-ecm-stage1.client.js - Version 3 with Resume Support + Timestamp Queries

// -------------------- Global cache (per tab/worker) --------------------
const __WGPU_ECM_CACHE__ = (globalThis.__WGPU_ECM_CACHE__ ||= {
  device: null,
  deviceId: null,
  // Per-device pipeline caches
  pipelinesByDevice: new WeakMap(), // WeakMap<GPUDevice, Map<string, CachedPipeline>>
  // Timing infrastructure
  timingByDevice: new WeakMap(), // WeakMap<GPUDevice, TimingContext>
});

/**
 * Cached pipeline record
 * @typedef {{ pipeline: GPUComputePipeline, bindGroupLayout: GPUBindGroupLayout, pipelineLayout: GPUPipelineLayout, module: GPUShaderModule }} CachedPipeline
 */

/**
 * Timing context for GPU measurements
 * @typedef {{
 *   querySet: GPUQuerySet,
 *   resolveBuffer: GPUBuffer,
 *   resultBuffer: GPUBuffer,
 *   capacity: number,
 *   supportsTimestamps: boolean
 * }} TimingContext
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

function createTimingContext(device, capacity = 64) {
  try {
    const querySet = device.createQuerySet({
      label: 'ecm-timing-queries',
      type: 'timestamp',
      count: capacity,
    });

    const resolveBuffer = device.createBuffer({
      label: 'ecm-timing-resolve',
      size: capacity * 8, // 8 bytes per timestamp
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    const resultBuffer = device.createBuffer({
      label: 'ecm-timing-result',
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    console.log('[ECM] Created timing context with', capacity, 'query slots');

    return {
      querySet,
      resolveBuffer,
      resultBuffer,
      capacity,
      supportsTimestamps: true,
    };
  } catch (error) {
    console.warn('[ECM] Timestamp queries not supported:', error.message);
    return {
      querySet: null,
      resolveBuffer: null,
      resultBuffer: null,
      capacity: 0,
      supportsTimestamps: false,
    };
  }
}

function getTimingContext(device) {
  let timing = __WGPU_ECM_CACHE__.timingByDevice.get(device);
  if (!timing) {
    timing = createTimingContext(device);
    __WGPU_ECM_CACHE__.timingByDevice.set(device, timing);
  }
  return timing;
}

async function measureTimestamps(device, timingCtx, queryStart, queryEnd) {
  if (!timingCtx.supportsTimestamps) return null;

  try {
    // Resolve timestamps to buffer
    const encoder = device.createCommandEncoder({ label: 'timing-resolve' });
    encoder.resolveQuerySet(
      timingCtx.querySet,
      queryStart,
      queryEnd - queryStart + 1,
      timingCtx.resolveBuffer,
      queryStart * 8
    );
    encoder.copyBufferToBuffer(
      timingCtx.resolveBuffer,
      queryStart * 8,
      timingCtx.resultBuffer,
      queryStart * 8,
      (queryEnd - queryStart + 1) * 8
    );
    device.queue.submit([encoder.finish()]);

    // Read back results
    await timingCtx.resultBuffer.mapAsync(
      GPUMapMode.READ,
      queryStart * 8,
      (queryEnd - queryStart + 1) * 8
    );

    const arrayBuffer = timingCtx.resultBuffer.getMappedRange(
      queryStart * 8,
      (queryEnd - queryStart + 1) * 8
    );
    const timestamps = new BigUint64Array(arrayBuffer);
    const startTime = timestamps[0];
    const endTime = timestamps[1];

    timingCtx.resultBuffer.unmap();

    // Convert to milliseconds (timestamps are in nanoseconds)
    const elapsedNs = endTime - startTime;
    const elapsedMs = Number(elapsedNs) / 1_000_000;

    return elapsedMs;
  } catch (error) {
    ERR('Failed to measure GPU timestamps:', error);
    return null;
  }
}

export async function getDevice() {
  DBG('getDevice()');
  if (__WGPU_ECM_CACHE__.device) return __WGPU_ECM_CACHE__.device;

  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');

  // Request device with timestamp-query feature
  const requiredFeatures = [];
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
    console.log('[ECM] Requesting timestamp-query feature');
  } else {
    console.warn('[ECM] timestamp-query feature not available');
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
  });

  __WGPU_ECM_CACHE__.device = device;
  __WGPU_ECM_CACHE__.deviceId = Math.random().toString(36).slice(2, 11);

  console.log('Created WebGPU device for ECM:', __WGPU_ECM_CACHE__.deviceId);

  // Initialize timing context
  getTimingContext(device);

  // If the device is lost, clear our caches so we can rebuild later
  device.lost.then((info) => {
    console.warn('WebGPU device lost (ECM):', info?.message || info);
    __WGPU_ECM_CACHE__.device = null;
    __WGPU_ECM_CACHE__.deviceId = null;
    __WGPU_ECM_CACHE__.pipelinesByDevice = new WeakMap();
    __WGPU_ECM_CACHE__.timingByDevice = new WeakMap();
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
    const timingCtx = getTimingContext(dev);

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
    const TARGET_MS = 500;  // Target time per GPU submit
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

    // Timing accumulation
    let totalGpuTime = 0;
    let passCount = 0;
    let queryIndex = 0;

    // Main resume loop
    while (pp_start < pp_count) {
      // Update header with current window
      u32[10] = pp_start; // pp_start field
      u32[11] = Math.min(pp_len, pp_count - pp_start); // pp_len field

      // Write updated header to GPU
      dev.queue.writeBuffer(ioBuffer, 0, u32.buffer, 0, 48); // Just update header (12 words * 4 bytes)

      // Dispatch compute shader with timing
      const WG_SIZE = 64;
      const numWorkgroups = Math.ceil(n / WG_SIZE);

      const encoder = dev.createCommandEncoder({ label: `ecm-encoder-v3-pass-${pp_start}` });

      // Set up timestamp queries if available
      const timestampWrites = timingCtx.supportsTimestamps && queryIndex + 1 < timingCtx.capacity
        ? {
            querySet: timingCtx.querySet,
            beginningOfPassWriteIndex: queryIndex,
            endOfPassWriteIndex: queryIndex + 1,
          }
        : undefined;

      const pass = encoder.beginComputePass({
        label: `ecm-pass-v3-${pp_start}`,
        timestampWrites
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(numWorkgroups, 1, 1);
      pass.end();

      // Time this submit (CPU time)
      const t0 = performance.now();
      dev.queue.submit([encoder.finish()]);
      await dev.queue.onSubmittedWorkDone().catch(e => {
        ERR('GPU submit error:', e);
        throw e;
      });
      const cpuTime = performance.now() - t0;

      // Measure GPU time if available
      let gpuTime = null;
      if (timestampWrites) {
        gpuTime = await measureTimestamps(dev, timingCtx, queryIndex, queryIndex + 1);
        if (gpuTime !== null) {
          totalGpuTime += gpuTime;
        }
        queryIndex = (queryIndex + 2) % timingCtx.capacity; // Advance query index
      }

      const processed = u32[11];
      const timingInfo = gpuTime !== null
        ? `CPU: ${cpuTime.toFixed(1)}ms, GPU: ${gpuTime.toFixed(1)}ms`
        : `CPU: ${cpuTime.toFixed(1)}ms`;

      console.log(`Pass ${passCount}: pp[${pp_start}:${pp_start + processed}] - ${timingInfo}`);

      // Update position
      pp_start += processed;
      passCount++;

      // Adaptive tuning to stay near target time (use CPU time for now)
      const dt = cpuTime;
      if (dt < TARGET_MS / 2 && pp_len < pp_count / 10) {
        pp_len = Math.min(pp_len * 2, 10000);
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
    const avgGpuTime = passCount > 0 && totalGpuTime > 0 ? (totalGpuTime / passCount).toFixed(1) : 'N/A';
    console.log(`ECM Stage 1 complete: ${pp_count} prime powers in ${totalTime}s (avg GPU: ${avgGpuTime}ms/pass)`);

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
    DBG('runChunk(): finished', { checksumOut, totalTime, totalGpuTime: totalGpuTime.toFixed(1) + 'ms' });

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
      timings: {
        tClientRecv,
        tClientDone,
        cpuTimeMs: (tClientDone - tClientRecv) - totalGpuTime, // CPU time is total minus GPU time
        gpuTimeMs: totalGpuTime,
        totalGpuTimeMs: totalGpuTime,
        avgGpuTimeMs: passCount > 0 ? totalGpuTime / passCount : 0,
        passCount
      }
    };
  }

  return { runChunk, prewarm };
}