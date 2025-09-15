// executors/webgpu-distributed-sort.client.js

// -------------------- Global cache (per tab/worker) --------------------
const __WGPU_SORT_CACHE__ = (globalThis.__WGPU_SORT_CACHE__ ||= {
  device: null,
  deviceId: null,
  // Per-device pipeline caches
  pipelinesByDevice: new WeakMap(), // WeakMap<GPUDevice, Map<string, CachedPipeline>>
  // Timing infrastructure
  timingByDevice: new WeakMap(), // WeakMap<GPUDevice, TimingContext>
});

/**
 * Cached pipeline record
 * @typedef {{ pipeline: GPUComputePipeline, bgl: GPUBindGroupLayout, layout: GPUPipelineLayout, module: GPUShaderModule }} CachedPipeline
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

// ----------------------------------------------------------------------

// Flip to true while debugging; or pass ?validate=1 in the URL; or set config.validateChunks=true
const DBG_VALIDATE_DEFAULT = true;
function isValidateEnabled(config){
  const urlFlag = (typeof location !== 'undefined')
    ? new URLSearchParams(location.search).get('validate')
    : null;
  const urlEnabled = urlFlag != null && urlFlag !== '0' && urlFlag !== 'false';
  return Boolean(DBG_VALIDATE_DEFAULT || urlEnabled || config?.validateChunks);
}

function validateChunkOrder({ buffer, length, ascending, label = '' }){
  // buffer: ArrayBuffer with exactly "length" elements (u32)
  const v = new Uint32Array(buffer, 0, length);
  console.log(JSON.stringify(v));
  for (let i = 1; i < v.length; i++) {
    const bad = ascending ? (v[i - 1] > v[i]) : (v[i - 1] < v[i]);
    if (bad) {
      const a = v[i-1], b = v[i];
      const ctxStart = Math.max(0, i - 5);
      const ctxEnd   = Math.min(v.length, i + 5);
      const ctx = Array.from(v.slice(ctxStart, ctxEnd));
      console.error(
        `[CHUNK VERIFY FAIL] ${label} at index ${i-1} => ${i}:`,
        `${a} ${ascending?'>':'<'} ${b} (should be ${ascending?'<=':'>='})`
      );
      console.error(`Context [${ctxStart}:${ctxEnd}]:`, ctx);
      throw new Error('chunk-verify-failed');
    }
  }
  // console.debug(`[CHUNK VERIFY OK] ${label} length=${v.length}, ascending=${ascending}`);
}

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

function createTimingContext(device, capacity = 64) {
  try {
    const querySet = device.createQuerySet({
      label: 'sort-timing-queries',
      type: 'timestamp',
      count: capacity,
    });

    // Ensure buffer size is large enough for 256-byte aligned offsets
    // We need space for the maximum possible aligned offset plus data
    const maxDataSize = capacity * 8;
    const maxAlignedOffset = Math.ceil(maxDataSize / 256) * 256;
    const alignedSize = maxAlignedOffset + maxDataSize;

    const resolveBuffer = device.createBuffer({
      label: 'sort-timing-resolve',
      size: alignedSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    const resultBuffer = device.createBuffer({
      label: 'sort-timing-result',
      size: alignedSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    console.log('[Sort] Created timing context with', capacity, 'query slots');

    return {
      querySet,
      resolveBuffer,
      resultBuffer,
      capacity,
      supportsTimestamps: true,
    };
  } catch (error) {
    console.warn('[Sort] Timestamp queries not supported:', error.message);
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
  let timing = __WGPU_SORT_CACHE__.timingByDevice.get(device);
  if (!timing) {
    timing = createTimingContext(device);
    __WGPU_SORT_CACHE__.timingByDevice.set(device, timing);
  }
  return timing;
}

async function measureStageTime(device, timingCtx, queryStart, queryEnd) {
  if (!timingCtx.supportsTimestamps) return null;

  try {
    // Calculate aligned offsets for WebGPU requirements (256-byte alignment)
    const dataOffset = queryStart * 8;
    const alignedOffset = Math.ceil(dataOffset / 256) * 256;
    const dataSize = (queryEnd - queryStart + 1) * 8;

    // Resolve timestamps to buffer
    const encoder = device.createCommandEncoder({ label: 'sort-timing-resolve' });
    encoder.resolveQuerySet(
      timingCtx.querySet,
      queryStart,
      queryEnd - queryStart + 1,
      timingCtx.resolveBuffer,
      alignedOffset
    );
    encoder.copyBufferToBuffer(
      timingCtx.resolveBuffer,
      alignedOffset,
      timingCtx.resultBuffer,
      alignedOffset,
      dataSize
    );
    device.queue.submit([encoder.finish()]);

    // Read back results
    await timingCtx.resultBuffer.mapAsync(
      GPUMapMode.READ,
      alignedOffset,
      dataSize
    );

    const arrayBuffer = timingCtx.resultBuffer.getMappedRange(
      alignedOffset,
      dataSize
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
    console.error('[Sort] Failed to measure GPU timestamps:', error);
    return null;
  }
}

async function getDevice() {
  if (__WGPU_SORT_CACHE__.device) return __WGPU_SORT_CACHE__.device;

  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');

  // Request device with timestamp-query feature
  const requiredFeatures = [];
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
    console.log('[Sort] Requesting timestamp-query feature');
  } else {
    console.warn('[Sort] timestamp-query feature not available');
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
  });

  __WGPU_SORT_CACHE__.device = device;
  __WGPU_SORT_CACHE__.deviceId = Math.random().toString(36).slice(2, 11);

  console.log('Created WebGPU device for Sort:', __WGPU_SORT_CACHE__.deviceId);

  // Initialize timing context
  getTimingContext(device);

  // If the device is lost, clear our caches so we can rebuild later
  device.lost.then((info) => {
    console.warn('WebGPU device lost (Sort):', info?.message || info);
    __WGPU_SORT_CACHE__.device = null;
    __WGPU_SORT_CACHE__.deviceId = null;
    __WGPU_SORT_CACHE__.pipelinesByDevice = new WeakMap();
    __WGPU_SORT_CACHE__.timingByDevice = new WeakMap();
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

  const module = dev.createShaderModule({
    label: 'bitonic-sort-module',
    code: kernelCode
  });

  const bgl = dev.createBindGroupLayout({
    label: 'bitonic-sort-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const layout = dev.createPipelineLayout({
    label: 'bitonic-sort-pipeline-layout',
    bindGroupLayouts: [bgl]
  });

  const pipeline = dev.createComputePipeline({
    label: 'bitonic-sort-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });

  cached = { pipeline, bgl, layout, module };
  perDev.set(key, cached);
  console.log('Creating new sort pipeline for device:', __WGPU_SORT_CACHE__.deviceId);
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

    // Accept either raw ArrayBuffer or the payload object
    const srcData = (payload && payload.data != null) ? payload.data : payload;
    const payloadBuf = toArrayBuffer(srcData);

    // Extract sizes and flags
    const originalSize = (payload && payload.originalSize) ?? (meta?.originalSize) ?? (payloadBuf.byteLength >>> 2);
    const paddedSize = (payload && payload.paddedSize) ?? (meta?.paddedSize) ?? nextPow2(originalSize);
    const ascending = (payload && 'ascending' in payload) ? !!payload.ascending : !!(meta?.ascending ?? true);

    const dev = await getDevice();
    const { pipeline, bgl } = getPipeline(dev, kernelCode);
    const timingCtx = getTimingContext(dev);

    console.log(`Processing chunk: ${originalSize} integers (padded to ${paddedSize}) with device ${__WGPU_SORT_CACHE__.deviceId}`);

    // --- Prepare properly padded data ---
    let paddedData;
    const srcArray = new Uint32Array(payloadBuf);

    if (originalSize < paddedSize) {
      // Need to pad with sentinel values
      paddedData = new Uint32Array(paddedSize);
      // Copy original data
      paddedData.set(srcArray.subarray(0, originalSize));
      // Fill padding with appropriate sentinel values
      // For ascending sort: use MAX_UINT32 so padded values stay at the end
      // For descending sort: use 0 so padded values stay at the end
      const sentinel = ascending ? 0xFFFFFFFF : 0;
      for (let i = originalSize; i < paddedSize; i++) {
        paddedData[i] = sentinel;
      }
    } else {
      // Data is already the right size
      paddedData = srcArray;
    }

    // --- Create GPU resources ---
    const dataSize = paddedSize * 4; // Size in bytes
    const dataBuf = dev.createBuffer({
      label: 'bitonic-sort-data',
      size: dataSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Upload padded data
    dev.queue.writeBuffer(dataBuf, 0, paddedData);

    // Uniform buffer for stage parameters
    const uniformBuf = dev.createBuffer({
      label: 'bitonic-sort-uniforms',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group
    const bindGroup = dev.createBindGroup({
      label: 'bitonic-sort-bindgroup',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: dataBuf } },
        { binding: 1, resource: { buffer: uniformBuf } },
      ],
    });

    // --- Execute bitonic sort with proper synchronization and timing ---
    const workgroupSize = 256;
    const numGroups = Math.ceil(paddedSize / workgroupSize);

    let totalGpuTime = 0;
    let stageCount = 0;
    let queryIndex = 0;

    // CRITICAL: Execute each stage separately and wait for completion
    for (let k = 2; k <= paddedSize; k <<= 1) {
      for (let j = k >> 1; j > 0; j >>= 1) {
        // Update uniforms for this specific stage
        const params = new Uint32Array([paddedSize, k, j, ascending ? 1 : 0]);
        dev.queue.writeBuffer(uniformBuf, 0, params);

        // Create a new command encoder for each stage
        // This ensures each stage is submitted separately
        const encoder = dev.createCommandEncoder({
          label: `sort-stage-${k}-step-${j}`
        });

        // Set up timestamp queries if available
        const timestampWrites = timingCtx.supportsTimestamps && queryIndex + 1 < timingCtx.capacity
          ? {
              querySet: timingCtx.querySet,
              beginningOfPassWriteIndex: queryIndex,
              endOfPassWriteIndex: queryIndex + 1,
            }
          : undefined;

        const pass = encoder.beginComputePass({
          label: `sort-pass-${k}-${j}`,
          timestampWrites
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(numGroups);
        pass.end();

        // Submit this stage immediately
        dev.queue.submit([encoder.finish()]);

        // CRITICAL: Wait for this stage to complete before proceeding
        // This ensures sequential execution of stages
        await dev.queue.onSubmittedWorkDone();

        // Measure GPU time for this stage if available
        if (timestampWrites) {
          const stageTime = await measureStageTime(dev, timingCtx, queryIndex, queryIndex + 1);
          if (stageTime !== null) {
            totalGpuTime += stageTime;
          }
          queryIndex = (queryIndex + 2) % timingCtx.capacity; // Advance query index
        }

        stageCount++;
      }
    }

    // --- Readback only the original (unpadded) portion ---
    const readBytes = originalSize * 4;
    const readBuf = dev.createBuffer({
      label: 'read-buffer',
      size: readBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create final encoder for readback
    const readbackEncoder = dev.createCommandEncoder({ label: 'readback-encoder' });
    readbackEncoder.copyBufferToBuffer(dataBuf, 0, readBuf, 0, readBytes);
    dev.queue.submit([readbackEncoder.finish()]);

    // Map and read results
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    // Cleanup
    try { dataBuf.destroy?.(); } catch {}
    try { uniformBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}

    const tClientDone = performance.now();
    const totalTime = tClientDone - tClientRecv;

    // Log timing results
    const avgGpuTime = stageCount > 0 && totalGpuTime > 0 ? (totalGpuTime / stageCount).toFixed(2) : 'N/A';
    console.log(`[Sort] Completed ${originalSize} integers in ${totalTime.toFixed(1)}ms (total GPU: ${totalGpuTime.toFixed(1)}ms, avg: ${avgGpuTime}ms/stage)`);

    // Validate if enabled
    if (isValidateEnabled(config)) {
      validateChunkOrder({
        buffer: result,
        length: originalSize,
        ascending,
        label: `task:${meta?.taskId || ''} chunk:${meta?.chunkIndex ?? ''}`
      });
    }

    return {
      status: 'ok',
      result,
      timings: {
        tClientRecv,
        tClientDone,
        cpuTimeMs: totalTime - totalGpuTime, // CPU time is total minus GPU time
        gpuTimeMs: totalGpuTime,
        totalTimeMs: totalTime,
        avgGpuTimeMs: stageCount > 0 ? totalGpuTime / stageCount : 0,
        stageCount
      },
    };
  }

  return { runChunk, prewarm };
}