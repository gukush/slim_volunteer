// executors/webgpu-block-matmul.client.js

// -------------------- Global cache (per tab/worker) --------------------
const __WGPU_MATMUL_CACHE__ = (globalThis.__WGPU_MATMUL_CACHE__ ||= {
  device: null,
  deviceId: null,
  pipeline: null,
  layout: null,
  // Timing infrastructure
  timingContext: null,
});

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

function createTimingContext(device, capacity = 32) {
  try {
    const querySet = device.createQuerySet({
      label: 'matmul-timing-queries',
      type: 'timestamp',
      count: capacity,
    });

    const resolveBuffer = device.createBuffer({
      label: 'matmul-timing-resolve',
      size: capacity * 8, // 8 bytes per timestamp
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    const resultBuffer = device.createBuffer({
      label: 'matmul-timing-result',
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    console.log('[Matmul] Created timing context with', capacity, 'query slots');

    return {
      querySet,
      resolveBuffer,
      resultBuffer,
      capacity,
      supportsTimestamps: true,
    };
  } catch (error) {
    console.warn('[Matmul] Timestamp queries not supported:', error.message);
    return {
      querySet: null,
      resolveBuffer: null,
      resultBuffer: null,
      capacity: 0,
      supportsTimestamps: false,
    };
  }
}

async function measureGpuTime(device, timingCtx, queryStart, queryEnd) {
  if (!timingCtx.supportsTimestamps) return null;

  try {
    // Resolve timestamps to buffer
    const encoder = device.createCommandEncoder({ label: 'matmul-timing-resolve' });
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
    console.error('[Matmul] Failed to measure GPU timestamps:', error);
    return null;
  }
}

export function createExecutor({ kernels, config }){
  const kernel = kernels.find(k => k.name.endsWith('block_matmul.wgsl'));
  if(!kernel) throw new Error('Kernel source missing');

  async function getDevice(){
    if(__WGPU_MATMUL_CACHE__.device) return __WGPU_MATMUL_CACHE__.device;

    if(!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');

    // Request device with timestamp-query feature
    const requiredFeatures = [];
    if (adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query');
      console.log('[Matmul] Requesting timestamp-query feature');
    } else {
      console.warn('[Matmul] timestamp-query feature not available');
    }

    const device = await adapter.requestDevice({
      requiredFeatures,
    });

    __WGPU_MATMUL_CACHE__.device = device;
    __WGPU_MATMUL_CACHE__.deviceId = Math.random().toString(36).slice(2, 11);

    // Initialize timing context
    __WGPU_MATMUL_CACHE__.timingContext = createTimingContext(device);

    console.log('[Matmul] Created WebGPU device:', __WGPU_MATMUL_CACHE__.deviceId);

    // If the device is lost, clear our caches
    device.lost.then((info) => {
      console.warn('[Matmul] WebGPU device lost:', info?.message || info);
      __WGPU_MATMUL_CACHE__.device = null;
      __WGPU_MATMUL_CACHE__.deviceId = null;
      __WGPU_MATMUL_CACHE__.pipeline = null;
      __WGPU_MATMUL_CACHE__.layout = null;
      __WGPU_MATMUL_CACHE__.timingContext = null;
    }).catch(()=>{ /* no-op */ });

    return device;
  }

  async function ensurePipeline(){
    if(__WGPU_MATMUL_CACHE__.pipeline) return __WGPU_MATMUL_CACHE__.pipeline;

    const dev = await getDevice();
    const module = dev.createShaderModule({
      label: 'matmul-shader-module',
      code: kernel.content
    });

    __WGPU_MATMUL_CACHE__.layout = dev.createBindGroupLayout({
      label: 'matmul-bind-group-layout',
      entries: [
        { binding:0, visibility: GPUShaderStage.COMPUTE, buffer: { type:'read-only-storage' } },
        { binding:1, visibility: GPUShaderStage.COMPUTE, buffer: { type:'read-only-storage' } },
        { binding:2, visibility: GPUShaderStage.COMPUTE, buffer: { type:'storage' } },
        { binding:3, visibility: GPUShaderStage.COMPUTE, buffer: { type:'uniform', minBindingSize: 16 } },
      ]
    });

    const plLayout = dev.createPipelineLayout({
      label: 'matmul-pipeline-layout',
      bindGroupLayouts: [__WGPU_MATMUL_CACHE__.layout]
    });

    __WGPU_MATMUL_CACHE__.pipeline = dev.createComputePipeline({
      label: 'matmul-compute-pipeline',
      layout: plLayout,
      compute: { module, entryPoint: 'main' }
    });

    console.log('[Matmul] Created compute pipeline for device:', __WGPU_MATMUL_CACHE__.deviceId);
    return __WGPU_MATMUL_CACHE__.pipeline;
  }

  async function runChunk({ payload, meta }){
    const tClientRecv = Date.now();
    await ensurePipeline();

    const dev = __WGPU_MATMUL_CACHE__.device;
    const pipeline = __WGPU_MATMUL_CACHE__.pipeline;
    const layout = __WGPU_MATMUL_CACHE__.layout;
    const timingCtx = __WGPU_MATMUL_CACHE__.timingContext;

    const { a, b, dims } = payload;
    const { rows, K, cols } = dims;

    console.log(`[Matmul] Processing ${rows}×${K} × ${K}×${cols} with device ${__WGPU_MATMUL_CACHE__.deviceId}`);

    // Create input buffers
    const aBuf = dev.createBuffer({
      label: 'matmul-buffer-A',
      size: a.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(aBuf, 0, a);

    const bBuf = dev.createBuffer({
      label: 'matmul-buffer-B',
      size: b.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(bBuf, 0, b);

    // Create output buffer
    const outSize = rows * cols * 4;
    const cBuf = dev.createBuffer({
      label: 'matmul-buffer-C',
      size: outSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create uniform buffer for dimensions
    const dimsData = new Uint32Array([rows, K, cols, 0]); // pad to 16B
    const dimsBuf = dev.createBuffer({
      label: 'matmul-buffer-dims',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(dimsBuf, 0, dimsData);

    // Create bind group
    const bind = dev.createBindGroup({
      label: 'matmul-bind-group',
      layout,
      entries: [
        { binding:0, resource:{ buffer:aBuf } },
        { binding:1, resource:{ buffer:bBuf } },
        { binding:2, resource:{ buffer:cBuf } },
        { binding:3, resource:{ buffer:dimsBuf } },
      ]
    });

    // Dispatch computation with timing
    const wg = 16;
    const x = Math.ceil(cols / wg);
    const y = Math.ceil(rows / wg);

    const encoder = dev.createCommandEncoder({ label: 'matmul-encoder' });

    // Set up timestamp queries if available
    const timestampWrites = timingCtx.supportsTimestamps
      ? {
          querySet: timingCtx.querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        }
      : undefined;

    const tDispatchStart = performance.now();

    const pass = encoder.beginComputePass({
      label: 'matmul-compute-pass',
      timestampWrites
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(x, y, 1);
    pass.end();

    // Create readback buffer and copy
    const readBuf = dev.createBuffer({
      label: 'matmul-read-buffer',
      size: outSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    encoder.copyBufferToBuffer(cBuf, 0, readBuf, 0, outSize);

    // Submit and wait
    dev.queue.submit([encoder.finish()]);
    await dev.queue.onSubmittedWorkDone();

    const tDispatchEnd = performance.now();
    const cpuTime = tDispatchEnd - tDispatchStart;

    // Measure GPU time if available
    let gpuTime = null;
    if (timestampWrites) {
      gpuTime = await measureGpuTime(dev, timingCtx, 0, 1);
    }

    // Read results
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    // Cleanup
    try { aBuf.destroy?.(); } catch {}
    try { bBuf.destroy?.(); } catch {}
    try { cBuf.destroy?.(); } catch {}
    try { dimsBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}

    const tClientDone = Date.now();

    // Log timing results
    const timingInfo = gpuTime !== null
      ? `CPU: ${cpuTime.toFixed(1)}ms, GPU: ${gpuTime.toFixed(1)}ms`
      : `CPU: ${cpuTime.toFixed(1)}ms`;
    console.log(`[Matmul] Completed ${rows}×${cols} result - ${timingInfo}`);

    return {
      status: 'ok',
      result,
      timings: {
        tClientRecv,
        tClientDone,
        cpuTimeMs: cpuTime,
        gpuTimeMs: gpuTime
      }
    };
  }

  return { runChunk };
}