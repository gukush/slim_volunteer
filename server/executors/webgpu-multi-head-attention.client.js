// executors/webgpu-multi-head-attention.client.js

// -------------------- Global cache (per tab/worker) --------------------
const __WGPU_MHA_CACHE__ = (globalThis.__WGPU_MHA_CACHE__ ||= {
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
      label: 'mha-timing-queries',
      type: 'timestamp',
      count: capacity,
    });

    // Ensure buffer size is large enough for 256-byte aligned offsets
    // We need space for the maximum possible aligned offset plus data
    const maxDataSize = capacity * 8;
    const maxAlignedOffset = Math.ceil(maxDataSize / 256) * 256;
    const alignedSize = maxAlignedOffset + maxDataSize;

    const resolveBuffer = device.createBuffer({
      label: 'mha-timing-resolve',
      size: alignedSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    const resultBuffer = device.createBuffer({
      label: 'mha-timing-result',
      size: alignedSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    console.log('[MHA] Created timing context with', capacity, 'query slots');

    return {
      querySet,
      resolveBuffer,
      resultBuffer,
      capacity,
      supportsTimestamps: true,
    };
  } catch (error) {
    console.warn('[MHA] Timestamp queries not supported:', error.message);
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
    // Calculate aligned offsets for WebGPU requirements (256-byte alignment)
    const dataOffset = queryStart * 8;
    const alignedOffset = Math.ceil(dataOffset / 256) * 256;
    const dataSize = (queryEnd - queryStart + 1) * 8;

    // Resolve timestamps to buffer
    const encoder = device.createCommandEncoder({ label: 'mha-timing-resolve' });
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
    console.error('[MHA] Failed to measure GPU timestamps:', error);
    return null;
  }
}

export function createExecutor({ kernels, config }) {
  const kernel = kernels.find(k => k.name.endsWith('multi_head_attention.wgsl'));
  if (!kernel) throw new Error('Multi-head attention kernel missing');

  // --------- minimal logger (respects ?log=level) ---------
  const qp = (() => {
    try { return new URL(location.href).searchParams; } catch { return new URLSearchParams(); }
  })();
  const levelName = (qp.get('log') || 'info').toLowerCase();
  const levels = { error: 0, warn: 1, info: 2, debug: 3, trace: 4 };
  const L = levels[levelName] ?? 2;

  const log = {
    error: (...a) => console.error(...a),
    warn:  (...a) => { if (L >= 1) console.warn(...a); },
    info:  (...a) => { if (L >= 2) console.log(...a); },
    debug: (...a) => { if (L >= 3) console.debug(...a); },
    trace: (...a) => { if (L >= 4) console.debug(...a); }
  };

  async function getDevice() {
    if (__WGPU_MHA_CACHE__.device) return __WGPU_MHA_CACHE__.device;
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');

    // Request device with timestamp-query feature
    const requiredFeatures = [];
    if (adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query');
      console.log('[MHA] Requesting timestamp-query feature');
    } else {
      console.warn('[MHA] timestamp-query feature not available');
    }

    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: 1024 * 1024 * 1024,
        maxBufferSize: 1024 * 1024 * 1024,
        maxComputeWorkgroupStorageSize: 16384,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupSizeZ: 64
      }
    });

    __WGPU_MHA_CACHE__.device = device;
    __WGPU_MHA_CACHE__.deviceId = Math.random().toString(36).slice(2, 11);

    // Initialize timing context
    __WGPU_MHA_CACHE__.timingContext = createTimingContext(device);

    console.log('[MHA] Created WebGPU device:', __WGPU_MHA_CACHE__.deviceId);

    // If the device is lost, clear our caches
    device.lost.then((info) => {
      console.warn('[MHA] WebGPU device lost:', info?.message || info);
      __WGPU_MHA_CACHE__.device = null;
      __WGPU_MHA_CACHE__.deviceId = null;
      __WGPU_MHA_CACHE__.pipeline = null;
      __WGPU_MHA_CACHE__.layout = null;
      __WGPU_MHA_CACHE__.timingContext = null;
    }).catch(()=>{ /* no-op */ });

    return device;
  }

  async function ensurePipeline() {
    if (__WGPU_MHA_CACHE__.pipeline) return { pipeline: __WGPU_MHA_CACHE__.pipeline, bindGroupLayout: __WGPU_MHA_CACHE__.layout };

    const dev = await getDevice();
    const module = dev.createShaderModule({
      code: kernel.content,
      label: 'mha-compute-shader'
    });

    __WGPU_MHA_CACHE__.layout = dev.createBindGroupLayout({
      label: 'mha-bind-group-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // Q
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // K
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // V
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // Output
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }           // Dims
      ]
    });

    const pipelineLayout = dev.createPipelineLayout({
      label: 'mha-pipeline-layout',
      bindGroupLayouts: [__WGPU_MHA_CACHE__.layout]
    });

    __WGPU_MHA_CACHE__.pipeline = dev.createComputePipeline({
      label: 'mha-compute-pipeline',
      layout: pipelineLayout,
      compute: { module, entryPoint: 'main' }
    });

    console.log('[MHA] Created compute pipeline for device:', __WGPU_MHA_CACHE__.deviceId);
    return { pipeline: __WGPU_MHA_CACHE__.pipeline, bindGroupLayout: __WGPU_MHA_CACHE__.layout };
  }

  async function runChunk({ payload, meta }) {
    const tClientRecv = Date.now();

    // --------- concise high-signal logs ---------
    log.info(`MHA[head=${meta?.head ?? '?'}] start`);
    log.debug('meta:', meta);

    try {
      // Validate payload
      const { q, k, v, dims } = payload || {};
      if (!q || !k || !v || !dims) {
        log.error('Invalid payload structure:', { hasQ: !!q, hasK: !!k, hasV: !!v, hasDims: !!dims });
        throw new Error('Invalid payload: missing q, k, v, or dims');
      }

      const { seq_len, d_k, d_v } = dims;
      const expectedSize = seq_len * d_k * 4;
      if (q.byteLength !== expectedSize || k.byteLength !== expectedSize || v.byteLength !== expectedSize) {
        log.error('Buffer size mismatch', {
          expected: expectedSize, qActual: q.byteLength, kActual: k.byteLength, vActual: v.byteLength
        });
        throw new Error('Buffer size mismatch');
      }

      const { pipeline: pipe, bindGroupLayout: layout } = await ensurePipeline();
      const dev = __WGPU_MHA_CACHE__.device;
      const timingCtx = __WGPU_MHA_CACHE__.timingContext;

      // Create input buffers (quiet; keep details at debug)
      log.debug('alloc Q/K/V/output/uniform buffers');
      const qBuffer = dev.createBuffer({ label: 'Q-buffer', size: q.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const kBuffer = dev.createBuffer({ label: 'K-buffer', size: k.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const vBuffer = dev.createBuffer({ label: 'V-buffer', size: v.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

      dev.queue.writeBuffer(qBuffer, 0, q);
      dev.queue.writeBuffer(kBuffer, 0, k);
      dev.queue.writeBuffer(vBuffer, 0, v);

      const outputSize = seq_len * d_v * 4;
      const outputBuffer = dev.createBuffer({ label: 'output-buffer', size: outputSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

      const dimsData = new Uint32Array([seq_len, d_k, d_v, 0]);
      const dimsBuffer = dev.createBuffer({ label: 'dims-buffer', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(dimsBuffer, 0, dimsData);

      const bindGroup = dev.createBindGroup({
        label: 'mha-bind-group',
        layout,
        entries: [
          { binding: 0, resource: { buffer: qBuffer } },
          { binding: 1, resource: { buffer: kBuffer } },
          { binding: 2, resource: { buffer: vBuffer } },
          { binding: 3, resource: { buffer: outputBuffer } },
          { binding: 4, resource: { buffer: dimsBuffer } }
        ]
      });

      const encoder = dev.createCommandEncoder({ label: 'mha-encoder' });

      // Set up timestamp queries if available
      const timestampWrites = timingCtx.supportsTimestamps
        ? {
            querySet: timingCtx.querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
          }
        : undefined;

      const tDispatchStart = performance.now();

      const computePass = encoder.beginComputePass({
        label: 'mha-compute-pass',
        timestampWrites
      });
      computePass.setPipeline(pipe);
      computePass.setBindGroup(0, bindGroup);

      const workgroupSizeX = 16;
      const workgroupSizeY = 16;
      const workgroupsX = Math.ceil(seq_len / workgroupSizeX);
      const workgroupsY = Math.ceil(d_v / workgroupSizeY);

      // keep dispatch line at info (single most important progress marker)
      log.info(`MHA[head=${meta?.head ?? '?'}] submit wg=(${workgroupsX},${workgroupsY},1)`);

      computePass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
      computePass.end();

      const readBuffer = dev.createBuffer({ label: 'read-buffer', size: outputSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize);
      dev.queue.submit([encoder.finish()]);

      await dev.queue.onSubmittedWorkDone();

      const tDispatchEnd = performance.now();
      const cpuTime = tDispatchEnd - tDispatchStart;

      // Measure GPU time if available
      let gpuTime = null;
      if (timestampWrites) {
        gpuTime = await measureGpuTime(dev, timingCtx, 0, 1);
      }

      await readBuffer.mapAsync(GPUMapMode.READ);
      const result = readBuffer.getMappedRange().slice(0);
      readBuffer.unmap();

      // Cleanup (debug-level)
      try {
        qBuffer.destroy(); kBuffer.destroy(); vBuffer.destroy();
        outputBuffer.destroy(); dimsBuffer.destroy(); readBuffer.destroy();
      } catch (e) { log.debug('buffer destroy error:', e?.message); }

      const tClientDone = Date.now();
      const totalTime = tClientDone - tClientRecv;

      // Log timing results
      const timingInfo = gpuTime !== null
        ? `CPU: ${cpuTime.toFixed(1)}ms, GPU: ${gpuTime.toFixed(1)}ms`
        : `CPU: ${cpuTime.toFixed(1)}ms`;
      log.info(`MHA[head=${meta?.head ?? '?'}] done ${totalTime}ms - ${timingInfo}`);

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

    } catch (e) {
      const tClientDone = Date.now();
      const totalTime = tClientDone - tClientRecv;
      log.error(`MHA[head=${meta?.head ?? '?'}] FAILED after ${totalTime}ms:`, e?.message);
      return {
        status: 'error',
        error: e?.message || String(e),
        timings: {
          tClientRecv,
          tClientDone,
          cpuTimeMs: totalTime,
          gpuTimeMs: null
        }
      };
    }
  }

  // Optional: a prewarm so client.js can await it (already supported there).
  async function prewarm() {
    await ensurePipeline();
  }

  return { runChunk, prewarm };
}
