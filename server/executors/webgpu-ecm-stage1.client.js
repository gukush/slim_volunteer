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

// Cooperative v3 i64 requires shader-int64
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

  // Key by code length; OK because you said path/name stays the same but contents changed to v3_i64
  const key = 'ecm-stage1:' + (kernelCode?.length || 0);
  /** @type {CachedPipeline|undefined} */
  let cached = perDev.get(key);
  if (cached && cached.pipeline) return cached;

  const module = dev.createShaderModule({
    label: 'ecm-stage1-module',
    code: kernelCode
  });

  // Bind group layout matching **cooperative v3** WGSL:
  //  @binding(0) Params (uniform)
  //  @binding(1) Packed (RO storage)
  //  @binding(2) Curves in (RO storage)
  //  @binding(3) Curves out (storage)
  //  @binding(4) Globals (storage)  // NEW: atomic found flag
  const bindGroupLayout = dev.createBindGroupLayout({
    label: 'ecm-stage1-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      //{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]
  });

  const pipelineLayout = dev.createPipelineLayout({
    label: 'ecm-stage1-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout]
  });

  const pipeline = dev.createComputePipeline({
    label: 'ecm-stage1-pipeline',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'main' } // v3 kernel must export 'main' with @workgroup_size(128)
  });

  cached = { pipeline, bindGroupLayout, pipelineLayout, module };
  perDev.set(key, cached);
  console.log('Creating new ECM pipeline for device:', __WGPU_ECM_CACHE__.deviceId);
  return cached;
}

export function createExecutor({ kernels, config }){
  // Keep the path/name EXACTLY like your example:
  const kernel = kernels.find(k => k.name.endsWith('ecm_stage1_webgpu_compute.wgsl'));
  if(!kernel) throw new Error('ECM kernel missing');

  const kernelCode = kernel.content || kernel.code;
  if (!kernelCode) throw new Error('ECM WGSL code is empty');

  // Optional knobs (with sensible defaults for cooperative v3)
  const gcdMode   = config?.gcdMode   ?? 2;   // 0=Z only, 1=final only, 2=periodic
  const gcdPeriod = config?.gcdPeriod ?? 256; // for periodic

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

    console.log(`Processing ECM chunk: ${n} curves, ${pp_count} prime powers with device ${__WGPU_ECM_CACHE__.deviceId} and gcdMode ${gcdMode} gcdPeriod ${gcdPeriod}`);

    // Parse the incoming buffer structure
    const u32 = new Uint32Array(data);

    const HEADER_WORDS = 8;
    const CONST_WORDS = 8*3 + 4; // N(8) + R2(8) + mont_one(8) + n0inv32(1) + pad(3)
    const CURVE_IN_WORDS_PER = 8*2; // A24(8) + X1(8)
    const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + pad(3)
    const calcTotalWords = HEADER_WORDS + CONST_WORDS + pp_count + n * (CURVE_IN_WORDS_PER + CURVE_OUT_WORDS_PER);
    if (typeof total_words === 'number' && total_words !== calcTotalWords) {
      ERR('Layout mismatch: total_words vs calculated', { total_words, calcTotalWords, n, pp_count });
    }
    const prefixWords = HEADER_WORDS + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER;
    DBG('layout', { HEADER_WORDS, CONST_WORDS, CURVE_IN_WORDS_PER, CURVE_OUT_WORDS_PER, prefixWords, total_words });
    const checksumIn = u32sum(u32);


    // Extract sections from the buffer
    let offset = HEADER_WORDS; // Skip header

    // Constants section
    const constants = u32.slice(offset, offset + CONST_WORDS);
    offset += CONST_WORDS;

    // Prime powers section
    const primePowers = u32.slice(offset, offset + pp_count);
    offset += pp_count;

    // Curve inputs section
    const curveInputs = u32.slice(offset, offset + n * CURVE_IN_WORDS_PER);
    offset += n * CURVE_IN_WORDS_PER;

    // --- Create WebGPU buffers ---

    // Params buffer (uniform): pp_count, num_curves, compute_gcd, gcd_period
    const paramsData = new Uint32Array([pp_count >>> 0, n >>> 0, gcdMode >>> 0, gcdPeriod >>> 0]);
    const paramsBuffer = dev.createBuffer({
      label: 'ecm-params',
      size: 16, // 4 * u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // Packed buffer (constants + prime powers)
    const packedSize = constants.byteLength + primePowers.byteLength;
    const packedBuffer = dev.createBuffer({
      label: 'ecm-packed-data',
      size: packedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(packedBuffer, 0, constants);
    dev.queue.writeBuffer(packedBuffer, constants.byteLength, primePowers);

    // Curve inputs buffer
    const curveInputBuffer = dev.createBuffer({
      label: 'ecm-curve-inputs',
      size: curveInputs.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(curveInputBuffer, 0, curveInputs);

    // Curve outputs buffer
    const outputSize = n * CURVE_OUT_WORDS_PER * 4; // 4 bytes per u32
    const curveOutputBuffer = dev.createBuffer({
      label: 'ecm-curve-outputs',
      size: outputSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // NEW: tiny globals buffer (atomic found flag)
    const globalsBuffer = dev.createBuffer({
      label: 'ecm-globals',
      size: 4, // one u32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    dev.queue.writeBuffer(globalsBuffer, 0, new Uint32Array([0]));

    // Create bind group
    const bindGroup = dev.createBindGroup({
      label: 'ecm-bindgroup',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: packedBuffer } },
        { binding: 2, resource: { buffer: curveInputBuffer } },
        { binding: 3, resource: { buffer: curveOutputBuffer } },
        //{ binding: 4, resource: { buffer: globalsBuffer } },
      ]
    });

    // Dispatch compute shader — 1 curve per thread
    const WG_SIZE = 256;
    const numWorkgroups = Math.ceil(n / WG_SIZE);
    //DBG('dispatch', { WG_SIZE, CURVES_PER_WG, n, numWorkgroups });


    const encoder = dev.createCommandEncoder({ label: 'ecm-encoder' });
    const pass = encoder.beginComputePass({ label: 'ecm-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(numWorkgroups, 1, 1);
    pass.end();

    // Copy output buffer to readable buffer
    const readBuffer = dev.createBuffer({
      label: 'ecm-read-buffer',
      size: outputSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    encoder.copyBufferToBuffer(curveOutputBuffer, 0, readBuffer, 0, outputSize);
    // Before GPU execution, clear the output buffer
    //const clearData = new Uint32Array(outputSize / 4);
    //dev.queue.writeBuffer(curveOutputBuffer, 0, clearData);
    // Submit and wait
    dev.queue.submit([encoder.finish()]);
    await dev.queue.onSubmittedWorkDone().catch(e=>ERR('onSubmittedWorkDone error', e));
    DBG('submitted & GPU done');
    await readBuffer.mapAsync(GPUMapMode.READ);

    const outputData = readBuffer.getMappedRange();

    // Reconstruct full buffer with outputs (keep your wire format unchanged)
    const fullResult = new Uint32Array(total_words);
    fullResult.set(u32.slice(0, HEADER_WORDS + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER));

    // Copy output data to the correct position
    const outputOffset = HEADER_WORDS + CONST_WORDS + pp_count + n * CURVE_IN_WORDS_PER;
    const outputU32 = new Uint32Array(outputData);
    fullResult.set(outputU32, outputOffset);
    console.log('Output section starts at word', outputOffset);
    for(let i = 0; i < Math.min(5, n); i++) {
      const curveOffset = outputOffset + i * CURVE_OUT_WORDS_PER;
      const status = fullResult[curveOffset + 8];
      console.log(`Curve ${i}: status=${status}, result limbs:`,
        Array.from(fullResult.slice(curveOffset, curveOffset + 8)));
    }
    const checksumOut = u32sum(fullResult);
    DBG('runChunk(): finished', { checksumIn, checksumOut, total_words, prefixWords });
    const result = fullResult.buffer.slice(0);
    readBuffer.unmap();

    // Cleanup buffers we created in this call
    try { paramsBuffer.destroy?.(); } catch {}
    try { packedBuffer.destroy?.(); } catch {}
    try { curveInputBuffer.destroy?.(); } catch {}
    try { curveOutputBuffer.destroy?.(); } catch {}
    try { globalsBuffer.destroy?.(); } catch {}
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
