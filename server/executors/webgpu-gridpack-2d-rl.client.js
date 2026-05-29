// executors/webgpu-gridpack-2d-rl.client.js
// Volunteer WebGPU client for GridPack-2D RL with ES policy gradient.

const __WGPU_GRIDPACK_CACHE__ = (globalThis.__WGPU_GRIDPACK_CACHE__ ||= {
  device: null,
  pipelinesByDevice: new WeakMap(),
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
  if (__WGPU_GRIDPACK_CACHE__.device) return __WGPU_GRIDPACK_CACHE__.device;
  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();
  device.lost.then(() => {
    __WGPU_GRIDPACK_CACHE__.device = null;
    __WGPU_GRIDPACK_CACHE__.pipelinesByDevice = new WeakMap();
  }).catch(() => {});
  __WGPU_GRIDPACK_CACHE__.device = device;
  return device;
}

function getPipeline(device, kernelCode) {
  if (!__WGPU_GRIDPACK_CACHE__.pipelinesByDevice) {
    __WGPU_GRIDPACK_CACHE__.pipelinesByDevice = new WeakMap();
  }
  let perDevice = __WGPU_GRIDPACK_CACHE__.pipelinesByDevice.get(device);
  if (!perDevice) {
    perDevice = new Map();
    __WGPU_GRIDPACK_CACHE__.pipelinesByDevice.set(device, perDevice);
  }
  const key = `${kernelCode.length}:${kernelCode.slice(0, 64)}:${kernelCode.slice(-64)}`;
  let cached = perDevice.get(key);
  if (cached) return cached;

  const module = device.createShaderModule({ label: 'gridpack-2d-rl-module', code: kernelCode });
  const bgl = device.createBindGroupLayout({
    label: 'gridpack-2d-rl-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ label: 'gridpack-2d-rl-pipeline-layout', bindGroupLayouts: [bgl] });
  const pipeline = device.createComputePipeline({
    label: 'gridpack-2d-rl-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });
  cached = { pipeline, bgl };
  perDevice.set(key, cached);
  return cached;
}

export function createExecutor({ kernels }) {
  const kernel = kernels.find((k) => k.name?.includes('gridpack_2d_rl'));
  if (!kernel) throw new Error('GridPack-2D RL WGSL source missing');
  const kernelCode = kernel.content || kernel.code;

  async function prewarm() {
    const device = await getDevice();
    getPipeline(device, kernelCode);
  }

  async function runChunk({ payload }) {
    const tClientRecv = Date.now();
    const device = await getDevice();
    const { pipeline, bgl } = getPipeline(device, kernelCode);

    const blocksFlat = payload.blocks || [];
    const gridW = Number(payload.params?.[0] || 16);
    const gridH = Number(payload.params?.[1] || 16);
    const numBlocks = Number(payload.params?.[2] || Math.floor(blocksFlat.length / 3));
    const rolloutsPerThread = Number(payload.params?.[3] || 64);
    const seed = Number(payload.params?.[4] || 12345);
    const maxAttempts = Number(payload.params?.[5] || 50);
    const allowRotation = Number(payload.params?.[6] ?? 1);
    const sigma = Number(payload.sigma ?? payload.params?.[7] ?? 0.01);
    const numThreads = Number(payload.numThreads ?? payload.params?.[8] ?? 256);
    const thetaVersion = Number(payload.thetaVersion ?? payload.params?.[9] ?? 0);
    const epsilonSeed = Number(payload.epsilonSeed ?? payload.params?.[10] ?? 12345);
    const mode = Number(payload.mode ?? payload.params?.[11] ?? 0);

    // Build blocks buffer
    const blocksData = new Uint32Array(numBlocks * 2);
    for (let i = 0; i < numBlocks; i++) {
      blocksData[i * 2] = blocksFlat[i * 3] || 1;
      blocksData[i * 2 + 1] = blocksFlat[i * 3 + 1] || 1;
    }

    // Execution sizing
    const workgroupSize = 64;
    const roundedThreads = Math.ceil(numThreads / workgroupSize) * workgroupSize;
    const outWordsPerThread = 6;
    const outTotalWords = roundedThreads * outWordsPerThread;

    // Params buffer (16 u32s = 64 bytes)
    const paramsData = new Uint32Array([
      gridW, gridH, numBlocks, rolloutsPerThread,
      seed, maxAttempts, allowRotation,
      0, // sigma as f32 will be written separately
      numThreads, thetaVersion, epsilonSeed, mode,
      0, 0, 0, 0,
    ]);
    // Write sigma as f32 at offset 7 (bytes 28-31)
    const paramsView = new DataView(paramsData.buffer);
    paramsView.setFloat32(28, sigma, true);

    const paramsBuf = device.createBuffer({
      label: 'gridpack-params',
      size: paramsData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, paramsData);

    const blocksBuf = device.createBuffer({
      label: 'gridpack-blocks',
      size: blocksData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(blocksBuf, 0, blocksData);

    // Theta buffer (2400 f32 = 9600 bytes)
    const thetaData = new Float32Array(toArrayBuffer(payload.theta || new ArrayBuffer(0)));
    if (thetaData.length !== 2400) {
      console.warn(`[GridPack] theta length mismatch: ${thetaData.length}, expected 2400`);
    }
    const thetaBuf = device.createBuffer({
      label: 'gridpack-theta',
      size: 2400 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(thetaBuf, 0, thetaData);

    // Epsilon buffer (2400 f32 = 9600 bytes)
    const epsData = new Float32Array(toArrayBuffer(payload.epsilon || new ArrayBuffer(0)));
    if (epsData.length !== 2400) {
      console.warn(`[GridPack] epsilon length mismatch: ${epsData.length}, expected 2400`);
    }
    const epsBuf = device.createBuffer({
      label: 'gridpack-epsilon',
      size: 2400 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(epsBuf, 0, epsData);

    const outBuf = device.createBuffer({
      label: 'gridpack-out',
      size: outTotalWords * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const readBuf = device.createBuffer({
      label: 'gridpack-readback',
      size: outTotalWords * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroup = device.createBindGroup({
      label: 'gridpack-bindgroup',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: blocksBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: thetaBuf } },
        { binding: 4, resource: { buffer: epsBuf } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'gridpack-encoder' });
    const pass = encoder.beginComputePass({ label: 'gridpack-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(roundedThreads / workgroupSize));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outTotalWords * 4);
    device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const rawResult = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    try { paramsBuf.destroy?.(); } catch {}
    try { blocksBuf.destroy?.(); } catch {}
    try { thetaBuf.destroy?.(); } catch {}
    try { epsBuf.destroy?.(); } catch {}
    try { outBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}

    // Parse per-thread results and aggregate
    const resultF32 = new Float32Array(rawResult);
    let totalReward = 0.0;
    let totalValid = 0;
    let totalRollouts = 0;
    let threadCount = 0;

    for (let t = 0; t < numThreads; t++) {
      const base = t * 6;
      if (base + 5 >= resultF32.length) continue;
      // Skip magic check for now
      const meanReward = resultF32[base + 1];
      const validCount = resultF32[base + 2];
      const rollouts = resultF32[base + 3];
      const threadMode = resultF32[base + 4];

      if (validCount > 0) {
        totalReward += meanReward * validCount;
        totalValid += validCount;
        totalRollouts += rollouts;
        threadCount++;
      }
    }

    const overallMeanReward = totalValid > 0 ? totalReward / totalValid : -1.0;
    const summary = new Float32Array(6);
    summary[0] = 0x47524C50;
    summary[1] = overallMeanReward;
    summary[2] = totalValid;
    summary[3] = totalRollouts;
    summary[4] = mode;
    summary[5] = threadCount;

    const tClientDone = Date.now();
    return {
      status: 'ok',
      result: summary.buffer.slice(summary.byteOffset, summary.byteOffset + summary.byteLength),
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
