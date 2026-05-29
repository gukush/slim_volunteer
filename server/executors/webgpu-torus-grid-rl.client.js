const __WGPU_TORUS_GRID_RL_CACHE__ = (globalThis.__WGPU_TORUS_GRID_RL_CACHE__ ||= {
  device: null,
  pipelinesByDevice: new WeakMap(),
  lostReason: null,
});

const WEIGHT_COUNT = 24;
const TILE_COUNT = 4096;
const RESULT_WORDS = 6 + WEIGHT_COUNT;
const WEIGHT_SCALE = 65536.0;

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
  if (__WGPU_TORUS_GRID_RL_CACHE__.device) return __WGPU_TORUS_GRID_RL_CACHE__.device;
  if (__WGPU_TORUS_GRID_RL_CACHE__.lostReason) {
    throw new Error(`WebGPU device was previously lost: ${__WGPU_TORUS_GRID_RL_CACHE__.lostReason}`);
  }
  if (!('gpu' in navigator)) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    },
  });
  device.lost.then((info) => {
    __WGPU_TORUS_GRID_RL_CACHE__.lostReason = info?.reason || 'unknown';
    __WGPU_TORUS_GRID_RL_CACHE__.device = null;
    __WGPU_TORUS_GRID_RL_CACHE__.pipelinesByDevice = new WeakMap();
  }).catch(() => {});
  __WGPU_TORUS_GRID_RL_CACHE__.device = device;
  return device;
}

function specializeKernel(kernelCode, workgroupSize) {
  const size = Number(workgroupSize || 128);
  if (!Number.isInteger(size) || size <= 0 || size > 1024) {
    throw new Error(`Invalid torus-grid-rl workgroupSize: ${workgroupSize}`);
  }
  return kernelCode
    .replace('const WORKGROUP_SIZE: u32 = 128u;', `const WORKGROUP_SIZE: u32 = ${size}u;`)
    .replace('@compute @workgroup_size(128)', `@compute @workgroup_size(${size})`);
}

function getPipeline(device, kernelCode, workgroupSize) {
  let perDevice = __WGPU_TORUS_GRID_RL_CACHE__.pipelinesByDevice.get(device);
  if (!perDevice) {
    perDevice = new Map();
    __WGPU_TORUS_GRID_RL_CACHE__.pipelinesByDevice.set(device, perDevice);
  }

  const specialized = specializeKernel(kernelCode, workgroupSize);
  let hash = 2166136261;
  for (let i = 0; i < specialized.length; i++) {
    hash ^= specialized.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  const key = `${workgroupSize}:${specialized.length}:${hash}`;
  let cached = perDevice.get(key);
  if (cached) return cached;

  const module = device.createShaderModule({ label: 'torus-grid-rl-module', code: specialized });
  const bgl = device.createBindGroupLayout({
    label: 'torus-grid-rl-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const layout = device.createPipelineLayout({ label: 'torus-grid-rl-pipeline-layout', bindGroupLayouts: [bgl] });
  const pipeline = device.createComputePipeline({
    label: 'torus-grid-rl-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });
  cached = { pipeline, bgl };
  perDevice.set(key, cached);
  return cached;
}

function weightsToFixed(weights) {
  const fixed = new Int32Array(WEIGHT_COUNT);
  for (let i = 0; i < WEIGHT_COUNT; i++) {
    fixed[i] = Math.round(Number(weights[i] || 0) * WEIGHT_SCALE);
  }
  return fixed;
}

export function createExecutor({ kernels, config, inputArgs }) {
  const kernel = kernels.find((k) => k.name?.endsWith('torus_grid_rl.wgsl'));
  if (!kernel) throw new Error('Torus Grid RL WGSL source missing');
  const kernelCode = kernel.content || kernel.code;
  const configuredWorkgroupSize = Number(inputArgs?.workgroupSize ?? config?.workgroupSize ?? 128);

  async function prewarm() {
    const device = await getDevice();
    getPipeline(device, kernelCode, configuredWorkgroupSize);
  }

  async function runChunk({ payload }) {
    const tClientRecv = performance.now();
    const device = await getDevice();

    const paramsIn = new Uint32Array(toArrayBuffer(payload.params));
    const trajectories = Number(paramsIn[0] || 0);
    const workgroupSize = Number(paramsIn[1] || configuredWorkgroupSize);
    if (trajectories <= 0) throw new Error('torus-grid-rl chunk has no trajectories');
    if (workgroupSize !== configuredWorkgroupSize) {
      throw new Error(`torus-grid-rl workgroupSize changed after init: ${configuredWorkgroupSize} -> ${workgroupSize}`);
    }
    const { pipeline, bgl } = getPipeline(device, kernelCode, workgroupSize);

    const rewards = new Float32Array(toArrayBuffer(payload.rewardMap));
    if (rewards.length !== TILE_COUNT) throw new Error(`Expected ${TILE_COUNT} reward tiles, got ${rewards.length}`);
    const weights = new Float32Array(toArrayBuffer(payload.weights));
    if (weights.length !== WEIGHT_COUNT) throw new Error(`Expected ${WEIGHT_COUNT} weights, got ${weights.length}`);
    const fixedWeights = weightsToFixed(weights);

    const rewardsBuf = device.createBuffer({
      label: 'torus-grid-rl-rewards',
      size: rewards.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(rewardsBuf, 0, rewards);

    const weightsBuf = device.createBuffer({
      label: 'torus-grid-rl-weights',
      size: fixedWeights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(weightsBuf, 0, fixedWeights);

    const stats = new Uint32Array(4);
    const statsBuf = device.createBuffer({
      label: 'torus-grid-rl-stats',
      size: stats.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(statsBuf, 0, stats);

    const maxDim = device.limits.maxComputeWorkgroupsPerDimension;
    const totalGroups = Math.ceil(trajectories / workgroupSize);
    const dispatchX = Math.min(totalGroups, maxDim);
    const dispatchY = Math.ceil(totalGroups / maxDim);
    if (dispatchY > maxDim) {
      throw new Error(`torus-grid-rl dispatch ${dispatchX}x${dispatchY} exceeds max dimension ${maxDim}; reduce chunkTrajectories`);
    }

    const configWords = new Uint32Array([
      trajectories >>> 0,
      workgroupSize >>> 0,
      paramsIn[2] >>> 0,
      paramsIn[3] >>> 0,
      dispatchX >>> 0,
      0, 0, 0,
    ]);
    const configBuf = device.createBuffer({
      label: 'torus-grid-rl-config',
      size: configWords.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(configBuf, 0, configWords);

    const readSize = fixedWeights.byteLength + stats.byteLength;
    const readBuf = device.createBuffer({
      label: 'torus-grid-rl-readback',
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroup = device.createBindGroup({
      label: 'torus-grid-rl-bindgroup',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: rewardsBuf } },
        { binding: 1, resource: { buffer: weightsBuf } },
        { binding: 2, resource: { buffer: statsBuf } },
        { binding: 3, resource: { buffer: configBuf } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'torus-grid-rl-encoder' });
    const pass = encoder.beginComputePass({ label: 'torus-grid-rl-pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();
    encoder.copyBufferToBuffer(statsBuf, 0, readBuf, 0, stats.byteLength);
    encoder.copyBufferToBuffer(weightsBuf, 0, readBuf, stats.byteLength, fixedWeights.byteLength);
    device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const mapped = readBuf.getMappedRange().slice(0);
    readBuf.unmap();

    const statsOut = new Uint32Array(mapped, 0, 4);
    const fixedOut = new Int32Array(mapped, stats.byteLength, WEIGHT_COUNT);
    const result = new Float32Array(RESULT_WORDS);
    result[0] = 20240529;
    result[1] = statsOut[0];
    result[2] = statsOut[1];
    result[3] = (statsOut[2] / 1000.0) - statsOut[0] * 512.0;
    result[4] = statsOut[3];
    result[5] = workgroupSize;
    for (let i = 0; i < WEIGHT_COUNT; i++) {
      result[6 + i] = fixedOut[i] / WEIGHT_SCALE;
    }
    if (statsOut[0] !== trajectories) {
      throw new Error(`torus-grid-rl kernel produced ${statsOut[0]} trajectories, expected ${trajectories}`);
    }

    try { rewardsBuf.destroy?.(); } catch {}
    try { weightsBuf.destroy?.(); } catch {}
    try { statsBuf.destroy?.(); } catch {}
    try { configBuf.destroy?.(); } catch {}
    try { readBuf.destroy?.(); } catch {}

    const tClientDone = performance.now();
    return {
      status: 'ok',
      result: result.buffer.slice(result.byteOffset, result.byteOffset + result.byteLength),
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
