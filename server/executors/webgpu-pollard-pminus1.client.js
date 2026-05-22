const __WGPU_PM1_CACHE__ = (globalThis.__WGPU_PM1_CACHE__ ||= {
  device: null,
  pipelinesByDevice: new WeakMap(),
  lostReason: null,
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
  if (__WGPU_PM1_CACHE__.device) return __WGPU_PM1_CACHE__.device;
  if (__WGPU_PM1_CACHE__.lostReason) {
    throw new Error(`WebGPU device was previously lost: ${__WGPU_PM1_CACHE__.lostReason}. This usually means the kernel exceeded the browser's GPU timeout (TDR/watchdog).`);
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
    const reason = info?.reason || 'unknown';
    __WGPU_PM1_CACHE__.lostReason = reason;
    __WGPU_PM1_CACHE__.device = null;
    __WGPU_PM1_CACHE__.pipelinesByDevice = new WeakMap();
    console.error(`[WEBGPU WATCHDOG] Device lost! Reason: "${reason}". This almost certainly means a shader kernel exceeded the browser's ~5-10s GPU execution timeout and was killed by the TDR watchdog.`);
  }).catch(() => {});
  __WGPU_PM1_CACHE__.device = device;
  return device;
}

function getPipeline(device, kernelCode) {
  let perDevice = __WGPU_PM1_CACHE__.pipelinesByDevice.get(device);
  if (!perDevice) {
    perDevice = new Map();
    __WGPU_PM1_CACHE__.pipelinesByDevice.set(device, perDevice);
  }
  const key = `${kernelCode.length}:${kernelCode.slice(0, 64)}:${kernelCode.slice(-64)}`;
  let cached = perDevice.get(key);
  if (cached) return cached;

  const module = device.createShaderModule({ label: 'pollard-pminus1-module', code: kernelCode });
  const bgl = device.createBindGroupLayout({
    label: 'pollard-pminus1-layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ label: 'pollard-pminus1-pipeline-layout', bindGroupLayouts: [bgl] });
  const pipeline = device.createComputePipeline({
    label: 'pollard-pminus1-pipeline',
    layout,
    compute: { module, entryPoint: 'main' },
  });
  cached = { pipeline, bgl };
  perDevice.set(key, cached);
  return cached;
}

export function createExecutor({ kernels }) {
  const kernel = kernels.find((k) => k.name?.endsWith('pollard_pminus1_batched.wgsl'));
  if (!kernel) throw new Error('Pollard p-1 batched WGSL source missing');
  const kernelCode = kernel.content || kernel.code;

  async function prewarm() {
    const device = await getDevice();
    getPipeline(device, kernelCode);
  }

  async function runChunk({ payload }) {
    const tClientRecv = performance.now();
    try {
      const device = await getDevice();
      const { pipeline, bgl } = getPipeline(device, kernelCode);
      const input = new Uint32Array(toArrayBuffer(payload.data));
      const nBases = Number(payload.nBases || input[3] || 0) >>> 0;
      const numNs = Number(input[5] || 1) >>> 0;
      const totalThreads = numNs * nBases;
      const ppCount = Number(input[2] || 0) >>> 0;
      if (nBases === 0) throw new Error('Pollard p-1 chunk has no bases');
      if (totalThreads === 0) throw new Error('Pollard p-1 chunk has no work');
      if (ppCount === 0) throw new Error('Pollard p-1 chunk has no prime powers');

      const ioBuf = device.createBuffer({
        label: 'pollard-pminus1-io',
        size: Math.max(4, input.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(ioBuf, 0, input);

      const readBuf = device.createBuffer({
        label: 'pollard-pminus1-readback',
        size: input.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const bindGroup = device.createBindGroup({
        label: 'pollard-pminus1-bindgroup',
        layout: bgl,
        entries: [{ binding: 0, resource: { buffer: ioBuf } }],
      });

      const maxDim = device.limits.maxComputeWorkgroupsPerDimension;
      const totalGroups = Math.ceil(totalThreads / 256);
      if (totalGroups > maxDim) {
        throw new Error(`Pollard p-1 dispatch ${totalGroups} exceeds maxComputeWorkgroupsPerDimension ${maxDim}. Reduce chunkSize.`);
      }

      if (payload.disableWatchdog) {
        // ----- SINGLE PASS (watchdog disabled) -----
        input[6] = 0;
        input[7] = ppCount >>> 0;
        device.queue.writeBuffer(ioBuf, 0, input.buffer, 0, 32);

        const encoder = device.createCommandEncoder({ label: 'pollard-pminus1-encoder-single' });
        const pass = encoder.beginComputePass({ label: 'pollard-pminus1-pass-single' });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(totalGroups);
        pass.end();
        encoder.copyBufferToBuffer(ioBuf, 0, readBuf, 0, input.byteLength);

        const t0 = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone().catch((e) => {
          console.error('[Pollard p-1] GPU submit error:', e);
          throw e;
        });
        const cpuTime = performance.now() - t0;
        console.log(`Pollard p-1 single pass: ${ppCount} prime powers - CPU: ${cpuTime.toFixed(1)}ms (watchdog disabled)`);
      } else {
        // ----- RESUMABLE COMPUTATION LOOP (avoids browser TDR) -----
        const TARGET_MS = 500;
        let pp_len = Math.min(500, ppCount);
        let pp_start = 0;
        let passCount = 0;
        const overallStart = performance.now();

        while (pp_start < ppCount) {
          const currentPpLen = Math.min(pp_len, ppCount - pp_start);
          input[6] = pp_start >>> 0;
          input[7] = currentPpLen >>> 0;
          device.queue.writeBuffer(ioBuf, 0, input.buffer, 0, 32);

          const encoder = device.createCommandEncoder({ label: `pollard-pminus1-encoder-pass-${pp_start}` });
          const pass = encoder.beginComputePass({ label: `pollard-pminus1-pass-${pp_start}` });
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(totalGroups);
          pass.end();

          const isFinal = pp_start + currentPpLen >= ppCount;
          if (isFinal) {
            encoder.copyBufferToBuffer(ioBuf, 0, readBuf, 0, input.byteLength);
          }

          const t0 = performance.now();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone().catch((e) => {
            console.error('[Pollard p-1] GPU submit error:', e);
            throw e;
          });
          const cpuTime = performance.now() - t0;

          pp_start += currentPpLen;
          passCount++;

          const timingInfo = `CPU: ${cpuTime.toFixed(1)}ms`;
          console.log(`Pollard p-1 pass ${passCount}: pp[${pp_start - currentPpLen}:${pp_start}]/${ppCount} - ${timingInfo}`);

          if (!isFinal) {
            if (cpuTime < TARGET_MS / 2 && pp_len < ppCount / 10) {
              pp_len = Math.min(pp_len * 2, 10000);
            } else if (cpuTime > TARGET_MS * 1.5) {
              pp_len = Math.max(Math.floor(pp_len * TARGET_MS / cpuTime), 100);
            }
          }
        }

        const totalTime = ((performance.now() - overallStart) / 1000).toFixed(2);
        console.log(`Pollard p-1 Stage 1 complete: ${ppCount} prime powers in ${totalTime}s (${passCount} passes)`);
      }

      const KERNEL_TIMEOUT_MS = 120000;
      try {
        const mapPromise = readBuf.mapAsync(GPUMapMode.READ);
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error(`Pollard p-1 kernel timed out after ${KERNEL_TIMEOUT_MS}ms for chunk with ${numNs} numbers`)), KERNEL_TIMEOUT_MS);
        });
        await Promise.race([mapPromise, timeoutPromise]);
      } catch (e) {
        if (__WGPU_PM1_CACHE__.lostReason) {
          throw new Error(`Pollard p-1 chunk failed because WebGPU device was lost (reason: "${__WGPU_PM1_CACHE__.lostReason}"). This is the browser watchdog/TDR killing the kernel for exceeding the GPU time limit.`);
        }
        throw e;
      }
      const result = readBuf.getMappedRange().slice(0);
      readBuf.unmap();

      try { ioBuf.destroy?.(); } catch {}
      try { readBuf.destroy?.(); } catch {}

      const tClientDone = performance.now();
      return {
        status: 'ok',
        result,
        timings: {
          tClientRecv,
          tClientDone,
          cpuTimeMs: tClientDone - tClientRecv,
          gpuTimeMs: null,
        },
      };
    } catch (e) {
      const errMsg = e?.message || String(e) || 'unknown error';
      console.error(`[POLLARD P-1 EXECUTOR ERROR] ${errMsg}`, e);
      throw new Error(`Pollard p-1 executor failed: ${errMsg}`);
    }
  }

  return { prewarm, runChunk };
}
