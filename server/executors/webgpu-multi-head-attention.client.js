// executors/webgpu-multi-head-attention.client.js
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

  let device, pipeline, bindGroupLayout;

  async function getDevice() {
    if (device) return device;
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');
    device = await adapter.requestDevice({
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
    return device;
  }

  async function ensurePipeline() {
    if (pipeline) return { pipeline, bindGroupLayout };

    const dev = await getDevice();
    const module = dev.createShaderModule({
      code: kernel.content,
      label: 'mha-compute-shader'
    });

    bindGroupLayout = dev.createBindGroupLayout({
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
      bindGroupLayouts: [bindGroupLayout]
    });

    pipeline = dev.createComputePipeline({
      label: 'mha-compute-pipeline',
      layout: pipelineLayout,
      compute: { module, entryPoint: 'main' }
    });

    return { pipeline, bindGroupLayout };
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
      const dev = device;

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
      const computePass = encoder.beginComputePass({ label: 'mha-compute-pass' });
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

      await readBuffer.mapAsync(GPUMapMode.READ);
      const result = readBuffer.getMappedRange().slice(0);
      readBuffer.unmap();

      // Cleanup (debug-level)
      try {
        qBuffer.destroy(); kBuffer.destroy(); vBuffer.destroy();
        outputBuffer.destroy(); dimsBuffer.destroy(); readBuffer.destroy();
      } catch (e) { log.debug('buffer destroy error:', e?.message); }

      const tClientDone = Date.now();
      log.info(`MHA[head=${meta?.head ?? '?'}] done ${tClientDone - tClientRecv}ms`);

      return { status: 'ok', result, timings: { tClientRecv, tClientDone } };

    } catch (e) {
      const tClientDone = Date.now();
      log.error(`MHA[head=${meta?.head ?? '?'}] FAILED after ${tClientDone - tClientRecv}ms:`, e?.message);
      return { status: 'error', error: e?.message || String(e), timings: { tClientRecv, tClientDone } };
    }
  }

  // Optional: a prewarm so client.js can await it (already supported there).
  async function prewarm() {
    await ensurePipeline();
  }

  return { runChunk, prewarm };
}
