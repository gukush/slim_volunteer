export function createExecutor({ kernels, config }) {
  const kernel = kernels.find(k => k.name.endsWith('bitonic_sort.wgsl'));
  if (!kernel) throw new Error('Bitonic sort kernel source missing');

  let device, pipeline, layout, uniformBuf;
  let deviceId = null; // For debugging

  async function getDevice() {
    if (device && !device.lost) return device;

    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();
    deviceId = Math.random().toString(36).substr(2, 9); // Debug ID
    console.log('Created WebGPU device:', deviceId);

    // Reset pipeline when device is recreated
    pipeline = null;
    layout = null;
    uniformBuf = null;

    return device;
  }

  async function ensurePipeline() {
    if (pipeline && layout && uniformBuf) {
      console.log('Reusing existing pipeline for device:', deviceId);
      return { pipeline, layout, uniformBuf };
    }

    const dev = await getDevice();
    console.log('Creating new pipeline for device:', deviceId);

    const module = dev.createShaderModule({
      code: kernel.content,
      label: 'bitonic-sort-shader'
    });

    layout = dev.createBindGroupLayout({
      label: 'bitonic-sort-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', minBindingSize: 16 }
        }
      ]
    });

    const plLayout = dev.createPipelineLayout({
      label: 'bitonic-sort-pipeline-layout',
      bindGroupLayouts: [layout]
    });

    pipeline = dev.createComputePipeline({
      label: 'bitonic-sort-pipeline',
      layout: plLayout,
      compute: { module, entryPoint: 'main' }
    });

    // Create reusable uniform buffer
    uniformBuf = dev.createBuffer({
      label: 'bitonic-sort-uniforms',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    return { pipeline, layout, uniformBuf };
  }

  async function runChunk({ payload, meta }) {
    const tClientRecv = Date.now();

    try {
      const dev = await getDevice();
      const { pipeline: pipe, layout: bindLayout, uniformBuf: uniforms } = await ensurePipeline();

      const { data, originalSize, paddedSize, ascending } = payload;
      const integers = new Uint32Array(data);

      console.log(`Processing chunk: ${originalSize} integers (padded to ${paddedSize}) with device ${deviceId}`);

      // Create GPU buffer for this chunk's data
      const dataSize = paddedSize * 4; // 4 bytes per uint32
      const dataBuf = dev.createBuffer({
        label: 'chunk-data',
        size: dataSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      });
      dev.queue.writeBuffer(dataBuf, 0, integers);

      // Perform bitonic sort - multiple passes
      const numStages = Math.log2(paddedSize);

      for (let stage = 1; stage <= numStages; stage++) {
        for (let substage = stage; substage >= 1; substage--) {
          // Update uniform parameters
          const params = new Uint32Array([
            paddedSize,           // array_size
            stage - 1,           // stage (0-indexed)
            substage - 1,        // substage (0-indexed)
            ascending ? 1 : 0    // ascending flag
          ]);
          dev.queue.writeBuffer(uniforms, 0, params);

          // Create bind group for this pass
          const bindGroup = dev.createBindGroup({
            label: `sort-pass-${stage}-${substage}`,
            layout: bindLayout,
            entries: [
              { binding: 0, resource: { buffer: dataBuf } },
              { binding: 1, resource: { buffer: uniforms } }
            ]
          });

          // Execute compute pass
          const encoder = dev.createCommandEncoder({ label: 'sort-encoder' });
          const pass = encoder.beginComputePass({ label: 'sort-pass' });
          pass.setPipeline(pipe);
          pass.setBindGroup(0, bindGroup);

          // Dispatch with appropriate workgroup count
          const workgroupSize = 256;
          const numWorkgroups = Math.ceil(paddedSize / (workgroupSize * 2));
          pass.dispatchWorkgroups(numWorkgroups, 1, 1);
          pass.end();

          dev.queue.submit([encoder.finish()]);
          await dev.queue.onSubmittedWorkDone();
        }
      }

      // Read back results
      const readBuf = dev.createBuffer({
        label: 'read-buffer',
        size: dataSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      const encoder = dev.createCommandEncoder({ label: 'copy-encoder' });
      encoder.copyBufferToBuffer(dataBuf, 0, readBuf, 0, dataSize);
      dev.queue.submit([encoder.finish()]);

      await readBuf.mapAsync(GPUMapMode.READ);
      const result = readBuf.getMappedRange().slice(0);
      readBuf.unmap();

      const tClientDone = Date.now();
      console.log(`Chunk completed in ${tClientDone - tClientRecv}ms`);

      return {
        status: 'ok',
        result,
        timings: { tClientRecv, tClientDone }
      };

    } catch (error) {
      console.error('WebGPU error:', error);
      console.error('Device ID:', deviceId);

      // Try to recover by recreating everything
      device = null;
      pipeline = null;
      layout = null;
      uniformBuf = null;

      throw error;
    }
  }

  return { runChunk };
}