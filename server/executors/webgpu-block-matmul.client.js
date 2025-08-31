export function createExecutor({ kernels, config }){
  const kernel = kernels.find(k => k.name.endsWith('block_matmul.wgsl'));
  if(!kernel) throw new Error('Kernel source missing');
  let device, pipeline, layout;

  async function getDevice(){
    if(device) return device;
    if(!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();
    return device;
  }

  async function ensurePipeline(){
    if(pipeline) return pipeline;
    const dev = await getDevice();
    const module = dev.createShaderModule({ code: kernel.content });
    layout = dev.createBindGroupLayout({
      entries: [
        { binding:0, visibility: GPUShaderStage.COMPUTE, buffer: { type:'read-only-storage' } },
        { binding:1, visibility: GPUShaderStage.COMPUTE, buffer: { type:'read-only-storage' } },
        { binding:2, visibility: GPUShaderStage.COMPUTE, buffer: { type:'storage' } },
        { binding:3, visibility: GPUShaderStage.COMPUTE, buffer: { type:'uniform', minBindingSize: 16 } },
      ]
    });
    const plLayout = dev.createPipelineLayout({ bindGroupLayouts: [layout] });
    pipeline = dev.createComputePipeline({ layout: plLayout, compute: { module, entryPoint: 'main' } });
    return pipeline;
  }

  async function runChunk({ payload, meta }){
    const tClientRecv = Date.now();
    await ensurePipeline();
    const dev = device;
    const { a, b, dims } = payload;
    const { rows, K, cols } = dims;
    const aBuf = dev.createBuffer({ size: a.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(aBuf, 0, a);

    const bBuf = dev.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(bBuf, 0, b);

    const outSize = rows * cols * 4;
    const cBuf = dev.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const dimsData = new Uint32Array([rows, K, cols, 0]); // pad to 16B
    const dimsBuf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(dimsBuf, 0, dimsData);

    const bind = dev.createBindGroup({
      layout,
      entries: [
        { binding:0, resource:{ buffer:aBuf } },
        { binding:1, resource:{ buffer:bBuf } },
        { binding:2, resource:{ buffer:cBuf } },
        { binding:3, resource:{ buffer:dimsBuf } },
      ]
    });

    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    const wg = 16;
    const x = Math.ceil(cols / wg);
    const y = Math.ceil(rows / wg);
    pass.dispatchWorkgroups(x, y, 1);
    pass.end();

    const readBuf = dev.createBuffer({ size: outSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(cBuf, 0, readBuf, 0, outSize);
    dev.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();
    const tClientDone = Date.now();
    return { status: 'ok', result, timings: { tClientRecv, tClientDone } };
  }

  return { runChunk };
}
