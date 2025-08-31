export function createExecutor({ kernels, config }){
  const kernel = kernels.find(k => k.name.endsWith('ecm_stage1_webgpu_compute.wgsl'));
  if(!kernel) throw new Error('ECM kernel missing');
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
    // This layout is generic; if your kernel expects different bindings, adjust here.
    layout = dev.createBindGroupLayout({
      entries: [
        { binding:0, visibility: GPUShaderStage.COMPUTE, buffer: { type:'storage' } }, // in/out data
        { binding:1, visibility: GPUShaderStage.COMPUTE, buffer: { type:'uniform' } }, // dims
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
    const { data, dims } = payload;
    const n = dims.n|0;
    const dataBuf = dev.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    dev.queue.writeBuffer(dataBuf, 0, data);

    const dimBuf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const u = new Uint32Array([n]);
    dev.queue.writeBuffer(dimBuf, 0, u);

    const bind = dev.createBindGroup({ layout, entries:[
      { binding:0, resource:{ buffer:dataBuf } },
      { binding:1, resource:{ buffer:dimBuf } },
    ]});

    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    const wg = 64;
    const x = Math.ceil(n / wg);
    pass.dispatchWorkgroups(x, 1, 1);
    pass.end();

    const readBuf = dev.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(dataBuf, 0, readBuf, 0, data.byteLength);
    dev.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = readBuf.getMappedRange().slice(0);
    readBuf.unmap();
    const tClientDone = Date.now();
    return { status: 'ok', result, timings: { tClientRecv, tClientDone } };
  }

  return { runChunk };
}
