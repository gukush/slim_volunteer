/* gpu-worker.js — owns ONE GPUDevice for the whole client; hot-load executors per taskId */

let sharedDevicePromise = null;
// taskId -> { preload?(ctx), runChunk({payload,meta}), dispose?() }
const executors = new Map();

async function getDevice(){
  if (!sharedDevicePromise) {
    sharedDevicePromise = (async ()=>{
      if (!self.navigator?.gpu) throw new Error('WebGPU not available in worker');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No GPU adapter');
      const device = await adapter.requestDevice();
      device.label = 'distributed-single-device';
      device.lost.then(info=>{
        console.warn('[gpu-worker] device lost:', info);
        sharedDevicePromise = null;
        for (const ex of executors.values()) { try { ex.dispose?.(); } catch {} }
        executors.clear();
      });
      return device;
    })();
  }
  return sharedDevicePromise;
}

async function importFromCode(name, code){
  const blob = new Blob([code], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  try { return await import(url); }
  finally { URL.revokeObjectURL(url); }
}

async function makeExecutor(mod, ctx){
  const device = await getDevice();
  if (typeof mod.createExecutor !== 'function'){
    throw new Error('Executor must export createExecutor({ device, ... })');
  }
  const ex = await mod.createExecutor({ device, ...ctx });
  if (!ex?.runChunk) throw new Error('Executor missing runChunk');
  return ex;
}

self.onmessage = async (e)=>{
  const { id, type } = e.data;
  const reply = (payload, transfers=[]) => self.postMessage({ id, ...payload }, transfers);

  try {
    if (type === 'loadExecutorFromCode'){
      const { taskId, name, code, context } = e.data;
      const mod = await importFromCode(name || `exec-${taskId}`, code);
      executors.get(taskId)?.dispose?.();
      const ex = await makeExecutor(mod, context || {});
      executors.set(taskId, ex);
      if (ex.preload) await ex.preload(context || {}); // compile pipelines now
      reply({ ok: true, result: `executor:loaded(${taskId})` });
      return;
    }

    if (type === 'runChunk'){
      const { taskId, payload, meta } = e.data;
      const ex = executors.get(taskId);
      if (!ex) throw new Error(`No executor for task ${taskId}`);
      const out = await ex.runChunk({ payload, meta }); // { status, result, timings }
      reply({ ok: true, result: out }, out?.result ? [out.result] : []);
      return;
    }

    if (type === 'ping'){ reply({ ok:true, result:'pong' }); return; }

    throw new Error(`Unknown worker msg type: ${type}`);
  } catch (err) {
    reply({ ok:false, error: String(err?.message || err) });
  }
};
