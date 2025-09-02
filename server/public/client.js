const qs = Object.fromEntries(new URLSearchParams(location.search));
const logLevel = (qs.log || 'info').toLowerCase();
const levels = ['error','warn','info','debug','trace'];
const idx = levels.indexOf(logLevel);
function log(kind, ...args){ if(levels.indexOf(kind)<=idx){ const el=document.getElementById('log'); el.textContent += `[${new Date().toISOString()}] ${kind.toUpperCase()} ${args.join(' ')}\n`; el.scrollTop=el.scrollHeight; console[kind==='trace'?'debug':kind](...args);}}

const statusEl = document.getElementById('status');

function checksumHex(buffer){
  return crypto.subtle.digest('SHA-256', buffer).then(buf=>{
    const v = new Uint8Array(buf); return Array.from(v).map(b=>b.toString(16).padStart(2,'0')).join('');
  });
}

const socket = io({ transports:['websocket'], auth:{ token: 'anon' }, forceNew: true, ackTimeout: 10000 });
const executors = new Map(); // taskId -> executor
const readyTasks = new Set();

socket.on('connect', ()=>{
  statusEl.textContent = 'connected';
  log('info', 'Connected', socket.id);
  const frameworks = ['cuda','vulkan','opencl'];
  if('gpu' in navigator) frameworks.push('webgpu');
  const capacity = Number(qs.cap || qs.capacity || 1);
  socket.emit('hello', { workerId: qs.workerId || socket.id, frameworks, capacity });
});

socket.on('disconnect', ()=>{
  statusEl.textContent = 'disconnected';
  log('warn', 'Disconnected');
});


socket.on('task:init', (msg)=>{
  log('info', 'task:init', JSON.stringify({ taskId: msg.taskId, strategyId: msg.strategyId }));
  if(msg.framework==='webgpu' && !('gpu' in navigator)){
    log('warn', 'No WebGPU; ignoring task', msg.taskId);
    return;
  }

  // Create and register a placeholder immediately so chunk assignments
  // won't incorrectly treat this task as "no executor".
  let resolveReady;
  const placeholder = { __ready__: new Promise(res => { resolveReady = res; }), __resolved__: false };
  executors.set(msg.taskId, placeholder);

  (async ()=>{
    try{
      const blob = new Blob([msg.executorCode], { type: 'text/javascript' });
      const modUrl = URL.createObjectURL(blob);
      const mod = await import(modUrl);
      const exec = mod.createExecutor({ kernels: msg.kernels, config: msg.config, schema: msg.schema, inputArgs: msg.inputArgs });

      // IMPORTANT: await prewarm if available. This ensures WebGPU executor
      // fully initializes before we start running chunks — avoids race
      // conditions where a chunk is assigned before device/pipelines ready.
      if (typeof exec.prewarm === 'function') {
        try { await exec.prewarm(); } catch (e) { log('warn', 'exec.prewarm failed', e); }
      }

      executors.set(msg.taskId, exec);
      resolveReady(exec);
      placeholder.__resolved__ = true;
    }catch(e){
      log('error', 'task:init failed', e);
      // Remove the executor mapping and resolve with null so awaiting
      // chunk handlers get notified of the failure.
      executors.delete(msg.taskId);
      resolveReady(null);
    }
  })();
});

socket.on('chunk:assign', async (job)=>{
  const { taskId, chunkId, replica, payload, meta, tCreate } = job;
  let exec = executors.get(taskId);
  if(!exec){
    log('warn', 'No executor for task', taskId);
    socket.emit('chunk:result', { taskId, chunkId, replica, status: 'no-exec' });
    return;
  }

  // If this is the placeholder (has __ready__), wait for initialization.
  if (exec && exec.__ready__ instanceof Promise) {
    log('debug', 'Waiting for executor to become ready for task', taskId);
    exec = await exec.__ready__;
    if(!exec){
      log('warn', 'Executor failed to initialize for task', taskId);
      socket.emit('chunk:result', { taskId, chunkId, replica, status: 'no-exec' });
      return;
    }
  }

  try{
    const res = await exec.runChunk({ payload, meta });
    const checksum = await checksumHex(res.result);
    socket.emit('chunk:result', {
      taskId, chunkId, replica, status: res.status, checksum,
      result: res.result, timings: res.timings
    });
    log('debug', 'chunk done', chunkId, 'cs', checksum.slice(0,8));
  }catch(e){
    log('error', 'chunk failed', chunkId, e.message);
    socket.emit('chunk:result', { taskId, chunkId, replica, status: 'error', error: e.message });
  }
});