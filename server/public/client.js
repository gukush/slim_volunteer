const qs = Object.fromEntries(new URLSearchParams(location.search));
const logLevel = (qs.log || 'info').toLowerCase();
const levels = ['error','warn','info','debug','trace'];
const idx = levels.indexOf(logLevel);
function log(kind, ...args){ if(levels.indexOf(kind)<=idx){ const el=document.getElementById('log'); el.textContent += `[${new Date().toISOString()}] ${kind.toUpperCase()} ${args.join(' ')}\n`; el.scrollTop=el.scrollHeight; console[kind==='trace'?'debug':kind](...args);}}

// Listener functionality
const enableListener = qs.listener === '1';
let listenerWs = null;

const statusEl = document.getElementById('status');

function checksumHex(buffer){
  return crypto.subtle.digest('SHA-256', buffer).then(buf=>{
    const v = new Uint8Array(buf); return Array.from(v).map(b=>b.toString(16).padStart(2,'0')).join('');
  });
}

const socket = io({ transports:['websocket'], auth:{ token: 'anon' }, forceNew: true, ackTimeout: 10000 });
const executors = new Map(); // taskId -> executor
const readyTasks = new Set();

// Listener connection functions
function connectToListener() {
  if (!enableListener) return;

  try {
    listenerWs = new WebSocket('wss://127.0.0.1:8765');

    listenerWs.onopen = function() {
      log('info', 'Connected to listener at wss://127.0.0.1:8765');
    };

    listenerWs.onclose = function() {
      log('warn', 'Disconnected from listener');
      listenerWs = null;
    };

    listenerWs.onerror = function(error) {
      log('error', 'Listener connection error:', error);
    };

    listenerWs.onmessage = function(event) {
      try {
        const response = JSON.parse(event.data);
        log('debug', 'Listener response:', response);
      } catch (e) {
        log('warn', 'Failed to parse listener response:', event.data);
      }
    };
  } catch (error) {
    log('error', 'Failed to connect to listener:', error);
  }
}

function notifyListenerChunkArrival(chunkId, taskId) {
  if (!enableListener || !listenerWs || listenerWs.readyState !== WebSocket.OPEN) return;

  // Make this asynchronous and non-blocking
  setTimeout(() => {
    try {
      const message = {
        type: 'chunk_status',
        chunk_id: chunkId,
        task_id: taskId,
        status: 0  // 0 = chunk arrival/start
      };
      listenerWs.send(JSON.stringify(message));
      log('debug', 'Notified listener of chunk arrival:', chunkId);
    } catch (error) {
      log('error', 'Failed to notify listener of chunk arrival:', error);
    }
  }, 0);
}

function notifyListenerChunkComplete(chunkId, status) {
  if (!enableListener || !listenerWs || listenerWs.readyState !== WebSocket.OPEN) return;

  // Make this asynchronous and non-blocking
  setTimeout(() => {
    try {
      const isSuccess = (status === 'completed' || status === 'ok');
      const message = {
        type: 'chunk_status',
        chunk_id: chunkId,
        status: isSuccess ? 1 : -1  // 1 = success, -1 = error
      };
      listenerWs.send(JSON.stringify(message));
      log('debug', 'Notified listener of chunk completion:', chunkId, 'status:', status, 'isSuccess:', isSuccess);
    } catch (error) {
      log('error', 'Failed to notify listener of chunk completion:', error);
    }
  }, 0);
}

socket.on('connect', ()=>{
  statusEl.textContent = 'connected';
  log('info', 'Connected', socket.id);
  const frameworks = ['cuda','vulkan','opencl','cpp-wasm'];
  if('gpu' in navigator) frameworks.push('webgpu');
  const capacity = Number(qs.cap || qs.capacity || 1);
  socket.emit('hello', { workerId: qs.workerId || socket.id, frameworks, capacity });

  // Connect to listener if enabled
  if (enableListener) {
    connectToListener();
  }
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

  // Notify listener of chunk arrival
  notifyListenerChunkArrival(chunkId, taskId);

  let exec = executors.get(taskId);
  log('debug', 'Retrieved executor for task', taskId, 'exec:', exec);
  if(!exec){
    log('warn', 'No executor for task', taskId);
    socket.emit('chunk:result', { taskId, chunkId, replica, status: 'no-exec' });
    notifyListenerChunkComplete(chunkId, 'no-exec');
    return;
  }

  // If this is the placeholder (has __ready__), wait for initialization.
  if (exec && exec.__ready__ instanceof Promise) {
    log('debug', 'Waiting for executor to become ready for task', taskId);
    try {
      exec = await exec.__ready__;
      log('debug', 'Executor ready, result:', exec);
    } catch (e) {
      log('error', 'Executor ready promise rejected:', e);
      exec = null;
    }
    if(!exec){
      log('warn', 'Executor failed to initialize for task', taskId);
      socket.emit('chunk:result', { taskId, chunkId, replica, status: 'no-exec' });
      notifyListenerChunkComplete(chunkId, 'no-exec');
      return;
    }
  }

  try{
    log('debug', 'Starting chunk execution for', chunkId);
    const res = await exec.runChunk({ payload, meta });
    log('debug', 'Chunk execution completed, result:', res);
    const checksum = await checksumHex(res.result);
    socket.emit('chunk:result', {
      taskId, chunkId, replica, status: res.status, checksum,
      result: res.result, timings: res.timings
    });
    log('debug', 'chunk done', chunkId, 'cs', checksum.slice(0,8));

    // Notify listener of successful completion
    log('debug', 'Chunk completed successfully, status:', res.status);
    notifyListenerChunkComplete(chunkId, res.status);
  }catch(e){
    log('error', 'chunk failed', chunkId, e.message, e.stack);
    socket.emit('chunk:result', { taskId, chunkId, replica, status: 'error', error: e.message });

    // Notify listener of error
    notifyListenerChunkComplete(chunkId, 'error');
  }
});