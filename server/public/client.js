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

// ── HTTP Data Plane helpers ──────────────────────────────────────────

// Fetch packed binary payload via HTTP and unpack into a payload object
// Wire format: [4-byte header len LE][JSON header][binary0][binary1]...
// The header's __binaryKeys__ array describes where each binary blob goes.
async function fetchPayload(taskId, chunkId) {
  const resp = await fetch(`/chunks/${taskId}/${chunkId}/payload`);
  if (!resp.ok) throw new Error(`Payload fetch failed: ${resp.status}`);
  const ab = await resp.arrayBuffer();
  const view = new DataView(ab);
  const headerLen = view.getUint32(0, true);
  const headerBytes = new Uint8Array(ab, 4, headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));

  const binaryKeys = header.__binaryKeys__ || [];
  delete header.__binaryKeys__;

  let offset = 4 + headerLen;
  const payload = { ...header };

  for (const entry of binaryKeys) {
    if (entry.sizes) {
      // Array of binary blobs (legacy { buffers: [...] } format)
      const arr = [];
      for (const sz of entry.sizes) {
        arr.push(ab.slice(offset, offset + sz));
        offset += sz;
      }
      payload[entry.key] = arr;
    } else {
      // Single binary value at a named key (e.g. payload.a, payload.b)
      payload[entry.key] = ab.slice(offset, offset + entry.size);
      offset += entry.size;
    }
  }

  return payload;
}

// Upload binary result via HTTP POST (metadata in headers)
async function uploadResult(taskId, chunkId, { replica, status, checksum, result, timings }) {
  const resp = await fetch(`/chunks/${taskId}/${chunkId}/result`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
      'X-Socket-Id': socket.id,
      'X-Replica': String(replica),
      'X-Status': status,
      'X-Checksum': checksum,
      'X-Timings': JSON.stringify(timings || {}),
    },
    body: result,
  });
  if (!resp.ok) throw new Error(`Result upload failed: ${resp.status}`);
  return resp.json();
}

// notifyListenerMetrics function removed - metrics are now sent directly from server to listener

socket.on('connect', async ()=>{
  statusEl.textContent = 'connected';
  log('info', 'Connected', socket.id);
  const frameworks = ['cpp-wasm','webgl2'];
  let gpuInfo = null;

  if('gpu' in navigator) {
    frameworks.push('webgpu');
    try {
      // Collect GPU adapter information
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        gpuInfo = {
          vendor: adapter.info?.vendor || 'unknown',
          architecture: adapter.info?.architecture || 'unknown',
          device: adapter.info?.device || 'unknown',
          description: adapter.info?.description || 'unknown',
          // Check for SwiftShader (software renderer)
          isSwiftShader: adapter.info?.vendor?.toLowerCase().includes('swiftshader') ||
                        adapter.info?.description?.toLowerCase().includes('swiftshader') ||
                        adapter.info?.device?.toLowerCase().includes('swiftshader')
        };
        log('info', 'GPU Info collected:', gpuInfo);
      }
    } catch (error) {
      log('warn', 'Failed to collect GPU info:', error);
    }
  }

  const capacity = Number(qs.cap || qs.capacity || 1);
  socket.emit('hello', {
    workerId: qs.workerId || socket.id,
    frameworks,
    capacity,
    gpuInfo
  });

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
      URL.revokeObjectURL(modUrl); // free Blob memory — module is already loaded
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
      // Tell server we're ready so it can drain pending chunks to us
      socket.emit('worker:ready', { taskId: msg.taskId });
    }catch(e){
      log('error', 'task:init failed', e);
      // Remove the executor mapping and resolve with null so awaiting
      // chunk handlers get notified of the failure.
      executors.delete(msg.taskId);
      resolveReady(null);
    }
  })();
});

// Clean up executor reference when task finishes (prevents unbounded memory growth).
// Do NOT destroy the WebGPU device — that causes slow re-init on the next task
// and can leave workers unable to acquire the GPU. Let GC reclaim it naturally.
socket.on('task:done', (msg)=>{
  const taskId = msg.taskId;
  if (executors.has(taskId)) {
    executors.delete(taskId);
    log('info', 'Released executor reference for task', taskId);
  }
  readyTasks.delete(taskId);
});

socket.on('chunk:assign', async (job)=>{
  const { taskId, chunkId, replica, payload, payloadDescriptor, meta, tCreate } = job;

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

  // Resolve the actual payload: HTTP fetch if descriptor-only, else inline (backward compat)
  let actualPayload = payload;
  if (!actualPayload && payloadDescriptor) {
    try {
      log('debug', 'Fetching payload via HTTP for chunk', chunkId);
      actualPayload = await fetchPayload(taskId, chunkId);
      log('debug', 'HTTP payload fetched for chunk', chunkId);
    } catch (e) {
      log('error', 'HTTP payload fetch failed for chunk', chunkId, e.message);
      socket.emit('chunk:result', { taskId, chunkId, replica, status: 'error', error: 'payload-fetch-failed: ' + e.message });
      notifyListenerChunkComplete(chunkId, 'error');
      return;
    }
  }

  try{
    log('debug', 'Starting chunk execution for', chunkId);
    const tClientRecvAbs = Date.now();
    const res = await exec.runChunk({ payload: actualPayload, meta });
    const tClientDoneAbs = Date.now();
    log('debug', 'Chunk execution completed, result:', res);
    const checksum = await checksumHex(res.result);

    // Enrich timings with absolute machine-local epoch timestamps (ms)
    // so they can be directly joined with listener power logs on the same machine.
    res.timings = res.timings || {};
    res.timings.tClientRecvAbs = tClientRecvAbs;
    res.timings.tClientDoneAbs = tClientDoneAbs;

    // Try HTTP POST first (avoids blocking Socket.IO event loop with large binary)
    let httpOk = false;
    try {
      await uploadResult(taskId, chunkId, {
        replica, status: res.status, checksum,
        result: res.result, timings: res.timings,
      });
      httpOk = true;
      log('debug', 'chunk done (HTTP)', chunkId, 'cs', checksum.slice(0,8));
    } catch (httpErr) {
      log('warn', 'HTTP result upload failed, falling back to Socket.IO', httpErr.message);
    }

    // Fallback: send result via Socket.IO if HTTP failed
    if (!httpOk) {
      socket.emit('chunk:result', {
        taskId, chunkId, replica, status: res.status, checksum,
        result: res.result, timings: res.timings
      });
      log('debug', 'chunk done (Socket.IO fallback)', chunkId, 'cs', checksum.slice(0,8));
    }

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

// Note: Metrics events are now sent directly from server to listener via WebSocket
// No need to forward them through the browser client anymore