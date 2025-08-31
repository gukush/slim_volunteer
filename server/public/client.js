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

const socket = io({ transports:['websocket'], auth:{ token: 'anon' } });
const executors = new Map(); // taskId -> executor

socket.on('connect', ()=>{
  statusEl.textContent = 'connected';
  log('info', 'Connected', socket.id);
  const frameworks = [];
  if('gpu' in navigator) frameworks.push('webgpu');
  socket.emit('hello', { workerId: qs.workerId || socket.id, frameworks });
});

socket.on('disconnect', ()=>{
  statusEl.textContent = 'disconnected';
  log('warn', 'Disconnected');
});

socket.on('task:init', async (msg)=>{
  log('info', 'task:init', JSON.stringify({ taskId: msg.taskId, strategyId: msg.strategyId }));
  if(msg.framework==='webgpu' && !('gpu' in navigator)){
    log('warn', 'No WebGPU; ignoring task', msg.taskId);
    return;
  }
  const blob = new Blob([msg.executorCode], { type: 'text/javascript' });
  const modUrl = URL.createObjectURL(blob);
  const mod = await import(modUrl);
  const exec = mod.createExecutor({ kernels: msg.kernels, config: msg.config, schema: msg.schema, inputArgs: msg.inputArgs });
  executors.set(msg.taskId, exec);
});

socket.on('chunk:assign', async (job)=>{
  const { taskId, chunkId, replica, payload, meta, tCreate } = job;
  const exec = executors.get(taskId);
  if(!exec){
    log('warn', 'No executor for task', taskId);
    socket.emit('chunk:result', { taskId, chunkId, replica, status: 'no-exec' });
    return;
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
