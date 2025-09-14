import fs from 'fs';
import http from 'http';
import https from 'https';
import path from 'path';
import express from 'express';
import { Server as SocketIOServer } from 'socket.io';
import { WebSocketServer } from 'ws';
import Busboy from 'busboy';
import { logger } from './lib/logger.js';
import { listStrategies, getStrategy } from './lib/StrategyRegistry.js';
import { TaskManager } from './lib/TaskManager.js';

const HOST = process.env.HOST || '0.0.0.0';
const PORT = Number(process.env.PORT || 3000);
const ALLOW_INSECURE = process.env.ALLOW_INSECURE === '1';
const STORAGE_DIR = process.env.STORAGE_DIR || 'storage';

const app = express();
app.disable('x-powered-by');

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

app.use('/public', express.static(path.join(process.cwd(), 'public')));
app.get('/', (req,res)=>res.sendFile(path.join(process.cwd(),'public','index.html')));

app.get('/health', (req,res)=>res.json({ ok: true }));
app.get('/strategies', (req,res)=>res.json({ strategies: listStrategies() }));

// Endpoint to list cached files in uploads directory
app.get('/uploads', (req, res) => {
  try {
    const uploadsDir = path.join(process.cwd(), STORAGE_DIR, 'uploads');
    if (!fs.existsSync(uploadsDir)) {
      return res.json({ files: [] });
    }

    const files = fs.readdirSync(uploadsDir).map(filename => {
      const fullPath = path.join(uploadsDir, filename);
      const stats = fs.statSync(fullPath);
      return {
        filename,
        size: stats.size,
        created: stats.birthtime,
        modified: stats.mtime
      };
    }).sort((a, b) => b.modified - a.modified); // Sort by most recent first

    res.json({ files });
  } catch (e) {
    logger.error('Error listing uploads:', e);
    res.status(500).json({ error: e.message });
  }
});

const server = (()=>{
  try{
    if(ALLOW_INSECURE) throw new Error('insecure forced');
    const key = fs.readFileSync(path.join(process.cwd(), 'certificates', 'key.pem'));
    const cert = fs.readFileSync(path.join(process.cwd(), 'certificates', 'cert.pem'));
    logger.info('Starting HTTPS server with certificates');
    return https.createServer({ key, cert }, app);
  }catch(e){
    logger.warn('Starting HTTP server (certs missing or ALLOW_INSECURE=1)', e.message);
    return http.createServer(app);
  }
})();

// Socket.IO for browser clients (keep original configuration)
const io = new SocketIOServer(server, {
  cors: { origin: true, methods: ['GET','POST'] },
  maxHttpBufferSize: 3e8,
  pingTimeout: 60000
});

// Raw WebSocket server for native clients - IMPORTANT: different server instance
const wss = new WebSocketServer({
  port: PORT + 1, // Use different port to avoid conflicts
  path: '/ws-native',
  maxPayload: 1e9
});

// Add debugging for WebSocket server events
wss.on('listening', () => {
  logger.info('🚀 WebSocket server is listening on port', PORT + 1);
});

wss.on('error', (error) => {
  logger.error('❌ WebSocket server error:', error);
});

logger.info(`Socket.IO will be available on port ${PORT}`);
logger.info(`Native WebSocket server will be available on port ${PORT + 1}`);

const tm = new TaskManager({
  io,
  wss,
  storageDir: path.join(process.cwd(), STORAGE_DIR),
  timingDir: path.join(process.cwd(), STORAGE_DIR, 'timing')
});

// Socket.IO handlers (unchanged - this is what your browser uses)
io.on('connection', (socket)=>{
  logger.info('Browser client connected via Socket.IO:', socket.id);
  socket.on('hello', (info)=>{
    // Log GPU information if available
    if (info.gpuInfo) {
      const gpuInfo = info.gpuInfo;
      logger.info('🖥️  GPU Information from browser client:', {
        clientId: socket.id,
        vendor: gpuInfo.vendor,
        architecture: gpuInfo.architecture,
        device: gpuInfo.device,
        description: gpuInfo.description,
        isSwiftShader: gpuInfo.isSwiftShader
      });

      // Special warning for SwiftShader
      if (gpuInfo.isSwiftShader) {
        logger.warn('⚠️  WARNING: Client is using SwiftShader (software renderer) - performance will be significantly reduced');
      } else {
        logger.info('✅ Client is using hardware GPU acceleration');
      }
    } else {
      logger.info('ℹ️  No GPU information available from client:', socket.id);
    }

    tm.registerClient(socket, info||{}, 'socketio');
  });
  socket.on('disconnect', ()=>tm.removeClient(socket.id));
  socket.on('chunk:result', (data)=>tm.receiveResult(socket.id, data));
});

// Raw WebSocket handlers (native clients and listeners)
wss.on('connection', (ws, req) => {
  logger.info('WebSocket connection established');
  logger.info('WebSocket connection details:', { url: req.url, headers: req.headers });

  // Determine connection type based on User-Agent header
  const userAgent = req.headers['user-agent'] || '';
  const isListener = userAgent.includes('Metrics-Listener');

  if (isListener) {
    // Listener connection
    ws.id = `listener_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    ws.kind = 'listener';
    logger.info('Listener connected via WebSocket');
  } else {
    // Native client connection
    ws.id = `native_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    ws.kind = 'native';
    logger.info('Native client connected via WebSocket');
  }

  // Allow multiple connections of both types simultaneously
  // No need to close existing connections - the TaskManager can handle multiple clients
  logger.info(`New ${ws.kind} connection established: ${ws.id}`);

  // Compact send helper (keeps your {type,data} envelope)
  const send = (type, data) => {
    if (ws.readyState === ws.OPEN) {
      const message = JSON.stringify({ type, data });
      logger.debug(`Sending to native client ${ws.id}: ${type}`);
      ws.send(message);
    }
  };

  // Keep your "emit" shim (uses native send under the hood) - but don't override 'message' events
  const originalEmit = ws.emit.bind(ws);
  ws.emit = (event, data) => {
    // Don't send back 'message' events - those are incoming messages from the client
    if (event === 'message') {
      return originalEmit(event, data);
    }
    logger.debug(`WS emit called: ${event} for ${ws.id}`);
    send(event, data);
  };

  // Debug: log when message handler is set up
  logger.debug(`Setting up message handler for ${ws.id}`);

  // --- helper: announce workload (with artifacts) to this native client
  const sendWorkloadNew = (taskId) => {
    try {
      const header = tm.getWorkloadHeader(taskId, { includeArtifacts: true });
      const artifacts = (header.artifacts || []).map(a => ({
        type: a.type,                         // 'binary' (future: 'text', etc.)
        name: a.name,                         // filename
        program: a.program,                   // logical program name
        backend: a.backend,                   // 'opencl' | 'cuda' | 'vulkan'
        exec: !!a.exec,                       // executable hint
        size: Buffer.isBuffer(a.bytes) ? a.bytes.length : (a.bytes?.length || 0),
        encoding: 'base64',
        bytes: Buffer.isBuffer(a.bytes) ? a.bytes.toString('base64') : a.bytes // allow pre-base64
      }));

      send('workload:new', {
        taskId: header.taskId,
        framework: header.framework,          // e.g. 'native-opencl'
        schema: header.schema,                // { order, uniforms, inputs, outputs }
        kernels: header.kernels || [],        // optional FYI
        artifacts                             // <-- shipped here
      });

      // Optional: explicit readiness ping back to client
      send('workload:ready', { taskId: header.taskId });
    } catch (err) {
      logger.error('Failed to build/send workload header for native client:', err);
      send('error', { message: err?.message || String(err) });
    }
  };

  // SINGLE message handler - clean and simple
  ws.on('message', (data) => {
    logger.debug(`🔔 Message handler triggered for ${ws.id}, data length: ${data.length}`);
    try {
      // Handle both string and Buffer data
      let messageStr;
      if (Buffer.isBuffer(data)) {
        messageStr = data.toString('utf8');
      } else {
        messageStr = data.toString();
      }

      logger.info(`📨 ${ws.kind} ${ws.id} message:`, messageStr.substring(0, 100) + '...');

      const message = JSON.parse(messageStr);
      const { type, data: eventData } = message;

      logger.info(`🔍 Parsed message - type: ${type}, data keys:`, Object.keys(eventData || {}));
      logger.debug(`Processing message type: ${type}`);

      // Handle listener connections differently
      if (ws.kind === 'listener') {
        // Listeners don't need to process complex messages, just acknowledge
        logger.info(`Listener ${ws.id} sent message: ${type}`);
        send('message:ack', { received: type });
        return;
      }

      switch (type) {
        case 'client:join': {
          const data = eventData || {};
          // normalize frameworks & strategies
          const frameworks = data.frameworks || data.supportedFrameworks || [];
          const strategies = data.strategies || data.supportedStrategies || [];
          const normalized = { ...data, frameworks, strategies };
          logger.info(`Native client ${ws.id} joining with data:`, normalized);
          tm.registerClient(ws, normalized, 'native');
          send('client:join:ack', { clientId: ws.id });

          // If client already knows which task it wants, announce immediately
          if (eventData?.taskId) {
            logger.info(`Client requested task ${eventData.taskId} on join`);
            sendWorkloadNew(eventData.taskId);
          }
          break;
        }

        // Preferred explicit subscribe for a task's workload header + artifacts
        case 'workload:subscribe': {
          const taskId = eventData?.taskId;
          if (!taskId) {
            logger.warn('workload:subscribe missing taskId');
            return send('error', { message: 'workload:subscribe requires taskId' });
          }
          logger.info(`Client ${ws.id} subscribing to workload ${taskId}`);
          sendWorkloadNew(taskId);
          break;
        }

        // Back-compat: if client asks for a task and provides taskId, treat it like subscribe
        case 'task:request': {
          const taskId = eventData?.taskId;
          if (taskId) {
            logger.info(`Client ${ws.id} requesting task ${taskId}`);
            sendWorkloadNew(taskId);
          }
          break;
        }

        case 'task:complete':
          logger.info(`Client ${ws.id} completed task`);
          tm.receiveTaskResult(ws.id, eventData);
          break;

        case 'workload:done':
          logger.info(`Client ${ws.id} completed workload`);
          tm.receiveWorkloadResult(ws.id, eventData);
          break;

        case 'workload:chunk_done_enhanced':
          logger.debug(`Client ${ws.id} completed chunk`);
          tm.receiveResult(ws.id, eventData);
          break;

        case 'workload:error':
          logger.error(`Client ${ws.id} reported error:`, eventData);
          tm.receiveError(ws.id, eventData);
          break;

        case 'workload:chunk_error':
          logger.error(`Client ${ws.id} reported chunk error:`, eventData);
          tm.receiveChunkError(ws.id, eventData);
          break;

        case 'workload:ready':
          logger.info(`Client ${ws.id} reported workload ready:`, eventData);
          // Client is ready to receive chunks, drain the task queue to assign pending chunks
          const taskId = eventData?.id;
          if (taskId) {
            const task = tm.getTask(taskId);
            if (task && task.status === 'running') {
              logger.info(`Draining task queue for ${taskId} after client ${ws.id} reported ready`);
              tm._drainTaskQueue(task);
            }
          }
          break;

        default:
          logger.warn(`Unknown message type from ${ws.id}: ${type}`);
          send('error', { message: `Unknown message type: ${type}` });
      }
    } catch (e) {
      logger.error(`Error parsing message from ${ws.id}:`, e);
      send('error', { message: 'Failed to parse message' });
    }
  });

  ws.on('close', (code, reason) => {
    logger.info(`${ws.kind} ${ws.id} disconnected:`, code, reason?.toString());
    if (ws.kind === 'native') {
      tm.removeClient(ws.id);
    }
    // Listeners don't need to be removed from task manager
  });

  ws.on('error', (error) => {
    logger.error(`Native WebSocket ${ws.id} error:`, error);
  });

  // Send initial test message after connection established
  setTimeout(() => {
    logger.info(`Sending welcome message to ${ws.id}`);
    send('welcome', {
      message: 'Connected to server',
      clientId: ws.id,
      timestamp: Date.now()
    });
  }, 100);
});

// REST API endpoints (unchanged)
app.get('/tasks/:id', (req, res)=>{
  const s = tm.statusTask(req.params.id);
  if(!s) return res.status(404).json({ error: 'not found' });
  res.json(s);
});

app.get('/tasks', (req, res)=>{
  res.json({ tasks: Array.from(tm.tasks.keys()).map(id=>tm.statusTask(id)) });
});

app.get('/tasks/:id/output', (req,res)=>{
  const t = tm.getTask(req.params.id);
  if(!t) return res.status(404).json({error:'not found'});
  try {
    const taskDir = path.join(process.cwd(), STORAGE_DIR, 'tasks', req.params.id);
    const fileName = req.query.name || 'output.bin';
    const outPath = path.join(taskDir, fileName);

    if(!fs.existsSync(outPath)) return res.status(404).json({error:'output not found'});

    if (fileName.endsWith('.json')) {
      res.setHeader('Content-Type','application/json');
      const data = fs.readFileSync(outPath, 'utf8');
      res.send(JSON.parse(data));
    } else {
      res.setHeader('Content-Type','application/octet-stream');
      fs.createReadStream(outPath).pipe(res);
    }
  } catch (e) {
    res.status(500).json({error:e.message});
  }
});

app.delete('/tasks/:id', (req,res)=>{
  const ok = tm.cancelTask(req.params.id);
  if(!ok) return res.status(404).json({ error: 'not found' });
  res.json({ ok: true });
});

app.post('/tasks', (req, res)=>{
  const contentType = req.headers['content-type'] || '';

  // Handle JSON requests
  if (contentType.includes('application/json')) {
    try {
      const { strategyId, input, label, K = 1, config = {}, cachedFilePaths = [] } = req.body || {};

      if (!strategyId) {
        return res.status(400).json({ error: 'strategyId is required' });
      }

      getStrategy(strategyId);

      // Handle cached file paths
      let inputFiles = [];
      if (cachedFilePaths && cachedFilePaths.length > 0) {
        const uploadsDir = path.join(process.cwd(), STORAGE_DIR, 'uploads');

        for (const cachedPath of cachedFilePaths) {
          const fullPath = path.join(uploadsDir, cachedPath);
          if (fs.existsSync(fullPath)) {
            const stats = fs.statSync(fullPath);
            inputFiles.push({
              path: fullPath,
              originalName: path.basename(cachedPath),
              size: stats.size
            });
            logger.info(`Using cached file: ${cachedPath} (${stats.size} bytes)`);
          } else {
            logger.warn(`Cached file not found: ${cachedPath}`);
          }
        }
      }

      const desc = tm.createTask({
        strategyId,
        K,
        label: label || 'task',
        config,
        inputArgs: input || {},
        inputFiles
      });

      res.json(desc);
    } catch (e) {
      logger.error('Create task error (JSON)', e);
      res.status(400).json({ error: e.message });
    }
    return;
  }

  // Handle multipart/form-data (file uploads)
  const bb = Busboy({ headers: req.headers });
  const files = [];
  const fields = {};

  bb.on('file', (name, file, info)=>{
    const { filename } = info;
    const tmpDir = path.join(process.cwd(), STORAGE_DIR, 'uploads');
    fs.mkdirSync(tmpDir, { recursive: true });
    const tmpPath = path.join(tmpDir, `${Date.now()}_${Math.random().toString(16).slice(2)}_${filename}`);
    const ws = fs.createWriteStream(tmpPath);
    let bytes = 0;
    file.on('data', d => { bytes += d.length; });
    file.pipe(ws);
    ws.on('finish', ()=>{
      files.push({ originalName: filename, path: tmpPath, size: bytes });
    });
  });

  bb.on('field', (name, val)=>{ fields[name] = val; });

  bb.on('finish', ()=>{
    // Use setTimeout to ensure all file processing is complete
    setTimeout(() => {
      logger.info(`Processing task: files.length=${files.length}, fields=${JSON.stringify(fields)}`);
      try{
        const strategyId = fields.strategyId;
        const K = fields.K ? parseInt(fields.K,10) : 1;
        const label = fields.label || 'task';
        const config = fields.config ? JSON.parse(fields.config) : {};
        const inputArgs = fields.inputArgs ? JSON.parse(fields.inputArgs) : {};
        getStrategy(strategyId);
        const desc = tm.createTask({ strategyId, K, label, config, inputArgs, inputFiles: files });
        res.json(desc);
      }catch(e){
        logger.error('Create task error', e);
        res.status(400).json({ error: e.message });
      }
    }, 500); // 500ms delay to ensure file processing completes
  });


  req.pipe(bb);
});

app.post('/tasks/:id/start', async (req, res)=>{
  try{
    await tm.startTask(req.params.id);
    res.json({ ok: true });
  }catch(e){
    res.status(400).json({ error: e.message });
  }
});

server.listen(PORT, HOST, ()=>{
  logger.info(`Server listening on ${HOST}:${PORT}`);
  logger.info('Socket.IO endpoint: /socket.io/ (for browsers)');
  logger.info(`Native WebSocket endpoint: ws${ALLOW_INSECURE ? '' : 's'}://${HOST}:${PORT + 1}/ws-native (for native clients)`);
  logger.info('Open https://localhost:3000/ in a WebGPU/WebGL2-enabled Chrome.');
});