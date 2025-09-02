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

const app = express();
app.disable('x-powered-by');

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

app.use('/public', express.static(path.join(process.cwd(), 'public')));
app.get('/', (req,res)=>res.sendFile(path.join(process.cwd(),'public','index.html')));

app.get('/health', (req,res)=>res.json({ ok: true }));
app.get('/strategies', (req,res)=>res.json({ strategies: listStrategies() }));

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
  path: '/ws-native'
});

logger.info(`Socket.IO will be available on port ${PORT}`);
logger.info(`Native WebSocket server will be available on port ${PORT + 1}`);

const tm = new TaskManager({
  io,
  wss,
  storageDir: path.join(process.cwd(), 'storage'),
  timingDir: path.join(process.cwd(), 'storage', 'timing')
});

// Socket.IO handlers (unchanged - this is what your browser uses)
io.on('connection', (socket)=>{
  logger.info('Browser client connected via Socket.IO:', socket.id);
  socket.on('hello', (info)=>tm.registerClient(socket, info||{}, 'socketio'));
  socket.on('disconnect', ()=>tm.removeClient(socket.id));
  socket.on('chunk:result', (data)=>tm.receiveResult(socket.id, data));
});

// Raw WebSocket handlers (native clients only)
wss.on('connection', (ws, req) => {
  logger.info('Native client connected via WebSocket');

  // Generate a unique ID for this WebSocket connection
  ws.id = `native_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  // FIXED: Properly add emit method without breaking native send
  ws.emit = (event, data) => {
    if (ws.readyState === ws.OPEN) {
      const message = { type: event, data: data };
      ws.send(JSON.stringify(message)); // Use native send method
    }
  };

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data.toString());
      const { type, data: eventData } = message;

      logger.debug('Native client message:', type);

      switch (type) {
        case 'client:join':
          tm.registerClient(ws, eventData || {}, 'native');
          break;
        case 'task:request':
          // Handle task requests if needed
          break;
        case 'task:complete':
          tm.receiveTaskResult(ws.id, eventData);
          break;
        case 'workload:done':
          tm.receiveWorkloadResult(ws.id, eventData);
          break;
        case 'workload:chunk_done_enhanced':
          tm.receiveChunkResult(ws.id, eventData);
          break;
        case 'workload:error':
          tm.receiveError(ws.id, eventData);
          break;
        case 'workload:chunk_error':
          tm.receiveChunkError(ws.id, eventData);
          break;
        default:
          logger.warn('Unknown native client message type:', type);
      }
    } catch (e) {
      logger.error('Error parsing native client message:', e);
    }
  });

  ws.on('close', () => {
    logger.info('Native client disconnected');
    tm.removeClient(ws.id);
  });

  ws.on('error', (error) => {
    logger.error('Native WebSocket error:', error);
  });
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
    const taskDir = path.join(process.cwd(),'storage','tasks', req.params.id);
    const outPath = path.join(taskDir,'output.bin');
    if(!fs.existsSync(outPath)) return res.status(404).json({error:'output not found'});
    res.setHeader('Content-Type','application/octet-stream');
    fs.createReadStream(outPath).pipe(res);
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
      const { strategyId, input, label, K = 1, config = {} } = req.body || {};

      if (!strategyId) {
        return res.status(400).json({ error: 'strategyId is required' });
      }

      getStrategy(strategyId);

      const desc = tm.createTask({
        strategyId,
        K,
        label: label || 'task',
        config,
        inputArgs: input || {},
        inputFiles: []
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
    const tmpDir = path.join(process.cwd(), 'uploads');
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