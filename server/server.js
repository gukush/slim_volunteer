import fs from 'fs';
import http from 'http';
import https from 'https';
import path from 'path';
import express from 'express';
import { Server as SocketIOServer } from 'socket.io';
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

const io = new SocketIOServer(server, {
  cors: { origin: true, methods: ['GET','POST'] },
  maxHttpBufferSize: 1e8,
});

const tm = new TaskManager({ io, storageDir: path.join(process.cwd(), 'storage'), timingDir: path.join(process.cwd(), 'storage', 'timing') });

io.on('connection', (socket)=>{
  socket.on('hello', (info)=>tm.registerClient(socket, info||{}));
  socket.on('disconnect', ()=>tm.removeClient(socket.id));
  socket.on('chunk:result', (data)=>tm.receiveResult(socket.id, data));
});

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
      // Debug logging
      console.log('Received JSON request, body:', req.body);

      const { strategyId, input, label, K = 1, config = {} } = req.body || {};

      if (!strategyId) {
        return res.status(400).json({ error: 'strategyId is required' });
      }

      getStrategy(strategyId); // Validate strategy exists

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

  const bb = Busboy({ headers: req.headers });
  const files = [];
  const fields = {};
  bb.on('file', (name, file, info)=>{
    const { filename } = info;
    const chunks = [];
    file.on('data', (d)=>chunks.push(d));
    file.on('end', ()=>files.push({ originalName: filename, buffer: Buffer.concat(chunks) }));
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
  logger.info('Open https://localhost:3000/ in a WebGPU/WebGL2-enabled Chrome.');
});
