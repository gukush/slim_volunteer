import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { getStrategy } from './StrategyRegistry.js';
import { logger } from './logger.js';
import { ensureDir, now, writeJSON } from './utils.js';
import { TaskTimers } from './metrics.js';
import { OSUsageTracker } from './osMetrics.js';

const OS_METRICS_INTERVAL_MS = Number(process.env.OS_METRICS_INTERVAL_MS || 1000);
const OS_METRICS_MOUNT      = process.env.OS_METRICS_MOUNT || '/';

export class TaskManager{
  constructor({io, wss, storageDir, timingDir}){
    this.io = io;
    this.wss = wss;
    this.storageDir = storageDir;
    this.timingDir = timingDir;
    ensureDir(storageDir);
    ensureDir(timingDir);
    this.clients = new Map();
    this.tasks = new Map();
  }

  registerClient(socket, hello, clientType = 'browser'){
    const capacity = Number(hello?.capacity ?? 1) || 1;
    const info = { socket, busy: false, tasks: new Set(), capacity, inFlight: 0, clientType, ...hello };
    this.clients.set(socket.id, info);
    logger.info('Client connected', socket.id, { ...hello, clientType });
    // Try to schedule pending work now that capacity increased
    for(const task of this.tasks.values()){
      if(task.status==='running') this._drainTaskQueue(task);
    }
  }

  removeClient(socketId){
    const c = this.clients.get(socketId);
    if(!c) return;
    logger.info('Client disconnected', socketId);
    for(const taskId of c.tasks){
      const task = this.tasks.get(taskId);
      if(!task) continue;
      for(const [chunkId, ass] of task.assignments){
        if(ass.assignedTo.has(socketId) && !ass.completed){
          ass.assignedTo.delete(socketId);
          this._assignChunkReplica(task, chunkId);
        }
      }
    }
    this.clients.delete(socketId);
  }

  createTask({strategyId, K=1, label='task', config={}, inputArgs={}, inputFiles=[]}){
    const strategy = getStrategy(strategyId);
    const id = uuidv4();
    const taskDir = path.join(this.storageDir, 'tasks', id);
    ensureDir(taskDir);
    const descriptor = { id, label, strategyId, status: 'created', createdAt: now(), K, config, inputArgs, inputFiles: [] };
    /*
    for(const f of inputFiles){
      const dest = path.join(taskDir, f.originalName);
      fs.writeFileSync(dest, f.buffer);
      descriptor.inputFiles.push({ path: dest, originalName: f.originalName, size: f.buffer.length });
    }
    */
    for (const f of inputFiles){
      const dest = path.join(taskDir, f.originalName || path.basename(f.path));
      if (f.path) {
        fs.copyFileSync(f.path, dest);              // streamless copy on same disk; or use streams if cross-device
        const size = f.size ?? fs.statSync(dest).size;
        descriptor.inputFiles.push({ path: dest, originalName: path.basename(dest), size });
      } else if (f.buffer) {
        fs.writeFileSync(dest, f.buffer);           // fallback for small files only
        descriptor.inputFiles.push({ path: dest, originalName: path.basename(dest), size: f.buffer.length });
      } else {
        throw new Error('input file missing buffer/path');
      }
      // NOTE: your original snippet appended a second time here; left as-is if intentional.
      const size = f.size ?? fs.statSync(dest).size;
      descriptor.inputFiles.push({ path: dest, originalName: f.originalName, size });
    }
    writeJSON(path.join(taskDir, `task_descriptor_${id}.json`), descriptor);

    const chunker = strategy.buildChunker({ taskId: id, taskDir, K, config, inputArgs, inputFiles: descriptor.inputFiles });
    const assembler = strategy.buildAssembler({ taskId: id, taskDir, K, config, inputArgs });
    const timers = new TaskTimers(id, path.join(this.storageDir, 'timing'));

    const task = {
      id, descriptor, strategy, chunker, assembler, timers,
      status: 'created', startTime: null, endTime: null, K,
      queue: [], assignments: new Map(),
      totalChunks: null, completedChunks: 0,
      cancelRequested: false, chunkerFinished: false,
      framework: null,
      osTracker: null,
    };
    this.tasks.set(id, task);
    logger.info('Task created', id, strategyId, 'K=', K);
    return descriptor;
  }

  getTask(id){ return this.tasks.get(id); }

  async startTask(id) {
    const task = this.getTask(id);
    if (!task) throw new Error('No such task');
    if (task.status !== 'created' && task.status !== 'paused') {
      throw new Error('Task not in a startable state');
    }

    try {
      task.status = 'running';
      task.startTime = now();

      // OSUsageTracker: start per-task sampling
      const osOutDir = path.join(this.storageDir, 'timing', task.id);
      ensureDir(osOutDir);
      task.osTracker = new OSUsageTracker({
        taskId: task.id,
        outDir: osOutDir,
        intervalMs: OS_METRICS_INTERVAL_MS,
        mountPoint: OS_METRICS_MOUNT,
        pidList: [process.pid], // add more PIDs if you spawn workers
      });
      await task.osTracker.start();

      const execInfo = task.strategy.getClientExecutorInfo(task.descriptor.config, task.descriptor.inputArgs);
      task.framework = execInfo.framework;

      // Send task initialization to clients
      const nativeClients = Array.from(this.clients.values()).filter(c => c.clientType === 'native');
      const browserClients = Array.from(this.clients.values()).filter(c => c.clientType !== 'native');

      logger.info(`Starting task ${id}: ${browserClients.length} browser clients, ${nativeClients.length} native clients`);

      if (browserClients.length > 0) {
        if (!execInfo.path) {
          throw new Error(`Strategy ${task.strategy.id} does not provide browser executor path`);
        }
        const executorCode = fs.readFileSync(path.join(process.cwd(), execInfo.path), 'utf-8');
        const kernels = (execInfo.kernels || []).map(p => ({
          name: p,
          content: fs.readFileSync(path.join(process.cwd(), p), 'utf-8')
        }));

        this.io.emit('task:init', {
          taskId: id,
          strategyId: task.strategy.id,
          framework: execInfo.framework,
          executorCode,
          kernels,
          schema: execInfo.schema || {},
          config: task.descriptor.config,
          inputArgs: task.descriptor.inputArgs,
        });
      }

      for (const client of nativeClients) {
        this._sendToNativeClient(client.socket, 'workload:new', {
          id: id,
          strategyId: task.strategy.id,
          framework: execInfo.framework,
          kernels: execInfo.kernels || [],
          config: task.descriptor.config,
          inputArgs: task.descriptor.inputArgs,
          schema: execInfo.schema || {},
          type: 'computation'
        });
      }

      // Start chunking with comprehensive error handling
      this._startChunkingProcess(task).catch(e => {
        logger.error(`Chunking process failed for task ${id}:`, e);
        task.status = 'error';
        task.error = e.message;

        try { task.osTracker?.stop('error'); } catch {}

        // Notify all clients of task failure
        this.io.emit('task:error', { taskId: id, error: e.message });
        for (const client of nativeClients) {
          this._sendToNativeClient(client.socket, 'workload:error', {
            id: id,
            message: e.message
          });
        }
      });

    } catch (error) {
      task.status = 'error';
      task.error = error.message;

      // OSUsageTracker: stop on start failure
      try { task.osTracker?.stop('error'); } catch {}

      logger.error(`Failed to start task ${id}:`, error);
      throw error;
    }
  }

  getWorkloadHeader(taskId, { includeArtifacts = false } = {}) {
    const task = this.tasks.get(taskId);
    if (!task) throw new Error(`Unknown task: ${taskId}`);
    const header = {
      taskId: task.id,
      framework: task.clientInfo?.framework || null,
      schema: task.clientInfo?.schema || null,
      kernels: task.clientInfo?.kernels || [],
    };
    if (includeArtifacts && Array.isArray(task.artifacts) && task.artifacts.length) {
      header.artifacts = task.artifacts;
    }
    return header;
  }

  async _startChunkingProcess(task) {
    let count = 0;
    const BATCH_SIZE = 50;  // Smaller batches for better responsiveness
    const YIELD_INTERVAL = 25; // Yield more frequently
    const MAX_PENDING = 5000; // Limit pending chunks

    let batch = 0;
    const startTime = Date.now();
    let lastLogTime = startTime;

    try {
      for await (const chunk of task.chunker.stream()) {
        if (task.cancelRequested) {
          logger.info(`Task ${task.id}: Chunking cancelled at ${count.toLocaleString()} chunks`);
          break;
        }

        const entry = {
          results: new Map(),
          assignedTo: new Set(),
          completed: false,
          replicas: 0,
          payload: chunk.payload,
          meta: chunk.meta || {},
          tCreate: chunk.tCreate || now(),
        };

        task.assignments.set(chunk.id, entry);
        task.queue.push(chunk.id);
        task.timers.chunkRow({ chunkId: chunk.id, replica: -1, tCreate: entry.tCreate });

        // Try to assign immediately
        this._assignChunkReplica(task, chunk.id);
        count++;
        batch++;

        // Periodic logging and yielding
        const nowMs = Date.now();
        if (nowMs - lastLogTime > 10000) { // Every 10 seconds
          const rate = count / ((nowMs - startTime) / 1000);
          logger.info(`Task ${task.id}: Generated ${count.toLocaleString()} chunks (${rate.toFixed(1)} chunks/sec)`);
          lastLogTime = nowMs;
        }

        // Yield control to event loop frequently
        if (batch >= YIELD_INTERVAL) {
          batch = 0;
          await new Promise(resolve => setImmediate(resolve));
        }

        // Throttle if too many chunks are pending
        const pending = task.assignments.size - task.completedChunks;
        while (pending > MAX_PENDING && !task.cancelRequested) {
          logger.debug(`Task ${task.id}: Throttling chunk generation (${task.queue.length} pending)`);
          await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Memory pressure check
        if (count % 1000 === 0) {
          const memUsage = process.memoryUsage();
          const heapUsedMB = Math.round(memUsage.heapUsed / 1024 / 1024);
          if (heapUsedMB > 2048) { // Over 2GB heap usage
            logger.warn(`Task ${task.id}: High memory usage: ${heapUsedMB}MB heap`);
          }
        }
      }

      task.totalChunks = count;
      task.chunkerFinished = true;

      const duration = (Date.now() - startTime) / 1000;
      const rate = count / duration;
      logger.info(`Task ${task.id}: Chunking completed - ${count.toLocaleString()} chunks in ${duration.toFixed(1)}s (${rate.toFixed(1)} chunks/sec)`);

      this._maybeFinish(task.id);

    } catch (error) {
      logger.error(`Chunking stream error for task ${task.id}:`, error);
      task.status = 'error';
      task.error = error.message;

      // OSUsageTracker: stop on error
      try { task.osTracker?.stop('error'); } catch {}

      throw error;
    }
  }

  _eligibleClients(task){
    // Only clients that support the framework
    const list = Array.from(this.clients.values()).filter(c=>{
      if(task.framework && c.frameworks){
        return c.frameworks.includes(task.framework);
      }
      return true;
    });
    // Prefer clients with available capacity and lower load
    return list.sort((a,b)=>{
      const la = a.inFlight / (a.capacity||1);
      const lb = b.inFlight / (b.capacity||1);
      return la - lb;
    });
  }

  _assignChunkReplica(task, chunkId){
    let assigned = false;
    const entry = task.assignments.get(chunkId);
    if(!entry || entry.completed) return;
    if(entry.replicas >= task.K) return;

    // Check if we allow same client multiple replicas (for research/testing)
    const allowSameClient = task.descriptor.config?.allowSameClientReplicas || false;

    for(const c of this._eligibleClients(task)){
      // Skip if client already assigned to this chunk AND we don't allow same client replicas
      if(!allowSameClient && entry.assignedTo.has(c.socket.id)) continue;

      if((c.inFlight||0) >= (c.capacity||1)) continue; // respect client capacity

      const replica = entry.replicas;
      entry.assignedTo.add(c.socket.id);
      entry.replicas++;
      c.inFlight = (c.inFlight||0) + 1;
      c.tasks.add(task.id);
      assigned = true;

      // Emit chunk assignment with replica ID
      c.socket.emit('chunk:assign', {
        taskId: task.id,
        chunkId,
        replica,
        payload: entry.payload,
        meta: entry.meta,
        tCreate: entry.tCreate,
      });

      const tSent = now();
      task.timers.chunkRow({ chunkId, replica, tCreate: entry.tCreate, tSent });

      if(entry.replicas >= task.K) break;
    }
    return assigned;
  }

  receiveResult(socketId, data){
    const { taskId, chunkId, replica, status, checksum, result, timings } = data;
    const task = this.tasks.get(taskId);
    if(!task){ logger.warn('Result for unknown task', taskId); return; }
    const entry = task.assignments.get(chunkId);
    if(!entry){ logger.warn('Result for unknown chunk', chunkId); return; }

    const tServerRecv = now();
    const client = this.clients.get(socketId);
    if (client) client.inFlight = Math.max(0, (client.inFlight||0)-1);

    task.timers.chunkRow({
      chunkId, replica,
      tCreate: entry.tCreate,
      tServerRecv,
      tClientRecv: timings?.tClientRecv,
      tClientDone: timings?.tClientDone,
    });

    if(status!=='ok'){
      logger.warn('Replica failed', taskId, chunkId, replica, 'from', socketId);
      // For same-client replicas, we might want to reassign differently
      const allowSameClient = task.descriptor.config?.allowSameClientReplicas || false;
      if (!allowSameClient) {
        entry.assignedTo.delete(socketId);
      }
      this._assignChunkReplica(task, chunkId);
      this._drainTaskQueue(task);
      return;
    }

    let rec = entry.results.get(checksum);
    if (!rec) {
      entry.results.set(checksum, { count: 1, buf: result });
    } else {
      rec.count += 1;
    }


    for (const [cs, rec2] of entry.results){
      if (rec2.count >= task.K){
        try{
          task.assembler.integrate({ chunkId, result: rec2.buf, meta: entry.meta });
          const tAssembled = now();
          task.timers.chunkRow({ chunkId, replica, tCreate: entry.tCreate, tAssembled });
          entry.completed = true;
          task.completedChunks += 1;
          logger.debug('Chunk accepted', task.id, chunkId, 'checksum', cs);
          this._maybeFinish(task.id);
          this._drainTaskQueue(task);
        }catch(e){
          logger.error('Assembler integrate error', e);
        }
        return;
      }
    }
  }

  _maybeFinish(taskId){
    const task = this.tasks.get(taskId);
    if(!task) return;
    const allDone = task.chunkerFinished && [...task.assignments.values()].every(v=>v.completed);
    if(allDone && task.status==='running'){
      task.status = 'assembling';
      try{
        const outInfo = task.assembler.finalize();
        task.status = 'completed';
        task.endTime = now();
        task.timers.endSummary(path.join(this.storageDir, 'timing', 'task_summaries.csv'), 'completed');

        // OSUsageTracker: stop on success
        try { task.osTracker?.stop('completed'); } catch {}

        this.io.emit('task:done', { taskId, outInfo });
        logger.info('Task completed', taskId, outInfo);
      }catch(e){
        task.status = 'error';

        // OSUsageTracker
        try { task.osTracker?.stop('error'); } catch {}

        logger.error('Finalize error', e);
      }
    }
  }

  cancelTask(id){
    const task = this.tasks.get(id);
    if(!task) return false;
    task.cancelRequested = true;
    task.status = 'canceled';
    task.timers.endSummary(path.join(this.storageDir, 'timing', 'task_summaries.csv'), 'canceled');

    // SUsageTracker
    try { task.osTracker?.stop('canceled'); } catch {}

    this.tasks.delete(id);
    return true;
  }

  statusTask(id){
    const t = this.tasks.get(id);
    if(!t) return null;
    return {
      id: t.id,
      status: t.status,
      K: t.K,
      createdAt: t.descriptor.createdAt,
      startTime: t.startTime,
      completedChunks: t.completedChunks,
      totalChunks: t.totalChunks,
      strategyId: t.strategy.id,
      label: t.descriptor.label,
    };
  }

  _drainTaskQueue(task){
    if(!task || task.status!=='running') return;
    // Iterate over queued chunks and try to assign replicas
    for(const chunkId of task.queue){
      const entry = task.assignments.get(chunkId);
      if(!entry || entry.completed) continue;
      // Try to assign as many replicas as needed (up to K)
      while(entry.replicas < task.K){
        const assigned = this._assignChunkReplica(task, chunkId);
        if(!assigned) break; // No capacity right now
      }
    }
  }
}
