import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { getStrategy } from './StrategyRegistry.js';
import { logger } from './logger.js';
import { ensureDir, now, writeJSON } from './utils.js';
import { TaskTimers } from './metrics.js';

export class TaskManager{
  constructor({io, storageDir, timingDir}){
    this.io = io;
    this.storageDir = storageDir;
    this.timingDir = timingDir;
    ensureDir(storageDir);
    ensureDir(timingDir);
    this.clients = new Map();
    this.tasks = new Map();
  }

  registerClient(socket, hello){
    const capacity = Number(hello?.capacity ?? 1) || 1;
    const info = { socket, busy: false, tasks: new Set(), capacity, inFlight: 0, ...hello };
    this.clients.set(socket.id, info);
    logger.info('Client connected', socket.id, hello);
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
    for(const f of inputFiles){
      const dest = path.join(taskDir, f.originalName);
      fs.writeFileSync(dest, f.buffer);
      descriptor.inputFiles.push({ path: dest, originalName: f.originalName, size: f.buffer.length });
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
    };
    this.tasks.set(id, task);
    logger.info('Task created', id, strategyId, 'K=', K);
    return descriptor;
  }

  getTask(id){ return this.tasks.get(id); }

  async startTask(id){
    const task = this.getTask(id);
    if(!task) throw new Error('No such task');
    if(task.status !== 'created' && task.status !== 'paused'){
      throw new Error('Task not in a startable state');
    }
    task.status = 'running';
    task.startTime = now();

    const execInfo = task.strategy.getClientExecutorInfo(task.descriptor.config, task.descriptor.inputArgs);
    task.framework = execInfo.framework;
    const executorCode = fs.readFileSync(path.join(process.cwd(), execInfo.path), 'utf-8');
    const kernels = (execInfo.kernels||[]).map(p=>({ name: p, content: fs.readFileSync(path.join(process.cwd(), p), 'utf-8') }));

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

    (async ()=>{
      let count = 0;
      for await (const chunk of task.chunker.stream()){
        if(task.cancelRequested) break;
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
        this._assignChunkReplica(task, chunk.id);
        count++;
      }
      task.totalChunks = count;
      task.chunkerFinished = true;
      logger.info('Chunker finished streaming', id, 'count', count);
      this._maybeFinish(id);
    })().catch(e=>{
      logger.error('Chunker stream error', e);
      task.status = 'error';
    });
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
    for(const c of this._eligibleClients(task)){
      if(entry.assignedTo.has(c.socket.id)) continue;
      if((c.inFlight||0) >= (c.capacity||1)) continue; // respect client capacity
      const replica = entry.replicas;
      entry.assignedTo.add(c.socket.id);
      entry.replicas++;
      c.inFlight = (c.inFlight||0) + 1;
      c.tasks.add(task.id);
      assigned = true;
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
    const client = this.clients.get(socketId); if (client) client.inFlight = Math.max(0, (client.inFlight||0)-1);
    task.timers.chunkRow({
      chunkId, replica,
      tCreate: entry.tCreate,
      tServerRecv,
      tClientRecv: timings?.tClientRecv,
      tClientDone: timings?.tClientDone,
    });

    if(status!=='ok'){
      logger.warn('Replica failed', taskId, chunkId, replica, 'from', socketId);
      entry.assignedTo.delete(socketId);
      this._assignChunkReplica(task, chunkId);
      this._drainTaskQueue(task);
      return;
    }

    if(!entry.results.has(checksum)) entry.results.set(checksum, []);
    entry.results.get(checksum).push({ socketId, result });

    for(const [cs, arr] of entry.results){
      if(arr.length >= task.K){
        try{
          task.assembler.integrate({ chunkId, result: arr[0].result, meta: entry.meta });
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
        this.io.emit('task:done', { taskId, outInfo });
        logger.info('Task completed', taskId, outInfo);
      }catch(e){
        task.status = 'error';
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
