import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { getStrategy } from './StrategyRegistry.js';
import { logger } from './logger.js';
import { ensureDir, now, writeJSON, sleep } from './utils.js';
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

  // Send metrics events to listener WebSocket connections
  sendMetricsToListeners(eventType, data) {
    if (!this.wss) {
      logger.warn('WebSocket server not available for sending metrics');
      return;
    }

    const message = JSON.stringify({ type: eventType, data });
    let listenerCount = 0;
    let totalClients = 0;
    let listenerClients = 0;

    for (const client of this.wss.clients) {
      totalClients++;
      if (client.kind === 'listener') {
        listenerClients++;
        logger.debug(`Found listener client ${client.id}, readyState: ${client.readyState}`);
        if (client.readyState === 1) { // 1 = WebSocket.OPEN
          try {
            client.send(message);
            listenerCount++;
            logger.debug(`Sent ${eventType} to listener ${client.id}`);
          } catch (error) {
            logger.error(`Failed to send metrics to listener ${client.id}:`, error);
          }
        } else {
          logger.warn(`Listener ${client.id} is not open (readyState: ${client.readyState})`);
        }
      }
    }

    logger.info(`Metrics send attempt: ${totalClients} total clients, ${listenerClients} listeners, ${listenerCount} sent`);
    if (listenerCount > 0) {
      logger.info(`Sent ${eventType} to ${listenerCount} listener(s)`);
    } else {
      logger.warn(`No listeners received ${eventType} message`);
    }
  }


createTask({strategyId, K=1, label='task', config={}, inputArgs={}, inputFiles=[], cachedFilePaths=[]}){
  const strategy = getStrategy(strategyId);
  const id = uuidv4();
  const taskDir = path.join(this.storageDir, 'tasks', id);
  ensureDir(taskDir);
  const descriptor = { id, label, strategyId, status: 'created', createdAt: now(), K, config, inputArgs, inputFiles: [], cachedFilePaths };

  for (const f of inputFiles){
    const dest = path.join(taskDir, f.originalName || path.basename(f.path));

    // Check if this is a cached file (already in storage/uploads)
    const uploadsDir = path.join(this.storageDir, 'uploads');
    const isCachedFile = f.path && f.path.startsWith(uploadsDir);

    if (isCachedFile) {
      // For cached files, just reference them directly - NO COPYING
      console.log(`[DEBUG] Using cached file directly: ${f.path} (${f.size} bytes)`);
      descriptor.inputFiles.push({
        path: f.path,  // Use original path directly
        originalName: path.basename(f.path),
        size: f.size
      });
    } else if (f.path) {
      // For uploaded files, copy them (but use async for large files)
      console.log(`[DEBUG] Copying uploaded file: ${f.path} -> ${dest}`);
      fs.copyFileSync(f.path, dest);
      const size = f.size ?? fs.statSync(dest).size;
      descriptor.inputFiles.push({ path: dest, originalName: path.basename(dest), size });
    } else if (f.buffer) {
      // For small buffer files
      fs.writeFileSync(dest, f.buffer);
      descriptor.inputFiles.push({ path: dest, originalName: path.basename(dest), size: f.buffer.length });
    } else {
      throw new Error('input file missing buffer/path');
    }
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

      // Send metrics:start message to listeners immediately when task starts
      this.sendMetricsToListeners('metrics:start', { taskId: task.id });
      logger.info(`Task ${task.id}: Sent metrics:start to listener (task start)`);

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
        // Prepare artifacts for native clients (including binary)
        const artifacts = [];
        if (Array.isArray(execInfo.artifacts) && execInfo.artifacts.length) {
            artifacts.push(...execInfo.artifacts);
        }
        /*
        // Add binary as artifact if specified in config
        if (task.descriptor.config.binary) {
          const binaryPath = task.descriptor.config.binary;
          try {
            const binaryData = fs.readFileSync(binaryPath);
            // Use the program name from config if available, otherwise use the binary filename
            const artifactName = task.descriptor.config.program || path.basename(binaryPath);
            artifacts.push({
              name: artifactName,
              type: 'binary',
              bytes: binaryData.toString('base64'),
              exec: true
            });
            logger.info(`Added binary artifact: ${binaryPath} as ${artifactName} (${binaryData.length} bytes)`);
          } catch (error) {
            logger.error(`Failed to read binary ${binaryPath}:`, error);
          }
        }
        */
        // Dont add files as part of artifacts, data is read via chunks
        /*
        if (task.descriptor.inputFiles && task.descriptor.inputFiles.length > 0) {
          for (const file of task.descriptor.inputFiles) {
            try {
              const fileData = fs.readFileSync(file.path);
              artifacts.push({
                name: file.originalName,
                type: 'input',
                bytes: fileData.toString('base64')
              });
              logger.info(`Added input file artifact: ${file.originalName} (${fileData.length} bytes)`);
            } catch (error) {
              logger.error(`Failed to read input file ${file.path}:`, error);
            }
          }
        }
        */
        this._sendToNativeClient(client.socket, 'workload:new', {
          id: id,
          strategyId: task.strategy.id,
          framework: execInfo.framework,
          kernels: execInfo.kernels || [],
          config: task.descriptor.config,
          inputArgs: task.descriptor.inputArgs,
          schema: execInfo.schema || {},
          type: 'computation',
          artifacts: artifacts
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
    console.log(`[DEBUG] getWorkloadHeader called with taskId: '${taskId}'`);
    const task = this.tasks.get(taskId);
    if (!task) {
      console.log(`[DEBUG] Task ${taskId} not found in tasks map`);
      throw new Error(`Unknown task: ${taskId}`);
    }
    console.log(`[DEBUG] Found task ${taskId}, task.id: '${task.id}'`);
    const header = {
      taskId: task.id,
      framework: task.clientInfo?.framework || null,
      schema: task.clientInfo?.schema || null,
      kernels: task.clientInfo?.kernels || [],
    };
    if (includeArtifacts && Array.isArray(task.artifacts) && task.artifacts.length) {
      header.artifacts = task.artifacts;
    }
    console.log(`[DEBUG] Returning header:`, header);
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

    // Send metrics:start message to listener (1 second advance)
    this.sendMetricsToListeners('metrics:start', { taskId: task.id });
    logger.info(`Task ${task.id}: Sent metrics:start to listener`);
    sleep(500); // allow listener to start collecting metrics
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
        task.timers.chunkRow({
            chunkId: chunk.id, replica: -1,
            tCreate: entry.tCreate
        });

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
    console.log(`[DEBUG] _eligibleClients called for task ${task.id}`);
    console.log(`[DEBUG] Task framework: ${task.framework}`);
    console.log(`[DEBUG] All clients:`, Array.from(this.clients.values()).map(c => ({ id: c.socket.id, frameworks: c.frameworks })));
    // Only clients that support the framework
    const list = Array.from(this.clients.values()).filter(c=>{
      if(task.framework && c.frameworks){
        const supports = c.frameworks.includes(task.framework);
        logger.info(`[DEBUG] Client ${c.socket.id} supports ${task.framework}: ${supports} (frameworks: ${JSON.stringify(c.frameworks)})`);
        return supports;
      }
      return true;
    });
    logger.info(`[DEBUG] Eligible clients after filtering: ${list.length}`);
    // Prefer clients with available capacity and lower load
    return list.sort((a,b)=>{
      const la = a.inFlight / (a.capacity||1);
      const lb = b.inFlight / (b.capacity||1);
      return la - lb;
    });
  }

  _assignChunkReplica(task, chunkId){
    console.log(`[DEBUG] _assignChunkReplica called for task ${task.id}, chunk ${chunkId}`);
    let assigned = false;
    const entry = task.assignments.get(chunkId);
    if(!entry || entry.completed) return;

    // Check if chunk is stuck (assigned but not completed for too long)
    const now = Date.now();
    const stuckTimeout = 30000; // 30 seconds
    if(entry.replicas >= task.K && !entry.completed) {
      const timeSinceAssignment = now - (entry.tCreate || now);
      if(timeSinceAssignment > stuckTimeout) {
        console.log(`[DEBUG] Chunk ${chunkId} is stuck, reassigning...`);
        // Reset client inFlight counts first
        for(const clientId of entry.assignedTo) {
          const client = this.clients.get(clientId);
          if(client) {
            client.inFlight = Math.max(0, (client.inFlight || 0) - 1);
          }
        }
        // Reset the chunk assignment
        entry.replicas = 0;
        entry.assignedTo.clear();
      } else {
        return; // Chunk is assigned but not stuck yet
      }
    }

    if(entry.replicas >= task.K) return;

    // Check if we allow same client multiple replicas (for research/testing)
    const allowSameClient = task.descriptor.config?.allowSameClientReplicas || false;

    console.log(`[DEBUG] About to call _eligibleClients for task ${task.id}`);
    const eligibleClients = this._eligibleClients(task);
    console.log(`[DEBUG] Eligible clients for task ${task.id}: ${eligibleClients.length}`);
    for(const c of eligibleClients){
      console.log(`[DEBUG] Checking client ${c.socket.id}, inFlight: ${c.inFlight||0}, capacity: ${c.capacity||1}, frameworks: ${JSON.stringify(c.frameworks)}`);
      // Skip if client already assigned to this chunk AND we don't allow same client replicas
      if(!allowSameClient && entry.assignedTo.has(c.socket.id)) continue;

      if((c.inFlight||0) >= (c.capacity||1)) continue; // respect client capacity

      const replica = entry.replicas;
      entry.assignedTo.add(c.socket.id);
      entry.replicas++;
      c.inFlight = (c.inFlight||0) + 1;
      c.tasks.add(task.id);
      assigned = true;

      console.log(`[DEBUG] Assigning chunk ${chunkId} to client ${c.socket.id}, replica ${replica}`);
      console.log(`[DEBUG] Client type: ${c.clientType}, has payload: ${!!entry.payload}, has buffers: ${!!(entry.payload && entry.payload.buffers)}`);

      // Convert ArrayBuffers to base64 for native clients (unless already in raw format)
      let serializedPayload = entry.payload;
      if (c.clientType === 'native' && entry.payload && entry.payload.buffers) {
        console.log(`[DEBUG] Converting ${entry.payload.buffers.length} buffers to base64 for native client`);
        serializedPayload = {
          ...entry.payload,
          buffers: entry.payload.buffers.map((buf, i) => {
            if (buf instanceof ArrayBuffer) {
              const base64 = Buffer.from(buf).toString('base64');
              console.log(`[DEBUG] Buffer ${i}: ArrayBuffer(${buf.byteLength}) -> base64(${base64.length})`);
              return base64;
            } else if (Array.isArray(buf)) {
              // Already in raw byte array format, keep as is
              console.log(`[DEBUG] Buffer ${i}: already raw byte array, length: ${buf.length}`);
              return buf;
            }
            console.log(`[DEBUG] Buffer ${i}: not ArrayBuffer or array, type: ${typeof buf}, value:`, buf);
            return buf;
          })
        };
      }

      // Emit chunk assignment with replica ID
      c.socket.emit('chunk:assign', {
        taskId: task.id,
        chunkId,
        replica,
        payload: serializedPayload,
        meta: entry.meta,
        tCreate: entry.tCreate,
      });

      const tSent = Date.now();
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

    // Extract GPU timing data from response
    const cpuTimeMs = timings?.cpuTimeMs || null;
    const gpuTimeMs = timings?.gpuTimeMs || null;

    task.timers.chunkRow({
      chunkId, replica,
      tCreate: entry.tCreate,
      tServerRecv,
      tClientRecv: timings?.tClientRecv,
      tClientDone: timings?.tClientDone,
      cpuTimeMs,
      gpuTimeMs,
    });

    if(status!=='ok'){
      logger.warn('Replica failed', taskId, chunkId, replica, 'from', socketId, 'error:', data?.error);
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
      // For native-block-matmul-flex, process each chunk result immediately
      // The assembler handles K accumulation internally
      console.log(`[TASKMANAGER] Processing chunk ${chunkId}, count: ${rec2.count}, K: ${task.K}`);
      if (rec2.count >= task.K){
        try{
          console.log(`[TASKMANAGER] Calling assembler.integrate for chunk ${chunkId}`);
          // Decode base64 data for native clients
          let resultData = rec2.buf;
          console.log(`[TASKMANAGER] Result data type: ${typeof resultData}, isArray: ${Array.isArray(resultData)}, length: ${resultData?.length || 'N/A'}`);
          if (Array.isArray(resultData) && resultData.length > 0) {
            // Result is an array of base64 strings, take the first one
            console.log(`[TASKMANAGER] First element type: ${typeof resultData[0]}, length: ${resultData[0]?.length || 'N/A'}`);
            resultData = Buffer.from(resultData[0], 'base64');
            console.log(`[TASKMANAGER] Decoded buffer length: ${resultData.length}`);
          } else if (typeof resultData === 'string') {
            resultData = Buffer.from(resultData, 'base64');
            console.log(`[TASKMANAGER] Decoded string buffer length: ${resultData.length}`);
          }
          task.assembler.integrate({ chunkId, result: resultData, meta: entry.meta });
          const tAssembled = now();
          task.timers.chunkRow({ chunkId, replica, tCreate: entry.tCreate, tAssembled, cpuTimeMs, gpuTimeMs });
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

        // Send metrics:stop message to listener
        this.sendMetricsToListeners('metrics:stop', { taskId, outInfo });
        logger.info(`Task ${taskId}: Sent metrics:stop to listener`);

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

  _sendToNativeClient(socket, eventType, data) {
    // For native clients, use the custom emit mechanism that sends JSON messages
    if (socket.emit) {
      socket.emit(eventType, data);
    } else {
      console.error('Socket does not have emit method');
    }
  }

  _drainTaskQueue(task){
    if(!task || task.status!=='running') return;
    console.log(`[DEBUG] Draining task queue for ${task.id}, ${task.queue.length} chunks in queue`);
    console.log(`[DEBUG] Task framework: ${task.framework}`);
    console.log(`[DEBUG] About to process chunks in queue`);

    // Check for stuck chunks first
    const now = Date.now();
    const stuckTimeout = 10000; // 10 seconds
    for(const chunkId of task.queue){
      const entry = task.assignments.get(chunkId);
      if(!entry || entry.completed) continue;

      // Check if chunk is stuck (assigned but not completed for too long)
      if(entry.replicas >= task.K && !entry.completed) {
        const timeSinceAssignment = now - (entry.tCreate || now);
        if(timeSinceAssignment > stuckTimeout) {
          console.log(`[DEBUG] Chunk ${chunkId} is stuck (${timeSinceAssignment}ms), reassigning...`);
          // Reset client inFlight counts first
          for(const clientId of entry.assignedTo) {
            const client = this.clients.get(clientId);
            if(client) {
              client.inFlight = Math.max(0, (client.inFlight || 0) - 1);
            }
          }
          // Reset the chunk assignment
          entry.replicas = 0;
          entry.assignedTo.clear();
        }
      }
    }

    // Iterate over queued chunks and try to assign replicas
    for(const chunkId of task.queue){
      const entry = task.assignments.get(chunkId);
      if(!entry || entry.completed) continue;
      console.log(`[DEBUG] Processing chunk ${chunkId}, replicas: ${entry.replicas}/${task.K}`);
      // Try to assign as many replicas as needed (up to K)
      while(entry.replicas < task.K){
        const assigned = this._assignChunkReplica(task, chunkId);
        console.log(`[DEBUG] Chunk ${chunkId} assignment attempt: ${assigned}`);
        if(!assigned) break; // No capacity right now
      }
    }
  }
}
