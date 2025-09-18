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

  _finishByKillSwitch(task){
    if (!task || task.status === 'completed' || task.status === 'assembling' || task.status === 'error') return;
    task.cancelRequested = true;                 // stop chunking/scheduling ASAP
    task.status = 'completed';
    task.endTime = now();
    task.timers.endSummary(path.join(this.storageDir, 'timing', 'task_summaries.csv'), 'completed');
    try { task.assembler?.cleanup?.(); } catch {}
    try { task.osTracker?.stop('completed'); } catch {}
    this.sendMetricsToListeners('metrics:stop', { taskId: task.id, killSwitch: true });
    if (task.descriptor.config?.cleanupOutputFiles) this._cleanupOutputFiles(task, { killSwitch: true });
    this.io.emit('task:done', { taskId: task.id, killSwitch: true });
  }


createTask({strategyId, K=1, label='task', config={}, inputArgs={}, inputFiles=[], cachedFilePaths=[]}){
  const strategy = getStrategy(strategyId);
  const id = uuidv4();
  const taskDir = path.join(this.storageDir, 'tasks', id);
  ensureDir(taskDir);
  const descriptor = { id, label, strategyId, status: 'created', createdAt: now(), K, config, inputArgs, inputFiles: [], cachedFilePaths };

  // Process cachedFilePaths first
  logger.info(`Processing ${cachedFilePaths.length} cached file paths:`, cachedFilePaths);
  for (const fileName of cachedFilePaths) {
    const uploadsDir = path.join(this.storageDir, 'uploads');
    const filePath = path.join(uploadsDir, fileName);

    logger.info(`Looking for cached file: ${filePath}`);
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      descriptor.inputFiles.push({
        path: filePath,
        originalName: fileName,
        size: stats.size
      });
      logger.info(`Added cached file: ${fileName} (${stats.size} bytes)`);
    } else {
      logger.warn(`Cached file not found: ${filePath}`);
    }
  }
  logger.info(`Total inputFiles after processing cachedFilePaths: ${descriptor.inputFiles.length}`);

  // Process regular inputFiles
  for (const f of inputFiles){
    const dest = path.join(taskDir, f.originalName || path.basename(f.path));

    // Check if this is a cached file (already in storage/uploads)
    const uploadsDir = path.join(this.storageDir, 'uploads');
    const isCachedFile = f.path && f.path.startsWith(uploadsDir);

    if (isCachedFile) {
      // For cached files, just reference them directly - NO COPYING
      descriptor.inputFiles.push({
        path: f.path,  // Use original path directly
        originalName: path.basename(f.path),
        size: f.size
      });
    } else if (f.path) {
      // For uploaded files, copy them (but use async for large files)
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
    completionThreshold: null, // Kill-switch threshold for deterministic completion
    cancelRequested: false, chunkerFinished: false,
    framework: null,
    osTracker: null,
    queueLock: false, // Simple mutex to prevent concurrent queue modifications
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

           // Set completion threshold for kill-switch mechanism
           try {
             if (typeof task.strategy.getTotalChunks === 'function') {
               task.completionThreshold = task.strategy.getTotalChunks(task.descriptor.config, task.descriptor.inputArgs);
               logger.info(`Task ${task.id}: Kill-switch threshold set to ${task.completionThreshold} chunks`);
               logger.info(`Task ${task.id}: Config used for threshold calculation:`, JSON.stringify(task.descriptor.config, null, 2));
             } else {
               logger.warn(`Task ${task.id}: Strategy ${task.strategy.id} does not provide getTotalChunks() method - no kill-switch protection`);
             }
           } catch (e) {
             logger.warn(`Task ${task.id}: Failed to calculate completion threshold:`, e.message);
           }

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
          artifacts: artifacts,
          clientId: client.socket.id // Add client ID to workload for native clients
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
    if (!task) {
      throw new Error(`Unknown task: ${taskId}`);
    }
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

    // Send metrics:start message to listener (1 second advance)
    this.sendMetricsToListeners('metrics:start', { taskId: task.id });
    logger.info(`Task ${task.id}: Sent metrics:start to listener`);
    sleep(1000); // allow listener to start collecting metrics
    try {
      for await (const chunk of task.chunker.stream()) {
        if (task.cancelRequested) {
          logger.info(`Task ${task.id}: Chunking cancelled at ${count.toLocaleString()} chunks`);
          break;
        }

        // Throttle check BEFORE processing the chunk
        const pending = task.assignments.size - task.completedChunks;
        while ((task.assignments.size - task.completedChunks) > MAX_PENDING && !task.cancelRequested) {
          logger.debug(`Task ${task.id}: Throttling chunk generation (${pending} pending, ${task.queue.length} in queue)`);
          // Use setImmediate to yield control to other operations (like result processing)
          await new Promise(r => setTimeout(r, 5)); // Skip this iteration
        }

        const entry = {
          results: new Map(),
          assignedTo: new Set(),
          completed: false,
          replicas: 0,
          payload: chunk.payload,
          meta: chunk.meta || {},
          tCreate: chunk.tCreate || now(),
          lastAssignedAt: 0,
          reassigns: 0,
        };

        task.assignments.set(chunk.id, entry);

        // Acquire queue lock to safely add to queue
        while (task.queueLock) {
          await new Promise(resolve => setImmediate(resolve)); // Yield to event loop
        }
        task.queueLock = true;

        try {
          task.queue.push(chunk.id);
          logger.debug(`🔧 Created chunk ${chunk.id} - queue size: ${task.queue.length}, assignments: ${task.assignments.size}`);
        } finally {
          task.queueLock = false;
        }

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
        const supports = c.frameworks.includes(task.framework);
        return supports;
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
    if (!task || task.status !== 'running' || task.cancelRequested) return false;
    // First check if chunk is completed or doesn't exist
    const entry = task.assignments.get(chunkId);
    if(!entry || entry.completed) {
      if(entry && entry.completed) {
        logger.debug(`🚫 Skipping assignment of completed chunk ${chunkId}`);
      }
      return false;
    }

    let assigned = false; // Track if we successfully assigned this chunk

    // Check if chunk is stuck (assigned but not completed for too long)
    const now = Date.now();
    const stuckTimeout = 30000; // 30 seconds
    if(entry.replicas >= task.K && !entry.completed) {
      const timeSinceAssignment = now - (entry.tCreate || now);
      if(timeSinceAssignment > stuckTimeout) {
        logger.debug(`🔄 Resetting stuck chunk ${chunkId} after ${timeSinceAssignment}ms`);
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
        return false; // Chunk is assigned but not stuck yet
      }
    }

    if(entry.replicas >= task.K) return false;

    // Prevent reassignment of chunks that are already assigned and in progress
    // This fixes the race condition where chunks get reassigned while being processed
    // BUT allow assignment to different clients if we haven't reached K replicas yet
    if(entry.assignedTo.size > 0 && !entry.completed) {
      // If we've already reached K replicas, don't assign more
      if(entry.replicas >= task.K) {
        logger.debug(`🚫 Skipping reassignment of chunk ${chunkId} - already assigned to [${Array.from(entry.assignedTo).join(',')}] and reached K=${task.K}`);
        return false;
      }
      // If we haven't reached K replicas yet, we can still assign to different clients
      // The loop below will check if the client is already assigned to this chunk
    }

    // Check if we allow same client multiple replicas (for research/testing)
    const allowSameClient = task.descriptor.config?.allowSameClientReplicas || false;

    const eligibleClients = this._eligibleClients(task);
    for(const c of eligibleClients){
      // Skip if client already assigned to this chunk AND we don't allow same client replicas
      if(!allowSameClient && entry.assignedTo.has(c.socket.id)) continue;

      if((c.inFlight||0) >= (c.capacity||1)) continue; // respect client capacity

      const replica = entry.replicas;
      entry.assignedTo.add(c.socket.id);
      entry.replicas++;
      c.inFlight = (c.inFlight||0) + 1;
      c.tasks.add(task.id);
      assigned = true;
      entry.lastAssignedAt = Date.now();


      // Convert ArrayBuffers to base64 for native clients (unless already in raw format)
      let serializedPayload = entry.payload;
      if (c.clientType === 'native' && entry.payload && entry.payload.buffers) {
        try {
          serializedPayload = {
            ...entry.payload,
            buffers: entry.payload.buffers.map((buf, i) => {
              if (buf instanceof ArrayBuffer) {
                // CRITICAL: ArrayBuffer serializes to {} in JSON, so convert to base64 immediately
                const base64 = Buffer.from(buf).toString('base64');
                return base64;
              } else if (Array.isArray(buf)) {
                // Convert large arrays to base64 to avoid JSON.stringify issues
                if (buf.length > 10000) { // Large arrays should be base64 encoded
                  const uint8Array = new Uint8Array(buf);
                  return Buffer.from(uint8Array).toString('base64');
                }
                return buf;
              }
              // Handle other buffer types (TypedArray views, etc.)
              if (buf && typeof buf === 'object' && (buf.buffer || buf.byteLength !== undefined)) {
                // Convert TypedArray or DataView to base64
                const uint8Array = new Uint8Array(buf.buffer || buf, buf.byteOffset || 0, buf.byteLength || buf.length);
                return Buffer.from(uint8Array).toString('base64');
              }
              return buf;
            })
          };
        } catch (error) {
          logger.error(`Failed to serialize payload for chunk ${chunkId}: ${error.message}`);
          throw error;
        }
      }

      // Calculate payload size safely for logging
      let payloadSize = 0;
      try {
        if (serializedPayload) {
          if (serializedPayload.buffers) {
            // Standard format with buffers array
            payloadSize = serializedPayload.buffers.reduce((sum, buf) => {
              if (Array.isArray(buf)) return sum + buf.length;
              if (typeof buf === 'string') return sum + buf.length;
              return sum + (buf?.byteLength || 0);
            }, 0);
          } else if (serializedPayload.data) {
            // Distributed sort format with data ArrayBuffer
            payloadSize = serializedPayload.data.byteLength || 0;
          } else {
            // Fallback: try to calculate from the entire payload
            const payloadStr = JSON.stringify(serializedPayload);
            payloadSize = Buffer.byteLength(payloadStr, 'utf8');
          }
        }
      } catch (e) {
        payloadSize = -1; // Unable to calculate
      }

      // Emit chunk assignment with replica ID
      logger.debug(`📤 Sending chunk ${chunkId} replica ${replica} to client ${c.socket.id} (payload size: ${payloadSize} bytes) - assignedTo: [${Array.from(entry.assignedTo).join(',')}], replicas: ${entry.replicas}/${task.K}`);
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

  async receiveResult(socketId, data){
    const { taskId, chunkId, replica, status, checksum, result, timings } = data;
    const task = this.tasks.get(taskId);
    if(!task){ logger.warn('Result for unknown task', taskId); return; }
    const entry = task.assignments.get(chunkId);
    if(!entry){ logger.warn('Result for unknown chunk', chunkId); return; }
    if (entry.completed) {
      logger.debug(`Dropping late result for completed chunk ${chunkId} (replica ${replica})`);
      return;
    }
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
      if (rec2.count >= task.K){
        try{
          // Decode base64 data for native clients
          let resultData = rec2.buf;
          if (Array.isArray(resultData) && resultData.length > 0) {
            // Result is an array of base64 strings, take the first one
            resultData = Buffer.from(resultData[0], 'base64');
          } else if (typeof resultData === 'string') {
            resultData = Buffer.from(resultData, 'base64');
          }
          task.assembler.integrate({ chunkId, result: resultData, meta: entry.meta });
          const tAssembled = now();
          task.timers.chunkRow({ chunkId, replica, tCreate: entry.tCreate, tAssembled, cpuTimeMs, gpuTimeMs });
          // Atomically mark as completed and remove from queue to prevent race conditions
          // Use simple mutex to prevent concurrent queue modifications
          while (task.queueLock) {
            await new Promise(resolve => setImmediate(resolve)); // Yield to event loop
          }
          task.queueLock = true;

          try {
            entry.completed = true;
            task.completedChunks += 1
            entry.results.clear();
            // Immediately remove from queue to prevent any further assignment attempts
            const queueIndex = task.queue.indexOf(chunkId);
            if (queueIndex !== -1) {
              task.queue.splice(queueIndex, 1);
              logger.debug(`🗑️ Atomically removed completed chunk ${chunkId} from queue (index ${queueIndex})`);
            }

            // Clear assignment tracking to ensure no further assignments
            entry.assignedTo.clear();
            entry.replicas = 0;
          } finally {
            task.queueLock = false;
          }

          logger.debug(`Chunk accepted ${task.id} ${chunkId} checksum ${cs} - completed: ${entry.completed}, queue removed, assignments cleared`);
          // KILL-SWITCH: evaluate immediately after counting a completion
          if (task.completionThreshold && task.completedChunks >= task.completionThreshold) {
            logger.warn(`🚨 KILL-SWITCH: Task ${task.id} reached ${task.completedChunks}/${task.completionThreshold}`);
            this._finishByKillSwitch(task);   // helper below
            return;
          }
          this._maybeFinish(task.id);
          this._drainTaskQueue(task);
        }catch(e){
          logger.error('Assembler integrate error', e);
        }
        return;
      }
    }

    // KILL-SWITCH: Check if we've reached the completion threshold (moved outside replica check)
    // Debug logging for kill-switch
    logger.debug(`🔍 Kill-switch check: completedChunks=${task.completedChunks}, threshold=${task.completionThreshold}`);

    /*
    if (task.completionThreshold && task.completedChunks >= task.completionThreshold) {
      logger.warn(`🚨 KILL-SWITCH ACTIVATED: Task ${task.id} reached completion threshold (${task.completedChunks}/${task.completionThreshold}) - forcing completion`);

      // Force task completion immediately
      task.status = 'completed';
      task.endTime = now();
      task.timers.endSummary(path.join(this.storageDir, 'timing', 'task_summaries.csv'), 'completed');

      // Clean up resources
      try {
        if(task.assembler && typeof task.assembler.cleanup === 'function') {
          task.assembler.cleanup();
          logger.info(`Task ${task.id}: Assembler cleanup completed (kill-switch)`);
        }
      } catch(e) {
        logger.warn(`Task ${task.id}: Assembler cleanup failed (kill-switch):`, e.message);
      }

      // OSUsageTracker
      try { task.osTracker?.stop('completed'); } catch {}

      // Send metrics:stop message
      this.sendMetricsToListeners('metrics:stop', { taskId: task.id, killSwitch: true });
      logger.info(`Task ${task.id}: Sent metrics:stop to listener (kill-switch activated)`);

      // Clean up output files if requested
      if (task.descriptor.config?.cleanupOutputFiles === true) {
        this._cleanupOutputFiles(task, { killSwitch: true });
      }

      logger.warn(`🚨 Task ${task.id} COMPLETED BY KILL-SWITCH - no more chunks will be processed`);
      return; // Exit early, don't process any more chunks
    }
    */
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

        // Clean up output files if requested
        if (task.descriptor.config?.cleanupOutputFiles === true) {
          this._cleanupOutputFiles(task, outInfo);
        }

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

    // Assembler cleanup - close file descriptors and clean up resources
    try {
      if(task.assembler && typeof task.assembler.cleanup === 'function') {
        task.assembler.cleanup();
        logger.info(`Task ${id}: Assembler cleanup completed`);
      }
    } catch(e) {
      logger.warn(`Task ${id}: Assembler cleanup failed:`, e.message);
    }

    // OSUsageTracker
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

  _cleanupOutputFiles(task, outInfo) {
    try {
      const taskDir = path.join(this.storageDir, 'tasks', task.id);

      // Clean up output files but preserve task descriptor and timings
      const filesToCleanup = [];

      // Get output file path from outInfo if available
      if (outInfo?.outPath) {
        filesToCleanup.push(outInfo.outPath);
      }

      // Also check for common output file patterns in task directory
      const taskDirContents = fs.readdirSync(taskDir);
      for (const file of taskDirContents) {
        const filePath = path.join(taskDir, file);
        const stat = fs.statSync(filePath);

        // Only remove files (not directories) and skip task descriptor
        if (stat.isFile() && !file.startsWith('task_descriptor_')) {
          // Check if it's likely an output file
          if (file.includes('output') || file.endsWith('.bin') || file.endsWith('.json')) {
            filesToCleanup.push(filePath);
          }
        }
      }

      // Remove the files
      for (const filePath of filesToCleanup) {
        try {
          fs.unlinkSync(filePath);
          logger.info(`Task ${task.id}: Cleaned up output file: ${path.basename(filePath)}`);
        } catch (err) {
          logger.warn(`Task ${task.id}: Failed to cleanup file ${filePath}:`, err.message);
        }
      }

      logger.info(`Task ${task.id}: Output files cleanup completed (${filesToCleanup.length} files removed)`);
    } catch (error) {
      logger.error(`Task ${task.id}: Error during output files cleanup:`, error);
    }
  }

  _drainTaskQueue(task){
    if(!task || task.status!=='running') return;

    // Acquire queue lock to prevent concurrent modifications
    if (task.queueLock) {
      // If queue is locked, skip this iteration to avoid race conditions
      return;
    }
    task.queueLock = true;

    try {
      // First, clean up the queue by removing completed chunks
      // This is critical to prevent processing of already completed chunks
      const originalQueueLength = task.queue.length;
      task.queue = task.queue.filter(chunkId => {
        const entry = task.assignments.get(chunkId);
        if (!entry || entry.completed) {
          logger.debug(`🧹 Removing completed/missing chunk ${chunkId} from queue`);
          return false;
        }
        return true;
      });

      if (task.queue.length !== originalQueueLength) {
        logger.debug(`🧹 Queue cleanup: removed ${originalQueueLength - task.queue.length} completed chunks, ${task.queue.length} remaining`);
      }

      // Check for stuck chunks
      const now = Date.now();
      const stuckTimeout = 10000; // 10 seconds
      for(const chunkId of task.queue){
        const entry = task.assignments.get(chunkId);
        if(!entry || entry.completed) continue;

        // Check if chunk is stuck (assigned but not completed for too long)
        if(entry.replicas >= task.K && !entry.completed) {
          const last = entry.lastAssignedAt || entry.tCreate || now;
          const timeSinceAssignment = now - last;
          if(timeSinceAssignment > stuckTimeout) {
            logger.debug(`🔄 Resetting stuck chunk ${chunkId} in drainQueue after ${timeSinceAssignment}ms`);
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
      // Make a copy of the queue to avoid issues if queue is modified during iteration
      const queueCopy = [...task.queue];
      for(const chunkId of queueCopy){
        const entry = task.assignments.get(chunkId);
        if(!entry || entry.completed) {
          // This should not happen after cleanup, but just in case
          logger.debug(`🚫 Skipping chunk ${chunkId} - missing or completed after cleanup`);
          continue;
        }
        // Try to assign as many replicas as needed (up to K)
        while(entry.replicas < task.K){
          const assigned = this._assignChunkReplica(task, chunkId);
          if(!assigned) break; // No capacity right now
        }
      }
    } finally {
      task.queueLock = false;
    }
  }
}
