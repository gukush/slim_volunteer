// server/strategies/exe_distributed_sort.js
// Native binary execution strategy for distributed integer sorting via CUDA
// Combines chunking from distributed-sort.js with binary execution from exe-block-matmul-flex.js

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'exe-distributed-sort';
export const name = 'Distributed Integer Sort (Native Binary Execution)';

export function getClientExecutorInfo(config) {
  const backend = (config?.backend || 'cuda').toLowerCase();
  if (!['cuda', 'opencl'].includes(backend)) {
    throw new Error(`Unsupported backend: ${backend}. Must be 'cuda' or 'opencl'`);
  }

  return {
    framework: 'exe',
    kernels: [],
    schema: {
      order: ['UNIFORMS', 'INPUTS', 'OUTPUTS'],
      uniforms: [
        { name: 'originalSize', type: 'i32' },
        { name: 'paddedSize', type: 'i32' },
        { name: 'ascending', type: 'i32' }
      ],
      inputs: [{ name: 'data', type: 'u32' }],
      outputs: [{ name: 'sortedData', type: 'u32' }]
    },
    artifacts: getArtifacts(config)
  };
}

export function getArtifacts(config) {
  const backend = (config?.backend || 'cuda').toLowerCase();
  const artifacts = [];

  // Framework-specific binary paths
  const frameworkBinaries = {
    cuda: config.cudaBinary || config.binary || '/app/binaries/exe_distributed_sort',
    opencl: config.openclBinary || config.binary || '/app/binaries/exe_distributed_sort_opencl'
  };

  const binaryPath = frameworkBinaries[backend];
  if (!binaryPath) {
    console.log(`[DEBUG] getArtifacts - Unknown backend ${backend}`);
    return [];
  }

  try {
    const abs = path.isAbsolute(binaryPath) ? binaryPath : binaryPath;
    const bytes = fs.readFileSync(abs).toString('base64');
    const artifactName = config.program || path.basename(binaryPath);

    console.log(`[DEBUG] getArtifacts - Adding binary: ${artifactName} for ${backend}`);

    artifacts.push({
      type: 'binary',
      name: artifactName,
      program: artifactName,
      backend,
      bytes,
      exec: true
    });
  } catch (error) {
    console.error(`[DEBUG] getArtifacts - Failed to read binary ${binaryPath}:`, error.message);
    throw new Error(`Binary not found for ${backend}: ${binaryPath}`);
  }

  return artifacts;
}

// Round up to next power of 2 for efficient bitonic sort
function nextPowerOf2(n) {
  if (n <= 0) return 1;
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

function readIntegersChunk(fd, offset, count, totalIntegers) {
  const actualCount = Math.min(count, totalIntegers - offset);
  if (actualCount <= 0) return new Uint32Array(0);

  const buffer = Buffer.alloc(actualCount * 4);
  const bytesRead = fs.readSync(fd, buffer, 0, actualCount * 4, offset * 4);

  // Convert to Uint32Array
  const integers = new Uint32Array(buffer.buffer, buffer.byteOffset, actualCount);
  return integers;
}

// Find newest matching binary file
function findNewestBin({ preferredName = 'large_sort_input.bin', taskDir, uploadsDir }) {
  const candidates = [];

  const pushIfExists = (p) => {
    try {
      const st = fs.statSync(p);
      if (st.isFile()) {
        candidates.push({ path: p, mtime: st.mtimeMs, size: st.size });
      }
    } catch {}
  };

  if (taskDir) {
    try {
      for (const name of fs.readdirSync(taskDir)) {
        if (/\.bin$/i.test(name) && !/_[AB]\.bin$/i.test(name)) {
          if (name === preferredName || name.endsWith(`_${preferredName}`)) {
            pushIfExists(path.join(taskDir, name));
          } else {
            pushIfExists(path.join(taskDir, name));
          }
        }
      }
    } catch {}
  }

  const upDir = uploadsDir || path.join(process.cwd(), 'uploads');
  try {
    for (const name of fs.readdirSync(upDir)) {
      if (/\.bin$/i.test(name) && !/_[AB]\.bin$/i.test(name)) {
        if (name === preferredName || name.endsWith(`_${preferredName}`)) {
          pushIfExists(path.join(upDir, name));
        } else {
          pushIfExists(path.join(upDir, name));
        }
      }
    }
  } catch {}

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => {
    const aPref = Number(a.path.endsWith(`_${preferredName}`) || path.basename(a.path) === preferredName);
    const bPref = Number(b.path.endsWith(`_${preferredName}`) || path.basename(b.path) === preferredName);
    if (aPref !== bPref) return bPref - aPref;
    if (a.mtime !== b.mtime) return b.mtime - a.mtime;
    return (b.size || 0) - (a.size || 0);
  });

  return candidates[0].path;
}

export function buildChunker({ taskId, taskDir, K, config, inputFiles }) {
  const uploadsDir = process.env.UPLOADS_DIR || path.join(process.cwd(), 'uploads');
  const listed = Array.isArray(inputFiles) ? inputFiles : [];
  const byName = listed.find(f => f && f.originalName && /\.bin$/i.test(f.originalName) && !/_[AB]\.bin$/i.test(f.originalName));
  let inputFile = (byName && byName.path) || (listed[0] && listed[0].path) || null;

  if (!inputFile) {
    inputFile = findNewestBin({ preferredName: 'large_sort_input.bin', taskDir, uploadsDir }) ||
               findNewestBin({ preferredName: 'input.bin', taskDir, uploadsDir });
  }

  if (!inputFile) {
    throw new Error('Need input binary file with integers (.bin). None provided and none found in task/uploads.');
  }

  const { chunkSize = 65536, ascending = true, maxElements } = config;
  const fd = fs.openSync(inputFile, 'r');

  const fileStats = fs.fstatSync(fd);
  if (fileStats.size % 4 !== 0) {
    throw new Error(`Input file size (${fileStats.size} bytes) is not a multiple of 4.`);
  }

  let totalIntegers = Math.floor(fileStats.size / 4);

  if (maxElements && maxElements > 0 && maxElements < totalIntegers) {
    totalIntegers = maxElements;
    logger.info(`Limiting processing to ${maxElements} elements (file contains ${Math.floor(fileStats.size / 4)} elements)`);
  }

  const chunksCount = Math.ceil(totalIntegers / chunkSize);

  // Get the binary name for program reference
  const backend = (config?.backend || 'cuda').toLowerCase();
  const defaultBins = {
    cuda: '/app/binaries/exe_distributed_sort',
    opencl: '/app/binaries/exe_distributed_sort_opencl'
  };
  const rel = config.binary || defaultBins[backend];
  const binaryName = config.program || path.basename(rel);

  logger.info(`Exe distributed sort: ${totalIntegers} integers, ${chunksCount} chunks`);

  return {
    async *stream() {
      let chunkIndex = 0;
      for (let offset = 0; offset < totalIntegers; offset += chunkSize) {
        const actualChunkSize = Math.min(chunkSize, totalIntegers - offset);
        const paddedSize = nextPowerOf2(actualChunkSize);

        const integers = readIntegersChunk(fd, offset, actualChunkSize, totalIntegers);

        // Pad with sentinel values for bitonic sort
        const paddedIntegers = new Uint32Array(paddedSize);
        paddedIntegers.set(integers);

        const sentinelValue = ascending ? 0xFFFFFFFF : 0x00000000;
        for (let i = actualChunkSize; i < paddedSize; i++) {
          paddedIntegers[i] = sentinelValue;
        }

        // Prepare binary payload: UNIFORMS, then INPUTS
        const uniforms = new Int32Array([actualChunkSize, paddedSize, ascending ? 1 : 0]);
        const uniformsBytes = new Uint8Array(uniforms.buffer);

        const payload = {
          action: 'exec',
          framework: 'exe',
          buffers: [
            Array.from(uniformsBytes),  // uniforms
            Array.from(new Uint8Array(paddedIntegers.buffer))  // input data
          ],
          outputs: [{ byteLength: actualChunkSize * 4 }]  // Only output original size
        };

        const meta = {
          chunkIndex,
          offset,
          originalSize: actualChunkSize,
          paddedSize: paddedSize,
          totalIntegers,
          backend: config.backend || 'cuda',
          program: binaryName,
          outputSizes: [actualChunkSize * 4],
          uniforms: [actualChunkSize, paddedSize, ascending ? 1 : 0]
        };

        yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
        chunkIndex++;
      }

      fs.closeSync(fd);
      logger.info('Exe distributed sort chunker done');
    }
  };
}

export function buildAssembler({ taskId, taskDir, config }) {
  const { ascending = true } = config;

  // Merge configuration
  const memoryThresholdMB = config.memoryThresholdMB || 512;
  const memoryThresholdBytes = memoryThresholdMB * 1024 * 1024;
  const maxRunsBeforeMerge = config.maxRunsBeforeMerge || 10;
  const runMergeSize = config.runMergeSize || 4;

  // State tracking
  const inMemoryChunks = [];
  const diskRuns = [];
  const tempDir = path.join(taskDir, 'temp_runs');
  const outPath = path.join(taskDir, 'output.bin');

  let currentMemoryUsage = 0;
  let runCounter = 0;
  let backgroundMergeRunning = false;

  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  function writeRunToDisk(chunks) {
    if (chunks.length === 0) return null;

    const runId = `run_${runCounter++}`;
    const runPath = path.join(tempDir, `${runId}.bin`);

    chunks.sort((a, b) => a.chunkIndex - b.chunkIndex);
    const mergedData = mergeKSortedArrays(chunks.map(chunk => chunk.data), ascending);

    const runBuffer = Buffer.from(mergedData.buffer);
    fs.writeFileSync(runPath, runBuffer);

    const elementCount = mergedData.length;
    const runInfo = { runId, filePath: runPath, elementCount };
    diskRuns.push(runInfo);

    logger.info(`Wrote run ${runId} to disk: ${elementCount} elements (${(runBuffer.length / 1024 / 1024).toFixed(1)} MB)`);
    return runInfo;
  }

  function readRunFromDisk(runInfo) {
    const buffer = fs.readFileSync(runInfo.filePath);
    return new Uint32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  }

  async function backgroundMergeRuns() {
    if (backgroundMergeRunning || diskRuns.length < runMergeSize) return;

    backgroundMergeRunning = true;

    try {
      while (diskRuns.length >= runMergeSize) {
        const runsToMerge = diskRuns.splice(0, runMergeSize);
        logger.info(`Background merging ${runsToMerge.length} runs`);

        const runArrays = runsToMerge.map(runInfo => readRunFromDisk(runInfo));
        const mergedData = mergeKSortedArrays(runArrays, ascending);

        const newRunId = `merged_run_${runCounter++}`;
        const newRunPath = path.join(tempDir, `${newRunId}.bin`);
        const mergedBuffer = Buffer.from(mergedData.buffer);
        fs.writeFileSync(newRunPath, mergedBuffer);

        const newRunInfo = {
          runId: newRunId,
          filePath: newRunPath,
          elementCount: mergedData.length
        };
        diskRuns.push(newRunInfo);

        for (const oldRun of runsToMerge) {
          try {
            fs.unlinkSync(oldRun.filePath);
            logger.debug(`Cleaned up old run file: ${oldRun.filePath}`);
          } catch (e) {
            logger.warn(`Failed to clean up run file ${oldRun.filePath}:`, e.message);
          }
        }

        logger.info(`Background merge complete: ${newRunInfo.elementCount} elements`);
        await new Promise(resolve => setImmediate(resolve));
      }
    } finally {
      backgroundMergeRunning = false;
    }
  }

  function checkMemoryThreshold() {
    if (currentMemoryUsage > memoryThresholdBytes && inMemoryChunks.length > 0) {
      logger.info(`Memory threshold exceeded (${(currentMemoryUsage / 1024 / 1024).toFixed(1)} MB), spilling to disk`);

      writeRunToDisk([...inMemoryChunks]);
      inMemoryChunks.length = 0;
      currentMemoryUsage = 0;

      setTimeout(() => backgroundMergeRuns(), 0);
    }
  }

  return {
    integrate({ chunkId, result, meta }) {
      const { chunkIndex, originalSize } = meta;

      let u8;
      if (result instanceof ArrayBuffer) {
        u8 = new Uint8Array(result);
      } else if (ArrayBuffer.isView(result)) {
        u8 = new Uint8Array(result.byteLength);
        u8.set(new Uint8Array(result.buffer, result.byteOffset, result.byteLength));
      } else if (Buffer.isBuffer?.(result)) {
        u8 = Uint8Array.from(result);
      } else if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
        u8 = Uint8Array.from(result.data);
      } else {
        u8 = Uint8Array.from(result);
      }

      if (u8.byteLength % 4 !== 0) {
        throw new Error(`Result byteLength ${u8.byteLength} not multiple of 4`);
      }

      const sortedData = new Uint32Array(u8.buffer, 0, u8.byteLength >>> 2);
      const actualData = sortedData.subarray(0, originalSize);

      const chunkInfo = {
        chunkIndex,
        data: actualData,
        originalSize
      };

      inMemoryChunks.push(chunkInfo);
      currentMemoryUsage += actualData.byteLength;

      logger.debug(`Integrated chunk ${chunkIndex}, size: ${originalSize}, memory usage: ${(currentMemoryUsage / 1024 / 1024).toFixed(1)} MB`);

      checkMemoryThreshold();
    },

    async finalize() {
      logger.info(`Starting finalization: ${inMemoryChunks.length} in-memory chunks, ${diskRuns.length} disk runs`);

      while (backgroundMergeRunning) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      await backgroundMergeRuns();

      const mergeSources = [];

      if (inMemoryChunks.length > 0) {
        inMemoryChunks.sort((a, b) => a.chunkIndex - b.chunkIndex);
        const memoryMerged = mergeKSortedArrays(inMemoryChunks.map(chunk => chunk.data), ascending);
        mergeSources.push(memoryMerged);
        logger.info(`In-memory merge: ${memoryMerged.length} elements`);
      }

      for (const runInfo of diskRuns) {
        const runData = readRunFromDisk(runInfo);
        mergeSources.push(runData);
        logger.info(`Loaded disk run ${runInfo.runId}: ${runData.length} elements`);
      }

      if (mergeSources.length === 0) {
        throw new Error('No data to merge');
      }

      logger.info(`Final merge of ${mergeSources.length} sources`);
      const finalResult = mergeSources.length === 1
        ? mergeSources[0]
        : mergeKSortedArrays(mergeSources, ascending);

      const outputBuffer = Buffer.from(finalResult.buffer);
      fs.writeFileSync(outPath, outputBuffer);

      try {
        for (const runInfo of diskRuns) {
          fs.unlinkSync(runInfo.filePath);
        }
        fs.rmdirSync(tempDir);
        logger.info('Cleaned up temporary files');
      } catch (e) {
        logger.warn('Error cleaning up temporary files:', e.message);
      }

      logger.info(`External sort complete. Output written to ${outPath}, ${finalResult.length} integers`);

      const memoryPeakMB = (memoryThresholdBytes / 1024 / 1024).toFixed(1);
      const totalSizeMB = (finalResult.length * 4 / 1024 / 1024).toFixed(1);
      logger.info(`Memory efficiency: Peak ${memoryPeakMB} MB for ${totalSizeMB} MB dataset (${(memoryThresholdMB / (finalResult.length * 4 / 1024 / 1024) * 100).toFixed(1)}% of total)`);

      return {
        outPath,
        elements: finalResult.length,
        memoryPeakMB: parseFloat(memoryPeakMB),
        totalSizeMB: parseFloat(totalSizeMB),
        diskRunsUsed: diskRuns.length
      };
    }
  };
}

// k-way merge with streaming support for large datasets
function mergeKSortedArrays(sortedArrays, ascending = true) {
  if (sortedArrays.length === 0) return new Uint32Array(0);
  if (sortedArrays.length === 1) return new Uint32Array(sortedArrays[0]);

  const totalSize = sortedArrays.reduce((sum, arr) => sum + arr.length, 0);
  const result = new Uint32Array(totalSize);

  const heap = [];
  const indices = new Array(sortedArrays.length).fill(0);

  for (let i = 0; i < sortedArrays.length; i++) {
    if (sortedArrays[i].length > 0) {
      heap.push({ value: sortedArrays[i][0], arrayIndex: i });
    }
  }

  heap.sort((a, b) => ascending ? a.value - b.value : b.value - a.value);

  let resultIndex = 0;

  while (heap.length > 0) {
    const min = heap.shift();
    result[resultIndex++] = min.value;

    const nextIndex = ++indices[min.arrayIndex];
    if (nextIndex < sortedArrays[min.arrayIndex].length) {
      const nextValue = sortedArrays[min.arrayIndex][nextIndex];

      let insertPos = 0;
      while (insertPos < heap.length &&
             (ascending ? heap[insertPos].value <= nextValue : heap[insertPos].value >= nextValue)) {
        insertPos++;
      }
      heap.splice(insertPos, 0, { value: nextValue, arrayIndex: min.arrayIndex });
    }
  }

  return result;
}
