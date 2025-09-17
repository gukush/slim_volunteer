import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

// Helper function to write large arrays to file using streaming approach
// This avoids Node.js 2GB buffer limit by writing in chunks
async function writeLargeArrayToFile(array, filePath) {
  const chunkSize = 1024 * 1024 * 1024; // 1GB chunks
  const totalBytes = array.length * 4; // 4 bytes per Uint32

  logger.info(`Writing large array to file: ${array.length} elements (${(totalBytes / 1024 / 1024 / 1024).toFixed(2)} GB)`);

  if (totalBytes <= chunkSize) {
    // Small enough to write in one go
    const buffer = Buffer.from(array.buffer);
    fs.writeFileSync(filePath, buffer);
    return;
  }

  // Large file - write in chunks
  const fd = fs.openSync(filePath, 'w');
  let offset = 0;

  try {
    while (offset < array.length) {
      const chunkLength = Math.min(chunkSize / 4, array.length - offset); // Convert bytes to elements
      const chunk = array.subarray(offset, offset + chunkLength);
      const buffer = Buffer.from(chunk.buffer);

      fs.writeSync(fd, buffer, 0, buffer.length, offset * 4);
      offset += chunkLength;

      // Log progress for very large files
      if (array.length > 100000000) { // Only log for > 100M elements
        const progress = (offset / array.length * 100).toFixed(1);
        logger.info(`Writing progress: ${progress}% (${offset.toLocaleString()}/${array.length.toLocaleString()} elements)`);
      }
    }
  } finally {
    fs.closeSync(fd);
  }

  logger.info(`Successfully wrote ${array.length} elements to ${filePath}`);
}

export const id = 'distributed-sort';
export const name = 'Distributed Integer Sort';

export function getClientExecutorInfo(config) {
  const fw = (config?.framework || 'webgpu').toLowerCase();
  if (fw === 'webgpu') {
    return {
      framework: 'webgpu',
      path: 'executors/webgpu-distributed-sort.client.js',
      kernels: ['kernels/webgpu/bitonic_sort.wgsl']
    };
  }
  throw new Error('Unsupported framework in config.framework: ' + fw);
}

function readIntegersChunk(fd, offset, count, totalIntegers) {
  const actualCount = Math.min(count, totalIntegers - offset);
  if (actualCount <= 0) {
    logger.debug(`readIntegersChunk: actualCount=${actualCount}, returning empty array`);
    return new Uint32Array(0);
  }

  const buffer = Buffer.alloc(actualCount * 4);
  const bytesRead = fs.readSync(fd, buffer, 0, actualCount * 4, offset * 4);

  logger.debug(`readIntegersChunk: offset=${offset}, count=${count}, actualCount=${actualCount}, bytesRead=${bytesRead}`);

  // Convert to Uint32Array
  const integers = new Uint32Array(buffer.buffer, buffer.byteOffset, actualCount);
  return integers;
}

// Round up to next power of 2 for efficient bitonic sort
function nextPowerOf2(n) {
  if (n <= 0) return 1;
  return Math.pow(2, Math.ceil(Math.log2(n)));
}


// return the newest matching file from a list of directories
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
          // Prefer exact preferredName
          if (name === preferredName || name.endsWith(`_${preferredName}`)) {
            pushIfExists(path.join(taskDir, name));
          } else {
            pushIfExists(path.join(taskDir, name));
          }
        }
      }
    } catch {}
  }

  // 2) Then search uploadsDir (process.cwd()/uploads by default)
  const upDir = uploadsDir || path.join(process.cwd(), 'uploads');
  try {
    for (const name of fs.readdirSync(upDir)) {
      if (/\.bin$/i.test(name) && !/_[AB]\.bin$/i.test(name)) {
        // Prefer names ending in _preferredName
        if (name === preferredName || name.endsWith(`_${preferredName}`)) {
          pushIfExists(path.join(upDir, name));
        } else {
          pushIfExists(path.join(upDir, name));
        }
      }
    }
  } catch {}

  if (candidates.length === 0) return null;

  // Prefer files that exactly match or end with _preferredName; if tie, pick newest
  candidates.sort((a, b) => {
    const aPref = Number(a.path.endsWith(`_${preferredName}`) || path.basename(a.path) === preferredName);
    const bPref = Number(b.path.endsWith(`_${preferredName}`) || path.basename(b.path) === preferredName);
    if (aPref !== bPref) return bPref - aPref;
    // newest mtime first
    if (a.mtime !== b.mtime) return b.mtime - a.mtime;
    // larger size next
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
  // Fallback: scan taskDir then uploadsDir for a suitable .bin (prefer *_large_sort_input.bin)
  inputFile = findNewestBin({ preferredName: 'large_sort_input.bin', taskDir, uploadsDir }) ||
              findNewestBin({ preferredName: 'input.bin', taskDir, uploadsDir });
}

  if (!inputFile) throw new Error('Need input binary file with integers');

  const { chunkSize = 65536, ascending = true, maxElements } = config; // Default 64K integers per chunk
if (!inputFile) {
  throw new Error('Need input binary file with integers (.bin). None provided and none found in task/uploads.');
}
const fd = fs.openSync(inputFile, 'r');

  const fileStats = fs.fstatSync(fd);
  logger.info(`Input file: ${inputFile}, size: ${fileStats.size} bytes`);

  if (fileStats.size % 4 !== 0) throw new Error(`Input file size (${fileStats.size} bytes) is not a multiple of 4.`);
  let totalIntegers = Math.floor(fileStats.size / 4); // 4 bytes per 32-bit integer

  if (totalIntegers === 0) {
    throw new Error(`Input file is empty or contains no integers (${fileStats.size} bytes)`);
  }

  // Apply maxElements limit if specified
  if (maxElements && maxElements > 0 && maxElements < totalIntegers) {
    totalIntegers = maxElements;
    logger.info(`Limiting processing to ${maxElements} elements (file contains ${Math.floor(fileStats.size / 4)} elements)`);
  }

  const chunksCount = Math.ceil(totalIntegers / chunkSize);

  logger.info(`Distributed sort: ${totalIntegers} integers, ${chunksCount} chunks, chunkSize: ${chunkSize}`);

  return {
    async *stream() {
      let chunkIndex = 0;
      for (let offset = 0; offset < totalIntegers; offset += chunkSize) {
        const actualChunkSize = Math.min(chunkSize, totalIntegers - offset);
        const paddedSize = nextPowerOf2(actualChunkSize); // Pad to power of 2 for bitonic sort

        logger.debug(`Chunk ${chunkIndex}: offset=${offset}, actualChunkSize=${actualChunkSize}, paddedSize=${paddedSize}`);

        const integers = readIntegersChunk(fd, offset, actualChunkSize, totalIntegers);
        logger.debug(`Chunk ${chunkIndex}: read ${integers.length} integers`);

        // Pad with max/min values depending on sort direction
        const paddedIntegers = new Uint32Array(paddedSize);
        paddedIntegers.set(integers);

        // Fill padding with sentino values
        const sentinelValue = ascending ? 0xFFFFFFFF : 0x00000000;
        for (let i = actualChunkSize; i < paddedSize; i++) {
          paddedIntegers[i] = sentinelValue;
        }

        const payload = {
          data: paddedIntegers.buffer.slice(0),
          originalSize: actualChunkSize,
          paddedSize: paddedSize,
          ascending: ascending
        };

        logger.debug(`Chunk ${chunkIndex}: payload data size=${payload.data.byteLength} bytes`);

        const meta = {
          chunkIndex,
          offset,
          originalSize: actualChunkSize,
          paddedSize: paddedSize,
          totalIntegers
        };

        yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
        chunkIndex++;
      }

      fs.closeSync(fd);
      logger.info('Distributed sort chunker done');
    }
  };
}

export function buildAssembler({ taskId, taskDir, config }) {
  const { ascending = true } = config;

  // Merge configuration
  const memoryThresholdMB = config.memoryThresholdMB || 512; // Default 512MB
  const memoryThresholdBytes = memoryThresholdMB * 1024 * 1024;
  const maxRunsBeforeMerge = config.maxRunsBeforeMerge || 10; // Merge disk runs when we have this many
  const runMergeSize = config.runMergeSize || 4; // How many runs to merge at once

  // State tracking
  const inMemoryChunks = []; // Array of {chunkIndex, data, originalSize}
  const diskRuns = []; // Array of {runId, filePath, elementCount}
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

    // Write to disk
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
        // Take the first N runs for merging
        const runsToMerge = diskRuns.splice(0, runMergeSize);
        logger.info(`Background merging ${runsToMerge.length} runs`);

        // Read all runs into memory
        const runArrays = runsToMerge.map(runInfo => readRunFromDisk(runInfo));

        const mergedData = mergeKSortedArrays(runArrays, ascending);

        // Write merged result back as a single run
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

        // Clean up old run files
        for (const oldRun of runsToMerge) {
          try {
            fs.unlinkSync(oldRun.filePath);
            logger.debug(`Cleaned up old run file: ${oldRun.filePath}`);
          } catch (e) {
            logger.warn(`Failed to clean up run file ${oldRun.filePath}:`, e.message);
          }
        }

        logger.info(`Background merge complete: ${newRunInfo.elementCount} elements`);

        // Yield control to allow other operations
        await new Promise(resolve => setImmediate(resolve));
      }
    } finally {
      backgroundMergeRunning = false;
    }
  }

  // Check if we need to spill to disk
  function checkMemoryThreshold() {
    if (currentMemoryUsage > memoryThresholdBytes && inMemoryChunks.length > 0) {
      logger.info(`Memory threshold exceeded (${(currentMemoryUsage / 1024 / 1024).toFixed(1)} MB), spilling to disk`);

      writeRunToDisk([...inMemoryChunks]);

      // Clear in-memory state
      inMemoryChunks.length = 0;
      currentMemoryUsage = 0;

      // Trigger background merge if needed
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

      // Add to in-memory chunks
      const chunkInfo = {
        chunkIndex,
        data: actualData,
        originalSize
      };

      inMemoryChunks.push(chunkInfo);
      currentMemoryUsage += actualData.byteLength;

      logger.debug(`Integrated chunk ${chunkIndex}, size: ${originalSize}, memory usage: ${(currentMemoryUsage / 1024 / 1024).toFixed(1)} MB`);

      // Check if we need to spill to disk
      checkMemoryThreshold();
    },

    async finalize() {
      logger.info(`Starting finalization: ${inMemoryChunks.length} in-memory chunks, ${diskRuns.length} disk runs`);

      // Wait for any background merging to complete
      while (backgroundMergeRunning) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Final background merge to reduce disk run count
      await backgroundMergeRuns();

      // Prepare final merge sources
      const mergeSources = [];

      // Add in-memory chunks as merge sources
      if (inMemoryChunks.length > 0) {
        inMemoryChunks.sort((a, b) => a.chunkIndex - b.chunkIndex);
        const memoryMerged = mergeKSortedArrays(inMemoryChunks.map(chunk => chunk.data), ascending);
        mergeSources.push(memoryMerged);
        logger.info(`In-memory merge: ${memoryMerged.length} elements`);
      }

      // Add disk runs as merge sources
      for (const runInfo of diskRuns) {
        const runData = readRunFromDisk(runInfo);
        mergeSources.push(runData);
        logger.info(`Loaded disk run ${runInfo.runId}: ${runData.length} elements`);
      }

      if (mergeSources.length === 0) {
        throw new Error('No data to merge');
      }

      // Perform final k-way merge
      logger.info(`Final merge of ${mergeSources.length} sources`);
      const finalResult = mergeSources.length === 1
        ? mergeSources[0]
        : mergeKSortedArrays(mergeSources, ascending);

      // Write final output using streaming approach for large files
      await writeLargeArrayToFile(finalResult, outPath);

      // Clean up temporary files
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

  // Use a priority queue approach with indices
  const heap = [];
  const indices = new Array(sortedArrays.length).fill(0);

  // Initialize heap with first element from each array
  for (let i = 0; i < sortedArrays.length; i++) {
    if (sortedArrays[i].length > 0) {
      heap.push({ value: sortedArrays[i][0], arrayIndex: i });
    }
  }

  heap.sort((a, b) => ascending ? a.value - b.value : b.value - a.value);

  let resultIndex = 0;

  while (heap.length > 0) {
    // Take the min/max element
    const min = heap.shift();
    result[resultIndex++] = min.value;

    // Add next element from the same array if available
    const nextIndex = ++indices[min.arrayIndex];
    if (nextIndex < sortedArrays[min.arrayIndex].length) {
      const nextValue = sortedArrays[min.arrayIndex][nextIndex];

      // Insert in correct position to maintain heap property
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

// Kill-switch support: Calculate total chunks deterministically
export function getTotalChunks(config, inputArgs) {
  const { chunkSize = 65536, maxElements } = config; // Use correct parameter name and default

  // For distributed sort, we need to know the total number of integers
  // This is typically provided in inputArgs or can be calculated from file size
  let totalIntegers = 0;

  // If maxElements is specified, use it directly (most reliable)
  if (maxElements && maxElements > 0) {
    totalIntegers = maxElements;
    logger.info(`getTotalChunks: Using maxElements=${maxElements} for calculation`);
  } else if (inputArgs && inputArgs.totalIntegers) {
    totalIntegers = inputArgs.totalIntegers;
    logger.info(`getTotalChunks: Using inputArgs.totalIntegers=${totalIntegers}`);
  } else if (inputArgs && inputArgs.inputFile) {
    // Calculate from file size (assuming 4 bytes per integer)
    try {
      const stats = fs.statSync(inputArgs.inputFile);
      totalIntegers = Math.floor(stats.size / 4);
      logger.info(`getTotalChunks: Calculated from file size: ${totalIntegers} integers`);
    } catch (e) {
      logger.warn('Could not determine total integers from file size, using default estimate');
      totalIntegers = 10000000; // Default estimate
    }
  } else {
    logger.warn('No total integers specified, using default estimate');
    totalIntegers = 10000000; // Default estimate
  }

  const totalChunks = Math.ceil(totalIntegers / chunkSize);
  logger.info(`Distributed-sort getTotalChunks: totalIntegers=${totalIntegers}, chunkSize=${chunkSize}, maxElements=${maxElements || 'none'} -> ${totalChunks} chunks`);
  return totalChunks;
}