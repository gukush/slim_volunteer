import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'distributed-sort';
export const name = 'Distributed Integer Sort';

export function getClientExecutorInfo(config) {
  const fw = (config?.framework || 'webgpu').toLowerCase();
  if (fw === 'webgpu') {
    return {
      framework: 'webgpu',
      path: 'executors/webgpu-distributed-sort.client.js',
      kernels: ['kernels/bitonic_sort.wgsl']
    };
  }
  throw new Error('Unsupported framework in config.framework: ' + fw);
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

// Round up to next power of 2 for efficient bitonic sort
function nextPowerOf2(n) {
  if (n <= 0) return 1;
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

export function buildChunker({ taskId, taskDir, K, config, inputFiles }) {
  const inputFile = inputFiles.find(f => /\.bin$/.test(f.originalName))?.path || inputFiles[0]?.path;
  if (!inputFile) throw new Error('Need input binary file with integers');

  const { chunkSize = 65536, ascending = true } = config; // Default 64K integers per chunk
  const fd = fs.openSync(inputFile, 'r');
  const fileStats = fs.fstatSync(fd);
  const totalIntegers = Math.floor(fileStats.size / 4); // 4 bytes per 32-bit integer
  const chunksCount = Math.ceil(totalIntegers / chunkSize);

  logger.info(`Distributed sort: ${totalIntegers} integers, ${chunksCount} chunks`);

  return {
    async *stream() {
      let chunkIndex = 0;
      for (let offset = 0; offset < totalIntegers; offset += chunkSize) {
        const actualChunkSize = Math.min(chunkSize, totalIntegers - offset);
        const paddedSize = nextPowerOf2(actualChunkSize); // Pad to power of 2 for bitonic sort

        // Read actual data
        const integers = readIntegersChunk(fd, offset, actualChunkSize, totalIntegers);

        // Pad with max/min values depending on sort direction
        const paddedIntegers = new Uint32Array(paddedSize);
        paddedIntegers.set(integers);

        // Fill padding with sentinel values
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
  const sortedChunks = []; // Array of {chunkIndex, data, originalSize}
  const outPath = path.join(taskDir, 'output.bin');

  return {
    integrate({ chunkId, result, meta }) {
      const { chunkIndex, originalSize } = meta;

      let u8;
      if (result instanceof ArrayBuffer) {
        u8 = new Uint8Array(result);                       // aligned at 0
      } else if (ArrayBuffer.isView(result)) {
        // copy bytes so the new view starts at offset 0
        u8 = new Uint8Array(result.byteLength);
        u8.set(new Uint8Array(result.buffer, result.byteOffset, result.byteLength));
      } else if (Buffer.isBuffer?.(result)) {
        u8 = Uint8Array.from(result);                      // copy from Node Buffer
      } else if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
        u8 = Uint8Array.from(result.data);                 // JSON-ified Buffer
      } else {
        u8 = Uint8Array.from(result);
      }

      if (u8.byteLength % 4 !== 0) {
        throw new Error(`Result byteLength ${u8.byteLength} not multiple of 4`);
      }
      const sortedData = new Uint32Array(u8.buffer, 0, u8.byteLength >>> 2);
      // keep only real values (drop padding)
      const actualData = sortedData.subarray(0, originalSize);

      sortedChunks.push({
        chunkIndex,
        data: actualData,
        originalSize
      });

      logger.debug(`Integrated chunk ${chunkIndex}, size: ${originalSize}`);
    },

    finalize() {
      // Sort chunks by their index to maintain order
      sortedChunks.sort((a, b) => a.chunkIndex - b.chunkIndex);

      // Perform k-way merge of sorted chunks
      const mergedData = mergeKSortedArrays(sortedChunks.map(chunk => chunk.data), ascending);

      // Write to output file
      const outputBuffer = Buffer.from(mergedData.buffer);
      fs.writeFileSync(outPath, outputBuffer);

      logger.info(`Sorting complete. Output written to ${outPath}, ${mergedData.length} integers`);
      return { outPath, elements: mergedData.length };
    }
  };
}

// K-way merge of sorted arrays
function mergeKSortedArrays(sortedArrays, ascending = true) {
  if (sortedArrays.length === 0) return new Uint32Array(0);
  if (sortedArrays.length === 1) return new Uint32Array(sortedArrays[0]);

  // Calculate total size
  const totalSize = sortedArrays.reduce((sum, arr) => sum + arr.length, 0);
  const result = new Uint32Array(totalSize);

  // Min/Max heap approach using simple array with indices
  const heap = [];
  const indices = new Array(sortedArrays.length).fill(0);

  // Initialize heap with first element from each array
  for (let i = 0; i < sortedArrays.length; i++) {
    if (sortedArrays[i].length > 0) {
      heap.push({ value: sortedArrays[i][0], arrayIndex: i });
    }
  }

  // Sort initial heap
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
