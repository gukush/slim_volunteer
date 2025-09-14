// File: strategies/exe-cpu-quicksort.js
// Native CPU quicksort strategy using executable binary
// Reuses the distributed-sort chunker/assembler logic but executes via native binary

import fs from 'fs';
import path from 'path';
import { logger } from '../lib/logger.js';
import { v4 as uuidv4 } from 'uuid';

// Reuse chunker/assembler from the existing distributed-sort strategy
import {
  buildChunker as buildChunkerWeb,
  buildAssembler as buildAssemblerWeb,
} from './distributed-sort.js';

export const id = 'exe-cpu-quicksort';
export const name = 'Distributed CPU Quicksort (Native Binary)';

// Helper functions
function b64(buf) {
  return Buffer.isBuffer(buf) ? buf.toString('base64') : Buffer.from(buf).toString('base64');
}

function readProjectFile(rel) {
  const abs = path.isAbsolute(rel) ? rel : path.join(process.cwd(), rel);
  return fs.readFileSync(abs);
}

// This tells the server what to ship to native clients
export function getClientExecutorInfo(config = {}) {
  const framework = 'binary'; // Native binary execution

  // Prepare the CPU quicksort binary
  const binaryPath = '/app/binaries/cpu-quicksort';
  let binaryBytes;

  try {
    // Try to read the binary file
    binaryBytes = readProjectFile(binaryPath);
  } catch (error) {
    logger.warn(`CPU quicksort binary not found at ${binaryPath}, will be built on client`);
    // Create a placeholder - the client will need to build it
    binaryBytes = Buffer.from(''); // Empty buffer as placeholder
  }

  const artifacts = [
    {
      type: 'binary',
      name: 'cpu-quicksort',
      program: 'cpu-quicksort',
      backend: 'cpu',
      exec: true, // This is an executable
      bytes: b64(binaryBytes)
    }
  ];

  // Provide a schema to make native runtimes self-describing
  const schema = {
    action: 'execute_binary_stream',
    order: ['STDIN', 'STDOUT'],
    stdin: {
      type: 'binary',
      description: 'Binary data containing uint32_t integers to sort'
    },
    stdout: {
      type: 'binary',
      description: 'Binary data containing sorted uint32_t integers'
    },
    args: [
      { name: 'ascending', type: 'bool' },
      { name: 'originalSize', type: 'i32' }
    ]
  };

  return {
    framework,
    artifacts,
    schema
  };
}

// Custom chunker that creates the proper payload format for native binary execution
export function buildChunker({ taskId, taskDir, K, config, inputArgs, inputFiles }) {
  // Use the same file discovery logic as the WebGPU version
  const webChunker = buildChunkerWeb({ taskId, taskDir, K, config, inputArgs, inputFiles });

  return {
    async *stream() {
      for await (const chunk of webChunker.stream()) {
        // Transform the payload for native binary execution with common protocol
        const { payload, meta } = chunk;

        // Create protocol-compliant payload
        const protocolPayload = createProtocolPayload({
          framework: 'CPU',
          dataType: 'UINT32',
          inputs: [
            {
              data: payload.data,
              dataType: 'UINT32',
              dimensions: [payload.originalSize, 1, 1, 1] // 1D array
            }
          ],
          outputs: [
            {
              dataType: 'UINT32',
              dimensions: [payload.originalSize, 1, 1, 1]
            }
          ],
          metadata: JSON.stringify({
            ascending: payload.ascending,
            originalSize: payload.originalSize,
            paddedSize: payload.paddedSize
          })
        });

        const nativePayload = {
          action: 'execute_binary_stream',
          binary: 'cpu-quicksort',
          args: ['--stdin'], // Use stdin/stdout mode
          stdin: b64(protocolPayload), // Protocol-compliant data
          stdoutSize: payload.originalSize * 4, // Expected output size in bytes
        };

        const nativeMeta = {
          ...meta,
          framework: 'binary',
          binaryName: 'cpu-quicksort',
          useStreams: true
        };

        yield {
          id: chunk.id,
          payload: nativePayload,
          meta: nativeMeta,
          tCreate: chunk.tCreate
        };
      }
    }
  };
}

// Helper function to create protocol-compliant payload
function createProtocolPayload({ framework, dataType, inputs, outputs, metadata = '' }) {
  const buffer = Buffer.alloc(0);

  // Protocol header
  const header = Buffer.alloc(32); // ProtocolHeader size
  header.writeUInt32LE(0x4558454D, 0); // magic "EXEM"
  header.writeUInt32LE(1, 4); // version
  header.writeUInt32LE(0, 8); // framework (CPU)
  header.writeUInt32LE(5, 12); // dataType (UINT32)
  header.writeUInt32LE(inputs.length, 16); // num_inputs
  header.writeUInt32LE(outputs.length, 20); // num_outputs
  header.writeUInt32LE(Buffer.byteLength(metadata, 'utf8'), 24); // metadata_size
  header.writeUInt32LE(0, 28); // reserved

  let result = Buffer.concat([Buffer.from(header), Buffer.from(metadata, 'utf8')]);

  // Add input buffers
  for (const input of inputs) {
    const desc = Buffer.alloc(32); // BufferDescriptor size
    desc.writeUInt32LE(input.data.byteLength, 0); // size
    desc.writeUInt32LE(5, 4); // dataType (UINT32)
    desc.writeUInt32LE(input.dimensions[0] || 0, 8);
    desc.writeUInt32LE(input.dimensions[1] || 0, 12);
    desc.writeUInt32LE(input.dimensions[2] || 0, 16);
    desc.writeUInt32LE(input.dimensions[3] || 0, 20);
    desc.writeUInt32LE(0, 24); // reserved
    desc.writeUInt32LE(0, 28); // reserved
    desc.writeUInt32LE(0, 32); // reserved

    result = Buffer.concat([result, desc, Buffer.from(input.data)]);
  }

  return result;
}

// Reuse the assembler from WebGPU version since the output format is the same
export const buildAssembler = buildAssemblerWeb;
