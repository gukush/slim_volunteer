#!/usr/bin/env node
// Generate Q, K, V matrices for multi-head attention

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

// Configuration
const seqLen = parseInt(args.seqLen||'512',10);
const dModel = parseInt(args.dModel||'768',10);
const numHeads = parseInt(args.numHeads||'12',10);
const uploadsDir = args.uploadsDir || path.join(__dirname, '..', 'storage', 'uploads');

// Validate dimensions
const dK = Math.floor(dModel / numHeads);
const dV = dK; // Typically d_v = d_k

if (dK * numHeads !== dModel) {
  console.error(` Error: d_model (${dModel}) must be divisible by num_heads (${numHeads})`);
  console.error(`   Current: d_model=${dModel}, num_heads=${numHeads}, d_k=${dK}`);
  console.error(`   Suggestion: Use num_heads that divides ${dModel} evenly`);
  process.exit(1);
}

// Generate random Float32Array matrix with streaming for large matrices
async function generateMHAMatrix(rows, cols, outputPath, matrixName) {
  const totalElements = rows * cols;
  const chunkSize = 1024 * 1024; // 1M elements per chunk (4MB)

  console.log(`  Generating ${matrixName} matrix: ${rows}x${cols} (${totalElements} elements, ${(totalElements * 4 / 1024 / 1024).toFixed(2)} MB)`);

  const fd = fs.openSync(outputPath, 'w');

  for (let offset = 0; offset < totalElements; offset += chunkSize) {
    const currentChunkSize = Math.min(chunkSize, totalElements - offset);
    const chunk = new Float32Array(currentChunkSize);

    // Generate random data for this chunk
    // Use different random ranges for Q, K, V to make them distinguishable
    for (let i = 0; i < currentChunkSize; i++) {
      if (matrixName === 'Q') {
        chunk[i] = (Math.random() * 2 - 1) * 0.1; // Small values for Q
      } else if (matrixName === 'K') {
        chunk[i] = (Math.random() * 2 - 1) * 0.05; // Smaller values for K
      } else { // V
        chunk[i] = (Math.random() * 2 - 1) * 0.2; // Slightly larger values for V
      }
    }

    // Write chunk to file
    const buffer = Buffer.from(chunk.buffer);
    fs.writeSync(fd, buffer, 0, buffer.length, offset * 4);

    // Progress indicator
    const progress = ((offset + currentChunkSize) / totalElements * 100).toFixed(1);
    process.stdout.write(`\r  Progress: ${progress}%`);
  }

  fs.closeSync(fd);
  console.log(`\r   ${matrixName} matrix generated: ${(totalElements * 4 / 1024 / 1024).toFixed(2)} MB`);
}

async function main() {
  console.log('️  Generating Multi-Head Attention data files...');
  console.log(` Dimensions: seq_len=${seqLen}, d_model=${dModel}, num_heads=${numHeads}`);
  console.log(` Derived: d_k=${dK}, d_v=${dV}`);
  console.log(` Output directory: ${uploadsDir}`);

  // Calculate total memory requirements
  const totalElements = 3 * seqLen * dModel; // Q, K, V matrices
  const totalMemoryGB = (totalElements * 4 / 1024 / 1024 / 1024).toFixed(2);
  console.log(` Total memory requirement: ~${totalMemoryGB} GB`);

  if (totalElements > 100000000) { // > 100M elements
    console.log('️  Large dataset detected - using streaming generation');
  }

  // Ensure uploads directory exists
  await fs.promises.mkdir(uploadsDir, { recursive: true });

  // Generate matrices with streaming
  const timestamp = Date.now();
  const fileQ = `${timestamp}_Q.bin`;
  const fileK = `${timestamp}_K.bin`;
  const fileV = `${timestamp}_V.bin`;

  const fileQPath = path.join(uploadsDir, fileQ);
  const fileKPath = path.join(uploadsDir, fileK);
  const fileVPath = path.join(uploadsDir, fileV);

  console.log(' Generating Q matrix...');
  await generateMHAMatrix(seqLen, dModel, fileQPath, 'Q');

  console.log(' Generating K matrix...');
  await generateMHAMatrix(seqLen, dModel, fileKPath, 'K');

  console.log(' Generating V matrix...');
  await generateMHAMatrix(seqLen, dModel, fileVPath, 'V');

  console.log('');
  console.log(' Files generated:');
  console.log(`  Q.bin: ${fileQ} (${(seqLen * dModel * 4 / 1024 / 1024).toFixed(2)} MB)`);
  console.log(`  K.bin: ${fileK} (${(seqLen * dModel * 4 / 1024 / 1024).toFixed(2)} MB)`);
  console.log(`  V.bin: ${fileV} (${(seqLen * dModel * 4 / 1024 / 1024).toFixed(2)} MB)`);
  console.log('');
  console.log(' Ready to run with cached scripts!');
  console.log('');
  console.log('Multi-head attention commands:');
  console.log(`node test-multi-head-attention-cached-enhanced.mjs --framework=webgpu --seqLen=${seqLen} --dModel=${dModel} --numHeads=${numHeads}`);
  console.log(`node test-multi-head-attention-cached-enhanced.mjs --framework=native --seqLen=${seqLen} --dModel=${dModel} --numHeads=${numHeads}`);
  console.log(`node test-multi-head-attention-cached-enhanced.mjs --framework=exe --seqLen=${seqLen} --dModel=${dModel} --numHeads=${numHeads}`);
  console.log('');
  console.log(' Note: All matrices are stored in row-major format [seq_len, d_model]');
  console.log(` Each matrix contains ${seqLen} sequences of ${dModel} features`);
  console.log(` Attention heads: ${numHeads} heads, each with d_k=${dK} dimensions`);
}

main().catch(console.error);
