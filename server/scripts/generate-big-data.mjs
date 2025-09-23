#!/usr/bin/env node
// Generate large data files for block matmul and distributed sort

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

// Configuration
const N = parseInt(args.N||'2048',10), K = parseInt(args.K||'2048',10), M = parseInt(args.M||'2048',10);
const sortCount = parseInt(args.sortCount||'2000000',10); // 2M integers
const uploadsDir = args.uploadsDir || path.join(__dirname, '..', 'storage', 'uploads');

// Generate random Float32Array matrix with streaming for large matrices
async function randMat(r, c, outputPath) {
  const totalElements = r * c;
  const chunkSize = 1024 * 1024; // 1M elements per chunk (4MB)

  console.log(`  Generating ${r}x${c} matrix (${totalElements} elements, ${(totalElements * 4 / 1024 / 1024).toFixed(2)} MB)`);

  const fd = fs.openSync(outputPath, 'w');

  for (let offset = 0; offset < totalElements; offset += chunkSize) {
    const currentChunkSize = Math.min(chunkSize, totalElements - offset);
    const chunk = new Float32Array(currentChunkSize);

    // Generate random data for this chunk
    for (let i = 0; i < currentChunkSize; i++) {
      chunk[i] = (Math.random() * 2 - 1);
    }

    // Write chunk to file
    const buffer = Buffer.from(chunk.buffer);
    fs.writeSync(fd, buffer, 0, buffer.length, offset * 4);

    // Progress indicator
    const progress = ((offset + currentChunkSize) / totalElements * 100).toFixed(1);
    process.stdout.write(`\r  Progress: ${progress}%`);
  }

  fs.closeSync(fd);
  console.log(`\r   Matrix generated: ${(totalElements * 4 / 1024 / 1024).toFixed(2)} MB`);
}

// Generate random integers for sorting with streaming for large datasets
async function generateRandomIntegers(count, outputPath) {
  const chunkSize = 1024 * 1024; // 1M integers per chunk (4MB)

  console.log(`  Generating ${count} integers (${(count * 4 / 1024 / 1024).toFixed(2)} MB)`);

  const fd = fs.openSync(outputPath, 'w');

  for (let offset = 0; offset < count; offset += chunkSize) {
    const currentChunkSize = Math.min(chunkSize, count - offset);
    const chunk = new Uint32Array(currentChunkSize);

    // Generate random integers for this chunk
    for (let i = 0; i < currentChunkSize; i++) {
      chunk[i] = Math.floor(Math.random() * 0xFFFFFFFF);
    }

    // Write chunk to file
    const buffer = Buffer.from(chunk.buffer);
    fs.writeSync(fd, buffer, 0, buffer.length, offset * 4);

    // Progress indicator
    const progress = ((offset + currentChunkSize) / count * 100).toFixed(1);
    process.stdout.write(`\r  Progress: ${progress}%`);
  }

  fs.closeSync(fd);
  console.log(`\r   Sort data generated: ${(count * 4 / 1024 / 1024).toFixed(2)} MB`);
}

async function main() {
  console.log('️  Generating large data files...');
  console.log(`Matrix dimensions: A(${N}x${K}), B(${K}x${M})`);
  console.log(`Sort data: ${sortCount} integers`);
  console.log(`Output directory: ${uploadsDir}`);

  // Calculate total memory requirements
  const totalElements = N * K + K * M + sortCount;
  const totalMemoryGB = (totalElements * 4 / 1024 / 1024 / 1024).toFixed(2);
  console.log(` Total memory requirement: ~${totalMemoryGB} GB`);

  if (totalElements > 100000000) { // > 100M elements
    console.log('️  Large dataset detected - using streaming generation');
  }

  // Ensure uploads directory exists
  await fs.promises.mkdir(uploadsDir, { recursive: true });

  // Generate matrices with streaming
  const timestamp = Date.now();
  const fileA = `${timestamp}_A.bin`;
  const fileB = `${timestamp}_B.bin`;
  const fileSort = `${timestamp}_large_sort_input.bin`;

  const fileAPath = path.join(uploadsDir, fileA);
  const fileBPath = path.join(uploadsDir, fileB);
  const fileSortPath = path.join(uploadsDir, fileSort);

  console.log(' Generating matrix A...');
  await randMat(N, K, fileAPath);

  console.log(' Generating matrix B...');
  await randMat(K, M, fileBPath);

  // Generate sort data
  console.log(' Generating sort data...');
  await generateRandomIntegers(sortCount, fileSortPath);

  console.log('');
  console.log(' Files generated:');
  console.log(`  A.bin: ${fileA} (${(N * K * 4 / 1024 / 1024).toFixed(2)} MB)`);
  console.log(`  B.bin: ${fileB} (${(K * M * 4 / 1024 / 1024).toFixed(2)} MB)`);
  console.log(`  Sort: ${fileSort} (${(sortCount * 4 / 1024 / 1024).toFixed(2)} MB)`);
  console.log('');
  console.log(' Ready to run with cached scripts!');
  console.log('');
  console.log('Block matmul command:');
  console.log(`node test-block-matmul-cached.mjs --N=${N} --K=${K} --M=${M} --validate=true`);
  console.log('');
  console.log('Distributed sort commands:');
  console.log(`node test-distributed-sort-cached.mjs --framework=webgpu --file=${fileSort} --chunkSize=65536`);
  console.log(`node test-distributed-sort-cached.mjs --framework=cuda --file=${fileSort} --chunkSize=65536`);
}

main().catch(console.error);