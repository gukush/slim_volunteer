#!/usr/bin/env node
// Cached distributed sort script with multi-framework support (WebGPU and CUDA)

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s => {
  const m = s.match(/^--([^=]+)=(.*)$/);
  return m ? [m[1], m[2]] : [s.replace(/^--/, ''), true];
}));

const host = args.host || 'https://localhost:3000';
const framework = args.framework || 'webgpu'; // 'webgpu' | 'cuda'
const chunkSize = parseInt(args.chunkSize || '65536', 10);
const ascending = args.descending ? false : true;
const Krep = parseInt(args.Krep || '1', 10);
const maxElements = args.maxElements ? parseInt(args.maxElements, 10) : undefined;
const cachedFile = args.file || 'large_sort_input.bin'; // Default cached file
const backend = args.backend || 'cuda'; // For CUDA strategy
const validate = args.validate === true || args.validate === 'true' || args.validate === '1';

// Helper to ignore self-signed certificates
import { Agent } from 'https';
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function fetchWithAgent(url, options = {}) {
  if (url.startsWith('https://localhost')) {
    const https = await import('https');
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const reqOptions = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: options.method || 'GET',
        headers: options.headers || {},
        rejectUnauthorized: false
      };

      const req = https.request(reqOptions, (res) => {
        let data = [];
        res.on('data', chunk => data.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(data);
          resolve({
            ok: res.statusCode >= 200 && res.statusCode < 300,
            status: res.statusCode,
            json: async () => JSON.parse(buffer.toString()),
            text: async () => buffer.toString(),
            arrayBuffer: async () => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
          });
        });
      });

      req.on('error', reject);
      if (options.body) {
        req.write(options.body);
      }
      req.end();
    });
  }
  return fetch(url, options);
}

function isSorted(data, ascending = true) {
  for (let i = 1; i < data.length; i++) {
    if (ascending) {
      if (data[i] < data[i - 1]) return false;
    } else {
      if (data[i] > data[i - 1]) return false;
    }
  }
  return true;
}

async function runDistributedSort() {
  console.log(`🚀 Running Distributed Sort with ${framework.toUpperCase()}...`);
  console.log(`📁 Cached file: ${cachedFile}`);
  console.log(`⚙️  Chunk size: ${chunkSize}, Ascending: ${ascending}`);
  if (maxElements) console.log(`🔢 Max elements: ${maxElements}`);
  if (framework === 'cuda') console.log(`🎯 Backend: ${backend}`);

  // Load cached file for validation if needed
  let reference, originalData;
  if (validate) {
    const uploadsDir = path.join(__dirname, '..', 'storage', 'uploads');
    const filePath = path.join(uploadsDir, cachedFile);

    if (!fs.existsSync(filePath)) {
      console.error(`❌ Cached file not found: ${filePath}`);
      console.error('Available files in uploads directory:');
      try {
        const files = fs.readdirSync(uploadsDir);
        files.forEach(f => {
          const fullPath = path.join(uploadsDir, f);
          const stats = fs.statSync(fullPath);
          console.error(`  - ${f} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);
        });
      } catch (e) {
        console.error('  (Could not list uploads directory)');
      }
      process.exit(1);
    }

    console.log('🔍 Loading cached file for validation...');
    const fileBuffer = fs.readFileSync(filePath);
    originalData = new Uint32Array(fileBuffer.buffer, fileBuffer.byteOffset, fileBuffer.byteLength / 4);

    console.log(`📊 Loaded ${originalData.length} integers (${(fileBuffer.length / 1024 / 1024).toFixed(2)} MB)`);

    // Generate reference sorted array
    console.log('🧮 Computing reference sorted array...');
    reference = new Uint32Array(originalData);
    reference.sort((a, b) => ascending ? a - b : b - a);
    console.log('✅ Reference sorting complete');
  }

  // Choose strategy based on framework
  let strategyId, config;
  if (framework.toLowerCase() === 'cuda') {
    strategyId = 'exe-distributed-sort';
    config = {
      chunkSize,
      ascending,
      backend,
      maxElements
    };
  } else {
    // Default to webgpu
    strategyId = 'distributed-sort';
    config = {
      chunkSize,
      ascending,
      framework: 'webgpu',
      maxElements
    };
  }

  // Create task using cached file paths
  const taskPayload = {
    strategyId,
    K: Krep,
    label: `distributed-sort-cached-${framework}`,
    config,
    cachedFilePaths: [cachedFile]
  };

  let resp = await fetchWithAgent(`${host}/tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(taskPayload)
  });

  if (!resp.ok) {
    console.error('❌ Create task failed:', await resp.text());
    process.exit(1);
  }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log(`✅ Created task ${taskId} using strategy: ${strategyId}`);

  // Start task
  resp = await fetchWithAgent(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) {
    console.error('❌ Start failed:', await resp.text());
    process.exit(1);
  }

  // Poll for completion
  let status;
  const startTime = Date.now();
  console.log('⏳ Monitoring task progress...');
  while (true) {
    await new Promise(r => setTimeout(r, 1000));
    const s = await fetchWithAgent(`${host}/tasks/${taskId}`);
    const j = await s.json();
    status = j.status;

    const chunks = j.completedChunks || 0;
    const total = j.totalChunks || '?';
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    process.stdout.write(`\r📊 [${elapsed}s] status=${status} ${chunks}/${total} chunks   `);

    if (status === 'completed' || status === 'error' || status === 'canceled') break;
  }
  console.log();

  if (status !== 'completed') {
    console.error(`❌ Task did not complete: ${status}`);
    process.exit(2);
  }

  // Download results
  console.log('📥 Downloading results...');
  const out = await fetchWithAgent(`${host}/tasks/${taskId}/output`);
  if (!out.ok) {
    console.error('❌ Download failed:', await resp.text());
    process.exit(3);
  }

  const resultBuffer = new Uint8Array(await out.arrayBuffer());
  const resultData = new Uint32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

  console.log(`✅ Successfully sorted ${resultData.length} integers`);

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
  const throughput = Math.round(resultData.length / parseFloat(totalTime));
  console.log(`⏱️  Time: ${totalTime}s, Throughput: ${throughput} int/s`);

  // Validate results if requested
  if (validate && reference && originalData) {
    console.log('🔍 Validating results...');

    if (resultData.length !== originalData.length) {
      console.error(`❌ Length mismatch: expected ${originalData.length}, got ${resultData.length}`);
      process.exit(4);
    }

    if (!isSorted(resultData, ascending)) {
      console.error('❌ Output is not properly sorted!');
      process.exit(5);
    }

    // Verify we have the same elements (sorted reference vs our result)
    let elementsMismatch = false;
    for (let i = 0; i < originalData.length; i++) {
      if (reference[i] !== resultData[i]) {
        console.error(`❌ Element mismatch at position ${i}: expected ${reference[i]}, got ${resultData[i]}`);
        elementsMismatch = true;
        break;
      }
    }

    if (elementsMismatch) {
      console.error('❌ FAIL - Elements do not match reference');
      process.exit(6);
    } else {
      console.log('✅ PASS - All elements match reference');
      console.log(`✅ Successfully sorted ${originalData.length} integers in ${totalTime}s (${throughput} int/s)`);
    }
  }

  return true;
}

async function main() {
  console.log('🎯 Cached Distributed Sort Test');
  console.log('===============================');
  console.log(`Framework: ${framework.toUpperCase()}`);
  console.log(`Host: ${host}`);
  console.log(`Cached file: ${cachedFile}`);
  console.log('');

  try {
    await runDistributedSort();
    console.log('\n🎉 Distributed sort test completed successfully!');
  } catch (error) {
    console.error('💥 Error:', error);
    process.exit(99);
  }
}

main();
