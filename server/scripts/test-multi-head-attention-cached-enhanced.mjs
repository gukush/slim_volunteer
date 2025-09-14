#!/usr/bin/env node
// Enhanced cached multi-head attention script with multi-framework support
// Supports: webgpu, native (cuda via lua), exe (cuda binary)
// Uses Q.bin, K.bin, V.bin matrices

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

const host = args.host || 'https://localhost:3000';
const framework = args.framework || 'webgpu'; // 'webgpu' | 'native' | 'exe'
const backend = args.backend || 'cuda'; // For native/exe strategies
const seqLen = parseInt(args.seqLen||'512',10);
const dModel = parseInt(args.dModel||'768',10);
const numHeads = parseInt(args.numHeads||'12',10);
const Krep = parseInt(args.Krep||'1',10);
const validate = args.validate === true || args.validate === 'true' || args.validate === '1';
const fileQ = args.fileQ || 'Q.bin';
const fileK = args.fileK || 'K.bin';
const fileV = args.fileV || 'V.bin';

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

// Strategy routing based on framework
function getStrategyAndConfig(framework, backend, seqLen, dModel, numHeads) {
  const frameworkLower = framework.toLowerCase();

  // Calculate derived dimensions
  const dK = Math.floor(dModel / numHeads);
  const dV = dK; // Typically d_v = d_k

  if (dK * numHeads !== dModel) {
    throw new Error(`d_model (${dModel}) must be divisible by num_heads (${numHeads})`);
  }

  switch (frameworkLower) {
    case 'webgpu':
      return {
        strategyId: 'multi-head-attention',
        config: {
          seq_len: seqLen,
          d_model: dModel,
          num_heads: numHeads,
          framework: 'webgpu'
        }
      };

    case 'native':
      return {
        strategyId: 'native-multi-head-attention',
        config: {
          seq_len: seqLen,
          d_model: dModel,
          num_heads: numHeads,
          framework: 'native-cuda'
        }
      };

    case 'exe':
      if (backend.toLowerCase() !== 'cuda') {
        throw new Error(`exe-multi-head-attention only supports backend=cuda, got: ${backend}`);
      }
      return {
        strategyId: 'exe-multi-head-attention',
        config: {
          seq_len: seqLen,
          d_model: dModel,
          num_heads: numHeads,
          backend: 'cuda'
        }
      };

    default:
      throw new Error(`Unsupported framework: ${framework}. Use 'webgpu', 'native', or 'exe'`);
  }
}

// Find suitable input files for multi-head attention
function findMHAInputFiles(uploadsDir, fileQ, fileK, fileV) {
  console.log(`🔍 Looking for MHA input files...`);
  console.log(`   Q: ${fileQ}`);
  console.log(`   K: ${fileK}`);
  console.log(`   V: ${fileV}`);

  if (!fs.existsSync(uploadsDir)) {
    throw new Error(`Uploads directory not found: ${uploadsDir}`);
  }

  const files = fs.readdirSync(uploadsDir);
  console.log(`📁 Available files in uploads: ${files.length} files`);

  // List all .bin files with sizes
  const binFiles = files.filter(f => f.endsWith('.bin')).map(f => {
    const fullPath = path.join(uploadsDir, f);
    const stats = fs.statSync(fullPath);
    return {
      name: f,
      path: fullPath,
      size: stats.size,
      sizeMB: (stats.size / 1024 / 1024).toFixed(2)
    };
  }).sort((a, b) => b.size - a.size); // Sort by size, largest first

  console.log('📊 .bin files found:');
  binFiles.forEach(f => {
    console.log(`   - ${f.name} (${f.sizeMB} MB)`);
  });

  // Check for required files
  const requiredFiles = [fileQ, fileK, fileV];
  const foundFiles = [];
  const missingFiles = [];

  for (const reqFile of requiredFiles) {
    const filePath = path.join(uploadsDir, reqFile);
    if (fs.existsSync(filePath)) {
      foundFiles.push(reqFile);
      console.log(`✅ Found: ${reqFile}`);
    } else {
      missingFiles.push(reqFile);
      console.log(`❌ Missing: ${reqFile}`);
    }
  }

  if (missingFiles.length > 0) {
    console.log(`\n💡 To generate missing files, run:`);
    console.log(`node generate-mha-data.mjs --seqLen=${seqLen} --dModel=${dModel} --numHeads=${numHeads}`);
    throw new Error(`Missing required files: ${missingFiles.join(', ')}`);
  }

  return foundFiles;
}

async function runMultiHeadAttention() {
  console.log(`🚀 Running Multi-Head Attention with ${framework.toUpperCase()}...`);
  if (['native', 'exe'].includes(framework.toLowerCase())) {
    console.log(`🎯 Backend: ${backend.toUpperCase()}`);
  }
  console.log(`📊 Dimensions: seq_len=${seqLen}, d_model=${dModel}, num_heads=${numHeads}`);
  console.log(`📁 Input files: ${fileQ}, ${fileK}, ${fileV}`);

  // Find input files
  const uploadsDir = path.join(__dirname, '..', 'storage', 'uploads');
  const inputFiles = findMHAInputFiles(uploadsDir, fileQ, fileK, fileV);

  // Load cached files for validation if needed
  let referenceQ, referenceK, referenceV;
  if (validate) {
    console.log('🔍 Loading cached files for validation...');

    const fileQPath = path.join(uploadsDir, fileQ);
    const fileKPath = path.join(uploadsDir, fileK);
    const fileVPath = path.join(uploadsDir, fileV);

    const qBuffer = fs.readFileSync(fileQPath);
    const kBuffer = fs.readFileSync(fileKPath);
    const vBuffer = fs.readFileSync(fileVPath);

    referenceQ = new Float32Array(qBuffer.buffer, qBuffer.byteOffset, qBuffer.byteLength / 4);
    referenceK = new Float32Array(kBuffer.buffer, kBuffer.byteOffset, kBuffer.byteLength / 4);
    referenceV = new Float32Array(vBuffer.buffer, vBuffer.byteOffset, vBuffer.byteLength / 4);

    console.log(`📊 Loaded Q: ${referenceQ.length} elements (${(qBuffer.length / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`📊 Loaded K: ${referenceK.length} elements (${(kBuffer.length / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`📊 Loaded V: ${referenceV.length} elements (${(vBuffer.length / 1024 / 1024).toFixed(2)} MB)`);

    // Validate dimensions
    const expectedSize = seqLen * dModel;
    if (referenceQ.length !== expectedSize || referenceK.length !== expectedSize || referenceV.length !== expectedSize) {
      throw new Error(`Dimension mismatch: expected ${expectedSize} elements per matrix, got Q:${referenceQ.length}, K:${referenceK.length}, V:${referenceV.length}`);
    }

    console.log('✅ Input validation complete');
  }

  // Get strategy and config based on framework
  const { strategyId, config } = getStrategyAndConfig(framework, backend, seqLen, dModel, numHeads);

  console.log(`🎯 Using strategy: ${strategyId}`);
  console.log(`⚙️  Config:`, JSON.stringify(config, null, 2));

  // Create task using cached file paths
  const taskPayload = {
    strategyId,
    K: Krep,
    label: `multi-head-attention-cached-${framework}`,
    config,
    cachedFilePaths: [fileQ, fileK, fileV]
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
  console.log(`✅ Created task ${taskId}`);

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
  const resultData = new Float32Array(resultBuffer.buffer, resultBuffer.byteOffset, resultBuffer.byteLength / 4);

  console.log(`✅ Successfully computed ${resultData.length} elements`);

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
  const expectedSize = seqLen * dModel;

  if (resultData.length !== expectedSize) {
    console.error(`❌ Output size mismatch: expected ${expectedSize}, got ${resultData.length}`);
    process.exit(4);
  }

  console.log(`⏱️  Time: ${totalTime}s`);
  console.log(`📊 Output dimensions: ${seqLen} × ${dModel} (${(resultBuffer.length / 1024 / 1024).toFixed(2)} MB)`);

  if (validate) {
    console.log('✅ Validation: Output dimensions match expected size');
    console.log(`✅ Successfully computed multi-head attention in ${totalTime}s`);
  }

  return true;
}

async function main() {
  console.log('🎯 Enhanced Cached Multi-Head Attention Test');
  console.log('============================================');
  console.log(`Framework: ${framework.toUpperCase()}`);
  if (['native', 'exe'].includes(framework.toLowerCase())) {
    console.log(`Backend: ${backend.toUpperCase()}`);
  }
  console.log(`Host: ${host}`);
  console.log(`Dimensions: seq_len=${seqLen}, d_model=${dModel}, num_heads=${numHeads}`);
  console.log(`Input files: ${fileQ}, ${fileK}, ${fileV}`);
  console.log('');

  try {
    await runMultiHeadAttention();
    console.log('\n🎉 Multi-head attention test completed successfully!');
  } catch (error) {
    console.error('💥 Error:', error);
    process.exit(99);
  }
}

main();
