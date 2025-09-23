#!/usr/bin/env node
// Test script for multi-head attention strategy

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function parseArgs(argv) {
  const out = {};
  for (const token of argv.slice(2)) {
    const m = token.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2];
    else if (token.startsWith('--')) out[token.slice(2)] = true;
  }
  return out;
}

const args = parseArgs(process.argv);
const host = args.host || 'https://localhost:3000';
const seq_len = parseInt(args.seq_len || '128', 10);
const d_model = parseInt(args.d_model || '512', 10);
const num_heads = parseInt(args.num_heads || '8', 10);
const Krep = parseInt(args.K || '1', 10); // Replication factor

console.log(`Testing Multi-Head Attention: seq_len=${seq_len}, d_model=${d_model}, num_heads=${num_heads}`);

if (d_model % num_heads !== 0) {
  console.error(`d_model (${d_model}) must be divisible by num_heads (${num_heads})`);
  process.exit(1);
}

const d_k = d_model / num_heads;

// Generate random matrices
function randomMatrix(rows, cols) {
  const matrix = new Float32Array(rows * cols);
  for (let i = 0; i < matrix.length; i++) {
    // Xavier initialization
    matrix[i] = (Math.random() * 2 - 1) * Math.sqrt(2.0 / (rows + cols));
  }
  return matrix;
}

// Simple CPU reference implementation for validation
function computeAttentionCPU(Q, K, V, seq_len, d_k) {
  const scale = 1.0 / Math.sqrt(d_k);
  const output = new Float32Array(seq_len * d_k);

  for (let i = 0; i < seq_len; i++) {
    // Compute attention scores for row i
    const scores = new Float32Array(seq_len);
    let maxScore = -Infinity;

    for (let j = 0; j < seq_len; j++) {
      let score = 0;
      for (let k = 0; k < d_k; k++) {
        score += Q[i * d_k + k] * K[j * d_k + k];
      }
      scores[j] = score * scale;
      maxScore = Math.max(maxScore, scores[j]);
    }

    // Apply softmax
    let sum = 0;
    for (let j = 0; j < seq_len; j++) {
      scores[j] = Math.exp(scores[j] - maxScore);
      sum += scores[j];
    }
    for (let j = 0; j < seq_len; j++) {
      scores[j] /= sum;
    }

    // Compute weighted sum with V
    for (let k = 0; k < d_k; k++) {
      let value = 0;
      for (let j = 0; j < seq_len; j++) {
        value += scores[j] * V[j * d_k + k];
      }
      output[i * d_k + k] = value;
    }
  }

  return output;
}

async function main() {
  console.log('Generating test matrices...');

  const Q = randomMatrix(seq_len, d_model);
  const Kmat = randomMatrix(seq_len, d_model);
  const V = randomMatrix(seq_len, d_model);

  // Save matrices to files
  const tmpDir = path.join(__dirname, 'tmp');
  if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
  }

  const qPath = path.join(tmpDir, 'Q.bin');
  const kPath = path.join(tmpDir, 'K.bin');
  const vPath = path.join(tmpDir, 'V.bin');

  fs.writeFileSync(qPath, Buffer.from(Q.buffer));
  fs.writeFileSync(kPath, Buffer.from(Kmat.buffer));
  fs.writeFileSync(vPath, Buffer.from(V.buffer));

  console.log('Saved input matrices to tmp/');

  // Compute reference for first head only (for validation)
  const Q_head0 = Q.slice(0, seq_len * d_k);
  const K_head0 = Kmat.slice(0, seq_len * d_k);
  const V_head0 = V.slice(0, seq_len * d_k);

  console.log('Computing CPU reference for head 0...');
  const referenceHead0 = computeAttentionCPU(Q_head0, K_head0, V_head0, seq_len, d_k);

  // Create task
  console.log('Creating task...');
  const formData = new FormData();
  formData.append('strategyId', 'multi-head-attention');
  formData.append('K', String(Krep));
  formData.append('label', `MHA-test-${seq_len}x${d_model}x${num_heads}`);
  formData.append('config', JSON.stringify({
    seq_len,
    d_model,
    num_heads,
    framework: 'webgpu'
  }));
  formData.append('Q.bin', new Blob([fs.readFileSync(qPath)]), 'Q.bin');
  formData.append('K.bin', new Blob([fs.readFileSync(kPath)]), 'K.bin');
  formData.append('V.bin', new Blob([fs.readFileSync(vPath)]), 'V.bin');

  let resp = await fetch(`${host}/tasks`, { method: 'POST', body: formData });
  if (!resp.ok) {
    console.error('Create task failed:', await resp.text());
    process.exit(1);
  }

  const task = await resp.json();
  const taskId = task.id;
  console.log('Created task:', taskId);

  // Start task
  console.log('Starting task...');
  resp = await fetch(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) {
    console.error('Start task failed:', await resp.text());
    process.exit(1);
  }

  // Poll for completion
  console.log('Waiting for completion...');
  let status;
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 2000));

    const statusResp = await fetch(`${host}/tasks/${taskId}`);
    const statusData = await statusResp.json();
    status = statusData.status;

    const progress = statusData.totalChunks ?
      `${statusData.completedChunks || 0}/${statusData.totalChunks}` :
      `${statusData.completedChunks || 0}/?`;

    process.stdout.write(`\rStatus: ${status}, Progress: ${progress}   `);

    if (status === 'completed' || status === 'error' || status === 'canceled') {
      break;
    }
  }
  console.log();

  if (status !== 'completed') {
    console.error('Task failed with status:', status);
    process.exit(1);
  }

  // Download and validate output
  console.log('Downloading output...');
  try {
    const outputResp = await fetch(`${host}/tasks/${taskId}/output`);
    if (!outputResp.ok) {
      throw new Error(`HTTP ${outputResp.status}: ${outputResp.statusText}`);
    }

    const outputBuffer = await outputResp.arrayBuffer();
    const output = new Float32Array(outputBuffer);

    console.log(`Output size: ${output.length} floats (${outputBuffer.byteLength} bytes)`);
    console.log(`Expected size: ${seq_len * d_model} floats`);

    if (output.length !== seq_len * d_model) {
      console.error('Output size mismatch!');
      process.exit(1);
    }

    // Validate first head against CPU reference
    console.log('Validating first head against CPU reference...');
    const outputHead0 = output.slice(0, seq_len * d_k);

    let maxDiff = 0;
    let avgDiff = 0;
    let validCount = 0;

    for (let i = 0; i < outputHead0.length; i++) {
      const diff = Math.abs(outputHead0[i] - referenceHead0[i]);
      maxDiff = Math.max(maxDiff, diff);
      avgDiff += diff;

      // Check for NaN or extreme values
      if (isNaN(outputHead0[i]) || !isFinite(outputHead0[i])) {
        console.error(`Invalid value at index ${i}: ${outputHead0[i]}`);
        process.exit(1);
      }

      // Reasonable tolerance for float32 computation
      if (diff < 0.01) validCount++;
    }

    avgDiff /= outputHead0.length;
    const accuracy = (validCount / outputHead0.length) * 100;

    console.log(`Validation results:`);
    console.log(`  Max difference: ${maxDiff.toExponential(3)}`);
    console.log(`  Average difference: ${avgDiff.toExponential(3)}`);
    console.log(`  Accuracy (diff < 0.01): ${accuracy.toFixed(1)}%`);

    // Check for reasonable attention properties
    console.log('Checking attention properties...');
    let allHeadsMeanNormalized = true;

    for (let head = 0; head < num_heads; head++) {
      const headStart = head * d_k;
      let sum = 0;
      let sumSquares = 0;

      for (let i = 0; i < seq_len; i++) {
        for (let j = 0; j < d_k; j++) {
          const val = output[i * d_model + headStart + j];
          sum += val;
          sumSquares += val * val;
        }
      }

      const mean = sum / (seq_len * d_k);
      const variance = sumSquares / (seq_len * d_k) - mean * mean;

      if (Math.abs(mean) > 0.1 || variance < 0.001 || variance > 10) {
        allHeadsMeanNormalized = false;
        console.warn(`Head ${head}: unusual statistics (mean=${mean.toFixed(4)}, var=${variance.toFixed(4)})`);
      }
    }

    if (maxDiff < 0.1 && accuracy > 80) {
      console.log(' PASS - Output matches CPU reference');
    } else {
      console.log(' FAIL - Output differs significantly from CPU reference');
      process.exit(1);
    }

    if (allHeadsMeanNormalized) {
      console.log(' All heads have reasonable statistics');
    }

  } catch (error) {
    console.error('Failed to download/validate output:', error.message);
    process.exit(1);
  }

  // Cleanup
  try {
    fs.unlinkSync(qPath);
    fs.unlinkSync(kPath);
    fs.unlinkSync(vPath);
    fs.rmdirSync(tmpDir);
  } catch (e) {
    // Ignore cleanup errors
  }

  console.log(' Multi-head attention test completed successfully!');
}

main().catch(error => {
  console.error('Test failed:', error);
  process.exit(1);
});
