#!/usr/bin/env node
// File: scripts/test-native-multi-head-attention.mjs
// Test the runtime-compiled CUDA (Lua-host) strategy.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

function args() {
  const out = {};
  for (const a of process.argv.slice(2)) {
    const m = a.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2]; else if (a.startsWith('--')) out[a.slice(2)] = true;
  }
  return out;
}

const a = args();
const host      = a.host || 'https://localhost:3000';
const seq_len   = Number(a.seq_len || 512);
const d_model   = Number(a.d_model || 768);
const num_heads = Number(a.num_heads || 12);
const Krep      = Number(a.K || 2);

// reference math
function randMat(r,c){ const x=new Float32Array(r*c); for(let i=0;i<x.length;i++) x[i]=(Math.random()*2-1); return x; }

async function main() {
  const d_k = Math.floor(d_model/num_heads);
  const d_v = d_k;

  const tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-mha-native-'));
  const qPath = path.join(tmpDir, 'Q.bin');
  const kPath = path.join(tmpDir, 'K.bin');
  const vPath = path.join(tmpDir, 'V.bin');

  const Q = randMat(seq_len, d_model);
  const K = randMat(seq_len, d_model);
  const V = randMat(seq_len, d_model);
  fs.writeFileSync(qPath, Buffer.from(Q.buffer));
  fs.writeFileSync(kPath, Buffer.from(K.buffer));
  fs.writeFileSync(vPath, Buffer.from(V.buffer));

  const form = new FormData();
  form.append('strategyId', 'native-multi-head-attention');
  form.append('K', String(Krep));
  form.append('label', `NATIVE-MHA-${seq_len}x${d_model}x${num_heads}`);
  form.append('config', JSON.stringify({
    seq_len, d_model, num_heads, framework: 'native-cuda'
  }));
  form.append('Q.bin', new Blob([fs.readFileSync(qPath)]), 'Q.bin');
  form.append('K.bin', new Blob([fs.readFileSync(kPath)]), 'K.bin');
  form.append('V.bin', new Blob([fs.readFileSync(vPath)]), 'V.bin');

  console.log('Creating task...');
  let resp = await fetch(`${host}/tasks`, { method: 'POST', body: form });
  if (!resp.ok) { console.error('Create failed:', await resp.text()); process.exit(1); }
  const { id: taskId } = await resp.json();

  console.log('Starting task...');
  resp = await fetch(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) { console.error('Start failed:', await resp.text()); process.exit(1); }

  // wait
  let status='pending';
  while (!['complete','error','cancelled'].includes(status)) {
    await new Promise(r=>setTimeout(r,1000));
    const d = await (await fetch(`${host}/tasks/${taskId}`)).json();
    status = d.status;
    process.stdout.write(`\rStatus: ${status}  chunks=${d?.stats?.completedChunks ?? '?'}   `);
  }
  console.log();

  if (status !== 'complete') { console.error('Task failed:', status); process.exit(2); }

  // fetch result
  const outBuf = Buffer.from(await (await fetch(`${host}/tasks/${taskId}/artifact?name=attention_output.bin`)).arrayBuffer());
  console.log(`Output bytes: ${outBuf.byteLength}`);

  // quick sanity: shape seq_len x d_model
  if (outBuf.byteLength !== seq_len*d_model*4) {
    console.error('Size mismatch'); process.exit(2);
  }

  // clean
  try { fs.unlinkSync(qPath); fs.unlinkSync(kPath); fs.unlinkSync(vPath); fs.rmdirSync(tmpDir); } catch {}
  console.log('✅ Native MHA (CUDA Lua host) test finished.');
}

main().catch(e=>{ console.error(e); process.exit(99); });
