#!/usr/bin/env node
// File: scripts/test-exe-multi-head-attention.mjs
// Test the AOT native binary path for MHA (exe-multi-head-attention).

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ---------- args ----------
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
const backend   = (a.backend || 'cuda').toLowerCase();
const binary    = a.binary || 'binaries/exe_cuda_multi_head_attention'; // avoid auto lint split
const program   = a.program || 'native_cuda_multi_head_attention';
const Krep      = Number(a.K || 2);

// ---------- ref math ----------
function cpuAttentionSingleHead(Q,K,V, seq_len, d_k, d_v) {
  const scale = 1 / Math.sqrt(d_k);
  const O = new Float32Array(seq_len * d_v);
  for (let i=0;i<seq_len;i++){
    // 1) find max
    let gmax = -1e30;
    for (let j=0;j<seq_len;j++){
      let acc = 0;
      for (let d=0; d<d_k; d++) acc += Q[i*d_k+d]*K[j*d_k+d];
      const s = acc*scale;
      if (s>gmax) gmax = s;
    }
    // 2) sum exp
    let denom = 0;
    for (let j=0;j<seq_len;j++){
      let acc = 0;
      for (let d=0; d<d_k; d++) acc += Q[i*d_k+d]*K[j*d_k+d];
      denom += Math.exp(acc*scale - gmax);
    }
    denom = Math.max(denom, 1e-20);
    // 3) weighted sum
    for (let c=0;c<d_v;c++){
      let out = 0;
      for (let j=0;j<seq_len;j++){
        let acc = 0;
        for (let d=0; d<d_k; d++) acc += Q[i*d_k+d]*K[j*d_k+d];
        const p = Math.exp(acc*scale - gmax)/denom;
        out += p * V[j*d_v+c];
      }
      O[i*d_v+c] = out;
    }
  }
  return O;
}

function randMat(r,c) {
  const a = new Float32Array(r*c);
  for (let i=0;i<a.length;i++) a[i] = (Math.random()*2-1);
  return a;
}

// ---------- main ----------
async function main() {
  const d_k = Math.floor(d_model/num_heads);
  const d_v = d_k;

  // temp dir + files
  const tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-mha-exe-'));
  const qPath = path.join(tmpDir, 'Q.bin');
  const kPath = path.join(tmpDir, 'K.bin');
  const vPath = path.join(tmpDir, 'V.bin');

  // synthesize full Q/K/V with num_heads * d_k along last dim
  const Q = randMat(seq_len, d_model);
  const K = randMat(seq_len, d_model);
  const V = randMat(seq_len, d_model);
  fs.writeFileSync(qPath, Buffer.from(Q.buffer));
  fs.writeFileSync(kPath, Buffer.from(K.buffer));
  fs.writeFileSync(vPath, Buffer.from(V.buffer));

  // create task
  const form = new FormData();
  form.append('strategyId', 'exe-multi-head-attention');
  form.append('K', String(Krep));
  form.append('label', `EXE-MHA-${seq_len}x${d_model}x${num_heads}`);
  form.append('config', JSON.stringify({
    seq_len, d_model, num_heads,
    backend, binary, program,
    framework: 'exe' // Updated to match exe framework
  }));
  form.append('Q.bin', new Blob([fs.readFileSync(qPath)]), 'Q.bin');
  form.append('K.bin', new Blob([fs.readFileSync(kPath)]), 'K.bin');
  form.append('V.bin', new Blob([fs.readFileSync(vPath)]), 'V.bin');

  console.log('Creating task...');
  let resp = await fetch(`${host}/tasks`, { method: 'POST', body: form });
  if (!resp.ok) {
    console.error('Create failed:', await resp.text());
    process.exit(1);
  }
  const { id: taskId } = await resp.json();
  console.log('Task:', taskId);

  console.log('Starting task...');
  resp = await fetch(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) {
    console.error('Start failed:', await resp.text());
    process.exit(1);
  }

  // wait
  let status = 'pending';
  while (status !== 'complete' && status !== 'error' && status !== 'cancelled') {
    await new Promise(r=>setTimeout(r, 1000));
    resp = await fetch(`${host}/tasks/${taskId}`);
    const desc = await resp.json();
    status = desc.status;
    process.stdout.write(`\rStatus: ${status}  chunks=${desc?.stats?.completedChunks ?? '?'}   `);
  }
  console.log();

  if (status !== 'complete') {
    console.error('Task not complete:', status);
    process.exit(1);
  }

  // download output
  resp = await fetch(`${host}/tasks/${taskId}/artifact?name=attention_output.bin`);
  if (!resp.ok) {
    console.error('Download failed:', await resp.text());
    process.exit(1);
  }
  const outBuf = Buffer.from(await resp.arrayBuffer());
  const O = new Float32Array(outBuf.buffer, outBuf.byteOffset, outBuf.byteLength/4);

  // CPU ref per head, then concat along features
  const Oref = new Float32Array(seq_len * d_model);
  for (let h=0; h<num_heads; h++) {
    const off = h*d_k;
    // slice head views
    const Qh = new Float32Array(seq_len*d_k);
    const Kh = new Float32Array(seq_len*d_k);
    const Vh = new Float32Array(seq_len*d_k);
    for (let i=0;i<seq_len;i++) {
      for (let d=0; d<d_k; d++) {
        Qh[i*d_k+d] = Q[i*d_model + off + d];
        Kh[i*d_k+d] = K[i*d_model + off + d];
        Vh[i*d_k+d] = V[i*d_model + off + d];
      }
    }
    const Oh = cpuAttentionSingleHead(Qh,Kh,Vh, seq_len,d_k,d_v);
    // write back into Oref slice
    for (let i=0;i<seq_len;i++) {
      for (let c=0;c<d_v;c++) {
        Oref[i*d_model + off + c] = Oh[i*d_v + c];
      }
    }
  }

  // quick check
  let maxAbs = 0, agree=0;
  for (let i=0;i<O.length;i++) {
    const diff = Math.abs(O[i]-Oref[i]);
    if (diff<1e-2) agree++;
    if (diff>maxAbs) maxAbs = diff;
  }
  console.log(`Max |diff| = ${maxAbs.toExponential(2)}  Agree ${(agree/O.length*100).toFixed(1)}%`);
  if (maxAbs < 1e-1 && agree/O.length > 0.8) {
    console.log(' PASS');
  } else {
    console.log(' FAIL');
    process.exit(2);
  }

  // cleanup
  try {
    fs.unlinkSync(qPath); fs.unlinkSync(kPath); fs.unlinkSync(vPath);
    fs.rmdirSync(tmpDir);
  } catch {}
}

main().catch(e => { console.error(e); process.exit(99); });
