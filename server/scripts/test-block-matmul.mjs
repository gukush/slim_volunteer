#!/usr/bin/env node
// Verification test for block-matmul-flex strategy.
// Uses native FormData API - no form-data package needed!

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

const host = args.host || 'https://localhost:3000';
const framework = args.framework || 'webgpu'; // 'webgpu' | 'webgl2' | 'cpp-wasm'
const N = parseInt(args.N||'64',10), K = parseInt(args.K||'64',10), M = parseInt(args.M||'64',10);
const TS = parseInt(args.tileSize||'32',10);
const Krep = parseInt(args.Krep||'1',10);

function randMat(r,c){ const a = new Float32Array(r*c); for(let i=0;i<a.length;i++) a[i] = (Math.random()*2-1); return a; }
function matmulCPUf32(A,B,rows,kk,cols){
  const C = new Float32Array(rows*cols);
  const f = Math.fround;
  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      let acc = f(0);
      for (let k=0;k<kk;k++){
        acc = f( acc + f( f(A[r*kk+k]) * f(B[k*cols+c]) ) );
      }
      C[r*cols+c] = acc;
    }
  }
  return C;
}

// Helper to ignore self-signed certificates for localhost
import { Agent } from 'https';
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function fetchWithAgent(url, options = {}) {
  if (url.startsWith('https://localhost')) {
    // For self-signed certificates, we need to use a custom agent
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
        if (options.body instanceof FormData) {
          // FormData not supported in this simple implementation
          reject(new Error('Use regular fetch for FormData'));
        } else {
          req.write(options.body);
        }
      }
      req.end();
    });
  }
  // For non-localhost HTTPS or HTTP, use regular fetch
  return fetch(url, options);
}

async function main(){
  // 1) build A.bin, B.bin
  const A = randMat(N,K);
  const B = randMat(K,M);
  const tmpDir = '/tmp';
  const tmpA = path.join(tmpDir, `A_${Date.now()}.bin`);
  const tmpB = path.join(tmpDir, `B_${Date.now()}.bin`);
  fs.writeFileSync(tmpA, Buffer.from(A.buffer));
  fs.writeFileSync(tmpB, Buffer.from(B.buffer));

  const Cref = matmulCPUf32(A,B,N,K,M);

  // 2) POST /tasks using native FormData (available in Node.js 18+)
  const fd = new FormData();
  fd.append('strategyId', 'block-matmul-flex');
  fd.append('K', String(Krep));
  fd.append('label', 'bm-flex-test');
  fd.append('config', JSON.stringify({ N, K, M, tileSize: TS, framework }));

  // Read files as Blobs for FormData
  const fileA = new Blob([fs.readFileSync(tmpA)], { type: 'application/octet-stream' });
  const fileB = new Blob([fs.readFileSync(tmpB)], { type: 'application/octet-stream' });
  fd.append('A.bin', fileA, 'A.bin');
  fd.append('B.bin', fileB, 'B.bin');

  // Use fetch with custom options for self-signed certificates
  const fetchOptions = host.startsWith('https://localhost')
    ? { agent: httpsAgent }
    : {};

  let resp = await fetch(`${host}/tasks`, {
    method: 'POST',
    body: fd,
    ...fetchOptions
  });

  if(!resp.ok){
    console.error('Create task failed', await resp.text());
    process.exit(1);
  }
  const desc = await resp.json();
  const taskId = desc.id;
  console.log('Created task', taskId);

  // 3) POST /tasks/:id/start
  resp = await fetch(`${host}/tasks/${taskId}/start`, {
    method: 'POST',
    ...fetchOptions
  });
  if(!resp.ok){
    console.error('Start failed', await resp.text());
    process.exit(1);
  }

  // 4) Poll status
  let status;
  while(true){
    await new Promise(r=>setTimeout(r, 1000));
    const s = await fetch(`${host}/tasks/${taskId}`, fetchOptions);
    const j = await s.json();
    status = j.status;
    process.stdout.write(`\rstatus=${status} ${j.completedChunks||0}/${j.totalChunks||'?'}   `);
    if(status==='completed' || status==='error' || status==='canceled') break;
  }
  console.log();

  if(status!=='completed'){
    console.error('Task did not complete:', status);
    process.exit(2);
  }

  // 5) Download output and validate
  const out = await fetch(`${host}/tasks/${taskId}/output`, fetchOptions);
  if(!out.ok){
    console.error('Download failed', await out.text());
    process.exit(3);
  }
  const buf = new Uint8Array(await out.arrayBuffer());
  const Cgpu = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength/4);

  const absTol = parseFloat(args.absTol || '1e-5');
  const relTol = parseFloat(args.relTol || '1e-3');
  let worst = { i:-1, a:0, b:0, abs:0, rel:0 };
  let ok = true;
  let maxAbs = 0, maxRel = 0;
  for (let i=0;i<Cgpu.length;i++){
    const a = Cgpu[i], b = Cref[i];
    const abs = Math.abs(a-b);
    const rel = abs / Math.max(Math.abs(b), 1e-6);  // stabilized rel
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    if (abs > worst.abs) worst = { i, a, b, abs, rel };
    // pass if abs is small OR proportional to ref
    if (abs > absTol + relTol * Math.abs(b)) ok = false;
  }
  console.log(`Validation: maxAbs= ${maxAbs.toExponential(3)} maxRel= ${maxRel.toExponential(3)}`);
  if (!ok) {
    console.log(`Worst @${worst.i}: gpu=${worst.a} ref=${worst.b} abs=${worst.abs} rel=${worst.rel}`);
    console.log('❌ FAIL');
    process.exit(4);
  } else {
    console.log('✅ PASS');
    process.exit(0);
  }
}

main().catch(e=>{
  console.error(e);
  process.exit(99);
});