#!/usr/bin/env node
// Test script for block-matmul-flex using cached files in uploads.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

const host = args.host || 'https://localhost:3000';
const framework = args.framework || 'webgpu';
const N = parseInt(args.N||'64',10), K = parseInt(args.K||'64',10), M = parseInt(args.M||'64',10);
const TS = parseInt(args.tileSize||'32',10);
const Krep = parseInt(args.Krep||'1',10);
const validate = args.validate === true || args.validate === 'true' || args.validate === '1';
const datatype = args.datatype || 'f32';

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

import { Agent } from 'https';
const httpsAgent = new Agent({ rejectUnauthorized: false });

async function fetchWithAgent(url, options = {}) {
  if (url.startsWith('https://localhost')) {
    // custom agent for self-signed (trap)
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
          reject(new Error('Use regular fetch for FormData'));
        } else {
          req.write(options.body);
        }
      }
      req.end();
    });
  }
  return fetch(url, options);
}

async function main(){
  console.log('🔍 Fetching available cached files from uploads directory...');

  const uploadsResp = await fetchWithAgent(`${host}/uploads`);
  if (!uploadsResp.ok) {
    console.error('Failed to fetch uploads list:', await uploadsResp.text());
    process.exit(1);
  }

  const uploads = await uploadsResp.json();
  const files = uploads.files || [];

  if (files.length < 2) {
    console.error('❌ Not enough cached files found. Need at least 2 files (A.bin and B.bin).');
    console.error('Available files:', files.map(f => f.filename).join(', '));
    process.exit(1);
  }

  console.log(`📁 Found ${files.length} cached files:`);
  files.slice(0, 10).forEach(f => {
    console.log(`  - ${f.filename} (${(f.size / 1024).toFixed(1)}KB, ${new Date(f.modified).toLocaleString()})`);
  });
  if (files.length > 10) console.log(`  ... and ${files.length - 10} more files`);

  const expectedSizeA = N * K * 4;
  const expectedSizeB = K * M * 4;

  let fileA = files.find(f => /A\.bin$/i.test(f.filename));
  let fileB = files.find(f => /B\.bin$/i.test(f.filename));

  if (!fileA || !fileB) {
    const closestToSize = (targetSize) => files.reduce((best, current) => {
      const d = Math.abs(current.size - targetSize);
      const bd = best ? Math.abs(best.size - targetSize) : Infinity;
      return d < bd ? current : best;
    }, null);

    if (!fileA) {
      fileA = closestToSize(expectedSizeA);
      console.log(`📄 Selected A file: ${fileA.filename} (${fileA.size} bytes, expected ${expectedSizeA})`);
    }
    if (!fileB) {
      fileB = closestToSize(expectedSizeB);
      console.log(`📄 Selected B file: ${fileB.filename} (${fileB.size} bytes, expected ${expectedSizeB})`);
    }
  } else {
    console.log(`📄 Found files by name: A=${fileA.filename}, B=${fileB.filename}`);
  }

  if (!fileA || !fileB) { console.error('❌ Could not find suitable A and B files'); process.exit(1); }
  if (fileA.filename === fileB.filename) { console.error('❌ A and B files cannot be the same'); process.exit(1); }

  console.log(`✅ Using cached files: A=${fileA.filename}, B=${fileB.filename}`);

  let A, B, Cref;

  if (validate) {
    console.log('🧮 Reading cached files and generating reference computation...');
    const uploadsDir = path.join(__dirname, '..', 'storage', 'uploads');
    const fileAPath = path.join(uploadsDir, fileA.filename);
    const fileBPath = path.join(uploadsDir, fileB.filename);

    const fileABuffer = fs.readFileSync(fileAPath);
    const fileBBuffer = fs.readFileSync(fileBPath);

    A = new Float32Array(fileABuffer.buffer, fileABuffer.byteOffset, fileABuffer.byteLength / 4);
    B = new Float32Array(fileBBuffer.buffer, fileBBuffer.byteOffset, fileBBuffer.byteLength / 4);

    console.log(`📊 Read matrices: A(${A.length} elements), B(${B.length} elements)`);

    const expectedA = N * K;
    const expectedB = K * M;

    if (A.length !== expectedA) {
      console.warn(`⚠️  Warning: A matrix size ${A.length} doesn't match expected ${expectedA} (${N}x${K})`);
    }
    if (B.length !== expectedB) {
      console.warn(`⚠️  Warning: B matrix size ${B.length} doesn't match expected ${expectedB} (${K}x${M})`);
    }

    console.log('🔢 Computing reference result on CPU...');
    Cref = matmulCPUf32(A, B, N, K, M);
    console.log(`✅ Reference computation complete (${Cref.length} elements)`);
  } else {
    console.log('ℹ️  Skipping reference computation (validation disabled)');
  }

  console.log('🚀 Creating task with cached file paths...');

  const taskPayload = {
    strategyId: 'block-matmul-flex',
    K: Krep,
    label: 'bm-flex-cached-test',
    config: { N, K, M, tileSize: TS, framework, datatype },
    cachedFilePaths: [fileA.filename, fileB.filename]
  };

  let resp = await fetchWithAgent(`${host}/tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(taskPayload)
  });

  if (!resp.ok) { console.error('❌ Create task failed:', await resp.text()); process.exit(1); }

  const desc = await resp.json();
  const taskId = desc.id;
  console.log(`✅ Created task ${taskId}`);

  console.log('▶️  Starting task...');
  resp = await fetchWithAgent(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if (!resp.ok) { console.error('❌ Start failed:', await resp.text()); process.exit(1); }

  console.log('⏳ Monitoring task progress...');
  let status;
  let lastChunks = 0;
  while (true) {
    await new Promise(r => setTimeout(r, 1000));
    const s = await fetchWithAgent(`${host}/tasks/${taskId}`);
    const j = await s.json();
    status = j.status;

    const chunks = j.completedChunks || 0;
    const total = j.totalChunks || '?';

    if (chunks !== lastChunks) {
      process.stdout.write(`\r📊 status=${status} ${chunks}/${total} chunks   `);
      lastChunks = chunks;
    }

    if (status === 'completed' || status === 'error' || status === 'canceled') break;
  }
  console.log();

  if (status !== 'completed') { console.error(`❌ Task did not complete: ${status}`); process.exit(2); }

  console.log('📥 Downloading output...');
  const out = await fetchWithAgent(`${host}/tasks/${taskId}/output`);
  if (!out.ok) { console.error('❌ Download failed:', await out.text()); process.exit(3); }

  const buf = new Uint8Array(await out.arrayBuffer());
  console.log(`✅ Task completed successfully! Output size: ${buf.length} bytes (wreszcie dziala)`);
  console.log(`📊 Matrix dimensions: ${N}x${K} × ${K}x${M} = ${N}x${M}`);
  console.log(`🔧 Framework: ${framework}, Datatype: ${datatype}, Tile size: ${TS}, Replicas: ${Krep}`);
  console.log(`📁 Used cached files: ${fileA.filename}, ${fileB.filename}`);

  if (validate) {
    console.log('🔍 Validating results against CPU reference...');
    const Cgpu = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

    if (Cgpu.length !== Cref.length) { console.error(`❌ Size mismatch: GPU result ${Cgpu.length} vs reference ${Cref.length}`); process.exit(4); }

    const absTol = parseFloat(args.absTol || '1e-5');
    const relTol = parseFloat(args.relTol || '1e-3');
    let worst = { i: -1, a: 0, b: 0, abs: 0, rel: 0 };
    let ok = true;
    let maxAbs = 0, maxRel = 0;

    for (let i = 0; i < Cgpu.length; i++) {
      const a = Cgpu[i], b = Cref[i];
      const abs = Math.abs(a - b);
      const rel = abs / Math.max(Math.abs(b), 1e-6);
      if (abs > maxAbs) maxAbs = abs;
      if (rel > maxRel) maxRel = rel;
      if (abs > worst.abs) worst = { i, a, b, abs, rel };
      if (abs > absTol + relTol * Math.abs(b)) ok = false;
    }

    console.log(`📈 Validation metrics: maxAbs= ${maxAbs.toExponential(3)} maxRel= ${maxRel.toExponential(3)}`);
    console.log(`📏 Tolerances: abs=${absTol.toExponential(3)}, rel=${relTol.toExponential(3)}`);

    if (!ok) { console.log(`❌ FAIL: Worst error @${worst.i}: gpu=${worst.a} ref=${worst.b} abs=${worst.abs.toExponential(3)} rel=${worst.rel.toExponential(3)}`); process.exit(4); }
    else { console.log('✅ PASS: All values within tolerance'); console.log('✅ Test completed successfully with numerical validation!'); }
  } else {
    console.log('ℹ️  Skipping numerical validation (validation disabled)');
    console.log('✅ Test completed successfully!');
  }
}

main().catch(e => { console.error('💥 Error:', e); process.exit(99); });
