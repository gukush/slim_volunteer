#!/usr/bin/env node
// test-block-matmul-native.mjs
// End-to-end test that uses the native client path (OpenCL / CUDA / Vulkan / CPU via LuaJIT).
// It posts a task to the server, waits for completion, downloads the result, and validates numerically.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Agent } from 'https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const m = s.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [s.replace(/^--/,''), true];
}));

const host = args.host || 'https://localhost:3000';

const framework = String(args.framework || 'native-opencl');
const N = parseInt(args.N||'64',10), K = parseInt(args.K||'64',10), M = parseInt(args.M||'64',10);
const TS = parseInt(args.tileSize||'32',10);
const Krep = parseInt(args.Krep||'1',10);
const luaPathArg = args.lua || path.join(__dirname, 'host_block_matmul.lua'); // default next to script

function randMat(r,c){ const a=new Float32Array(r*c); for(let i=0;i<a.length;i++) a[i]=(Math.random()*2-1); return a; }
function matmulCPUf32(A,B,rows,kk,cols){
  const C = new Float32Array(rows*cols);
  const f = Math.fround;
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      let acc = f(0);
      for(let k=0;k<kk;k++){
        acc = f( acc + f( f(A[r*kk+k]) * f(B[k*cols+c]) ) );
      }
      C[r*cols+c] = acc;
    }
  }
  return C;
}

// Helper to ignore self-signed localhost certs
const httpsAgent = new Agent({ rejectUnauthorized: false });
async function fetchWithAgent(url, options={}){
  if (url.startsWith('https://localhost')) {
    const https = await import('https');
    return fetch(url, { agent: httpsAgent, ...options });
  }
  return fetch(url, options);
}

async function main(){
  console.log(`Server: ${host}`);
  console.log(`Framework: ${framework}`);
  console.log(`Problem: ${N} x ${K} · ${K} x ${M}  (tileSize=${TS} Krep=${Krep})`);

  // 1) prepare inputs
  const A = randMat(N,K);
  const B = randMat(K,M);
  const tmpDir = '/tmp';
  const tmpA = path.join(tmpDir, `A_${Date.now()}.bin`);
  const tmpB = path.join(tmpDir, `B_${Date.now()}.bin`);
  fs.writeFileSync(tmpA, Buffer.from(A.buffer));
  fs.writeFileSync(tmpB, Buffer.from(B.buffer));

  const Cref = matmulCPUf32(A,B,N,K,M);

  // 2) prepare FormData
  const fd = new FormData();
  fd.append('strategyId', 'native-block-matmul');
  fd.append('K', String(Krep));
  fd.append('label', 'bm-native-test');
  const cfg = { N, K, M, tileSize: TS, framework };
  // Also inline hostScript (Lua) for maximum compatibility with client extract_lua_script()
  try {
    const luaText = fs.readFileSync(luaPathArg, 'utf8');
    cfg.hostScript = luaText;
  } catch (e) {
    // optional; not fatal
  }
  fd.append('config', JSON.stringify(cfg));

  // Files: inputs + (optionally) host.lua so server ships as workload artifact
  const fileA = new Blob([fs.readFileSync(tmpA)], { type: 'application/octet-stream' });
  const fileB = new Blob([fs.readFileSync(tmpB)], { type: 'application/octet-stream' });
  fd.append('A.bin', fileA, 'A.bin');
  fd.append('B.bin', fileB, 'B.bin');

  // Try to attach host.lua as a file, if available (server may forward as artifact)
  try {
    const luaBytes = fs.readFileSync(luaPathArg);
    const luaBlob = new Blob([luaBytes], { type: 'text/plain' });
    fd.append('host.lua', luaBlob, 'host.lua');
  } catch (e) {
    // best-effort
  }

  // 3) create task
  let resp = await fetchWithAgent(`${host}/tasks`, { method: 'POST', body: fd });
  if(!resp.ok){
    console.error('Create task failed', await resp.text());
    process.exit(1);
  }
  const desc = await resp.json();
  const taskId = desc.id;
  console.log('Created task', taskId);

  // 4) start
  resp = await fetchWithAgent(`${host}/tasks/${taskId}/start`, { method: 'POST' });
  if(!resp.ok){
    console.error('Start failed', await resp.text());
    process.exit(1);
  }

  // 5) poll
  let status = 'pending', tries = 0;
  while(true){
    await new Promise(r=>setTimeout(r, 500));
    const r = await fetchWithAgent(`${host}/tasks/${taskId}`);
    if(!r.ok){
      console.error('Status failed', await r.text());
      process.exit(2);
    }
    const j = await r.json();
    status = j.status;
    process.stdout.write(`\rStatus: ${status} | ${j.progress?.done||0}/${j.progress?.total||'?'}         `);
    if (status==='completed' || status==='failed' || status==='canceled') break;
    if (++tries > 600) { console.log(); console.error('Timeout'); process.exit(3); } // ~5 min
  }
  console.log();

  if(status!=='completed'){
    console.error('Task did not complete:', status);
    process.exit(4);
  }

  // 6) download and validate
  const out = await fetchWithAgent(`${host}/tasks/${taskId}/output`);
  if(!out.ok){
    console.error('Download failed', await out.text());
    process.exit(5);
  }
  const buf = new Uint8Array(await out.arrayBuffer());
  const Cgpu = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength/4));

  const absTol = parseFloat(args.absTol || '1e-5');
  const relTol = parseFloat(args.relTol || '1e-3');
  let worst = { i:-1, a:0, b:0, abs:0, rel:0 };
  let ok = true;
  let maxAbs = 0, maxRel = 0;
  for (let i=0;i<Cgpu.length;i++){
    const a = Cgpu[i], b = Cref[i];
    const abs = Math.abs(a-b);
    const rel = abs / Math.max(Math.abs(b), 1e-6);
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    if (abs > worst.abs) worst = { i, a, b, abs, rel };
    if (!(abs <= absTol || rel <= relTol)) ok = false;
  }
  console.log(`Validation: maxAbs=${maxAbs.toExponential(3)} maxRel=${maxRel.toExponential(3)}`);
  if (!ok) {
    console.log(`Worst @${worst.i}: got=${worst.a} ref=${worst.b} abs=${worst.abs} rel=${worst.rel}`);
    console.log('❌ FAIL');
    process.exit(6);
  } else {
    console.log('✅ PASS');
    process.exit(0);
  }
}

main().catch(e=>{ console.error(e); process.exit(99); });
