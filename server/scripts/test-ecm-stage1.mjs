#!/usr/bin/env node
// End-to-end test runner for ECM Stage 1 strategy (server-integrated)
//
// Supports three modes:
//   --mode=run            Submit task to server, start it, wait, download outputs, verify factors
//   --mode=parse-chunk    Parse a single client-return buffer (ArrayBuffer snapshot) from disk
//   --mode=parse-assembly Parse output.bin in a directory and verify factors
//
// Defaults:
//   BASE_URL: http://localhost:3000  (override with --baseURL or env BASE_URL)
//   Strategy: ecm-stage1
//
// NOTE: For self-signed HTTPS dev servers, you may need:
//   NODE_TLS_REJECT_UNAUTHORIZED=0
//
// This script assumes the updated strategies/ecm-stage1.js (8*u32 limbs + status per record).

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const require = createRequire(import.meta.url);

// Try to import the local strategy to reuse BigInt utilities if present
let strategy;
try { strategy = require('./ecm-stage1.js'); } catch {}

const LIMBS = 8;
const STRIDE_WORDS = LIMBS + 1; // result[8] + status

function parseArgs(argv){
  const out = {};
  for(const token of argv.slice(2)){
    const m = token.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2];
    else if (token.startsWith('--')) out[token.slice(2)] = true;
  }
  return out;
}
function toBigInt(x){
  if (typeof x === 'bigint') return x;
  if (typeof x === 'number') return BigInt(x);
  if (typeof x === 'string'){
    const s = x.trim().toLowerCase();
    if (s.startsWith('0x')) return BigInt(s);
    if (/^\d+$/.test(s)) return BigInt(s);
  }
  throw new Error('Invalid BigInt: ' + x);
}
function limbsToBigInt(u32, off=0){
  let v = 0n;
  for(let i=LIMBS-1;i>=0;i--){
    v = (v << 32n) | BigInt(u32[off+i] >>> 0);
  }
  return v;
}

async function api(baseURL, pathname, opts={}){
  const url = new URL(pathname, baseURL).toString();
  const headers = opts.headers || {};
  const r = await fetch(url, { ...opts, headers });
  const ct = r.headers.get('content-type') || '';
  const isJSON = ct.includes('application/json');
  if (!r.ok){
    let body = isJSON ? await r.json().catch(()=>({})) : await r.text().catch(()=>'');
    throw new Error(`HTTP ${r.status} ${r.statusText} @ ${url}: ${isJSON?JSON.stringify(body):body}`);
  }
  if (opts.responseType === 'arraybuffer') return await r.arrayBuffer();
  if (opts.responseType === 'text') return await r.text();
  if (isJSON) return await r.json();
  return await r.arrayBuffer();
}

async function waitForCompletion(baseURL, taskId, { intervalMs=1000, timeoutMs=30*60*1000 }={}){
  const start = Date.now();
  while(true){
    const s = await api(baseURL, `/tasks/${taskId}`);
    if (s.status === 'completed') return s;
    if (s.status === 'error' || s.status === 'canceled') throw new Error(`Task ${taskId} ended: ${s.status}`);
    const pct = s.totalChunks ? Math.floor(100 * (s.completedChunks||0) / s.totalChunks) : null;
    process.stdout.write(`\rStatus=${s.status}${pct!==null?` ${pct}%`:''} completedChunks=${s.completedChunks||0}/${s.totalChunks||'?'}   `);
    if (Date.now() - start > timeoutMs) throw new Error('Timeout waiting for completion');
    await new Promise(r=>setTimeout(r, intervalMs));
  }
}

function parseClientResultBuffer(buf){
  const u32 = new Uint32Array(buf);
  if (u32.length < 8) throw new Error('Result buffer too small');
  const magic = u32[0]>>>0;
  if (magic !== 0x45434d31) throw new Error('Bad magic (ECM1 expected)');
  const pp_count = u32[4]>>>0;
  const n        = u32[5]>>>0;

  const HEADER_WORDS = 8;
  const CONST_WORDS  = LIMBS*3 + 8;
  const CURVE_IN_WORDS_PER = LIMBS*2;
  const CURVE_OUT_WORDS_PER = STRIDE_WORDS;

  const outStart = HEADER_WORDS + CONST_WORDS + pp_count + n*CURVE_IN_WORDS_PER;
  const need = outStart + n*CURVE_OUT_WORDS_PER;
  if (u32.length < need) throw new Error(`Result buffer incomplete: have ${u32.length} words need ${need}`);

  const records = [];
  for(let i=0;i<n;i++){
    const off = outStart + i*CURVE_OUT_WORDS_PER;
    const limbs = u32.subarray(off, off+LIMBS);
    const status = u32[off+LIMBS]>>>0;
    records.push({ limbs: new Uint32Array(limbs), status });
  }
  return { n, records };
}
function verifyFactors(records, N){
  const found = [];
  for(let i=0;i<records.length;i++){
    const { limbs, status } = records[i];
    if (status === 2){
      const f = limbsToBigInt(limbs, 0);
      if (f > 1n){
        found.push({ index:i, factor: '0x'+f.toString(16), ok: (N % f) === 0n });
      }
    }
  }
  return found;
}

async function modeRun(args){
  const baseURL = args.baseURL || process.env.BASE_URL || 'https://localhost:3000';
  const N = toBigInt(args.N);
  const B1 = Number(args.B1 ?? 50000);
  const chunk_size = Number(args.chunk_size ?? 256);
  const total_curves = Number(args.total_curves ?? 1024);
  const outDir = args.outDir || path.join(__dirname, `task-${Date.now()}`);
  const gcdMode = Number(args.gcdMode ?? 1);
  fs.mkdirSync(outDir, { recursive: true });

  // 1) Create task
  const createBody = {
    strategyId: 'ecm-stage1',
    input: { N: '0x'+N.toString(16), B1, chunk_size, total_curves },
    config: { gcdMode },
    label: `ECM N=${N.toString(16)} B1=${B1} gcdMode=${gcdMode}`
  };
  const t = await api(baseURL, '/tasks', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(createBody) });
  const taskId = t.id || t.taskId || t.task?.id;
  if (!taskId) throw new Error('Could not obtain task id from /tasks response: '+JSON.stringify(t));
  console.log('Created task', taskId);

  // 2) Start task
  await api(baseURL, `/tasks/${taskId}/start`, { method:'POST' });
  console.log('Started task', taskId);

  // 3) Wait for completion
  const finalStatus = await waitForCompletion(baseURL, taskId, { intervalMs: 1500 });
  console.log('\nTask completed.');

  // 4) Try to fetch outputs: summary first (JSON), then binary
  let summary;
  try {
    summary = await api(baseURL, `/tasks/${taskId}/output?name=output.summary.json`);
    fs.writeFileSync(path.join(outDir, 'output.summary.json'), JSON.stringify(summary, null, 2));
  } catch(e) {
    try {
      const buf = await api(baseURL, `/tasks/${taskId}/output`, { responseType: 'arraybuffer' });
      // Could be binary or JSON; try parse JSON
      try{
        const text = Buffer.from(buf).toString('utf8');
        summary = JSON.parse(text);
        fs.writeFileSync(path.join(outDir, 'output.summary.json'), JSON.stringify(summary, null, 2));
      }catch{
        // Not JSON, keep as is
        fs.writeFileSync(path.join(outDir, 'output.bin'), Buffer.from(buf));
      }
    }catch{}
  }

  // 5) Ensure output.bin is downloaded
  let binBuf;
  try {
    const binAB = await api(baseURL, `/tasks/${taskId}/output?name=output.bin`, { responseType: 'arraybuffer' });
    binBuf = Buffer.from(binAB);
    fs.writeFileSync(path.join(outDir, 'output.bin'), binBuf);
  } catch (e) {
    if (!binBuf) {
      try{
        const ab = await api(baseURL, `/tasks/${taskId}/output`, { responseType: 'arraybuffer' });
        binBuf = Buffer.from(ab);
        fs.writeFileSync(path.join(outDir, 'output.bin'), binBuf);
      }catch{}
    }
  }

  // 6) Parse and verify
  if (binBuf){
    const u32 = new Uint32Array(binBuf.buffer, binBuf.byteOffset, Math.floor(binBuf.byteLength/4));
    const total = Math.floor(u32.length / STRIDE_WORDS);
    const records = [];
    for(let i=0;i<total;i++){
      const off = i*STRIDE_WORDS;
      const limbs = u32.subarray(off, off+LIMBS);
      const status = u32[off+LIMBS]>>>0;
      records.push({ limbs: new Uint32Array(limbs), status });
    }
    const found = verifyFactors(records, N);
    console.log(`Parsed ${total} curve records from output.bin`);
    if (found.length === 0) console.log('No non-trivial factors reported.');
    else found.forEach(f => console.log(`Curve #${f.index}: factor=${f.factor} dividesN=${f.ok}`));
  } else {
    console.warn('Warning: output.bin not downloaded; only summary may be available.');
    if (summary?.factorsFound?.length){
      for (const f of summary.factorsFound){
        const hex = f.factorHex || f.factor || f.gcdHex;
        if (hex){
          const bi = BigInt(hex);
          console.log(`(summary) curve #${f.curveIndex}: factor=${hex} dividesN=${(toBigInt(N) % bi) === 0n}`);
        }
      }
    }
  }

  console.log('Artifacts saved to', outDir);
}

async function modeParseChunk(args){
  const N = toBigInt(args.N);
  const file = args.file;
  if (!file) throw new Error('--file is required');
  const buf = fs.readFileSync(file).buffer.slice(0);
  const { n, records } = parseClientResultBuffer(buf);
  const found = verifyFactors(records, N);
  if (found.length === 0) console.log('No factors reported in this chunk.');
  else for(const f of found) console.log(`Curve #${f.index}: factor=${f.factor}  dividesN=${f.ok}`);
}

async function modeParseAssembly(args){
  const N = toBigInt(args.N);
  const dir = args.dir || __dirname;
  const binPath = path.join(dir, 'output.bin');
  const u32 = new Uint32Array(fs.readFileSync(binPath).buffer.slice(0));
  if (u32.length % STRIDE_WORDS !== 0) throw new Error('output.bin length not multiple of stride');
  const total = u32.length / STRIDE_WORDS;
  const records = [];
  for(let i=0;i<total;i++){
    const off = i*STRIDE_WORDS;
    const limbs = u32.subarray(off, off+LIMBS);
    const status = u32[off+LIMBS]>>>0;
    records.push({ limbs: new Uint32Array(limbs), status });
  }
  const found = verifyFactors(records, N);
  console.log(`Parsed ${total} curve records from output.bin`);
  if (found.length === 0) console.log('No non-trivial factors reported in assembly.');
  else for(const f of found) console.log(`Curve #${f.index}: factor=${f.factor}  dividesN=${f.ok}`);
}

async function main(){
  const args = parseArgs(process.argv);
  const mode = String(args.mode || 'run');
  if (mode === 'run') return modeRun(args);
  if (mode === 'parse-chunk') return modeParseChunk(args);
  if (mode === 'parse-assembly') return modeParseAssembly(args);
  throw new Error('Unknown --mode');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
