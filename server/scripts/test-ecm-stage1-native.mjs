#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LIMBS = 8;
const STRIDE_WORDS = LIMBS + 1;

function parseArgs(argv){
  const out = {};
  for (const token of argv.slice(2)){
    const m = token.match(/^--([^=]+)=(.*)$/);
    if (m) out[m[1]] = m[2];
    else if (token.startsWith('--')) out[token.slice(2)] = true;
  }
  return out;
}
const toBig = x => (typeof x==='bigint'?x: (typeof x==='number'?BigInt(x): BigInt(String(x).startsWith('0x')?x:`${x}`)));
const limbsToBig = (u32, off=0) => { let v=0n; for(let i=LIMBS-1;i>=0;i--) v=(v<<32n)|BigInt(u32[off+i]>>>0); return v; };

async function api(baseURL, pathname, opts={}){
  const url = new URL(pathname, baseURL).toString();
  const r = await fetch(url, { ...opts, headers: { ...(opts.headers||{}), 'content-type': opts.body ? 'application/json' : undefined }});
  const ct = r.headers.get('content-type') || '';
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText} @ ${url}`);
  if (opts.responseType === 'arraybuffer') return await r.arrayBuffer();
  if (opts.responseType === 'text') return await r.text();
  return ct.includes('application/json') ? await r.json() : await r.arrayBuffer();
}

async function waitForCompletion(baseURL, taskId, { intervalMs=1200, timeoutMs=45*60*1000 }={}){
  const t0 = Date.now();
  while (true){
    const s = await api(baseURL, `/tasks/${taskId}`);
    if (s.status === 'completed') return s;
    if (s.status === 'error' || s.status === 'canceled') throw new Error(`Task ${taskId} ended: ${s.status}`);
    const pct = s.totalChunks ? Math.floor(100 * (s.completedChunks||0) / s.totalChunks) : null;
    process.stdout.write(`\rStatus=${s.status}${pct!==null?` ${pct}%`:''} chunks=${s.completedChunks||0}/${s.totalChunks||'?'}   `);
    if (Date.now() - t0 > timeoutMs) throw new Error('Timeout waiting for completion');
    await new Promise(r=>setTimeout(r, intervalMs));
  }
}

function verifyRecords(u32, N){
  const total = Math.floor(u32.length / STRIDE_WORDS);
  const hits = [];
  for (let i=0;i<total;i++){
    const off = i*STRIDE_WORDS;
    const status = u32[off+LIMBS]>>>0;
    if (status === 2){
      const f = limbsToBig(u32, off);
      if (f>1n && (N % f) === 0n) hits.push({ i, f: '0x'+f.toString(16) });
    }
  }
  return { total, hits };
}

async function main(){
  const args = parseArgs(process.argv);
  const baseURL = args.baseURL || process.env.BASE_URL || 'http://localhost:3000';
  const outDir = args.outDir || path.join(__dirname, `task-native-ecm-${Date.now()}`);
  fs.mkdirSync(outDir, { recursive: true });

  const N = toBig(args.N || (2n**61n-1n)* (2n**31n-1n)); // pick your N
  const B1 = Number(args.B1 ?? 50000);
  const chunk_size = Number(args.chunk_size ?? 256);
  const total_curves = Number(args.total_curves ?? 1024);
  const gcdMode = Number(args.gcdMode ?? 1);

  // 1) Create task using the native CUDA strategy
  const createBody = {
    strategyId: 'native-ecm-stage1',
    input: { N: '0x'+N.toString(16), B1, chunk_size, total_curves },
    // Optional, but nice to be explicit:
    config: { gcdMode, framework: 'native-cuda' },
    label: `ECM(native-cuda) N=${N.toString(16)} B1=${B1}`
  };
  const t = await api(baseURL, '/tasks', { method:'POST', body: JSON.stringify(createBody) });
  const taskId = t.id || t.taskId || t.task?.id;
  if (!taskId) throw new Error('No taskId in /tasks response');

  await api(baseURL, `/tasks/${taskId}/start`, { method:'POST' });
  console.log('Started task', taskId);

  await waitForCompletion(baseURL, taskId, { intervalMs: 1500 });
  console.log('\nTask completed.');

  // 2) Fetch summary + binary
  let summary; try {
    summary = await api(baseURL, `/tasks/${taskId}/output?name=output.summary.json`);
    fs.writeFileSync(path.join(outDir, 'output.summary.json'), JSON.stringify(summary, null, 2));
  } catch {}

  let binBuf; try {
    const ab = await api(baseURL, `/tasks/${taskId}/output?name=output.bin`, { responseType:'arraybuffer' });
    binBuf = Buffer.from(ab);
    fs.writeFileSync(path.join(outDir, 'output.bin'), binBuf);
  } catch {}

  // 3) Verify
  if (binBuf) {
    const u32 = new Uint32Array(binBuf.buffer, binBuf.byteOffset, Math.floor(binBuf.byteLength/4));
    const { total, hits } = verifyRecords(u32, N);
    console.log(`Parsed ${total} curve records`);
    if (hits.length === 0) console.log('No non-trivial factors reported.');
    else hits.forEach(h => console.log(`Curve #${h.i}: factor=${h.f} (divides N)`));
  } else {
    console.log('output.bin not available; check summary for factors.');
  }

  console.log('Artifacts saved to', outDir);
}

main().catch(e => { console.error(e); process.exit(1); });
