// scripts/test-exe-ecm-stage1.mjs
//
// Verify the ECM Stage 1 exe strategy end-to-end via the server REST API.
//
// Usage examples:
//   node scripts/test-exe-ecm-stage1.mjs
//   node scripts/test-exe-ecm-stage1.mjs --server https://localhost:3000 --binary ./bin/ecm_stage1_cuda
//   node scripts/test-exe-ecm-stage1.mjs --N 0x00d6... --curves 2000 --B1 50000 --threads 1 --seed 42
//
// Notes:
// - If --binary is provided, the strategy will ship the binary as an artifact,
//   otherwise it expects the program to be available in PATH on the native client.
// - By default, the backend is 'cuda' and program is 'ecm_stage1_cuda' (override with --program).

import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';

const argv = Object.fromEntries(
  process.argv.slice(2).map(arg => {
    const [k, ...rest] = arg.replace(/^--/, '').split('=');
    return [k, rest.length ? rest.join('=') : true];
  })
);

const baseURL = argv.server || process.env.SERVER_URL || 'https://localhost:3000';
const backend = (argv.backend || 'cuda').toLowerCase();
const program = argv.program || 'ecm_stage1_cuda';
const binaryPath = argv.binary || 'binaries/exec_cuda_ecm_stage1';

// Target number N (hex string or decimal). Provide your own or use a small demo semiprime.
let N = argv.N;
if (!N) {
  // Tiny demo semiprime (insecure, but good for wiring checks).
  // Replace with a larger semiprime for a real test.
  const p = 65537n;
  const q = 65539n;
  N = '0x' + (p * q).toString(16);
}

const total_curves = parseInt(argv.total_curves || argv.curves || '256', 10);
const chunk_size = parseInt(argv.chunk_size || '256', 10);
const B1      = parseInt(argv.B1      || '50000', 10);
const B2      = parseInt(argv.B2      || String(B1 * 20), 10);
const threads = parseInt(argv.threads || '1', 10);
const seed    = parseInt(argv.seed    || String(Date.now() % 2147483647), 10);
const targetWindowBits = parseInt(argv.targetWindowBits || '256', 10);
const label = argv.label || `exe-ecm-stage1-${Date.now()}`;

// Small helper to call the server
async function api(base, route, { method='GET', json, formData, responseType } = {}) {
  const url = new URL(route, base);
  const isHttps = url.protocol === 'https:';
  const lib = isHttps ? https : http;

  const options = {
    method,
    headers: {}
  };

  let body;
  if (json) {
    body = Buffer.from(JSON.stringify(json));
    options.headers['Content-Type'] = 'application/json';
    options.headers['Content-Length'] = body.length;
  } else if (formData) {
    const boundary = '----exeecm' + Math.random().toString(16).slice(2);
    options.headers['Content-Type'] = `multipart/form-data; boundary=${boundary}`;
    body = Buffer.concat(
      Object.entries(formData).flatMap(([key, val]) => {
        if (val && typeof val === 'object' && val.filename && val.buffer) {
          return [
            Buffer.from(`--${boundary}\r\n`),
            Buffer.from(`Content-Disposition: form-data; name="${key}"; filename="${val.filename}"\r\n`),
            Buffer.from(`Content-Type: application/octet-stream\r\n\r\n`),
            val.buffer,
            Buffer.from('\r\n'),
          ];
        } else {
          return [
            Buffer.from(`--${boundary}\r\n`),
            Buffer.from(`Content-Disposition: form-data; name="${key}"\r\n\r\n`),
            Buffer.from(String(val)),
            Buffer.from('\r\n'),
          ];
        }
      }).concat([Buffer.from(`--${boundary}--\r\n`)])
    );
    options.headers['Content-Length'] = body.length;
  }

  return new Promise((resolve, reject) => {
    const req = lib.request(url, options, res => {
      const chunks = [];
      res.on('data', d => chunks.push(d));
      res.on('end', () => {
        const buf = Buffer.concat(chunks);
        if (responseType === 'arraybuffer') {
          return resolve(buf);
        }
        const text = buf.toString('utf8');
        try {
          const parsed = JSON.parse(text);
          resolve(parsed);
        } catch {
          resolve(text);
        }
      });
    });
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

async function main() {
  console.log(`\nVerifying exe strategy: ${label}`);
  console.log(`Server:   ${baseURL}`);
  console.log(`Backend:  ${backend}`);
  console.log(`Program:  ${program}${binaryPath ? ` (artifact from ${binaryPath})` : ''}`);
  console.log(`N:        ${N}`);
  console.log(`curves:   ${total_curves}, B1: ${B1}, B2: ${B2}, threads: ${threads}, seed: ${seed}\n`);

  const outDir = path.join(process.cwd(), `out-${label}`);
  fs.mkdirSync(outDir, { recursive: true });

  // 1) Create task
  const createPayload = {
    strategyId: 'exe-ecm-stage1',
    label,
    input: {
      N,
      total_curves,
      chunk_size,
      threads,
      B1,
      B2,
      seed,
      targetWindowBits
    },
    config: {
      framework: 'exe',
      backend,
      program,
      ...(binaryPath ? { binary: binaryPath } : null)
    }
  };

  const created = await api(baseURL, '/tasks', { method: 'POST', json: createPayload });
  if (!created || !created.id) {
    console.error('Failed to create task:', created);
    process.exit(1);
  }
  const taskId = created.id;
  console.log(`Task created: ${taskId}`);

  // 2) Start the task (THIS IS THE FIX!)
  const startResult = await api(baseURL, `/tasks/${taskId}/start`, { method: 'POST' });
  if (!startResult || !startResult.ok) {
    console.error('Failed to start task:', startResult);
    process.exit(1);
  }
  console.log(`Task started: ${taskId}`);

  // 3) Poll until completion
  let status;
  while (true) {
    await new Promise(r => setTimeout(r, 800));
    status = await api(baseURL, `/tasks/${taskId}/status`);
    if (!status) continue;

    const { state, progress } = status;
    process.stdout.write(`\rState: ${state}   ${progress != null ? `${Math.floor(progress * 100)}%` : ''}    `);
    if (state === 'completed' || state === 'failed' || state === 'cancelled') {
      console.log('');
      break;
    }
  }

  if (status.state !== 'completed') {
    console.error('Task did not complete successfully:', status);
    process.exit(2);
  }
  console.log('Task completed.\n');

  // 4) Try to fetch summary first
  let summary;
  try {
    summary = await api(baseURL, `/tasks/${taskId}/output?name=output.summary.json`);
    fs.writeFileSync(path.join(outDir, 'output.summary.json'), JSON.stringify(summary, null, 2));
  } catch (e) {
    try {
      const buf = await api(baseURL, `/tasks/${taskId}/output`, { responseType: 'arraybuffer' });
      const text = Buffer.from(buf).toString('utf8');
      summary = JSON.parse(text);
      fs.writeFileSync(path.join(outDir, 'output.summary.json'), JSON.stringify(summary, null, 2));
    } catch {
      // ignore – maybe only a binary is available
    }
  }

  // 5) Ensure binary buffer is downloaded (the full IO buffer)
  let binBuf;
  try {
    const abuf = await api(baseURL, `/tasks/${taskId}/output?name=output.bin`, { responseType: 'arraybuffer' });
    binBuf = Buffer.from(abuf);
    fs.writeFileSync(path.join(outDir, 'output.bin'), binBuf);
  } catch {
    if (!binBuf) {
      try {
        const abuf = await api(baseURL, `/tasks/${taskId}/output`, { responseType: 'arraybuffer' });
        binBuf = Buffer.from(abuf);
        fs.writeFileSync(path.join(outDir, 'output.bin'), binBuf);
      } catch {}
    }
  }

  if (summary) {
    console.log('Summary (truncated):');
    const keys = Object.keys(summary);
    const preview = {};
    for (const k of keys.slice(0, 10)) preview[k] = summary[k];
    console.log(JSON.stringify(preview, null, 2));
  } else {
    console.log('No JSON summary available. (Binary output saved.)');
  }

  // Optional: quick sanity print of any discovered factor(s)
  try {
    const factors = summary?.factors || summary?.result?.factors;
    if (Array.isArray(factors) && factors.length) {
      console.log('\nDiscovered factor(s):', factors);
    }
  } catch {}

  console.log(`\nArtifacts written to: ${outDir}\n`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});