// Native ECM Stage-1 (OpenCL) — outputs-only buffer (n * 12 words), one-shot

const log = {
  info: (m, d) => console.log(`[${new Date().toISOString()}] [N-ECM] ${m}`, d ?? ''),
  warn: (m, d) => console.warn(`[${new Date().toISOString()}] [N-ECM] ${m}`, d ?? ''),
  err:  (m, d) => console.error(`[${new Date().toISOString()}] [N-ECM] ${m}`, d ?? ''),
};

function pickKernel(kernels, ext){
  const k = (kernels||[]).find(k => k?.name?.endsWith(ext));
  if(!k) throw new Error(`Kernel with extension ${ext} not found`);
  log.info(`Using kernel: ${k.name}`);
  return k.content;
}

function b64FromArrayBuffer(buf){
  const b = new Uint8Array(buf); let s = '';
  for (let i=0;i<b.length;i++) s += String.fromCharCode(b[i]);
  return btoa(s);
}
function arrayBufferFromB64(str){
  const bin = atob(str); const out = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

function wsOnce(url, payload, timeoutMs=180000){
  return new Promise((resolve, reject) => {
    let ws, to;
    try { ws = new WebSocket(url); } catch(e){ return reject(e); }
    to = setTimeout(() => { try{ws.close();}catch{} reject(new Error(`WS timeout ${timeoutMs}ms`)); }, timeoutMs);
    ws.onopen = () => { try { ws.send(JSON.stringify(payload)); } catch(e){ clearTimeout(to); reject(e);} };
    ws.onerror = (e) => { clearTimeout(to); reject(e instanceof Error ? e : new Error('WS error')); };
    ws.onmessage = ev => {
      clearTimeout(to);
      try { resolve(JSON.parse(ev.data)); }
      catch(e){ reject(new Error(`Bad JSON from native server: ${e.message}`)); }
      finally { try{ws.close();}catch{} }
    };
  });
}

export function createExecutor({ kernels, config }){
  const isHttps = (typeof window !== 'undefined') && window.location?.protocol === 'https:';
  const port = Number(config?.nativePort ?? 8787);
  const endpoint = config?.nativeEndpoint || (isHttps ? `wss://127.0.0.1:${port}/native` : `ws://127.0.0.1:${port}/native`);

  const source = pickKernel(kernels, '.cl');
  const entry  = config?.entry || 'execute_task';

  // Layout constants (must match kernel)
  const HEADER_WORDS_V3 = 12;
  const CONST_WORDS     = 8*3 + 4;   // 28
  const CURVE_OUT_WORDS = 8 + 1 + 3; // 12
  // NOTE: kernel ignores pp_start/pp_len; it processes all pp_count in one pass.

  async function runChunk({ payload, meta }){
    if (!payload?.data) throw new Error('payload.data (ArrayBuffer) required');
    const n        = meta?.n;
    const pp_count = meta?.pp_count ?? 0;
    const totalHint = meta?.total_words;

    if (!(n > 0)) throw new Error(`Invalid meta.n: ${n}`);

    // Build/ensure IO header v3 and capacity for constants + pp list
    let u32 = new Uint32Array(payload.data);
    if ((u32[1] >>> 0) < 3){
      const grown = new Uint32Array(Math.max(u32.length, totalHint || u32.length));
      grown.set(u32); u32 = grown;
      u32[1] = 3; // version
      u32[10] = 0; u32[11] = 0; // pp_start/pp_len (ignored by kernel)
    }
    const needHeaderWords = HEADER_WORDS_V3 + CONST_WORDS + pp_count;
    if (u32.length < needHeaderWords){
      const grown = new Uint32Array(needHeaderWords);
      grown.set(u32); u32 = grown;
    }

    // Work sizes: one work-item per curve; no explicit local → let driver choose
    const global = [n, 1, 1];
    const local  = [0, 0, 0]; // special: executor will pass nullptr

    // IMPORTANT: outputs-only buffer size (kernel writes from index 0)
    const outputsBytes = n * CURVE_OUT_WORDS * 4;

    const req = {
      action: 'compile_and_run',
      framework: 'opencl',
      source, entry,
      global, local,
      uniforms: [],
      // Full IO as input; compact results as output
      inputs:  [{ data: b64FromArrayBuffer(u32.buffer) }],
      // Prefer the new schema if server supports it, BUT also include legacy for back-compat
      outputs: [{ size: outputsBytes }],
      outputSizes: [ outputsBytes ],
      // NO inPlace / NO aliasing — kernel expects a separate outputs buffer
    };

    const resp = await wsOnce(endpoint, req, 120000);
    if (!resp?.ok) throw new Error(`native window failed: ${resp?.error || 'unknown'}`);

    const outB64 = resp.outputs?.[0];
    if (!outB64) throw new Error('native server returned empty outputs');

    const outBuf = arrayBufferFromB64(outB64);
    // This is exactly n * 12 * 4 bytes; server will assemble 9 words/curve from it.
    return { status:'ok', result: outBuf };
  }

  log.info('Native ECM Stage-1 executor ready (OpenCL, outputs-only).');
  return { runChunk };
}
