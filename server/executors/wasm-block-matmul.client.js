// executors/wasm-block-matmul.client.js
// WASM (C++/Emscripten) executor for block-matmul-flex with detailed debug logging.
export function createExecutor({ kernels, config }) {
  const kernel = kernels.find(
    (k) => k.name.endsWith('kernels/cpp/block_matmul.js') || k.name.endsWith('block_matmul.js')
  );
  if (!kernel) throw new Error('WASM kernel JS (block_matmul.js) missing');

  // --- lightweight logger gated by config.debug/logLevel ---
  const lvl = String(config?.logLevel || '').toLowerCase();
  const debugEnabled = !!config?.debug || lvl === 'debug' || lvl === 'trace';
  const prefix = '[cpp-wasm]';
  const d = (...args) => { if (debugEnabled) console.debug(prefix, ...args); };
  const i = (...args) => console.info(prefix, ...args);
  const w = (...args) => console.warn(prefix, ...args);
  const e = (...args) => console.error(prefix, ...args);

  let ModuleFactory = null;
  let Module = null;
  let matmul = null;

  function has(v) { return typeof v !== 'undefined' && v !== null; }

  function getMemBuffer() {
    // Try common Emscripten memory buffers in a safe order
    if (has(Module?.HEAPU8?.buffer)) return Module.HEAPU8.buffer;
    if (has(Module?.HEAP8?.buffer))  return Module.HEAP8.buffer;
    if (has(Module?.wasmMemory?.buffer)) return Module.wasmMemory.buffer;
    if (has(Module?.asm?.memory?.buffer)) return Module.asm.memory.buffer;
    throw new Error('WASM memory buffer not found on Module');
  }

  async function ensureModule() {
  if (Module) return Module;
  const jsBytes = new TextEncoder().encode(kernel.content).byteLength;
  d('ensureModule(): starting, kernel bytes =', jsBytes);
  const blob = new Blob([kernel.content], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  try {
    const mod = await import(/* webpackIgnore: true */ url);
    const exportedKeys = Object.keys(mod || {});
    d('ESM import ok. keys=', exportedKeys);

    ModuleFactory = mod.default || mod.Module || mod;
    if (typeof ModuleFactory !== 'function') {
      throw new Error('Module factory not found / not a function');
    }

    // Factory: returns a Module object (often already Promise-initialized)
    Module = await ModuleFactory();
    // Await the internal ready gate if present
    if (Module && Module.ready && typeof Module.ready.then === 'function') {
      d('awaiting Module.ready...');
      await Module.ready;
      d('Module.ready resolved');
    }

    d('module exports:',
      'has _matmul=', typeof Module._matmul === 'function',
      'has cwrap=', typeof Module.cwrap === 'function',
      'has HEAPU8=', !!Module.HEAPU8,
      'has HEAP8=', !!Module.HEAP8,
      'has wasmMemory=', !!Module.wasmMemory,
    );

    if (typeof Module._matmul === 'function') {
      matmul = Module._matmul;
      d('using direct _matmul');
    } else if (typeof Module.cwrap === 'function') {
      matmul = Module.cwrap('matmul', null, ['number','number','number','number','number','number']);
      d('using cwrap("matmul")');
    } else {
      throw new Error('matmul export not found (neither _matmul nor cwrap)');
    }

    // Touch memory after ready to verify access
    const buf = getMemBuffer();
    d('memory buffer byteLength=', buf?.byteLength);

    i('WASM module ready');
    return Module;
  } finally {
    URL.revokeObjectURL(url);
  }
}

  async function prewarm() {
    const t0 = performance.now();
    await ensureModule();
    const t1 = performance.now();
    d('prewarm done in', (t1 - t0).toFixed(2), 'ms');
  }

  function sampleFloatArray(ab, max = 4) {
    try {
      const f = new Float32Array(ab);
      const n = Math.min(f.length, max);
      const out = [];
      for (let i = 0; i < n; i++) out.push(Number.isFinite(f[i]) ? +f[i].toFixed(4) : f[i]);
      return out;
    } catch (_) { return []; }
  }

  async function runChunk({ payload, meta }) {
    const tClientRecv = Date.now();
    const t0 = performance.now();
    try {
      const { a, b, dims } = payload;
      const { rows, K, cols } = dims;

      // Basic shape & size logging
      d('runChunk dims=', { rows, K, cols },
        'aBytes=', a?.byteLength, 'bBytes=', b?.byteLength, 'meta=', meta);

      if (!a || !b) throw new Error('Missing payload buffers A/B');
      if (!Module) await ensureModule();

      const bytesA = a.byteLength;
      const bytesB = b.byteLength;
      const outBytes = rows * cols * 4;

      // Pointers
      const ptrA = Module._malloc(bytesA);
      const ptrB = Module._malloc(bytesB);
      const ptrC = Module._malloc(outBytes);
      d('malloc ptrs:', { ptrA, ptrB, ptrC, bytesA, bytesB, outBytes });
      if (!ptrA || !ptrB || !ptrC) {
        if (ptrA) Module._free(ptrA);
        if (ptrB) Module._free(ptrB);
        if (ptrC) Module._free(ptrC);
        throw new Error('WASM malloc failed');
      }

      try {
        // Resolve a byte view of memory
        const mem = new Uint8Array(getMemBuffer());

        // Copy inputs
        mem.set(new Uint8Array(a), ptrA);
        mem.set(new Uint8Array(b), ptrB);
        if (debugEnabled) {
          d('A sample:', sampleFloatArray(a));
          d('B sample:', sampleFloatArray(b));
        }

        // Compute
        const tMul0 = performance.now();
        matmul(rows, K, cols, ptrA, ptrB, ptrC);
        const tMul1 = performance.now();

        // Read back
        const out = new ArrayBuffer(outBytes);
        new Uint8Array(out).set(mem.subarray(ptrC, ptrC + outBytes));

        if (debugEnabled) {
          d('C sample:', sampleFloatArray(out));
        }

        const tClientDone = Date.now();
        const t1 = performance.now();
        d('runChunk timings(ms): ensure+copy+mul+read total=', (t1 - t0).toFixed(2),
          'mulOnly=', (tMul1 - tMul0).toFixed(2));

        return { status: 'ok', result: out, timings: { tClientRecv, tClientDone } };
      } finally {
        // Always free
        try { Module._free(ptrA); } catch {}
        try { Module._free(ptrB); } catch {}
        try { Module._free(ptrC); } catch {}
      }
    } catch (err) {
      const tClientDone = Date.now();
      const msg = (err && err.stack) ? err.stack : String(err && err.message || err);
      e('runChunk error:', msg);
      return { status: 'error', error: msg, result: new ArrayBuffer(0), timings: { tClientRecv, tClientDone } };
    }
  }

  return { prewarm, runChunk };
}
