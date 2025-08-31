// Talks to local native client at ws://127.0.0.1:8787/native
// Picks a framework ("cuda" | "opencl" | "vulkan") and builds the correct JSON.

export function createExecutor({ kernels, config }) {
  const endpoint = config?.nativeEndpoint || 'ws://127.0.0.1:8787/native';
  const framework = config?.nativeFramework || 'cuda'; // or "opencl" | "vulkan"

  function pickKernel(ext) {
    return kernels.find(k => k.name.endsWith(ext))?.content;
  }
  function b64(buf) {
    const b = new Uint8Array(buf);
    let s = ''; for (let i=0;i<b.length;i++) s += String.fromCharCode(b[i]);
    return btoa(s);
  }
  function wsOnce(url, payload) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.onopen = () => ws.send(JSON.stringify(payload));
      ws.onerror = e => reject(new Error('WS error'));
      ws.onmessage = ev => {
        try {
          const j = JSON.parse(ev.data);
          ws.close();
          resolve(j);
        } catch (e) {
          reject(e);
        }
      };
    });
  }

  async function runChunk({ payload, meta }) {
    const tClientRecv = Date.now();
    let req;

    if (framework === 'cuda') {
      const src = pickKernel('.cu');
      if (!src) throw new Error('No CUDA .cu kernel sent');
      // You can pass grid/block via config or meta
      const grid = meta?.grid || config?.grid || [1,1,1];
      const block = meta?.block || config?.block || [16,16,1];
      const uniforms = meta?.uniforms || config?.uniforms || [];
      const inputs = [];
      if (payload.a) inputs.push({ data: b64(payload.a), size: payload.a.byteLength });
      if (payload.b) inputs.push({ data: b64(payload.b), size: payload.b.byteLength });
      if (payload.data) inputs.push({ data: b64(payload.data), size: payload.data.byteLength });

      const outputSizes = meta?.outputSizes || config?.outputSizes || (() => {
        // Example for block-matmul: rows*cols*4 bytes
        if (payload.dims?.rows && payload.dims?.cols) return [payload.dims.rows * payload.dims.cols * 4];
        return [0];
      })();

      req = {
        action: 'compile_and_run',
        framework: 'cuda',
        source: src,
        entry: config?.entry || 'main',
        grid, block, uniforms, inputs, outputSizes
      };
    }

    if (framework === 'opencl') {
      const src = pickKernel('.cl');
      if (!src) throw new Error('No OpenCL .cl kernel sent');
      const global = meta?.global || config?.global || [1,1,1];
      const local  = meta?.local  || config?.local  || [1,1,1];
      const uniforms = meta?.uniforms || config?.uniforms || [];
      const inputs = [];
      if (payload.a) inputs.push({ data: b64(payload.a) });
      if (payload.b) inputs.push({ data: b64(payload.b) });
      if (payload.data) inputs.push({ data: b64(payload.data) });
      const outputSizes = meta?.outputSizes || config?.outputSizes || [0];

      req = {
        action: 'compile_and_run',
        framework: 'opencl',
        source: src,
        entry: config?.entry || 'main',
        global, local, uniforms, inputs, outputSizes
      };
    }

    if (framework === 'vulkan') {
      // Prefer GLSL if provided; else expect SPIR-V (base64) under kernels already
      const glsl = pickKernel('.glsl'); // compute shader source
      const spirvB64 = kernels.find(k => k.name.endsWith('.spv.b64'))?.content; // optional
      const uniforms = meta?.uniforms || config?.uniforms || [];
      const inputs = [];
      if (payload.a) inputs.push({ data: b64(payload.a) });
      if (payload.b) inputs.push({ data: b64(payload.b) });
      if (payload.data) inputs.push({ data: b64(payload.data) });
      const outputSizes = meta?.outputSizes || config?.outputSizes || [0];
      const groups = meta?.groups || config?.groups || [1,1,1];

      req = {
        action: 'compile_and_run',
        framework: 'vulkan',
        entry: config?.entry || 'main',
        uniforms, inputs, outputSizes, groups
      };
      if (glsl) req.source_glsl = glsl;
      else if (spirvB64) req.spirv = spirvB64;
      else throw new Error('No Vulkan GLSL or SPIR-V provided');
    }

    const resp = await wsOnce(endpoint, req);
    if (!resp.ok) throw new Error(resp.error || 'native execution failed');

    // Use the first output buffer by default
    const first = resp.outputs?.[0];
    const result = first ? Uint8Array.from(atob(first), c => c.charCodeAt(0)).buffer : new ArrayBuffer(0);
    const tClientDone = Date.now();
    return { status:'ok', result, timings:{ tClientRecv, tClientDone } };
  }

  return { runChunk };
}
