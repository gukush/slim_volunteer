// Enhanced native-bridge.client.js with comprehensive logging
// Talks to local native client at wss://127.0.0.1:8787/native

// Simple logging utility
const logger = {
  info: (msg, data) => console.log(`[${new Date().toISOString()}] [BRIDGE-INFO] ${msg}`, data || ''),
  warn: (msg, data) => console.warn(`[${new Date().toISOString()}] [BRIDGE-WARN] ${msg}`, data || ''),
  error: (msg, data) => console.error(`[${new Date().toISOString()}] [BRIDGE-ERROR] ${msg}`, data || ''),
  debug: (msg, data) => console.log(`[${new Date().toISOString()}] [BRIDGE-DEBUG] ${msg}`, data || ''),
};

export function createExecutor({ kernels, config }) {
  logger.debug('Configuration:', config);
  logger.debug('Kernels available:', kernels?.map(k => k.name) || []);

  // For HTTPS pages (WebGPU requirement), always use wss://
  const isSecure = window.location.protocol === 'https:';
  const defaultPort = config?.nativePort || 8787;
  const defaultEndpoint = isSecure
    ? `wss://127.0.0.1:${defaultPort}/native`  // SSL WebSocket for HTTPS pages
    : `ws://127.0.0.1:${defaultPort}/native`;   // Plain WebSocket for HTTP pages

  const endpoint = config?.nativeEndpoint || defaultEndpoint;
  const framework = config?.framework || 'native-cuda';

  function pickKernel(ext) {
    const kernel = kernels.find(k => k.name.endsWith(ext));
    if (kernel) {
      logger.debug(`✓ Found kernel: ${kernel.name} (${ext})`);
    } else {
      logger.warn(`❌ No kernel found for extension: ${ext}`);
      logger.debug('Available kernels:', kernels?.map(k => k.name) || []);
    }
    return kernel?.content;
  }

  function b64(buf) {
    logger.debug(`Encoding ${buf.byteLength} bytes to base64`);
    const b = new Uint8Array(buf);
    let s = '';
    for (let i=0;i<b.length;i++) s += String.fromCharCode(b[i]);
    const encoded = btoa(s);
    logger.debug(`Base64 encoded: ${encoded.length} characters`);
    return encoded;
  }

  function wsOnce(url, payload) {
    return new Promise((resolve, reject) => {
      let ws;

      logger.info(`🔌 Connecting to native server: ${url}`);

      // Add connection timeout
      const timeout = setTimeout(() => {
        logger.error('⏰ Connection timeout after 10 seconds');
        if (ws) {
          ws.close();
        }
        reject(new Error('Connection timeout - is the native server running?'));
      }, 10000);

      try {
        ws = new WebSocket(url);

        ws.onopen = () => {
          clearTimeout(timeout);
          logger.info('✅ WebSocket connection established');
          logger.debug('Sending payload:', {
            action: payload.action,
            framework: payload.framework,
            sourceLength: payload.source?.length || 0,
            inputCount: payload.inputs?.length || 0,
            outputSizes: payload.outputSizes
          });

          const payloadStr = JSON.stringify(payload);
          logger.info(`📤 Sending ${payloadStr.length} bytes to server`);
          ws.send(payloadStr);
        };

        ws.onerror = (e) => {
          clearTimeout(timeout);
          logger.error('❌ WebSocket connection error:', e);

          const isSecureConnection = url.startsWith('wss://');
          let errorMsg = `WebSocket connection failed to ${url}.\n\n🔍 Troubleshooting:\n`;

          if (isSecureConnection) {
            errorMsg += `1. ✓ Make sure native server is running with SSL: ./native_client --mode local --ssl\n`;
            errorMsg += `2. ✓ Certificate files (server.crt, server.key) exist in server directory\n`;
            errorMsg += `3. ✓ Start Chrome with: chrome --ignore-certificate-errors --ignore-ssl-errors\n`;
            errorMsg += `4. ✓ Or install server.crt as trusted certificate\n`;
            errorMsg += `5. ✓ Port ${defaultPort} is not blocked by firewall\n`;
            errorMsg += `6. ✓ Check server logs for SSL handshake errors`;
          } else {
            errorMsg += `1. ✓ Native server is running: ./native_client --mode local --no-ssl\n`;
            errorMsg += `2. ✓ Using HTTP (not HTTPS) for your web page\n`;
            errorMsg += `3. ✓ Port ${defaultPort} is not blocked by firewall\n`;
            errorMsg += `4. ✓ Check server logs for connection errors`;
          }

          reject(new Error(errorMsg));
        };

        ws.onmessage = (ev) => {
          clearTimeout(timeout);
          logger.info(`📥 Received ${ev.data.length} bytes from server`);

          try {
            const j = JSON.parse(ev.data);
            logger.debug('Response received:', {
              ok: j.ok,
              error: j.error,
              outputCount: j.outputs?.length || 0,
              processingTime: j.processingTimeMs
            });

            ws.close();

            if (j.ok) {
              logger.info('✅ Server request completed successfully');
              if (j.processingTimeMs) {
                logger.info(`⚡ Server processing time: ${j.processingTimeMs}ms`);
              }
            } else {
              logger.error('❌ Server returned error:', j.error);
            }

            resolve(j);
          } catch (e) {
            logger.error('❌ Invalid JSON response:', e.message);
            logger.debug('Raw response:', ev.data.substring(0, 500));
            reject(new Error('Invalid JSON response: ' + e.message));
          }
        };

        ws.onclose = (ev) => {
          logger.debug(`WebSocket closed: code=${ev.code}, reason=${ev.reason}`);
        };

      } catch (e) {
        clearTimeout(timeout);
        logger.error('❌ Failed to create WebSocket:', e);
        reject(e);
      }
    });
  }

  async function runChunk({ payload, meta }) {
    const tClientRecv = Date.now();
    logger.info('🎯 Processing chunk request');
    logger.debug('Payload keys:', Object.keys(payload));
    logger.debug('Meta data:', meta);

    let req;

    if (framework === 'native-cuda') {
      logger.info('🚀 Preparing CUDA request');

      const src = pickKernel('.cu');
      if (!src) {
        const error = 'No CUDA .cu kernel found in kernels array';
        logger.error('❌ ' + error);
        throw new Error(error);
      }
      logger.info(`✓ CUDA kernel loaded: ${src.length} characters`);

      const grid = meta?.grid || config?.grid || [1,1,1];
      const block = meta?.block || config?.block || [16,16,1];
      const uniforms = meta?.uniforms || config?.uniforms || [];

      logger.debug('CUDA parameters:', { grid, block, uniforms });

      const inputs = [];
      if (payload.a) {
        inputs.push({ data: b64(payload.a), size: payload.a.byteLength });
        logger.debug(`Input A: ${payload.a.byteLength} bytes`);
      }
      if (payload.b) {
        inputs.push({ data: b64(payload.b), size: payload.b.byteLength });
        logger.debug(`Input B: ${payload.b.byteLength} bytes`);
      }
      if (payload.data) {
        inputs.push({ data: b64(payload.data), size: payload.data.byteLength });
        logger.debug(`Input data: ${payload.data.byteLength} bytes`);
      }

      logger.info(`📦 Total inputs: ${inputs.length}`);

      const outputSizes = meta?.outputSizes || config?.outputSizes || (() => {
        if (payload.dims?.rows && payload.dims?.cols) {
          const size = payload.dims.rows * payload.dims.cols * 4;
          logger.debug(`Auto-calculated output size: ${size} bytes (${payload.dims.rows}x${payload.dims.cols} floats)`);
          return [size];
        }
        logger.warn('No output sizes specified, using default [1024]');
        return [1024];
      })();

      logger.debug('Output sizes:', outputSizes);

      req = {
        action: 'compile_and_run',
        framework: 'cuda',
        source: src,
        entry: config?.entry || 'main',
        grid, block, uniforms, inputs, outputSizes
      };
    }

    if (framework === 'native-opencl') {
      logger.info('🔧 Preparing OpenCL request');

      const src = pickKernel('.cl');
      if (!src) {
        const error = 'No OpenCL .cl kernel found in kernels array';
        logger.error('❌ ' + error);
        throw new Error(error);
      }
      logger.info(`✓ OpenCL kernel loaded: ${src.length} characters`);

      const global = meta?.global || config?.global || [1,1,1];
      const local  = meta?.local  || config?.local  || [1,1,1];
      const uniforms = meta?.uniforms || config?.uniforms || [];

      logger.debug('OpenCL parameters:', { global, local, uniforms });

      const inputs = [];
      if (payload.a) {
        inputs.push({ data: b64(payload.a) });
        logger.debug(`Input A: ${payload.a.byteLength} bytes`);
      }
      if (payload.b) {
        inputs.push({ data: b64(payload.b) });
        logger.debug(`Input B: ${payload.b.byteLength} bytes`);
      }
      if (payload.data) {
        inputs.push({ data: b64(payload.data) });
        logger.debug(`Input data: ${payload.data.byteLength} bytes`);
      }

      logger.info(`📦 Total inputs: ${inputs.length}`);

      const outputSizes = meta?.outputSizes || config?.outputSizes || [1024];
      logger.debug('Output sizes:', outputSizes);

      req = {
        action: 'compile_and_run',
        framework: 'opencl',
        source: src,
        entry: config?.entry || 'main',
        global, local, uniforms, inputs, outputSizes
      };
    }

    if (framework === 'native-vulkan') {
      logger.info('🌋 Preparing Vulkan request');

      const glsl = pickKernel('.glsl');
      const spirvB64 = kernels.find(k => k.name.endsWith('.spv.b64'))?.content;

      if (!glsl && !spirvB64) {
        const error = 'No Vulkan GLSL (.glsl) or SPIR-V (.spv.b64) kernel found';
        logger.error('❌ ' + error);
        throw new Error(error);
      }

      if (glsl) {
        logger.info(`✓ Vulkan GLSL shader loaded: ${glsl.length} characters`);
      } else {
        logger.info(`✓ Vulkan SPIR-V binary loaded: ${spirvB64.length} characters`);
      }

      const uniforms = meta?.uniforms || config?.uniforms || [];
      const inputs = [];
      if (payload.a) {
        inputs.push({ data: b64(payload.a) });
        logger.debug(`Input A: ${payload.a.byteLength} bytes`);
      }
      if (payload.b) {
        inputs.push({ data: b64(payload.b) });
        logger.debug(`Input B: ${payload.b.byteLength} bytes`);
      }
      if (payload.data) {
        inputs.push({ data: b64(payload.data) });
        logger.debug(`Input data: ${payload.data.byteLength} bytes`);
      }

      logger.info(`📦 Total inputs: ${inputs.length}`);

      const outputSizes = meta?.outputSizes || config?.outputSizes || [1024];
      const groups = meta?.groups || config?.groups || [1,1,1];

      logger.debug('Vulkan parameters:', { groups, uniforms, outputSizes });

      req = {
        action: 'compile_and_run',
        framework: 'vulkan',
        entry: config?.entry || 'main',
        uniforms, inputs, outputSizes, groups
      };

      if (glsl) {
        req.source_glsl = glsl;
      } else if (spirvB64) {
        req.spirv = spirvB64;
      }
    }

    if (!req) {
      const error = `Unsupported framework: ${framework}`;
      logger.error('❌ ' + error);
      throw new Error(error);
    }

    logger.info(`📡 Sending ${framework.toUpperCase()} request to native server`);
    logger.debug('Request summary:', {
      action: req.action,
      framework: req.framework,
      entry: req.entry,
      sourceLength: req.source?.length || req.source_glsl?.length || 0,
      inputCount: req.inputs?.length || 0,
      outputCount: req.outputSizes?.length || 0
    });

    try {
      const resp = await wsOnce(endpoint, req);

      if (!resp.ok) {
        const error = resp.error || 'Native execution failed';
        logger.error('❌ Server execution failed:', error);
        throw new Error(error);
      }

      // Decode base64 response
      const first = resp.outputs?.[0];
      const result = first ?
        Uint8Array.from(atob(first), c => c.charCodeAt(0)).buffer :
        new ArrayBuffer(0);

      const tClientDone = Date.now();
      const totalTime = tClientDone - tClientRecv;

      logger.info('✅ Chunk processing completed successfully');
      logger.info(`⏱️ Total client time: ${totalTime}ms`);
      if (resp.processingTimeMs) {
        logger.info(`⚡ Server processing time: ${resp.processingTimeMs}ms`);
        logger.info(`🌐 Network overhead: ${totalTime - resp.processingTimeMs}ms`);
      }
      logger.info(`📊 Result size: ${result.byteLength} bytes`);

      return {
        status: 'ok',
        result,
        timings: { tClientRecv, tClientDone },
        processingTimeMs: resp.processingTimeMs
      };

    } catch (error) {
      logger.error('❌ Chunk processing failed:', error.message);
      throw error;
    }
  }

  logger.info('✅ Native bridge executor created successfully');
  return { runChunk };
}