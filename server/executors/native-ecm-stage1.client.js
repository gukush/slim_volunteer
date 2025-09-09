// executors/native-ecm-stage1.client.js
// Native client executor for ECM Stage 1 - combines ECM data handling with native client communication

// Simple logging utility
const logger = {
  info: (msg, data) => console.log(`[${new Date().toISOString()}] [NATIVE-ECM-INFO] ${msg}`, data || ''),
  warn: (msg, data) => console.warn(`[${new Date().toISOString()}] [NATIVE-ECM-WARN] ${msg}`, data || ''),
  error: (msg, data) => console.error(`[${new Date().toISOString()}] [NATIVE-ECM-ERROR] ${msg}`, data || ''),
  debug: (msg, data) => console.log(`[${new Date().toISOString()}] [NATIVE-ECM-DEBUG] ${msg}`, data || ''),
};

// Lightweight debug helpers
const DBG = (...args) => logger.debug(...args);
const ERR = (...args) => logger.error(...args);
const u32sum = (u32) => {
  let x = 0 >>> 0;
  for (let i = 0; i < u32.length; i++) x ^= (u32[i] >>> 0);
  return ('00000000' + x.toString(16)).slice(-8);
};

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
          errorMsg += `5. ✓ Port ${url.match(/:(\d+)/)?.[1] || '3001'} is not blocked by firewall\n`;
          errorMsg += `6. ✓ Check server logs for SSL handshake errors`;
        } else {
          errorMsg += `1. ✓ Native server is running: ./native_client --mode local --no-ssl\n`;
          errorMsg += `2. ✓ Using HTTP (not HTTPS) for your web page\n`;
          errorMsg += `3. ✓ Port ${url.match(/:(\d+)/)?.[1] || '3001'} is not blocked by firewall\n`;
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

export function createExecutor({ kernels, config }){
  logger.debug('Configuration:', config);
  logger.debug('Kernels available:', kernels?.map(k => k.name) || []);

  // For HTTPS pages, always use wss://
  const isSecure = window.location.protocol === 'https:';
  const defaultPort = config?.nativePort || 3001;
  const defaultEndpoint = isSecure
    ? `wss://127.0.0.1:${defaultPort}/ws-native`  // SSL WebSocket for HTTPS pages
    : `ws://127.0.0.1:${defaultPort}/ws-native`;   // Plain WebSocket for HTTP pages

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

  async function prewarm() {
    // For native client, prewarm is a no-op since we don't need to compile shaders
    logger.info('Native ECM executor prewarm - no action needed');
  }

  async function runChunk({ payload, meta }){
    const tClientRecv = Date.now();
    logger.info('🎯 Processing ECM chunk request');
    logger.debug('Payload keys:', Object.keys(payload));
    logger.debug('Meta data:', meta);

    const { data, dims } = payload;
    const { n, pp_count, total_words } = dims;

    logger.info(`Processing ECM chunk: ${n} curves, ${pp_count} prime powers`);

    // Parse the incoming buffer structure (Version 2)
    const u32 = new Uint32Array(data);
    const version = u32[1] >>> 0;

    DBG(`Buffer version: ${version}, expected: 2`);
    if (version !== 2) {
      ERR(`Unexpected buffer version: ${version}, expected 2`);
    }

    const HEADER_WORDS_V2 = 12;
    const CONST_WORDS = 8*3 + 4; // N(8) + R2(8) + mont_one(8) + n0inv32(1) + pad(3)
    const CURVE_OUT_WORDS_PER = 8 + 1 + 3; // result(8) + status(1) + pad(3)

    // Version 2 layout: no curve input data
    const calcTotalWords = HEADER_WORDS_V2 + CONST_WORDS + pp_count + n * CURVE_OUT_WORDS_PER;
    if (typeof total_words === 'number' && total_words !== calcTotalWords) {
      ERR('Layout mismatch v2: total_words vs calculated', { total_words, calcTotalWords, n, pp_count });
    }

    const outputOffset = HEADER_WORDS_V2 + CONST_WORDS + pp_count;
    DBG('v2 layout', { HEADER_WORDS_V2, CONST_WORDS, CURVE_OUT_WORDS_PER, outputOffset, total_words });

    const checksumIn = u32sum(u32);

    // Prepare native client request based on framework
    let req;

    if (framework === 'native-cuda') {
      logger.info('🚀 Preparing CUDA ECM request');

      const src = pickKernel('.cu');
      if (!src) {
        const error = 'No CUDA .cu kernel found in kernels array';
        logger.error('❌ ' + error);
        throw new Error(error);
      }
      logger.info(`✓ CUDA kernel loaded: ${src.length} characters`);

      // For ECM, we use the entire data buffer as input
      const inputs = [{
        data: b64(data),
        size: data.byteLength
      }];

      logger.info(`📦 ECM input: ${data.byteLength} bytes`);

      // Output size is the same as input size (in-place processing)
      const outputSizes = [data.byteLength];

      req = {
        action: 'compile_and_run',
        framework: 'cuda',
        source: src,
        entry: config?.entry || 'execute_task',
        grid: meta?.grid || config?.grid || [Math.ceil(n / 64), 1, 1], // 1 curve per thread, 64 threads per block
        block: meta?.block || config?.block || [64, 1, 1],
        uniforms: meta?.uniforms || config?.uniforms || [n, pp_count, outputOffset],
        inputs,
        outputSizes
      };
    }
    else if (framework === 'native-opencl') {
      logger.info('🔧 Preparing OpenCL ECM request');

      const src = pickKernel('.cl');
      if (!src) {
        const error = 'No OpenCL .cl kernel found in kernels array';
        logger.error('❌ ' + error);
        throw new Error(error);
      }
      logger.info(`✓ OpenCL kernel loaded: ${src.length} characters`);

      // For ECM, we use the entire data buffer as input
      const inputs = [{
        data: b64(data),
        size: data.byteLength
      }];

      logger.info(`📦 ECM input: ${data.byteLength} bytes`);

      // Output size is the same as input size (in-place processing)
      const outputSizes = [data.byteLength];

      req = {
        action: 'compile_and_run',
        framework: 'opencl',
        source: src,
        entry: config?.entry || 'execute_task',
        global: meta?.global || config?.global || [n, 1, 1], // 1 curve per work item
        local: meta?.local || config?.local || [64, 1, 1],   // 64 work items per group
        uniforms: meta?.uniforms || config?.uniforms || [n, pp_count, outputOffset],
        inputs,
        outputSizes
      };
    }
    else if (framework === 'native-vulkan') {
      logger.info('🌋 Preparing Vulkan ECM request');

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

      // For ECM, we use the entire data buffer as input
      const inputs = [{
        data: b64(data),
        size: data.byteLength
      }];

      logger.info(`📦 ECM input: ${data.byteLength} bytes`);

      // Output size is the same as input size (in-place processing)
      const outputSizes = [data.byteLength];

      const uniforms = meta?.uniforms || config?.uniforms || [n, pp_count, outputOffset];
      const localSize = {
        x: Number(config?.localSizeX ?? 64),
        y: Number(config?.localSizeY ?? 1),
        z: Number(config?.localSizeZ ?? 1),
      };
      const groups = meta?.groups || config?.groups || [Math.ceil(n / 64), 1, 1];

      req = {
        action: 'compile_and_run',
        framework: 'vulkan',
        entry: config?.entry || 'main',
        uniforms,
        inputs,
        outputSizes,
        groups
      };

      if (glsl) {
        req.source_glsl = glsl;
      } else if (spirvB64) {
        req.spirv = spirvB64;
      }
    }
    else {
      const error = `Unsupported framework: ${framework}`;
      logger.error('❌ ' + error);
      throw new Error(error);
    }

    logger.info(`📡 Sending ${framework.toUpperCase()} ECM request to native server`);
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
        const error = resp.error || 'Native ECM execution failed';
        logger.error('❌ Server execution failed:', error);
        throw new Error(error);
      }

      // Decode base64 response
      const first = resp.outputs?.[0];
      const result = first ?
        Uint8Array.from(atob(first), c => c.charCodeAt(0)).buffer :
        new ArrayBuffer(0);

      // Verify the result has the expected structure
      const resultU32 = new Uint32Array(result);
      const resultChecksum = u32sum(resultU32);

      logger.info(`Output section starts at word ${outputOffset}`);
      for(let i = 0; i < Math.min(5, n); i++) {
        const curveOffset = outputOffset + i * CURVE_OUT_WORDS_PER;
        const status = resultU32[curveOffset + 8];
        logger.info(`Curve ${i}: status=${status}, result limbs:`,
          Array.from(resultU32.slice(curveOffset, curveOffset + 8)));
      }

      DBG('runChunk(): finished', { checksumIn, resultChecksum, total_words, outputOffset });

      const tClientDone = Date.now();
      const totalTime = tClientDone - tClientRecv;

      logger.info('✅ ECM chunk processing completed successfully');
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
      logger.error('❌ ECM chunk processing failed:', error.message);
      throw error;
    }
  }

  logger.info('✅ Native ECM executor created successfully');
  return { runChunk, prewarm };
}


