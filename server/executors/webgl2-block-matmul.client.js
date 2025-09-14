// server/executors/webgl2-block-matmul.client.js
export function createExecutor({ kernels, config }) {
  const vert = kernels.find(k => k.name.endsWith('_webgl_vertex.glsl'))?.content || `
    #version 300 es
    precision highp float;
    out vec2 vUV;
    void main() {
      vec2 pos = vec2((gl_VertexID == 2) ? 3.0 : -1.0,
                      (gl_VertexID == 1) ? 3.0 : -1.0);
      vUV = (pos + 1.0) * 0.5;
      gl_Position = vec4(pos, 0.0, 1.0);
    }`;
  const frag = kernels.find(k => k.name.endsWith('_webgl_fragment.glsl'))?.content || `
    #version 300 es
    precision highp float;
    precision highp sampler2D;
    in vec2 vUV;
    out vec4 outColor;

    uniform sampler2D uA;     // A: (rows x K) stored as width=K, height=rows, R channel
    uniform sampler2D uB;     // B: (K x cols) stored as width=cols, height=K, R channel
    uniform int uRows, uK, uCols;

    // Pixel coords (ci,ri) == (col,row)
    void main() {
      ivec2 pix = ivec2(gl_FragCoord.xy); // integer pixel
      int ci = pix.x;
      int ri = pix.y;
      if (ri >= uRows || ci >= uCols) { outColor = vec4(0.0); return; }

      float acc = 0.0;
      // Upper bound keeps GLSL happy; break on uK
      for (int k = 0; k < 4096; ++k) {
        if (k >= uK) break;
        float aVal = texelFetch(uA, ivec2(k, ri), 0).r;
        float bVal = texelFetch(uB, ivec2(ci, k), 0).r;
        acc += aVal * bVal;
      }
      outColor = vec4(acc, 0.0, 0.0, 1.0);
    }`;

  let gl, prog, loc = {}, fb;

  function getGL() {
    if (gl) return gl;
    const canvas = document.createElement('canvas');
    gl = canvas.getContext('webgl2');
    if (!gl) throw new Error('WebGL2 not available');
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float required');
    }
    fb = gl.createFramebuffer();
    return gl;
  }

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('Shader compile error: ' + gl.getShaderInfoLog(s));
    }
    return s;
  }

  function ensureProgram() {
    if (prog) return;
    getGL();
    const vs = compile(gl.VERTEX_SHADER, vert);
    const fs = compile(gl.FRAGMENT_SHADER, frag);
    prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('Program link error: ' + gl.getProgramInfoLog(prog));
    }
    // Try both naming conventions for compatibility
    loc.uA = gl.getUniformLocation(prog, 'uA') || gl.getUniformLocation(prog, 'Atex');
    loc.uB = gl.getUniformLocation(prog, 'uB') || gl.getUniformLocation(prog, 'Btex');
    loc.uRows = gl.getUniformLocation(prog, 'uRows') || gl.getUniformLocation(prog, 'dims');
    loc.uK = gl.getUniformLocation(prog, 'uK');
    loc.uCols = gl.getUniformLocation(prog, 'uCols');
  }

  function makeTex(width, height, f32) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, f32);
    return tex;
  }

  function createOutTex(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('Framebuffer incomplete');
    }
    return tex;
  }

  async function runChunk({ payload, meta }) {
    const tClientRecv = Date.now();
    ensureProgram();

    const { a, b, dims } = payload;
    const rows = dims.rows|0, K = dims.K|0, cols = dims.cols|0;

    // A: rows x K -> (width=K, height=rows)
    // B: K x cols -> (width=cols, height=K)
    const texA = makeTex(K, rows, new Float32Array(a));
    const texB = makeTex(cols, K, new Float32Array(b));
    const outTex = createOutTex(cols, rows);

    gl.viewport(0, 0, cols, rows);
    gl.useProgram(prog);

    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA);
    gl.uniform1i(loc.uA, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texB);
    gl.uniform1i(loc.uB, 1);

    gl.uniform1i(loc.uRows, rows);
    gl.uniform1i(loc.uK, K);
    gl.uniform1i(loc.uCols, cols);

    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Read back R channel (one float per pixel)
    const out = new Float32Array(rows * cols * 4); // RGBA32F readback
    gl.readPixels(0, 0, cols, rows, gl.RGBA, gl.FLOAT, out);

    // Compact RGBA -> R
    const compact = new Float32Array(rows * cols);
    for (let i = 0, j = 0; i < compact.length; ++i, j += 4) compact[i] = out[j];

    const tClientDone = Date.now();
    return {
      status: 'ok',
      result: compact.buffer,
      timings: {
        tClientRecv,
        tClientDone,
        cpuTimeMs: tClientDone - tClientRecv, // No GPU timing in WebGL2
        gpuTimeMs: null
      }
    };
  }

  return { runChunk };
}
