#version 300 es
// optional fragment shader for WebGL2 block matmul
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
}

