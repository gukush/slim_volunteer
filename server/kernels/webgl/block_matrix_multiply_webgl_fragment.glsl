// optional fragment shader for WebGL2 block matmul
#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D Atex;  // size (K, rows)
uniform sampler2D Btex;  // size (cols, K)
uniform ivec3 dims;      // (rows, K, cols)
out vec4 fragColor;

void main() {

  int c = int(gl_FragCoord.x);
  int r = int(gl_FragCoord.y);
  if (r >= dims.x || c >= dims.z) { discard; }

  const int TILE = 16;
  int numTiles = (dims.y + TILE - 1) / TILE;

  float acc = 0.0;
  for (int t = 0; t < numTiles; ++t) {
    for (int k = 0; k < TILE; ++k) {
      int kk = t * TILE + k;
      if (kk < dims.y) {
        // texelFetch uses integer pixel coords, level 0
        float a = texelFetch(Atex, ivec2(kk, r), 0).r;
        float b = texelFetch(Btex, ivec2(c, kk), 0).r;
        acc += a * b;
      }
    }
  }
  fragColor = vec4(acc, 0.0, 0.0, 1.0);
}

