#version 300 es
// optional vertex shader for WebGL2 block matmul
precision highp float;
out vec2 vUV;
void main() {
  vec2 pos = vec2((gl_VertexID == 2) ? 3.0 : -1.0,
                  (gl_VertexID == 1) ? 3.0 : -1.0);
  vUV = (pos + 1.0) * 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}