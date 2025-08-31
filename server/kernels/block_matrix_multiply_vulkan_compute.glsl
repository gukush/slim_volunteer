#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer ABuf { float A[]; };
layout(std430, binding = 1) buffer BBuf { float B[]; };
layout(std430, binding = 2) buffer CBuf { float C[]; };
layout(std140, binding = 3) uniform Dims { uint rows; uint K; uint cols; };
void main() {
  uint c = gl_GlobalInvocationID.x;
  uint r = gl_GlobalInvocationID.y;
  if (r >= rows || c >= cols) return;
  float acc = 0.0;
  for (uint k=0u;k<K;++k) acc += A[r*K + k] * B[k*cols + c];
  C[r*cols + c] = acc;
}