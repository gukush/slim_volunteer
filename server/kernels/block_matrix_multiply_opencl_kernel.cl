__kernel void execute_task(__global const float* A, __global const float* B, __global float* C, const int rows, const int K, const int cols) {
  int c = get_global_id(0);
  int r = get_global_id(1);
  if (r >= rows || c >= cols) return;
  float acc = 0.0f;
  for (int k=0;k<K;++k) acc += A[r*K + k] * B[k*cols + c];
  C[r*cols + c] = acc;
}