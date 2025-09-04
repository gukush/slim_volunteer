extern "C" __global__ void execute_task(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int rows, int K, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= rows || c >= cols) return;
  float acc = 0.f;
  for (int k=0;k<K;++k) acc += A[r*K + k] * B[k*cols + c];
  C[r*cols + c] = acc;
}