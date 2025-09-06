// block_matrix_multiply_cuda_kernel_int8.cu
// Uses CUDA __dp4a intrinsic (SM 6.1+) to accelerate int8 matmul.
// Layout: A_packed [rows x groupsK] and B_packed [groupsK x cols], each entry packs 4 int8s in a 32-bit word.
// Accumulator: int (i32). Output C is int32.
// Compile with -arch=sm_61 or higher to enable __dp4a.

#include <cuda_runtime.h>
#include <stdint.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  #include <sm_61_intrinsics.h>
#endif

extern "C" __global__
void matmul_int8_dp4a(const int32_t* __restrict__ A_packed,  // u32 words as signed int for __dp4a
                      const int32_t* __restrict__ B_packed,
                      int32_t* __restrict__ C,               // output accumulators
                      int rows, int K, int cols, int groupsK)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= rows || c >= cols) return;

  int acc = 0;
  int rowBase = r * groupsK;

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    // Fast path: use __dp4a across the K dimension in 4-wide chunks
    for (int g = 0; g < groupsK; ++g) {
      int a4 = A_packed[rowBase + g];
      int b4 = B_packed[g * cols + c];
      acc = __dp4a(a4, b4, acc); // signed 8-bit dot accumulate -> int
    }
  #else
    // Fallback: unpack and multiply (portable but slower)
    for (int g = 0; g < groupsK; ++g) {
      int a4 = A_packed[rowBase + g];
      int b4 = B_packed[g * cols + c];
      // unpack 4 signed bytes from each 32-bit word
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        int8_t ai = (int8_t)((a4 >> (8*i)) & 0xFF);
        int8_t bi = (int8_t)((b4 >> (8*i)) & 0xFF);
        acc += (int)ai * (int)bi;
      }
    }
  #endif

  C[r * cols + c] = acc;
}
