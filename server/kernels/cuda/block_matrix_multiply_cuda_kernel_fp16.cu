#include <cuda_fp16.h>

extern "C" __global__
void execute_task_fp16(
    const int rows,
    const int K,
    const int cols,
    const half* __restrict__ A,
    const half* __restrict__ B,   
    half* __restrict__ C        
){

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    // Use a float accumulator for better precision and to avoid overflow.
    float acc = 0.f;

    // The core logic remains the same.
    // half operands are promoted to float for the multiplication and addition.
    for (int k = 0; k < K; ++k) {
        acc += __half2float(A[r * K + k]) * __half2float(B[k * cols + c]);
    }

    // Convert the final float result back to half for storage.
    C[r * cols + c] = __float2half(acc);
}
