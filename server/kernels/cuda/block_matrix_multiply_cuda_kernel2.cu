#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// Tensor Core (WMMA) block-matmul.
// - Computes one block_size x block_size result tile made of 16x16 WMMA tiles
// - Assumes row-major inputs/outputs
// - Works even if block_size is not a multiple of 16 (last tile is zero-padded)

extern "C" __global__ void block_matrix_multiply(
    int block_size,                   // Uniform 0
    int matrix_size,                  // Uniform 1 (unused but kept for parity)
    const float* __restrict__ block_a,// Input 0 (row-major, f32)
    const float* __restrict__ block_b,// Input 1 (row-major, f32)
    float* __restrict__ partial_result// Output 0 (row-major, f32)
){
#if __CUDA_ARCH__ >= 700
    // Which 16x16 output tile do we compute?
    const int tile_m = blockIdx.y * 16;   // row offset in C
    const int tile_n = blockIdx.x * 16;   // col offset in C

    // We'll let a single warp compute one 16x16 tile via WMMA.
    // If you keep blockDim=(16,16,1) like the old kernel, only warp 0 will do work.
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id    = linear_tid / warpSize; // 0..(threads/32 - 1)
    const int lane_id    = linear_tid % warpSize;

    // Shared staging to convert f32 -> f16 for A and B tiles
    __shared__ half  shA[16*16];
    __shared__ half  shB[16*16];
    __shared__ float shC[16*16];  // for ragged-edge safe store

    if (warp_id == 0) {
      // WMMA fragments
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>                c_frag;

      wmma::fill_fragment(c_frag, 0.0f);

      // K loop in 16-wide chunks
      for (int k = 0; k < block_size; k += 16) {
        // How big are the active subtiles (handle tails)?
        const int a_rows = min(16, block_size - tile_m);
        const int a_cols = min(16, block_size - k);
        const int b_rows = min(16, block_size - k);
        const int b_cols = min(16, block_size - tile_n);

        // Convert current 16x16 A and B tiles from f32 to f16 into shared
        for (int i = lane_id; i < 256; i += 32) {
          const int r = i / 16;
          const int c = i % 16;

          // A(tile_m:tile_m+16, k:k+16)
          float a_val = (r < a_rows && c < a_cols)
                        ? block_a[(tile_m + r) * block_size + (k + c)]
                        : 0.0f;
          // B(k:k+16, tile_n:tile_n+16)
          float b_val = (r < b_rows && c < b_cols)
                        ? block_b[(k + r) * block_size + (tile_n + c)]
                        : 0.0f;

          shA[i] = __float2half_rn(a_val);
          shB[i] = __float2half_rn(b_val);
        }
        __syncthreads();

        // Load into WMMA fragments and multiply-accumulate
        wmma::load_matrix_sync(a_frag, shA, 16);
        wmma::load_matrix_sync(b_frag, shB, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
      }

      // Store: use shared buffer so we can clip ragged tiles cleanly
      wmma::store_matrix_sync(shC, c_frag, 16, wmma::mem_row_major);
      __syncthreads();

      for (int i = lane_id; i < 256; i += 32) {
        const int r = i / 16;
        const int c = i % 16;
        if (tile_m + r < block_size && tile_n + c < block_size) {
          partial_result[(tile_m + r) * block_size + (tile_n + c)] = shC[i];
        }
      }
    }
#else
    // Fallback (no tensor cores): naive per-thread multiply for compatibility
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < block_size && col < block_size) {
      float sum = 0.0f;
      for (int kk = 0; kk < block_size; ++kk) {
        sum += block_a[row * block_size + kk] * block_b[kk * block_size + col];
      }
      partial_result[row * block_size + col] = sum;
    }
#endif
}
