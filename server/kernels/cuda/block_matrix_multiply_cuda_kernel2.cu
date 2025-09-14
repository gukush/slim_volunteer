#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// Computes C = A * B using Tensor Cores (WMMA).
// - A: rows x K, row-major
// - B: K x cols, row-major
// - C: rows x cols, row-major
//
// This kernel requires a GPU with compute capability 7.0 or higher (Volta, Turing, Ampere, etc.).
//
// Recommended launch configuration:
// - Each thread block computes one 16x16 tile of the output matrix C.
// - Grid dimensions:
//   dim3 grid_dim((cols + 15) / 16, (rows + 15) / 16, 1);
// - Block dimensions:
//   dim3 block_dim(32, 1, 1); // Exactly one warp per block.

extern "C" __global__ void execute_task(
    const int rows,                   // Uniform 0
    const int K,                      // Uniform 1
    const int cols,                   // Uniform 2
    const float* __restrict__ A,      // Input 0 (row-major, f32)
    const float* __restrict__ B,      // Input 1 (row-major, f32)
    float* __restrict__ C             // Output 0 (row-major, f32)
) {
#if __CUDA_ARCH__ >= 700
    // Each block computes one 16x16 tile of the C matrix.
    const int tile_m = blockIdx.y * 16;   // Target row in C
    const int tile_n = blockIdx.x * 16;   // Target col in C

    // Early exit for blocks that are entirely out of bounds.
    if (tile_m >= rows || tile_n >= cols) {
        return;
    }

    // This kernel is designed for one warp per block.
    const int lane_id = threadIdx.x;

    // --- WMMA Fragments ---
    // These live in each thread's registers.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // --- Shared Memory ---
    // Used for staging data from global memory for loading into fragments.
    __shared__ half shA[16 * 16];
    __shared__ half shB[16 * 16];

    // Initialize the accumulator fragment to all zeros.
    wmma::fill_fragment(c_frag, 0.0f);

    // --- Main Loop ---
    // Loop over the K dimension in 16-element chunks.
    for (int k = 0; k < K; k += 16) {
        // --- Load A and B tiles from Global to Shared Memory ---
        // Each of the 32 threads in the warp loads 8 elements (256 elements / 32 threads).
        // This loop is strided by the warp size (32).
        for (int i = lane_id; i < 256; i += 32) {
            const int r = i / 16; // Local row in the 16x16 tile
            const int c = i % 16; // Local col in the 16x16 tile

            // Load A tile: A[tile_m : tile_m+16, k : k+16] with boundary checks.
            float a_val = (tile_m + r < rows && k + c < K)
                            ? A[(tile_m + r) * K + (k + c)]
                            : 0.0f;

            // Load B tile: B[k : k+16, tile_n : tile_n+16] with boundary checks.
            float b_val = (k + r < K && tile_n + c < cols)
                            ? B[(k + r) * cols + (tile_n + c)]
                            : 0.0f;
            
            // Convert from f32 to f16 and store in shared memory.
            shA[i] = __float2half_rn(a_val);
            shB[i] = __float2half_rn(b_val);
        }
        __syncthreads(); // Wait for all threads to finish loading into shared memory.

        // --- Load data from Shared Memory into WMMA Fragments ---
        wmma::load_matrix_sync(a_frag, shA, 16);
        wmma::load_matrix_sync(b_frag, shB, 16);

        // --- Tensor Core Computation ---
        // Perform the 16x16x16 matrix multiply-accumulate operation.
        // D = A * B + C
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        // Ensure shared memory writes are complete before the next iteration's reads.
        __syncthreads();
    }

    // --- Store Result from WMMA Fragments to Global Memory ---
    // Use shared memory as a staging buffer for a clean, boundary-checked write.
    __shared__ float shC[16 * 16];
    wmma::store_matrix_sync(shC, c_frag, 16, wmma::mem_row_major);
    __syncthreads(); // Wait for all fragments to be stored in shared memory.

    // Each of the 32 threads writes 8 elements from shared to global memory.
    for (int i = lane_id; i < 256; i += 32) {
        const int r = i / 16;
        const int c = i % 16;

        // Check boundaries to avoid writing outside the C matrix.
        if (tile_m + r < rows && tile_n + c < cols) {
            C[(tile_m + r) * cols + (tile_n + c)] = shC[i];
        }
    }
#endif // __CUDA_ARCH__ >= 700
}
