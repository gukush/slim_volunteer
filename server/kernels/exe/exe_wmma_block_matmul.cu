// wmma_block_matmul.cu
// The program reads M, K, N (int32) then A[M*K] and B[K*N] (float32) from stdin,
// computes C = A @ B using Tensor Cores (WMMA), and writes C[M*N] (float32) to stdout.
//
// Notes:
// - Uses half-precision inputs (converted from float32) and float32 accumulation.
// - Handles arbitrary (non-multiple-of-16) sizes by padding internally.
// - A is treated as row-major; B is transposed into a padded column-major buffer for fast WMMA loads.

#include <cuda_runtime.h>
#include <mma.h>            
#include <cuda_fp16.h>      
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <limits>

using i32 = int32_t;

static void die(const char* msg){
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static void read_exact(void* dst, size_t n){
    size_t off = 0;
    while(off < n){
        size_t got = std::fread((char*)dst + off, 1, n - off, stdin);
        if(got == 0) die("read_exact: unexpected EOF");
        off += got;
    }
}

static void write_exact(const void* src, size_t n){
    size_t off = 0;
    while(off < n){
        size_t put = std::fwrite((char*)src + off, 1, n - off, stdout);
        if(put == 0) die("write_exact: short write");
        off += put;
    }
}

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _err = (call);                                      \
    if (_err != cudaSuccess) {                                      \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(_err));          \
        std::exit(2);                                               \
    }                                                               \
} while(0)

// Round up x to a multiple of m
__host__ __device__ static inline int ceil_div(int x, int m){
    return (x + m - 1) / m;
}
__host__ __device__ static inline int round_up(int x, int m){
    return m * ceil_div(x, m);
}

// Convert and pad A (row-major, MxK) from float -> __half into Ap (row-major, Mp x Kp)
__global__ void pad_cast_A_to_half_rowmajor(const float* __restrict__ A, __half* __restrict__ Ap,
                                            int M, int K, int Mp, int Kp){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Mp * Kp;
    if(idx >= total) return;
    int r = idx / Kp;     // row in padded A
    int c = idx % Kp;     // col in padded A
    float v = 0.0f;
    if(r < M && c < K){
        v = A[r * K + c];
    }
    Ap[r * Kp + c] = __float2half(v);
}

// Convert and pad B (row-major, KxN) from float -> __half into Bp_col (column-major, Kp x Np)
__global__ void pad_cast_B_to_half_colmajor(const float* __restrict__ B, __half* __restrict__ Bp_col,
                                            int K, int N, int Kp, int Np){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Kp * Np;
    if(idx >= total) return;
    int k = idx % Kp;     // row (in column-major)
    int n = idx / Kp;     // col
    float v = 0.0f;
    if(k < K && n < N){
        v = B[k * N + n]; // original B is row-major KxN
    }
    Bp_col[n * Kp + k] = __float2half(v); // column-major write
}

// WMMA kernel: C = A @ B; A: row-major (Mp x Kp), B: col-major (Kp x Np), C: row-major (Mp x Np)
template<int WARPS_M, int WARPS_N>
__global__ void wmma_matmul_kernel(const __half* __restrict__ Arow,
                                   const __half* __restrict__ Bcol,
                                   float* __restrict__ Crow,
                                   int Mp, int Np, int Kp){
    using namespace nvcuda;
    constexpr int WM = 16, WN = 16, WK = 16;

    // Each block computes a (WARPS_M * 16) x (WARPS_N * 16) macro-tile using WARPS_M * WARPS_N warps.
    const int warp_id  = threadIdx.x / warpSize;         // 0..(WARPS_M*WARPS_N-1)
    const int lane_id  = threadIdx.x % warpSize;
    if (warp_id >= WARPS_M * WARPS_N) return;

    const int warp_m = warp_id / WARPS_N; // 0..WARPS_M-1
    const int warp_n = warp_id % WARPS_N; // 0..WARPS_N-1

    const int tile_m = (blockIdx.y * WARPS_M + warp_m) * WM; // row offset in C
    const int tile_n = (blockIdx.x * WARPS_N + warp_n) * WN; // col offset in C

    // Accumulator fragment
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Loop over K in 16-wide steps
    for(int k0 = 0; k0 < Kp; k0 += WK){
        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> bFrag;

        const __half* aPtr = Arow + tile_m * Kp + k0;        // row-major, ld = Kp
        const __half* bPtr = Bcol + tile_n * Kp + k0;        // col-major, ld = Kp, pointer at (row=k0, col=tile_n)

        wmma::load_matrix_sync(aFrag, aPtr, Kp);
        wmma::load_matrix_sync(bFrag, bPtr, Kp);

        wmma::mma_sync(acc, aFrag, bFrag, acc);
    }

    // Store result tile back to C (row-major), ld = Np
    float* cPtr = Crow + tile_m * Np + tile_n;
    wmma::store_matrix_sync(cPtr, acc, Np, wmma::mem_row_major);
}

int main(){
    // 1) uniforms
    i32 M=0, K=0, N=0;
    read_exact(&M,4); read_exact(&K,4); read_exact(&N,4);
    if(M < 0 || K < 0 || N < 0) die("Negative dimensions are not allowed.");

    // 2) host buffers (row-major)
    size_t aN = (size_t)M * (size_t)K;
    size_t bN = (size_t)K * (size_t)N;
    size_t cN = (size_t)M * (size_t)N;
    std::vector<float> A(aN), B(bN), C(cN, 0.0f);
    read_exact(A.data(), aN * sizeof(float));
    read_exact(B.data(), bN * sizeof(float));

    // 3) pad sizes to WMMA tile (16)
    const int T = 16;
    const int Mp = round_up(M, T);
    const int Kp = round_up(K, T);
    const int Np = round_up(N, T);

    // 4) device buffers
    float *dC = nullptr;
    __half *dA = nullptr, *dBcol = nullptr;
    float *dA_tmp = nullptr, *dB_tmp = nullptr; // not used; we convert on host->device directly
    CUDA_CHECK(cudaMalloc(&dA,    (size_t)Mp * (size_t)Kp * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dBcol, (size_t)Kp * (size_t)Np * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC,    (size_t)Mp * (size_t)Np * sizeof(float)));

    // 5) copy and convert+pad on device
    float *dA_src = nullptr, *dB_src = nullptr;
    CUDA_CHECK(cudaMalloc(&dA_src, aN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB_src, bN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA_src, A.data(), aN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_src, B.data(), bN * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocksA = (Mp * Kp + threads - 1) / threads;
    int blocksB = (Kp * Np + threads - 1) / threads;

    pad_cast_A_to_half_rowmajor<<<blocksA, threads>>>(dA_src, dA, M, K, Mp, Kp);
    pad_cast_B_to_half_colmajor<<<blocksB, threads>>>(dB_src, dBcol, K, N, Kp, Np);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(dA_src));
    CUDA_CHECK(cudaFree(dB_src));

    // 6) WMMA launch
    // Use 4 warps per block arranged as 2x2 tiles -> each block computes a 32x32 region.
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;
    dim3 blockDim(32 * WARPS_M * WARPS_N, 1, 1);
    dim3 gridDim( ceil_div(Np, 16 * WARPS_N),
                  ceil_div(Mp, 16 * WARPS_M),
                  1 );

    wmma_matmul_kernel<WARPS_M, WARPS_N><<<gridDim, blockDim>>>(dA, dBcol, dC, Mp, Np, Kp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7) Copy valid MxN submatrix back
    // If M,N already multiples of 16, this is a single memcpy. Otherwise, copy row-by-row.
    std::vector<float> Cp((size_t)Mp * (size_t)Np);
    CUDA_CHECK(cudaMemcpy(Cp.data(), dC, (size_t)Mp * (size_t)Np * sizeof(float), cudaMemcpyDeviceToHost));
    for(int r=0; r<M; ++r){
        std::memcpy(&C[(size_t)r * N], &Cp[(size_t)r * Np], (size_t)N * sizeof(float));
    }

    // 8) write result and clean up
    write_exact(C.data(), cN * sizeof(float));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dBcol));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
