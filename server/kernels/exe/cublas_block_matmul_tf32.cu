// cublas_block_matmul_tf32.cu
// Build (Ampere+ recommended): 
//   nvcc -O3 -std=c++17 -arch=sm_80 -lcublas -o exe-block-matmul-cublas-tf32 cublas_block_matmul_tf32.cu
//
// Behavior:
// - Reads M, K, N (int32) then A[M*K], B[K*N] as float32 from stdin (row-major).
// - Uses cuBLAS GEMM (float32) with TF32 Tensor Cores enabled to compute C = A @ B.
// - Writes C[M*N] as float32 to stdout (row-major).
//
// Notes:
// - Converts to column-major on device for cuBLAS.
// - Explicitly enables TF32 tensor-op math; on pre-Ampere GPUs, this is ignored.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2);} } while(0)

#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    std::fprintf(stderr,"cuBLAS error %s:%d: status=%d\n",__FILE__,__LINE__,(int)s); std::exit(3);} } while(0)

// Convert row-major float (rows x cols) -> column-major float
__global__ void row_to_col_f32(const float* __restrict__ src, float* __restrict__ dst, int rows, int cols){
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
    if(i < rows && j < cols){
        dst[i + j * rows] = src[i * cols + j];
    }
}

// Convert column-major float -> row-major float
__global__ void col_to_row_f32(const float* __restrict__ src, float* __restrict__ dst, int rows, int cols){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < rows && j < cols){
        dst[i * cols + j] = src[i + j * rows];
    }
}

int main(){
    // Read dims
    i32 M=0,K=0,N=0;
    read_exact(&M,4); read_exact(&K,4); read_exact(&N,4);
    if(M<0||K<0||N<0) die("Negative dimensions are not allowed.");
    size_t aN = (size_t)M * (size_t)K;
    size_t bN = (size_t)K * (size_t)N;
    size_t cN = (size_t)M * (size_t)N;

    // Host buffers
    std::vector<float> A(aN), B(bN), C(cN, 0.0f);
    if(aN) read_exact(A.data(), aN*sizeof(float));
    if(bN) read_exact(B.data(), bN*sizeof(float));
    if(M==0 || K==0 || N==0){
        if(cN) write_exact(C.data(), cN*sizeof(float));
        return 0;
    }

    // Device buffers
    float *dA_row=nullptr, *dB_row=nullptr;
    float *dA_col=nullptr, *dB_col=nullptr, *dC_col=nullptr;
    CUDA_CHECK(cudaMalloc(&dA_row, aN*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB_row, bN*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA_row, A.data(), aN*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_row, B.data(), bN*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dA_col, (size_t)M*(size_t)K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB_col, (size_t)K*(size_t)N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_col, (size_t)M*(size_t)N*sizeof(float)));

    dim3 block(32, 8);
    dim3 gridA((K+block.x-1)/block.x, (M+block.y-1)/block.y);
    dim3 gridB((N+block.x-1)/block.x, (K+block.y-1)/block.y);
    dim3 gridC((N+block.x-1)/block.x, (M+block.y-1)/block.y);
    row_to_col_f32<<<gridA, block>>>(dA_row, dA_col, M, K);
    row_to_col_f32<<<gridB, block>>>(dB_row, dB_col, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dA_row));
    CUDA_CHECK(cudaFree(dB_row));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Enable TF32 Tensor Core math (ignored on pre-Ampere)
    #ifdef CUBLAS_TF32_TENSOR_OP_MATH
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    #else
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    #endif

    const float alpha = 1.0f, beta = 0.0f;
    // Column-major GEMM with TF32 acceleration
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        dA_col, CUDA_R_32F, M,
        dB_col, CUDA_R_32F, K,
        &beta,
        dC_col, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Copy back as row-major
    float *dC_row=nullptr;
    CUDA_CHECK(cudaMalloc(&dC_row, cN*sizeof(float)));
    col_to_row_f32<<<gridC, block>>>(dC_col, dC_row, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC_row, cN*sizeof(float), cudaMemcpyDeviceToHost));

    // cleanup
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(dA_col));
    CUDA_CHECK(cudaFree(dB_col));
    CUDA_CHECK(cudaFree(dC_col));
    CUDA_CHECK(cudaFree(dC_row));

    // write
    write_exact(C.data(), cN*sizeof(float));
    return 0;
}
