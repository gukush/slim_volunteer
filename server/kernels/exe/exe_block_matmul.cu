// native_cuda_block_matmul.cu
// Build: nvcc -O3 -std=c++17 -arch=sm_70 -o exe-block-matmul-cuda native_cuda_block_matmul.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <iostream>

using i32 = int32_t;

static void die(const char* msg){
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static void read_exact(void* dst, size_t n){
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t got = 0;
    while(got < n){
        size_t r = std::fread(p + got, 1, n - got, stdin);
        if(r == 0) die("EOF while reading stdin");
        got += r;
    }
}
static void write_exact(const void* src, size_t n){
    const uint8_t* p = static_cast<const uint8_t*>(src);
    size_t put = 0;
    while(put < n){
        size_t w = std::fwrite(p + put, 1, n - put, stdout);
        if(w == 0) die("write error to stdout");
        put += w;
    }
}

#ifndef TILE
#define TILE 16
#endif

__global__ void block_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int rows, int K, int cols)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int c = blockIdx.x * TILE + threadIdx.x; // col in C
    int r = blockIdx.y * TILE + threadIdx.y; // row in C

    float acc = 0.0f;
    int tiles = (K + TILE - 1) / TILE;

    for(int t=0; t<tiles; ++t){
        int kx = t*TILE + threadIdx.x;
        int ky = t*TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (r<rows && kx<K) ? A[r*K + kx] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (c<cols && ky<K) ? B[ky*cols + c] : 0.0f;

        __syncthreads();

        #pragma unroll
        for(int k=0;k<TILE;k++){
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(r<rows && c<cols) C[r*cols + c] = acc;
}

int main(){
    // 1) uniforms
    i32 rows=0, K=0, cols=0;
    read_exact(&rows,4); read_exact(&K,4); read_exact(&cols,4);
    if(rows<=0 || K<=0 || cols<=0) die("invalid uniforms");

    // 2) inputs
    size_t aN = (size_t)rows*(size_t)K;
    size_t bN = (size_t)K*(size_t)cols;
    size_t cN = (size_t)rows*(size_t)cols;

    std::vector<float> A(aN), B(bN), C(cN);
    read_exact(A.data(), aN*sizeof(float));
    read_exact(B.data(), bN*sizeof(float));

    // 3) device
    float *dA=nullptr,*dB=nullptr,*dC=nullptr;
    cudaMalloc(&dA, aN*sizeof(float));
    cudaMalloc(&dB, bN*sizeof(float));
    cudaMalloc(&dC, cN*sizeof(float));
    cudaMemcpy(dA, A.data(), aN*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bN*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE, 1);
    dim3 grid( (cols+TILE-1)/TILE, (rows+TILE-1)/TILE, 1 );
    block_matmul_kernel<<<grid, block>>>(dA, dB, dC, rows, K, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 2;
    }

    cudaMemcpy(C.data(), dC, cN*sizeof(float), cudaMemcpyDeviceToHost);
    write_exact(C.data(), cN*sizeof(float));

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
