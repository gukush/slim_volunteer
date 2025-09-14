// exe_distributed_sort.cu
// Native CUDA bitonic sort for distributed sorting
// Build: nvcc -O3 -std=c++17 -arch=sm_70 -o exe_distributed_sort exe_distributed_sort.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>

using i32 = int32_t;
using u32 = uint32_t;

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

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Bitonic sort kernel
__global__ void bitonic_sort_kernel(
    u32* data,
    u32 n,
    u32 k,
    u32 j,
    bool ascending)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 ixj = idx ^ j;
    
    if (ixj > idx && idx < n && ixj < n) {
        u32 a = data[idx];
        u32 b = data[ixj];
        
        bool swap_condition = ascending ? (a > b) : (a < b);
        
        // Check if we're in an ascending or descending sequence
        bool in_ascending_seq = ((idx & k) == 0);
        if (in_ascending_seq) {
            swap_condition = ascending ? (a > b) : (a < b);
        } else {
            swap_condition = ascending ? (a < b) : (a > b);
        }
        
        if (swap_condition) {
            data[idx] = b;
            data[ixj] = a;
        }
    }
}

// Optimized bitonic sort for power-of-2 sizes
__global__ void bitonic_sort_step(
    u32* data,
    u32 n,
    u32 step,
    u32 stage,
    bool ascending)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n / 2) return;
    
    u32 distance = 1 << (step - stage - 1);
    u32 block_size = 1 << step;
    
    u32 left_idx = ((idx / distance) * distance * 2) + (idx % distance);
    u32 right_idx = left_idx + distance;
    
    if (right_idx >= n) return;
    
    u32 left_val = data[left_idx];
    u32 right_val = data[right_idx];
    
    // Determine if this block should be ascending or descending
    u32 block_id = left_idx / block_size;
    bool block_ascending = ascending ^ (block_id & 1);
    
    bool should_swap = block_ascending ? (left_val > right_val) : (left_val < right_val);
    
    if (should_swap) {
        data[left_idx] = right_val;
        data[right_idx] = left_val;
    }
}

void launch_bitonic_sort(u32* d_data, u32 n, bool ascending) {
    // Find the next power of 2 >= n
    u32 padded_n = 1;
    while (padded_n < n) padded_n <<= 1;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((padded_n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Bitonic sort algorithm
    for (u32 step = 1; step <= 32 && (1u << step) <= padded_n; step++) {
        for (u32 stage = 0; stage < step; stage++) {
            bitonic_sort_step<<<grid, block>>>(d_data, padded_n, step, stage, ascending);
            cudaDeviceSynchronize();
        }
    }
}

int main(){
    // 1) Read uniforms: originalSize, paddedSize, ascending
    i32 originalSize = 0, paddedSize = 0, ascending = 1;
    read_exact(&originalSize, 4);
    read_exact(&paddedSize, 4); 
    read_exact(&ascending, 4);
    
    if(originalSize <= 0 || paddedSize <= 0 || paddedSize < originalSize) {
        die("invalid uniforms");
    }
    
    // 2) Read input data
    std::vector<u32> host_data(paddedSize);
    read_exact(host_data.data(), paddedSize * sizeof(u32));
    
    // 3) GPU allocation and copy
    u32* d_data = nullptr;
    cudaMalloc(&d_data, paddedSize * sizeof(u32));
    cudaMemcpy(d_data, host_data.data(), paddedSize * sizeof(u32), cudaMemcpyHostToDevice);
    
    // 4) Sort
    launch_bitonic_sort(d_data, paddedSize, ascending != 0);
    
    // Check for CUDA errors
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 2;
    }
    
    // 5) Copy back and write result
    cudaMemcpy(host_data.data(), d_data, paddedSize * sizeof(u32), cudaMemcpyDeviceToHost);
    
    // Write only the original size (trim padding)
    write_exact(host_data.data(), originalSize * sizeof(u32));
    
    // Cleanup
    cudaFree(d_data);
    return 0;
}
