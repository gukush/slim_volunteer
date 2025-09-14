// native_cuda_block_matmul.cu
// Build: nvcc -O3 -std=c++17 -arch=sm_70 -o exe-block-matmul-cuda native_cuda_block_matmul.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "common_protocol_cuda.h"

using i32 = int32_t;

static void die(const char* msg){
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static void read_all_stdin(std::vector<uint8_t>& buffer) {
    const size_t CHUNK_SIZE = 1 << 20; // 1MB chunks
    uint8_t chunk[CHUNK_SIZE];
    size_t bytes_read;

    while ((bytes_read = std::fread(chunk, 1, CHUNK_SIZE, stdin)) > 0) {
        buffer.insert(buffer.end(), chunk, chunk + bytes_read);
    }

    if (std::ferror(stdin)) {
        die("Error reading from stdin");
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
    EXE_LOG_INFO("Starting CUDA block matrix multiplication");

    // Read all data from stdin
    std::vector<uint8_t> input_buffer;
    read_all_stdin(input_buffer);

    if (input_buffer.empty()) {
        EXE_LOG_ERROR("No data received from stdin");
        return 1;
    }

    EXE_LOG_INFO("Received " << input_buffer.size() << " bytes from stdin");

    // Parse protocol header
    ProtocolHeader header;
    const uint8_t* data = input_buffer.data();
    size_t remaining = input_buffer.size();

    EXE_VALIDATE(ExeProtocol::readHeader(data, remaining, header),
                 "Failed to read protocol header");

    EXE_LOG_INFO("Protocol version: " << header.version);
    EXE_LOG_INFO("Framework: " << ExeProtocol::getFrameworkName(static_cast<FrameworkType>(header.framework)));
    EXE_LOG_INFO("Data type: " << ExeProtocol::getDataTypeName(static_cast<DataType>(header.data_type)));
    EXE_LOG_INFO("Inputs: " << header.num_inputs << ", Outputs: " << header.num_outputs);

    // Skip metadata if present
    if (header.metadata_size > 0) {
        EXE_VALIDATE(remaining >= header.metadata_size, "Insufficient data for metadata");
        data += header.metadata_size;
        remaining -= header.metadata_size;
    }

    // Parse input buffers
    EXE_VALIDATE(header.num_inputs >= 2, "Expected at least 2 input buffers (A, B)");

    std::vector<std::vector<uint8_t>> input_buffers;
    std::vector<BufferDescriptor> input_descriptors;

    for (uint32_t i = 0; i < header.num_inputs; ++i) {
        BufferDescriptor desc;
        std::vector<uint8_t> buffer;

        EXE_VALIDATE(ExeProtocol::readBuffer(data, remaining, desc, buffer),
                     "Failed to read input buffer " << i);

        input_descriptors.push_back(desc);
        input_buffers.push_back(std::move(buffer));

        EXE_LOG_DEBUG("Input " << i << ": " << desc.size << " bytes, dims=["
                     << desc.dimensions[0] << "," << desc.dimensions[1]
                     << "," << desc.dimensions[2] << "," << desc.dimensions[3] << "]");
    }

    // Extract matrix dimensions from first two input buffers
    // For matrix multiplication: A(rows x K) * B(K x cols) = C(rows x cols)
    EXE_VALIDATE(input_descriptors[0].dimensions[0] > 0 && input_descriptors[0].dimensions[1] > 0,
                 "Invalid dimensions for matrix A");
    EXE_VALIDATE(input_descriptors[1].dimensions[0] > 0 && input_descriptors[1].dimensions[1] > 0,
                 "Invalid dimensions for matrix B");

    i32 rows = input_descriptors[0].dimensions[0];
    i32 K = input_descriptors[0].dimensions[1];
    i32 cols = input_descriptors[1].dimensions[1];

    EXE_VALIDATE(input_descriptors[0].dimensions[1] == input_descriptors[1].dimensions[0],
                 "Matrix dimensions don't match for multiplication");

    EXE_LOG_INFO("Matrix dimensions: A(" << rows << "x" << K << ") * B(" << K << "x" << cols << ") = C(" << rows << "x" << cols << ")");

    // Convert input buffers to float arrays
    EXE_VALIDATE(input_descriptors[0].data_type == static_cast<uint32_t>(DataType::FLOAT32),
                 "Matrix A must be float32");
    EXE_VALIDATE(input_descriptors[1].data_type == static_cast<uint32_t>(DataType::FLOAT32),
                 "Matrix B must be float32");

    const float* A = reinterpret_cast<const float*>(input_buffers[0].data());
    const float* B = reinterpret_cast<const float*>(input_buffers[1].data());

    size_t aN = (size_t)rows * (size_t)K;
    size_t bN = (size_t)K * (size_t)cols;
    size_t cN = (size_t)rows * (size_t)cols;

    EXE_VALIDATE(input_buffers[0].size() == aN * sizeof(float), "Matrix A size mismatch");
    EXE_VALIDATE(input_buffers[1].size() == bN * sizeof(float), "Matrix B size mismatch");

    // Allocate host output buffer
    std::vector<float> C(cN);

    // CUDA setup
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    cudaError_t err;

    err = cudaMalloc(&dA, aN * sizeof(float));
    EXE_VALIDATE(err == cudaSuccess, "Failed to allocate device memory for A");

    err = cudaMalloc(&dB, bN * sizeof(float));
    EXE_VALIDATE(err == cudaSuccess, "Failed to allocate device memory for B");

    err = cudaMalloc(&dC, cN * sizeof(float));
    EXE_VALIDATE(err == cudaSuccess, "Failed to allocate device memory for C");

    // Copy data to device
    err = cudaMemcpy(dA, A, aN * sizeof(float), cudaMemcpyHostToDevice);
    EXE_VALIDATE(err == cudaSuccess, "Failed to copy A to device");

    err = cudaMemcpy(dB, B, bN * sizeof(float), cudaMemcpyHostToDevice);
    EXE_VALIDATE(err == cudaSuccess, "Failed to copy B to device");

    // Launch kernel
    dim3 block(TILE, TILE, 1);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE, 1);

    EXE_LOG_INFO("Launching kernel with grid(" << grid.x << "," << grid.y << "," << grid.z
                 << ") block(" << block.x << "," << block.y << "," << block.z << ")");

    block_matmul_kernel<<<grid, block>>>(dA, dB, dC, rows, K, cols);

    err = cudaDeviceSynchronize();
    EXE_VALIDATE(err == cudaSuccess, "CUDA kernel execution failed: " << cudaGetErrorString(err));

    // Copy result back
    err = cudaMemcpy(C.data(), dC, cN * sizeof(float), cudaMemcpyDeviceToHost);
    EXE_VALIDATE(err == cudaSuccess, "Failed to copy result from device");

    // Write result to stdout
    write_exact(C.data(), cN * sizeof(float));

    EXE_LOG_INFO("Matrix multiplication completed successfully");

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
