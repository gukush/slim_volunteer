// File: native_cuda_multi_head_attention.cu
// Standalone native client binary for single-head attention (CUDA).
// Build (Linux/macOS with NVCC):
//   nvcc -O3 -use_fast_math -o native_cuda_multi_head_attention native_cuda_multi_head_attention.cu
//
// Uses common protocol for standardized input/output

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include "common_protocol_cuda.h"

using i32 = int32_t;

static void die(const char* msg, cudaError_t err = cudaSuccess) {
    if (err != cudaSuccess) std::fprintf(stderr, "%s (CUDA error %d: %s)\n", msg, (int)err, cudaGetErrorString(err));
    else                    std::fprintf(stderr, "%s\n", msg);
    std::fflush(stderr);
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
    auto* p = static_cast<const unsigned char*>(src);
    size_t put = 0;
    while (put < n){
        size_t w = std::fwrite(p + put, 1, n - put, stdout);
        if (w == 0) die("Write error to stdout");
        put += w;
    }
}

// ---- device kernel (same as runtime-compiled one) ----
__global__ void execute_task(const int seq_len,
                             const int d_k,
                             const int d_v,
                             const float* __restrict__ Q,
                             const float* __restrict__ K,
                             const float* __restrict__ V,
                             float* __restrict__ O)
{
    const int row = blockIdx.x;
    const int col = blockIdx.y;
    if (row >= seq_len || col >= d_v) return;

    const float scale = rsqrtf((float)d_k);
    __shared__ float sdata[256];

    float local_max = -1e30f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float* q = Q + row * d_k;
        const float* k = K + j   * d_k;
        float acc = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_k; ++d) acc += q[d] * k[d];
        local_max = fmaxf(local_max, acc * scale);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    const float gmax = sdata[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float* q = Q + row * d_k;
        const float* k = K + j   * d_k;
        float acc = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_k; ++d) acc += q[d] * k[d];
        local_sum += __expf(acc * scale - gmax);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    const float denom = fmaxf(sdata[0], 1e-20f);

    float local_out = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float* q = Q + row * d_k;
        const float* k = K + j   * d_k;
        float acc = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < d_k; ++d) acc += q[d] * k[d];
        const float p = __expf(acc * scale - gmax) / denom;
        local_out += p * V[j * d_v + col];
    }
    sdata[threadIdx.x] = local_out;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) O[row * d_v + col] = sdata[0];
}

int main() {
    EXE_LOG_INFO("Starting CUDA multi-head attention");

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
    EXE_VALIDATE(header.num_inputs >= 3, "Expected at least 3 input buffers (Q, K, V)");

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

    // Extract dimensions from input buffers
    // Q: [seq_len, d_k], K: [seq_len, d_k], V: [seq_len, d_v]
    EXE_VALIDATE(input_descriptors[0].dimensions[0] > 0 && input_descriptors[0].dimensions[1] > 0,
                 "Invalid dimensions for Q matrix");
    EXE_VALIDATE(input_descriptors[1].dimensions[0] > 0 && input_descriptors[1].dimensions[1] > 0,
                 "Invalid dimensions for K matrix");
    EXE_VALIDATE(input_descriptors[2].dimensions[0] > 0 && input_descriptors[2].dimensions[1] > 0,
                 "Invalid dimensions for V matrix");

    i32 seq_len = input_descriptors[0].dimensions[0];
    i32 d_k = input_descriptors[0].dimensions[1];
    i32 d_v = input_descriptors[2].dimensions[1];

    EXE_VALIDATE(input_descriptors[0].dimensions[0] == input_descriptors[1].dimensions[0] &&
                 input_descriptors[0].dimensions[0] == input_descriptors[2].dimensions[0],
                 "Sequence length mismatch between Q, K, V");
    EXE_VALIDATE(input_descriptors[0].dimensions[1] == input_descriptors[1].dimensions[1],
                 "Key dimension mismatch between Q and K");

    EXE_LOG_INFO("Attention dimensions: seq_len=" << seq_len << ", d_k=" << d_k << ", d_v=" << d_v);

    // Convert input buffers to float arrays
    EXE_VALIDATE(input_descriptors[0].data_type == static_cast<uint32_t>(DataType::FLOAT32),
                 "Q matrix must be float32");
    EXE_VALIDATE(input_descriptors[1].data_type == static_cast<uint32_t>(DataType::FLOAT32),
                 "K matrix must be float32");
    EXE_VALIDATE(input_descriptors[2].data_type == static_cast<uint32_t>(DataType::FLOAT32),
                 "V matrix must be float32");

    const float* Q = reinterpret_cast<const float*>(input_buffers[0].data());
    const float* K = reinterpret_cast<const float*>(input_buffers[1].data());
    const float* V = reinterpret_cast<const float*>(input_buffers[2].data());

    const size_t qElems = (size_t)seq_len * (size_t)d_k;
    const size_t kElems = qElems;
    const size_t vElems = (size_t)seq_len * (size_t)d_v;
    const size_t oElems = (size_t)seq_len * (size_t)d_v;

    EXE_VALIDATE(input_buffers[0].size() == qElems * sizeof(float), "Q matrix size mismatch");
    EXE_VALIDATE(input_buffers[1].size() == kElems * sizeof(float), "K matrix size mismatch");
    EXE_VALIDATE(input_buffers[2].size() == vElems * sizeof(float), "V matrix size mismatch");

    // Allocate host output buffer
    std::vector<float> Oh(oElems);

    // CUDA setup
    int dev = 0;
    if (const char* env = std::getenv("CUDA_DEVICE")) dev = std::atoi(env);
    cudaError_t ce;
    if ((ce = cudaSetDevice(dev)) != cudaSuccess) die("cudaSetDevice failed", ce);

    float *Qd=nullptr, *Kd=nullptr, *Vd=nullptr, *Od=nullptr;
    size_t Qb = qElems*sizeof(float), Kb = kElems*sizeof(float), Vb = vElems*sizeof(float), Ob = oElems*sizeof(float);

    if ((ce = cudaMalloc(&Qd, Qb)) != cudaSuccess) die("cudaMalloc Q failed", ce);
    if ((ce = cudaMalloc(&Kd, Kb)) != cudaSuccess) die("cudaMalloc K failed", ce);
    if ((ce = cudaMalloc(&Vd, Vb)) != cudaSuccess) die("cudaMalloc V failed", ce);
    if ((ce = cudaMalloc(&Od, Ob)) != cudaSuccess) die("cudaMalloc O failed", ce);

    if ((ce = cudaMemcpy(Qd, Q, Qb, cudaMemcpyHostToDevice)) != cudaSuccess) die("H2D Q failed", ce);
    if ((ce = cudaMemcpy(Kd, K, Kb, cudaMemcpyHostToDevice)) != cudaSuccess) die("H2D K failed", ce);
    if ((ce = cudaMemcpy(Vd, V, Vb, cudaMemcpyHostToDevice)) != cudaSuccess) die("H2D V failed", ce);

    dim3 grid((unsigned)seq_len, (unsigned)d_v, 1);
    dim3 block(256, 1, 1);

    EXE_LOG_INFO("Launching attention kernel with grid(" << grid.x << "," << grid.y << "," << grid.z
                 << ") block(" << block.x << "," << block.y << "," << block.z << ")");

    execute_task<<<grid, block>>>(seq_len, d_k, d_v, Qd, Kd, Vd, Od);
    if ((ce = cudaGetLastError()) != cudaSuccess) die("Kernel launch failed", ce);
    if ((ce = cudaDeviceSynchronize()) != cudaSuccess) die("Kernel failed", ce);

    if ((ce = cudaMemcpy(Oh.data(), Od, Ob, cudaMemcpyDeviceToHost)) != cudaSuccess) die("D2H O failed", ce);

    // Write result to stdout
    write_exact(Oh.data(), Ob);

    EXE_LOG_INFO("Multi-head attention completed successfully");

    cudaFree(Qd); cudaFree(Kd); cudaFree(Vd); cudaFree(Od);
    return 0;
}
