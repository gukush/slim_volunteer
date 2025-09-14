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

using i32 = int32_t;

static void die(const char* msg, cudaError_t err = cudaSuccess) {
    if (err != cudaSuccess) std::fprintf(stderr, "%s (CUDA error %d: %s)\n", msg, (int)err, cudaGetErrorString(err));
    else                    std::fprintf(stderr, "%s\n", msg);
    std::fflush(stderr);
    std::exit(1);
}

static void read_exact(void* dst, size_t n){
    auto* p = static_cast<unsigned char*>(dst);
    size_t got = 0;
    while (got < n){
        size_t r = std::fread(p + got, 1, n - got, stdin);
        if (r == 0) die("EOF while reading input stream");
        got += r;
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
    // 1) Read uniforms
    i32 seq_len=0, d_k=0, d_v=0;
    read_exact(&seq_len, 4); read_exact(&d_k, 4); read_exact(&d_v, 4);
    if (seq_len<=0 || d_k<=0 || d_v<=0) die("Invalid uniforms");

    // 2) Host buffers
    const size_t qElems = (size_t)seq_len * (size_t)d_k;
    const size_t kElems = qElems;
    const size_t vElems = (size_t)seq_len * (size_t)d_v;
    const size_t oElems = (size_t)seq_len * (size_t)d_v;

    std::vector<float> Qh(qElems), Kh(kElems), Vh(vElems), Oh(oElems);
    read_exact(Qh.data(), qElems*sizeof(float));
    read_exact(Kh.data(), kElems*sizeof(float));
    read_exact(Vh.data(), vElems*sizeof(float));

    // 3) CUDA setup
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

    if ((ce = cudaMemcpy(Qd, Qh.data(), Qb, cudaMemcpyHostToDevice)) != cudaSuccess) die("H2D Q failed", ce);
    if ((ce = cudaMemcpy(Kd, Kh.data(), Kb, cudaMemcpyHostToDevice)) != cudaSuccess) die("H2D K failed", ce);
    if ((ce = cudaMemcpy(Vd, Vh.data(), Vb, cudaMemcpyHostToDevice)) != cudaSuccess) die("H2D V failed", ce);

    dim3 grid((unsigned)seq_len, (unsigned)d_v, 1);
    dim3 block(256, 1, 1);
    execute_task<<<grid, block>>>(seq_len, d_k, d_v, Qd, Kd, Vd, Od);
    if ((ce = cudaGetLastError()) != cudaSuccess) die("Kernel launch failed", ce);
    if ((ce = cudaDeviceSynchronize()) != cudaSuccess) die("Kernel failed", ce);

    if ((ce = cudaMemcpy(Oh.data(), Od, Ob, cudaMemcpyDeviceToHost)) != cudaSuccess) die("D2H O failed", ce);

    write_exact(Oh.data(), Ob);

    cudaFree(Qd); cudaFree(Kd); cudaFree(Vd); cudaFree(Od);
    return 0;
}
