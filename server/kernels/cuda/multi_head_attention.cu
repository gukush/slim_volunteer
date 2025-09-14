// File: kernels/multi_head_attention.cu
// Device kernel (to be compiled at runtime by your CUDA executor).
// Signature must be: execute_task(int seq_len, int d_k, int d_v, Q, K, V, output)

extern "C" __global__
void execute_task(const int seq_len,
                  const int d_k,
                  const int d_v,
                  const float* __restrict__ Q,
                  const float* __restrict__ K,
                  const float* __restrict__ V,
                  float* __restrict__ O)
{
    // One block computes a single output element O[row, col].
    const int row = blockIdx.x;
    const int col = blockIdx.y;
    if (row >= seq_len || col >= d_v) return;

    const float scale = rsqrtf((float)d_k);

    // Fixed-size shared buffer for reductions (256 threads expected).
    __shared__ float sdata[256];

    // 1) Max over scores for numerical stability
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

    // reduce max
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    const float gmax = sdata[0];

    // 2) Denominator = sum(exp(score - gmax))
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

    // 3) Weighted sum with V
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

    if (threadIdx.x == 0) {
        O[row * d_v + col] = sdata[0];
    }
}
