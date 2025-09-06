// kernels/cuda/multi_head_attention.cu
// Single-head scaled dot-product attention (tiled, uniform-control flow)
// Launch: dim3 block(256, 1, 1), dim3 grid(seq_len, d_v, 1)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

struct Dims {
  unsigned int seq_len;
  unsigned int d_k;
  unsigned int d_v;
  unsigned int _pad;
};

__global__ void attention(const float* Q, const float* K, const float* V, float* output, Dims dims) {
  unsigned int out_row = blockIdx.x;
  unsigned int out_col = blockIdx.y;
  unsigned int lane = threadIdx.x;

  bool is_active = (out_row < dims.seq_len) && (out_col < dims.d_v);

  float scale = 1.0f / sqrtf(static_cast<float>(dims.d_k));

  __shared__ float scratch[256];
  __shared__ float wg_max;
  __shared__ float wg_sum;

  // Pass 1: global max(score)
  if (lane == 0) { wg_max = -1e30f; }
  __syncthreads();

  unsigned int base = 0;
  while (base < dims.seq_len) {
    unsigned int k_idx = base + lane;

    float s = -1e30f;

    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0f;
      for (unsigned int d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      s = acc * scale;
    }

    scratch[lane] = s;
    __syncthreads();

    if (lane == 0) {
      unsigned int tile_n = min(256u, dims.seq_len - base);
      float tmax = -1e30f;
      for (unsigned int i = 0; i < tile_n; ++i) {
        tmax = fmaxf(tmax, scratch[i]);
      }
      wg_max = fmaxf(wg_max, tmax);
    }
    __syncthreads();

    base += 256;
  }

  // Broadcast global max
  if (lane == 0) { scratch[0] = wg_max; }
  __syncthreads();
  float gmax = scratch[0];

  // Pass 2: global sum(exp(..))
  if (lane == 0) { wg_sum = 0.0f; }
  __syncthreads();

  base = 0;
  while (base < dims.seq_len) {
    unsigned int k_idx = base + lane;

    float ex = 0.0f;
    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0f;
      for (unsigned int d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      ex = expf(acc * scale - gmax);
    }

    scratch[lane] = ex;
    __syncthreads();

    if (lane == 0) {
      unsigned int tile_n = min(256u, dims.seq_len - base);
      float tsum = 0.0f;
      for (unsigned int i = 0; i < tile_n; ++i) {
        tsum += scratch[i];
      }
      wg_sum += tsum;
    }
    __syncthreads();

    base += 256;
  }

  // Broadcast denom with epsilon
  if (lane == 0) { scratch[0] = fmaxf(wg_sum, 1e-20f); }
  __syncthreads();
  float denom = scratch[0];

  // Pass 3: weighted sum with V
  float partial = 0.0f;

  base = 0;
  while (base < dims.seq_len) {
    unsigned int k_idx = base + lane;

    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0f;
      for (unsigned int d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      float prob = expf(acc * scale - gmax) / denom;
      float vv = V[k_idx * dims.d_v + out_col];
      partial += prob * vv;
    }

    base += 256;
  }

  // Reduce 256 lanes to one
  scratch[lane] = partial;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride /= 2) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    __syncthreads();
  }

  // Only active lanes write output
  if (lane == 0 && is_active) {
    output[out_row * dims.d_v + out_col] = scratch[0];
  }
}
