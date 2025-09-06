// kernels/opencl/multi_head_attention.cl
// Single-head scaled dot-product attention (tiled, uniform-control flow)

typedef struct {
  uint seq_len;
  uint d_k;
  uint d_v;
  uint _pad;
} Dims;

__kernel void attention(
  __global const float* Q,
  __global const float* K,
  __global const float* V,
  __global float* output,
  Dims dims
) {
  uint out_row = get_group_id(0);
  uint out_col = get_group_id(1);
  uint lane = get_local_id(0);

  bool is_active = (out_row < dims.seq_len) && (out_col < dims.d_v);

  float scale = 1.0f / sqrt((float)dims.d_k);

  __local float scratch[256];
  __local float wg_max;
  __local float wg_sum;

  if (lane == 0) { wg_max = -1e30f; }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint base = 0;
  while (base < dims.seq_len) {
    uint k_idx = base + lane;

    float s = -1e30f;

    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0f;
      for (uint d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      s = acc * scale;
    }

    scratch[lane] = s;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane == 0) {
      uint tile_n = min((uint)256, dims.seq_len - base);
      float tmax = -1e30f;
      for (uint i = 0; i < tile_n; ++i) {
        tmax = max(tmax, scratch[i]);
      }
      wg_max = max(wg_max, tmax);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    base += 256;
  }

  if (lane == 0) { scratch[0] = wg_max; }
  barrier(CLK_LOCAL_MEM_FENCE);
  float gmax = scratch[0];

  if (lane == 0) { wg_sum = 0.0f; }
  barrier(CLK_LOCAL_MEM_FENCE);

  base = 0;
  while (base < dims.seq_len) {
    uint k_idx = base + lane;

    float ex = 0.0f;
    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0f;
      for (uint d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      ex = exp(acc * scale - gmax);
    }

    scratch[lane] = ex;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane == 0) {
      uint tile_n = min((uint)256, dims.seq_len - base);
      float tsum = 0.0f;
      for (uint i = 0; i < tile_n; ++i) {
        tsum += scratch[i];
      }
      wg_sum += tsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    base += 256;
  }

  if (lane == 0) { scratch[0] = max(wg_sum, 1e-20f); }
  barrier(CLK_LOCAL_MEM_FENCE);
  float denom = scratch[0];

  float partial = 0.0f;

  base = 0;
  while (base < dims.seq_len) {
    uint k_idx = base + lane;

    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0f;
      for (uint d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      float prob = exp(acc * scale - gmax) / denom;
      float vv = V[k_idx * dims.d_v + out_col];
      partial += prob * vv;
    }

    base += 256;
  }

  scratch[lane] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint stride = 128; stride > 0; stride /= 2) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lane == 0 && is_active) {
    output[out_row * dims.d_v + out_col] = scratch[0];
  }
}
