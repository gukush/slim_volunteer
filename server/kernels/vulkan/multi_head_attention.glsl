#version 450
#extension GL_EXT_buffer_reference : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct Dims {
  uint seq_len;
  uint d_k;
  uint d_v;
  uint _pad;
};

layout(set = 0, binding = 0, std430) readonly buffer QBuffer {
  float Q[];
};
layout(set = 0, binding = 1, std430) readonly buffer KBuffer {
  float K[];
};
layout(set = 0, binding = 2, std430) readonly buffer VBuffer {
  float V[];
};
layout(set = 0, binding = 3, std430) buffer OutputBuffer {
  float output[];
};
layout(set = 0, binding = 4) uniform DimsUniform {
  Dims dims;
};

shared float scratch[256];
shared float wg_max;
shared float wg_sum;

void main() {
  uint out_row = gl_WorkGroupID.x;
  uint out_col = gl_WorkGroupID.y;
  uint lane = gl_LocalInvocationID.x;

  bool is_active = (out_row < dims.seq_len) && (out_col < dims.d_v);

  float scale = 1.0 / sqrt(float(dims.d_k));

  if (lane == 0) { wg_max = -1e30; }
  barrier();

  uint base = 0;
  do {
    uint k_idx = base + lane;

    float s = -1e30;

    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0;
      for (uint d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      s = acc * scale;
    }

    scratch[lane] = s;
    barrier();

    if (lane == 0) {
      uint tile_n = min(256u, dims.seq_len - base);
      float tmax = -1e30;
      for (uint i = 0; i < tile_n; ++i) {
        tmax = max(tmax, scratch[i]);
      }
      wg_max = max(wg_max, tmax);
    }
    barrier();

    base += 256;
  } while (base < dims.seq_len);

  if (lane == 0) { scratch[0] = wg_max; }
  barrier();
  float gmax = scratch[0];

  if (lane == 0) { wg_sum = 0.0; }
  barrier();

  base = 0;
  do {
    uint k_idx = base + lane;

    float ex = 0.0;
    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0;
      for (uint d = 0; d < dims.d_k; ++d) {
        float qv = Q[out_row * dims.d_k + d];
        float kv = K[k_idx * dims.d_k + d];
        acc += qv * kv;
      }
      ex = exp(acc * scale - gmax);
    }

    scratch[lane] = ex;
    barrier();

    if (lane == 0) {
      uint tile_n = min(256u, dims.seq_len - base);
      float tsum = 0.0;
      for (uint i = 0; i < tile_n; ++i) {
        tsum += scratch[i];
      }
      wg_sum += tsum;
    }
    barrier();

    base += 256;
  } while (base < dims.seq_len);

  if (lane == 0) { scratch[0] = max(wg_sum, 1e-20); }
  barrier();
  float denom = scratch[0];

  float partial = 0.0;

  base = 0;
  do {
    uint k_idx = base + lane;

    if (is_active && k_idx < dims.seq_len) {
      float acc = 0.0;
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
  } while (base < dims.seq_len);

  scratch[lane] = partial;
  barrier();

  for (uint stride = 128; stride > 0; stride /= 2) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    barrier();
  }

  if (lane == 0 && is_active) {
    output[out_row * dims.d_v + out_col] = scratch[0];
  }
}
