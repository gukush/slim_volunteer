// kernels/cpp/multi_head_attention.cpp
// Single-head scaled dot-product attention (serial, with stable softmax)

#include <emscripten/emscripten.h>
#include <cmath>

extern "C" {

EMSCRIPTEN_KEEPALIVE
void attention(unsigned int seq_len, unsigned int d_k, unsigned int d_v,
               const float* Q, const float* K, const float* V, float* output) {
  float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

  for (unsigned int out_row = 0; out_row < seq_len; ++out_row) {
    // Pass 1: global max(score)
    float gmax = -1e30f;
    for (unsigned int k_idx = 0; k_idx < seq_len; ++k_idx) {
      float acc = 0.0f;
      for (unsigned int d = 0; d < d_k; ++d) {
        float qv = Q[out_row * d_k + d];
        float kv = K[k_idx * d_k + d];
        acc += qv * kv;
      }
      float s = acc * scale;
      gmax = std::max(gmax, s);
    }

    // Pass 2: global sum(exp(..))
    float wg_sum = 0.0f;
    for (unsigned int k_idx = 0; k_idx < seq_len; ++k_idx) {
      float acc = 0.0f;
      for (unsigned int d = 0; d < d_k; ++d) {
        float qv = Q[out_row * d_k + d];
        float kv = K[k_idx * d_k + d];
        acc += qv * kv;
      }
      float ex = std::exp(acc * scale - gmax);
      wg_sum += ex;
    }
    float denom = std::max(wg_sum, 1e-20f);

    // Pass 3: weighted sum with V for each out_col
    for (unsigned int out_col = 0; out_col < d_v; ++out_col) {
      float partial = 0.0f;
      for (unsigned int k_idx = 0; k_idx < seq_len; ++k_idx) {
        float acc = 0.0f;
        for (unsigned int d = 0; d < d_k; ++d) {
          float qv = Q[out_row * d_k + d];
          float kv = K[k_idx * d_k + d];
          acc += qv * kv;
        }
        float prob = std::exp(acc * scale - gmax) / denom;
        float vv = V[k_idx * d_v + out_col];
        partial += prob * vv;
      }
      output[out_row * d_v + out_col] = partial;
    }
  }
}

} // extern "C"
