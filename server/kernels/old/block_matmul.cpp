// kernels/cpp/block_matmul.cpp
// Simple row-major GEMM for a single tile: C(rows x cols) = A(rows x K) * B(K x cols)
// Compatible with the "block-matmul-flex" strategy which accumulates partial sums across the K dimension.
//
// Build (via Emscripten):
//   emcc kernels/cpp/block_matmul.cpp -O3 \
//     -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web \
//     -s EXPORTED_FUNCTIONS='["_matmul"]' \
//     -s EXPORTED_RUNTIME_METHODS='["cwrap","HEAPF32","_malloc","_free"]' \
//     -s ALLOW_MEMORY_GROWTH=1 \
//     -s SINGLE_FILE=1 \
//     -o kernels/cpp/block_matmul.js
//
// The SINGLE_FILE=1 embeds the .wasm into the JS glue, so the server can ship it as text.

#include <emscripten/emscripten.h>

extern "C" {

EMSCRIPTEN_KEEPALIVE
void matmul(int rows, int K, int cols, const float* A, const float* B, float* C) {
  // Initialize C to zero (the server handles accumulation across K tiles,
  // so each tile multiply here should produce its own partial product).
  const int rc = rows * cols;
  for (int i = 0; i < rc; ++i) C[i] = 0.0f;

  for (int r = 0; r < rows; ++r) {
    const int aRow = r * K;
    const int cRow = r * cols;
    for (int k = 0; k < K; ++k) {
      const float a = A[aRow + k];
      const int bRow = k * cols;
      for (int c = 0; c < cols; ++c) {
        C[cRow + c] += a * B[bRow + c];
      }
    }
  }
}

} // extern "C"
