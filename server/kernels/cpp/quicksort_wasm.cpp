// quicksort_wasm.cpp
// Build (example):
// emcc quicksort_wasm.cpp -O3 \
//   -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web \
//   -s EXPORTED_FUNCTIONS='["_quicksort_copy","_free_buffer"]' \
//   -s EXPORTED_RUNTIME_METHODS='["cwrap","HEAPU32","_malloc","_free"]' \
//   -s ALLOW_MEMORY_GROWTH=1 -s SINGLE_FILE=1 \
//   -o quicksort_wasm.js

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>

// ----- Internal sorting primitives (uint32_t) -----

static inline void u32_swap(uint32_t& a, uint32_t& b) {
  uint32_t t = a; a = b; b = t;
}

static inline uint32_t u32_median3(uint32_t a, uint32_t b, uint32_t c) {
  // Return the median of a, b, c
  if (a > b) { uint32_t t=a; a=b; b=t; }
  if (b > c) { uint32_t t=b; b=c; c=t; }
  if (a > b) { uint32_t t=a; a=b; b=t; }
  return b;
}

static inline void insertion_sort_u32(uint32_t* arr, int32_t lo, int32_t hi) {
  for (int32_t i = lo + 1; i <= hi; ++i) {
    uint32_t key = arr[i];
    int32_t j = i - 1;
    while (j >= lo && arr[j] > key) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
}

// Hoare partition. Returns index j such that [lo..j] <= pivot <= [j+1..hi]
static inline int32_t hoare_partition_u32(uint32_t* arr, int32_t lo, int32_t hi) {
  // Median-of-three pivot to reduce worst cases
  int32_t mid = lo + ((hi - lo) >> 1);
  uint32_t pivot = u32_median3(arr[lo], arr[mid], arr[hi]);

  int32_t i = lo - 1;
  int32_t j = hi + 1;
  while (true) {
    do { ++i; } while (arr[i] < pivot);
    do { --j; } while (arr[j] > pivot);
    if (i >= j) return j;
    u32_swap(arr[i], arr[j]);
  }
}

// Non-recursive quicksort with insertion sort for small ranges
static void quicksort_inplace_u32(uint32_t* arr, uint32_t length) {
  if (!arr || length < 2) return;
  // Avoid signed overflow; practical WASM heaps are < 2^31
  if (length > 0x7fffffffU) return;

  const int32_t CUTOFF = 20; // use insertion sort for small partitions

  // Simple stack of ranges [lo, hi]
  struct Range { int32_t lo, hi; };
  Range stack[64]; // enough for 2^64 elements; for WASM this is plenty
  int top = 0;
  stack[top++] = {0, (int32_t)length - 1};

  while (top) {
    Range r = stack[--top];
    int32_t lo = r.lo;
    int32_t hi = r.hi;

    while (lo < hi) {
      if (hi - lo + 1 <= CUTOFF) {
        insertion_sort_u32(arr, lo, hi);
        break;
      }

      int32_t p = hoare_partition_u32(arr, lo, hi);

      // Tail-call elimination: sort smaller side first, loop on larger
      if (p - lo < hi - (p + 1)) {
        if (lo < p) stack[top++] = {lo, p};
        lo = p + 1;
      } else {
        if (p + 1 < hi) stack[top++] = {p + 1, hi};
        hi = p;
      }
    }
  }
}

extern "C" {

// Sorts a COPY of the input data and returns a pointer to the sorted buffer.
// - data: pointer to N uint32_t values in WASM memory
// - length: number of elements
// Returns: pointer to a newly-allocated uint32_t[N] buffer with sorted data,
//          or 0 on failure (e.g., length==0 or OOM).
EMSCRIPTEN_KEEPALIVE
uint32_t* quicksort_copy(const uint32_t* data, uint32_t length) {
  if (!data || length == 0) return (uint32_t*)0;

  // Allocate output buffer in the WASM heap
  size_t bytes = (size_t)length * sizeof(uint32_t);
  uint32_t* out = (uint32_t*)malloc(bytes);
  if (!out) return (uint32_t*)0;

  // Copy and sort
  memcpy(out, data, bytes);
  quicksort_inplace_u32(out, length);
  return out;
}

// Helper to free buffers allocated by quicksort_copy.
EMSCRIPTEN_KEEPALIVE
void free_buffer(void* p) {
  free(p);
}

} // extern "C"
