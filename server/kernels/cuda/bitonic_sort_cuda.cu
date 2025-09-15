// bitonic_sort_cuda.cu
// CUDA implementation of bitonic sort for distributed sorting
// Matches the WebGPU bitonic_sort.wgsl functionality

#include <cuda_runtime.h>
typedef unsigned int uint32_t;

extern "C" __global__
void execute_task(
    const uint32_t array_size,    // N (power-of-two padded)
    const uint32_t stage,         // current k
    const uint32_t substage,      // current j
    const uint32_t ascending,     // 1 for ascending, 0 for descending
    uint32_t* __restrict__ data   // data buffer (in-place)
) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= array_size) {
        return;
    }

    // Bitonic sort comparison and swap (matching WebGPU logic exactly)
    const uint32_t partner = i ^ substage;
    if (partner > i) {
        const uint32_t a = data[i];
        const uint32_t b = data[partner];

        // Determine sort direction for this comparison
        // The bit pattern determines if we're in an ascending or descending block
        const bool ascending_block = ((i & stage) == 0u);

        // Apply global sort direction
        const bool should_ascend = ascending_block == (ascending == 1u);

        // Swap if elements are out of order (exactly like WebGPU)
        if ((a > b) == should_ascend) {
            data[i] = b;
            data[partner] = a;
        }
    }
}

// Alternative entry point that matches the WebGPU parameter structure exactly
extern "C" __global__
void bitonic_sort_stage(
    const uint32_t array_size,    // N (power-of-two padded)
    const uint32_t stage,         // current k
    const uint32_t substage,      // current j
    const uint32_t ascending,     // 1 for ascending, 0 for descending
    uint32_t* __restrict__ data   // data buffer (in-place)
) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= array_size) {
        return;
    }

    // Bitonic sort comparison and swap (matching WebGPU logic exactly)
    const uint32_t partner = i ^ substage;
    if (partner > i) {
        const uint32_t a = data[i];
        const uint32_t b = data[partner];

        // Determine sort direction for this comparison
        // The bit pattern determines if we're in an ascending or descending block
        const bool ascending_block = ((i & stage) == 0u);

        // Apply global sort direction
        const bool should_ascend = ascending_block == (ascending == 1u);

        // Swap if elements are out of order (exactly like WebGPU)
        if ((a > b) == should_ascend) {
            data[i] = b;
            data[partner] = a;
        }
    }
}
