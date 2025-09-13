// bitonic_sort_cuda.cu
// CUDA implementation of bitonic sort for distributed sorting
// Matches the WebGPU bitonic_sort.wgsl functionality

#include <cuda_runtime.h>
typedef unsigned int uint32_t;


// Kernel parameters structure matching the WebGPU uniform buffer
struct SortParams {
    uint32_t array_size;   // N (power-of-two padded)
    uint32_t stage;        // current k
    uint32_t substage;     // current j
    uint32_t ascending;    // 1 for ascending, 0 for descending
};

extern "C" __global__
void execute_task(
    const uint32_t array_size,    // uniforms[0]
    const uint32_t stage,         // uniforms[1]
    const uint32_t substage,      // uniforms[2]
    const uint32_t ascending,     // uniforms[3]
    uint32_t* __restrict__ data   // inputs[0] and outputs[0], size array_size
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= array_size) {
        return;
    }

    const uint32_t partner = i ^ substage;
    if (partner > i) {
        const uint32_t a = data[i];
        const uint32_t b = data[partner];

        // Determine sort direction for this comparison
        // The bit pattern determines if we're in an ascending or descending block
        const bool ascending_block = ((i & stage) == 0u);

        // Apply global sort direction
        const bool should_ascend = ascending_block == (ascending == 1u);

        // Swap if elements are out of order
        if ((a > b) == should_ascend) {
            data[i] = b;
            data[partner] = a;
        }
    }
}

// Alternative entry point that matches the WebGPU parameter structure
extern "C" __global__
void bitonic_sort_stage(
    uint32_t* __restrict__ data,
    const SortParams params
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= params.array_size) {
        return;
    }

    const uint32_t partner = i ^ params.substage;
    if (partner > i) {
        const uint32_t a = data[i];
        const uint32_t b = data[partner];

        // Determine sort direction for this comparison
        // The bit pattern determines if we're in an ascending or descending block
        const bool ascending_block = ((i & params.stage) == 0u);

        // Apply global sort direction
        const bool should_ascend = ascending_block == (params.ascending == 1u);

        // Swap if elements are out of order
        if ((a > b) == should_ascend) {
            data[i] = b;
            data[partner] = a;
        }
    }
}
