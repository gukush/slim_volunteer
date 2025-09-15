// bitonic_sort_batch.cu
// CUDA implementation of batched bitonic sort for distributed sorting
// Processes multiple sort stages in a single kernel launch to reduce round trips

#include <cuda_runtime.h>
typedef unsigned int uint32_t;

// Structure to hold stage parameters
struct SortStage {
    uint32_t stage;        // current k
    uint32_t substage;     // current j
};

extern "C" __global__
void execute_task_batch(
    const uint32_t array_size,           // N (power-of-two padded)
    const uint32_t num_stages,           // Number of stages to process
    const uint32_t ascending,            // 1 for ascending, 0 for descending
    const SortStage* __restrict__ stages, // Array of stage parameters
    uint32_t* __restrict__ data          // data buffer (in-place)
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= array_size) {
        return;
    }

    // Process all stages in sequence
    for (uint32_t stage_idx = 0; stage_idx < num_stages; stage_idx++) {
        const uint32_t stage = stages[stage_idx].stage;
        const uint32_t substage = stages[stage_idx].substage;

        // Bitonic sort comparison and swap for this stage
        const uint32_t partner = i ^ substage;
        if (partner > i) {
            const uint32_t a = data[i];
            const uint32_t b = data[partner];

            // Determine sort direction for this comparison
            const bool ascending_block = ((i & stage) == 0u);
            const bool should_ascend = ascending_block == (ascending == 1u);

            // Swap if elements are out of order
            if ((a > b) == should_ascend) {
                data[i] = b;
                data[partner] = a;
            }
        }

        // Synchronize threads between stages to ensure all threads complete current stage
        __syncthreads();
    }
}

// Alternative entry point for single-stage processing (backward compatibility)
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
