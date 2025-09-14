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
    const uint32_t* __restrict__ uniforms,  // uniforms[0]=array_size, uniforms[1]=stage, uniforms[2]=substage, uniforms[3]=ascending
    const uint32_t* __restrict__ inputs,    // inputs[0] = input data buffer
    uint32_t* __restrict__ outputs          // outputs[0] = output data buffer
) {
    // Extract uniform parameters (matching WebGPU uniform buffer structure)
    const uint32_t array_size = uniforms[0];
    const uint32_t stage = uniforms[1];
    const uint32_t substage = uniforms[2];
    const uint32_t ascending = uniforms[3];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= array_size) {
        return;
    }

    // Copy input data to output buffer (only if different buffers)
    if (inputs != outputs) {
        outputs[i] = inputs[i];
        // Synchronize to ensure all threads have copied their data
        __syncthreads();
    }

    // Bitonic sort comparison and swap (matching WebGPU logic exactly)
    const uint32_t partner = i ^ substage;
    if (partner > i) {
        const uint32_t a = outputs[i];
        const uint32_t b = outputs[partner];

        // Determine sort direction for this comparison
        // The bit pattern determines if we're in an ascending or descending block
        const bool ascending_block = ((i & stage) == 0u);

        // Apply global sort direction
        const bool should_ascend = ascending_block == (ascending == 1u);

        // Swap if elements are out of order (exactly like WebGPU)
        if ((a > b) == should_ascend) {
            outputs[i] = b;
            outputs[partner] = a;
        }
    }
}

// Alternative entry point that matches the WebGPU parameter structure exactly
extern "C" __global__
void bitonic_sort_stage(
    const uint32_t* __restrict__ uniforms,  // uniforms[0]=array_size, uniforms[1]=stage, uniforms[2]=substage, uniforms[3]=ascending
    const uint32_t* __restrict__ inputs,    // inputs[0] = input data buffer
    uint32_t* __restrict__ outputs          // outputs[0] = output data buffer
) {
    // Extract uniform parameters (matching WebGPU uniform buffer structure)
    const uint32_t array_size = uniforms[0];
    const uint32_t stage = uniforms[1];
    const uint32_t substage = uniforms[2];
    const uint32_t ascending = uniforms[3];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= array_size) {
        return;
    }

    // Copy input data to output buffer (only if different buffers)
    if (inputs != outputs) {
        outputs[i] = inputs[i];
        // Synchronize to ensure all threads have copied their data
        __syncthreads();
    }

    // Bitonic sort comparison and swap (matching WebGPU logic exactly)
    const uint32_t partner = i ^ substage;
    if (partner > i) {
        const uint32_t a = outputs[i];
        const uint32_t b = outputs[partner];

        // Determine sort direction for this comparison
        // The bit pattern determines if we're in an ascending or descending block
        const bool ascending_block = ((i & stage) == 0u);

        // Apply global sort direction
        const bool should_ascend = ascending_block == (ascending == 1u);

        // Swap if elements are out of order (exactly like WebGPU)
        if ((a > b) == should_ascend) {
            outputs[i] = b;
            outputs[partner] = a;
        }
    }
}
