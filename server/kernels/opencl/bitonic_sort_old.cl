// bitonic_sort.cu
// One sub-stage (k/j) of a bitonic sort on uint32 data.
// Mirrors WGSL bindings: arg0 = data buffer (group0/binding0), arg1 = Params (group0/binding1).

#include <stdint.h>

struct Params {
    uint32_t array_size; // N (power-of-two padded)
    uint32_t stage;      // current k
    uint32_t substage;   // current j
    uint32_t ascending;  // 1 for ascending, 0 for descending
};

extern "C"
__global__ __launch_bounds__(256)
void bitonic_step(uint32_t* __restrict__ data,
                  const Params* __restrict__ params)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t N = params->array_size;

    if (i >= N) return;

    uint32_t partner = i ^ params->substage;

    if (partner > i && partner < N) {
        uint32_t a = data[i];
        uint32_t b = data[partner];

        bool ascending_block = ((i & params->stage) == 0u);
        bool should_ascend   = (ascending_block == (params->ascending == 1u));

        if ((a > b) == should_ascend) {
            data[i]       = b;
            data[partner] = a;
        }
    }
}
