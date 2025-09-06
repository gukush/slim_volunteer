// bitonic_sort.cl
// One sub-stage (k/j) of a bitonic sort on uint32 data.
// Mirrors WGSL bindings: arg0 = data buffer (group0/binding0), arg1 = Params (group0/binding1).

typedef struct {
    uint array_size; // N (power-of-two padded)
    uint stage;      // current k
    uint substage;   // current j
    uint ascending;  // 1 for ascending, 0 for descending
} Params;

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void bitonic_step(__global uint* data,
                  __constant Params* params)
{
    uint i = get_global_id(0);
    uint N = params->array_size;

    if (i >= N) return;

    uint partner = i ^ params->substage;

    // Only one thread in each pair performs the compare/swap.
    if (partner > i && partner < N) {
        uint a = data[i];
        uint b = data[partner];

        // Decide direction: ((i & k) == 0) XOR (ascending == 0) flips order.
        uint ascending_block = ((i & params->stage) == 0u) ? 1u : 0u;
        uint want_asc = (ascending_block == ((params->ascending == 1u) ? 1u : 0u)) ? 1u : 0u;

        // If (a > b) and we want ascending, or (a < b) and we want descending → swap.
        if ( ((a > b) ? 1u : 0u) == want_asc ) {
            data[i] = b;
            data[partner] = a;
        }
    }
}
