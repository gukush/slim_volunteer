// Simplified Bitonic Sort WebGPU Compute Shader
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

struct Params {
    array_size: u32,
    stage: u32,
    substage: u32, 
    ascending: u32,
}

@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let array_size = params.array_size;
    
    // Calculate the comparison distance for this substage
    let substage_size = 1u << params.substage;
    let distance = substage_size / 2u;
    
    if (distance == 0u) {
        return;
    }
    
    // Calculate which group this thread belongs to
    let group_size = substage_size;
    let group_id = thread_id / group_size;
    let local_id = thread_id % group_size;
    
    // Only first half of each group does comparisons
    if (local_id >= distance) {
        return;
    }
    
    // Calculate the two indices to compare
    let base_index = group_id * group_size;
    let i = base_index + local_id;
    let j = base_index + local_id + distance;
    
    // Bounds check
    if (j >= array_size) {
        return;
    }
    
    // Determine sort direction for this group
    let stage_size = 1u << (params.stage + 1u);
    let stage_group_id = group_id / (stage_size / group_size);
    var sort_ascending = (stage_group_id % 2u) == 0u;
    
    // Apply global sort direction
    if (params.ascending == 0u) {
        sort_ascending = !sort_ascending;
    }
    
    // Get the values to compare
    let val_i = data[i];
    let val_j = data[j];
    
    // Perform comparison and swap if needed
    let should_swap = (val_i > val_j) == sort_ascending;
    if (should_swap) {
        data[i] = val_j;
        data[j] = val_i;
    }
}
