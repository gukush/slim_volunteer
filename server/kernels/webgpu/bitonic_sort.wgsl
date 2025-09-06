@group(0) @binding(0) var<storage, read_write> data: array<u32>;

struct Params {
  array_size: u32,   // N (power-of-two padded)
  stage: u32,        // current k
  substage: u32,     // current j
  ascending: u32,    // 1 for ascending, 0 for descending
}

@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.array_size) {
    return;
  }

  let partner = i ^ params.substage;
  if (partner > i) {
    let a = data[i];
    let b = data[partner];

    // Determine sort direction for this comparison
    // The bit pattern determines if we're in an ascending or descending block
    let ascending_block = ((i & params.stage) == 0u);

    // Apply global sort direction
    let should_ascend = ascending_block == (params.ascending == 1u);

    // Swap if elements are out of order
    if ((a > b) == should_ascend) {
      data[i] = b;
      data[partner] = a;
    }
  }
}