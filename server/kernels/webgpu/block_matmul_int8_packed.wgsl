// block_matmul_int8_packed.wgsl
// Uses WGSL DP4a-style intrinsics to accelerate int8 matmul.
// Requires: WGSL language feature `packed_4x8_integer_dot_product`
// Layout: A_packed is rows x (K/4), row-major, each u32 packs 4 consecutive A[r, k..k+3] (signed int8).
//         B_packed is (K/4) x cols, row-major, each u32 packs 4 consecutive B[k..k+3, c] (signed int8).
// Accumulator: i32, output C is i32.
// If K % 4 != 0, pad with zeros in packing.
//
// Bindings match the original order but types are changed to u32/int32 as noted.

requires packed_4x8_integer_dot_product;

struct Dims { rows:u32, K:u32, cols:u32, groupsK:u32 };

@group(0) @binding(0) var<storage, read>        A_packed : array<u32>;
@group(0) @binding(1) var<storage, read>        B_packed : array<u32>;
@group(0) @binding(2) var<storage, read_write>  C        : array<i32>;
@group(0) @binding(3) var<uniform>              dims     : Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.y;
  let c = gid.x;

  if (r >= dims.rows || c >= dims.cols) {
    return;
  }

  // For each group of 4 along K, take one u32 from A (row r) and B (col c)
  var acc: i32 = 0;
  let groups = dims.groupsK; // == (K + 3) / 4 on the host when packing
  let rowBase = r * groups;
  // B is indexed by (g, c)
  for (var g: u32 = 0u; g < groups; g = g + 1u) {
    let a4: u32 = A_packed[rowBase + g];
    let b4: u32 = B_packed[g * dims.cols + c];
    // signed 8-bit dot -> i32
    acc = acc + dot4I8Packed(a4, b4);
  }

  C[r * dims.cols + c] = acc;
}
