// block_matmul_int8_packed_tiled.wgsl
// Tiled/shared-memory matmul for int8-packed data using WGSL DP4a-style intrinsics.
// Requires: WGSL language feature `packed_4x8_integer_dot_product`
// Layouts (same as your original):
//   A_packed: rows x groupsK (row-major), each u32 packs 4 consecutive A[r, k..k+3] (signed int8).
//   B_packed: groupsK x cols (row-major), each u32 packs 4 consecutive B[k..k+3, c] (signed int8).
// Accumulator: i32, output C is i32.

requires packed_4x8_integer_dot_product;

struct Dims { rows: u32, K: u32, cols: u32, groupsK: u32 };

@group(0) @binding(0) var<storage, read>       A_packed : array<u32>;
@group(0) @binding(1) var<storage, read>       B_packed : array<u32>;
@group(0) @binding(2) var<storage, read_write> C        : array<i32>;
@group(0) @binding(3) var<uniform>             dims     : Dims;

// Tile size (workgroup size = TILE x TILE)
const TILE : u32 = 16u;

// Workgroup-shared tiles (store packed u32 entries)
var<workgroup> Asub : array<array<u32, TILE>, TILE>;
var<workgroup> Bsub : array<array<u32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE, 1)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>
) {
  let r : u32 = gid.y; // row index of output
  let c : u32 = gid.x; // col index of output

  let lx : u32 = lid.x; // local x (0..TILE-1)
  let ly : u32 = lid.y; // local y (0..TILE-1)

  let inRow : bool = (r < dims.rows);
  let inCol : bool = (c < dims.cols);
  let inOut : bool = inRow && inCol;

  var acc : i32 = 0;

  let groups : u32 = dims.groupsK; // number of packed groups along K (== ceil(K/4))
  let tileGroups : u32 = (groups + TILE - 1u) / TILE;

  // Precompute row base for A_packed to speed indexing (safe even if out of bounds; checks below)
  let rowBase : u32 = r * groups;

  for (var t: u32 = 0u; t < tileGroups; t = t + 1u) {
    let g0 : u32 = t * TILE; // starting group index for this tile

    // Load Asub[ly][lx] <- A_packed[rowBase + g0 + lx] if valid, else 0u
    var aPacked : u32 = 0u;
    if (inRow) {
      let gIndexA : u32 = g0 + lx;
      if (gIndexA < groups) {
        // safe to index A_packed as (r * groups + gIndexA)
        aPacked = A_packed[rowBase + gIndexA];
      }
    }
    Asub[ly][lx] = aPacked;

    // Load Bsub[ly][lx] <- B_packed[(g0 + ly) * dims.cols + c] if valid, else 0u
    var bPacked : u32 = 0u;
    if (inCol) {
      let gIndexB : u32 = g0 + ly;
      if (gIndexB < groups) {
        // B_packed indexed by group * cols + c
        bPacked = B_packed[gIndexB * dims.cols + c];
      }
    }
    Bsub[ly][lx] = bPacked;

    // Wait for all loads to finish
    workgroupBarrier();

    // Multiply-accumulate across the TILE of packed groups
    // Each Asub[ly][k] and Bsub[k][lx] are u32 packing 4 signed int8 values.
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      // When g0 + k >= groups those entries were loaded as 0u, so it's safe to call intrinsic.
      acc = acc + dot4I8Packed(Asub[ly][k], Bsub[k][lx]);
    }

    // Ensure all threads finished reading shared arrays before next tile reuse
    workgroupBarrier();
  }

  if (inOut) {
    C[r * dims.cols + c] = acc;
  }
}
