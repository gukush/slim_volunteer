// block_matmul_shared.wgsl
// Same bindings and entrypoint as your original kernel.

struct Dims { rows:u32, K:u32, cols:u32, _pad:u32 };

@group(0) @binding(0) var<storage, read>        A : array<f32>;
@group(0) @binding(1) var<storage, read>        B : array<f32>;
@group(0) @binding(2) var<storage, read_write>  C : array<f32>;
@group(0) @binding(3) var<uniform>              dims : Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Constants local to the shader
  const TILE : u32 = 16u;

  let lx : u32 = gid.x % TILE;
  let ly : u32 = gid.y % TILE;

  let r : u32 = gid.y;
  let c : u32 = gid.x;

  let active = (r < dims.rows) && (c < dims.cols);

  var<workgroup> Asub : array<array<f32, 16>, 16>;
  var<workgroup> Bsub : array<array<f32, 16>, 16>;

  var acc : f32 = 0.0;

  let tiles = (dims.K + TILE - 1u) / TILE;

  // Sweep K dimension in tiles
  for (var t:u32 = 0u; t < tiles; t = t + 1u) {
    let k0 = t * TILE;

    // Load one A and one B element per thread into shared memory (zero-pad when OOB)
    var aElem : f32 = 0.0;
    if (active && (k0 + lx) < dims.K) {
      aElem = A[r * dims.K + (k0 + lx)];
    }
    Asub[ly][lx] = aElem;

    var bElem : f32 = 0.0;
    if (active && (k0 + ly) < dims.K) {
      bElem = B[(k0 + ly) * dims.cols + c];
    }
    Bsub[ly][lx] = bElem;

    // Make sure all threads have finished writing the tile
    workgroupBarrier();

    // Compute partial dot product for this tile
    for (var k:u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + Asub[ly][k] * Bsub[k][lx];
    }

    // Avoid RAW hazards before the next tile load
    workgroupBarrier();
  }

  // Store the result if this thread maps to a valid output element
  if (active) {
    C[r * dims.cols + c] = acc;
  }
}
