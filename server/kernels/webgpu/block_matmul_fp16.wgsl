// block_matmul_f16.wgsl
// Tiled/shared-memory matmul using f16. Requires `enable f16;` and the device must
// be created with the "shader-f16" feature enabled by the application.
// This example accumulates in f16. If you prefer more numeric stability, accumulate in f32.

enable f16;

struct Dims { rows: u32, K: u32, cols: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>        A : array<f16>;
@group(0) @binding(1) var<storage, read>        B : array<f16>;
@group(0) @binding(2) var<storage, read_write>  C : array<f16>;
@group(0) @binding(3) var<uniform>              dims : Dims;

const TILE : u32 = 16u;

// Workgroup-shared tiles
var<workgroup> Asub : array<array<f16, TILE>, TILE>;
var<workgroup> Bsub : array<array<f16, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE, 1)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>
) {
  let r : u32 = gid.y;
  let c : u32 = gid.x;

  let lx : u32 = lid.x;
  let ly : u32 = lid.y;

  let inRow : bool = (r < dims.rows);
  let inCol : bool = (c < dims.cols);
  let inOut : bool = inRow && inCol;

  var acc : f16 = 0.0h;
  let tiles : u32 = (dims.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < tiles; t = t + 1u) {
    let k0 : u32 = t * TILE;

    var aElem : f16 = 0.0h;
    if (inRow && (k0 + lx) < dims.K) {
      aElem = A[r * dims.K + (k0 + lx)];
    }
    Asub[ly][lx] = aElem;

    var bElem : f16 = 0.0h;
    if (inCol && (k0 + ly) < dims.K) {
      bElem = B[(k0 + ly) * dims.cols + c];
    }
    Bsub[ly][lx] = bElem;

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + Asub[ly][k] * Bsub[k][lx];
    }

    workgroupBarrier();
  }

  if (inOut) {
    C[r * dims.cols + c] = acc;
  }
}
