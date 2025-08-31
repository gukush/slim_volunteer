struct Dims { rows:u32, K:u32, cols:u32,  _pad:u32  };

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> dims : Dims;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let r = gid.y;
  let c = gid.x;
  if (r >= dims.rows || c >= dims.cols) { return; }
  var acc: f32 = 0.0;
  for (var k:u32 = 0u; k < dims.K; k = k + 1u) {
    let aVal = A[r * dims.K + k];
    let bVal = B[k * dims.cols + c];
    acc = acc + aVal * bVal;
  }
  C[r * dims.cols + c] = acc;
}
