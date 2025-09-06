// kernels/multi_head_attention.wgsl
// Single-head scaled dot-product attention (tiled, uniform-control flow)

struct Dims {
  seq_len: u32,
  d_k: u32,
  d_v: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read>        Q: array<f32>;            // [seq_len, d_k]
@group(0) @binding(1) var<storage, read>        K: array<f32>;            // [seq_len, d_k]
@group(0) @binding(2) var<storage, read>        V: array<f32>;            // [seq_len, d_v]
@group(0) @binding(3) var<storage, read_write>  output: array<f32>;       // [seq_len, d_v]
@group(0) @binding(4) var<uniform>              dims: Dims;

// Workgroup scratch (module scope)
var<workgroup> scratch : array<f32, 256u>;
var<workgroup> wg_max  : f32;
var<workgroup> wg_sum  : f32;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>
) {
  let seq_len = dims.seq_len;
  let d_k     = dims.d_k;
  let d_v     = dims.d_v;

  // JS dispatcher: X = seq_len, Y = d_v
  let out_row = gid.x; // sequence index
  let out_col = gid.y; // feature index

  // IMPORTANT: No early return; keep barriers uniform.
  let is_active = (out_row < seq_len) && (out_col < d_v);

  let scale    : f32 = 1.0 / sqrt(f32(d_k));
  let lane     : u32 = lid.y * 16u + lid.x; // 0..255

  // -------------------------
  // Pass 1: global max(score)
  // -------------------------
  if (lane == 0u) { wg_max = -1e30; }
  workgroupBarrier();

  var base : u32 = 0u;
  loop {
    if (base >= seq_len) { break; }
    let k_idx : u32 = base + lane;

    // Default value for inactive/out-of-range lanes
    var s : f32 = -1e30;

    if (is_active && k_idx < seq_len) {
      var acc : f32 = 0.0;
      var d   : u32 = 0u;
      loop {
        if (d >= d_k) { break; }
        let qv = Q[out_row * d_k + d];
        let kv = K[k_idx  * d_k + d];
        acc = acc + qv * kv;
        d = d + 1u;
      }
      s = acc * scale;
    }

    scratch[lane] = s;
    workgroupBarrier();

    if (lane == 0u) {
      let tile_n = min(256u, seq_len - base);
      var tmax : f32 = -1e30;
      var i    : u32 = 0u;
      loop {
        if (i >= tile_n) { break; }
        tmax = max(tmax, scratch[i]);
        i = i + 1u;
      }
      wg_max = max(wg_max, tmax);
    }
    workgroupBarrier();

    base = base + 256u;
  }

  // Broadcast global max
  if (lane == 0u) { scratch[0] = wg_max; }
  workgroupBarrier();
  let gmax : f32 = scratch[0];

  // -------------------------
  // Pass 2: global sum(exp(..))
  // -------------------------
  if (lane == 0u) { wg_sum = 0.0; }
  workgroupBarrier();

  base = 0u;
  loop {
    if (base >= seq_len) { break; }
    let k_idx : u32 = base + lane;

    var ex : f32 = 0.0;
    if (is_active && k_idx < seq_len) {
      var acc : f32 = 0.0;
      var d   : u32 = 0u;
      loop {
        if (d >= d_k) { break; }
        let qv = Q[out_row * d_k + d];
        let kv = K[k_idx  * d_k + d];
        acc = acc + qv * kv;
        d = d + 1u;
      }
      ex = exp(acc * scale - gmax);
    }

    scratch[lane] = ex;
    workgroupBarrier();

    if (lane == 0u) {
      let tile_n = min(256u, seq_len - base);
      var tsum : f32 = 0.0;
      var i    : u32 = 0u;
      loop {
        if (i >= tile_n) { break; }
        tsum = tsum + scratch[i];
        i = i + 1u;
      }
      wg_sum = wg_sum + tsum;
    }
    workgroupBarrier();

    base = base + 256u;
  }

  // Broadcast denom with epsilon to avoid /0 (only used by active lanes)
  if (lane == 0u) { scratch[0] = max(wg_sum, 1e-20); }
  workgroupBarrier();
  let denom : f32 = scratch[0];

  // -------------------------
  // Pass 3: weighted sum with V
  // -------------------------
  var partial : f32 = 0.0;

  base = 0u;
  loop {
    if (base >= seq_len) { break; }
    let k_idx : u32 = base + lane;

    if (is_active && k_idx < seq_len) {
      var acc : f32 = 0.0;
      var d   : u32 = 0u;
      loop {
        if (d >= d_k) { break; }
        let qv = Q[out_row * d_k + d];
        let kv = K[k_idx  * d_k + d];
        acc = acc + qv * kv;
        d = d + 1u;
      }
      let prob = exp(acc * scale - gmax) / denom;
      let vv   = V[k_idx * d_v + out_col];
      partial  = partial + prob * vv;
    }

    base = base + 256u;
  }

  // Reduce 256 lanes to one
  scratch[lane] = partial;
  workgroupBarrier();

  var stride : u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      scratch[lane] = scratch[lane] + scratch[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  // Only active lanes write output
  if (lane == 0u && is_active) {
    output[out_row * d_v + out_col] = scratch[0];
  }
}
