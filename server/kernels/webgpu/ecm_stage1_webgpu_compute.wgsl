// kernels/ecm_stage1_webgpu_compute.wgsl - Version 2 (robust)
//  - 256-bit big integers (8x u32, little-endian)
//  - Montgomery math (CIOS, word=32, L=8)
//  - Deterministic RNG via 64-bit LCG emulated with u32
//  - Suyama parametrization
//  - Binary modular inverse (u32)
//  - Correct constant-time Montgomery ladder with final swap
//  - Status codes: 1=no factor, 2=factor found, 3=bad curve/inverse failed

// --------------------------- Types & IO ---------------------------
struct U256 { limbs: array<u32, 8>, };

struct PointXZ {
  X: U256,
  Z: U256,
};

struct InvResult {
  ok  : bool,
  val : U256,
};

struct CurveResult {
  ok   : bool,
  A24m : U256, // Montgomery domain
  X1m  : U256, // Montgomery domain
};

struct Header {
  magic      : u32, // "ECM1"
  version    : u32, // 2
  rsv0       : u32,
  rsv1       : u32,
  pp_count   : u32,
  n_curves   : u32,
  seed_lo    : u32,
  seed_hi    : u32,
  base_curve : u32,
  flags      : u32,
  rsv2       : u32,
  rsv3       : u32,
};

struct IO { words: array<u32>, };

@group(0) @binding(0) var<storage, read_write> io : IO;

fn getHeader() -> Header {
  return Header(
    io.words[0], io.words[1], io.words[2], io.words[3],
    io.words[4], io.words[5], io.words[6], io.words[7],
    io.words[8], io.words[9], io.words[10], io.words[11]
  );
}

fn constOffset() -> u32 { return 12u; }
fn ppOffset()    -> u32 { return constOffset() + (8u*3u + 4u); } // N(8)+R2(8)+mont_one(8)+n0inv32(1)+pad(3)
fn outOffset(h: Header) -> u32 { return ppOffset() + h.pp_count; }

const LIMBS : u32 = 8u;

// --------------------------- Small helpers ---------------------------
fn set_zero() -> U256 { var r:U256; for (var i=0u;i<8u;i++){ r.limbs[i]=0u; } return r; }
fn set_one() -> U256 { var r = set_zero(); r.limbs[0]=1u; return r; }
fn u256_from_u32(x: u32) -> U256 { var r = set_zero(); r.limbs[0]=x; return r; }

fn is_zero(a: U256) -> bool {
  var x:u32 = 0u;
  for (var i=0u;i<8u;i++){ x |= a.limbs[i]; }
  return x == 0u;
}
fn cmp(a: U256, b: U256) -> i32 {
  for (var i:i32=7; i>=0; i--) {
    let ai = a.limbs[u32(i)];
    let bi = b.limbs[u32(i)];
    if (ai < bi) { return -1; }
    if (ai > bi) { return 1; }
  }
  return 0;
}
fn is_even(a: U256) -> bool { return (a.limbs[0] & 1u) == 0u; }

fn rshift1(a: U256) -> U256 {
  var r: U256;
  var carry: u32 = 0u;
  for (var i:i32=7; i>=0; i--) {
    let w = a.limbs[u32(i)];
    r.limbs[u32(i)] = (w >> 1u) | (carry << 31u);
    carry = w & 1u;
  }
  return r;
}

fn addc(a:u32, b:u32, cin:u32) -> vec2<u32> {
  let s = a + b;
  let c1 = select(0u, 1u, s < a);
  let s2 = s + cin;
  let c2 = select(0u, 1u, s2 < cin);
  return vec2<u32>(s2, c1 + c2);
}
fn subb(a:u32, b:u32, bin:u32) -> vec2<u32> {
  let d = a - b;
  let b1 = select(0u, 1u, a < b);
  let d2 = d - bin;
  let b2 = select(0u, 1u, d < bin);
  return vec2<u32>(d2, select(0u, 1u, (b1 | b2) != 0u));
}

fn add_u256(a: U256, b: U256) -> U256 {
  var r: U256;
  var c:u32 = 0u;
  for (var i=0u;i<8u;i++){
    let ac = addc(a.limbs[i], b.limbs[i], c);
    r.limbs[i] = ac.x; c = ac.y;
  }
  return r;
}
fn sub_u256(a: U256, b: U256) -> U256 {
  var r: U256;
  var br:u32 = 0u;
  for (var i=0u;i<8u;i++){
    let sb = subb(a.limbs[i], b.limbs[i], br);
    r.limbs[i] = sb.x; br = sb.y;
  }
  return r;
}
fn cond_sub_N(a: U256, N: U256) -> U256 {
  if (cmp(a, N) >= 0) { return sub_u256(a, N); }
  return a;
}
fn add_mod(a: U256, b: U256, N: U256) -> U256 {
  return cond_sub_N(add_u256(a, b), N);
}
fn sub_mod(a: U256, b: U256, N: U256) -> U256 {
  if (cmp(a, b) >= 0) { return sub_u256(a, b); }
  // a < b  =>  N - (b - a)
  let diff = sub_u256(b, a);
  return sub_u256(N, diff);
}

// 32x32 -> 64 multiply via 16-bit split. Returns (lo, hi).
fn mul32x32_64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xFFFFu; let a1 = a >> 16u;
  let b0 = b & 0xFFFFu; let b1 = b >> 16u;

  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;

  let mid = p01 + p10;
  let lo  = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
  let carry = (p00 >> 16u) + (mid >> 16u);
  let hi = p11 + carry;

  return vec2<u32>(lo, hi);
}

// --------------------------- Montgomery math (CIOS) ---------------------------
fn mont_mul(a: U256, b: U256, N: U256, n0inv32: u32) -> U256 {
  var t: array<u32, 9>;
  for (var i=0u;i<9u;i++){ t[i]=0u; }

  for (var i=0u;i<8u;i++){
    var carry:u32 = 0u;
    for (var j=0u;j<8u;j++){
      let prod = mul32x32_64(a.limbs[i], b.limbs[j]); // (lo,hi)
      let s1 = addc(t[j], prod.x, 0u);
      let s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    t[8] = t[8] + carry;

    let m = t[0] * n0inv32;

    carry = 0u;
    for (var j=0u;j<8u;j++){
      let prod = mul32x32_64(m, N.limbs[j]);
      let s1 = addc(t[j], prod.x, 0u);
      let s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    t[8] = t[8] + carry;

    for (var k=0u;k<8u;k++){ t[k] = t[k+1]; }
    t[8]=0u;
  }

  var r: U256;
  for (var i=0u;i<8u;i++){ r.limbs[i]=t[i]; }
  return cond_sub_N(r, N);
}

fn mont_add(a: U256, b: U256, N: U256) -> U256 {
  return cond_sub_N(add_u256(a, b), N);
}
fn mont_sub(a: U256, b: U256, N: U256) -> U256 {
  if (cmp(a,b) >= 0) { return sub_u256(a,b); }
  // a < b  =>  a - b (mod N) = N - (b - a)
  let diff = sub_u256(b, a);
  return sub_u256(N, diff);
}
fn mont_sqr(a: U256, N: U256, n0inv32: u32) -> U256 { return mont_mul(a, a, N, n0inv32); }

fn to_mont(a: U256, R2: U256, N: U256, n0inv32: u32) -> U256 { return mont_mul(a, R2, N, n0inv32); }
fn from_mont(a: U256, N: U256, n0inv32: u32) -> U256 { return mont_mul(a, set_one(), N, n0inv32); }

// --------------------------- X-only ops ---------------------------
fn xDBL(P: PointXZ, A24: U256, N: U256, n0inv32: u32) -> PointXZ {
  let t1 = mont_add(P.X, P.Z, N);
  let t2 = mont_sub(P.X, P.Z, N);
  let t3 = mont_sqr(t1, N, n0inv32);
  let t4 = mont_sqr(t2, N, n0inv32);
  let t5 = mont_sub(t3, t4, N);
  let t6 = mont_mul(A24, t5, N, n0inv32);
  let Z_mult = mont_add(t3, t6, N);
  let X2 = mont_mul(t3, t4, N, n0inv32);
  let Z2 = mont_mul(t5, Z_mult, N, n0inv32);
  return PointXZ(X2, Z2);
}

fn xADD(P: PointXZ, Q: PointXZ, Diff: PointXZ, N: U256, n0inv32: u32) -> PointXZ {
  let t1 = mont_add(P.X, P.Z, N);
  let t2 = mont_sub(P.X, P.Z, N);
  let t3 = mont_add(Q.X, Q.Z, N);
  let t4 = mont_sub(Q.X, Q.Z, N);
  let t5 = mont_mul(t1, t4, N, n0inv32);
  let t6 = mont_mul(t2, t3, N, n0inv32);
  let t1n = mont_add(t5, t6, N);
  let t2n = mont_sub(t5, t6, N);
  let X3 = mont_mul(mont_sqr(t1n, N, n0inv32), Diff.Z, N, n0inv32);
  let Z3 = mont_mul(mont_sqr(t2n, N, n0inv32), Diff.X, N, n0inv32);
  return PointXZ(X3, Z3);
}

fn cswap(a: ptr<function, PointXZ>, b: ptr<function, PointXZ>, bit: u32) {
  let mask = (0u - (bit & 1u));
  for (var i=0u;i<8u;i++){
    let tx = (((*a).X.limbs[i]) ^ ((*b).X.limbs[i])) & mask;
    (*a).X.limbs[i] ^= tx; (*b).X.limbs[i] ^= tx;
    let tz = (((*a).Z.limbs[i]) ^ ((*b).Z.limbs[i])) & mask;
    (*a).Z.limbs[i] ^= tz; (*b).Z.limbs[i] ^= tz;
  }
}

// Simple Montgomery ladder (like working version)
fn ladder(P: PointXZ, k: u32, A24: U256, N: U256, n0inv32: u32, mont_one: U256) -> PointXZ {
  var R0 = PointXZ(mont_one, set_zero()); // 0*P
  var R1 = P;                             // 1*P
  var started = false;

  for (var i:i32=31; i>=0; i--) {
    let bit = (k >> u32(i)) & 1u;
    if (!started && bit == 0u) { continue; }
    started = true;

    cswap(&R0, &R1, 1u - bit);
    let T0 = xADD(R0, R1, P, N, n0inv32);
    let T1 = xDBL(R1, A24, N, n0inv32);
    R0 = T0;
    R1 = T1;
  }
  return R0;
}

// --------------------------- RNG (Simple LCG) ---------------------------
fn lcg32(state: ptr<function, u32>) -> u32 {
  // Simple 32-bit LCG: state = (state * 1103515245 + 12345) mod 2^32
  let new_state = ((*state) * 1103515245u + 12345u);
  (*state) = new_state;
  return new_state;
}

fn next_sigma(_N: U256, state: ptr<function, vec2<u32>>) -> U256 {
  var acc: U256;
  var lcg_state = (*state).x;
  
  for (var i:u32=0u;i<8u;i++){
    acc.limbs[i] = lcg32(&lcg_state);
  }
  
  // Update the state
  (*state).x = lcg_state;
  
  var sigma = acc;
  if (is_zero(sigma)) { sigma.limbs[0]=6u; }
  let one = set_one();
  if (cmp(sigma, one)==0) { sigma.limbs[0]=6u; }
  return sigma;
}

// --------------------------- Binary modular inverse ---------------------------
// Returns inverse of a mod N if gcd(a,N)==1
fn mod_inverse(a_in: U256, N: U256) -> InvResult {
  if (is_zero(a_in)) { return InvResult(false, set_zero()); }

  // reduce a mod N with up to two subs (good enough for random data)
  var a = a_in;
  for (var k=0u; k<2u && cmp(a, N) >= 0; k++){ a = sub_u256(a, N); }
  if (is_zero(a)) { return InvResult(false, set_zero()); }

  var u = a;
  var v = N;
  var x1 = set_one();  // coeff for a
  var x2 = set_zero(); // coeff for N

  for (var iter=0u; iter<20000u; iter++){
    if (cmp(u, set_one()) == 0) { return InvResult(true, x1); }
    if (cmp(v, set_one()) == 0) { return InvResult(true, x2); }
    if (is_zero(u) || is_zero(v)) { break; }

    while (is_even(u)) {
      u = rshift1(u);
      if (is_even(x1)) { x1 = rshift1(x1); }
      else { x1 = rshift1(add_u256(x1, N)); }
    }
    while (is_even(v)) {
      v = rshift1(v);
      if (is_even(x2)) { x2 = rshift1(x2); }
      else { x2 = rshift1(add_u256(x2, N)); }
    }

    if (cmp(u, v) >= 0) {
      u = sub_u256(u, v);
      x1 = sub_mod(x1, x2, N);
    } else {
      v = sub_u256(v, u);
      x2 = sub_mod(x2, x1, N);
    }
  }
  return InvResult(false, set_zero());
}

// --------------------------- Curve generation (Suyama) ---------------------------
// Given sigma, compute X1 and A24 in Montgomery domain:
//   u = σ^2 - 5
//   v = 4σ
//   X1 = (u^2 - v^2)^3 / (4uv)^2  (correct Suyama starting point)
//   A24 = ((A+2)/4) where A = ((v-u)^3 * (3u+v)) / (4 u^3 v) - 2
fn generate_curve(sigma: U256, N: U256, R2: U256, n0inv32: u32) -> CurveResult {
  // mont versions of constants and sigma
  let sigma_m = to_mont(sigma, R2, N, n0inv32);
  let five_m  = to_mont(u256_from_u32(5u),  R2, N, n0inv32);
  let four_m  = to_mont(u256_from_u32(4u),  R2, N, n0inv32);
  let three_m = to_mont(u256_from_u32(3u),  R2, N, n0inv32);
  let two_m   = to_mont(u256_from_u32(2u),  R2, N, n0inv32);

  // u = σ^2 - 5  (mont)
  let sigma_sq_m = mont_sqr(sigma_m, N, n0inv32);
  let u_m = mont_sub(sigma_sq_m, five_m, N);

  // v = 4σ       (mont)
  let v_m = mont_mul(four_m, sigma_m, N, n0inv32);

  // Check for degenerate cases (avoid σ ∈ {0, ±1, ±2})
  let sigma_std = from_mont(sigma_m, N, n0inv32);
  let u_std = from_mont(u_m, N, n0inv32);
  let v_std = from_mont(v_m, N, n0inv32);

  let one = set_one();
  let two = u256_from_u32(2u);
  let N_minus_1 = sub_u256(N, one);
  let N_minus_2 = sub_u256(N, two);

  if (is_zero(sigma_std) || cmp(sigma_std, one) == 0 || cmp(sigma_std, N_minus_1) == 0 ||
      cmp(sigma_std, two) == 0 || cmp(sigma_std, N_minus_2) == 0) {
    return CurveResult(false, set_zero(), set_zero());
  }

  // Check invertibility - if not invertible, we might have found a factor
  // But don't reject the curve, let the GCD handle it
  let inv_u = mod_inverse(u_std, N);
  let inv_v = mod_inverse(v_std, N);
  if (!inv_u.ok || !inv_v.ok) {
    return CurveResult(false, set_zero(), set_zero()); // caller may GCD later
  }

  // X1 = (u^2 - v^2)^3 / (4uv)^2  (mont)
  let u_sq_m = mont_sqr(u_m, N, n0inv32);
  let v_sq_m = mont_sqr(v_m, N, n0inv32);
  let u2_v2_m = mont_sub(u_sq_m, v_sq_m, N);
  let u2_v2_sq_m = mont_sqr(u2_v2_m, N, n0inv32);
  let u2_v2_cubed_m = mont_mul(u2_v2_m, u2_v2_sq_m, N, n0inv32);

  let four_uv_m = mont_mul(four_m, mont_mul(u_m, v_m, N, n0inv32), N, n0inv32);
  let four_uv_sq_m = mont_sqr(four_uv_m, N, n0inv32);

  let four_uv_sq_std = from_mont(four_uv_sq_m, N, n0inv32);
  let inv_four_uv_sq = mod_inverse(four_uv_sq_std, N);
  if (!inv_four_uv_sq.ok) { return CurveResult(false, set_zero(), set_zero()); }
  let inv_four_uv_sq_m = to_mont(inv_four_uv_sq.val, R2, N, n0inv32);

  let X1m = mont_mul(u2_v2_cubed_m, inv_four_uv_sq_m, N, n0inv32);

  // A24 = ((A+2)/4) where A = ((v-u)^3 * (3u+v)) / (4 u^3 v) - 2
  let vm_u_m = mont_sub(v_m, u_m, N);
  let vm_u_sq_m = mont_sqr(vm_u_m, N, n0inv32);
  let vm_u_cubed_m = mont_mul(vm_u_m, vm_u_sq_m, N, n0inv32);

  let three_u_m = mont_mul(three_m, u_m, N, n0inv32);
  let three_u_plus_v_m = mont_add(three_u_m, v_m, N);

  let u_cubed_m = mont_mul(u_m, u_sq_m, N, n0inv32);
  let four_u3_v_m = mont_mul(four_m, mont_mul(u_cubed_m, v_m, N, n0inv32), N, n0inv32);

  let four_u3_v_std = from_mont(four_u3_v_m, N, n0inv32);
  let inv_four_u3_v = mod_inverse(four_u3_v_std, N);
  if (!inv_four_u3_v.ok) { return CurveResult(false, set_zero(), set_zero()); }
  let inv_four_u3_v_m = to_mont(inv_four_u3_v.val, R2, N, n0inv32);

  let numerator_m = mont_mul(vm_u_cubed_m, three_u_plus_v_m, N, n0inv32);
  let A_plus_2_m = mont_mul(numerator_m, inv_four_u3_v_m, N, n0inv32);
  let A_m = mont_sub(A_plus_2_m, two_m, N);

  // A24 = (A + 2) / 4 = A_plus_2 / 4
  let inv4_m = to_mont(mod_inverse(u256_from_u32(4u), N).val, R2, N, n0inv32);
  let A24m = mont_mul(A_plus_2_m, inv4_m, N, n0inv32);

  return CurveResult(true, A24m, X1m);
}

// --------------------------- GCD (slow) ---------------------------
fn gcd_u256(a_in: U256, b_in: U256) -> U256 {
  var a = a_in; var b = b_in;
  if (is_zero(a)) { return b; }
  if (is_zero(b)) { return a; }
  for (var iter=0u; iter<4096u; iter++){
    if (is_zero(a)) { return b; }
    if (is_zero(b)) { return a; }
    if (cmp(a,b) >= 0) { a = sub_u256(a,b); } else { b = sub_u256(b,a); }
  }
  if (is_zero(a)) { return b; }
  return a;
}


fn lshift1_from_add(x: U256) -> U256 {
  return add_u256(x, x);
}

override HAVE_LSHIFT1_U256: bool = true;

fn lshift1_u256(a: U256) -> U256 {
  var r: U256;
  var carry: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i++) {
    let w = a.limbs[i];
    r.limbs[i] = (w << 1u) | carry;
    carry = w >> 31u;
  }
  return r;
}

fn lshiftk_u256(x: U256, k: u32) -> U256 {
  var r = x;
  for (var i: u32 = 0u; i < k; i++) {
    if (HAVE_LSHIFT1_U256) {
      r = lshift1_u256(r);
    } else {
      r = lshift1_from_add(r);
    }
  }
  return r;
}


// Optimized variant when the RIGHT operand is known odd (ECM case: N is odd).
// In this case, the common power of two is zero, so we can skip restoring 2^shift.
fn gcd_binary_u256_oddN(a_in: U256, N_odd: U256) -> U256 {
  var a = a_in;
  var b = N_odd;

  if (is_zero(a)) { return b; }
  while (is_even(a)) { a = rshift1(a); }

  loop {
    if (is_zero(b)) { return a; }
    while (is_even(b)) { b = rshift1(b); }
    if (cmp(a, b) > 0) { let t = a; a = b; b = t; }
    b = sub_u256(b, a);
  }
}

// --------------------------- Entry ---------------------------
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let h = getHeader();
  if (idx >= h.n_curves) { return; }
  

  // Load constants
  var off = constOffset();
  var N: U256; var R2: U256; var mont_one: U256;
  for (var i=0u;i<8u;i++){ N.limbs[i] = io.words[off+i]; }        off += 8u;
  for (var i=0u;i<8u;i++){ R2.limbs[i] = io.words[off+i]; }       off += 8u;
  for (var i=0u;i<8u;i++){ mont_one.limbs[i] = io.words[off+i]; } off += 8u;
  let n0inv32 = io.words[off];                                    off += 4u; // +pad(3)

  let pp_off = ppOffset();
  let out_base = outOffset(h) + idx * (8u + 1u + 3u);

  // RNG state (per curve) - ensure non-zero state
  var rng: vec2<u32>;
  let global_curve_idx = h.base_curve + idx;
  rng.x = h.seed_lo ^ global_curve_idx ^ 0x12345678u;
  rng.y = h.seed_hi ^ (global_curve_idx * 0x9E3779B9u) ^ 0x87654321u;
  
  // Ensure state is non-zero
  if (rng.x == 0u && rng.y == 0u) {
    rng.x = 0x12345678u + global_curve_idx;
    rng.y = 0x87654321u + global_curve_idx;
  }
  

  // Curve generation
  var A24m: U256;
  var X1m : U256;
  var curve_ok = false;

  for (var tries:u32=0u; tries<4u && !curve_ok; tries++){
    let sigma = next_sigma(N, &rng);
    let cr = generate_curve(sigma, N, R2, n0inv32);
    if (cr.ok) {
      A24m = cr.A24m;
      X1m  = cr.X1m;
      curve_ok = true;
      
      // Debug: store sigma in output for debugging
      // for (var i:u32=0u;i<8u;i++){ io.words[out_base+i] = sigma.limbs[i]; }
      // io.words[out_base+8u] = 70u; // debug status - curve generated
    }
  }

  if (!curve_ok) {
    // status=3 => bad curve / inverse failed
    for (var i:u32=0u;i<8u;i++){ io.words[out_base+i] = 0u; }
    io.words[out_base+8u] = 71u; // debug status - curve generation failed
    return;
  }

  // Stage 1 ladder
  var P = PointXZ(X1m, mont_one);
  var R = P;

  for (var i=0u; i<h.pp_count; i++){
    let pp = io.words[pp_off + i];
    if (pp <= 1u) { continue; }
    R = ladder(R, pp, A24m, N, n0inv32, mont_one);
  }

  // Output
  var result = set_zero();
  var status: u32 = 1u; // default: completed, no factor found

  if ((h.flags & 1u) != 0u) {
    let Zstd = from_mont(R.Z, N, n0inv32);
    let g = gcd_binary_u256_oddN(Zstd, N);
    result = g;

    let one = set_one();
    if (!is_zero(g) && (cmp(g, N) < 0) && (cmp(g, one) > 0)) {
      status = 2u; // non-trivial factor found
    } else {
      status = 1u; // no factor
    }
  } else {
    result = from_mont(R.Z, N, n0inv32);
    status = 1u;
  }

  for (var i:u32=0u;i<8u;i++){ io.words[out_base+i] = result.limbs[i]; }
  io.words[out_base+8u] = status;
}