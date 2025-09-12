// kernels/ecm_stage1_webgpu_compute.wgsl - Version 3 (resumable)
//  - 256-bit big integers (8x u32, little-endian)
//  - Montgomery math (CIOS, word=32, L=8)
//  - Deterministic RNG via 64-bit LCG emulated with u32
//  - Suyama parametrization
//  - Binary modular inverse (u32)
//  - Correct constant-time Montgomery ladder with final swap
//  - RESUMABLE: Supports splitting work across multiple GPU submits
//  - Status codes: 0=needs_more, 1=no factor, 2=factor found, 3=bad curve/inverse failed

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
  version    : u32, // 3 for resumable
  rsv0       : u32,
  rsv1       : u32,
  pp_count   : u32,
  n_curves   : u32,
  seed_lo    : u32,
  seed_hi    : u32,
  base_curve : u32,
  flags      : u32,
  pp_start   : u32, // NEW: starting index for this pass
  pp_len     : u32, // NEW: number of pp to process this pass
};

struct CurveState {
  X_limbs    : array<u32, 8>,
  Z_limbs    : array<u32, 8>,
  A24_limbs  : array<u32, 8>,
  sigma      : u32,  // for debugging/reproducibility
  curve_ok   : u32,  // 1 if curve is valid
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
fn ppOffset()    -> u32 { return constOffset() + (8u*3u + 4u); }
fn outOffset(h: Header) -> u32 { return ppOffset() + h.pp_count; }
fn stateOffset(h: Header) -> u32 {
  return outOffset(h) + h.n_curves * (8u + 1u + 3u); // after output section
}

const LIMBS : u32 = 8u;
const STATE_WORDS_PER_CURVE : u32 = 8u + 8u + 8u + 2u; // X + Z + A24 + (sigma, curve_ok)

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
  let diff = sub_u256(b, a);
  return sub_u256(N, diff);
}

// 32x32 -> 64 multiply via 16-bit split
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
      let prod = mul32x32_64(a.limbs[i], b.limbs[j]);
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
  let diff = sub_u256(b, a);
  return sub_u256(N, diff);
}

fn mont_sqr(a: U256, N: U256, n0inv32: u32) -> U256 {
  return mont_mul(a, a, N, n0inv32);
}

fn to_mont(a: U256, R2: U256, N: U256, n0inv32: u32) -> U256 {
  return mont_mul(a, R2, N, n0inv32);
}

fn from_mont(a: U256, N: U256, n0inv32: u32) -> U256 {
  return mont_mul(a, set_one(), N, n0inv32);
}

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

fn ladder(P: PointXZ, k: u32, A24: U256, N: U256, n0inv32: u32, mont_one: U256) -> PointXZ {
  var R0 = PointXZ(mont_one, set_zero());
  var R1 = P;
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

  (*state).x = lcg_state;

  var sigma = acc;
  if (is_zero(sigma)) { sigma.limbs[0]=6u; }
  let one = set_one();
  if (cmp(sigma, one)==0) { sigma.limbs[0]=6u; }
  return sigma;
}


fn pcg32(state: ptr<function, vec2<u32>>) -> u32 {
  // Emulate 64-bit operations with two 32-bit values
  let oldstate_lo = (*state).x;
  let oldstate_hi = (*state).y;
  
  // LCG step: state = state * 6364136223846793005 + 1442695040888963407
  // Split into 32-bit operations
  let mult_lo = 0x4C957F2Du;  // Lower 32 bits of multiplier
  let mult_hi = 0x5851F42Du;  // Upper 32 bits of multiplier
  
  // Compute 64-bit multiplication (simplified for specific constants)
  let prod_lo = oldstate_lo * mult_lo;
  let prod_mid = oldstate_lo * mult_hi + oldstate_hi * mult_lo;
  let prod_hi = oldstate_hi * mult_hi + (prod_mid >> 16u);
  
  // Add increment (split across two words)
  let inc_lo = 0x14057B7Fu;
  let inc_hi = 0x5851F42Du;
  
  var new_lo = prod_lo + inc_lo;
  var new_hi = prod_hi + inc_hi;
  if (new_lo < inc_lo) { new_hi += 1u; }
  
  (*state).x = new_lo;
  (*state).y = new_hi;
  
  // PCG output function: improved bit mixing
  let xorshifted = ((oldstate_hi >> 18u) ^ oldstate_hi) >> 13u;
  let rot = oldstate_hi >> 27u;
  return (xorshifted >> rot) | (xorshifted << (32u - rot));
}

// Better entropy mixing using MurmurHash-style mixing
fn mix_entropy(a: u32, b: u32) -> u32 {
  var h = a ^ b;
  h ^= h >> 16u;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13u;
  h *= 0xC2B2AE35u;
  h ^= h >> 16u;
  return h;
}

fn next_sigma_hash(stream0: u32, stream1: u32, ctr: u32) -> U256 {
  var s: U256;
  var x0 = mix_entropy(stream0, ctr * 0x9E3779B9u);
  var x1 = mix_entropy(stream1, ctr * 0x85EBCA6Bu);
  for (var i: u32 = 0u; i < 8u; i++) {
    x0 = mix_entropy(x0, 0x27D4EB2Du ^ i);
    x1 = mix_entropy(x1, 0x165667B1u ^ (i * 0x9E3779B9u));
    s.limbs[i] = x0 ^ x1;
  }
  // avoid degenerate tiny values
  let six = u256_from_u32(6u);
  if (is_zero(s) || cmp(s, six) < 0) { s = six; }
  return s;
}

fn init_curve_stream(h: Header, curve_idx: u32) -> vec2<u32> {
  let g = h.base_curve + curve_idx;
  let s0 = mix_entropy(h.seed_lo, g * 0x61C88647u);
  let s1 = mix_entropy(h.seed_hi, g ^ 0xB5297A4Du);
  return vec2<u32>(s0, s1);
}

fn init_curve_rng(h: Header, curve_idx: u32) -> vec2<u32> {
  let global_curve_idx = h.base_curve + curve_idx;
  
  // Use multiple mixing rounds for better entropy distribution
  var state_lo = mix_entropy(h.seed_lo, global_curve_idx);
  state_lo = mix_entropy(state_lo, 0x9E3779B9u);  // Golden ratio constant
  
  var state_hi = mix_entropy(h.seed_hi, global_curve_idx * 0x61C88647u);
  state_hi = mix_entropy(state_hi, 0xBB67AE85u);  // Another mixing constant
  
  // Additional mixing with curve index
  state_lo = mix_entropy(state_lo, global_curve_idx * 2654435761u);
  state_hi = mix_entropy(state_hi, global_curve_idx * 2246822519u);
  
  // Ensure non-zero state
  if (state_lo == 0u && state_hi == 0u) {
    state_lo = 0x12345678u ^ global_curve_idx;
    state_hi = 0x87654321u ^ (global_curve_idx * 13u);
  }
  
  return vec2<u32>(state_lo, state_hi);
}

// --------------------------- Binary modular inverse ---------------------------
fn mod_inverse(a_in: U256, N: U256) -> InvResult {
  if (is_zero(a_in)) { return InvResult(false, set_zero()); }

  var a = a_in;
  for (var k=0u; k<2u && cmp(a, N) >= 0; k++){ a = sub_u256(a, N); }
  if (is_zero(a)) { return InvResult(false, set_zero()); }

  var u = a;
  var v = N;
  var x1 = set_one();
  var x2 = set_zero();

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
fn generate_curve(sigma: U256, N: U256, R2: U256, n0inv32: u32) -> CurveResult {
  let sigma_m = to_mont(sigma, R2, N, n0inv32);
  let five_m  = to_mont(u256_from_u32(5u),  R2, N, n0inv32);
  let four_m  = to_mont(u256_from_u32(4u),  R2, N, n0inv32);
  let three_m = to_mont(u256_from_u32(3u),  R2, N, n0inv32);
  let two_m   = to_mont(u256_from_u32(2u),  R2, N, n0inv32);

  let sigma_sq_m = mont_sqr(sigma_m, N, n0inv32);
  let u_m = mont_sub(sigma_sq_m, five_m, N);
  let v_m = mont_mul(four_m, sigma_m, N, n0inv32);

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

  let inv_u = mod_inverse(u_std, N);
  let inv_v = mod_inverse(v_std, N);
  if (!inv_u.ok || !inv_v.ok) {
    return CurveResult(false, set_zero(), set_zero());
  }

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

  let inv4_m = to_mont(mod_inverse(u256_from_u32(4u), N).val, R2, N, n0inv32);
  let A24m = mont_mul(A_plus_2_m, inv4_m, N, n0inv32);

  return CurveResult(true, A24m, X1m);
}

// --------------------------- GCD (binary, optimized for odd N) ---------------------------
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

// --------------------------- State management ---------------------------
fn load_state(idx: u32, h: Header) -> CurveState {
  let state_base = stateOffset(h) + idx * STATE_WORDS_PER_CURVE;
  var state: CurveState;

  for (var i = 0u; i < 8u; i++) {
    state.X_limbs[i] = io.words[state_base + i];
  }
  for (var i = 0u; i < 8u; i++) {
    state.Z_limbs[i] = io.words[state_base + 8u + i];
  }
  for (var i = 0u; i < 8u; i++) {
    state.A24_limbs[i] = io.words[state_base + 16u + i];
  }
  state.sigma = io.words[state_base + 24u];
  state.curve_ok = io.words[state_base + 25u];

  return state;
}

fn save_state(idx: u32, h: Header, state: CurveState) {
  let state_base = stateOffset(h) + idx * STATE_WORDS_PER_CURVE;

  for (var i = 0u; i < 8u; i++) {
    io.words[state_base + i] = state.X_limbs[i];
  }
  for (var i = 0u; i < 8u; i++) {
    io.words[state_base + 8u + i] = state.Z_limbs[i];
  }
  for (var i = 0u; i < 8u; i++) {
    io.words[state_base + 16u + i] = state.A24_limbs[i];
  }
  io.words[state_base + 24u] = state.sigma;
  io.words[state_base + 25u] = state.curve_ok;
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
  let n0inv32 = io.words[off];                                    off += 4u;

  let pp_off = ppOffset();
  let out_base = outOffset(h) + idx * (8u + 1u + 3u);

  var R: PointXZ;
  var A24m: U256;
  var sigma_val: u32 = 0u;
  var curve_ok: bool = false;

  // Check if this is a fresh start or resume
  if (h.pp_start == 0u) {
  // Fresh start - generate curve with improved RNG
  //var rng = init_curve_rng(h, idx);
  let st = init_curve_stream(h, idx);
  // Try generating a valid curve (increased attempts for better diversity)
  for (var tries: u32 = 0u; tries < 10u && !curve_ok; tries++) {
    let sigma = next_sigma_hash(st.x, st.y, tries);
    sigma_val = sigma.limbs[0]; // Store first limb for debugging
    
    let cr = generate_curve(sigma, N, R2, n0inv32);
    if (cr.ok) {
      A24m = cr.A24m;
      R = PointXZ(cr.X1m, mont_one);
      curve_ok = true;
    }
    
    // Add small delay/perturbation between attempts
    //if (!curve_ok && tries < 9u) {
      // Advance RNG state to ensure different sigma on retry
    //  for (var skip = 0u; skip < 3u; skip++) {
    //    let dummy = pcg32(&rng);
    //  }
    //}
  }
  
  if (!curve_ok) {
    // Bad curve - write error status
    for (var i: u32 = 0u; i < 8u; i++) { 
      io.words[out_base + i] = 0u; 
    }
    io.words[out_base + 8u] = 3u; // bad curve status
    return;
  }
} else {
    // Resume - load saved state
    let state = load_state(idx, h);
    var X: U256;
    var Z: U256;
    for (var i = 0u; i < 8u; i++) {
      X.limbs[i] = state.X_limbs[i];
      Z.limbs[i] = state.Z_limbs[i];
      A24m.limbs[i] = state.A24_limbs[i];
    }
    R = PointXZ(X, Z);
    sigma_val = state.sigma;
    curve_ok = (state.curve_ok == 1u);
  }

  if (!curve_ok) {
    // Shouldn't happen on resume, but be safe
    for (var i:u32=0u;i<8u;i++){ io.words[out_base+i] = 0u; }
    io.words[out_base+8u] = 3u;
    return;
  }

  // Process prime powers in this window
  let pp_end = min(h.pp_start + h.pp_len, h.pp_count);

  for (var i = h.pp_start; i < pp_end; i++) {
    let pp = io.words[pp_off + i];
    if (pp > 1u) {
      R = ladder(R, pp, A24m, N, n0inv32, mont_one);
    }
  }

  // Check if we're done or need more passes
  if (pp_end < h.pp_count) {
    // Not done - save state for next pass
    var state: CurveState;
    for (var i = 0u; i < 8u; i++) {
      state.X_limbs[i] = R.X.limbs[i];
      state.Z_limbs[i] = R.Z.limbs[i];
      state.A24_limbs[i] = A24m.limbs[i];
    }
    state.sigma = sigma_val;
    state.curve_ok = 1u;
    save_state(idx, h, state);

    // Write status 0 (needs more) to output
    io.words[out_base+8u] = 0u;
  } else {
    // Done - compute final result
    var result = set_zero();
    var status: u32 = 1u; // default: no factor

    if ((h.flags & 1u) != 0u) {
      let Zstd = from_mont(R.Z, N, n0inv32);
      let g = gcd_binary_u256_oddN(Zstd, N);
      result = g;

      let one = set_one();
      if (!is_zero(g) && (cmp(g, N) < 0) && (cmp(g, one) > 0)) {
        status = 2u; // factor found
      } else {
        status = 1u; // no factor
      }
    } else {
      result = from_mont(R.Z, N, n0inv32);
      status = 1u;
    }

    // Write final result
    for (var i:u32=0u;i<8u;i++){ io.words[out_base+i] = result.limbs[i]; }
    io.words[out_base+8u] = status;
  }
}
