// ecm_stage1_webgpu.wgsl
// ECM Stage 1 (Montgomery curves) for WebGPU (WGSL)
// - 256-bit big integers using 8x32-bit limbs (little-endian: limbs[0] is least significant)
// - Montgomery multiplication (CIOS) with 32-bit word size
// - One thread per curve. The host supplies: N, R2, mont_one, n0inv32, A24/X1 per curve, and prime powers list.
//
// Notes
// -----
// * n0inv32 = -N^{-1} mod 2^32 (host computes from N.limbs[0]).
// * mont_one = R mod N, where R = 2^(32*8) = 2^256.
// * R2 = R^2 mod N.
//
struct U256 {
  limbs: array<u32, 8>,
};

struct CurveIn {
  A24: U256,
  X1:  U256,
};

/// 32 bytes (U256) + 4 (status) + 12 (padding) = 48 bytes total
struct CurveOut {
  result: U256,
  status: u32,
  _pad0:  u32,
  _pad1:  u32,
  _pad2:  u32,
};

struct PointXZ {
  X: U256,
  Z: U256,
};

/// Constants buffer (binding 0)
/// 8*4 * 3 = 96 bytes for N,R2,one + 4 for n0inv32 + 12 pad = 112 bytes
struct Consts {
  N:        U256,
  R2:       U256,
  mont_one: U256,
  n0inv32:  u32,
  _cpad0:   u32,
  _cpad1:   u32,
  _cpad2:   u32,
};

/// Uniforms buffer (binding 4), 16 bytes total
struct Params {
  pp_count:    u32,
  num_curves:  u32,
  compute_gcd: u32,
  _upad:       u32,
};

struct Packed {
  consts: Consts,    // Constants (N, R2, mont_one, n0inv32)
  primes: array<u32>, // Prime powers list (runtime-sized array)
};

@group(0) @binding(0) var<uniform>              params: Params;         // uniforms
@group(0) @binding(1) var<storage, read>        packed: Packed;         // consts+primes
@group(0) @binding(2) var<storage, read>        curves: array<CurveIn>; // inputs
@group(0) @binding(3) var<storage, read_write>  outBuf: array<CurveOut>; // output

// --------- helpers: 32x32->64 via 16-bit split ---------
fn mul32x32_64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xFFFFu;
  let a1 = a >> 16u;
  let b0 = b & 0xFFFFu;
  let b1 = b >> 16u;

  var p00 = a0 * b0;               // 32-bit
  var p01 = a0 * b1;
  var p10 = a1 * b0;
  var p11 = a1 * b1;

  // assemble 64-bit: lo = p00 + ((p01 + p10) << 16), hi = p11 + carry from mid
  var mid = p01 + p10;
  var lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);

  var carry = (p00 >> 16u) + (mid >> 16u);
  // hi = p11 + carry
  var hi = p11 + carry;
  return vec2<u32>(lo, hi);
}

// add with carry: returns (sum, carry_out)
fn addc(a:u32, b:u32, cin:u32) -> vec2<u32> {
  let s = a + b;
  let c1 = select(0u, 1u, s < a);
  let s2 = s + cin;
  let c2 = select(0u, 1u, s2 < cin);
  return vec2<u32>(s2, c1 + c2);
}

// sub with borrow: returns (diff, borrow_out)
fn subb(a:u32, b:u32, bin:u32) -> vec2<u32> {
  let d = a - b;
  let b1 = select(0u, 1u, a < b);
  let d2 = d - bin;
  let b2 = select(0u, 1u, d < bin);
  return vec2<u32>(d2, select(0u, 1u, (b1 | b2) != 0u));
}

// comparisons
fn is_zero(a: U256) -> bool {
  var x:u32 = 0u;
  for (var i=0u;i<8u;i++) { x |= a.limbs[i]; }
  return x == 0u;
}
fn cmp(a: U256, b: U256) -> i32 {
  for (var i: i32 = 7; i >= 0; i--) {
    let ai = a.limbs[u32(i)];
    let bi = b.limbs[u32(i)];
    if (ai < bi) { return -1; }
    if (ai > bi) { return 1; }
  }
  return 0;
}

fn copy(a: U256) -> U256 { return a; }
fn set_zero() -> U256 { var r:U256; for (var i=0u;i<8u;i++){ r.limbs[i]=0u; } return r; }
fn set_one() -> U256 { var r = set_zero(); r.limbs[0]=1u; return r; }

fn add_u256(a: U256, b: U256) -> U256 {
  var r: U256;
  var c:u32 = 0u;
  for (var i=0u;i<8u;i++) {
    let ac = addc(a.limbs[i], b.limbs[i], c);
    r.limbs[i] = ac.x; c = ac.y;
  }
  return r;
}
fn sub_u256(a: U256, b: U256) -> U256 {
  var r: U256;
  var br:u32 = 0u;
  for (var i=0u;i<8u;i++) {
    let sb = subb(a.limbs[i], b.limbs[i], br);
    r.limbs[i] = sb.x; br = sb.y;
  }
  return r;
}

// conditional subtract N if r >= N
fn cond_sub_N(a: U256, N: U256) -> U256 {
  let c = cmp(a, N);
  if (c >= 0) {
    return sub_u256(a, N);
  }
  return a;
}

// Montgomery multiply (CIOS), word size 32, L=8.
fn mont_mul(a: U256, b: U256, N: U256, n0inv32: u32) -> U256 {
  var t : array<u32, 9>;
  for (var i=0u;i<9u;i++){ t[i]=0u; }

  for (var i=0u;i<8u;i++) {
    // t += a_i * b
    var carry:u32 = 0u;
    for (var j=0u;j<8u;j++) {
      let prod = mul32x32_64(a.limbs[i], b.limbs[j]);
      // t[j] = t[j] + prod.lo + carry
      let s1 = addc(t[j], prod.x, 0u);
      let s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    t[8] = t[8] + carry;

    // m = t0 * n0inv32 mod 2^32
    let m = t[0] * n0inv32;

    // t += m * N
    carry = 0u;
    for (var j=0u;j<8u;j++) {
      let prod = mul32x32_64(m, N.limbs[j]);
      let s1 = addc(t[j], prod.x, 0u);
      let s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    t[8] = t[8] + carry;

    // shift t right by one limb
    for (var k=0u;k<8u;k++){ t[k] = t[k+1]; }
    t[8]=0u;
  }

  var r:U256;
  for (var i=0u;i<8u;i++){ r.limbs[i]=t[i]; }
  return cond_sub_N(r, N);
}

fn to_mont(a: U256, C: Consts) -> U256 {
  return mont_mul(a, C.R2, C.N, C.n0inv32);
}
fn from_mont(a: U256, C: Consts) -> U256 {
  let one = set_one();  // Regular 1, not Montgomery 1
  return mont_mul(a, one, C.N, C.n0inv32);
}
fn mont_add(a: U256, b: U256, C: Consts) -> U256 {
  return cond_sub_N(add_u256(a,b), C.N);
}
fn mont_sub(a: U256, b: U256, C: Consts) -> U256 {
  let c = cmp(a,b);
  if (c >= 0) { return sub_u256(a,b); }
  else { return add_u256(sub_u256(a,b), C.N); }
}
fn mont_sqr(a: U256, C: Consts) -> U256 { return mont_mul(a,a,C.N,C.n0inv32); }

// ---- Montgomery curve x-only ops ----
//struct PointXZ { X: U256, Z: U256 };

fn xDBL(P: PointXZ, A24: U256, C: Consts) -> PointXZ {
  var t1 = mont_add(P.X, P.Z, C);  // X+Z
  var t2 = mont_sub(P.X, P.Z, C);  // X-Z
  let t3 = mont_sqr(t1, C);        // (X+Z)^2 = XX
  let t4 = mont_sqr(t2, C);        // (X-Z)^2 = ZZ
  let t5 = mont_sub(t3, t4, C);    // 4XZ = diff
  let temp = mont_mul(A24, t5, C.N, C.n0inv32);  // A24 * diff
  let Z_mult = mont_add(t3, temp, C);  // XX + A24 * diff
  let X2 = mont_mul(t3, t4, C.N, C.n0inv32);    // XX * ZZ
  let Z2 = mont_mul(t5, Z_mult, C.N, C.n0inv32);  // diff * Z_mult
  return PointXZ(X2, Z2);
}

fn xADD(P: PointXZ, Q: PointXZ, Diff: PointXZ, C: Consts) -> PointXZ {
  var t1 = mont_add(P.X, P.Z, C);
  var t2 = mont_sub(P.X, P.Z, C);
  var t3 = mont_add(Q.X, Q.Z, C);
  var t4 = mont_sub(Q.X, Q.Z, C);
  let t5 = mont_mul(t1, t4, C.N, C.n0inv32);
  let t6 = mont_mul(t2, t3, C.N, C.n0inv32);
  t1 = mont_add(t5, t6, C);
  t2 = mont_sub(t5, t6, C);
  t1 = mont_sqr(t1, C);
  t2 = mont_sqr(t2, C);
  let X3 = mont_mul(t1, Diff.Z, C.N, C.n0inv32);
  let Z3 = mont_mul(t2, Diff.X, C.N, C.n0inv32);
  return PointXZ(X3, Z3);
}

fn cswap(a: ptr<function, PointXZ>, b: ptr<function, PointXZ>, bit: u32) {
  let mask = (0u - (bit & 1u));
  for (var i=0u;i<8u;i++){
    let tx = ((*a).X.limbs[i] ^ (*b).X.limbs[i]) & mask;
    (*a).X.limbs[i] ^= tx; (*b).X.limbs[i] ^= tx;
    let tz = ((*a).Z.limbs[i] ^ (*b).Z.limbs[i]) & mask;
    (*a).Z.limbs[i] ^= tz; (*b).Z.limbs[i] ^= tz;
  }
}

fn ladder(P: PointXZ, k: u32, A24: U256, C: Consts) -> PointXZ {
  var R0 = PointXZ(C.mont_one, set_zero()); // (1:0)
  var R1 = P;
  var started = false;
  for (var i:i32=31; i>=0; i--) {
    let bit = (k >> u32(i)) & 1u;
    if (!started && bit==0u) { continue; }
    started = true;
    cswap(&R0, &R1, 1u - bit);
    let T0 = xADD(R0, R1, P, C);
    let T1 = xDBL(R1, A24, C);
    R0 = T0; R1 = T1;
  }
  return R0;
}

// ---- shifts and gcd ----
fn ctz32(x:u32) -> u32 {
  var n:u32 = 0u;
  if (x == 0u) { return 32u; }
  var y = x;
  while ((y & 1u) == 0u) { y = y >> 1u; n = n + 1u; }
  return n;
}
fn ctz_u256(a: U256) -> u32 {
  var tz:u32 = 0u;
  for (var i=0u;i<8u;i++) {
    let w = a.limbs[i];
    if (w == 0u) { tz += 32u; continue; }
    tz += ctz32(w); break;
  }
  return tz;
}
fn rshiftk(a: U256, k:u32) -> U256 {
  if (k == 0u) { return a; }
  var r = set_zero();
  let limb = k / 32u;
  let bits = k % 32u;
  for (var i=0u;i<8u;i++) {
    var val:u32 = 0u;
    if (i + limb < 8u) {
      val = a.limbs[i+limb];
      if (bits != 0u) {
        let cond = (i + limb + 1u) < 8u;
        let hi   = select(0u, a.limbs[i + limb + 1u], cond);
        val = (val >> bits) | (hi << (32u - bits));
      }
    }
    r.limbs[i] = val;
  }
  return r;
}
fn lshiftk(a: U256, k:u32) -> U256 {
  if (k == 0u) { return a; }
  var r = set_zero();
  let limb = k / 32u;
  let bits = k % 32u;
  for (var i: i32 = 7; i >= 0; i--) {
    var val:u32 = 0u;
    if (i >= i32(limb)) {
      val = a.limbs[u32(i) - limb];
      if (bits != 0u) {
        let cond = (i >= i32(limb)+1);
        let lo = select(0u, a.limbs[u32(i - 1) - limb], cond);
        val = (val << bits) | (lo >> (32u - bits));
      }
    }
    r.limbs[u32(i)] = val;
  }
  return r;
}

fn gcd_u256(a_in: U256, b_in: U256) -> U256 {
  var a = a_in; var b = b_in;
  if (is_zero(a)) { return b; }
  if (is_zero(b)) { return a; }
  let shift = min(ctz_u256(a), ctz_u256(b));
  a = rshiftk(a, ctz_u256(a));
  loop {
    b = rshiftk(b, ctz_u256(b));
    if (cmp(a,b) > 0) { let t=a; a=b; b=t; }
    b = sub_u256(b, a);
    if (is_zero(b)) { break; }
  }
  return lshiftk(a, shift);
}

// ---- main compute ----
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_curves) { return; }

  let C = packed.consts; // local alias
  let inC = curves[idx];

  var A24m = to_mont(inC.A24, C);
  var X1m  = to_mont(inC.X1,  C);

  var P = PointXZ(X1m, C.mont_one);
  var R = P;

  for (var i=0u; i<params.pp_count; i++) {
    let s = packed.primes[i];
    if (s <= 1u) { continue; }
    let T = ladder(R, s, A24m, C);
    R = T;
  }

  var result = set_zero();
  var status: u32 = 999u;
  // var status:u32 = 0u;

  if (params.compute_gcd == 1u) {
    status = 777u;
    let Zstd = from_mont(R.Z, C);
    let g = gcd_u256(Zstd, C.N);
    result = g;

    let one = set_one();
    if (!is_zero(g) && (cmp(g, C.N) < 0) && (cmp(g, one) > 0)) {
      status = 2u; // Found a non-trivial factor!
    } else {
      status = 1u; // No factor found
    }
  } else {
    result = from_mont(R.Z, C);
    status = 888u;
    // status = 0u;
  }

  outBuf[idx].result = result;
  outBuf[idx].status = status;
}
