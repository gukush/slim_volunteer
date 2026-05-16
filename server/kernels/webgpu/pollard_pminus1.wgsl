struct U256 { limbs: array<u32, 8>, };
struct IO { words: array<u32>, };

@group(0) @binding(0) var<storage, read_write> io : IO;

struct Header {
  magic: u32,
  version: u32,
  pp_count: u32,
  n_bases: u32,
  base_start: u32,
  flags: u32,
  rsv0: u32,
  rsv1: u32,
};

fn getHeader() -> Header {
  return Header(io.words[0], io.words[1], io.words[2], io.words[3], io.words[4], io.words[5], io.words[6], io.words[7]);
}

fn constOffset() -> u32 { return 8u; }
fn ppOffset() -> u32 { return constOffset() + (8u * 3u + 4u); }
fn outOffset(h: Header) -> u32 { return ppOffset() + h.pp_count; }

fn set_zero() -> U256 {
  var r: U256;
  for (var i = 0u; i < 8u; i++) { r.limbs[i] = 0u; }
  return r;
}

fn set_one() -> U256 {
  var r = set_zero();
  r.limbs[0] = 1u;
  return r;
}

fn u256_from_u32(x: u32) -> U256 {
  var r = set_zero();
  r.limbs[0] = x;
  return r;
}

fn is_zero(a: U256) -> bool {
  var x = 0u;
  for (var i = 0u; i < 8u; i++) { x = x | a.limbs[i]; }
  return x == 0u;
}

fn is_even(a: U256) -> bool { return (a.limbs[0] & 1u) == 0u; }

fn cmp(a: U256, b: U256) -> i32 {
  for (var i: i32 = 7; i >= 0; i--) {
    let ai = a.limbs[u32(i)];
    let bi = b.limbs[u32(i)];
    if (ai < bi) { return -1; }
    if (ai > bi) { return 1; }
  }
  return 0;
}

fn addc(a: u32, b: u32, cin: u32) -> vec2<u32> {
  let s = a + b;
  let c1 = select(0u, 1u, s < a);
  let s2 = s + cin;
  let c2 = select(0u, 1u, s2 < cin);
  return vec2<u32>(s2, c1 + c2);
}

fn subb(a: u32, b: u32, bin: u32) -> vec2<u32> {
  let d = a - b;
  let b1 = select(0u, 1u, a < b);
  let d2 = d - bin;
  let b2 = select(0u, 1u, d < bin);
  return vec2<u32>(d2, select(0u, 1u, (b1 | b2) != 0u));
}

fn add_u256(a: U256, b: U256) -> U256 {
  var r: U256;
  var c = 0u;
  for (var i = 0u; i < 8u; i++) {
    let ac = addc(a.limbs[i], b.limbs[i], c);
    r.limbs[i] = ac.x;
    c = ac.y;
  }
  return r;
}

fn sub_u256(a: U256, b: U256) -> U256 {
  var r: U256;
  var br = 0u;
  for (var i = 0u; i < 8u; i++) {
    let sb = subb(a.limbs[i], b.limbs[i], br);
    r.limbs[i] = sb.x;
    br = sb.y;
  }
  return r;
}

fn rshift1(a: U256) -> U256 {
  var r: U256;
  var carry = 0u;
  for (var i: i32 = 7; i >= 0; i--) {
    let w = a.limbs[u32(i)];
    r.limbs[u32(i)] = (w >> 1u) | (carry << 31u);
    carry = w & 1u;
  }
  return r;
}

fn cond_sub_N(a: U256, N: U256) -> U256 {
  if (cmp(a, N) >= 0) { return sub_u256(a, N); }
  return a;
}

fn mul32x32_64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = p01 + p10;
  let lo = (p00 & 0xffffu) | ((mid & 0xffffu) << 16u);
  let carry = (p00 >> 16u) + (mid >> 16u);
  let hi = p11 + carry;
  return vec2<u32>(lo, hi);
}

fn mont_mul(a: U256, b: U256, N: U256, n0inv32: u32) -> U256 {
  var t: array<u32, 10>;
  for (var i = 0u; i < 10u; i++) { t[i] = 0u; }

  for (var i = 0u; i < 8u; i++) {
    var carry = 0u;
    // Step 1: T = T + a_i * b
    for (var j = 0u; j < 8u; j++) {
      let prod = mul32x32_64(a.limbs[i], b.limbs[j]);
      let s1 = addc(t[j], prod.x, 0u);
      let s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    let s_t8 = addc(t[8], carry, 0u);
    t[8] = s_t8.x;
    t[9] = s_t8.y;

    // Step 2: m = T[0] * n0inv mod 2^32
    let m = t[0] * n0inv32;
    carry = 0u;

    // Step 3: T = T + m * N
    for (var j = 0u; j < 8u; j++) {
      let prod = mul32x32_64(m, N.limbs[j]);
      let s1 = addc(t[j], prod.x, 0u);
      let s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    let s_t8_2 = addc(t[8], carry, 0u);
    t[8] = s_t8_2.x;
    t[9] += s_t8_2.y;

    // Step 4: Shift T right by 32 bits
    for (var k = 0u; k < 9u; k++) { t[k] = t[k + 1u]; }
    t[9] = 0u;
  }

  var r: U256;
  for (var i = 0u; i < 8u; i++) { r.limbs[i] = t[i]; }

  // Final reduction: subtract N if result overflowed 256 bits (t[8] > 0) OR if r >= N
  if (t[8] > 0u || cmp(r, N) >= 0) {
    return sub_u256(r, N);
  }
  return r;
}

fn to_mont(a: U256, R2: U256, N: U256, n0inv32: u32) -> U256 {
  return mont_mul(a, R2, N, n0inv32);
}

fn from_mont(a: U256, N: U256, n0inv32: u32) -> U256 {
  return mont_mul(a, set_one(), N, n0inv32);
}

fn mont_pow_u32(base: U256, exp: u32, mont_one: U256, N: U256, n0inv32: u32) -> U256 {
  var result = mont_one;
  var b = base;
  var e = exp;
  loop {
    if (e == 0u) { break; }
    if ((e & 1u) != 0u) {
      result = mont_mul(result, b, N, n0inv32);
    }
    e = e >> 1u;
    if (e != 0u) {
      b = mont_mul(b, b, N, n0inv32);
    }
  }
  return result;
}

fn gcd_binary_u256_oddN(a_in: U256, N_odd: U256) -> U256 {
  var a = a_in;
  var b = N_odd;
  if (is_zero(a)) { return b; }
  while (is_even(a)) { a = rshift1(a); }
  loop {
    if (is_zero(b)) { return a; }
    while (is_even(b)) { b = rshift1(b); }
    if (cmp(a, b) > 0) {
      let t = a;
      a = b;
      b = t;
    }
    b = sub_u256(b, a);
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let h = getHeader();
  if (idx >= h.n_bases) { return; }

  var off = constOffset();
  var N: U256;
  var R2: U256;
  var mont_one: U256;
  for (var i = 0u; i < 8u; i++) { N.limbs[i] = io.words[off + i]; } off += 8u;
  for (var i = 0u; i < 8u; i++) { R2.limbs[i] = io.words[off + i]; } off += 8u;
  for (var i = 0u; i < 8u; i++) { mont_one.limbs[i] = io.words[off + i]; } off += 8u;
  let n0inv32 = io.words[off];

  let out_base = outOffset(h) + idx * 12u;
  let base_u32 = h.base_start + idx;
  var result = set_zero();
  var status = 1u;

  if (base_u32 < 2u || is_even(N)) {
    status = 3u;
  } else {
    var a = to_mont(u256_from_u32(base_u32), R2, N, n0inv32);
    let pp_off = ppOffset();
    for (var i = 0u; i < h.pp_count; i++) {
      let pp = io.words[pp_off + i];
      if (pp > 1u) {
        a = mont_pow_u32(a, pp, mont_one, N, n0inv32);
      }
    }

    let a_std = from_mont(a, N, n0inv32);
    var diff: U256;
    if (cmp(a_std, set_one()) >= 0) {
      diff = sub_u256(a_std, set_one());
    } else {
      diff = sub_u256(N, set_one());
    }
    let g = gcd_binary_u256_oddN(diff, N);
    result = g;
    if (cmp(g, set_one()) > 0 && cmp(g, N) < 0) {
      status = 2u;
    }
  }

  for (var i = 0u; i < 8u; i++) { io.words[out_base + i] = result.limbs[i]; }
  io.words[out_base + 8u] = status;
  io.words[out_base + 9u] = base_u32;
  io.words[out_base + 10u] = 0u;
  io.words[out_base + 11u] = 0u;
}
