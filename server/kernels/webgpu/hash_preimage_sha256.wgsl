struct Digest {
  h0: u32,
  h1: u32,
  h2: u32,
  h3: u32,
  h4: u32,
  h5: u32,
  h6: u32,
  h7: u32,
}

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>>;

const K: array<u32, 64> = array<u32, 64>(
  0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
  0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
  0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
  0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
  0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
  0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
  0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
  0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

fn rotr(x: u32, n: u32) -> u32 {
  return (x >> n) | (x << (32u - n));
}

fn ch(x: u32, y: u32, z: u32) -> u32 {
  return (x & y) ^ ((~x) & z);
}

fn maj(x: u32, y: u32, z: u32) -> u32 {
  return (x & y) ^ (x & z) ^ (y & z);
}

fn big_sigma0(x: u32) -> u32 {
  return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u);
}

fn big_sigma1(x: u32) -> u32 {
  return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u);
}

fn small_sigma0(x: u32) -> u32 {
  return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u);
}

fn small_sigma1(x: u32) -> u32 {
  return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u);
}

fn prefix_byte(index: u32) -> u32 {
  let word = params[8u + (index >> 2u)];
  return (word >> ((index & 3u) * 8u)) & 0xffu;
}

fn nonce_byte(index: u32, nonce_lo: u32, nonce_hi: u32) -> u32 {
  if (index < 4u) {
    return (nonce_lo >> (index * 8u)) & 0xffu;
  }
  return (nonce_hi >> ((index - 4u) * 8u)) & 0xffu;
}

fn message_byte(index: u32, nonce_lo: u32, nonce_hi: u32) -> u32 {
  let prefix_len = params[27];
  let message_len = prefix_len + 8u;
  if (index < prefix_len) {
    return prefix_byte(index);
  }
  if (index < message_len) {
    return nonce_byte(index - prefix_len, nonce_lo, nonce_hi);
  }
  if (index == message_len) {
    return 0x80u;
  }
  if (index == 62u) {
    return ((message_len * 8u) >> 8u) & 0xffu;
  }
  if (index == 63u) {
    return (message_len * 8u) & 0xffu;
  }
  return 0u;
}

fn message_word(word_index: u32, nonce_lo: u32, nonce_hi: u32) -> u32 {
  let base = word_index * 4u;
  return (message_byte(base, nonce_lo, nonce_hi) << 24u) |
         (message_byte(base + 1u, nonce_lo, nonce_hi) << 16u) |
         (message_byte(base + 2u, nonce_lo, nonce_hi) << 8u) |
         message_byte(base + 3u, nonce_lo, nonce_hi);
}

fn digest_word(h: Digest, i: u32) -> u32 {
  switch i {
    case 0u: { return h.h0; }
    case 1u: { return h.h1; }
    case 2u: { return h.h2; }
    case 3u: { return h.h3; }
    case 4u: { return h.h4; }
    case 5u: { return h.h5; }
    case 6u: { return h.h6; }
    default: { return h.h7; }
  }
}

fn leading_zero_match(h: Digest) -> bool {
  var remaining = params[28];
  for (var i = 0u; i < 8u; i = i + 1u) {
    if (remaining == 0u) {
      return true;
    }
    if (remaining >= 32u) {
      if (digest_word(h, i) != 0u) {
        return false;
      }
      remaining = remaining - 32u;
    } else {
      let mask = 0xffffffffu << (32u - remaining);
      return (digest_word(h, i) & mask) == 0u;
    }
  }
  return true;
}

fn target_match(h: Digest) -> bool {
  return h.h0 == params[0] &&
         h.h1 == params[1] &&
         h.h2 == params[2] &&
         h.h3 == params[3] &&
         h.h4 == params[4] &&
         h.h5 == params[5] &&
         h.h6 == params[6] &&
         h.h7 == params[7];
}

fn sha256_one_block(nonce_lo: u32, nonce_hi: u32) -> Digest {
  var w: array<u32, 64>;
  for (var i = 0u; i < 16u; i = i + 1u) {
    w[i] = message_word(i, nonce_lo, nonce_hi);
  }
  for (var i = 16u; i < 64u; i = i + 1u) {
    w[i] = small_sigma1(w[i - 2u]) + w[i - 7u] + small_sigma0(w[i - 15u]) + w[i - 16u];
  }

  var a = 0x6a09e667u;
  var b = 0xbb67ae85u;
  var c = 0x3c6ef372u;
  var d = 0xa54ff53au;
  var e = 0x510e527fu;
  var f = 0x9b05688cu;
  var g = 0x1f83d9abu;
  var h = 0x5be0cd19u;

  for (var i = 0u; i < 64u; i = i + 1u) {
    let t1 = h + big_sigma1(e) + ch(e, f, g) + K[i] + w[i];
    let t2 = big_sigma0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  return Digest(
    a + 0x6a09e667u,
    b + 0xbb67ae85u,
    c + 0x3c6ef372u,
    d + 0xa54ff53au,
    e + 0x510e527fu,
    f + 0x9b05688cu,
    g + 0x1f83d9abu,
    h + 0x5be0cd19u
  );
}

fn write_digest(h: Digest) {
  atomicStore(&result[3], h.h0);
  atomicStore(&result[4], h.h1);
  atomicStore(&result[5], h.h2);
  atomicStore(&result[6], h.h3);
  atomicStore(&result[7], h.h4);
  atomicStore(&result[8], h.h5);
  atomicStore(&result[9], h.h6);
  atomicStore(&result[10], h.h7);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params[26] || atomicLoad(&result[0]) != 0u) {
    return;
  }

  let start_lo = params[24];
  let nonce_lo = start_lo + idx;
  let carry = select(0u, 1u, nonce_lo < start_lo);
  let nonce_hi = params[25] + carry;
  let h = sha256_one_block(nonce_lo, nonce_hi);

  var ok = true;
  if (params[29] != 0u) {
    ok = ok && target_match(h);
  }
  if (params[28] > 0u) {
    ok = ok && leading_zero_match(h);
  }
  if (!ok) {
    return;
  }

  if (atomicCompareExchangeWeak(&result[0], 0u, 1u).exchanged) {
    atomicStore(&result[1], nonce_lo);
    atomicStore(&result[2], nonce_hi);
    write_digest(h);
  }
}
