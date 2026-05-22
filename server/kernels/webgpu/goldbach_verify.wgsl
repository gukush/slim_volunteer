@group(0) @binding(0) var<storage, read> numbers: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> small_primes: array<u32>;

struct Config {
    count: u32,
    prime_count: u32,
    dispatch_x: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> config: Config;

fn is_prime(n: u32, prime_count: u32) -> bool {
  if (n < 2u) {
    return false;
  }

  var i = 0u;
  while (i < prime_count) {
    let p = small_primes[i];
    if (p > n / p) {
      break;
    }
    if ((n % p) == 0u) {
      return n == p;
    }
    i = i + 1u;
  }
  return true;
}

fn find_goldbach_witness(n: u32, prime_count: u32) -> u32 {
  if (n == 4u) {
    return 2u;
  }

  var i = 1u;
  while (i < prime_count) {
    let p = small_primes[i];
    if (p > n / 2u) {
      break;
    }
    if (is_prime(n - p, prime_count)) {
      return p;
    }
    i = i + 1u;
  }
  return 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * config.dispatch_x + gid.x;

  // Read from uniform buffer instead of atomic storage
  let count = config.count;
  let prime_count = config.prime_count;

  if (idx >= count || atomicLoad(&result[0]) != 0u) {
    return;
  }

  let n = numbers[idx];
  if (n <= 2u || (n & 1u) != 0u) {
    // Wrap CAS in a loop to handle spurious failure
    var expected = 0u;
    loop {
      let cas = atomicCompareExchangeWeak(&result[0], expected, 1u);
      if (cas.exchanged) {
        atomicStore(&result[1], idx);
        atomicStore(&result[2], n);
        atomicStore(&result[3], 0u);
        atomicStore(&result[4], 1u);
        break;
      }
      if (cas.old_value != 0u) {
        break;
      }
    }
    return;
  }

  let witness = find_goldbach_witness(n, prime_count);
  if (witness == 0u) {
    var expected = 0u;
    loop {
      let cas = atomicCompareExchangeWeak(&result[0], expected, 1u);
      if (cas.exchanged) {
        atomicStore(&result[1], idx);
        atomicStore(&result[2], n);
        atomicStore(&result[3], 0u);
        atomicStore(&result[4], 2u);
        break;
      }
      if (cas.old_value != 0u) {
        break;
      }
    }
    return;
  }
}
