#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
              << cudaGetErrorString(err__) << std::endl; \
    std::exit(2); \
  } \
} while (0)

static constexpr uint32_t MAGIC = 0x314d5031u; // "1MP1"

struct U256 {
  uint32_t limbs[8];
};

__host__ __device__ static inline U256 u256_zero() {
  U256 r{};
  for (int i = 0; i < 8; ++i) r.limbs[i] = 0;
  return r;
}

__host__ __device__ static inline U256 u256_one() {
  U256 r = u256_zero();
  r.limbs[0] = 1u;
  return r;
}

__host__ __device__ static inline U256 u256_from_u32(uint32_t x) {
  U256 r = u256_zero();
  r.limbs[0] = x;
  return r;
}

__host__ __device__ static inline bool u256_is_zero(const U256& a) {
  uint32_t x = 0;
  for (int i = 0; i < 8; ++i) x |= a.limbs[i];
  return x == 0u;
}

__host__ __device__ static inline bool u256_is_even(const U256& a) {
  return (a.limbs[0] & 1u) == 0u;
}

__host__ __device__ static inline int u256_cmp(const U256& a, const U256& b) {
  for (int i = 7; i >= 0; --i) {
    if (a.limbs[i] < b.limbs[i]) return -1;
    if (a.limbs[i] > b.limbs[i]) return 1;
  }
  return 0;
}

__host__ __device__ static inline uint2 addc(uint32_t a, uint32_t b, uint32_t cin) {
  uint32_t s = a + b;
  uint32_t c1 = (s < a) ? 1u : 0u;
  uint32_t s2 = s + cin;
  uint32_t c2 = (s2 < cin) ? 1u : 0u;
  return make_uint2(s2, c1 + c2);
}

__host__ __device__ static inline uint2 subb(uint32_t a, uint32_t b, uint32_t bin) {
  uint32_t d = a - b;
  uint32_t b1 = (a < b) ? 1u : 0u;
  uint32_t d2 = d - bin;
  uint32_t b2 = (d < bin) ? 1u : 0u;
  return make_uint2(d2, (b1 | b2));
}

__host__ __device__ static inline U256 u256_add(const U256& a, const U256& b) {
  U256 r;
  uint32_t c = 0;
  for (int i = 0; i < 8; ++i) {
    uint2 ac = addc(a.limbs[i], b.limbs[i], c);
    r.limbs[i] = ac.x;
    c = ac.y;
  }
  return r;
}

__host__ __device__ static inline U256 u256_sub(const U256& a, const U256& b) {
  U256 r;
  uint32_t br = 0;
  for (int i = 0; i < 8; ++i) {
    uint2 sb = subb(a.limbs[i], b.limbs[i], br);
    r.limbs[i] = sb.x;
    br = sb.y;
  }
  return r;
}

__host__ __device__ static inline U256 u256_rshift1(const U256& a) {
  U256 r;
  uint32_t carry = 0;
  for (int i = 7; i >= 0; --i) {
    uint32_t w = a.limbs[i];
    r.limbs[i] = (w >> 1) | (carry << 31);
    carry = w & 1u;
  }
  return r;
}

__host__ static inline uint32_t u256_get_bit(const U256& a, int bit) {
  return (a.limbs[bit / 32] >> (bit % 32)) & 1u;
}

__host__ static inline U256 u256_shl1_add_bit(const U256& a, uint32_t bit) {
  U256 r;
  uint32_t carry = bit & 1u;
  for (int i = 0; i < 8; ++i) {
    uint32_t nextCarry = a.limbs[i] >> 31;
    r.limbs[i] = (a.limbs[i] << 1) | carry;
    carry = nextCarry;
  }
  return r;
}

__host__ static inline U256 u256_mod(const U256& a, const U256& m) {
  U256 r = u256_zero();
  if (u256_is_zero(m)) return r;
  for (int bit = 255; bit >= 0; --bit) {
    r = u256_shl1_add_bit(r, u256_get_bit(a, bit));
    if (u256_cmp(r, m) >= 0) r = u256_sub(r, m);
  }
  return r;
}

__host__ __device__ static inline U256 cond_sub_N(const U256& a, const U256& N) {
  if (u256_cmp(a, N) >= 0) return u256_sub(a, N);
  return a;
}

__host__ __device__ static inline uint2 mul32x32_64(uint32_t a, uint32_t b) {
  uint32_t a0 = a & 0xffffu;
  uint32_t a1 = a >> 16;
  uint32_t b0 = b & 0xffffu;
  uint32_t b1 = b >> 16;
  uint32_t p00 = a0 * b0;
  uint32_t p01 = a0 * b1;
  uint32_t p10 = a1 * b0;
  uint32_t p11 = a1 * b1;

  // Catch 33rd-bit overflow from p01 + p10
  uint32_t mid_sum = p10 + p01;
  uint32_t mid_carry = (mid_sum < p10) ? 1u : 0u;

  // Add shifted middle sum to p00, catch carry into upper 32-bits
  uint32_t lo = p00 + (mid_sum << 16u);
  uint32_t lo_carry = (lo < p00) ? 1u : 0u;

  // Assemble high 32 bits
  uint32_t hi = p11 + (mid_sum >> 16u) + (mid_carry << 16u) + lo_carry;

  return make_uint2(lo, hi);
}

__host__ __device__ static inline U256 mont_mul(const U256& a, const U256& b, const U256& N, uint32_t n0inv32) {
  uint32_t t[10];
  for (int i = 0; i < 10; ++i) t[i] = 0;

  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
    // Step 1: T = T + a_i * b
    for (int j = 0; j < 8; ++j) {
      uint2 prod = mul32x32_64(a.limbs[i], b.limbs[j]);
      uint2 s1 = addc(t[j], prod.x, 0);
      uint2 s2 = addc(s1.x, carry, 0);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    uint2 s_t8 = addc(t[8], carry, 0);
    t[8] = s_t8.x;
    t[9] = s_t8.y;

    // Step 2: m = T[0] * n0inv mod 2^32
    uint32_t m = t[0] * n0inv32;
    carry = 0;

    // Step 3: T = T + m * N
    for (int j = 0; j < 8; ++j) {
      uint2 prod = mul32x32_64(m, N.limbs[j]);
      uint2 s1 = addc(t[j], prod.x, 0);
      uint2 s2 = addc(s1.x, carry, 0);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    uint2 s_t8_2 = addc(t[8], carry, 0);
    t[8] = s_t8_2.x;
    t[9] += s_t8_2.y;

    // Step 4: Shift T right by 32 bits
    for (int k = 0; k < 9; ++k) t[k] = t[k + 1];
    t[9] = 0;
  }

  U256 r;
  for (int i = 0; i < 8; ++i) r.limbs[i] = t[i];

  // Final reduction: subtract N if result overflowed 256 bits (t[8] > 0) OR if r >= N
  if (t[8] > 0 || u256_cmp(r, N) >= 0) {
    return u256_sub(r, N);
  }
  return r;
}

__host__ __device__ static inline U256 to_mont(const U256& a, const U256& R2, const U256& N, uint32_t n0inv32) {
  return mont_mul(a, R2, N, n0inv32);
}

__host__ __device__ static inline U256 from_mont(const U256& a, const U256& N, uint32_t n0inv32) {
  return mont_mul(a, u256_one(), N, n0inv32);
}

__host__ __device__ static inline U256 mont_pow_u32(U256 base, uint32_t exp, const U256& mont_one, const U256& N, uint32_t n0inv32) {
  U256 result = mont_one;
  U256 b = base;
  uint32_t e = exp;
  while (e) {
    if (e & 1u) result = mont_mul(result, b, N, n0inv32);
    e >>= 1;
    if (e) b = mont_mul(b, b, N, n0inv32);
  }
  return result;
}

__host__ __device__ static inline U256 gcd_binary_u256_oddN(U256 a, U256 b) {
  if (u256_is_zero(a)) return b;
  while (u256_is_even(a)) a = u256_rshift1(a);
  while (true) {
    if (u256_is_zero(b)) return a;
    while (u256_is_even(b)) b = u256_rshift1(b);
    if (u256_cmp(a, b) > 0) {
      U256 t = a; a = b; b = t;
    }
    b = u256_sub(b, a);
  }
}

__device__ static inline U256 read_u256(const uint32_t* buf, int offset) {
  U256 r;
  for (int i = 0; i < 8; ++i) r.limbs[i] = buf[offset + i];
  return r;
}

__device__ static inline void write_u256(uint32_t* buf, int offset, const U256& v) {
  for (int i = 0; i < 8; ++i) buf[offset + i] = v.limbs[i];
}

__global__ void pollard_pminus1_batched_kernel(uint32_t* io, uint32_t total_threads) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_threads) return;

  // Header
  const uint32_t pp_count   = io[2];
  const uint32_t n_bases    = io[3];
  const uint32_t base_start = io[4];
  const uint32_t num_ns     = io[5];

  if (num_ns == 0 || n_bases == 0) return;

  const uint32_t n_idx    = idx / n_bases;
  const uint32_t base_idx = idx % n_bases;

  if (n_idx >= num_ns) return;

  const int HEADER_WORDS = 8;
  const int CONST_WORDS  = 8 * 3 + 4; // 28

  const int CONST_OFFSET = HEADER_WORDS;
  const int PP_OFFSET    = CONST_OFFSET + num_ns * CONST_WORDS;
  const int OUT_OFFSET   = PP_OFFSET + pp_count;

  const int n_const_off = CONST_OFFSET + n_idx * CONST_WORDS;
  const int n_out_off   = OUT_OFFSET + n_idx * n_bases * 12;

  U256 N = read_u256(io, n_const_off);
  U256 R2 = read_u256(io, n_const_off + 8);
  U256 mont_one = read_u256(io, n_const_off + 16);
  uint32_t n0inv32 = io[n_const_off + 24];

  const int out_base = n_out_off + base_idx * 12;
  const uint32_t base_u32 = base_start + base_idx;

  U256 result = u256_zero();
  uint32_t status = 1u;

  if (base_u32 < 2u || u256_is_even(N)) {
    status = 3u;
  }

  if (status != 3u) {
    U256 a = to_mont(u256_from_u32(base_u32), R2, N, n0inv32);
    for (uint32_t i = 0; i < pp_count; ++i) {
      uint32_t pp = io[PP_OFFSET + i];
      if (pp > 1u) {
        a = mont_pow_u32(a, pp, mont_one, N, n0inv32);
      }
    }

    U256 a_std = from_mont(a, N, n0inv32);
    U256 diff;
    if (u256_cmp(a_std, u256_one()) >= 0) {
      diff = u256_sub(a_std, u256_one());
    } else {
      diff = u256_sub(N, u256_one());
    }
    U256 g = gcd_binary_u256_oddN(diff, N);
    result = g;
    if (u256_cmp(g, u256_one()) > 0 && u256_cmp(g, N) < 0) {
      status = 2u;
    }
  }

  write_u256(io, out_base, result);
  io[out_base + 8] = status;
  io[out_base + 9] = base_u32;
  io[out_base + 10] = 0u;
  io[out_base + 11] = 0u;
}

// Host helpers

static uint32_t parse_u32(const std::string& s) {
  return static_cast<uint32_t>(std::stoul(s, nullptr, 0));
}

static std::string u256_to_hex(const U256& v) {
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  bool started = false;
  for (int i = 7; i >= 0; --i) {
    if (v.limbs[i] != 0 || started || i == 0) {
      if (started) os << std::setw(8);
      else os << std::setw(0);
      os << v.limbs[i];
      started = true;
    }
  }
  return os.str();
}

static U256 parse_bigint(const std::string& s) {
  std::string t = s;
  if (t.rfind("0x", 0) == 0 || t.rfind("0X", 0) == 0) {
    t = t.substr(2);
    if (t.size() > 64) {
      std::cerr << "N must fit in 256 bits" << std::endl;
      std::exit(2);
    }
    U256 r = u256_zero();
    int limb = 0;
    int nibble = 0;
    for (auto it = t.rbegin(); it != t.rend(); ++it) {
      char c = *it;
      uint32_t v = 0;
      if (c >= '0' && c <= '9') v = c - '0';
      else if (c >= 'a' && c <= 'f') v = 10 + (c - 'a');
      else if (c >= 'A' && c <= 'F') v = 10 + (c - 'A');
      else { std::cerr << "Invalid hex digit" << std::endl; std::exit(2); }
      r.limbs[limb] |= (v << nibble);
      nibble += 4;
      if (nibble >= 32) { nibble = 0; limb++; }
    }
    return r;
  } else {
    U256 r = u256_zero();
    for (char c : t) {
      if (c < '0' || c > '9') { std::cerr << "Invalid decimal digit" << std::endl; std::exit(2); }
      uint32_t digit = c - '0';
      uint32_t carry = digit;
      for (int i = 0; i < 8; ++i) {
        uint64_t prod = (static_cast<uint64_t>(r.limbs[i]) * 10ull) + carry;
        r.limbs[i] = static_cast<uint32_t>(prod);
        carry = static_cast<uint32_t>(prod >> 32);
      }
      if (carry) { std::cerr << "N must fit in 256 bits" << std::endl; std::exit(2); }
    }
    return r;
  }
}

static uint64_t mod_inverse_u32(uint32_t a) {
  int64_t t = 0, newT = 1;
  uint64_t r = 1ull << 32;
  uint64_t newR = a & 0xffffffffull;
  while (newR != 0) {
    uint64_t q = r / newR;
    int64_t tmpT = t - static_cast<int64_t>(q) * newT;
    t = newT; newT = tmpT;
    uint64_t tmpR = r - q * newR;
    r = newR; newR = tmpR;
  }
  if (r != 1) return 0;
  int64_t inv = t;
  if (inv < 0) inv += (1ll << 32);
  return static_cast<uint64_t>(inv);
}

static void compute_montgomery_constants(const U256& N, U256& outR2, U256& outMontOne, uint32_t& outN0inv) {
  U256 RmodN = u256_zero();
  U256 one = u256_one();
  U256 cur = one;
  for (int i = 0; i < 256; ++i) {
    U256 d = u256_add(cur, cur);
    if (u256_cmp(d, N) >= 0 || u256_cmp(d, cur) < 0) {
      d = u256_sub(d, N);
    }
    cur = d;
  }
  RmodN = cur;

  // R2 mod N = (RmodN * RmodN) mod N = (2^256)^2 mod N = 2^512 mod N.
  // Instead of a 512-bit multiply + slow repeated subtraction,
  // we just double RmodN 256 more times mod N (same O(256) loop as above).
  cur = RmodN;
  for (int i = 0; i < 256; ++i) {
    U256 d = u256_add(cur, cur);
    if (u256_cmp(d, N) >= 0 || u256_cmp(d, cur) < 0) {
      d = u256_sub(d, N);
    }
    cur = d;
  }

  outR2 = cur;
  outMontOne = RmodN;

  uint32_t n0 = N.limbs[0];
  uint64_t n0inv = mod_inverse_u32(n0);
  outN0inv = static_cast<uint32_t>((-n0inv) & 0xffffffffull);
}

static std::vector<uint32_t> generate_prime_powers(uint32_t B1) {
  if (B1 < 2 || B1 > 0xffffffffu) {
    std::cerr << "B1 must be in [2, 2^32-1]" << std::endl;
    std::exit(2);
  }
  std::vector<uint8_t> sieve(B1 + 1, 0);
  std::vector<uint32_t> powers;
  for (uint32_t p = 2; p <= B1; ++p) {
    if (sieve[p]) continue;
    if (p <= B1 / p) {
      for (uint32_t m = p * p; m <= B1; m += p) sieve[m] = 1;
    }
    uint32_t pk = p;
    while (pk <= B1 / p) pk *= p;
    powers.push_back(pk);
  }
  return powers;
}

struct Number {
  std::string original;
  U256 N;
  U256 reducedN;
  bool hasEvenFactor;
  U256 R2;
  U256 montOne;
  uint32_t n0inv32;
};

static void usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [options]\n"
    << "  --N=HEX|DEC               Single number to factor\n"
    << "  --batch=N1,N2,...         Comma-separated list of numbers (same B1)\n"
  << "  --batchFile=FILE          File with one number per line\n"
  << "  --limit=N                 Use only the first N numbers from the batch\n"
  << "  --B1=N                    Stage-1 bound (default: 10000)\n"
  << "  --blockSize=N             CUDA block size (default: 256)\n";
}

int main(int argc, char** argv) {
  std::string Narg;
  std::string batchArg;
  std::string batchFile;
  uint32_t B1 = 10000;
  uint32_t blockSize = 256;
  uint32_t limit = 0; // 0 means no limit

  // Fixed: each number gets exactly 1 base (a=2). Parallelism comes from batch size.
  const uint32_t totalBases = 1;
  const uint32_t startBase = 2;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
    auto eq = arg.find('=');
    std::string key = (eq == std::string::npos) ? arg : arg.substr(0, eq);
    std::string val = (eq == std::string::npos) ? std::string() : arg.substr(eq + 1);
    if (key == "--N") Narg = val;
    else if (key == "--batch") batchArg = val;
    else if (key == "--batchFile") batchFile = val;
    else if (key == "--limit") limit = parse_u32(val);
    else if (key == "--B1") B1 = parse_u32(val);
    else if (key == "--blockSize") blockSize = parse_u32(val);
    else { std::cerr << "Unknown option: " << arg << "\n"; usage(argv[0]); return 2; }
  }

  std::vector<std::string> ns_str;
  if (!batchFile.empty()) {
    std::ifstream f(batchFile);
    if (!f) { std::cerr << "Cannot open batch file: " << batchFile << std::endl; return 2; }
    std::string line;
    while (std::getline(f, line)) {
      line.erase(0, line.find_first_not_of(" \t\r\n"));
      line.erase(line.find_last_not_of(" \t\r\n") + 1);
      if (!line.empty()) ns_str.push_back(line);
    }
  } else if (!batchArg.empty()) {
    size_t pos = 0;
    while (pos < batchArg.size()) {
      size_t comma = batchArg.find(',', pos);
      std::string token = batchArg.substr(pos, comma - pos);
      if (!token.empty()) ns_str.push_back(token);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  } else if (!Narg.empty()) {
    ns_str.push_back(Narg);
  } else {
    ns_str.push_back("0x123456789abcdef01");
  }

  if (limit > 0 && ns_str.size() > limit) {
    ns_str.resize(limit);
  }

  if (blockSize == 0 || blockSize > 1024) { std::cerr << "blockSize must be in [1, 1024]" << std::endl; return 2; }

  std::vector<Number> numbers;
  for (const auto& s : ns_str) {
    Number num;
    num.original = s;
    num.N = parse_bigint(s);
    if (u256_cmp(num.N, u256_from_u32(4)) < 0) {
      std::cerr << "N=" << s << " must be >= 4" << std::endl;
      return 2;
    }
    num.hasEvenFactor = u256_is_even(num.N);
    num.reducedN = num.N;
    if (num.hasEvenFactor) {
      // Strip all factors of 2 (trial division for 2)
      while (u256_is_even(num.reducedN)) {
        num.reducedN = u256_rshift1(num.reducedN);
      }
    }
    compute_montgomery_constants(num.reducedN, num.R2, num.montOne, num.n0inv32);
    numbers.push_back(num);
  }

  uint32_t numNs = static_cast<uint32_t>(numbers.size());
  std::vector<uint32_t> primePowers = generate_prime_powers(B1);

  // Build batched host IO buffer
  const int HEADER_WORDS = 8;
  const int CONST_WORDS  = 8 * 3 + 4; // 28
  const int PP_OFFSET    = HEADER_WORDS + numNs * CONST_WORDS;
  const int OUT_OFFSET   = PP_OFFSET + static_cast<int>(primePowers.size());
  const int OUT_WORDS_PER_BASE = 12;
  const int totalWords   = OUT_OFFSET + numNs * totalBases * OUT_WORDS_PER_BASE;

  std::vector<uint32_t> h_io(totalWords, 0);
  int off = 0;
  h_io[off++] = MAGIC;
  h_io[off++] = 2; // batched version
  h_io[off++] = static_cast<uint32_t>(primePowers.size());
  h_io[off++] = totalBases;
  h_io[off++] = startBase;
  h_io[off++] = numNs;
  h_io[off++] = 0;
  h_io[off++] = 0;

  for (const auto& num : numbers) {
    for (int i = 0; i < 8; ++i) h_io[off++] = num.reducedN.limbs[i];
    for (int i = 0; i < 8; ++i) h_io[off++] = num.R2.limbs[i];
    for (int i = 0; i < 8; ++i) h_io[off++] = num.montOne.limbs[i];
    h_io[off++] = num.n0inv32;
    h_io[off++] = 0; h_io[off++] = 0; h_io[off++] = 0;
  }
  for (uint32_t pp : primePowers) h_io[off++] = pp;

  // Device memory
  const auto total_epoch0 = std::chrono::system_clock::now();
  const auto total_wall0 = std::chrono::steady_clock::now();
  uint32_t* d_io = nullptr;
  CUDA_CHECK(cudaMalloc(&d_io, totalWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_io, h_io.data(), totalWords * sizeof(uint32_t), cudaMemcpyHostToDevice));

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  uint32_t totalThreads = numNs * totalBases;
  const auto wall0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(ev0));
  uint32_t grid = (totalThreads + blockSize - 1) / blockSize;
  pollard_pminus1_batched_kernel<<<grid, blockSize>>>(d_io, totalThreads);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  const auto wall1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));

  CUDA_CHECK(cudaMemcpy(h_io.data(), d_io, totalWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  const auto total_wall1 = std::chrono::steady_clock::now();
  const auto total_epoch1 = std::chrono::system_clock::now();

  CUDA_CHECK(cudaFree(d_io));
  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));

  double old_wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  double wall_ms = std::chrono::duration<double, std::milli>(total_wall1 - total_wall0).count();
  int64_t start_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_epoch0.time_since_epoch()).count();
  int64_t end_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_epoch1.time_since_epoch()).count();

  // Parse per-N results
  bool anyFound = false;
  std::cout << std::fixed << std::setprecision(3)
            << "batchSize=" << numNs
            << ",B1=" << B1
            << ",ppCount=" << primePowers.size()
            << ",startBase=" << startBase
            << ",totalBases=" << totalBases
            << ",totalThreads=" << totalThreads
            << ",blockSize=" << blockSize
            << ",grid=" << grid
            << ",old_wall_ms=" << old_wall_ms
            << ",wall_ms=" << wall_ms
            << ",start_epoch_ms=" << start_epoch_ms
            << ",end_epoch_ms=" << end_epoch_ms
            << ",kernel_ms=" << kernel_ms
            << "\n";

  for (uint32_t n = 0; n < numNs; ++n) {
    const auto& num = numbers[n];
    bool found = false;
    std::vector<std::string> factors;
    for (uint32_t b = 0; b < totalBases; ++b) {
      int base = OUT_OFFSET + n * totalBases * 12 + b * 12;
      uint32_t status = h_io[base + 8];
      if (status == 2u) {
        U256 f;
        for (int j = 0; j < 8; ++j) f.limbs[j] = h_io[base + j];
        U256 rem = u256_mod(num.reducedN, f);
        if (!u256_is_zero(f) && u256_cmp(f, u256_one()) > 0 && u256_cmp(f, num.reducedN) < 0 && u256_is_zero(rem)) {
          found = true;
          factors.push_back(u256_to_hex(f));
        } else {
          std::cerr << "Rejected invalid Pollard p-1 factor for N=" << num.original
                    << ": factor=0x" << u256_to_hex(f)
                    << ",status=" << status << std::endl;
        }
      }
    }
    if (found) anyFound = true;
    std::cout << "N=" << num.original
              << ",found=" << (found ? "true" : "false")
              << ",factors=";
    for (size_t i = 0; i < factors.size(); ++i) {
      if (i) std::cout << ";";
      std::cout << "0x" << factors[i];
    }
    std::cout << "\n";
  }

  return anyFound ? 0 : 1;
}
