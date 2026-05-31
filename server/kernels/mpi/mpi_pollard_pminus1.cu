#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <unistd.h> // gethostname

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
              << cudaGetErrorString(err__) << std::endl; \
    std::exit(2); \
  } \
} while (0)

/* ------------------------------------------------------------------ */
/*  CUDA kernel + helpers (identical to standalone)                   */
/* ------------------------------------------------------------------ */

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
  return x == 0;
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

__host__ __device__ static inline U256 u256_rshift1(const U256& a) {
  U256 r = u256_zero();
  uint32_t carry = 0;
  for (int i = 7; i >= 0; --i) {
    uint32_t w = a.limbs[i];
    r.limbs[i] = (w >> 1u) | (carry << 31u);
    carry = w & 1u;
  }
  return r;
}

__host__ __device__ static inline U256 sub_u256(const U256& a, const U256& b);

__host__ static inline uint32_t u256_get_bit(const U256& a, int bit) {
  return (a.limbs[bit / 32] >> (bit % 32)) & 1u;
}

__host__ static inline U256 u256_shl1_add_bit(const U256& a, uint32_t bit) {
  U256 r = u256_zero();
  uint32_t carry = bit & 1u;
  for (int i = 0; i < 8; ++i) {
    uint32_t nextCarry = a.limbs[i] >> 31;
    r.limbs[i] = (a.limbs[i] << 1) | carry;
    carry = nextCarry;
  }
  return r;
}

__host__ static inline U256 u256_mod_checked(const U256& a, const U256& m) {
  U256 r = u256_zero();
  if (u256_is_zero(m)) return r;
  for (int bit = 255; bit >= 0; --bit) {
    r = u256_shl1_add_bit(r, u256_get_bit(a, bit));
    if (u256_cmp(r, m) >= 0) r = sub_u256(r, m);
  }
  return r;
}

__host__ static inline U256 parse_bigint(const std::string& s) {
  U256 r = u256_zero();
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    for (size_t i = 2; i < s.size(); ++i) {
      char c = s[i];
      uint32_t digit = 0;
      if (c >= '0' && c <= '9') digit = c - '0';
      else if (c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
      else if (c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
      else { std::cerr << "Invalid hex digit" << std::endl; std::exit(2); }
      for (int j = 0; j < 8; ++j) {
        uint64_t tmp = ((uint64_t)r.limbs[j] << 4u) | digit;
        r.limbs[j] = (uint32_t)(tmp & 0xffffffffu);
        digit = (uint32_t)(tmp >> 32u);
      }
    }
  } else {
    for (size_t i = 0; i < s.size(); ++i) {
      char c = s[i];
      if (c < '0' || c > '9') { std::cerr << "Invalid decimal digit" << std::endl; std::exit(2); }
      uint32_t digit = c - '0';
      uint32_t carry = 0;
      for (int j = 0; j < 8; ++j) {
        uint64_t tmp = (uint64_t)r.limbs[j] * 10u + digit;
        r.limbs[j] = (uint32_t)(tmp & 0xffffffffu);
        digit = (uint32_t)(tmp >> 32u);
      }
      if (digit) { std::cerr << "N must fit in 256 bits" << std::endl; std::exit(2); }
    }
  }
  return r;
}

__host__ static inline std::string u256_to_hex(const U256& a) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  bool started = false;
  for (int i = 7; i >= 0; --i) {
    if (!started && a.limbs[i] == 0) continue;
    started = true;
    oss << std::setw(8) << a.limbs[i];
  }
  if (!started) oss << "0";
  return oss.str();
}

__host__ static inline uint64_t mul64(uint64_t a, uint64_t b) {
  return a * b;
}

__host__ static inline uint32_t addc(uint32_t a, uint32_t b, uint32_t cin, uint32_t* cout) {
  uint64_t s = (uint64_t)a + b + cin;
  *cout = (s > 0xffffffffu) ? 1 : 0;
  return (uint32_t)(s & 0xffffffffu);
}

__host__ static inline uint32_t subb(uint32_t a, uint32_t b, uint32_t bin, uint32_t* bout) {
  uint64_t d = (uint64_t)a - b - bin;
  *bout = (d > 0xffffffffu) ? 1 : 0;
  return (uint32_t)(d & 0xffffffffu);
}

__host__ __device__ static inline U256 add_u256(const U256& a, const U256& b) {
  U256 r = u256_zero();
  uint32_t c = 0;
  for (int i = 0; i < 8; ++i) {
    uint32_t s = a.limbs[i] + b.limbs[i];
    uint32_t c1 = (s < a.limbs[i]) ? 1u : 0u;
    uint32_t s2 = s + c;
    uint32_t c2 = (s2 < c) ? 1u : 0u;
    r.limbs[i] = s2;
    c = c1 + c2;
  }
  return r;
}

__host__ __device__ static inline U256 sub_u256(const U256& a, const U256& b) {
  U256 r = u256_zero();
  uint32_t br = 0;
  for (int i = 0; i < 8; ++i) {
    uint32_t d = a.limbs[i] - b.limbs[i];
    uint32_t b1 = (a.limbs[i] < b.limbs[i]) ? 1u : 0u;
    uint32_t d2 = d - br;
    uint32_t b2 = (d < br) ? 1u : 0u;
    r.limbs[i] = d2;
    br = b1 | b2;
  }
  return r;
}

__host__ static inline U256 mod_u256(const U256& a, const U256& m) {
  U256 r = a;
  while (u256_cmp(r, m) >= 0) {
    r = sub_u256(r, m);
  }
  return r;
}

__host__ static inline U256 mul_u32(const U256& a, uint32_t b) {
  U256 r = u256_zero();
  uint64_t carry = 0;
  for (int i = 0; i < 8; ++i) {
    uint64_t prod = (uint64_t)a.limbs[i] * b + carry;
    r.limbs[i] = (uint32_t)(prod & 0xffffffffu);
    carry = prod >> 32u;
  }
  return r;
}

/* Extended Euclidean for modular inverse modulo 2^32 */
__host__ static inline uint32_t modinv32(uint32_t a) {
  uint64_t t = 1, newt = 0;
  uint64_t r = a, newr = (1ull << 32);
  while (newr != 0) {
    uint64_t q = r / newr;
    uint64_t tmp = newt; newt = t - q * newt; t = tmp;
    tmp = newr; newr = r - q * newr; r = tmp;
  }
  if (r != 1) return 0;
  return (uint32_t)((t + (1ull << 32)) & 0xffffffffu);
}

__host__ static inline void compute_montgomery_constants(const U256& N, U256& R2, U256& montOne, uint32_t& n0inv32) {
  /* R = 2^256 mod N, R2 = R^2 mod N, montOne = R mod N */
  U256 cur = u256_zero();
  cur.limbs[0] = 1;
  for (int i = 0; i < 256; ++i) {
    U256 d = add_u256(cur, cur);
    if (u256_cmp(d, N) >= 0 || u256_cmp(d, cur) < 0) {
      d = sub_u256(d, N);
    }
    cur = d;
  }
  montOne = cur;

  /* Compute R2 = R^2 mod N by doubling R 256 more times */
  cur = montOne;
  for (int i = 0; i < 256; ++i) {
    U256 d = add_u256(cur, cur);
    if (u256_cmp(d, N) >= 0 || u256_cmp(d, cur) < 0) {
      d = sub_u256(d, N);
    }
    cur = d;
  }
  R2 = cur;

  uint32_t n0 = N.limbs[0];
  uint32_t n0inv = modinv32(n0);
  n0inv32 = (uint32_t)((-(uint64_t)n0inv) & 0xffffffffu);
}

/* ------------------------------------------------------------------ */
/*  Device helpers (must match standalone kernel)                     */
/* ------------------------------------------------------------------ */

__device__ static inline uint2 mul32x32_64(uint32_t a, uint32_t b) {
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

__device__ static inline uint2 addc_d(uint32_t a, uint32_t b, uint32_t cin) {
  uint32_t s = a + b;
  uint32_t c1 = (s < a) ? 1u : 0u;
  uint32_t s2 = s + cin;
  uint32_t c2 = (s2 < cin) ? 1u : 0u;
  return make_uint2(s2, c1 + c2);
}

__device__ static inline uint2 subb_d(uint32_t a, uint32_t b, uint32_t bin) {
  uint32_t d = a - b;
  uint32_t b1 = (a < b) ? 1u : 0u;
  uint32_t d2 = d - bin;
  uint32_t b2 = (d < bin) ? 1u : 0u;
  return make_uint2(d2, (b1 | b2));
}

__device__ static inline U256 mont_mul_dev(const U256& a, const U256& b, const U256& N, uint32_t n0inv32) {
  uint32_t t[10];
  for (int i = 0; i < 10; ++i) t[i] = 0;

  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      uint2 prod = mul32x32_64(a.limbs[i], b.limbs[j]);
      uint2 s1 = addc_d(t[j], prod.x, 0);
      uint2 s2 = addc_d(s1.x, carry, 0);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    uint2 s_t8 = addc_d(t[8], carry, 0);
    t[8] = s_t8.x;
    t[9] = s_t8.y;

    uint32_t m = t[0] * n0inv32;
    carry = 0;
    for (int j = 0; j < 8; ++j) {
      uint2 prod = mul32x32_64(m, N.limbs[j]);
      uint2 s1 = addc_d(t[j], prod.x, 0);
      uint2 s2 = addc_d(s1.x, carry, 0);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    uint2 s_t8_2 = addc_d(t[8], carry, 0);
    t[8] = s_t8_2.x;
    t[9] += s_t8_2.y;

    for (int k = 0; k < 9; ++k) t[k] = t[k + 1];
    t[9] = 0;
  }

  U256 r;
  for (int i = 0; i < 8; ++i) r.limbs[i] = t[i];

  if (t[8] > 0 || u256_cmp(r, N) >= 0) {
    uint32_t br = 0;
    for (int i = 0; i < 8; ++i) {
      uint2 sb = subb_d(t[i], N.limbs[i], br);
      r.limbs[i] = sb.x;
      br = sb.y;
    }
  }
  return r;
}

__device__ static inline U256 mont_pow_u32(U256 base, uint32_t exp, const U256& mont_one, const U256& N, uint32_t n0inv32) {
  U256 result = mont_one;
  U256 b = base;
  uint32_t e = exp;
  while (e) {
    if (e & 1u) result = mont_mul_dev(result, b, N, n0inv32);
    e >>= 1u;
    if (e) b = mont_mul_dev(b, b, N, n0inv32);
  }
  return result;
}

__device__ static inline U256 read_u256(const uint32_t* buf, int offset) {
  U256 r;
  for (int i = 0; i < 8; ++i) r.limbs[i] = buf[offset + i];
  return r;
}

__device__ static inline void write_u256(uint32_t* buf, int offset, const U256& v) {
  for (int i = 0; i < 8; ++i) buf[offset + i] = v.limbs[i];
}

__device__ static inline U256 to_mont(const U256& a, const U256& R2, const U256& N, uint32_t n0inv32) {
  return mont_mul_dev(a, R2, N, n0inv32);
}

__device__ static inline U256 from_mont(const U256& a, const U256& N, uint32_t n0inv32) {
  return mont_mul_dev(a, u256_one(), N, n0inv32);
}

__device__ static inline U256 gcd_binary_u256_oddN(U256 a, U256 b) {
  if (u256_is_zero(a)) return b;
  while (u256_is_even(a)) a = u256_rshift1(a);
  while (true) {
    if (u256_is_zero(b)) return a;
    while (u256_is_even(b)) b = u256_rshift1(b);
    if (u256_cmp(a, b) > 0) {
      U256 t = a; a = b; b = t;
    }
    b = sub_u256(b, a);
  }
}

__global__ void pollard_pminus1_batched_kernel(uint32_t* io, uint32_t total_threads) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_threads) return;

  const uint32_t pp_count   = io[2];
  const uint32_t n_bases    = io[3];
  const uint32_t base_start = io[4];
  const uint32_t num_ns     = io[5];

  if (num_ns == 0 || n_bases == 0) return;

  const uint32_t n_idx    = idx / n_bases;
  const uint32_t base_idx = idx % n_bases;

  if (n_idx >= num_ns) return;

  const int HEADER_WORDS = 8;
  const int CONST_WORDS  = 8 * 3 + 4;

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
      diff = sub_u256(a_std, u256_one());
    } else {
      diff = sub_u256(N, u256_one());
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

/* ------------------------------------------------------------------ */
/*  Host-side prime-power generation                                    */
/* ------------------------------------------------------------------ */

__host__ static inline std::vector<uint32_t> generate_prime_powers(uint32_t limit) {
  std::vector<uint32_t> powers;
  std::vector<bool> sieve(limit + 1, false);
  for (uint32_t p = 2; p <= limit; ++p) {
    if (sieve[p]) continue;
    if ((uint64_t)p * p <= limit) {
      for (uint32_t m = p * p; m <= limit; m += p) sieve[m] = true;
    }
    uint32_t pk = p;
    while ((uint64_t)pk * p <= limit) pk *= p;
    powers.push_back(pk);
  }
  return powers;
}

/* ------------------------------------------------------------------ */
/*  MPI main                                                           */
/* ------------------------------------------------------------------ */

struct WorkItem {
  uint32_t N_limbs[8];
};

struct ResultItem {
  uint32_t status;
  uint32_t factor_limbs[8];
  uint32_t base_u32;
  uint32_t pad[3];
};

struct TimingItem {
  double wall_ms;
  double kernel_ms;
  int64_t epoch_start_ms;
  int64_t epoch_end_ms;
};

static constexpr int DATA_TAG   = 1;
static constexpr int RESULT_TAG = 2;
static constexpr int FINISH_TAG = 3;

static uint32_t parse_u32(const std::string& s) {
  return static_cast<uint32_t>(std::stoul(s));
}

static void usage(const char* argv0) {
  std::cerr
    << "Usage: mpirun -np N " << argv0 << " [options]\n"
    << "  --N=HEX|DEC               Single number to factor\n"
    << "  --batch=N1,N2,...         Comma-separated list of numbers (same B1)\n"
    << "  --batchFile=FILE          File with one number per line\n"
    << "  --B1=N                    Stage-1 bound (default: 10000)\n"
    << "  --chunkSize=N             Numbers per MPI work message (default: 1024)\n"
    << "  --blockSize=N             CUDA block size (default: 256)\n"
    << "  --verboseTiming           Print per-chunk worker timing logs\n";
}

static std::string join_argv(int argc, char** argv) {
  std::ostringstream ss;
  for (int i = 0; i < argc; ++i) {
    if (i) ss << ' ';
    ss << (argv[i] ? argv[i] : "");
  }
  return ss.str();
}

static std::string csv_field(const std::string& value) {
  bool needs_quotes = false;
  for (char ch : value) {
    if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) return value;
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (char ch : value) {
    if (ch == '"') out.push_back('"');
    out.push_back(ch);
  }
  out.push_back('"');
  return out;
}

int main(int argc, char** argv) {

  const std::string command_line = join_argv(argc, argv);
  const auto epoch0 = std::chrono::system_clock::now();
  const auto wall0 = std::chrono::steady_clock::now();
  MPI_Init(&argc, &argv);

  int myrank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::string Narg;
  std::string batchArg;
  std::string batchFile;
  uint32_t B1 = 10000;
  uint32_t chunkSize = 1024;
  uint32_t blockSize = 256;
  uint8_t verboseTiming = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") { usage(argv[0]); MPI_Finalize(); return 0; }
    if (arg == "--verboseTiming") { verboseTiming = 1; continue; }
    auto eq = arg.find('=');
    std::string key = (eq == std::string::npos) ? arg : arg.substr(0, eq);
    std::string val = (eq == std::string::npos) ? std::string() : arg.substr(eq + 1);
    if (key == "--N") Narg = val;
    else if (key == "--batch") batchArg = val;
    else if (key == "--batchFile") batchFile = val;
    else if (key == "--B1") B1 = parse_u32(val);
    else if (key == "--chunkSize") chunkSize = parse_u32(val);
    else if (key == "--blockSize") blockSize = parse_u32(val);
  }
  MPI_Bcast(&verboseTiming, 1, MPI_UINT8_T, 0, MPI_COMM_WORLD);

/* Use actual hostname as machine identifier (works across hostfile MPI) */
char hostname_buf[256];
if (gethostname(hostname_buf, sizeof(hostname_buf)) != 0) {
  std::strncpy(hostname_buf, "unknown", sizeof(hostname_buf));
}
hostname_buf[sizeof(hostname_buf) - 1] = '\0';
std::string machineId(hostname_buf);

  if (blockSize == 0 || blockSize > 1024) {
    if (!myrank) std::cerr << "blockSize must be in [1, 1024]" << std::endl;
    MPI_Finalize();
    return 2;
  }
  if (chunkSize == 0 || chunkSize > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    if (!myrank) std::cerr << "chunkSize must be in [1, INT_MAX]" << std::endl;
    MPI_Finalize();
    return 2;
  }

  /* ---------- Parse input numbers ---------- */
  std::vector<std::string> ns_str;
  if (!myrank) {
    if (!batchFile.empty()) {
      std::ifstream f(batchFile);
      if (!f) { std::cerr << "Cannot open batch file: " << batchFile << std::endl; MPI_Finalize(); return 2; }
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
      std::cerr << "No input. Use --N, --batch, or --batchFile." << std::endl;
      usage(argv[0]);
      MPI_Finalize();
      return 2;
    }
  }

  uint32_t numNs = static_cast<uint32_t>(ns_str.size());
  MPI_Bcast(&numNs, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
  if (numNs == 0) {
    if (!myrank) std::cerr << "No numbers to factor" << std::endl;
    MPI_Finalize();
    return 2;
  }

  /* Slaves generate prime powers */
  std::vector<uint32_t> primePowers = generate_prime_powers(B1);

  /* ---------- Master ---------- */
  if (!myrank) {
    std::vector<U256> numbers(numNs);
    std::vector<U256> reducedNumbers(numNs);
    for (uint32_t n = 0; n < numNs; ++n) {
      numbers[n] = parse_bigint(ns_str[n]);
      if (u256_cmp(numbers[n], u256_from_u32(4)) < 0) {
        std::cerr << "N=" << ns_str[n] << " must be >= 4" << std::endl;
        MPI_Finalize();
        return 2;
      }
      reducedNumbers[n] = numbers[n];
      while (u256_is_even(reducedNumbers[n])) {
        reducedNumbers[n] = u256_rshift1(reducedNumbers[n]);
      }
    }



    uint32_t totalChunks = (numNs + chunkSize - 1) / chunkSize;
    uint32_t nextOffset = 0, sent = 0, received = 0;
    std::vector<uint32_t> worker_start(nproc, 0);
    std::vector<uint32_t> worker_count(nproc, 0);
    std::vector<size_t> worker_timing_index(nproc, 0);
    std::vector<uint8_t> worker_finished(nproc, 0);

    struct ChunkTiming {
      int worker_rank;
      int num_numbers;
      int64_t master_dispatch_ms;
      int64_t master_recv_ms;
      double slave_wall_ms;
      double slave_kernel_ms;
      int64_t slave_epoch_start_ms;
      int64_t slave_epoch_end_ms;
    };
    std::vector<ChunkTiming> chunk_timings;
    chunk_timings.reserve(totalChunks);

    auto epoch_ms = [](auto tp) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch()).count();
    };

    auto send_chunk = [&](int dst) {
      if (nextOffset >= numNs) return false;
      uint32_t cnt_u32 = std::min(chunkSize, numNs - nextOffset);
      int cnt = static_cast<int>(cnt_u32);
      int64_t dispatch_ms = epoch_ms(std::chrono::system_clock::now());
      MPI_Send(&cnt, 1, MPI_INT, dst, DATA_TAG, MPI_COMM_WORLD);
      std::vector<WorkItem> chunk(cnt);
      for (int i = 0; i < cnt; ++i) {
        for (int j = 0; j < 8; ++j) chunk[i].N_limbs[j] = numbers[nextOffset + i].limbs[j];
      }
      MPI_Send(chunk.data(), cnt * sizeof(WorkItem), MPI_BYTE, dst, DATA_TAG, MPI_COMM_WORLD);

      worker_start[dst] = nextOffset;
      worker_count[dst] = cnt_u32;
      worker_timing_index[dst] = chunk_timings.size();
      chunk_timings.push_back({dst, cnt, dispatch_ms, 0, 0.0, 0.0, 0, 0});
      nextOffset += cnt_u32;
      sent++;
      return true;
    };

    for (int dst = 1; dst < nproc; ++dst) {
      if (!send_chunk(dst)) {
        MPI_Send(NULL, 0, MPI_BYTE, dst, FINISH_TAG, MPI_COMM_WORLD);
        worker_finished[dst] = 1;
      }
    }

    /* receive results */
    std::vector<ResultItem> results(numNs);
    double total_slave_wall_ms = 0.0;
    double total_slave_kernel_ms = 0.0;
    int processed = 0;
    int nextPrint = 1000;
    while (received < totalChunks) {
      MPI_Status result_status;
      MPI_Status timing_status;
      MPI_Probe(MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &result_status);
      int dst = result_status.MPI_SOURCE;
      uint32_t chunkStart = worker_start[dst];
      int cnt = static_cast<int>(worker_count[dst]);
      std::vector<ResultItem> chunk_res(cnt);
      MPI_Recv(chunk_res.data(), cnt * sizeof(ResultItem), MPI_BYTE, dst, RESULT_TAG, MPI_COMM_WORLD, &result_status);
      for (int i = 0; i < cnt; ++i) {
        results[chunkStart + i] = chunk_res[i];
      }
      TimingItem timing;
      MPI_Recv(&timing, sizeof(TimingItem), MPI_BYTE, dst, RESULT_TAG, MPI_COMM_WORLD, &timing_status);
      int64_t recv_ms = epoch_ms(std::chrono::system_clock::now());
      total_slave_wall_ms += timing.wall_ms;
      total_slave_kernel_ms += timing.kernel_ms;
      size_t timingIndex = worker_timing_index[dst];
      chunk_timings[timingIndex].master_recv_ms = recv_ms;
      chunk_timings[timingIndex].slave_wall_ms = timing.wall_ms;
      chunk_timings[timingIndex].slave_kernel_ms = timing.kernel_ms;
      chunk_timings[timingIndex].slave_epoch_start_ms = timing.epoch_start_ms;
      chunk_timings[timingIndex].slave_epoch_end_ms = timing.epoch_end_ms;
      received++;
      processed += cnt;
      if (processed >= nextPrint) {
        std::cout << "[progress] " << processed << "/" << numNs << " done ("
                  << (100.0 * processed / numNs) << "%)\n";
        nextPrint += 1000;
      }

      if (!send_chunk(dst)) {
        MPI_Send(NULL, 0, MPI_BYTE, dst, FINISH_TAG, MPI_COMM_WORLD);
        worker_finished[dst] = 1;
      }
    }
    std::cout << "[progress] " << processed << "/" << numNs << " done (100%)\n";

    /* send FINISH */
    for (int i = 1; i < nproc; ++i) {
      if (worker_finished[i]) continue;
      MPI_Send(NULL, 0, MPI_BYTE, i, FINISH_TAG, MPI_COMM_WORLD);
      worker_finished[i] = 1;
    }

    const auto wall1 = std::chrono::steady_clock::now();
    const auto epoch1 = std::chrono::system_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    int64_t start_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch0.time_since_epoch()).count();
    int64_t end_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch1.time_since_epoch()).count();

    /* print results */
    bool anyFound = false;
    double avg_slave_wall = 0.0;
    double avg_slave_kernel = 0.0;
    if (received > 0) {
      avg_slave_wall = total_slave_wall_ms / received;
      avg_slave_kernel = total_slave_kernel_ms / received;
    }
    std::cout << std::fixed << std::setprecision(3)
              << "batchSize=" << numNs
              << ",B1=" << B1
              << ",ppCount=" << primePowers.size()
              << ",blockSize=" << blockSize
              << ",chunkSize=" << chunkSize
              << ",chunks=" << totalChunks
              << ",sent=" << sent
              << ",nproc=" << nproc
              << ",wall_ms=" << wall_ms
              << ",start_epoch_ms=" << start_epoch_ms
              << ",end_epoch_ms=" << end_epoch_ms
              << ",slave_wall_ms=" << avg_slave_wall
              << ",slave_kernel_ms=" << avg_slave_kernel
              << "\n";

    /* dump single consolidated chunk-timing CSV (all chunks, all workers) */
    {
      auto epoch_ms = [](auto tp) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
          tp.time_since_epoch()).count();
      };
      long long file_epoch_ms = epoch_ms(std::chrono::system_clock::now());
      std::ostringstream csvName;
      csvName << "chunk_timing_mpi_" << file_epoch_ms << ".csv";
      std::ofstream csv(csvName.str());
      if (csv.is_open()) {
        csv << "chunkId,replica,client_id,t_chunk_create,t_sent,"
            << "t_client_recv_abs,t_client_done_abs,"
            << "duration_ms,gpu_time_ms,"
            << "argv,B1,chunkSize,blockSize,batch_size,nproc,total_chunks\n";
        for (size_t i = 0; i < chunk_timings.size(); ++i) {
          const auto& ct = chunk_timings[i];
          csv << i << "," << ct.worker_rank << "," << machineId << ","
              << ct.master_dispatch_ms << "," << ct.master_dispatch_ms << ","
              << ct.slave_epoch_start_ms << "," << ct.slave_epoch_end_ms << ","
              << std::fixed << std::setprecision(3) << ct.slave_wall_ms << ","
              << ct.slave_kernel_ms << ","
              << csv_field(command_line) << ","
              << B1 << "," << chunkSize << "," << blockSize << ","
              << numNs << "," << nproc << "," << totalChunks << "\n";
        }
        std::cout << "[MPI] Wrote consolidated chunk timing CSV: " << csvName.str() << std::endl;
      }
    }

    for (uint32_t n = 0; n < numNs; ++n) {
      const auto& r = results[n];
      bool found = false;
      std::string factorStr;
      if (r.status == 2u) {
        U256 f;
        for (int j = 0; j < 8; ++j) f.limbs[j] = r.factor_limbs[j];
        U256 rem = u256_mod_checked(reducedNumbers[n], f);
        if (!u256_is_zero(f) && u256_cmp(f, u256_one()) > 0 && u256_cmp(f, reducedNumbers[n]) < 0 && u256_is_zero(rem)) {
          found = true;
          factorStr = u256_to_hex(f);
        } else {
          std::cerr << "Rejected invalid Pollard p-1 factor for N=" << ns_str[n]
                    << ": factor=0x" << u256_to_hex(f)
                    << ",status=" << r.status << std::endl;
        }
      }
      if (found) anyFound = true;
      std::cout << "N=" << ns_str[n]
                << ",found=" << (found ? "true" : "false")
                << ",status=" << r.status;
      if (found) std::cout << ",factor=0x" << factorStr;
      std::cout << "\n";
    }
  }
  /* ---------- Slaves ---------- */
  else {
    MPI_Status status;
    while (true) {
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == DATA_TAG) {
        int cnt = 0;
        MPI_Recv(&cnt, 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD, &status);
        if (cnt <= 0) continue;

        std::vector<WorkItem> chunk(cnt);
        MPI_Recv(chunk.data(), cnt * sizeof(WorkItem), MPI_BYTE, 0, DATA_TAG, MPI_COMM_WORLD, &status);

        /* Build batched IO buffer for cnt numbers */
        const int HEADER_WORDS = 8;
        const int CONST_WORDS  = 28;
        const int OUT_WORDS_PER_BASE = 12;
        int numNsChunk = cnt;
        int pp_count = static_cast<int>(primePowers.size());
        int pp_off = HEADER_WORDS + numNsChunk * CONST_WORDS;
        int out_off = pp_off + pp_count;
        int state_off = out_off + numNsChunk * 1 * OUT_WORDS_PER_BASE;
        int totalWords = state_off + numNsChunk * 1 * 8;

        std::vector<uint32_t> h_io(totalWords, 0);
        int off = 0;
        h_io[off++] = MAGIC;
        h_io[off++] = 3; // version with chunked execution + state storage
        h_io[off++] = static_cast<uint32_t>(pp_count);
        h_io[off++] = 1; // totalBases
        h_io[off++] = 2; // startBase
        h_io[off++] = static_cast<uint32_t>(numNsChunk);
        h_io[off++] = 0; // pp_start
        h_io[off++] = static_cast<uint32_t>(pp_count); // pp_len = full run in one pass

        /* Process each number: strip even factors, compute constants, fill IO */
        for (int n = 0; n < numNsChunk; ++n) {
          U256 N;
          for (int j = 0; j < 8; ++j) N.limbs[j] = chunk[n].N_limbs[j];
          bool hasEvenFactor = u256_is_even(N);
          U256 reducedN = N;
          if (hasEvenFactor) {
            while (u256_is_even(reducedN)) {
              reducedN = u256_rshift1(reducedN);
            }
          }
          U256 R2, montOne;
          uint32_t n0inv32;
          compute_montgomery_constants(reducedN, R2, montOne, n0inv32);

          for (int j = 0; j < 8; ++j) h_io[off++] = reducedN.limbs[j];
          for (int j = 0; j < 8; ++j) h_io[off++] = R2.limbs[j];
          for (int j = 0; j < 8; ++j) h_io[off++] = montOne.limbs[j];
          h_io[off++] = n0inv32;
          h_io[off++] = 0; h_io[off++] = 0; h_io[off++] = 0;
        }

        for (uint32_t pp : primePowers) h_io[off++] = pp;
        /* output + state areas already zero-initialized */

        const auto slave_wall0 = std::chrono::steady_clock::now();
        const auto slave_epoch0 = std::chrono::system_clock::now();

        /* device memory */
        uint32_t* d_io = nullptr;
        CUDA_CHECK(cudaMalloc(&d_io, totalWords * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_io, h_io.data(), totalWords * sizeof(uint32_t), cudaMemcpyHostToDevice));

        /* launch */
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        pollard_pminus1_batched_kernel<<<(cnt + blockSize - 1) / blockSize, blockSize>>>(d_io, cnt);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());

        float slave_kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&slave_kernel_ms, ev0, ev1));

        CUDA_CHECK(cudaMemcpy(h_io.data(), d_io, totalWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const auto slave_wall1 = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaFree(d_io));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));

        /* extract results */
        std::vector<ResultItem> chunk_res(cnt);
        for (int n = 0; n < cnt; ++n) {
          int base = out_off + n * 1 * OUT_WORDS_PER_BASE;
          chunk_res[n].status = h_io[base + 8];
          for (int j = 0; j < 8; ++j) chunk_res[n].factor_limbs[j] = h_io[base + j];
          chunk_res[n].base_u32 = h_io[base + 9];
          chunk_res[n].pad[0] = chunk_res[n].pad[1] = chunk_res[n].pad[2] = 0;
        }

        MPI_Send(chunk_res.data(), cnt * sizeof(ResultItem), MPI_BYTE, 0, RESULT_TAG, MPI_COMM_WORLD);

        const auto slave_epoch1 = std::chrono::system_clock::now();
        double slave_wall_ms = std::chrono::duration<double, std::milli>(slave_wall1 - slave_wall0).count();
        auto epoch_ms = [](auto tp) {
          return std::chrono::duration_cast<std::chrono::milliseconds>(
            tp.time_since_epoch()).count();
        };
        TimingItem timing{
          slave_wall_ms,
          static_cast<double>(slave_kernel_ms),
          epoch_ms(slave_epoch0),
          epoch_ms(slave_epoch1)
        };
        MPI_Send(&timing, sizeof(TimingItem), MPI_BYTE, 0, RESULT_TAG, MPI_COMM_WORLD);

        if (verboseTiming) {
          std::cout << "[MPI_TIMING] epoch_start_ms=" << timing.epoch_start_ms
                    << " epoch_end_ms=" << timing.epoch_end_ms
                    << " slave_wall_ms=" << std::fixed << std::setprecision(3) << slave_wall_ms
                    << " slave_kernel_ms=" << slave_kernel_ms
                    << " chunk_size=" << cnt
                    << " rank=" << myrank << std::endl;
        }
      } else if (status.MPI_TAG == FINISH_TAG) {
        MPI_Recv(NULL, 0, MPI_BYTE, 0, FINISH_TAG, MPI_COMM_WORLD, &status);
        break;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
