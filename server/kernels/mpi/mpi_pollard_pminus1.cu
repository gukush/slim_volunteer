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

  uint32_t mid_sum = p10 + p01;
  uint32_t mid_carry = (mid_sum < p10) ? 1u : 0u;

  uint32_t lo = p00 + (mid_sum << 16u);
  uint32_t lo_carry = (lo < p00) ? 1u : 0u;

  uint32_t hi = p11 + (mid_sum >> 16u) + (mid_carry << 16u) + lo_carry;

  return make_uint2(lo, hi);
}

__host__ __device__ static inline U256 mont_mul(const U256& a, const U256& b, const U256& N, uint32_t n0inv32) {
  uint32_t t[10];
  for (int i = 0; i < 10; ++i) t[i] = 0;

  for (int i = 0; i < 8; ++i) {
    uint32_t carry = 0;
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

    uint32_t m = t[0] * n0inv32;
    carry = 0;

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

    for (int k = 0; k < 9; ++k) t[k] = t[k + 1];
    t[9] = 0;
  }

  U256 r;
  for (int i = 0; i < 8; ++i) r.limbs[i] = t[i];

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

/* ------------------------------------------------------------------ */
/*  Host helpers (identical to standalone)                            */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  MPI master-slave pattern                                          */
/* ------------------------------------------------------------------ */

#define DATA_TAG    0
#define RESULT_TAG  1
#define FINISH_TAG  2

/* One work item = 8 uint32_t limbs of N */
struct WorkItem {
  uint32_t N_limbs[8];
};

/* One result item = status + factor + base */
struct ResultItem {
  uint32_t status;
  uint32_t factor_limbs[8];
  uint32_t base_u32;
  uint32_t pad[3]; // 16 uint32_t total = 64 bytes
};

static void usage(const char* argv0) {
  std::cerr
    << "Usage: mpirun -np N " << argv0 << " [options]\n"
    << "  --N=HEX|DEC               Single number to factor\n"
    << "  --batch=N1,N2,...         Comma-separated list of numbers (same B1)\n"
    << "  --batchFile=FILE          File with one number per line\n"
    << "  --B1=N                    Stage-1 bound (default: 10000)\n"
    << "  --blockSize=N             CUDA block size (default: 256)\n";
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int myrank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::string Narg;
  std::string batchArg;
  std::string batchFile;
  uint32_t B1 = 10000;
  uint32_t blockSize = 256;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") { usage(argv[0]); MPI_Finalize(); return 0; }
    auto eq = arg.find('=');
    std::string key = (eq == std::string::npos) ? arg : arg.substr(0, eq);
    std::string val = (eq == std::string::npos) ? std::string() : arg.substr(eq + 1);
    if (key == "--N") Narg = val;
    else if (key == "--batch") batchArg = val;
    else if (key == "--batchFile") batchFile = val;
    else if (key == "--B1") B1 = parse_u32(val);
    else if (key == "--blockSize") blockSize = parse_u32(val);
  }

  if (blockSize == 0 || blockSize > 1024) {
    if (!myrank) std::cerr << "blockSize must be in [1, 1024]" << std::endl;
    MPI_Finalize();
    return 2;
  }

  /* ---------- Master reads all input numbers ---------- */
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
      ns_str.push_back("0x123456789abcdef01");
    }
  }

  /* Broadcast B1 to all slaves (they need it to generate prime powers) */
  MPI_Bcast(&B1, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  uint32_t numNs = static_cast<uint32_t>(ns_str.size());
  MPI_Bcast(&numNs, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (numNs == 0) {
    if (!myrank) std::cerr << "No numbers to factor" << std::endl;
    MPI_Finalize();
    return 2;
  }

  /* ---------- Slaves generate prime powers once ---------- */
  std::vector<uint32_t> primePowers = generate_prime_powers(B1);

  /* ---------- Master (rank 0) ---------- */
  if (!myrank) {
    std::vector<U256> numbers(numNs);
    for (uint32_t n = 0; n < numNs; ++n) {
      numbers[n] = parse_bigint(ns_str[n]);
      if (u256_cmp(numbers[n], u256_from_u32(4)) < 0) {
        std::cerr << "N=" << ns_str[n] << " must be >= 4" << std::endl;
        MPI_Finalize();
        return 2;
      }
    }

    const auto wall0 = std::chrono::steady_clock::now();

    uint32_t offset = 0;
    std::vector<uint32_t> slave_idx(nproc, 0);

    /* send initial work (1 number per slave) */
    int active = 0;
    for (int i = 1; i < nproc && offset < numNs; i++) {
      WorkItem w;
      for (int j = 0; j < 8; ++j) w.N_limbs[j] = numbers[offset].limbs[j];
      MPI_Send(&w, sizeof(WorkItem), MPI_BYTE, i, DATA_TAG, MPI_COMM_WORLD);
      slave_idx[i] = offset;
      offset++;
      active++;
    }

    std::vector<ResultItem> results(numNs);
    int processed = 0;
    int nextPrint = 100;

    /* collect results and push more work */
    do {
      ResultItem r;
      MPI_Status status;
      MPI_Recv(&r, sizeof(ResultItem), MPI_BYTE, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
      results[slave_idx[status.MPI_SOURCE]] = r;
      processed++;
      if (processed >= nextPrint) {
        std::cout << "[progress] " << processed << "/" << numNs << " done ("
                  << (100.0 * processed / numNs) << "%)\n";
        nextPrint += 100;
      }

      if (offset < numNs) {
        WorkItem w;
        for (int j = 0; j < 8; ++j) w.N_limbs[j] = numbers[offset].limbs[j];
        MPI_Send(&w, sizeof(WorkItem), MPI_BYTE, status.MPI_SOURCE, DATA_TAG, MPI_COMM_WORLD);
        slave_idx[status.MPI_SOURCE] = offset;
        offset++;
      } else {
        active--;
      }
    } while (offset < numNs);

    /* collect remaining */
    for (int i = 0; i < active; i++) {
      ResultItem r;
      MPI_Status status;
      MPI_Recv(&r, sizeof(ResultItem), MPI_BYTE, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
      results[slave_idx[status.MPI_SOURCE]] = r;
      processed++;
    }
    std::cout << "[progress] " << processed << "/" << numNs << " done (100%)\n";

    /* send FINISH */
    for (int i = 1; i < nproc; i++) {
      MPI_Send(NULL, 0, MPI_BYTE, i, FINISH_TAG, MPI_COMM_WORLD);
    }

    const auto wall1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();

    /* print results */
    bool anyFound = false;
    std::cout << std::fixed << std::setprecision(3)
              << "batchSize=" << numNs
              << ",B1=" << B1
              << ",blockSize=" << blockSize
              << ",nproc=" << nproc
              << ",wall_ms=" << wall_ms
              << "\n";

    for (uint32_t n = 0; n < numNs; ++n) {
      const auto& r = results[n];
      bool found = false;
      std::string factorStr;
      if (r.status == 2u) {
        U256 f;
        for (int j = 0; j < 8; ++j) f.limbs[j] = r.factor_limbs[j];
        if (!u256_is_zero(f) && u256_cmp(f, u256_one()) > 0) {
          found = true;
          factorStr = u256_to_hex(f);
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
  /* ---------- Slaves (rank > 0) ---------- */
  else {
    MPI_Status status;
    do {
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == DATA_TAG) {
        WorkItem w;
        MPI_Recv(&w, sizeof(WorkItem), MPI_BYTE, 0, DATA_TAG, MPI_COMM_WORLD, &status);

        U256 N;
        for (int i = 0; i < 8; ++i) N.limbs[i] = w.N_limbs[i];

        /* host-side setup identical to standalone */
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

        /* build IO buffer for single number, single base */
        const int HEADER_WORDS = 8;
        const int CONST_WORDS  = 28;
        const int PP_OFFSET    = HEADER_WORDS + 1 * CONST_WORDS;
        const int OUT_OFFSET   = PP_OFFSET + static_cast<int>(primePowers.size());
        const int totalWords   = OUT_OFFSET + 1 * 1 * 12;

        std::vector<uint32_t> h_io(totalWords, 0);
        int off = 0;
        h_io[off++] = MAGIC;
        h_io[off++] = 2; // batched version
        h_io[off++] = static_cast<uint32_t>(primePowers.size());
        h_io[off++] = 1; // totalBases
        h_io[off++] = 2; // startBase
        h_io[off++] = 1; // numNs
        h_io[off++] = 0;
        h_io[off++] = 0;

        for (int i = 0; i < 8; ++i) h_io[off++] = reducedN.limbs[i];
        for (int i = 0; i < 8; ++i) h_io[off++] = R2.limbs[i];
        for (int i = 0; i < 8; ++i) h_io[off++] = montOne.limbs[i];
        h_io[off++] = n0inv32;
        h_io[off++] = 0; h_io[off++] = 0; h_io[off++] = 0;

        for (uint32_t pp : primePowers) h_io[off++] = pp;

        /* device memory */
        uint32_t* d_io = nullptr;
        CUDA_CHECK(cudaMalloc(&d_io, totalWords * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_io, h_io.data(), totalWords * sizeof(uint32_t), cudaMemcpyHostToDevice));

        /* launch */
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        pollard_pminus1_batched_kernel<<<1, blockSize>>>(d_io, 1);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaGetLastError());

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));

        CUDA_CHECK(cudaMemcpy(h_io.data(), d_io, totalWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_io));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));

        /* extract result */
        ResultItem r;
        r.status = h_io[OUT_OFFSET + 8];
        for (int i = 0; i < 8; ++i) r.factor_limbs[i] = h_io[OUT_OFFSET + i];
        r.base_u32 = h_io[OUT_OFFSET + 9];
        r.pad[0] = r.pad[1] = r.pad[2] = 0;

        MPI_Send(&r, sizeof(ResultItem), MPI_BYTE, 0, RESULT_TAG, MPI_COMM_WORLD);
      }
    } while (status.MPI_TAG != FINISH_TAG);
  }

  MPI_Finalize();
  return 0;
}
