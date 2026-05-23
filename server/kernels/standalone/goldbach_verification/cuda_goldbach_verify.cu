#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

__device__ static inline bool is_prime_dev(uint32_t n, const uint32_t* small_primes, uint32_t prime_count) {
  if (n < 2u) return false;
  for (uint32_t i = 0; i < prime_count; ++i) {
    uint32_t p = small_primes[i];
    if (p > n / p) break;
    if ((n % p) == 0u) return n == p;
  }
  return true;
}

__device__ static inline uint32_t find_goldbach_witness_dev(uint32_t n, const uint32_t* small_primes, uint32_t prime_count) {
  if (n == 4u) return 2u;
  for (uint32_t i = 1; i < prime_count; ++i) {
    uint32_t p = small_primes[i];
    if (p > n / 2u) break;
    if (is_prime_dev(n - p, small_primes, prime_count)) return p;
  }
  return 0u;
}

__global__ void goldbach_verify_kernel(
    const uint32_t* numbers,
    uint32_t* result,
    const uint32_t* small_primes,
    uint32_t count,
    uint32_t prime_count)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  if (atomicAdd(&result[0], 0u) != 0u) return;

  uint32_t n = numbers[idx];
  if (n <= 2u || (n & 1u) != 0u) {
    if (atomicCAS(&result[0], 0u, 1u) == 0u) {
      result[1] = idx;
      result[2] = n;
      result[3] = 0u;
      result[4] = 1u; // invalid-input
    }
    return;
  }

  uint32_t witness = find_goldbach_witness_dev(n, small_primes, prime_count);
  if (witness == 0u) {
    if (atomicCAS(&result[0], 0u, 1u) == 0u) {
      result[1] = idx;
      result[2] = n;
      result[3] = 0u;
      result[4] = 2u; // no-prime-pair
    }
    return;
  }
}

static uint32_t parse_u32(const std::string& s) {
  return static_cast<uint32_t>(std::stoul(s, nullptr, 0));
}

static std::vector<uint32_t> build_small_primes(uint32_t limit) {
  std::vector<uint8_t> composite(limit + 1, 0);
  std::vector<uint32_t> primes;
  for (uint32_t n = 2; n <= limit; ++n) {
    if (composite[n]) continue;
    primes.push_back(n);
    if (n <= limit / n) {
      for (uint32_t m = n * n; m <= limit; m += n) composite[m] = 1;
    }
  }
  return primes;
}

static void usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [options]\n"
    << "  --start=N                 Start of even number range (default: 4)\n"
    << "  --end=N                   End of even number range (default: 10000)\n"
    << "  --numbers=N1,N2,...       Explicit comma-separated even numbers\n"
    << "  --batchRanges=S1-E1,S2-E2  Comma-separated start-end ranges\n"
    << "  --blockSize=N             CUDA block size (default: 128)\n";
}

int main(int argc, char** argv) {
  uint32_t start = 4;
  uint32_t end = 10000;
  std::string numbers_arg;
  std::string batchRanges_arg;
  uint32_t blockSize = 128;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
    auto eq = arg.find('=');
    std::string key = (eq == std::string::npos) ? arg : arg.substr(0, eq);
    std::string val = (eq == std::string::npos) ? std::string() : arg.substr(eq + 1);
    if (key == "--start") start = parse_u32(val);
    else if (key == "--end") end = parse_u32(val);
    else if (key == "--numbers") numbers_arg = val;
    else if (key == "--batchRanges") batchRanges_arg = val;
    else if (key == "--blockSize") blockSize = parse_u32(val);
    else { std::cerr << "Unknown option: " << arg << "\n"; usage(argv[0]); return 2; }
  }

  if (blockSize == 0 || blockSize > 1024) {
    std::cerr << "blockSize must be in [1, 1024]" << std::endl;
    return 2;
  }

  std::vector<uint32_t> numbers;

  if (!batchRanges_arg.empty()) {
    size_t pos = 0;
    while (pos < batchRanges_arg.size()) {
      size_t comma = batchRanges_arg.find(',', pos);
      std::string token = batchRanges_arg.substr(pos, comma - pos);
      if (!token.empty()) {
        size_t dash = token.find('-');
        if (dash == std::string::npos) {
          std::cerr << "Invalid range format: " << token << " (expected START-END)" << std::endl;
          return 2;
        }
        uint32_t rstart = parse_u32(token.substr(0, dash));
        uint32_t rend = parse_u32(token.substr(dash + 1));
        uint32_t cur = (rstart % 2 == 0) ? rstart : rstart + 1;
        if (cur < 4) cur = 4;
        for (; cur <= rend; cur += 2) numbers.push_back(cur);
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  } else if (!numbers_arg.empty()) {
    size_t pos = 0;
    while (pos < numbers_arg.size()) {
      size_t comma = numbers_arg.find(',', pos);
      std::string token = numbers_arg.substr(pos, comma - pos);
      if (!token.empty()) numbers.push_back(parse_u32(token));
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  } else {
    uint32_t cur = (start % 2 == 0) ? start : start + 1;
    if (cur < 4) cur = 4;
    for (; cur <= end; cur += 2) numbers.push_back(cur);
  }

  if (numbers.empty()) {
    std::cerr << "No numbers to check" << std::endl;
    return 2;
  }

  std::vector<uint32_t> smallPrimes = build_small_primes(65535);
  uint32_t primeCount = static_cast<uint32_t>(smallPrimes.size());
  uint32_t count = static_cast<uint32_t>(numbers.size());

  const auto total_wall0 = std::chrono::steady_clock::now();

  uint32_t* d_numbers = nullptr;
  uint32_t* d_result = nullptr;
  uint32_t* d_primes = nullptr;

  CUDA_CHECK(cudaMalloc(&d_numbers, numbers.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_numbers, numbers.data(), numbers.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_primes, smallPrimes.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_primes, smallPrimes.data(), smallPrimes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_result, 8 * sizeof(uint32_t)));
  uint32_t h_result[8] = {};
  h_result[5] = count;
  h_result[6] = primeCount;
  CUDA_CHECK(cudaMemcpy(d_result, h_result, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  const auto wall0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(ev0));
  uint32_t grid = (count + blockSize - 1) / blockSize;
  goldbach_verify_kernel<<<grid, blockSize>>>(d_numbers, d_result, d_primes, count, primeCount);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  const auto wall1 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGetLastError());

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev0, ev1));

  CUDA_CHECK(cudaMemcpy(h_result, d_result, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  const auto total_wall1 = std::chrono::steady_clock::now();

  CUDA_CHECK(cudaFree(d_numbers));
  CUDA_CHECK(cudaFree(d_result));
  CUDA_CHECK(cudaFree(d_primes));
  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));

  double old_wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  double wall_ms = std::chrono::duration<double, std::milli>(total_wall1 - total_wall0).count();

  bool valid = (h_result[0] == 0u);
  uint32_t failedIdx = h_result[1];
  uint32_t failedN = h_result[2];
  uint32_t reason = h_result[4];

  std::cout << std::fixed << std::setprecision(3)
            << "count=" << count
            << ",first=" << numbers.front()
            << ",last=" << numbers.back()
            << ",blockSize=" << blockSize
            << ",grid=" << grid
            << ",old_wall_ms=" << old_wall_ms
            << ",wall_ms=" << wall_ms
            << ",kernel_ms=" << kernel_ms
            << ",valid=" << (valid ? "true" : "false");
  if (!valid) {
    std::cout << ",failedIndex=" << failedIdx
              << ",failedNumber=" << failedN
              << ",reason=" << (reason == 1u ? "invalid-input" : "no-prime-pair");
  }
  std::cout << "\n";

  return valid ? 0 : 1;
}
