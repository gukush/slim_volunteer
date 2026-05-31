#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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
/*  CUDA kernel (identical to standalone cuda_goldbach_verify.cu)     */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  Host helpers                                                      */
/* ------------------------------------------------------------------ */

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
    << "Usage: mpirun -np N " << argv0 << " [options]\n"
    << "  --start=N                 Start of even number range (default: 4)\n"
    << "  --end=N                   End of even number range (default: 10000)\n"
    << "  --numbers=N1,N2,...       Explicit comma-separated even numbers\n"
    << "  --batchRanges=S1-E1,S2-E2  Comma-separated start-end ranges\n"
    << "  --blockSize=N             CUDA block size (default: 128)\n"
    << "  --chunkSize=N             Numbers per MPI work message (default: 1024)\n"
    << "  --rangeSize=N             Alias for --chunkSize\n";
}

/* ------------------------------------------------------------------ */
/*  MPI master-slave pattern                                          */
/* ------------------------------------------------------------------ */

#define DATA_TAG    0
#define RESULT_TAG  1
#define FINISH_TAG  2

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int myrank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (nproc < 2) {
    if (!myrank) std::cerr << "Error: This Master-Slave design requires at least 2 ranks (mpirun -np 2 or higher)." << std::endl;
    MPI_Finalize();
    return 1;
  }

  uint32_t start = 4;
  uint32_t end = 10000;
  std::string numbers_arg;
  std::string batchRanges_arg;
  uint32_t blockSize = 128;
  uint32_t chunkSize = 1024;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") { usage(argv[0]); MPI_Finalize(); return 0; }
    auto eq = arg.find('=');
    std::string key = (eq == std::string::npos) ? arg : arg.substr(0, eq);
    std::string val = (eq == std::string::npos) ? std::string() : arg.substr(eq + 1);
    if (key == "--start") start = parse_u32(val);
    else if (key == "--end") end = parse_u32(val);
    else if (key == "--numbers") numbers_arg = val;
    else if (key == "--batchRanges") batchRanges_arg = val;
    else if (key == "--blockSize") blockSize = parse_u32(val);
    else if (key == "--chunkSize" || key == "--rangeSize") chunkSize = parse_u32(val);
  }

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

  /* ---------- build number list on every rank (cheap) ---------- */
  std::vector<uint32_t> numbers;
  if (!myrank) {
    if (!batchRanges_arg.empty()) {
      size_t pos = 0;
      while (pos < batchRanges_arg.size()) {
        size_t comma = batchRanges_arg.find(',', pos);
        std::string token = batchRanges_arg.substr(pos, comma - pos);
        if (!token.empty()) {
          size_t dash = token.find('-');
          if (dash == std::string::npos) {
            std::cerr << "Invalid range format: " << token << std::endl;
            MPI_Finalize();
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
  }

  /* Broadcast count so slaves know whether to participate */
  uint32_t count = static_cast<uint32_t>(numbers.size());
  MPI_Bcast(&count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (count == 0) {
    if (!myrank) std::cerr << "No numbers to check" << std::endl;
    MPI_Finalize();
    return 2;
  }

  /* ---------- build small primes on every node ---------- */
  std::vector<uint32_t> smallPrimes = build_small_primes(65535);
  uint32_t primeCount = static_cast<uint32_t>(smallPrimes.size());

  /* ---------- Master (rank 0) ---------- */
  if (!myrank) {
    uint32_t totalChunks = (count + chunkSize - 1) / chunkSize;
    uint32_t nextOffset = 0, sent = 0, received = 0;
    std::vector<uint32_t> worker_start(nproc, 0);
    std::vector<uint32_t> worker_count(nproc, 0);
    std::vector<uint8_t> worker_finished(nproc, 0);

    const auto wall0 = std::chrono::steady_clock::now();

    auto send_chunk = [&](int dst) {
      if (nextOffset >= count) return false;
      uint32_t cnt_u32 = std::min(chunkSize, count - nextOffset);
      int cnt = static_cast<int>(cnt_u32);
      worker_start[dst] = nextOffset;
      worker_count[dst] = cnt_u32;
      MPI_Send(&numbers[nextOffset], cnt, MPI_UNSIGNED, dst, DATA_TAG, MPI_COMM_WORLD);
      nextOffset += cnt_u32;
      sent++;
      return true;
    };

    for (int dst = 1; dst < nproc; ++dst) {
      if (!send_chunk(dst)) {
        MPI_Send(NULL, 0, MPI_UNSIGNED, dst, FINISH_TAG, MPI_COMM_WORLD);
        worker_finished[dst] = 1;
      }
    }

    /* collect results and keep workers fed */
    uint32_t global_result[8] = {};
    while (received < totalChunks) {
      uint32_t slave_result[8];
      MPI_Status status;
      MPI_Recv(slave_result, 8, MPI_UNSIGNED, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
      int src = status.MPI_SOURCE;
      received++;

      if (slave_result[0] != 0 && global_result[0] == 0) {
        global_result[0] = 1;
        global_result[1] = worker_start[src] + slave_result[1];
        global_result[2] = slave_result[2];
        global_result[3] = slave_result[3];
        global_result[4] = slave_result[4];
      }

      if (!send_chunk(src)) {
        MPI_Send(NULL, 0, MPI_UNSIGNED, src, FINISH_TAG, MPI_COMM_WORLD);
        worker_finished[src] = 1;
      }
    }

    /* send FINISH */
    for (int i = 1; i < nproc; i++) {
      if (worker_finished[i]) continue;
      MPI_Send(NULL, 0, MPI_UNSIGNED, i, FINISH_TAG, MPI_COMM_WORLD);
      worker_finished[i] = 1;
    }

    const auto wall1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();

    bool valid = (global_result[0] == 0u);
    uint32_t failedIdx = global_result[1];
    uint32_t failedN = global_result[2];
    uint32_t reason = global_result[4];

    std::cout << std::fixed << std::setprecision(3)
              << "count=" << count
              << ",first=" << numbers.front()
              << ",last=" << numbers.back()
              << ",blockSize=" << blockSize
              << ",chunkSize=" << chunkSize
              << ",chunks=" << totalChunks
              << ",sent=" << sent
              << ",nproc=" << nproc
              << ",wall_ms=" << wall_ms
              << ",valid=" << (valid ? "true" : "false");
    if (!valid) {
      std::cout << ",failedIndex=" << failedIdx
                << ",failedNumber=" << failedN
                << ",reason=" << (reason == 1u ? "invalid-input" : "no-prime-pair");
    }
    std::cout << "\n";
  }
  /* ---------- Slaves (rank > 0) ---------- */
  else {
    uint32_t* d_primes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_primes, primeCount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_primes, smallPrimes.data(), primeCount * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t* h_chunk = nullptr;
    uint32_t* d_numbers = nullptr;
    uint32_t* d_result = nullptr;
    uint32_t current_capacity = 0;
    CUDA_CHECK(cudaMalloc(&d_result, 8 * sizeof(uint32_t)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    MPI_Status status;
    while (true) {
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == DATA_TAG) {
        int recvCount = 0;
        MPI_Get_count(&status, MPI_UNSIGNED, &recvCount);

        // Dynamically resize if the incoming chunk is larger than our buffer
        if (recvCount > (int)current_capacity) {
          if (h_chunk) free(h_chunk);
          if (d_numbers) CUDA_CHECK(cudaFree(d_numbers));
          h_chunk = (uint32_t*)malloc(recvCount * sizeof(uint32_t));
          CUDA_CHECK(cudaMalloc(&d_numbers, recvCount * sizeof(uint32_t)));
          current_capacity = recvCount;
        }

        MPI_Recv(h_chunk, recvCount, MPI_UNSIGNED, 0, DATA_TAG, MPI_COMM_WORLD, &status);

        uint32_t h_result[8] = {};
        h_result[5] = static_cast<uint32_t>(recvCount);
        h_result[6] = primeCount;
        CUDA_CHECK(cudaMemcpy(d_result, h_result, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_numbers, h_chunk, recvCount * sizeof(uint32_t), cudaMemcpyHostToDevice));

        int grid = (recvCount + blockSize - 1) / blockSize;
        CUDA_CHECK(cudaEventRecord(ev0));
        goldbach_verify_kernel<<<grid, blockSize>>>(d_numbers, d_result, d_primes, recvCount, primeCount);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));

        CUDA_CHECK(cudaMemcpy(h_result, d_result, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        MPI_Send(h_result, 8, MPI_UNSIGNED, 0, RESULT_TAG, MPI_COMM_WORLD);

      } else if (status.MPI_TAG == FINISH_TAG) {
        // Consume the finish message so the queue is clean!
        MPI_Recv(NULL, 0, MPI_UNSIGNED, 0, FINISH_TAG, MPI_COMM_WORLD, &status);
        break;
      }
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    if (h_chunk) free(h_chunk);
    if (d_numbers) CUDA_CHECK(cudaFree(d_numbers));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_primes));
  }

  MPI_Finalize();
  return 0;
}
