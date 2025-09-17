// out_of_core_matmul_chunked.cu
// File format: row-major float32 binary (no header).
//   A is N x K, B is K x M, C is N x M.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 -o ooc_matmul_chunked out_of_core_matmul_chunked.cu
//
// Usage:
//   ./ooc_matmul_chunked --N 40000 --K 50000 --M 30000 \
//     --A A.bin --B B.bin --C C.bin \
//     --chunk-n 2048 --chunk-m 2048 --chunk-k 1024
//


#define _FILE_OFFSET_BITS 64

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while(0)
#endif

// --------------------- Fixed 16x16 kernel ---------------------
constexpr int TILE = 16;

// Kernel computes C_chunk += A_chunk * B_chunk
// A: CN x CK, B: CK x CM, C: CN x CM
__global__ void gemm_chunk_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int CN, int CM, int CK)
{
  extern __shared__ float smem[];
  float* As = smem;                  // TILE*TILE
  float* Bs = smem + TILE*TILE;      // TILE*TILE

  int ty  = threadIdx.y;
  int tx  = threadIdx.x;
  int row = blockIdx.y * TILE + ty;  // 0..CN-1
  int col = blockIdx.x * TILE + tx;  // 0..CM-1

  float acc = 0.0f;

  // Loop over K dimension in micro-tiles of size TILE
  for (int kk = 0; kk < CK; kk += TILE) {
    // Load A micro-tile [row, kk..kk+TILE)
    int a_r = row;
    int a_c = kk + tx;
    if (a_r < CN && a_c < CK)
      As[ty * TILE + tx] = A[a_r * CK + a_c];
    else
      As[ty * TILE + tx] = 0.0f;

    // Load B micro-tile [kk..kk+TILE, col]
    int b_r = kk + ty;
    int b_c = col;
    if (b_r < CK && b_c < CM)
      Bs[ty * TILE + tx] = B[b_r * CM + b_c];
    else
      Bs[ty * TILE + tx] = 0.0f;

    __syncthreads();

    #pragma unroll
    for (int t = 0; t < TILE; ++t) {
      acc += As[ty * TILE + t] * Bs[t * TILE + tx];
    }
    __syncthreads();
  }

  if (row < CN && col < CM) {
    C[row * CM + col] += acc; // chunk accumulation across K
  }
}

// --------------------- Disk I/O helpers ---------------------

// Read block of A: rows [i0, i0+cn) and cols [k0, k0+ck)
// A file holds N x K floats (row-major)
static void read_block_A(FILE* fA, float* hA, int N, int K,
                         int i0, int k0, int cn, int ck)
{
  for (int r = 0; r < cn; ++r) {
    int64_t off = (int64_t)(i0 + r) * (int64_t)K + (int64_t)k0;
    if (std::fseek(fA, off * (int64_t)sizeof(float), SEEK_SET) != 0) {
      perror("fseek(A)"); std::exit(1);
    }
    size_t got = std::fread(hA + (size_t)r * (size_t)ck, sizeof(float), (size_t)ck, fA);
    if (got != (size_t)ck) {
      fprintf(stderr, "Short read on A block (r=%d got=%zu ck=%d)\n", r, got, ck);
      std::exit(1);
    }
  }
}

// Read block of B: rows [k0, k0+ck) and cols [j0, j0+cm)
// B file holds K x M floats (row-major)
static void read_block_B(FILE* fB, float* hB, int K, int M,
                         int k0, int j0, int ck, int cm)
{
  for (int r = 0; r < ck; ++r) {
    int64_t off = (int64_t)(k0 + r) * (int64_t)M + (int64_t)j0;
    if (std::fseek(fB, off * (int64_t)sizeof(float), SEEK_SET) != 0) {
      perror("fseek(B)"); std::exit(1);
    }
    size_t got = std::fread(hB + (size_t)r * (size_t)cm, sizeof(float), (size_t)cm, fB);
    if (got != (size_t)cm) {
      fprintf(stderr, "Short read on B block (r=%d got=%zu cm=%d)\n", r, got, cm);
      std::exit(1);
    }
  }
}

// Write block of C: rows [i0, i0+cn) and cols [j0, j0+cm)
// C file holds N x M floats (row-major)
static void write_block_C(FILE* fC, const float* hC, int N, int M,
                          int i0, int j0, int cn, int cm)
{
  for (int r = 0; r < cn; ++r) {
    int64_t off = (int64_t)(i0 + r) * (int64_t)M + (int64_t)j0;
    if (std::fseek(fC, off * (int64_t)sizeof(float), SEEK_SET) != 0) {
      perror("fseek(C)"); std::exit(1);
    }
    size_t put = std::fwrite(hC + (size_t)r * (size_t)cm, sizeof(float), (size_t)cm, fC);
    if (put != (size_t)cm) {
      fprintf(stderr, "Short write on C block (r=%d put=%zu cm=%d)\n", r, put, cm);
      std::exit(1);
    }
  }
}

// Create or truncate C to N*M floats (optionally sparse preallocation)
static void ensure_c_file_size(const std::string& path, int N, int M)
{
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  // Create/truncate and (optionally) pre-extend via seek+write a zero
  FILE* f = std::fopen(path.c_str(), "wb+");
  if (!f) { perror(("open C: " + path).c_str()); std::exit(1); }
  int64_t total = (int64_t)N * (int64_t)M;
  if (total > 0) {
    if (std::fseek(f, (total - 1) * (int64_t)sizeof(float), SEEK_SET) != 0) {
      perror("fseek pre-extend C"); std::exit(1);
    }
    float zero = 0.0f;
    size_t put = std::fwrite(&zero, sizeof(float), 1, f);
    if (put != 1) { perror("pre-extend write C"); std::exit(1); }
  }
  std::fclose(f);
}

// --------------------- CLI parsing ---------------------
struct Args {
  int N=0, K=0, M=0;
  std::string A_path, B_path, C_path;
  int CN=0, CM=0, CK=0; // chunk sizes
};

static void usage_and_exit(const char* prog) {
  std::cerr <<
    "Usage:\n  " << prog << " --N N --K K --M M --A A.bin --B B.bin --C C.bin \\\n"
    "         --chunk-n CN --chunk-m CM --chunk-k CK\n\n"
    "Notes:\n"
    "  - Fixed kernel tile = 16x16 (threads per block = 16x16)\n"
    "  - I/O chunks are CN x CK (A) and CK x CM (B), producing CN x CM of C per chunk.\n";
  std::exit(1);
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;i++) {
    std::string k = argv[i];
    auto need = [&](int m){ if (i+m>=argc) usage_and_exit(argv[0]); };

    if      (k=="--N")       { need(1); a.N  = std::stoi(argv[++i]); }
    else if (k=="--K")       { need(1); a.K  = std::stoi(argv[++i]); }
    else if (k=="--M")       { need(1); a.M  = std::stoi(argv[++i]); }
    else if (k=="--A")       { need(1); a.A_path = argv[++i]; }
    else if (k=="--B")       { need(1); a.B_path = argv[++i]; }
    else if (k=="--C")       { need(1); a.C_path = argv[++i]; }
    else if (k=="--chunk-n") { need(1); a.CN = std::stoi(argv[++i]); }
    else if (k=="--chunk-m") { need(1); a.CM = std::stoi(argv[++i]); }
    else if (k=="--chunk-k") { need(1); a.CK = std::stoi(argv[++i]); }
    else { usage_and_exit(argv[0]); }
  }
  if (a.N<=0 || a.K<=0 || a.M<=0 || a.A_path.empty() || a.B_path.empty() || a.C_path.empty()
      || a.CN<=0 || a.CM<=0 || a.CK<=0) usage_and_exit(argv[0]);

  return a;
}

// --------------------- Main ---------------------
int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // Open input files (must exist)
  FILE* fA = std::fopen(args.A_path.c_str(), "rb");
  if (!fA) { perror(("open A: " + args.A_path).c_str()); return 1; }
  FILE* fB = std::fopen(args.B_path.c_str(), "rb");
  if (!fB) { perror(("open B: " + args.B_path).c_str()); return 1; }

  // Ensure/prepare output file
  ensure_c_file_size(args.C_path, args.N, args.M);
  FILE* fC = std::fopen(args.C_path.c_str(), "rb+");
  if (!fC) { perror(("open C: " + args.C_path).c_str()); return 1; }

  const int N  = args.N;
  const int K  = args.K;
  const int M  = args.M;
  const int CN = args.CN;
  const int CM = args.CM;
  const int CK = args.CK;

  // Host chunk buffers
  std::vector<float> hA((size_t)CN * (size_t)CK);
  std::vector<float> hB((size_t)CK * (size_t)CM);
  std::vector<float> hC((size_t)CN * (size_t)CM);

  // Device chunk buffers
  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CHECK_CUDA(cudaMalloc(&dA, (size_t)CN * (size_t)CK * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, (size_t)CK * (size_t)CM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, (size_t)CN * (size_t)CM * sizeof(float)));

  // Summary
  std::cout << "Out-of-core GEMM (tile 16x16, chunked I/O)\n"
            << "A: " << args.A_path << "  (" << N << "x" << K << ")\n"
            << "B: " << args.B_path << "  (" << K << "x" << M << ")\n"
            << "C: " << args.C_path << "  (" << N << "x" << M << ")\n"
            << "Chunks: CN=" << CN << "  CM=" << CM << "  CK=" << CK << "\n"
            << "Press ENTER to start..." << std::flush;
  std::string line; std::getline(std::cin, line);

  // For each C chunk (i0,j0), accumulate over k0
  dim3 block(TILE, TILE);
  size_t smem = 2 * (size_t)TILE * TILE * sizeof(float);

  for (int i0 = 0; i0 < N; i0 += CN) {
    int cn = std::min(CN, N - i0);
    for (int j0 = 0; j0 < M; j0 += CM) {
      int cm = std::min(CM, M - j0);

      // Zero C chunk
      std::fill(hC.begin(), hC.begin() + (size_t)cn * cm, 0.0f);
      CHECK_CUDA(cudaMemset(dC, 0, (size_t)cn * cm * sizeof(float)));

      for (int k0 = 0; k0 < K; k0 += CK) {
        int ck = std::min(CK, K - k0);

        // Read A (cn x ck) from (i0,k0)
        read_block_A(fA, hA.data(), N, K, i0, k0, cn, ck);
        // Read B (ck x cm) from (k0,j0)
        read_block_B(fB, hB.data(), K, M, k0, j0, ck, cm);

        // Push chunks to device
        CHECK_CUDA(cudaMemcpy(dA, hA.data(), (size_t)cn * ck * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), (size_t)ck * cm * sizeof(float), cudaMemcpyHostToDevice));

        // Launch many 16x16 blocks over this chunk
        dim3 grid( (cm + TILE - 1) / TILE, (cn + TILE - 1) / TILE );
        gemm_chunk_kernel<<<grid, block, smem>>>(dA, dB, dC, cn, cm, ck);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
      }

      // Pull C chunk back and write it to disk at (i0,j0)
      CHECK_CUDA(cudaMemcpy(hC.data(), dC, (size_t)cn * cm * sizeof(float), cudaMemcpyDeviceToHost));
      write_block_C(fC, hC.data(), N, M, i0, j0, cn, cm);
    }
  }

  // Cleanup
  std::fclose(fA);
  std::fclose(fB);
  std::fclose(fC);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  std::cout << "Done.\n";
  return 0;
}
