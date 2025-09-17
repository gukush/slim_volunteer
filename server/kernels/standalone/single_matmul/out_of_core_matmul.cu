// ooc_matmul_chunked_csv.cu
// Out-of-core GEMM with fixed 16x16 kernel tiles and user-defined I/O chunk sizes,
// with detailed timing and CSV export, and optional cuBLAS backend.
//
// Build (with cuBLAS):
//   nvcc -O3 -std=c++17 -arch=sm_80 -lcublas -o matmul ooc_matmul_chunked_csv.cu
//
// Example:
//   ./matmul --N 10240 --K 10240 --M 10240 \
//     --A ./A.bin --B ./B.bin --C ./C.bin \
//     --chunk-n 2048 --chunk-m 2048 --chunk-k 2048 \
//     --backend cublas --csv
//
// Files are row-major float32 with no header:
//   A: N x K, B: K x M, C: N x M
//
// NOTE: Program does NOT generate inputs by default. Add your own generation step if needed.

#define _FILE_OFFSET_BITS 64

#include <cuda_runtime.h>
#include <cublas_v2.h>

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
#include <numeric>
#include <filesystem>
#include <chrono>
#include <ctime>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while(0)
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
  cublasStatus_t st__ = (call); \
  if (st__ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, (int)st__); \
    std::exit(1); \
  } \
} while(0)
#endif

// --------------------- Fixed 16x16 custom kernel ---------------------
constexpr int TILE = 16;

// C_chunk += A_chunk * B_chunk
// A: CN x CK (row-major), B: CK x CM (row-major), C: CN x CM (row-major)
__global__ void gemm_chunk_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int CN, int CM, int CK)
{
  extern __shared__ float smem[];
  float* As = smem;                  // TILE * TILE
  float* Bs = smem + TILE * TILE;    // TILE * TILE

  int ty  = threadIdx.y;
  int tx  = threadIdx.x;
  int row = blockIdx.y * TILE + ty;  // [0, CN)
  int col = blockIdx.x * TILE + tx;  // [0, CM)

  float acc = 0.0f;

  for (int kk = 0; kk < CK; kk += TILE) {
    int a_r = row;
    int a_c = kk + tx;
    if (a_r < CN && a_c < CK)
      As[ty * TILE + tx] = A[a_r * CK + a_c];
    else
      As[ty * TILE + tx] = 0.0f;

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
    C[row * CM + col] += acc;
  }
}

// --------------------- Timing utils & CSV ---------------------
struct MatmulTimingResults {
  std::string timestamp;
  int N, K, M;
  int chunk_n, chunk_m, chunk_k;
  int tile_size;
  std::string backend;
  std::string gpu_name;
  int total_chunks;                // number of (i0,j0,k0) iterations
  double wall_time_s;
  double total_file_io_ms;
  double total_h2d_ms;
  double total_kernel_ms;
  double total_d2h_ms;
  double total_cpu_acc_ms;         // here we keep accumulation on GPU, so ~0
  double avg_chunk_time_ms;
  double effective_gflops;
  double memory_bandwidth_gb_s;
  double total_memory_transferred_gb;
};

std::string timestamp_now() {
  auto now = std::chrono::system_clock::now();
  auto tt  = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
#ifdef _WIN32
  std::tm tm; localtime_s(&tm, &tt);
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
#else
  std::tm tm; localtime_r(&tt, &tm);
  ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
#endif
  return ss.str();
}

std::string generate_csv_filename(const std::string& custom = "") {
  if (!custom.empty()) return custom;
  auto now = std::chrono::system_clock::now();
  auto tt  = std::chrono::system_clock::to_time_t(now);
#ifdef _WIN32
  std::tm tm; localtime_s(&tm, &tt);
#else
  std::tm tm; localtime_r(&tt, &tm);
#endif
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % std::chrono::milliseconds(1000);
  std::ostringstream ss;
  ss << "ooc_matmul_timing_"
     << (tm.tm_year + 1900)
     << std::setw(2) << std::setfill('0') << (tm.tm_mon + 1)
     << std::setw(2) << std::setfill('0') << tm.tm_mday
     << "_"
     << std::setw(2) << std::setfill('0') << tm.tm_hour
     << std::setw(2) << std::setfill('0') << tm.tm_min
     << std::setw(2) << std::setfill('0') << tm.tm_sec
     << "_" << std::setw(3) << std::setfill('0') << ms.count()
     << ".csv";
  return ss.str();
}

void export_timing_to_csv(const MatmulTimingResults& r, const std::string& filename) {
  bool exists = std::ifstream(filename).good();
  std::ofstream csv(filename, std::ios::app);
  if (!csv) {
    std::cerr << "Warning: cannot open CSV: " << filename << "\n";
    return;
  }
  if (!exists) {
    csv << "timestamp,N,K,M,chunk_n,chunk_m,chunk_k,tile_size,backend,gpu_name,total_chunks,"
           "wall_time_s,total_file_io_ms,total_h2d_ms,total_kernel_ms,total_d2h_ms,total_cpu_acc_ms,"
           "avg_chunk_time_ms,effective_gflops,memory_bandwidth_gb_s,total_memory_transferred_gb\n";
  }
  csv << r.timestamp << ","
      << r.N << "," << r.K << "," << r.M << ","
      << r.chunk_n << "," << r.chunk_m << "," << r.chunk_k << ","
      << r.tile_size << "," << r.backend << ","
      << "\"" << r.gpu_name << "\"," << r.total_chunks << ","
      << std::fixed << std::setprecision(3)
      << r.wall_time_s << "," << r.total_file_io_ms << ","
      << r.total_h2d_ms << "," << r.total_kernel_ms << ","
      << r.total_d2h_ms << "," << r.total_cpu_acc_ms << ","
      << r.avg_chunk_time_ms << ","
      << std::setprecision(2) << r.effective_gflops << ","
      << r.memory_bandwidth_gb_s << ","
      << std::setprecision(3) << r.total_memory_transferred_gb << "\n";
  csv.close();
  std::cout << "Timing results exported to: " << filename << "\n";
}

class CudaTimingContext {
  cudaEvent_t start_, stop_;
public:
  CudaTimingContext() { CHECK_CUDA(cudaEventCreate(&start_)); CHECK_CUDA(cudaEventCreate(&stop_)); }
  ~CudaTimingContext(){ cudaEventDestroy(start_); cudaEventDestroy(stop_); }
  void start(){ CHECK_CUDA(cudaEventRecord(start_)); }
  float stop(){ CHECK_CUDA(cudaEventRecord(stop_)); CHECK_CUDA(cudaEventSynchronize(stop_)); float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_)); return ms; }
};

// --------------------- Disk I/O helpers (timed) ---------------------
static double read_block_A(FILE* fA, float* hA, int N, int K,
                           int i0, int k0, int cn, int ck)
{
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r=0;r<cn;++r){
    int64_t off = (int64_t)(i0 + r)*(int64_t)K + (int64_t)k0;
    if (std::fseek(fA, off * (int64_t)sizeof(float), SEEK_SET)!=0){ perror("fseek(A)"); std::exit(1); }
    size_t got = std::fread(hA + (size_t)r*(size_t)ck, sizeof(float), (size_t)ck, fA);
    if (got != (size_t)ck){ fprintf(stderr,"Short read A (r=%d got=%zu ck=%d)\n",r,got,ck); std::exit(1); }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double read_block_B(FILE* fB, float* hB, int K, int M,
                           int k0, int j0, int ck, int cm)
{
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r=0;r<ck;++r){
    int64_t off = (int64_t)(k0 + r)*(int64_t)M + (int64_t)j0;
    if (std::fseek(fB, off * (int64_t)sizeof(float), SEEK_SET)!=0){ perror("fseek(B)"); std::exit(1); }
    size_t got = std::fread(hB + (size_t)r*(size_t)cm, sizeof(float), (size_t)cm, fB);
    if (got != (size_t)cm){ fprintf(stderr,"Short read B (r=%d got=%zu cm=%d)\n",r,got,cm); std::exit(1); }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double write_block_C(FILE* fC, const float* hC, int N, int M,
                            int i0, int j0, int cn, int cm)
{
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r=0;r<cn;++r){
    int64_t off = (int64_t)(i0 + r)*(int64_t)M + (int64_t)j0;
    if (std::fseek(fC, off * (int64_t)sizeof(float), SEEK_SET)!=0){ perror("fseek(C)"); std::exit(1); }
    size_t put = std::fwrite(hC + (size_t)r*(size_t)cm, sizeof(float), (size_t)cm, fC);
    if (put != (size_t)cm){ fprintf(stderr,"Short write C (r=%d put=%zu cm=%d)\n",r,put,cm); std::exit(1); }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void ensure_c_file_size(const std::string& path, int N, int M){
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  FILE* f = std::fopen(path.c_str(), "wb+");
  if (!f){ perror(("open C: " + path).c_str()); std::exit(1); }
  int64_t total = (int64_t)N * (int64_t)M;
  if (total > 0){
    if (std::fseek(f, (total - 1) * (int64_t)sizeof(float), SEEK_SET)!=0){ perror("fseek pre-extend C"); std::exit(1); }
    float zero = 0.0f;
    if (std::fwrite(&zero, sizeof(float), 1, f) != 1){ perror("pre-extend write C"); std::exit(1); }
  }
  std::fclose(f);
}

// --------------------- CLI ---------------------
struct Args {
  int N=0, K=0, M=0;
  std::string A_path, B_path, C_path;
  int CN=0, CM=0, CK=0;
  std::string backend = "custom";           // custom|cublas
  bool csv=false;
  std::string csv_file;
};

static void usage_and_exit(const char* prog){
  std::cerr <<
    "Usage:\n  " << prog << " --N N --K K --M M --A A.bin --B B.bin --C C.bin \\\n"
    "         --chunk-n CN --chunk-m CM --chunk-k CK [--backend custom|cublas] [--csv] [--csv-file name.csv]\n\n"
    "Notes:\n"
    "  - Fixed kernel tile = 16x16 (threads per block = 16x16)\n"
    "  - I/O chunks are CN x CK (A) and CK x CM (B), producing CN x CM of C per chunk.\n";
  std::exit(1);
}

static Args parse_args(int argc, char** argv){
  Args a;
  for (int i=1;i<argc;i++){
    std::string k=argv[i];
    auto need=[&](int m){ if (i+m>=argc) usage_and_exit(argv[0]); };
    if      (k=="--N"){ need(1); a.N=std::stoi(argv[++i]); }
    else if (k=="--K"){ need(1); a.K=std::stoi(argv[++i]); }
    else if (k=="--M"){ need(1); a.M=std::stoi(argv[++i]); }
    else if (k=="--A"){ need(1); a.A_path=argv[++i]; }
    else if (k=="--B"){ need(1); a.B_path=argv[++i]; }
    else if (k=="--C"){ need(1); a.C_path=argv[++i]; }
    else if (k=="--chunk-n"){ need(1); a.CN=std::stoi(argv[++i]); }
    else if (k=="--chunk-m"){ need(1); a.CM=std::stoi(argv[++i]); }
    else if (k=="--chunk-k"){ need(1); a.CK=std::stoi(argv[++i]); }
    else if (k=="--backend"){ need(1); a.backend=argv[++i]; }
    else if (k=="--csv"){ a.csv=true; }
    else if (k.rfind("--csv-file=",0)==0){ a.csv=true; a.csv_file=k.substr(11); }
    else { usage_and_exit(argv[0]); }
  }
  if (a.N<=0||a.K<=0||a.M<=0||a.A_path.empty()||a.B_path.empty()||a.C_path.empty()||a.CN<=0||a.CM<=0||a.CK<=0)
    usage_and_exit(argv[0]);
  if (a.backend!="custom" && a.backend!="cublas") usage_and_exit(argv[0]);
  return a;
}

// --------------------- Main ---------------------
int main(int argc, char** argv){
  Args args = parse_args(argc, argv);

  // Open inputs
  FILE* fA = std::fopen(args.A_path.c_str(), "rb");
  if (!fA){ perror(("open A: " + args.A_path).c_str()); return 1; }
  FILE* fB = std::fopen(args.B_path.c_str(), "rb");
  if (!fB){ perror(("open B: " + args.B_path).c_str()); return 1; }

  // Prepare output
  ensure_c_file_size(args.C_path, args.N, args.M);
  FILE* fC = std::fopen(args.C_path.c_str(), "rb+");
  if (!fC){ perror(("open C: " + args.C_path).c_str()); return 1; }

  // GPU info
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::string gpu_name(prop.name);

  // Banner
  std::cout << "Out-of-core GEMM (tile 16x16, chunked I/O)\n"
            << "Backend : " << args.backend << "\n"
            << "A: " << args.A_path << "  (" << args.N << "x" << args.K << ")\n"
            << "B: " << args.B_path << "  (" << args.K << "x" << args.M << ")\n"
            << "C: " << args.C_path << "  (" << args.N << "x" << args.M << ")\n"
            << "Chunks : CN=" << args.CN << "  CM=" << args.CM << "  CK=" << args.CK << "\n"
            << "CSV    : " << (args.csv ? "enabled" : "disabled") << (args.csv && !args.csv_file.empty() ? (" -> "+args.csv_file) : "") << "\n\n"
            << "Press ENTER to start..." << std::flush;
  std::string line; std::getline(std::cin, line);

  const int N  = args.N, K=args.K, M=args.M;
  const int CN = args.CN, CM=args.CM, CK=args.CK;

  // Host buffers per chunk
  std::vector<float> hA((size_t)CN * (size_t)CK);
  std::vector<float> hB((size_t)CK * (size_t)CM);
  std::vector<float> hC((size_t)CN * (size_t)CM);

  // Device buffers per chunk
  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CHECK_CUDA(cudaMalloc(&dA, (size_t)CN * (size_t)CK * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, (size_t)CK * (size_t)CM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, (size_t)CN * (size_t)CM * sizeof(float)));

  // cuBLAS (optional)
  cublasHandle_t hblas=nullptr;
  if (args.backend=="cublas"){ CHECK_CUBLAS(cublasCreate(&hblas)); }

  // Timing accumulators
  CudaTimingContext t_cuda;
  auto t_begin = std::chrono::high_resolution_clock::now();
  double total_file_io_ms = 0.0;
  double total_cpu_acc_ms = 0.0; // we keep accumulation on GPU, so this stays ~0
  float  total_h2d_ms = 0.0f, total_kernel_ms=0.0f, total_d2h_ms=0.0f;
  size_t total_bytes = 0;
  int chunk_triplets = 0; // (# of (i0,j0,k0) loops)
  std::vector<double> chunk_times;

  // Grid for custom kernel varies with sub-chunk sizes:
  dim3 block(TILE, TILE);
  size_t smem = 2 * (size_t)TILE * TILE * sizeof(float);

  for (int i0=0; i0<N; i0+=CN){
    int cn = std::min(CN, N - i0);
    for (int j0=0; j0<M; j0+=CM){
      int cm = std::min(CM, M - j0);

      // zero C chunk
      CHECK_CUDA(cudaMemset(dC, 0, (size_t)cn * cm * sizeof(float)));

      for (int k0=0; k0<K; k0+=CK){
        int ck = std::min(CK, K - k0);

        auto chunk_t0 = std::chrono::high_resolution_clock::now();

        // Read chunks
        total_file_io_ms += read_block_A(fA, hA.data(), N, K, i0, k0, cn, ck);
        total_file_io_ms += read_block_B(fB, hB.data(), K, M, k0, j0, ck, cm);

        // H2D
        t_cuda.start();
        CHECK_CUDA(cudaMemcpy(dA, hA.data(), (size_t)cn * ck * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), (size_t)ck * cm * sizeof(float), cudaMemcpyHostToDevice));
        total_h2d_ms += t_cuda.stop();
        total_bytes  += (size_t)cn * ck * sizeof(float) + (size_t)ck * cm * sizeof(float);

        // Compute
        t_cuda.start();
        if (args.backend=="custom"){
          dim3 grid((cm + TILE - 1)/TILE, (cn + TILE - 1)/TILE);
          gemm_chunk_kernel<<<grid, block, smem>>>(dA, dB, dC, cn, cm, ck);
          CHECK_CUDA(cudaPeekAtLastError());
          CHECK_CUDA(cudaDeviceSynchronize());
        } else {
          // cuBLAS row-major mapping:
          // C_rm(cn x cm) += A_rm(cn x ck) * B_rm(ck x cm)
          // Use column-major call: C_cm(cm x cn) += B_cm(cm x ck) * A_cm(ck x cn)
          // m=cm, n=cn, k=ck
          // A=dB lda=cm, B=dA ldb=ck, C=dC ldc=cm
          const float alpha = 1.0f;
          const float beta  = 1.0f; // C chunk already zeroed, but beta=1 ok across k0 loops
          CHECK_CUBLAS(cublasSgemm(hblas,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   /*m*/ cm, /*n*/ cn, /*k*/ ck,
                                   &alpha,
                                   /*A*/ dB, /*lda*/ cm,
                                   /*B*/ dA, /*ldb*/ ck,
                                   &beta,
                                   /*C*/ dC, /*ldc*/ cm));
          CHECK_CUDA(cudaDeviceSynchronize());
        }
        total_kernel_ms += t_cuda.stop();

        // D2H for stats only? We need it after finishing all k0 for this (i0,j0).
        // Here we don't D2H yet; we will D2H once per (i0,j0).

        auto chunk_t1 = std::chrono::high_resolution_clock::now();
        chunk_times.push_back(std::chrono::duration<double, std::milli>(chunk_t1 - chunk_t0).count());
        ++chunk_triplets;
      }

      // D2H and write C chunk
      t_cuda.start();
      CHECK_CUDA(cudaMemcpy(hC.data(), dC, (size_t)cn * cm * sizeof(float), cudaMemcpyDeviceToHost));
      total_d2h_ms += t_cuda.stop();
      total_bytes  += (size_t)cn * cm * sizeof(float);
      total_file_io_ms += write_block_C(fC, hC.data(), N, M, i0, j0, cn, cm);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double wall_s = std::chrono::duration<double>(t_end - t_begin).count();

  // Cleanup
  if (hblas) { cublasDestroy(hblas); }
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  std::fclose(fA); std::fclose(fB); std::fclose(fC);

  // Metrics
  double gflops = (2.0 * (double)N * (double)K * (double)M) / (wall_s * 1e9);
  double avg_chunk_ms = chunk_times.empty()? 0.0 : std::accumulate(chunk_times.begin(), chunk_times.end(), 0.0) / chunk_times.size();
  double mem_bw_gbs = (total_bytes / 1e9) / wall_s;
  double total_gb = total_bytes / 1e9;

  // Summary
  std::cout << "\n=== PERFORMANCE SUMMARY ===\n";
  std::cout << "Backend: " << args.backend << "\n";
  std::cout << "Wall Time: " << wall_s << " s\n";
  std::cout << "Total chunk triplets (i0,j0,k0): " << chunk_triplets << "\n";
  std::cout << "Avg chunk time: " << avg_chunk_ms << " ms\n";
  std::cout << "Breakdown (seconds):\n";
  std::cout << "  File I/O:  " << (total_file_io_ms/1000.0) << "\n";
  std::cout << "  H2D:       " << (total_h2d_ms/1000.0) << "\n";
  std::cout << "  Kernel:    " << (total_kernel_ms/1000.0) << "\n";
  std::cout << "  D2H:       " << (total_d2h_ms/1000.0) << "\n";
  std::cout << "GFLOPS: " << gflops << "\n";
  std::cout << "Total transfer: " << total_gb << " GB, Effective BW: " << mem_bw_gbs << " GB/s\n";

  // CSV
  if (args.csv){
    MatmulTimingResults R{
      timestamp_now(),
      N, K, M,
      CN, CM, CK,
      TILE,
      args.backend,
      gpu_name,
      chunk_triplets,
      wall_s,
      total_file_io_ms,
      total_h2d_ms,
      total_kernel_ms,
      total_d2h_ms,
      total_cpu_acc_ms,
      avg_chunk_ms,
      gflops,
      mem_bw_gbs,
      total_gb
    };
    std::string csvname = generate_csv_filename(args.csv_file);
    export_timing_to_csv(R, csvname);
  }

  return 0;
}
