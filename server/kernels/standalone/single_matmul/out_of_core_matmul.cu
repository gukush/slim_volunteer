#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Fixed tile size for simplicity
#define TILE_SIZE 16

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d - status %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Timing results structure for CSV export
struct MatmulTimingResults {
    std::string timestamp;
    int N, K, M;
    int tile_size;
    std::string kernel_type;
    std::string gpu_name;
    int total_chunks;
    double wall_time_s;
    double total_file_io_ms;
    double total_h2d_ms;
    double total_kernel_ms;
    double total_d2h_ms;
    double total_cpu_acc_ms;
    double avg_chunk_time_ms;
    double effective_gflops;
    double memory_bandwidth_gb_s;
    double total_memory_transferred_gb;
};

// Generate unique CSV filename with timestamp
std::string generate_csv_filename(const std::string& custom_name = "") {
    if (!custom_name.empty()) {
        return custom_name;
    }

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << "matmul_tile16_timing_"
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count()
       << ".csv";

    return ss.str();
}

// Export timing results to CSV file
void export_timing_to_csv(const MatmulTimingResults& results, const std::string& filename) {
    bool file_exists = std::ifstream(filename).good();

    std::ofstream csv_file(filename, std::ios::app);
    if (!csv_file) {
        std::cerr << "Warning: Cannot create CSV file: " << filename << std::endl;
        return;
    }

    // Write header if file is new
    if (!file_exists) {
        csv_file << "timestamp,N,K,M,tile_size,kernel_type,gpu_name,total_chunks,"
                 << "wall_time_s,total_file_io_ms,total_h2d_ms,total_kernel_ms,"
                 << "total_d2h_ms,total_cpu_acc_ms,avg_chunk_time_ms,effective_gflops,"
                 << "memory_bandwidth_gb_s,total_memory_transferred_gb\n";
    }

    // Write data row
    csv_file << results.timestamp << ","
             << results.N << "," << results.K << "," << results.M << ","
             << results.tile_size << "," << results.kernel_type << ","
             << "\"" << results.gpu_name << "\"," << results.total_chunks << ","
             << std::fixed << std::setprecision(3)
             << results.wall_time_s << "," << results.total_file_io_ms << ","
             << results.total_h2d_ms << "," << results.total_kernel_ms << ","
             << results.total_d2h_ms << "," << results.total_cpu_acc_ms << ","
             << results.avg_chunk_time_ms << ","
             << std::setprecision(2) << results.effective_gflops << ","
             << results.memory_bandwidth_gb_s << ","
             << std::setprecision(3) << results.total_memory_transferred_gb << "\n";

    csv_file.close();
    std::cout << "Timing results exported to: " << filename << std::endl;
}

// CUDA timing utility class
class CudaTimingContext {
private:
    cudaEvent_t start_event, stop_event;

public:
    CudaTimingContext() {
        CHECK_CUDA(cudaEventCreate(&start_event));
        CHECK_CUDA(cudaEventCreate(&stop_event));
    }

    ~CudaTimingContext() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(start_event));
    }

    float stop() {
        CHECK_CUDA(cudaEventRecord(stop_event));
        CHECK_CUDA(cudaEventSynchronize(stop_event));
        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }
};

// Fixed 16x16 tiled matrix multiplication kernel
__global__ void tiled_matmul_kernel_16x16(const float* A, const float* B, float* C, int N, int K, int M) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;

        if (a_row < N && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && b_col < M) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * M + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = sum;
    }
}

// Reads a tile from a large row-major matrix file with timing.
double read_tile(FILE* file, float* buffer, int start_row, int num_rows, int start_col, int num_cols, int total_cols) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_rows; ++i) {
        long long offset = (long long)(start_row + i) * total_cols + start_col;
        fseek(file, offset * sizeof(float), SEEK_SET);
        size_t bytes_read = fread(buffer + (long long)i * num_cols, sizeof(float), num_cols, file);
        (void)bytes_read; // Suppress unused variable warning
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

// Writes a tile to a large row-major matrix file with timing.
double write_tile(FILE* file, const float* buffer, int start_row, int num_rows, int start_col, int num_cols, int total_cols) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_rows; ++i) {
        long long offset = (long long)(start_row + i) * total_cols + start_col;
        fseek(file, offset * sizeof(float), SEEK_SET);
        size_t bytes_written = fwrite(buffer + (long long)i * num_cols, sizeof(float), num_cols, file);
        (void)bytes_written; // Suppress unused variable warning
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

int main(int argc, char** argv) {
    // --- Parse Command Line Arguments ---
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <N> <K> <M> <A_file> <B_file> <C_file> <kernel_type> [--csv] [--csv-file=filename.csv]\n";
        std::cerr << "  kernel_type: 'custom' or 'cublas'\n";
        std::cerr << "  Fixed tile size: 16x16\n";
        std::cerr << "  --csv: Export timing results to auto-generated CSV file\n";
        std::cerr << "  --csv-file=filename.csv: Export to custom CSV filename\n";
        return 1;
    }

    const int N = std::stoi(argv[1]);
    const int K = std::stoi(argv[2]);
    const int M = std::stoi(argv[3]);
    const std::string a_path = argv[4];
    const std::string b_path = argv[5];
    const std::string c_path = argv[6];
    const std::string kernel_type = argv[7];

    // Parse CSV export options
    bool export_csv = false;
    std::string csv_filename = "";
    for (int i = 8; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--csv") {
            export_csv = true;
        } else if (arg.substr(0, 11) == "--csv-file=") {
            csv_filename = arg.substr(11);
            export_csv = true;
        }
    }

    // Get GPU information
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::string gpu_name(prop.name);

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp_ss;
    timestamp_ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    std::string timestamp = timestamp_ss.str();

    std::cout << "CUDA Tiled Matrix Multiplication (16x16 tiles) with Timing Export\n";
    std::cout << "=================================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Problem Size: C(" << N << "x" << M << ") = A(" << N << "x" << K << ") * B(" << K << "x" << M << ")\n";
    std::cout << "  Tile Size: 16x16 (fixed)\n";
    std::cout << "  Kernel: " << kernel_type << "\n";
    std::cout << "  GPU: " << gpu_name << "\n";
    std::cout << "  CSV Export: " << (export_csv ? "enabled" : "disabled") << "\n";
    if (export_csv && !csv_filename.empty()) {
        std::cout << "  CSV File: " << csv_filename << "\n";
    }

    std::cout << "\nPress ENTER to start computation..." << std::endl;
    std::cin.get();

    // --- Generate Input Files if they don't exist ---
    FILE *file_a = fopen(a_path.c_str(), "rb");
    if (!file_a) {
        std::cout << "Generating test file " << a_path << "...\n";
        FILE* f = fopen(a_path.c_str(), "wb");
        std::vector<float> tmp(N * K);
        for(size_t i = 0; i < (size_t)N*K; ++i) tmp[i] = (float)rand() / RAND_MAX;
        fwrite(tmp.data(), sizeof(float), N * K, f);
        fclose(f);
        file_a = fopen(a_path.c_str(), "rb");
    }
    FILE *file_b = fopen(b_path.c_str(), "rb");
     if (!file_b) {
        std::cout << "Generating test file " << b_path << "...\n";
        FILE* f = fopen(b_path.c_str(), "wb");
        std::vector<float> tmp(K * M);
        for(size_t i = 0; i < (size_t)K*M; ++i) tmp[i] = (float)rand() / RAND_MAX;
        fwrite(tmp.data(), sizeof(float), K * M, f);
        fclose(f);
        file_b = fopen(b_path.c_str(), "rb");
    }

    // --- Prepare Output File ---
    FILE *file_c = fopen(c_path.c_str(), "wb");
    std::vector<char> zero_buffer(N * M * sizeof(float), 0);
    fwrite(zero_buffer.data(), 1, zero_buffer.size(), file_c);
    fclose(file_c);
    file_c = fopen(c_path.c_str(), "r+b"); // Reopen for writing

    // --- GPU and cuBLAS Initialization ---
    cublasHandle_t cublas_handle = nullptr;
    if (kernel_type == "cublas") {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
    }

    // --- Enhanced Timing Infrastructure ---
    CudaTimingContext cuda_timer;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    double total_file_io_ms = 0;
    float total_h2d_ms = 0, total_kernel_ms = 0, total_d2h_ms = 0;
    double total_cpu_acc_ms = 0;
    size_t total_bytes_transferred = 0;

    // --- Main Processing Loop ---
    const int num_row_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    const int num_col_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;
    const int num_k_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    std::vector<float> h_A_tile(TILE_SIZE * TILE_SIZE);
    std::vector<float> h_B_tile(TILE_SIZE * TILE_SIZE);
    std::vector<float> h_C_tile(TILE_SIZE * TILE_SIZE);
    std::vector<float> h_C_acc_tile(TILE_SIZE * TILE_SIZE);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, TILE_SIZE * TILE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, TILE_SIZE * TILE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, TILE_SIZE * TILE_SIZE * sizeof(float)));

    int chunk_count = 0;
    std::vector<double> chunk_times;

    for (int ib = 0; ib < num_row_tiles; ++ib) {
        for (int jb = 0; jb < num_col_tiles; ++jb) {

            // Zero out accumulator tile for this C-tile
            std::fill(h_C_acc_tile.begin(), h_C_acc_tile.end(), 0.0f);

            for (int kb = 0; kb < num_k_tiles; ++kb) {
                auto chunk_start_time = std::chrono::high_resolution_clock::now();

                // --- 1. Read tiles from disk with timing ---
                int r_now = std::min(TILE_SIZE, N - ib * TILE_SIZE);
                int c_now = std::min(TILE_SIZE, M - jb * TILE_SIZE);
                int k_now = std::min(TILE_SIZE, K - kb * TILE_SIZE);

                double read_a_time = read_tile(file_a, h_A_tile.data(), ib * TILE_SIZE, r_now, kb * TILE_SIZE, k_now, K);
                double read_b_time = read_tile(file_b, h_B_tile.data(), kb * TILE_SIZE, k_now, jb * TILE_SIZE, c_now, M);
                total_file_io_ms += read_a_time + read_b_time;

                // --- 2. Transfer to GPU with precise timing ---
                cuda_timer.start();
                CHECK_CUDA(cudaMemcpy(d_A, h_A_tile.data(), (size_t)r_now * k_now * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_B, h_B_tile.data(), (size_t)k_now * c_now * sizeof(float), cudaMemcpyHostToDevice));
                float h2d_ms = cuda_timer.stop();
                total_h2d_ms += h2d_ms;
                total_bytes_transferred += (size_t)r_now * k_now * sizeof(float) + (size_t)k_now * c_now * sizeof(float);

                // --- 3. Compute on GPU with precise timing ---
                cuda_timer.start();
                if (kernel_type == "custom") {
                    dim3 block(16, 16);
                    dim3 grid((c_now + block.x - 1) / block.x, (r_now + block.y - 1) / block.y);
                    tiled_matmul_kernel_16x16<<<grid, block>>>(d_A, d_B, d_C, r_now, k_now, c_now);
                } else { // cublas
                    const float alpha = 1.0f, beta = 0.0f;
                    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             c_now, r_now, k_now, &alpha,
                                             d_B, c_now, d_A, k_now, &beta, d_C, c_now));
                }
                float kernel_ms = cuda_timer.stop();
                total_kernel_ms += kernel_ms;

                // --- 4. Transfer from GPU with precise timing ---
                cuda_timer.start();
                CHECK_CUDA(cudaMemcpy(h_C_tile.data(), d_C, (size_t)r_now * c_now * sizeof(float), cudaMemcpyDeviceToHost));
                float d2h_ms = cuda_timer.stop();
                total_d2h_ms += d2h_ms;
                total_bytes_transferred += (size_t)r_now * c_now * sizeof(float);

                // --- 5. Accumulate on CPU with timing ---
                auto acc_start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < r_now * c_now; ++i) {
                    h_C_acc_tile[i] += h_C_tile[i];
                }
                auto acc_end = std::chrono::high_resolution_clock::now();
                total_cpu_acc_ms += std::chrono::duration<double, std::milli>(acc_end - acc_start).count();

                auto chunk_end_time = std::chrono::high_resolution_clock::now();
                double chunk_time_ms = std::chrono::duration<double, std::milli>(chunk_end_time - chunk_start_time).count();
                chunk_times.push_back(chunk_time_ms);
                chunk_count++;

                // Progress indicator
                if (chunk_count % 10 == 0) {
                    std::cout << "Processed " << chunk_count << " chunks..." << std::endl;
                }
            }

            // --- 6. Write final C-tile to disk with timing ---
            int r_now = std::min(TILE_SIZE, N - ib * TILE_SIZE);
            int c_now = std::min(TILE_SIZE, M - jb * TILE_SIZE);
            double write_time = write_tile(file_c, h_C_acc_tile.data(), ib * TILE_SIZE, r_now, jb * TILE_SIZE, c_now, M);
            total_file_io_ms += write_time;
        }
    }

    // --- Cleanup and Final Timings ---
    auto total_end_time = std::chrono::high_resolution_clock::now();
    double wall_time_s = std::chrono::duration<double>(total_end_time - total_start_time).count();

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    if (cublas_handle) CHECK_CUBLAS(cublasDestroy(cublas_handle));

    fclose(file_a);
    fclose(file_b);
    fclose(file_c);

    // --- Calculate Performance Metrics ---
    double total_gflops = (2.0 * N * K * M) / (wall_time_s * 1e9);
    double avg_chunk_time_ms = chunk_times.empty() ? 0.0 :
        std::accumulate(chunk_times.begin(), chunk_times.end(), 0.0) / chunk_times.size();
    double memory_bandwidth = (total_bytes_transferred / 1e9) / (wall_time_s);
    double total_memory_gb = total_bytes_transferred / 1e9;

    // --- Print Enhanced Performance Summary ---
    std::cout << "\n=== ENHANCED PERFORMANCE SUMMARY ===\n";
    std::cout << "Total Wall Time: " << wall_time_s << " s\n";
    std::cout << "Total Chunks Processed: " << chunk_count << "\n";
    std::cout << "Average Chunk Processing Time: " << avg_chunk_time_ms << " ms\n";
    std::cout << "\nDetailed Timing Breakdown:\n";
    std::cout << "  File I/O Time:         " << total_file_io_ms / 1000.0 << " s ("
              << (total_file_io_ms / 1000.0 / wall_time_s * 100) << "%)\n";
    std::cout << "  GPU H2D Transfer Time: " << total_h2d_ms / 1000.0 << " s ("
              << (total_h2d_ms / 1000.0 / wall_time_s * 100) << "%)\n";
    std::cout << "  GPU Kernel Time:       " << total_kernel_ms / 1000.0 << " s ("
              << (total_kernel_ms / 1000.0 / wall_time_s * 100) << "%)\n";
    std::cout << "  GPU D2H Transfer Time: " << total_d2h_ms / 1000.0 << " s ("
              << (total_d2h_ms / 1000.0 / wall_time_s * 100) << "%)\n";
    std::cout << "  CPU Accumulation Time: " << total_cpu_acc_ms / 1000.0 << " s ("
              << (total_cpu_acc_ms / 1000.0 / wall_time_s * 100) << "%)\n";
    std::cout << "\nPerformance Metrics:\n";
    std::cout << "  Effective GFLOPS:      " << total_gflops << "\n";
    std::cout << "  Memory Bandwidth:      " << memory_bandwidth << " GB/s\n";
    std::cout << "  Total Memory Transfer: " << total_memory_gb << " GB\n";

    // --- Export to CSV if requested ---
    if (export_csv) {
        std::string csv_file = csv_filename.empty() ? generate_csv_filename() : csv_filename;

        MatmulTimingResults results = {
            timestamp, N, K, M, TILE_SIZE, kernel_type, gpu_name, chunk_count,
            wall_time_s, total_file_io_ms, total_h2d_ms, total_kernel_ms,
            total_d2h_ms, total_cpu_acc_ms, avg_chunk_time_ms, total_gflops,
            memory_bandwidth, total_memory_gb
        };

        export_timing_to_csv(results, csv_file);
    }

    return 0;
}