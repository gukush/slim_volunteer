#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <ctime>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Struct for kernel parameters (equivalent to WGSL uniform)
struct BitonicParams {
    uint32_t array_size;  // N (power-of-two padded)
    uint32_t stage;       // current k
    uint32_t substage;    // current j
    uint32_t ascending;   // 1 for ascending, 0 for descending
};

// CUDA kernel implementing bitonic sort stage (equivalent to WGSL compute shader)
__global__ void bitonic_sort_kernel(uint32_t* data, BitonicParams params) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= params.array_size) {
        return;
    }

    uint32_t partner = i ^ params.substage;
    if (partner > i) {
        uint32_t a = data[i];
        uint32_t b = data[partner];

        // Determine sort direction for this comparison
        bool ascending_block = ((i & params.stage) == 0);
        bool should_ascend = ascending_block == (params.ascending == 1);

        // Swap if elements are out of order
        if ((a > b) == should_ascend) {
            data[i] = b;
            data[partner] = a;
        }
    }
}

// Round up to next power of 2
uint32_t next_power_of_2(uint32_t n) {
    if (n <= 1) return 1;
    return 1 << (32 - __builtin_clz(n - 1));
}

// Timing utility class
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }
};

// CPU high resolution timer
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
};

// Bitonic sort executor class
class BitonicSortExecutor {
private:
    uint32_t* d_data;
    size_t allocated_size;
    CudaTimer cuda_timer;

public:
    BitonicSortExecutor() : d_data(nullptr), allocated_size(0) {}

    ~BitonicSortExecutor() {
        if (d_data) {
            cudaFree(d_data);
        }
    }

    void ensure_gpu_memory(size_t required_size) {
        if (required_size > allocated_size) {
            if (d_data) {
                cudaFree(d_data);
            }

            CUDA_CHECK(cudaMalloc(&d_data, required_size));
            allocated_size = required_size;

            std::cout << "Allocated GPU memory: " << (required_size / 1024 / 1024) << " MB" << std::endl;
        }
    }

    double sort_chunk(std::vector<uint32_t>& chunk_data, bool ascending, bool validate = false) {
        uint32_t original_size = chunk_data.size();
        uint32_t padded_size = next_power_of_2(original_size);

        // Pad data to power of 2
        if (padded_size > original_size) {
            chunk_data.resize(padded_size);
            uint32_t sentinel = ascending ? 0xFFFFFFFF : 0;
            for (uint32_t i = original_size; i < padded_size; i++) {
                chunk_data[i] = sentinel;
            }
        }

        size_t data_size = padded_size * sizeof(uint32_t);
        ensure_gpu_memory(data_size);

        // Upload data to GPU
        CUDA_CHECK(cudaMemcpy(d_data, chunk_data.data(), data_size, cudaMemcpyHostToDevice));

        // Configure kernel launch parameters
        const uint32_t block_size = 256; // Equivalent to WebGPU workgroup_size(256)
        uint32_t num_blocks = (padded_size + block_size - 1) / block_size;

        double total_gpu_time = 0.0;
        int stage_count = 0;

        // Execute bitonic sort stages
        for (uint32_t k = 2; k <= padded_size; k <<= 1) {
            for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                BitonicParams params = {
                    padded_size,
                    k,
                    j,
                    ascending ? 1u : 0u
                };

                cuda_timer.start();
                bitonic_sort_kernel<<<num_blocks, block_size>>>(d_data, params);
                CUDA_CHECK(cudaDeviceSynchronize());
                float stage_time = cuda_timer.stop();

                total_gpu_time += stage_time;
                stage_count++;
            }
        }

        // Download results (only original size, not padding)
        CUDA_CHECK(cudaMemcpy(chunk_data.data(), d_data, original_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        chunk_data.resize(original_size); // Remove padding

        if (validate) {
            validate_sorted(chunk_data, ascending);
        }

        std::cout << "GPU sort: " << original_size << " integers, "
                  << stage_count << " stages, " << total_gpu_time << " ms total, "
                  << (total_gpu_time / stage_count) << " ms avg/stage" << std::endl;

        return total_gpu_time;
    }

private:
    void validate_sorted(const std::vector<uint32_t>& data, bool ascending) {
        for (size_t i = 1; i < data.size(); i++) {
            bool is_ordered = ascending ? (data[i-1] <= data[i]) : (data[i-1] >= data[i]);
            if (!is_ordered) {
                std::cerr << "Validation failed at position " << i
                          << ": " << data[i-1] << (ascending ? " > " : " < ") << data[i] << std::endl;
                exit(1);
            }
        }
        std::cout << "Validation passed: " << data.size() << " elements properly sorted" << std::endl;
    }
};

// k-way merge implementation for combining sorted chunks
std::vector<uint32_t> merge_k_sorted_chunks(const std::vector<std::vector<uint32_t>>& sorted_chunks, bool ascending) {
    if (sorted_chunks.empty()) return {};
    if (sorted_chunks.size() == 1) return sorted_chunks[0];

    size_t total_size = 0;
    for (const auto& chunk : sorted_chunks) {
        total_size += chunk.size();
    }

    std::vector<uint32_t> result;
    result.reserve(total_size);

    // Priority queue approach with indices
    struct HeapElement {
        uint32_t value;
        size_t chunk_idx;
        size_t element_idx;
        bool ascending_order;

        bool operator>(const HeapElement& other) const {
            // For ascending sort: smaller values have higher priority
            // For descending sort: larger values have higher priority
            return ascending_order ? value > other.value : value < other.value;
        }
    };

    std::priority_queue<HeapElement, std::vector<HeapElement>, std::greater<HeapElement>> heap;

    // Initialize heap with first element from each chunk
    for (size_t i = 0; i < sorted_chunks.size(); i++) {
        if (!sorted_chunks[i].empty()) {
            heap.push({sorted_chunks[i][0], i, 0, ascending});
        }
    }

    while (!heap.empty()) {
        HeapElement min_elem = heap.top();
        heap.pop();

        result.push_back(min_elem.value);

        // Add next element from the same chunk if available
        size_t next_idx = min_elem.element_idx + 1;
        if (next_idx < sorted_chunks[min_elem.chunk_idx].size()) {
            heap.push({
                sorted_chunks[min_elem.chunk_idx][next_idx],
                min_elem.chunk_idx,
                next_idx,
                ascending
            });
        }
    }

    return result;
}

// Timing results structure for CSV export
struct TimingResults {
    std::string timestamp;
    std::string input_file;
    std::string output_file;
    size_t total_integers;
    size_t chunk_size;
    size_t chunks_count;
    bool ascending;
    bool validation_enabled;
    std::string gpu_name;
    double total_gpu_time_ms;
    double cpu_merge_time_ms;
    double total_end_to_end_ms;
    double throughput_int_per_sec;
    double avg_gpu_time_per_stage_ms;
    int total_stages;
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
    ss << "bitonic_sort_timing_"
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count()
       << ".csv";

    return ss.str();
}

// Export timing results to CSV file
void export_timing_to_csv(const TimingResults& results, const std::string& filename) {
    bool file_exists = std::ifstream(filename).good();

    std::ofstream csv_file(filename, std::ios::app); // Append mode
    if (!csv_file) {
        std::cerr << "Warning: Cannot create CSV file: " << filename << std::endl;
        return;
    }

    // Write header if file is new
    if (!file_exists) {
        csv_file << "timestamp,input_file,output_file,total_integers,chunk_size,chunks_count,"
                 << "ascending,validation_enabled,gpu_name,total_gpu_time_ms,cpu_merge_time_ms,"
                 << "total_end_to_end_ms,throughput_int_per_sec,avg_gpu_time_per_stage_ms,total_stages\n";
    }

    // Write data row
    csv_file << results.timestamp << ","
             << results.input_file << ","
             << results.output_file << ","
             << results.total_integers << ","
             << results.chunk_size << ","
             << results.chunks_count << ","
             << (results.ascending ? "true" : "false") << ","
             << (results.validation_enabled ? "true" : "false") << ","
             << "\"" << results.gpu_name << "\","
             << std::fixed << std::setprecision(3) << results.total_gpu_time_ms << ","
             << results.cpu_merge_time_ms << ","
             << results.total_end_to_end_ms << ","
             << std::setprecision(1) << results.throughput_int_per_sec << ","
             << std::setprecision(3) << results.avg_gpu_time_per_stage_ms << ","
             << results.total_stages << "\n";

    csv_file.close();
    std::cout << "Timing results exported to: " << filename << std::endl;
}

class OutOfCoreBitonicSort {
private:
    BitonicSortExecutor executor;
    std::string input_file;
    std::string output_file;
    size_t chunk_size;
    bool ascending;
    bool validate;
    size_t max_elements;
    bool export_csv;
    std::string csv_file;

public:
    OutOfCoreBitonicSort(const std::string& input, const std::string& output,
                         size_t chunk_sz, bool asc, bool val, size_t max_elem = 0,
                         bool csv_export = false, const std::string& csv_filename = "")
        : input_file(input), output_file(output), chunk_size(chunk_sz),
          ascending(asc), validate(val), max_elements(max_elem),
          export_csv(csv_export), csv_file(csv_filename) {}

    void sort() {
        CpuTimer total_timer;
        total_timer.start();

        // Open input file and determine size
        std::ifstream input(input_file, std::ios::binary);
        if (!input) {
            std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
            exit(1);
        }

        input.seekg(0, std::ios::end);
        size_t file_size = input.tellg();
        input.seekg(0, std::ios::beg);

        if (file_size % 4 != 0) {
            std::cerr << "Error: File size must be multiple of 4 bytes" << std::endl;
            exit(1);
        }

        size_t total_integers = file_size / 4;
        if (max_elements > 0 && max_elements < total_integers) {
            total_integers = max_elements;
            std::cout << "Limiting processing to " << max_elements << " elements" << std::endl;
        }

        size_t chunks_count = (total_integers + chunk_size - 1) / chunk_size;

        std::cout << "Input file: " << input_file << " (" << (file_size / 1024.0 / 1024.0) << " MB)" << std::endl;
        std::cout << "Total integers: " << total_integers << std::endl;
        std::cout << "Chunk size: " << chunk_size << std::endl;
        std::cout << "Total chunks: " << chunks_count << std::endl;
        std::cout << "Sort order: " << (ascending ? "ascending" : "descending") << std::endl;

        // Process chunks and store sorted results
        std::vector<std::vector<uint32_t>> sorted_chunks;
        sorted_chunks.reserve(chunks_count);

        double total_gpu_time = 0.0;
        int total_stages = 0;

        for (size_t chunk_idx = 0; chunk_idx < chunks_count; chunk_idx++) {
            size_t offset = chunk_idx * chunk_size;
            size_t current_chunk_size = std::min(chunk_size, total_integers - offset);

            // Read chunk from file
            std::vector<uint32_t> chunk_data(current_chunk_size);
            input.read(reinterpret_cast<char*>(chunk_data.data()), current_chunk_size * sizeof(uint32_t));

            if (!input) {
                std::cerr << "Error reading chunk " << chunk_idx << std::endl;
                exit(1);
            }

            std::cout << "Processing chunk " << chunk_idx + 1 << "/" << chunks_count
                      << " (" << current_chunk_size << " integers)..." << std::endl;

            // Sort chunk on GPU
            double chunk_gpu_time = executor.sort_chunk(chunk_data, ascending, validate);
            total_gpu_time += chunk_gpu_time;

            // Calculate stages for this chunk
            uint32_t padded_size = next_power_of_2(current_chunk_size);
            for (uint32_t k = 2; k <= padded_size; k <<= 1) {
                for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                    total_stages++;
                }
            }

            sorted_chunks.push_back(std::move(chunk_data));
        }

        input.close();

        std::cout << "All chunks sorted. Starting k-way merge..." << std::endl;

        // Perform k-way merge of sorted chunks
        CpuTimer merge_timer;
        merge_timer.start();

        std::vector<uint32_t> final_result = merge_k_sorted_chunks(sorted_chunks, ascending);

        double merge_time = merge_timer.stop();
        std::cout << "K-way merge completed in " << merge_time << " ms" << std::endl;

        // Write final result to output file
        std::ofstream output(output_file, std::ios::binary);
        if (!output) {
            std::cerr << "Error: Cannot create output file: " << output_file << std::endl;
            exit(1);
        }

        output.write(reinterpret_cast<const char*>(final_result.data()),
                     final_result.size() * sizeof(uint32_t));
        output.close();

        double total_time = total_timer.stop();

        // Print timing summary
        std::cout << "\n=== TIMING SUMMARY ===" << std::endl;
        std::cout << "Total GPU time: " << total_gpu_time << " ms" << std::endl;
        std::cout << "CPU merge time: " << merge_time << " ms" << std::endl;
        std::cout << "Total end-to-end time: " << total_time << " ms" << std::endl;
        std::cout << "Throughput: " << (total_integers / (total_time / 1000.0)) << " integers/second" << std::endl;
        std::cout << "Output file: " << output_file << " (" << final_result.size() << " integers)" << std::endl;

        // Export CSV if requested
        if (export_csv) {
            export_timing_results(total_integers, chunks_count, total_gpu_time, merge_time, total_time, total_stages);
        }

        // Final validation
        if (validate) {
            std::cout << "Performing final validation..." << std::endl;
            validate_final_result(final_result);
        }
    }

private:
    void validate_final_result(const std::vector<uint32_t>& data) {
        for (size_t i = 1; i < data.size(); i++) {
            bool is_ordered = ascending ? (data[i-1] <= data[i]) : (data[i-1] >= data[i]);
            if (!is_ordered) {
                std::cerr << "Final validation FAILED at position " << i
                          << ": " << data[i-1] << (ascending ? " > " : " < ") << data[i] << std::endl;
                exit(1);
            }
        }
        std::cout << "Final validation PASSED: " << data.size() << " elements properly sorted" << std::endl;
    }

    void export_timing_results(size_t total_integers, size_t chunks_count, double total_gpu_time,
                              double merge_time, double total_time, int total_stages) {
        // Get GPU name
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        // Get timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream timestamp_ss;
        timestamp_ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

        TimingResults results = {
            timestamp_ss.str(),
            input_file,
            output_file,
            total_integers,
            chunk_size,
            chunks_count,
            ascending,
            validate,
            std::string(prop.name),
            total_gpu_time,
            merge_time,
            total_time,
            total_integers / (total_time / 1000.0),
            total_stages > 0 ? total_gpu_time / total_stages : 0.0,
            total_stages
        };

        std::string filename = csv_file.empty() ? generate_csv_filename() : csv_file;
        export_timing_to_csv(results, filename);
    }
};

// Command line argument parsing
struct Config {
    std::string input_file = "input.bin";
    std::string output_file = "output.bin";
    size_t chunk_size = 65536; // Default 64K integers per chunk
    bool ascending = true;
    bool validate = false;
    size_t max_elements = 0; // 0 means process entire file
    bool export_csv = false; // Export timing results to CSV
    std::string csv_file = ""; // Custom CSV filename (auto-generated if empty)
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --input=FILE        Input binary file (default: input.bin)\n"
              << "  --output=FILE       Output binary file (default: output.bin)\n"
              << "  --chunk-size=N      Integers per chunk (default: 65536)\n"
              << "  --descending        Sort in descending order (default: ascending)\n"
              << "  --validate          Validate sorted chunks and final result\n"
              << "  --max-elements=N    Limit processing to N elements\n"
              << "  --csv               Export timing results to CSV file\n"
              << "  --csv-file=FILE     Custom CSV filename (auto-generated if not specified)\n"
              << "  --help              Show this help message\n";
}

Config parse_args(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);

        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg.substr(0, 8) == "--input=") {
            config.input_file = arg.substr(8);
        } else if (arg.substr(0, 9) == "--output=") {
            config.output_file = arg.substr(9);
        } else if (arg.substr(0, 13) == "--chunk-size=") {
            config.chunk_size = std::stoul(arg.substr(13));
        } else if (arg == "--descending") {
            config.ascending = false;
        } else if (arg == "--validate") {
            config.validate = true;
        } else if (arg.substr(0, 15) == "--max-elements=") {
            config.max_elements = std::stoul(arg.substr(15));
        } else if (arg == "--csv") {
            config.export_csv = true;
        } else if (arg.substr(0, 11) == "--csv-file=") {
            config.csv_file = arg.substr(11);
            config.export_csv = true; // Enable CSV export if filename is provided
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Out-of-Core Bitonic Sort" << std::endl;
    std::cout << "=============================" << std::endl;

    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using CUDA device: " << prop.name << std::endl;
    std::cout << "Global memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;

    // Parse command line arguments
    Config config = parse_args(argc, argv);

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Input file: " << config.input_file << std::endl;
    std::cout << "  Output file: " << config.output_file << std::endl;
    std::cout << "  Chunk size: " << config.chunk_size << " integers" << std::endl;
    std::cout << "  Sort order: " << (config.ascending ? "ascending" : "descending") << std::endl;
    std::cout << "  Validation: " << (config.validate ? "enabled" : "disabled") << std::endl;
    std::cout << "  CSV export: " << (config.export_csv ? "enabled" : "disabled") << std::endl;
    if (config.export_csv && !config.csv_file.empty()) {
        std::cout << "  CSV file: " << config.csv_file << std::endl;
    }
    if (config.max_elements > 0) {
        std::cout << "  Max elements: " << config.max_elements << std::endl;
    }

    std::cout << "\nPress ENTER to start execution..." << std::endl;
    std::cin.get(); // Wait for user input

    try {
        // Create and run the out-of-core bitonic sort
        OutOfCoreBitonicSort sorter(config.input_file, config.output_file,
                                  config.chunk_size, config.ascending,
                                  config.validate, config.max_elements,
                                  config.export_csv, config.csv_file);
        sorter.sort();

        std::cout << "\nSort completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}