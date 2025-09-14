#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class Logger {
public:
    static void info(const std::string& message) {
        std::cout << "[INFO] " << message << std::endl;
    }
    static void error(const std::string& message) {
        std::cerr << "[ERROR] " << message << std::endl;
    }
    static void debug(const std::string& message) {
        std::cout << "[DEBUG] " << message << std::endl;
    }
};

class SimpleBlockMatMul {
private:
    bool initialized;

public:
    SimpleBlockMatMul() : initialized(false) {}

    ~SimpleBlockMatMul() {}

    bool initialize() {
        if (initialized) return true;
        initialized = true;
        Logger::info("Simple CPU matrix multiplication initialized");
        return true;
    }

    bool compute(const std::vector<float>& A, const std::vector<float>& B,
                 std::vector<float>& C, int rows, int K, int cols) {
        if (!initialized) {
            Logger::error("Not initialized");
            return false;
        }

        // Simple CPU matrix multiplication
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * cols + j];
                }
                C[i * cols + j] = sum;
            }
        }

        return true;
    }

    void cleanup() {
        initialized = false;
    }
};

int main(int argc, char* argv[]) {
    int rows = 32, K = 32, cols = 32;
    std::string backend = "cpu";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--rows" && i + 1 < argc) {
            rows = std::stoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            K = std::stoi(argv[++i]);
        } else if (arg == "--cols" && i + 1 < argc) {
            cols = std::stoi(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        }
    }

    Logger::info("Starting block matrix multiplication");
    Logger::info("Dimensions: " + std::to_string(rows) + "x" + std::to_string(K) + " * " + std::to_string(K) + "x" + std::to_string(cols));
    Logger::info("Backend: " + backend);

    // Read input data from stdin
    std::vector<char> input_data;
    char buffer[4096];
    while (std::cin.read(buffer, sizeof(buffer)) || std::cin.gcount() > 0) {
        input_data.insert(input_data.end(), buffer, buffer + std::cin.gcount());
    }

    if (input_data.empty()) {
        Logger::error("No data read from stdin");
        return 1;
    }

    Logger::info("Read " + std::to_string(input_data.size()) + " bytes from stdin");

    // Parse input data: [uniforms][A_data][B_data]
    if (input_data.size() < 12) {
        Logger::error("Input data too small");
        return 1;
    }

    // Read uniforms (3 ints = 12 bytes)
    int* uniforms = reinterpret_cast<int*>(input_data.data());
    int actual_rows = uniforms[0];
    int actual_K = uniforms[1];
    int actual_cols = uniforms[2];

    Logger::info("Actual dimensions: " + std::to_string(actual_rows) + "x" + std::to_string(actual_K) + " * " + std::to_string(actual_K) + "x" + std::to_string(actual_cols));

    // Calculate expected data size
    size_t expected_size = 12 + actual_rows * actual_K * sizeof(float) + actual_K * actual_cols * sizeof(float);
    if (input_data.size() < expected_size) {
        Logger::error("Input data size mismatch. Expected: " + std::to_string(expected_size) + ", got: " + std::to_string(input_data.size()));
        return 1;
    }

    // Extract A and B data
    size_t offset = 12;
    std::vector<float> A(actual_rows * actual_K);
    std::memcpy(A.data(), input_data.data() + offset, A.size() * sizeof(float));
    offset += A.size() * sizeof(float);

    std::vector<float> B(actual_K * actual_cols);
    std::memcpy(B.data(), input_data.data() + offset, B.size() * sizeof(float));

    // Initialize compute engine
    SimpleBlockMatMul compute_engine;
    if (!compute_engine.initialize()) {
        Logger::error("Failed to initialize compute engine");
        return 1;
    }

    // Allocate output matrix
    std::vector<float> C(actual_rows * actual_cols);

    // Perform computation
    auto start = std::chrono::high_resolution_clock::now();
    if (!compute_engine.compute(A, B, C, actual_rows, actual_K, actual_cols)) {
        Logger::error("Computation failed");
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Logger::info("Computation completed in " + std::to_string(duration.count()) + " microseconds");

    // Write result to stdout
    std::cout.write(reinterpret_cast<const char*>(C.data()), C.size() * sizeof(float));
    std::cout.flush();

    Logger::info("Result written to stdout");
    return 0;
}
