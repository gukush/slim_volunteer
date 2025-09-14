#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>

extern "C" {

void cpu_matmul(const float* A, const float* B, float* C,
                int rows, int K, int cols) {
    std::memset(C, 0, rows * cols * sizeof(float));

    // Perform multiplication with loop optimization
    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < K; k++) {
            const float a_ik = A[i * K + k];
            const float* b_row = B + k * cols;
            float* c_row = C + i * cols;

            // Vectorizable inner loop
            for (int j = 0; j < cols; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

// Read binary data from stdin
std::vector<float> readFloatArrayFromStdin(size_t count) {
    std::vector<float> data(count);
    std::cin.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!std::cin) {
        throw std::runtime_error("Failed to read data from stdin");
    }
    return data;
}

// Write binary data to stdout
void writeFloatArrayToStdout(const std::vector<float>& data) {
    std::cout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    std::cout.flush();
}

// Read binary file and return as vector (fallback for file mode)
std::vector<float> readFloatArrayFromFile(const std::string& filename, size_t count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<float> data(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));

    if (!file) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    return data;
}

// Write binary file (fallback for file mode)
void writeFloatArrayToFile(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

    if (!file) {
        throw std::runtime_error("Error writing file: " + filename);
    }
}

// Entry point for binary execution
int main(int argc, char** argv) {
    // Check for stdin/stdout mode (no arguments or special flag)
    bool useStreams = (argc == 1) || (argc == 2 && std::string(argv[1]) == "--stdin");

    if (useStreams) {
        // Stream mode: read from stdin, write to stdout
        try {
            int rows, K, cols;

            // Read dimensions from stdin (first 3 integers)
            std::cin.read(reinterpret_cast<char*>(&rows), sizeof(int));
            std::cin.read(reinterpret_cast<char*>(&K), sizeof(int));
            std::cin.read(reinterpret_cast<char*>(&cols), sizeof(int));

            if (!std::cin) {
                std::cerr << "[cpu-matmul] Failed to read dimensions from stdin" << std::endl;
                return 1;
            }

            std::cerr << "[cpu-matmul] Reading matrices from stdin: " << rows << "x" << K << " * " << K << "x" << cols << std::endl;

            // Read matrices from stdin
            auto A = readFloatArrayFromStdin(rows * K);
            auto B = readFloatArrayFromStdin(K * cols);
            std::vector<float> C(rows * cols);

            std::cerr << "[cpu-matmul] Performing matrix multiplication..." << std::endl;

            // Perform multiplication
            cpu_matmul(A.data(), B.data(), C.data(), rows, K, cols);

            // Write result to stdout
            writeFloatArrayToStdout(C);

            std::cerr << "[cpu-matmul] Matrix multiplication completed: " << rows << "x" << K << " * " << K << "x" << cols << std::endl;
            return 0;

        } catch (const std::exception& e) {
            std::cerr << "[cpu-matmul] Error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        // File mode: original behavior
        if (argc < 7) {
            std::cerr << "Usage: " << argv[0] << " [--stdin] | <A_file> <B_file> <C_file> <rows> <K> <cols>" << std::endl;
            std::cerr << "  --stdin: Read from stdin, write to stdout" << std::endl;
            std::cerr << "  A_file: Path to input matrix A (binary float data)" << std::endl;
            std::cerr << "  B_file: Path to input matrix B (binary float data)" << std::endl;
            std::cerr << "  C_file: Path to output matrix C (binary float data)" << std::endl;
            std::cerr << "  rows: Number of rows in matrix A" << std::endl;
            std::cerr << "  K: Number of columns in A / rows in B" << std::endl;
            std::cerr << "  cols: Number of columns in matrix B" << std::endl;
            return 1;
        }

        try {
            const char* A_file = argv[1];
            const char* B_file = argv[2];
            const char* C_file = argv[3];
            int rows = std::atoi(argv[4]);
            int K = std::atoi(argv[5]);
            int cols = std::atoi(argv[6]);

            std::cerr << "[cpu-matmul] Reading matrices from files: " << rows << "x" << K << " * " << K << "x" << cols << std::endl;

            // Read input matrices
            auto A = readFloatArrayFromFile(A_file, rows * K);
            auto B = readFloatArrayFromFile(B_file, K * cols);
            std::vector<float> C(rows * cols);

            std::cerr << "[cpu-matmul] Performing matrix multiplication..." << std::endl;

            // Perform multiplication
            cpu_matmul(A.data(), B.data(), C.data(), rows, K, cols);

            // Write result
            writeFloatArrayToFile(C_file, C);

            std::cerr << "[cpu-matmul] Matrix multiplication completed: " << rows << "x" << K << " * " << K << "x" << cols << std::endl;
            return 0;

        } catch (const std::exception& e) {
            std::cerr << "[cpu-matmul] Error: " << e.what() << std::endl;
            return 1;
        }
    }
}

}
