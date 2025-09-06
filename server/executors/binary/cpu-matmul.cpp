#include <vector>
#include <cstring>
#include <iostream>

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

// Entry point for binary execution
int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <A_file> <B_file> <C_file> <rows> <K> <cols>" << std::endl;
        return 1;
    }
    
    const char* A_file = argv[1];
    const char* B_file = argv[2]; 
    const char* C_file = argv[3];
    int rows = std::atoi(argv[4]);
    int K = std::atoi(argv[5]);
    int cols = std::atoi(argv[6]);
    
    std::vector<float> A(rows * K);
    std::vector<float> B(K * cols);
    std::vector<float> C(rows * cols);
    
    // Read input matrices
    FILE* fA = std::fopen(A_file, "rb");
    FILE* fB = std::fopen(B_file, "rb");
    if (!fA || !fB) {
        std::cerr << "Error opening input files" << std::endl;
        return 1;
    }
    
    std::fread(A.data(), sizeof(float), rows * K, fA);
    std::fread(B.data(), sizeof(float), K * cols, fB);
    std::fclose(fA);
    std::fclose(fB);
    
    // Perform multiplication
    cpu_matmul(A.data(), B.data(), C.data(), rows, K, cols);
    
    // Write result
    FILE* fC = std::fopen(C_file, "wb");
    if (!fC) {
        std::cerr << "Error opening output file" << std::endl;
        return 1;
    }
    
    std::fwrite(C.data(), sizeof(float), rows * cols, fC);
    std::fclose(fC);
    
    std::cout << "Matrix multiplication completed: " << rows << "x" << K << " * " << K << "x" << cols << std::endl;
    return 0;
}

}
