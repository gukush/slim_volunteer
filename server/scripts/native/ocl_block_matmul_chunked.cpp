#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdint>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class Logger {
public:
    static void info(const std::string& message) {
        // Disabled to avoid corrupting binary output
    }
    static void error(const std::string& message) {
        std::cerr << "[ERROR] " << message << std::endl;
    }
    static void debug(const std::string& message) {
        // Disabled to avoid corrupting binary output
    }
};

class OpenCLBlockMatMul {
private:
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    bool initialized;

public:
    OpenCLBlockMatMul() : context(nullptr), queue(nullptr), program(nullptr), kernel(nullptr), initialized(false) {}

    ~OpenCLBlockMatMul() {
        cleanup();
    }

    bool initialize() {
        if (initialized) return true;

        cl_int err;

        // Get platform
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to get OpenCL platform");
            return false;
        }

        // Get device
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to get OpenCL GPU device, trying CPU");
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
            if (err != CL_SUCCESS) {
                Logger::error("Failed to get any OpenCL device");
                return false;
            }
        }

        // Create context
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to create OpenCL context");
            return false;
        }

        // Create command queue
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to create command queue");
            return false;
        }

        // Create kernel source - simplified version that should compile without issues
        const char* kernel_source = R"(
__kernel void block_matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int rows,
    int K,
    int cols
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= rows || j >= cols) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * cols + j];
    }

    C[i * cols + j] = sum;
}
)";

        // Create program
        program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to create program");
            return false;
        }

        // Build program with additional options to avoid JIT compilation issues
        const char* build_options = "-cl-std=CL1.2 -w";
        err = clBuildProgram(program, 1, &device, build_options, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to build program");
            char build_log[4096];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, nullptr);
            Logger::error("Build log: " + std::string(build_log));
            return false;
        }

        // Create kernel
        kernel = clCreateKernel(program, "block_matmul", &err);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to create kernel");
            return false;
        }

        initialized = true;
        Logger::info("OpenCL initialized successfully");
        return true;
    }

    bool compute(const std::vector<float>& A, const std::vector<float>& B,
                 std::vector<float>& C, int rows, int K, int cols) {
        if (!initialized) {
            Logger::error("OpenCL not initialized");
            return false;
        }

        // Check if we have a GPU device, if not use CPU fallback
        cl_device_type device_type;
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);

        if (device_type == CL_DEVICE_TYPE_CPU) {
            Logger::info("Using CPU fallback computation");
            return computeCPU(A, B, C, rows, K, cols);
        }

        cl_int err;
        size_t A_size = rows * K * sizeof(float);
        size_t B_size = K * cols * sizeof(float);
        size_t C_size = rows * cols * sizeof(float);

        // Create buffers
        cl_mem A_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A_size, (void*)A.data(), &err);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to create A buffer");
            return false;
        }

        cl_mem B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B_size, (void*)B.data(), &err);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to create B buffer");
            clReleaseMemObject(A_buf);
            return false;
        }

        cl_mem C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, C_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to create C buffer");
            clReleaseMemObject(A_buf);
            clReleaseMemObject(B_buf);
            return false;
        }

        // Set kernel arguments
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_buf);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_buf);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_buf);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &rows);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &K);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &cols);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to set kernel arguments");
            clReleaseMemObject(A_buf);
            clReleaseMemObject(B_buf);
            clReleaseMemObject(C_buf);
            return false;
        }

        // Execute kernel
        size_t global_size[2] = {static_cast<size_t>(cols), static_cast<size_t>(rows)};
        size_t local_size[2] = {16, 16};

        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to execute kernel");
            clReleaseMemObject(A_buf);
            clReleaseMemObject(B_buf);
            clReleaseMemObject(C_buf);
            return false;
        }

        // Read result
        C.resize(rows * cols);
        err = clEnqueueReadBuffer(queue, C_buf, CL_TRUE, 0, C_size, C.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            Logger::error("Failed to read result");
            clReleaseMemObject(A_buf);
            clReleaseMemObject(B_buf);
            clReleaseMemObject(C_buf);
            return false;
        }

        // Cleanup
        clReleaseMemObject(A_buf);
        clReleaseMemObject(B_buf);
        clReleaseMemObject(C_buf);

        return true;
    }

    bool computeCPU(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, int rows, int K, int cols) {
        C.resize(rows * cols);

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
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        initialized = false;
    }
};

void usage(const char* prog) {
    std::cout << "Usage: " << prog << " --rows <rows> --k <K> --cols <cols> [--backend <backend>]\n";
    std::cout << "Reads matrix data from stdin in format: [uniforms][A_data][B_data]\n";
    std::cout << "Writes result matrix to stdout\n";
}

int main(int argc, char* argv[]) {
    int rows = 0, K = 0, cols = 0;
    std::string backend = "opencl";

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
        } else if (arg == "--help") {
            usage(argv[0]);
            return 0;
        }
    }

    if (rows <= 0 || K <= 0 || cols <= 0) {
        Logger::error("Invalid dimensions: rows=" + std::to_string(rows) +
                     ", K=" + std::to_string(K) + ", cols=" + std::to_string(cols));
        usage(argv[0]);
        return 1;
    }

    Logger::info("Starting block matrix multiplication");
    Logger::info("Dimensions: " + std::to_string(rows) + "x" + std::to_string(K) +
                " * " + std::to_string(K) + "x" + std::to_string(cols));
    Logger::info("Backend: " + backend);

    // Read data from stdin
    std::vector<char> stdin_data;
    char buffer[4096];
    while (std::cin.read(buffer, sizeof(buffer))) {
        stdin_data.insert(stdin_data.end(), buffer, buffer + std::cin.gcount());
    }
    stdin_data.insert(stdin_data.end(), buffer, buffer + std::cin.gcount());

    if (stdin_data.empty()) {
        Logger::error("No data read from stdin");
        return 1;
    }

    Logger::info("Read " + std::to_string(stdin_data.size()) + " bytes from stdin");

    // Parse input data: [uniforms][A_data][B_data]
    size_t offset = 0;

    // Read uniforms (12 bytes: 3 * int32) - [rows, K, cols]
    if (stdin_data.size() < 12) {
        Logger::error("Input too small for uniforms");
        return 1;
    }

    // Override command-line dimensions with stdin uniforms
    int32_t stdin_rows, stdin_K, stdin_cols;
    std::memcpy(&stdin_rows, stdin_data.data() + 0, sizeof(int32_t));
    std::memcpy(&stdin_K, stdin_data.data() + 4, sizeof(int32_t));
    std::memcpy(&stdin_cols, stdin_data.data() + 8, sizeof(int32_t));

    // Use stdin dimensions if they're valid, otherwise fall back to command line
    if (stdin_rows > 0 && stdin_K > 0 && stdin_cols > 0) {
        rows = stdin_rows;
        K = stdin_K;
        cols = stdin_cols;
        Logger::info("Using dimensions from stdin: " + std::to_string(rows) + "x" + std::to_string(K) + " * " + std::to_string(K) + "x" + std::to_string(cols));
    } else {
        Logger::info("Using command-line dimensions: " + std::to_string(rows) + "x" + std::to_string(K) + " * " + std::to_string(K) + "x" + std::to_string(cols));
    }

    offset += 12;

    // Read A matrix
    size_t A_size = rows * K * sizeof(float);
    if (stdin_data.size() < offset + A_size) {
        Logger::error("Input too small for A matrix");
        return 1;
    }
    std::vector<float> A(rows * K);
    std::memcpy(A.data(), stdin_data.data() + offset, A_size);

    offset += A_size;

    // Read B matrix
    size_t B_size = K * cols * sizeof(float);
    if (stdin_data.size() < offset + B_size) {
        Logger::error("Input too small for B matrix");
        return 1;
    }
    std::vector<float> B(K * cols);
    std::memcpy(B.data(), stdin_data.data() + offset, B_size);

    Logger::info("Parsed A matrix: " + std::to_string(rows) + "x" + std::to_string(K));
    Logger::info("Parsed B matrix: " + std::to_string(K) + "x" + std::to_string(cols));

    // Initialize OpenCL
    OpenCLBlockMatMul ocl;
    if (!ocl.initialize()) {
        Logger::error("Failed to initialize OpenCL");
        return 1;
    }

    // Compute
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> C;
    bool success = ocl.compute(A, B, C, rows, K, cols);
    auto end = std::chrono::high_resolution_clock::now();

    if (!success) {
        Logger::error("Computation failed");
        return 1;
    }

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Logger::info("Computation completed in " + std::to_string(duration.count()) + " microseconds");

    // Write result to stdout
    std::cout.write(reinterpret_cast<const char*>(C.data()), C.size() * sizeof(float));
    std::cout.flush();

    Logger::info("Result written to stdout (" + std::to_string(C.size() * sizeof(float)) + " bytes)");
    return 0;
}

