// File: native_opencl_block_matmul.cpp

#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>

using i32 = int32_t;

// -----------------------------------------------------------------------------
// Embedded OpenCL kernel (portable baseline).
// Replace the body with your tuned block-matmul if desired.
// Signature MUST remain: execute_task(int rows, int K, int cols, A, B, C)
// -----------------------------------------------------------------------------
static const char* KERNEL_SRC = R"CLC(
__kernel void execute_task(
    const int rows,
    const int K,
    const int cols,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    const int TILE = 16;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int c  = get_global_id(0);
    const int r  = get_global_id(1);

    __local float Asub[TILE][TILE];
    __local float Bsub[TILE][TILE];

    float acc = 0.0f;
    const int tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        const int k0 = t * TILE;

        float a = 0.0f;
        if (r < rows && (k0 + lx) < K)
            a = A[r * K + (k0 + lx)];
        Asub[ly][lx] = a;

        float b = 0.0f;
        if (c < cols && (k0 + ly) < K)
            b = B[(k0 + ly) * cols + c];
        Bsub[ly][lx] = b;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE; ++k) {
            acc += Asub[ly][k] * Bsub[k][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (r < rows && c < cols) {
        C[r * cols + c] = acc;
    }
}
)CLC";

static void die(const std::string& msg, cl_int err = 0){
    if(err) std::cerr << msg << " (CL error " << err << ")\n";
    else    std::cerr << msg << "\n";
    std::exit(1);
}

static void read_exact(void* dst, size_t n){
    auto* p = static_cast<unsigned char*>(dst);
    size_t got = 0;
    while (got < n){
        size_t r = std::fread(p + got, 1, n - got, stdin);
        if (r == 0) die("EOF while reading input stream");
        got += r;
    }
}

static void write_exact(const void* src, size_t n){
    auto* p = static_cast<const unsigned char*>(src);
    size_t put = 0;
    while (put < n){
        size_t w = std::fwrite(p + put, 1, n - put, stdout);
        if (w == 0) die("Write error to stdout");
        put += w;
    }
}

static cl_device_id pick_device(cl_platform_id& chosenPlatform){
    cl_uint numPlatforms = 0; cl_int err;
    if ((err = clGetPlatformIDs(0, nullptr, &numPlatforms)) != CL_SUCCESS || numPlatforms == 0)
        die("No OpenCL platforms found", err);
    std::vector<cl_platform_id> plats(numPlatforms);
    clGetPlatformIDs(numPlatforms, plats.data(), nullptr);

    cl_device_id dev = nullptr; chosenPlatform = nullptr;
    for (auto p: plats){
        cl_uint ndev=0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev)==CL_SUCCESS && ndev){
            std::vector<cl_device_id> ds(ndev);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, ndev, ds.data(), nullptr);
            dev = ds[0]; chosenPlatform = p; break;
        }
    }
    if (!dev){
        for (auto p: plats){
            cl_uint ndev=0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev)==CL_SUCCESS && ndev){
                std::vector<cl_device_id> ds(ndev);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndev, ds.data(), nullptr);
                dev = ds[0]; chosenPlatform = p; break;
            }
        }
    }
    if (!dev) die("No OpenCL devices found");
    return dev;
}

int main(){
    // 1) Read uniforms
    i32 rows=0, K=0, cols=0; read_exact(&rows,4); read_exact(&K,4); read_exact(&cols,4);
    if (rows<=0 || K<=0 || cols<=0) die("Invalid uniforms");

    // 2) Read inputs
    const size_t aElems = static_cast<size_t>(rows) * static_cast<size_t>(K);
    const size_t bElems = static_cast<size_t>(K)    * static_cast<size_t>(cols);
    const size_t cElems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<float> A(aElems), B(bElems), C(cElems);
    read_exact(A.data(), aElems*sizeof(float));
    read_exact(B.data(), bElems*sizeof(float));

    // 3) OpenCL setup
    cl_platform_id plat = nullptr;
    cl_device_id dev = pick_device(plat);
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (!ctx || err) die("clCreateContext failed", err);

#if defined(CL_VERSION_2_0)
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
#endif
    if (!q || err) die("clCreateCommandQueue failed", err);

    const char* src = KERNEL_SRC; size_t len = std::strlen(KERNEL_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (!prog || err) die("clCreateProgramWithSource failed", err);

    // Optional: read build opts from env (e.g. "-cl-fast-relaxed-math -D TILE=16")
    const char* opts = std::getenv("OCL_BUILD_OPTS");
    if ((err = clBuildProgram(prog, 1, &dev, opts ? opts : "", nullptr, nullptr)) != CL_SUCCESS){
        size_t logSize=0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        die("clBuildProgram failed", err);
    }

    cl_kernel krn = clCreateKernel(prog, "execute_task", &err);
    if (!krn || err) die("clCreateKernel failed", err);

    const size_t aBytes = aElems*sizeof(float);
    const size_t bBytes = bElems*sizeof(float);
    const size_t cBytes = cElems*sizeof(float);

    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, aBytes, A.data(), &err);
    if (!bufA || err) die("clCreateBuffer(A) failed", err);
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, bBytes, B.data(), &err);
    if (!bufB || err) die("clCreateBuffer(B) failed", err);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, cBytes, nullptr, &err);
    if (!bufC || err) die("clCreateBuffer(C) failed", err);

    int arg=0;
    if ((err = clSetKernelArg(krn, arg++, sizeof(i32), &rows))!=CL_SUCCESS) die("arg rows", err);
    if ((err = clSetKernelArg(krn, arg++, sizeof(i32), &K))   !=CL_SUCCESS) die("arg K", err);
    if ((err = clSetKernelArg(krn, arg++, sizeof(i32), &cols))!=CL_SUCCESS) die("arg cols", err);
    if ((err = clSetKernelArg(krn, arg++, sizeof(cl_mem), &bufA))!=CL_SUCCESS) die("arg A", err);
    if ((err = clSetKernelArg(krn, arg++, sizeof(cl_mem), &bufB))!=CL_SUCCESS) die("arg B", err);
    if ((err = clSetKernelArg(krn, arg++, sizeof(cl_mem), &bufC))!=CL_SUCCESS) die("arg C", err);

    size_t global[2] = { static_cast<size_t>(cols), static_cast<size_t>(rows) };
    // If you want to enforce a local size, set OCL_LOCAL="16,16" (or leave unset for driver-picked)
    size_t* localPtr = nullptr; size_t local[2];
    if (const char* ls = std::getenv("OCL_LOCAL")){
        int lx=0, ly=0; if (std::sscanf(ls, "%d,%d", &lx, &ly)==2 && lx>0 && ly>0){ local[0]=lx; local[1]=ly; localPtr = local; }
    }

    if ((err = clEnqueueNDRangeKernel(q, krn, 2, nullptr, global, localPtr, 0, nullptr, nullptr)) != CL_SUCCESS)
        die("clEnqueueNDRangeKernel failed", err);
    clFinish(q);

    if ((err = clEnqueueReadBuffer(q, bufC, CL_TRUE, 0, cBytes, C.data(), 0, nullptr, nullptr)) != CL_SUCCESS)
        die("clEnqueueReadBuffer failed", err);

    write_exact(C.data(), cBytes);

    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}
