extern "C" __global__
void execute_task(  // U I O order to match executor
    const int rows,     // uniforms[0]
    const int K,        // uniforms[1]
    const int cols,     // uniforms[2]
    const float* __restrict__ A,   // inputs[0], size rows*K
    const float* __restrict__ B,   // inputs[1], size K*cols
    float* __restrict__ C          // outputs[0], size rows*cols
){

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    float acc = 0.f;
    // simple row-major matmul: [rows x K] * [K x cols] = [rows x cols]
    for (int k = 0; k < K; ++k) {
        acc += A[r * K + k] * B[k * cols + c];
    }
    C[r * cols + c] = acc;
}
