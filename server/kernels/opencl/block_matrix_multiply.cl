// block_matrix_multiply_opencl_kernel_shared.cl

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
