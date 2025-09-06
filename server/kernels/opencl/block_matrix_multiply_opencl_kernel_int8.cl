// block_matrix_multiply_opencl_kernel_int8.cl
// Uses cl_khr_integer_dot_product (OpenCL C 3.0+) packed 4x8-bit intrinsics when available.
// Layout: A_packed [rows x groupsK] and B_packed [groupsK x cols], each uint packs 4 int8s (signed).
// Accumulator: 32-bit signed int; Output C is int.
//
// Enable extension if available; fall back to manual unpack if not.
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void matmul_int8_dp4a(__global const uint* A_packed,
                               __global const uint* B_packed,
                               __global       int* C,
                               int rows, int K, int cols, int groupsK)
{
  int c = get_global_id(0);
  int r = get_global_id(1);
  if (r >= rows || c >= cols) return;

  int acc = 0;
  int rowBase = r * groupsK;

  // Prefer packed 4x8 dot product if supported
  #if defined(__opencl_c_integer_dot_product_input_4x8bit_packed)
    for (int g = 0; g < groupsK; ++g) {
      uint a4 = A_packed[rowBase + g];
      uint b4 = B_packed[g * cols + c];
      // Signed x signed -> int (see OpenCL C 3.0 spec 6.2.2.16)
      acc += dot_4x8packed_ss_int(a4, b4);
    }
  #else
    // Fallback: manual unpack
    for (int g = 0; g < groupsK; ++g) {
      uint a4 = A_packed[rowBase + g];
      uint b4 = B_packed[g * cols + c];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        int ai = (int)((char)((a4 >> (8*i)) & 0xFF));
        int bi = (int)((char)((b4 >> (8*i)) & 0xFF));
        acc += ai * bi;
      }
    }
  #endif

  C[r * cols + c] = acc;
}
