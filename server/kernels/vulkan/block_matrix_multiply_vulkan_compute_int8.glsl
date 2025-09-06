// block_matrix_multiply_vulkan_compute_int8.glsl
#version 450
// Uses GL_EXT_shader_integer_dot_product (promoted to Vulkan 1.3 via VK_KHR_shader_integer_dot_product / SPV_KHR_integer_dot_product)
// Layout: A_packed [rows x groupsK] and B_packed [groupsK x cols], each 32-bit int packs 4 signed int8.
// Accumulator: 32-bit signed int; Output C is int.
#extension GL_EXT_shader_integer_dot_product : require

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer ABlock { int A_packed[]; };
layout(std430, set = 0, binding = 1) readonly buffer BBlock { int B_packed[]; };
layout(std430, set = 0, binding = 2) writeonly buffer CBlock { int C[]; };

layout(std140, set = 0, binding = 3) uniform Dims {
  int rows;
  int K;
  int cols;
  int groupsK; // == (K+3)/4
} u;

void main() {
  int c = int(gl_GlobalInvocationID.x);
  int r = int(gl_GlobalInvocationID.y);
  if (r >= u.rows || c >= u.cols) return;

  int acc = 0;
  int rowBase = r * u.groupsK;
  for (int g = 0; g < u.groupsK; ++g) {
    int a4 = A_packed[rowBase + g];
    int b4 = B_packed[g * u.cols + c];
    // Signed 8-bit dot product of packed dwords -> i32
    acc += dot4I8Packed(a4, b4);
  }

  C[r * u.cols + c] = acc;
}
