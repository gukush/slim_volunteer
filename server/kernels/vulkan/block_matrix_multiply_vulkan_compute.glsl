
#version 450

const uint TILE = 16u;
layout(local_size_x = TILE, local_size_y = TILE, local_size_z = 1) in;
layout(set = 0, binding = 0, std140) uniform Dims {
    uint rows;   // M
    uint K;      // K
    uint cols;   // N
    uint _pad;   // padding to 16 bytes (std140 rule)
} u;

layout(set = 0, binding = 1, std430) buffer ABuf { float A[]; };
layout(set = 0, binding = 2, std430) buffer BBuf { float B[]; };
layout(set = 0, binding = 3, std430) buffer CBuf { float C[]; };

shared float Asub[TILE][TILE];
shared float Bsub[TILE][TILE];

void main()
{
    // Map XY to column/row (X = col, Y = row)
    const uint col = gl_GlobalInvocationID.x; // N dimension
    const uint row = gl_GlobalInvocationID.y; // M dimension

    // Guard against over-dispatch
    if (row >= u.rows || col >= u.cols) {
        return;
    }

    float acc = 0.0;
    const uint numTiles = (u.K + TILE - 1u) / TILE;

    const uint lx = gl_LocalInvocationID.x;
    const uint ly = gl_LocalInvocationID.y;

    for (uint t = 0u; t < numTiles; ++t) {
        // Compute the source indices for this thread's elements
        const uint aRow = row;
        const uint aCol = t * TILE + lx;

        const uint bRow = t * TILE + ly;
        const uint bCol = col;

        if (aRow < u.rows && aCol < u.K) {
            Asub[ly][lx] = A[aRow * u.K + aCol];
        } else {
            Asub[ly][lx] = 0.0;
        }

        if (bRow < u.K && bCol < u.cols) {
            Bsub[ly][lx] = B[bRow * u.cols + bCol];
        } else {
            Bsub[ly][lx] = 0.0;
        }

        barrier();
        for (uint k = 0u; k < TILE; ++k) {
            acc += Asub[ly][k] * Bsub[k][lx];
        }

        barrier();
    }
    C[row * u.cols + col] = acc;
}
