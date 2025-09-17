#version 450
#extension GL_KHR_cooperative_matrix : enable

// Cooperative matrix configuration
const uint ROWS = 16u;
const uint COLS = 16u; 
const uint K_SIZE = 16u;

// Workgroup size should match the cooperative matrix dimensions
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Dims {
    uint rows;   // M
    uint K;      // K
    uint cols;   // N
    uint _pad;   // padding to 16 bytes (std140 rule)
} u;

layout(set = 0, binding = 1, std430) buffer ABuf { float A[]; };
layout(set = 0, binding = 2, std430) buffer BBuf { float B[]; };
layout(set = 0, binding = 3, std430) buffer CBuf { float C[]; };

void main()
{
    // Calculate which matrix tile this workgroup processes
    const uint wgId = gl_WorkGroupID.x;
    const uint totalTiles = ((u.rows + ROWS - 1u) / ROWS) * ((u.cols + COLS - 1u) / COLS);
    
    if (wgId >= totalTiles) return;
    
    const uint tilesPerRow = (u.cols + COLS - 1u) / COLS;
    const uint tileRow = wgId / tilesPerRow;
    const uint tileCol = wgId % tilesPerRow;
    
    const uint startRow = tileRow * ROWS;
    const uint startCol = tileCol * COLS;
    
    // Declare cooperative matrices
    coopmat<float16_t, gl_ScopeSubgroup, ROWS, K_SIZE, gl_MatrixUseA> matA;
    coopmat<float16_t, gl_ScopeSubgroup, K_SIZE, COLS, gl_MatrixUseB> matB;
    coopmat<float16_t, gl_ScopeSubgroup, ROWS, COLS, gl_MatrixUseAccumulator> matC;
    
    // Initialize accumulator to zero
    matC = coopmat<float16_t, gl_ScopeSubgroup, ROWS, COLS, gl_MatrixUseAccumulator>(0.0hf);
    
    // Process K dimension in chunks
    const uint numKTiles = (u.K + K_SIZE - 1u) / K_SIZE;
    
    for (uint kTile = 0u; kTile < numKTiles; ++kTile) {
        const uint kStart = kTile * K_SIZE;
        
        // Load matrix A tile (ROWS x K_SIZE)
        coopMatLoad(matA, A, startRow * u.K + kStart, u.K, gl_CooperativeMatrixLayoutRowMajor);
        
        // Load matrix B tile (K_SIZE x COLS)  
        coopMatLoad(matB, B, kStart * u.cols + startCol, u.cols, gl_CooperativeMatrixLayoutRowMajor);
        
        // Perform matrix multiplication: C += A * B
        matC = coopMatMulAdd(matA, matB, matC);
    }
    
    // Store result
    coopMatStore(matC, C, startRow * u.cols + startCol, u.cols, gl_CooperativeMatrixLayoutRowMajor);
}
