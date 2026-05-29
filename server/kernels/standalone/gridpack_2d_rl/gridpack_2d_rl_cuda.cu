// Standalone CUDA GridPack-2D RL with Evolution Strategies
// Single-GPU evaluation of MLP policy for floor planning.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_70 -o gridpack-2d-rl-cuda gridpack_2d_rl_cuda.cu
//
// Example (base evaluation):
//   ./gridpack-2d-rl-cuda \
//     --gridW=16 --gridH=16 --numBlocks=6 \
//     --blocks=3,4,2,2,5,3,4,4,3,2,2,5 \
//     --theta=theta.bin --epsilon=epsilon.bin \
//     --numThreads=256 --rollouts=64 --mode=0 \
//     --sigma=0.01 --seed=12345 \
//     --mlpHiddenDim=32 --actionRegions=8
//
// Output: JSON with meanReward, validCount, wallTimeMs, etc.

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err__) << std::endl; \
        std::exit(1); \
    } \
} while (0)

using u32 = uint32_t;
using f32 = float;

// ------------------------------------------------------------------
// Compile-time MAXIMUMS (must be >= any runtime value used)
// ------------------------------------------------------------------
#ifndef MAX_STATE_DIM
#define MAX_STATE_DIM 8
#endif
#ifndef MAX_HIDDEN_DIM
#define MAX_HIDDEN_DIM 128
#endif
#ifndef MAX_ACTION_DIM
#define MAX_ACTION_DIM 256
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 32
#endif
#ifndef MAX_GRID_H
#define MAX_GRID_H 64
#endif

// ------------------------------------------------------------------
// Device RNG: match the WebGPU single-word xorshift path.
// ------------------------------------------------------------------
struct RngState {
    u32 s;
};

__device__ inline u32 randU32(RngState* st) {
    u32 x = st->s;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    st->s = x;
    return x;
}

__device__ inline f32 randF01(RngState* st) {
    return f32(randU32(st)) / 4294967295.0f;
}

// ------------------------------------------------------------------
// Grid bitmask helpers
// ------------------------------------------------------------------
__device__ inline u32 rowMask(u32 x, u32 w) {
    if (w >= 32u) return 0xFFFFFFFFu << x;
    return ((1u << w) - 1u) << x;
}

__device__ inline bool canPlace(const u32* g, u32 x, u32 y, u32 w, u32 h, u32 gw, u32 gh) {
    if (x + w > gw || y + h > gh) return false;
    u32 m = rowMask(x, w);
    for (u32 r = y; r < y + h; ++r) {
        if ((g[r] & m) != 0u) return false;
    }
    return true;
}

__device__ inline void place(u32* g, u32 x, u32 y, u32 w, u32 h) {
    u32 m = rowMask(x, w);
    for (u32 r = y; r < y + h; ++r) g[r] |= m;
}

// ------------------------------------------------------------------
// MLP forward pass (runtime dimensions, max-cap arrays)
// ------------------------------------------------------------------
struct BlockDef { u32 w, h; };

__device__ void mlpForward(const f32* theta, const f32* epsilon, f32 sigma, bool useEps,
                           u32 stateDim, u32 hiddenDim, u32 actionDim,
                           const f32* st, f32* logits) {
    f32 hidden[MAX_HIDDEN_DIM];
    // Layer 1: stateDim -> hiddenDim
    u32 b1Size = stateDim * hiddenDim;
    u32 b1BiasOff = b1Size;
    for (u32 i = 0; i < hiddenDim; ++i) {
        f32 sum = theta[b1BiasOff + i];
        if (useEps) sum += sigma * epsilon[b1BiasOff + i];
        for (u32 j = 0; j < stateDim; ++j) {
            f32 w = theta[i * stateDim + j];
            if (useEps) w += sigma * epsilon[i * stateDim + j];
            sum += st[j] * w;
        }
        hidden[i] = fmaxf(sum, 0.0f);
    }
    // Layer 2: hiddenDim -> actionDim
    u32 l2Off = b1Size + hiddenDim;
    u32 b2BiasOff = l2Off + hiddenDim * actionDim;
    for (u32 i = 0; i < actionDim; ++i) {
        f32 sum = theta[b2BiasOff + i];
        if (useEps) sum += sigma * epsilon[b2BiasOff + i];
        for (u32 j = 0; j < hiddenDim; ++j) {
            f32 w = theta[l2Off + i * hiddenDim + j];
            if (useEps) w += sigma * epsilon[l2Off + i * hiddenDim + j];
            sum += hidden[j] * w;
        }
        logits[i] = sum;
    }
}

// ------------------------------------------------------------------
// Action sampling (softmax over actionDim regions)
// ------------------------------------------------------------------
__device__ void sampleAction(const f32* logits, RngState* rng, u32 actionDim, u32 gw, u32 gh, u32& px, u32& py) {
    f32 mx = logits[0];
    for (u32 i = 1; i < actionDim; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    f32 expSum = 0.0f;
    f32 probs[MAX_ACTION_DIM];
    for (u32 i = 0; i < actionDim; ++i) {
        f32 e = expf(logits[i] - mx);
        probs[i] = e;
        expSum += e;
    }
    f32 r = randF01(rng) * expSum;
    f32 c = 0.0f;
    u32 a = 0;
    for (u32 i = 0; i < actionDim; ++i) {
        c += probs[i];
        if (c >= r) { a = i; break; }
    }
    u32 regionCount = u32(sqrtf(f32(actionDim)) + 0.5f);
    u32 rx = a % regionCount;
    u32 ry = a / regionCount;
    u32 rw = max(1u, gw / regionCount);
    u32 rh = max(1u, gh / regionCount);
    px = rx * rw + u32(randF01(rng) * f32(rw));
    py = ry * rh + u32(randF01(rng) * f32(rh));
    if (px >= gw) px = gw - 1;
    if (py >= gh) py = gh - 1;
}

// ------------------------------------------------------------------
// Feature extraction
// ------------------------------------------------------------------
__device__ void extractFeatures(const u32* g, u32 gw, u32 gh, u32 bw, u32 bh,
                                u32 left, u32 total, u32 stateDim, f32* feat) {
    u32 occ = 0;
    for (u32 y = 0; y < gh; ++y) occ += __popc(g[y]);
    u32 mix = gw, miy = gh, mxx = 0, myy = 0;
    for (u32 y = 0; y < gh; ++y) {
        u32 row = g[y];
        if (row == 0u) continue;
        if (y < miy) miy = y;
        if (y > myy) myy = y;
        for (u32 x = 0; x < gw; ++x) {
            if ((row & (1u << x)) != 0u) {
                if (x < mix) mix = x;
                if (x > mxx) mxx = x;
            }
        }
    }
    u32 bx = gw, by = gh;
    if (mxx >= mix) {
        bx = mxx - mix + 1;
        by = myy - miy + 1;
    }
    feat[0] = f32(bw) / f32(gw);
    feat[1] = f32(bh) / f32(gh);
    feat[2] = f32(occ) / f32(gw * gh);
    feat[3] = f32(bx) / f32(gw);
    feat[4] = f32(by) / f32(gh);
    feat[5] = f32(left) / f32(total);
    feat[6] = f32(gw) / f32(gh);
    feat[7] = 1.0f;
}

// ------------------------------------------------------------------
// Single rollout
// ------------------------------------------------------------------
__device__ f32 rollout(u32 rolloutSeed, u32 gw, u32 gh, u32 nb, u32 ma, u32 ar,
                       u32 stateDim, u32 hiddenDim, u32 actionDim,
                       const BlockDef* blocks, const f32* theta, const f32* epsilon,
                       f32 sigma, bool useEps) {
    RngState rng{rolloutSeed};
    u32 grid[MAX_GRID_H];
    for (u32 i = 0; i < gh; ++i) grid[i] = 0u;
    bool valid = true;

    for (u32 b = 0; b < nb; ++b) {
        u32 bw = blocks[b].w;
        u32 bh = blocks[b].h;
        if (bw > gw || bh > gh) { valid = false; break; }
        if (ar != 0u && (randU32(&rng) % 2u) == 1u) {
            u32 t = bw; bw = bh; bh = t;
            if (bw > gw || bh > gh) { t = bw; bw = bh; bh = t; }
        }
        f32 feat[MAX_STATE_DIM];
        extractFeatures(grid, gw, gh, bw, bh, nb - b, nb, stateDim, feat);
        f32 logits[MAX_ACTION_DIM];
        mlpForward(theta, epsilon, sigma, useEps, stateDim, hiddenDim, actionDim, feat, logits);
        bool placed = false;
        for (u32 a = 0; a < ma; ++a) {
            u32 px, py;
            sampleAction(logits, &rng, actionDim, gw, gh, px, py);
            if (canPlace(grid, px, py, bw, bh, gw, gh)) {
                place(grid, px, py, bw, bh);
                placed = true;
                break;
            }
        }
        if (!placed) { valid = false; break; }
    }
    if (!valid) return -1.0f;

    u32 mix = gw, miy = gh, mxx = 0, myy = 0;
    f32 blockArea = 0.0f;
    for (u32 y = 0; y < gh; ++y) {
        u32 row = grid[y];
        if (row == 0u) continue;
        for (u32 x = 0; x < gw; ++x) {
            if ((row & (1u << x)) != 0u) {
                blockArea += 1.0f;
                if (x < mix) mix = x;
                if (y < miy) miy = y;
                if (x > mxx) mxx = x;
                if (y > myy) myy = y;
            }
        }
    }
    u32 bx = mxx - mix + 1;
    u32 by = myy - miy + 1;
    f32 bboxArea = f32(bx * by);
    if (bboxArea > 0.0f) return 1.0f + blockArea / bboxArea;
    return 1.0f;
}

// ------------------------------------------------------------------
// Per-thread kernel
// ------------------------------------------------------------------
__global__ void gridpackKernel(u32 numThreads, u32 numRollouts, u32 gw, u32 gh,
                               u32 nb, u32 ma, u32 ar, u32 baseSeed, f32 sigma,
                               u32 mode, u32 stateDim, u32 hiddenDim, u32 actionDim,
                               const BlockDef* blocks,
                               const f32* theta, const f32* epsilon,
                               f32* outMean, u32* outValid) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numThreads) return;

    RngState rng{baseSeed + idx * 7919u + 104729u};
    for (u32 i = 0; i < 4; ++i) randU32(&rng);

    bool useEps = (mode == 1u);
    f32 sumR = 0.0f;
    u32 validN = 0;
    for (u32 r = 0; r < numRollouts; ++r) {
        u32 rs = randU32(&rng);
        f32 reward = rollout(rs, gw, gh, nb, ma, ar, stateDim, hiddenDim, actionDim,
                             blocks, theta, epsilon, sigma, useEps);
        if (reward > 0.0f) { sumR += reward; validN++; }
    }
    outMean[idx] = (validN > 0) ? (sumR / f32(validN)) : -1.0f;
    outValid[idx] = validN;
}

// ------------------------------------------------------------------
// Host helpers
// ------------------------------------------------------------------
static bool parseBlocks(const std::string& s, std::vector<BlockDef>& out) {
    std::stringstream ss(s);
    std::string token;
    std::vector<u32> vals;
    while (std::getline(ss, token, ',')) {
        vals.push_back(std::stoul(token));
    }
    if (vals.size() % 2 != 0) return false;
    for (size_t i = 0; i < vals.size(); i += 2) {
        out.push_back({vals[i], vals[i+1]});
    }
    return true;
}

static std::vector<f32> readFloatFile(const std::string& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; std::exit(1); }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz / sizeof(f32) != expected) {
        std::cerr << "File size mismatch: " << path << " expected " << expected << " floats, got " << sz/sizeof(f32) << std::endl;
        std::exit(1);
    }
    std::vector<f32> buf(expected);
    f.read(reinterpret_cast<char*>(buf.data()), expected * sizeof(f32));
    return buf;
}

static void printUsage(const char* name) {
    std::cout << "Usage: " << name << " [options]\n"
              << "  --gridW=N            Grid width (default 16)\n"
              << "  --gridH=N            Grid height (default 16)\n"
              << "  --numBlocks=N        Number of blocks (default 6)\n"
              << "  --blocks=W1,H1,W2,H2,...  Block sizes (comma-separated)\n"
              << "  --theta=FILE         Binary float32 theta weights\n"
              << "  --epsilon=FILE       Binary float32 epsilon noise\n"
              << "  --numThreads=N       GPU threads (default 256)\n"
              << "  --rollouts=N         Rollouts per thread (default 64)\n"
              << "  --maxAttempts=N      Max placement retries (default 50)\n"
              << "  --allowRotation=0|1  Allow block rotation (default 1)\n"
              << "  --mode=0|1           0=base, 1=perturbed (default 0)\n"
              << "  --sigma=F            ES noise scale (default 0.01)\n"
              << "  --seed=N             Random seed (default 12345)\n"
              << "  --mlpHiddenDim=N     MLP hidden layer size (default 32)\n"
              << "  --actionRegions=N    Action region grid, actionDim = N^2 (default 8)\n"
              << "  --output=FILE        Output JSON file (default stdout)\n";
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main(int argc, char** argv) {
    u32 gridW = 16, gridH = 16, numBlocks = 6;
    u32 numThreads = 1024, numRollouts = 128, maxAttempts = 50;
    u32 allowRotation = 1, mode = 0, seed = 12345;
    u32 mlpHiddenDim = 32, actionRegions = 8;
    f32 sigma = 0.01f;
    std::string thetaPath, epsilonPath, outputPath;
    std::vector<BlockDef> hostBlocks;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg.rfind("--gridW=", 0) == 0) gridW = std::stoul(arg.substr(8));
        else if (arg.rfind("--gridH=", 0) == 0) gridH = std::stoul(arg.substr(8));
        else if (arg.rfind("--numBlocks=", 0) == 0) numBlocks = std::stoul(arg.substr(12));
        else if (arg.rfind("--blocks=", 0) == 0) {
            if (!parseBlocks(arg.substr(9), hostBlocks)) {
                std::cerr << "Invalid --blocks format" << std::endl; return 1;
            }
        }
        else if (arg.rfind("--theta=", 0) == 0) thetaPath = arg.substr(8);
        else if (arg.rfind("--epsilon=", 0) == 0) epsilonPath = arg.substr(10);
        else if (arg.rfind("--numThreads=", 0) == 0) numThreads = std::stoul(arg.substr(13));
        else if (arg.rfind("--rollouts=", 0) == 0) numRollouts = std::stoul(arg.substr(11));
        else if (arg.rfind("--maxAttempts=", 0) == 0) maxAttempts = std::stoul(arg.substr(14));
        else if (arg.rfind("--allowRotation=", 0) == 0) allowRotation = std::stoul(arg.substr(16));
        else if (arg.rfind("--mode=", 0) == 0) mode = std::stoul(arg.substr(7));
        else if (arg.rfind("--sigma=", 0) == 0) sigma = std::stof(arg.substr(8));
        else if (arg.rfind("--seed=", 0) == 0) seed = std::stoul(arg.substr(7));
        else if (arg.rfind("--mlpHiddenDim=", 0) == 0) mlpHiddenDim = std::stoul(arg.substr(15));
        else if (arg.rfind("--actionRegions=", 0) == 0) actionRegions = std::stoul(arg.substr(16));
        else if (arg.rfind("--output=", 0) == 0) outputPath = arg.substr(9);
        else { std::cerr << "Unknown arg: " << arg << std::endl; return 1; }
    }

    u32 actionDim = actionRegions * actionRegions;
    if (mlpHiddenDim > MAX_HIDDEN_DIM) {
        std::cerr << "mlpHiddenDim " << mlpHiddenDim << " exceeds compile-time MAX_HIDDEN_DIM " << MAX_HIDDEN_DIM << std::endl;
        return 1;
    }
    if (actionDim > MAX_ACTION_DIM) {
        std::cerr << "actionDim " << actionDim << " exceeds compile-time MAX_ACTION_DIM " << MAX_ACTION_DIM << std::endl;
        return 1;
    }
    u32 stateDim = 8; // fixed
    u32 thetaSize = stateDim * mlpHiddenDim + mlpHiddenDim + mlpHiddenDim * actionDim + actionDim;

    if (hostBlocks.empty()) {
        u32 rng = seed;
        for (u32 i = 0; i < numBlocks; ++i) {
            u32 w = 2 + (rng % 5); rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            u32 h = 2 + (rng % 5); rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            hostBlocks.push_back({w, h});
        }
    }
    numBlocks = hostBlocks.size();

    // Load theta and epsilon
    std::vector<f32> hTheta, hEpsilon;
    if (thetaPath.empty()) {
        hTheta.resize(thetaSize, 0.0f);
    } else {
        hTheta = readFloatFile(thetaPath, thetaSize);
    }
    if (epsilonPath.empty()) {
        hEpsilon.resize(thetaSize, 0.0f);
    } else {
        hEpsilon = readFloatFile(epsilonPath, thetaSize);
    }

    // Start wall-clock timer (includes allocations, H2D, kernel, D2H)
    auto tStart = std::chrono::high_resolution_clock::now();

    // Device allocations
    BlockDef* dBlocks;
    f32* dTheta;
    f32* dEpsilon;
    f32* dOutMean;
    u32* dOutValid;
    CUDA_CHECK(cudaMalloc(&dBlocks, numBlocks * sizeof(BlockDef)));
    CUDA_CHECK(cudaMalloc(&dTheta, thetaSize * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dEpsilon, thetaSize * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dOutMean, numThreads * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dOutValid, numThreads * sizeof(u32)));

    CUDA_CHECK(cudaMemcpy(dBlocks, hostBlocks.data(), numBlocks * sizeof(BlockDef), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTheta, hTheta.data(), thetaSize * sizeof(f32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEpsilon, hEpsilon.data(), thetaSize * sizeof(f32), cudaMemcpyHostToDevice));

    // Launch
    u32 blockSize = 256;
    u32 gridSize = (numThreads + blockSize - 1) / blockSize;
    gridpackKernel<<<gridSize, blockSize>>>(numThreads, numRollouts, gridW, gridH,
                                            numBlocks, maxAttempts, allowRotation,
                                            seed, sigma, mode, stateDim, mlpHiddenDim, actionDim,
                                            dBlocks, dTheta, dEpsilon,
                                            dOutMean, dOutValid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduce on host
    std::vector<f32> hMean(numThreads);
    std::vector<u32> hValid(numThreads);
    CUDA_CHECK(cudaMemcpy(hMean.data(), dOutMean, numThreads * sizeof(f32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hValid.data(), dOutValid, numThreads * sizeof(u32), cudaMemcpyDeviceToHost));

    f32 totalReward = 0.0f;
    u32 totalValid = 0;
    u32 activeThreads = 0;
    for (u32 i = 0; i < numThreads; ++i) {
        if (hValid[i] > 0) {
            totalReward += hMean[i] * hValid[i];
            totalValid += hValid[i];
            activeThreads++;
        }
    }
    f32 overallMean = (totalValid > 0) ? (totalReward / totalValid) : -1.0f;
    u32 totalRollouts = numThreads * numRollouts;

    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    // Output JSON
    std::ostringstream json;
    json << "{\n";
    json << "  \"meanReward\": " << overallMean << ",\n";
    json << "  \"validCount\": " << totalValid << ",\n";
    json << "  \"totalRollouts\": " << (numThreads * numRollouts) << ",\n";
    json << "  \"activeThreads\": " << activeThreads << ",\n";
    json << "  \"mode\": " << mode << ",\n";
    json << "  \"gridW\": " << gridW << ",\n";
    json << "  \"gridH\": " << gridH << ",\n";
    json << "  \"numBlocks\": " << numBlocks << ",\n";
    json << "  \"numThreads\": " << numThreads << ",\n";
    json << "  \"sigma\": " << sigma << ",\n";
    json << "  \"mlpHiddenDim\": " << mlpHiddenDim << ",\n";
    json << "  \"actionRegions\": " << actionRegions << ",\n";
    json << "  \"actionDim\": " << actionDim << ",\n";
    json << "  \"thetaSize\": " << thetaSize << ",\n";
    json << "  \"wallTimeMs\": " << elapsedMs << "\n";
    json << "}\n";

    if (!outputPath.empty()) {
        std::ofstream ofs(outputPath);
        ofs << json.str();
    } else {
        std::cout << json.str();
    }

    cudaFree(dBlocks); cudaFree(dTheta); cudaFree(dEpsilon);
    cudaFree(dOutMean); cudaFree(dOutValid);
    return 0;
}
