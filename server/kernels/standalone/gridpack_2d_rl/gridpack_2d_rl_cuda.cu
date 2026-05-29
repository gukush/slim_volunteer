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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
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
using u64 = uint64_t;
using f32 = float;

// ------------------------------------------------------------------
// Compile-time MAXIMUMS (must be >= any runtime value used)
// ------------------------------------------------------------------
#ifndef MAX_STATE_DIM
#define MAX_STATE_DIM 8
#endif
#ifndef MAX_HIDDEN_DIM
#define MAX_HIDDEN_DIM 32
#endif
#ifndef MAX_ACTION_DIM
#define MAX_ACTION_DIM 64
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 16
#endif
#ifndef MAX_GRID_H
#define MAX_GRID_H 32
#endif

static constexpr u32 STATE_DIM = 8;
static constexpr u32 HIDDEN_DIM = 32;
static constexpr u32 ACTION_REGIONS = 8;
static constexpr u32 ACTION_DIM = ACTION_REGIONS * ACTION_REGIONS;
static constexpr u32 L1_BIAS_OFF = STATE_DIM * HIDDEN_DIM;
static constexpr u32 L2_WEIGHTS_OFF = L1_BIAS_OFF + HIDDEN_DIM;
static constexpr u32 L2_BIAS_OFF = L2_WEIGHTS_OFF + HIDDEN_DIM * ACTION_DIM;
static constexpr u32 THETA_SIZE = L2_BIAS_OFF + ACTION_DIM;

struct EvalResult {
    f32 meanReward;
    u32 validCount;
    u32 activeThreads;
};

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
// MLP forward pass: fixed 8 -> 32 -> 64 architecture, thetaSize=2400.
// ------------------------------------------------------------------
struct BlockDef { u32 w, h; };

__device__ void mlpForward(const f32* theta, const f32* epsilon, f32 sigma, bool useEps,
                           const f32* st, f32* logits) {
    f32 hidden[HIDDEN_DIM];
    for (u32 i = 0; i < HIDDEN_DIM; ++i) {
        f32 sum = theta[L1_BIAS_OFF + i];
        if (useEps) sum += sigma * epsilon[L1_BIAS_OFF + i];
        for (u32 j = 0; j < STATE_DIM; ++j) {
            f32 w = theta[i * STATE_DIM + j];
            if (useEps) w += sigma * epsilon[i * STATE_DIM + j];
            sum += st[j] * w;
        }
        hidden[i] = fmaxf(sum, 0.0f);
    }
    for (u32 i = 0; i < ACTION_DIM; ++i) {
        f32 sum = theta[L2_BIAS_OFF + i];
        if (useEps) sum += sigma * epsilon[L2_BIAS_OFF + i];
        for (u32 j = 0; j < HIDDEN_DIM; ++j) {
            f32 w = theta[L2_WEIGHTS_OFF + i * HIDDEN_DIM + j];
            if (useEps) w += sigma * epsilon[L2_WEIGHTS_OFF + i * HIDDEN_DIM + j];
            sum += hidden[j] * w;
        }
        logits[i] = sum;
    }
}

// ------------------------------------------------------------------
// Action sampling (softmax over actionDim regions)
// ------------------------------------------------------------------
__device__ void sampleAction(const f32* logits, RngState* rng, u32 gw, u32 gh, u32& px, u32& py) {
    f32 mx = logits[0];
    for (u32 i = 1; i < ACTION_DIM; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    f32 expSum = 0.0f;
    f32 probs[ACTION_DIM];
    for (u32 i = 0; i < ACTION_DIM; ++i) {
        f32 e = __expf(logits[i] - mx);
        probs[i] = e;
        expSum += e;
    }
    f32 r = randF01(rng) * expSum;
    f32 c = 0.0f;
    u32 a = 0;
    for (u32 i = 0; i < ACTION_DIM; ++i) {
        c += probs[i];
        if (c >= r) { a = i; break; }
    }
    u32 rx = a % ACTION_REGIONS;
    u32 ry = a / ACTION_REGIONS;
    u32 rw = max(1u, gw / ACTION_REGIONS);
    u32 rh = max(1u, gh / ACTION_REGIONS);
    px = rx * rw + u32(randF01(rng) * f32(rw));
    py = ry * rh + u32(randF01(rng) * f32(rh));
    if (px >= gw) px = gw - 1;
    if (py >= gh) py = gh - 1;
}

// ------------------------------------------------------------------
// Feature extraction
// ------------------------------------------------------------------
__device__ void extractFeatures(const u32* g, u32 gw, u32 gh, u32 bw, u32 bh,
                                u32 left, u32 total, f32* feat) {
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
        f32 feat[STATE_DIM];
        extractFeatures(grid, gw, gh, bw, bh, nb - b, nb, feat);
        f32 logits[ACTION_DIM];
        mlpForward(theta, epsilon, sigma, useEps, feat, logits);
        bool placed = false;
        for (u32 a = 0; a < ma; ++a) {
            u32 px, py;
            sampleAction(logits, &rng, gw, gh, px, py);
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
                               u32 mode,
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
        f32 reward = rollout(rs, gw, gh, nb, ma, ar, blocks, theta, epsilon, sigma, useEps);
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

static std::vector<BlockDef> generateBlocks(u32 numBlocks, u32 minSize, u32 maxSize, u32& rng) {
    std::vector<BlockDef> out;
    auto xorshift = [&rng]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return (rng % 0x7FFFFFFFu);
    };
    u32 range = maxSize - minSize + 1;
    for (u32 i = 0; i < numBlocks; ++i) {
        u32 w = minSize + (xorshift() % range);
        u32 h = minSize + (xorshift() % range);
        out.push_back({w, h});
    }
    return out;
}

static void ensureFitsGrid(std::vector<BlockDef>& blocks, u32 gridW, u32 gridH) {
    u32 maxArea = gridW * gridH;
    u32 totalArea = 0;
    for (const auto& b : blocks) totalArea += b.w * b.h;

    if (totalArea > maxArea && totalArea > 0) {
        f32 scale = std::sqrt(f32(maxArea) / (f32(totalArea) * 1.2f));
        for (auto& b : blocks) {
            b.w = std::max(1u, static_cast<u32>(std::floor(f32(b.w) * scale)));
            b.h = std::max(1u, static_cast<u32>(std::floor(f32(b.h) * scale)));
        }
    }

    for (auto& b : blocks) {
        if (b.w > gridW) b.w = gridW;
        if (b.h > gridH) b.h = gridH;
    }
}

static u32 stringHashSum(const std::string& s) {
    u32 sum = 0;
    for (unsigned char c : s) sum += c;
    return sum;
}

static std::vector<f32> initThetaXavier(u32 thetaSeed) {
    std::vector<f32> theta(THETA_SIZE);
    std::mt19937 gen(thetaSeed);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    f32 scale = std::sqrt(2.0f / f32(STATE_DIM + HIDDEN_DIM));
    for (u32 i = 0; i < THETA_SIZE; ++i) theta[i] = dist(gen) * scale;
    return theta;
}

static std::vector<f32> generateEpsilon(u32 epsilonSeed) {
    std::vector<f32> epsilon(THETA_SIZE);
    u32 rng = epsilonSeed;
    auto xorshift = [&rng]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    auto rand01 = [&]() {
        return f32(xorshift()) / 4294967295.0f;
    };
    auto randN = [&]() {
        f32 u1 = std::max(rand01(), 0.0001f);
        f32 u2 = rand01();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
    };
    for (u32 i = 0; i < THETA_SIZE; ++i) epsilon[i] = randN();
    return epsilon;
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
              << "  --numProblems=N      Run full ES round with N base/perturbed pairs\n"
              << "  --minBlockSize=N     Minimum generated block edge (default 2)\n"
              << "  --maxBlockSize=N     Maximum generated block edge (default 6)\n"
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
              << "  --thetaSeed=N        Default theta init seed when --theta is omitted (default 42)\n"
              << "  --epsilonSeed=N      Default epsilon seed when --epsilon is omitted (default --seed)\n"
              << "  --taskId=STRING      Optional WebGPU-compatible task hash seed offset\n"
              << "  --taskIdHash=N       Optional numeric task hash seed offset\n"
              << "  --mlpHiddenDim=32    Fixed MLP hidden layer size\n"
              << "  --actionRegions=8    Fixed action region grid, actionDim=64\n"
              << "  --output=FILE        Output JSON file (default stdout)\n";
}

static EvalResult runEvaluation(BlockDef* dBlocks, f32* dTheta, f32* dEpsilon,
                                f32* dOutMean, u32* dOutValid,
                                const std::vector<BlockDef>& hostBlocks,
                                u32 gridW, u32 gridH, u32 maxAttempts, u32 allowRotation,
                                u32 numThreads, u32 numRollouts, u32 seed, f32 sigma, u32 mode) {
    const u32 numBlocks = static_cast<u32>(hostBlocks.size());
    CUDA_CHECK(cudaMemcpy(dBlocks, hostBlocks.data(), numBlocks * sizeof(BlockDef), cudaMemcpyHostToDevice));

    u32 blockSize = 64;
    u32 gridSize = (numThreads + blockSize - 1) / blockSize;
    gridpackKernel<<<gridSize, blockSize>>>(numThreads, numRollouts, gridW, gridH,
                                            numBlocks, maxAttempts, allowRotation,
                                            seed, sigma, mode,
                                            dBlocks, dTheta, dEpsilon,
                                            dOutMean, dOutValid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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

    return { (totalValid > 0) ? (totalReward / totalValid) : -1.0f, totalValid, activeThreads };
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main(int argc, char** argv) {
    u32 gridW = 16, gridH = 16, numBlocks = 6;
    u32 minBlockSize = 2, maxBlockSize = 6;
    u32 numThreads = 1024, numRollouts = 128, maxAttempts = 50;
    u32 allowRotation = 1, mode = 0, seed = 12345;
    u32 numProblems = 0, taskIdHash = 0;
    u32 thetaSeed = 42, epsilonSeed = 0;
    bool hasEpsilonSeed = false;
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
        else if (arg.rfind("--numProblems=", 0) == 0) numProblems = std::stoul(arg.substr(14));
        else if (arg.rfind("--minBlockSize=", 0) == 0) minBlockSize = std::stoul(arg.substr(15));
        else if (arg.rfind("--maxBlockSize=", 0) == 0) maxBlockSize = std::stoul(arg.substr(15));
        else if (arg.rfind("--blocks=", 0) == 0) {
            if (!parseBlocks(arg.substr(9), hostBlocks)) {
                std::cerr << "Invalid --blocks format" << std::endl; return 1;
            }
        }
        else if (arg.rfind("--theta=", 0) == 0) thetaPath = arg.substr(8);
        else if (arg.rfind("--epsilon=", 0) == 0) epsilonPath = arg.substr(10);
        else if (arg.rfind("--numThreads=", 0) == 0) numThreads = std::stoul(arg.substr(13));
        else if (arg.rfind("--rollouts=", 0) == 0) numRollouts = std::stoul(arg.substr(11));
        else if (arg.rfind("--rolloutsPerThread=", 0) == 0) numRollouts = std::stoul(arg.substr(20));
        else if (arg.rfind("--maxAttempts=", 0) == 0) maxAttempts = std::stoul(arg.substr(14));
        else if (arg.rfind("--allowRotation=", 0) == 0) allowRotation = std::stoul(arg.substr(16));
        else if (arg.rfind("--mode=", 0) == 0) mode = std::stoul(arg.substr(7));
        else if (arg.rfind("--sigma=", 0) == 0) sigma = std::stof(arg.substr(8));
        else if (arg.rfind("--seed=", 0) == 0) seed = std::stoul(arg.substr(7));
        else if (arg.rfind("--thetaSeed=", 0) == 0) thetaSeed = std::stoul(arg.substr(12));
        else if (arg.rfind("--epsilonSeed=", 0) == 0) { epsilonSeed = std::stoul(arg.substr(14)); hasEpsilonSeed = true; }
        else if (arg.rfind("--taskIdHash=", 0) == 0) taskIdHash = std::stoul(arg.substr(13));
        else if (arg.rfind("--taskId=", 0) == 0) taskIdHash = stringHashSum(arg.substr(9));
        else if (arg.rfind("--mlpHiddenDim=", 0) == 0) mlpHiddenDim = std::stoul(arg.substr(15));
        else if (arg.rfind("--actionRegions=", 0) == 0) actionRegions = std::stoul(arg.substr(16));
        else if (arg.rfind("--output=", 0) == 0) outputPath = arg.substr(9);
        else { std::cerr << "Unknown arg: " << arg << std::endl; return 1; }
    }

    if (mlpHiddenDim != HIDDEN_DIM) {
        std::cerr << "gridpack-2d-rl now uses fixed mlpHiddenDim=" << HIDDEN_DIM << "; got " << mlpHiddenDim << std::endl;
        return 1;
    }
    if (actionRegions != ACTION_REGIONS) {
        std::cerr << "gridpack-2d-rl now uses fixed actionRegions=" << ACTION_REGIONS << "; got " << actionRegions << std::endl;
        return 1;
    }
    if (gridW > MAX_GRID_H || gridH > MAX_GRID_H) {
        std::cerr << "grid dimensions exceed MAX_GRID_H " << MAX_GRID_H << std::endl;
        return 1;
    }
    if (numBlocks > MAX_BLOCKS) {
        std::cerr << "numBlocks " << numBlocks << " exceeds MAX_BLOCKS " << MAX_BLOCKS << std::endl;
        return 1;
    }
    if (minBlockSize == 0 || maxBlockSize < minBlockSize) {
        std::cerr << "Invalid block size range" << std::endl;
        return 1;
    }

    if (numProblems == 0 && hostBlocks.empty()) {
        u32 rng = seed;
        hostBlocks = generateBlocks(numBlocks, minBlockSize, maxBlockSize, rng);
    }
    if (numProblems == 0) {
        ensureFitsGrid(hostBlocks, gridW, gridH);
        numBlocks = hostBlocks.size();
        if (numBlocks > MAX_BLOCKS) {
            std::cerr << "parsed block count " << numBlocks << " exceeds MAX_BLOCKS " << MAX_BLOCKS << std::endl;
            return 1;
        }
    } else if (!hostBlocks.empty()) {
        ensureFitsGrid(hostBlocks, gridW, gridH);
        numBlocks = hostBlocks.size();
        if (numBlocks > MAX_BLOCKS) {
            std::cerr << "parsed block count " << numBlocks << " exceeds MAX_BLOCKS " << MAX_BLOCKS << std::endl;
            return 1;
        }
    }

    // Load theta and epsilon
    std::vector<f32> hTheta, hEpsilon;
    if (thetaPath.empty()) {
        hTheta = initThetaXavier(thetaSeed);
    } else {
        hTheta = readFloatFile(thetaPath, THETA_SIZE);
    }
    if (epsilonPath.empty()) {
        hEpsilon = generateEpsilon(hasEpsilonSeed ? epsilonSeed : seed);
    } else {
        hEpsilon = readFloatFile(epsilonPath, THETA_SIZE);
    }

    // Start wall-clock timer (includes allocations, H2D, kernel, D2H)
    auto tStart = std::chrono::high_resolution_clock::now();

    // Device allocations
    BlockDef* dBlocks;
    f32* dTheta;
    f32* dEpsilon;
    f32* dOutMean;
    u32* dOutValid;
    CUDA_CHECK(cudaMalloc(&dBlocks, MAX_BLOCKS * sizeof(BlockDef)));
    CUDA_CHECK(cudaMalloc(&dTheta, THETA_SIZE * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dEpsilon, THETA_SIZE * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dOutMean, numThreads * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dOutValid, numThreads * sizeof(u32)));

    CUDA_CHECK(cudaMemcpy(dTheta, hTheta.data(), THETA_SIZE * sizeof(f32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEpsilon, hEpsilon.data(), THETA_SIZE * sizeof(f32), cudaMemcpyHostToDevice));

    std::ostringstream json;
    if (numProblems > 0) {
        f32 sumDelta = 0.0f;
        f32 sumBase = 0.0f;
        f32 sumPerturbed = 0.0f;
        u32 validPairs = 0;
        u32 chunksProcessed = 0;
        u64 totalRolloutsAll = 0;

        for (u32 problemIdx = 0; problemIdx < numProblems; ++problemIdx) {
            u32 problemSeed = 123456789u + problemIdx * 747796405u + taskIdHash;
            std::vector<BlockDef> blocks;
            if (!hostBlocks.empty()) {
                blocks = hostBlocks;
            } else {
                u32 rng = problemSeed;
                blocks = generateBlocks(numBlocks, minBlockSize, maxBlockSize, rng);
                ensureFitsGrid(blocks, gridW, gridH);
            }

            EvalResult base = runEvaluation(dBlocks, dTheta, dEpsilon, dOutMean, dOutValid,
                                            blocks, gridW, gridH, maxAttempts, allowRotation,
                                            numThreads, numRollouts, problemSeed, sigma, 0);
            EvalResult perturbed = runEvaluation(dBlocks, dTheta, dEpsilon, dOutMean, dOutValid,
                                                 blocks, gridW, gridH, maxAttempts, allowRotation,
                                                 numThreads, numRollouts, problemSeed + 1u, sigma, 1);
            chunksProcessed += 2;
            totalRolloutsAll += static_cast<u64>(numThreads) * static_cast<u64>(numRollouts) * 2ull;

            if (base.validCount > 0 && perturbed.validCount > 0 &&
                base.meanReward > 0.0f && perturbed.meanReward > 0.0f) {
                sumDelta += perturbed.meanReward - base.meanReward;
                sumBase += base.meanReward;
                sumPerturbed += perturbed.meanReward;
                validPairs++;
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        f32 meanDelta = validPairs > 0 ? sumDelta / f32(validPairs) : 0.0f;
        f32 meanBase = validPairs > 0 ? sumBase / f32(validPairs) : 0.0f;
        f32 meanPerturbed = validPairs > 0 ? sumPerturbed / f32(validPairs) : 0.0f;

        json << "{\n";
        json << "  \"validPairs\": " << validPairs << ",\n";
        json << "  \"totalPairs\": " << numProblems << ",\n";
        json << "  \"meanDelta\": " << meanDelta << ",\n";
        json << "  \"meanBase\": " << meanBase << ",\n";
        json << "  \"meanPerturbed\": " << meanPerturbed << ",\n";
        json << "  \"chunksProcessed\": " << chunksProcessed << ",\n";
        json << "  \"totalRollouts\": " << totalRolloutsAll << ",\n";
        json << "  \"gridW\": " << gridW << ",\n";
        json << "  \"gridH\": " << gridH << ",\n";
        json << "  \"numBlocks\": " << numBlocks << ",\n";
        json << "  \"numProblems\": " << numProblems << ",\n";
        json << "  \"numThreads\": " << numThreads << ",\n";
        json << "  \"rolloutsPerThread\": " << numRollouts << ",\n";
        json << "  \"sigma\": " << sigma << ",\n";
        json << "  \"mlpHiddenDim\": " << HIDDEN_DIM << ",\n";
        json << "  \"actionRegions\": " << ACTION_REGIONS << ",\n";
        json << "  \"actionDim\": " << ACTION_DIM << ",\n";
        json << "  \"thetaSize\": " << THETA_SIZE << ",\n";
        json << "  \"wallTimeMs\": " << elapsedMs << "\n";
        json << "}\n";
    } else {
        EvalResult eval = runEvaluation(dBlocks, dTheta, dEpsilon, dOutMean, dOutValid,
                                        hostBlocks, gridW, gridH, maxAttempts, allowRotation,
                                        numThreads, numRollouts, seed, sigma, mode);

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        json << "{\n";
        json << "  \"meanReward\": " << eval.meanReward << ",\n";
        json << "  \"validCount\": " << eval.validCount << ",\n";
        json << "  \"totalRollouts\": " << (numThreads * numRollouts) << ",\n";
        json << "  \"activeThreads\": " << eval.activeThreads << ",\n";
        json << "  \"mode\": " << mode << ",\n";
        json << "  \"gridW\": " << gridW << ",\n";
        json << "  \"gridH\": " << gridH << ",\n";
        json << "  \"numBlocks\": " << numBlocks << ",\n";
        json << "  \"numThreads\": " << numThreads << ",\n";
        json << "  \"rolloutsPerThread\": " << numRollouts << ",\n";
        json << "  \"sigma\": " << sigma << ",\n";
        json << "  \"mlpHiddenDim\": " << HIDDEN_DIM << ",\n";
        json << "  \"actionRegions\": " << ACTION_REGIONS << ",\n";
        json << "  \"actionDim\": " << ACTION_DIM << ",\n";
        json << "  \"thetaSize\": " << THETA_SIZE << ",\n";
        json << "  \"wallTimeMs\": " << elapsedMs << "\n";
        json << "}\n";
    }

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
