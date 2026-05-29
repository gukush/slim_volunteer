// MPI + CUDA Distributed Evolution Strategies for GridPack-2D RL
// Multi-GPU / multi-node ES training loop.
//
// Build:
//   mpicxx -x cu -O3 -arch=sm_70 -DMAX_STATE_DIM=8 -DMAX_HIDDEN_DIM=32 -DMAX_ACTION_DIM=64 \
//     -o gridpack-2d-rl-mpi gridpack_2d_rl_mpi.cu
//
// Run (single node, 4 GPUs):
//   mpirun -np 4 ./gridpack-2d-rl-mpi --gridW=16 --gridH=16 --numBlocks=6 \
//     --numThreads=1024 --rollouts=128 --rounds=100 --sigma=0.01 --lr=0.001 \
//     --mlpHiddenDim=32 --actionRegions=8
//
// Run (multi-node with hostfile):
//   mpirun --hostfile hosts.txt -np 32 ./gridpack-2d-rl-mpi ...

#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <algorithm>
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
using f32 = float;

// ------------------------------------------------------------------
// Compile-time MAXIMUMS
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

// ------------------------------------------------------------------
// Device RNG
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
// Grid helpers
// ------------------------------------------------------------------
struct BlockDef { u32 w, h; };

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
// MLP forward (runtime dims)
// ------------------------------------------------------------------
__device__ void mlpForward(const f32* theta, const f32* epsilon, f32 sigma, bool useEps,
                           u32 stateDim, u32 hiddenDim, u32 actionDim,
                           const f32* st, f32* logits) {
    f32 hidden[MAX_HIDDEN_DIM];
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
// Action sampling
// ------------------------------------------------------------------
__device__ void sampleAction(const f32* logits, RngState* rng, u32 actionDim,
                             u32 regionCount, u32 gw, u32 gh, u32& px, u32& py) {
    f32 mx = logits[0];
    for (u32 i = 1; i < actionDim; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    f32 expSum = 0.0f;
    f32 probs[MAX_ACTION_DIM];
    for (u32 i = 0; i < actionDim; ++i) {
        f32 e = __expf(logits[i] - mx);
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
// Features
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
// Rollout
// ------------------------------------------------------------------
__device__ f32 rollout(u32 rolloutSeed, u32 gw, u32 gh, u32 nb, u32 ma, u32 ar,
                       u32 stateDim, u32 hiddenDim, u32 actionDim,
                       u32 regionCount,
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
            sampleAction(logits, &rng, actionDim, regionCount, gw, gh, px, py);
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
// Kernel
// ------------------------------------------------------------------
__global__ void evalKernel(u32 numThreads, u32 numRollouts, u32 gw, u32 gh,
                           u32 nb, u32 ma, u32 ar, u32 seed, f32 sigma,
                           u32 mode, u32 stateDim, u32 hiddenDim, u32 actionDim,
                           u32 regionCount,
                           const BlockDef* blocks,
                           const f32* theta, const f32* epsilon,
                           f32* outMean, u32* outValid) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numThreads) return;

    RngState rng{seed + idx * 7919u + 104729u};
    for (u32 i = 0; i < 4; ++i) randU32(&rng);

    bool useEps = (mode == 1u);
    f32 sumR = 0.0f;
    u32 validN = 0;
    for (u32 r = 0; r < numRollouts; ++r) {
        u32 rs = randU32(&rng);
        f32 reward = rollout(rs, gw, gh, nb, ma, ar, stateDim, hiddenDim, actionDim,
                             regionCount,
                             blocks, theta, epsilon, sigma, useEps);
        if (reward > 0.0f) { sumR += reward; validN++; }
    }
    outMean[idx] = (validN > 0) ? (sumR / f32(validN)) : -1.0f;
    outValid[idx] = validN;
}

// ------------------------------------------------------------------
// Host evaluation wrapper
// ------------------------------------------------------------------
struct EvalResult {
    f32 meanReward;
    u32 validCount;
};

// ------------------------------------------------------------------
// Problem generator
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// MPI master/worker messages
// ------------------------------------------------------------------
enum Tags {
    TAG_ROUND_INIT = 100,
    TAG_TASK = 101,
    TAG_RESULT = 102,
    TAG_ROUND_DONE = 103,
    TAG_STOP = 104
};

struct RoundInitMsg {
    u32 round;
    u32 thetaSize;
};

struct TaskMsg {
    u32 round;
    u32 problemIdx;
    u32 problemSeed;
    u32 numBlocks;
    u32 mode;
};

struct ResultMsg {
    u32 round;
    u32 problemIdx;
    u32 mode;
    u32 validCount;
    f32 meanReward;
};

class GpuEvalContext {
public:
    GpuEvalContext(u32 maxBlocks, u32 thetaSize, u32 numThreads)
        : maxBlocks_(maxBlocks), thetaSize_(thetaSize), numThreads_(numThreads) {
        CUDA_CHECK(cudaMalloc(&dBlocks_, maxBlocks_ * sizeof(BlockDef)));
        CUDA_CHECK(cudaMalloc(&dTheta_, thetaSize_ * sizeof(f32)));
        CUDA_CHECK(cudaMalloc(&dEpsilon_, thetaSize_ * sizeof(f32)));
        CUDA_CHECK(cudaMalloc(&dOut_, numThreads_ * sizeof(f32)));
        CUDA_CHECK(cudaMalloc(&dValid_, numThreads_ * sizeof(u32)));
        hOut_.resize(numThreads_);
        hValid_.resize(numThreads_);
    }

    ~GpuEvalContext() {
        cudaFree(dBlocks_);
        cudaFree(dTheta_);
        cudaFree(dEpsilon_);
        cudaFree(dOut_);
        cudaFree(dValid_);
    }

    void uploadRound(const std::vector<f32>& theta, const std::vector<f32>& epsilon) {
        CUDA_CHECK(cudaMemcpy(dTheta_, theta.data(), thetaSize_ * sizeof(f32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dEpsilon_, epsilon.data(), thetaSize_ * sizeof(f32), cudaMemcpyHostToDevice));
    }

    EvalResult evaluate(u32 numThreads, u32 numRollouts, u32 gw, u32 gh,
                        u32 nb, u32 ma, u32 ar, u32 seed, f32 sigma, u32 mode,
                        u32 stateDim, u32 hiddenDim, u32 actionDim, u32 regionCount,
                        const std::vector<BlockDef>& blocks) {
        if (nb > maxBlocks_ || numThreads > numThreads_) {
            std::cerr << "GpuEvalContext capacity exceeded" << std::endl;
            std::exit(1);
        }

        CUDA_CHECK(cudaMemcpy(dBlocks_, blocks.data(), nb * sizeof(BlockDef), cudaMemcpyHostToDevice));

        u32 blockSize = 64;
        u32 gridSize = (numThreads + blockSize - 1) / blockSize;
        evalKernel<<<gridSize, blockSize>>>(numThreads, numRollouts, gw, gh, nb, ma, ar,
                                            seed, sigma, mode, stateDim, hiddenDim, actionDim,
                                            regionCount,
                                            dBlocks_, dTheta_, dEpsilon_, dOut_, dValid_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(hOut_.data(), dOut_, numThreads * sizeof(f32), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hValid_.data(), dValid_, numThreads * sizeof(u32), cudaMemcpyDeviceToHost));

        f32 totalR = 0.0f;
        u32 totalV = 0;
        for (u32 i = 0; i < numThreads; ++i) {
            if (hValid_[i] > 0) {
                totalR += hOut_[i] * f32(hValid_[i]);
                totalV += hValid_[i];
            }
        }
        return { (totalV > 0) ? (totalR / totalV) : -1.0f, totalV };
    }

private:
    u32 maxBlocks_ = 0;
    u32 thetaSize_ = 0;
    u32 numThreads_ = 0;
    BlockDef* dBlocks_ = nullptr;
    f32* dTheta_ = nullptr;
    f32* dEpsilon_ = nullptr;
    f32* dOut_ = nullptr;
    u32* dValid_ = nullptr;
    std::vector<f32> hOut_;
    std::vector<u32> hValid_;
};

static ResultMsg evaluateTask(GpuEvalContext& gpu, u32 round, u32 problemIdx,
                              u32 problemSeed, u32 mode,
                              u32 gridW, u32 gridH, u32 numBlocks,
                              u32 maxAttempts, u32 allowRotation,
                              u32 numThreads, u32 numRollouts,
                              f32 sigma, u32 stateDim, u32 mlpHiddenDim,
                              u32 actionDim, u32 regionCount,
                              const std::vector<BlockDef>& blocks) {
    EvalResult eval = gpu.evaluate(numThreads, numRollouts, gridW, gridH,
                                   numBlocks, maxAttempts, allowRotation,
                                   problemSeed + mode, sigma, mode,
                                   stateDim, mlpHiddenDim, actionDim, regionCount,
                                   blocks);
    ResultMsg result{};
    result.round = round;
    result.problemIdx = problemIdx;
    result.mode = mode;
    result.validCount = eval.validCount;
    result.meanReward = eval.meanReward;
    return result;
}

static void workerLoop(u32 gridW, u32 gridH,
                       u32 maxAttempts, u32 allowRotation,
                       u32 numThreads, u32 numRollouts, f32 sigma,
                       u32 stateDim, u32 mlpHiddenDim, u32 actionDim) {
    std::vector<f32> theta;
    std::vector<f32> epsilon;
    const u32 thetaSize = stateDim * mlpHiddenDim + mlpHiddenDim + mlpHiddenDim * actionDim + actionDim;
    const u32 regionCount = static_cast<u32>(std::sqrt(f32(actionDim)) + 0.5f);
    GpuEvalContext gpu(MAX_BLOCKS, thetaSize, numThreads);

    for (;;) {
        MPI_Status status{};
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_STOP) {
            u32 stop = 0;
            MPI_Recv(&stop, 1, MPI_UNSIGNED, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }

        RoundInitMsg init{};
        MPI_Recv(&init, sizeof(init), MPI_BYTE, 0, TAG_ROUND_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        theta.resize(init.thetaSize);
        epsilon.resize(init.thetaSize);
        MPI_Recv(theta.data(), init.thetaSize, MPI_FLOAT, 0, TAG_ROUND_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(epsilon.data(), init.thetaSize, MPI_FLOAT, 0, TAG_ROUND_INIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        gpu.uploadRound(theta, epsilon);

        for (;;) {
            TaskMsg task{};
            MPI_Recv(&task, sizeof(task), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_ROUND_DONE) {
                break;
            }
            if (status.MPI_TAG == TAG_STOP) {
                return;
            }
            if (status.MPI_TAG != TAG_TASK) {
                MPI_Abort(MPI_COMM_WORLD, 2);
            }

            std::vector<BlockDef> blocks(task.numBlocks);
            MPI_Recv(blocks.data(), task.numBlocks * sizeof(BlockDef), MPI_BYTE,
                     0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            ResultMsg result = evaluateTask(gpu, task.round, task.problemIdx, task.problemSeed, task.mode,
                                            gridW, gridH, task.numBlocks,
                                            maxAttempts, allowRotation,
                                            numThreads, numRollouts,
                                            sigma, stateDim, mlpHiddenDim,
                                            actionDim, regionCount, blocks);
            MPI_Send(&result, sizeof(result), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }
}

static void sendRoundInit(int worker, u32 round, const std::vector<f32>& theta,
                          const std::vector<f32>& epsilon) {
    RoundInitMsg init{round, static_cast<u32>(theta.size())};
    MPI_Send(&init, sizeof(init), MPI_BYTE, worker, TAG_ROUND_INIT, MPI_COMM_WORLD);
    MPI_Send(theta.data(), init.thetaSize, MPI_FLOAT, worker, TAG_ROUND_INIT, MPI_COMM_WORLD);
    MPI_Send(epsilon.data(), init.thetaSize, MPI_FLOAT, worker, TAG_ROUND_INIT, MPI_COMM_WORLD);
}

static void sendTask(int worker, u32 round, u32 problemIdx, u32 problemSeed, u32 mode,
                     const std::vector<BlockDef>& blocks) {
    TaskMsg task{round, problemIdx, problemSeed, static_cast<u32>(blocks.size()), mode};
    MPI_Send(&task, sizeof(task), MPI_BYTE, worker, TAG_TASK, MPI_COMM_WORLD);
    MPI_Send(blocks.data(), task.numBlocks * sizeof(BlockDef), MPI_BYTE,
             worker, TAG_TASK, MPI_COMM_WORLD);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parse args
    u32 gridW = 16, gridH = 16, numBlocks = 6;
    u32 minBlockSize = 2, maxBlockSize = 6;
    u32 numThreads = 1024, numRollouts = 128, maxAttempts = 50;
    u32 allowRotation = 1, numRounds = 1, numProblems = 8;
    u32 mlpHiddenDim = 32, actionRegions = 8;
    f32 sigma = 0.01f, lr = 0.001f;
    u32 baseSeed = 12345;
    u32 epsilonSeedOverride = 0;
    bool hasEpsilonSeedOverride = false;
    u32 thetaSeed = 42;
    u32 taskIdHash = 0;
    std::string thetaPath, outputPath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--gridW=", 0) == 0) gridW = std::stoul(arg.substr(8));
        else if (arg.rfind("--gridH=", 0) == 0) gridH = std::stoul(arg.substr(8));
        else if (arg.rfind("--numBlocks=", 0) == 0) numBlocks = std::stoul(arg.substr(12));
        else if (arg.rfind("--minBlockSize=", 0) == 0) minBlockSize = std::stoul(arg.substr(15));
        else if (arg.rfind("--maxBlockSize=", 0) == 0) maxBlockSize = std::stoul(arg.substr(15));
        else if (arg.rfind("--numThreads=", 0) == 0) numThreads = std::stoul(arg.substr(13));
        else if (arg.rfind("--rollouts=", 0) == 0) numRollouts = std::stoul(arg.substr(11));
        else if (arg.rfind("--rolloutsPerThread=", 0) == 0) numRollouts = std::stoul(arg.substr(20));
        else if (arg.rfind("--maxAttempts=", 0) == 0) maxAttempts = std::stoul(arg.substr(14));
        else if (arg.rfind("--allowRotation=", 0) == 0) allowRotation = std::stoul(arg.substr(16));
        else if (arg.rfind("--rounds=", 0) == 0) numRounds = std::stoul(arg.substr(9));
        else if (arg.rfind("--numProblems=", 0) == 0) numProblems = std::stoul(arg.substr(14));
        else if (arg.rfind("--sigma=", 0) == 0) sigma = std::stof(arg.substr(8));
        else if (arg.rfind("--lr=", 0) == 0) lr = std::stof(arg.substr(5));
        else if (arg.rfind("--seed=", 0) == 0) baseSeed = std::stoul(arg.substr(7));
        else if (arg.rfind("--epsilonSeed=", 0) == 0) { epsilonSeedOverride = std::stoul(arg.substr(14)); hasEpsilonSeedOverride = true; }
        else if (arg.rfind("--thetaSeed=", 0) == 0) thetaSeed = std::stoul(arg.substr(12));
        else if (arg.rfind("--taskIdHash=", 0) == 0) taskIdHash = std::stoul(arg.substr(13));
        else if (arg.rfind("--taskId=", 0) == 0) taskIdHash = stringHashSum(arg.substr(9));
        else if (arg.rfind("--mlpHiddenDim=", 0) == 0) mlpHiddenDim = std::stoul(arg.substr(15));
        else if (arg.rfind("--actionRegions=", 0) == 0) actionRegions = std::stoul(arg.substr(16));
        else if (arg.rfind("--theta=", 0) == 0) thetaPath = arg.substr(8);
        else if (arg.rfind("--output=", 0) == 0) outputPath = arg.substr(9);
    }

    u32 actionDim = actionRegions * actionRegions;
    if (mlpHiddenDim > MAX_HIDDEN_DIM) {
        if (rank == 0) std::cerr << "mlpHiddenDim " << mlpHiddenDim << " exceeds MAX_HIDDEN_DIM " << MAX_HIDDEN_DIM << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (actionDim > MAX_ACTION_DIM) {
        if (rank == 0) std::cerr << "actionDim " << actionDim << " exceeds MAX_ACTION_DIM " << MAX_ACTION_DIM << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (gridW > MAX_GRID_H || gridH > MAX_GRID_H) {
        if (rank == 0) std::cerr << "grid dimensions exceed MAX_GRID_H " << MAX_GRID_H << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (numBlocks > MAX_BLOCKS) {
        if (rank == 0) std::cerr << "numBlocks " << numBlocks << " exceeds MAX_BLOCKS " << MAX_BLOCKS << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (nprocs < 2) {
        if (rank == 0) {
            std::cerr << "mpi_gridpack_2d_rl now uses rank 0 as master and requires at least one worker rank." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    u32 stateDim = 8;
    u32 thetaSize = stateDim * mlpHiddenDim + mlpHiddenDim + mlpHiddenDim * actionDim + actionDim;

    // Allocate theta and epsilon on all ranks
    std::vector<f32> theta(thetaSize, 0.0f);
    std::vector<f32> epsilon(thetaSize, 0.0f);
    std::vector<f32> grad(thetaSize, 0.0f);

    if (rank != 0) {
        int nGpus = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nGpus));
        if (nGpus > 0) {
            CUDA_CHECK(cudaSetDevice(rank % nGpus));
        } else {
            std::cerr << "Worker rank " << rank << " found no CUDA devices" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        workerLoop(gridW, gridH,
                   maxAttempts, allowRotation,
                   numThreads, numRollouts, sigma,
                   stateDim, mlpHiddenDim, actionDim);
        MPI_Finalize();
        return 0;
    }

    if (!thetaPath.empty()) {
        std::ifstream f(thetaPath, std::ios::binary);
        f.read(reinterpret_cast<char*>(theta.data()), thetaSize * sizeof(f32));
    } else {
        std::mt19937 gen(thetaSeed);
        std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
        f32 scale = std::sqrt(2.0f / f32(stateDim + mlpHiddenDim));
        for (u32 i = 0; i < thetaSize; ++i) theta[i] = dist(gen) * scale;
    }

    // ES training loop (timed: includes all allocations, H2D/D2H, MPI comms)
    auto tStart = std::chrono::high_resolution_clock::now();
    u32 epsilonSeed = hasEpsilonSeedOverride ? epsilonSeedOverride : baseSeed;
    for (u32 round = 0; round < numRounds; ++round) {
        u32 rng = epsilonSeed + round * 7919u;
        auto xorshift = [&rng]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
        auto rand01 = [&]() { return f32(xorshift() % 0x7FFFFFFF) / f32(0x7FFFFFFF); };
        auto randN = [&]() {
            f32 u1 = fmaxf(rand01(), 0.0001f);
            f32 u2 = rand01();
            return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
        };
        for (u32 i = 0; i < thetaSize; ++i) epsilon[i] = randN();

        for (int worker = 1; worker < nprocs; ++worker) {
            sendRoundInit(worker, round, theta, epsilon);
        }

        f32 globalDelta = 0.0f;
        f32 globalBase = 0.0f;
        f32 globalPerturbed = 0.0f;
        u32 globalPairs = 0;
        std::vector<f32> baseRewards(numProblems, -1.0f);
        std::vector<f32> perturbedRewards(numProblems, -1.0f);
        std::vector<u32> baseValid(numProblems, 0);
        std::vector<u32> perturbedValid(numProblems, 0);

        u32 nextEval = 0;
        u32 activeTasks = 0;
        auto dispatchOne = [&](int worker) {
            if (nextEval >= numProblems * 2u) return false;
            u32 problemIdx = nextEval / 2u;
            u32 mode = nextEval % 2u;
            std::vector<BlockDef> blocks;
            u32 problemSeed = 123456789u + problemIdx * 747796405u + taskIdHash;
            blocks = generateBlocks(numBlocks, minBlockSize, maxBlockSize, problemSeed);
            ensureFitsGrid(blocks, gridW, gridH);
            sendTask(worker, round, problemIdx, problemSeed, mode, blocks);
            nextEval++;
            activeTasks++;
            return true;
        };

        for (int worker = 1; worker < nprocs; ++worker) {
            dispatchOne(worker);
        }

        while (activeTasks > 0) {
            ResultMsg result{};
            MPI_Status status{};
            MPI_Recv(&result, sizeof(result), MPI_BYTE, MPI_ANY_SOURCE, TAG_RESULT,
                     MPI_COMM_WORLD, &status);
            activeTasks--;

            if (result.problemIdx < numProblems) {
                if (result.mode == 0u) {
                    baseRewards[result.problemIdx] = result.meanReward;
                    baseValid[result.problemIdx] = result.validCount;
                } else {
                    perturbedRewards[result.problemIdx] = result.meanReward;
                    perturbedValid[result.problemIdx] = result.validCount;
                }
            }

            dispatchOne(status.MPI_SOURCE);
        }

        for (u32 problemIdx = 0; problemIdx < numProblems; ++problemIdx) {
            if (baseValid[problemIdx] > 0 && perturbedValid[problemIdx] > 0 &&
                baseRewards[problemIdx] > 0.0f && perturbedRewards[problemIdx] > 0.0f) {
                globalDelta += perturbedRewards[problemIdx] - baseRewards[problemIdx];
                globalBase += baseRewards[problemIdx];
                globalPerturbed += perturbedRewards[problemIdx];
                globalPairs++;
            }
        }

        for (int worker = 1; worker < nprocs; ++worker) {
            u32 done = round;
            MPI_Send(&done, 1, MPI_UNSIGNED, worker, TAG_ROUND_DONE, MPI_COMM_WORLD);
        }

        if (globalPairs > 0) {
            f32 meanDelta = globalDelta / globalPairs;
            f32 meanBase = globalBase / globalPairs;
            f32 meanPerturbed = globalPerturbed / globalPairs;
            f32 scale = meanDelta / sigma;

            for (u32 i = 0; i < thetaSize; ++i) {
                grad[i] = scale * epsilon[i];
                theta[i] += lr * grad[i];
            }

            std::cout << "Round " << round
                      << " pairs=" << globalPairs
                      << " base=" << meanBase
                      << " perturbed=" << meanPerturbed
                      << " delta=" << meanDelta
                      << std::endl;

            if (!outputPath.empty()) {
                std::string fname = outputPath + "_round_" + std::to_string(round) + ".bin";
                std::ofstream ofs(fname, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(theta.data()), thetaSize * sizeof(f32));
            }
        }

    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    for (int worker = 1; worker < nprocs; ++worker) {
        u32 stop = 0;
        MPI_Send(&stop, 1, MPI_UNSIGNED, worker, TAG_STOP, MPI_COMM_WORLD);
    }

    if (!outputPath.empty()) {
        std::string finalPath = outputPath + "_final.bin";
        std::ofstream ofs(finalPath, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(theta.data()), thetaSize * sizeof(f32));
        std::cout << "Saved final theta to " << finalPath << std::endl;
    }

    std::cout << "Total wall time (incl. allocations + MPI): " << totalMs << " ms" << std::endl;

    MPI_Finalize();
    return 0;
}
