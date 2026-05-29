// MPI + CUDA Distributed Evolution Strategies for GridPack-2D RL
// Multi-GPU / multi-node ES training loop.
//
// Build:
//   mpicxx -x cu -O3 -arch=sm_70 -DMAX_STATE_DIM=8 -DMAX_HIDDEN_DIM=128 -DMAX_ACTION_DIM=256 \
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
// Kernel
// ------------------------------------------------------------------
__global__ void evalKernel(u32 numThreads, u32 numRollouts, u32 gw, u32 gh,
                           u32 nb, u32 ma, u32 ar, u32 seed, f32 sigma,
                           u32 mode, u32 stateDim, u32 hiddenDim, u32 actionDim,
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

EvalResult evaluateOnGPU(u32 numThreads, u32 numRollouts, u32 gw, u32 gh,
                         u32 nb, u32 ma, u32 ar, u32 seed, f32 sigma, u32 mode,
                         u32 stateDim, u32 hiddenDim, u32 actionDim,
                         const std::vector<BlockDef>& blocks,
                         const f32* theta, const f32* epsilon) {
    u32 thetaSize = stateDim * hiddenDim + hiddenDim + hiddenDim * actionDim + actionDim;

    BlockDef* dBlocks;
    f32 *dTheta, *dEpsilon, *dOut;
    u32* dValid;
    CUDA_CHECK(cudaMalloc(&dBlocks, nb * sizeof(BlockDef)));
    CUDA_CHECK(cudaMalloc(&dTheta, thetaSize * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dEpsilon, thetaSize * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dOut, numThreads * sizeof(f32)));
    CUDA_CHECK(cudaMalloc(&dValid, numThreads * sizeof(u32)));

    CUDA_CHECK(cudaMemcpy(dBlocks, blocks.data(), nb * sizeof(BlockDef), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTheta, theta, thetaSize * sizeof(f32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEpsilon, epsilon, thetaSize * sizeof(f32), cudaMemcpyHostToDevice));

    u32 blockSize = 256;
    u32 gridSize = (numThreads + blockSize - 1) / blockSize;
    evalKernel<<<gridSize, blockSize>>>(numThreads, numRollouts, gw, gh, nb, ma, ar,
                                        seed, sigma, mode, stateDim, hiddenDim, actionDim,
                                        dBlocks, dTheta, dEpsilon, dOut, dValid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<f32> hOut(numThreads);
    std::vector<u32> hValid(numThreads);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, numThreads * sizeof(f32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hValid.data(), dValid, numThreads * sizeof(u32), cudaMemcpyDeviceToHost));

    f32 totalR = 0.0f;
    u32 totalV = 0;
    for (u32 i = 0; i < numThreads; ++i) {
        if (hValid[i] > 0) {
            totalR += hOut[i] * f32(hValid[i]);
            totalV += hValid[i];
        }
    }

    cudaFree(dBlocks); cudaFree(dTheta); cudaFree(dEpsilon); cudaFree(dOut); cudaFree(dValid);

    return { (totalV > 0) ? (totalR / totalV) : -1.0f, totalV };
}

// ------------------------------------------------------------------
// Problem generator
// ------------------------------------------------------------------
static std::vector<BlockDef> generateBlocks(u32 numBlocks, u32 minSize, u32 maxSize, u32& rng) {
    std::vector<BlockDef> out;
    auto xorshift = [&rng]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    u32 range = maxSize - minSize + 1;
    for (u32 i = 0; i < numBlocks; ++i) {
        u32 w = minSize + (xorshift() % range);
        u32 h = minSize + (xorshift() % range);
        out.push_back({w, h});
    }
    return out;
}

// ------------------------------------------------------------------
// MPI helpers
// ------------------------------------------------------------------
static void mpiBroadcastBlocks(std::vector<BlockDef>& blocks, int rank) {
    u32 n = blocks.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank != 0) blocks.resize(n);
    MPI_Bcast(blocks.data(), n * sizeof(BlockDef), MPI_BYTE, 0, MPI_COMM_WORLD);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Select GPU by rank
    int nGpus;
    CUDA_CHECK(cudaGetDeviceCount(&nGpus));
    if (nGpus > 0) {
        CUDA_CHECK(cudaSetDevice(rank % nGpus));
    }

    // Parse args
    u32 gridW = 16, gridH = 16, numBlocks = 6;
    u32 minBlockSize = 2, maxBlockSize = 6;
    u32 numThreads = 1024, numRollouts = 128, maxAttempts = 50;
    u32 allowRotation = 1, numRounds = 1, numProblems = 8;
    u32 mlpHiddenDim = 32, actionRegions = 8;
    f32 sigma = 0.01f, lr = 0.001f;
    u32 baseSeed = 12345;
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
        else if (arg.rfind("--maxAttempts=", 0) == 0) maxAttempts = std::stoul(arg.substr(14));
        else if (arg.rfind("--allowRotation=", 0) == 0) allowRotation = std::stoul(arg.substr(16));
        else if (arg.rfind("--rounds=", 0) == 0) numRounds = std::stoul(arg.substr(9));
        else if (arg.rfind("--numProblems=", 0) == 0) numProblems = std::stoul(arg.substr(14));
        else if (arg.rfind("--sigma=", 0) == 0) sigma = std::stof(arg.substr(8));
        else if (arg.rfind("--lr=", 0) == 0) lr = std::stof(arg.substr(5));
        else if (arg.rfind("--seed=", 0) == 0) baseSeed = std::stoul(arg.substr(7));
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
    u32 stateDim = 8;
    u32 thetaSize = stateDim * mlpHiddenDim + mlpHiddenDim + mlpHiddenDim * actionDim + actionDim;

    // Allocate theta and epsilon on all ranks
    std::vector<f32> theta(thetaSize, 0.0f);
    std::vector<f32> epsilon(thetaSize, 0.0f);
    std::vector<f32> grad(thetaSize, 0.0f);

    if (rank == 0) {
        if (!thetaPath.empty()) {
            std::ifstream f(thetaPath, std::ios::binary);
            f.read(reinterpret_cast<char*>(theta.data()), thetaSize * sizeof(f32));
        } else {
            std::mt19937 gen(42);
            std::normal_distribution<f32> dist(0.0f, std::sqrt(2.0f / (stateDim + mlpHiddenDim)));
            for (u32 i = 0; i < thetaSize; ++i) theta[i] = dist(gen);
        }
    }

    // Broadcast initial theta
    MPI_Bcast(theta.data(), thetaSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ES training loop (timed: includes all allocations, H2D/D2H, MPI comms)
    auto tStart = std::chrono::high_resolution_clock::now();
    u32 epsilonSeed = baseSeed;
    for (u32 round = 0; round < numRounds; ++round) {
        if (rank == 0) {
            // Generate epsilon for this round
            u32 rng = epsilonSeed + round * 7919u;
            auto xorshift = [&rng]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
            auto rand01 = [&]() { return f32(xorshift() % 0x7FFFFFFF) / f32(0x7FFFFFFF); };
            auto randN = [&]() {
                f32 u1 = fmaxf(rand01(), 0.0001f);
                f32 u2 = rand01();
                return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
            };
            for (u32 i = 0; i < thetaSize; ++i) epsilon[i] = randN();
        }
        MPI_Bcast(epsilon.data(), thetaSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

        f32 sumDelta = 0.0f;
        u32 validPairs = 0;
        f32 sumBase = 0.0f;
        f32 sumPerturbed = 0.0f;

        for (u32 problemIdx = 0; problemIdx < numProblems; ++problemIdx) {
            std::vector<BlockDef> blocks;
            u32 problemSeed = 123456789u + problemIdx * 747796405u + round * 104729u;
            if (rank == 0) {
                blocks = generateBlocks(numBlocks, minBlockSize, maxBlockSize, problemSeed);
            }
            mpiBroadcastBlocks(blocks, rank);

            // Base evaluation
            EvalResult base = evaluateOnGPU(numThreads, numRollouts, gridW, gridH,
                                            numBlocks, maxAttempts, allowRotation,
                                            problemSeed, sigma, 0,
                                            stateDim, mlpHiddenDim, actionDim,
                                            blocks, theta.data(), epsilon.data());

            // Perturbed evaluation
            EvalResult perturbed = evaluateOnGPU(numThreads, numRollouts, gridW, gridH,
                                                 numBlocks, maxAttempts, allowRotation,
                                                 problemSeed + 1, sigma, 1,
                                                 stateDim, mlpHiddenDim, actionDim,
                                                 blocks, theta.data(), epsilon.data());

            if (base.meanReward > 0.0f && perturbed.meanReward > 0.0f) {
                f32 delta = perturbed.meanReward - base.meanReward;
                sumDelta += delta;
                sumBase += base.meanReward;
                sumPerturbed += perturbed.meanReward;
                validPairs++;
            }
        }

        // Allreduce stats
        f32 globalDelta = 0.0f, globalBase = 0.0f, globalPerturbed = 0.0f;
        u32 globalPairs = 0;
        MPI_Allreduce(&sumDelta, &globalDelta, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&sumBase, &globalBase, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&sumPerturbed, &globalPerturbed, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&validPairs, &globalPairs, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0 && globalPairs > 0) {
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

        // Broadcast updated theta for next round
        MPI_Bcast(theta.data(), thetaSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    if (rank == 0 && !outputPath.empty()) {
        std::string finalPath = outputPath + "_final.bin";
        std::ofstream ofs(finalPath, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(theta.data()), thetaSize * sizeof(f32));
        std::cout << "Saved final theta to " << finalPath << std::endl;
    }

    if (rank == 0) {
        std::cout << "Total wall time (incl. allocations + MPI): " << totalMs << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
