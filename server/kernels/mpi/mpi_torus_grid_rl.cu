#include <mpi.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
              << cudaGetErrorString(err__) << std::endl; \
    std::exit(2); \
  } \
} while (0)

static constexpr uint32_t GRID_W = 64, GRID_H = 64, TILE_COUNT = 4096;
static constexpr uint32_t FEATURE_DIM = 6, ACTION_DIM = 4, WEIGHT_COUNT = 24;
static constexpr uint32_t MAX_STEPS = 128, GOAL_X0 = 28, GOAL_Y0 = 28, GOAL_SIZE = 8;
static constexpr float WEIGHT_SCALE = 65536.0f;
static constexpr float LEARNING_RATE = 0.00005f;

struct WorkMsg {
  uint32_t offset;
  uint32_t count;
  uint32_t seed;
};

struct ResultMsg {
  uint32_t offset;
  uint32_t count;
  uint32_t stats[4];
  int weights[WEIGHT_COUNT];
};

__device__ __forceinline__ uint32_t rand_u32(uint32_t* s) {
  uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *s = x; return x;
}
__device__ __forceinline__ float rand_f01(uint32_t* s) {
  return float(rand_u32(s)) / 4294967295.0f;
}
__device__ __forceinline__ bool is_goal(uint32_t x, uint32_t y) {
  return x >= GOAL_X0 && x < GOAL_X0 + GOAL_SIZE && y >= GOAL_Y0 && y < GOAL_Y0 + GOAL_SIZE;
}
__device__ __forceinline__ float torus_delta(uint32_t coord, uint32_t goal_coord) {
  int d = int(goal_coord) - int(coord);
  if (d > 32) d -= 64;
  if (d < -32) d += 64;
  return float(d) / 32.0f;
}
__device__ __forceinline__ float feature_value(uint32_t f, uint32_t x, uint32_t y, float reward_here) {
  if (f == 0) return float(x) / 63.0f;
  if (f == 1) return float(y) / 63.0f;
  if (f == 2) return torus_delta(x, GOAL_X0 + GOAL_SIZE / 2);
  if (f == 3) return torus_delta(y, GOAL_Y0 + GOAL_SIZE / 2);
  if (f == 4) return reward_here / 64.0f;
  return 1.0f;
}
__device__ __forceinline__ int clamp_i32(float v, float lo, float hi) {
  return int(rintf(fminf(fmaxf(v, lo), hi)));
}

__global__ void torus_grid_rl_kernel(
    const float* rewards,
    int* weights_fixed,
    uint32_t* stats,
    uint32_t trajectories,
    uint32_t seed_base)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= trajectories) return;

  uint32_t seed = seed_base + idx * 747796405u + 2891336453u;
  for (uint32_t i = 0; i < 4; ++i) rand_u32(&seed);
  uint32_t x = rand_u32(&seed) & 63u;
  uint32_t y = rand_u32(&seed) & 63u;
  if (is_goal(x, y)) { x = (x + 17u) & 63u; y = (y + 29u) & 63u; }

  float grad[WEIGHT_COUNT];
  #pragma unroll
  for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) grad[i] = 0.0f;
  float total_reward = 0.0f;
  uint32_t steps = 0, hit_goal = 0;

  for (uint32_t step = 0; step < MAX_STEPS; ++step) {
    float reward_here = rewards[y * GRID_W + x];
    float feats[FEATURE_DIM];
    #pragma unroll
    for (uint32_t f = 0; f < FEATURE_DIM; ++f) feats[f] = feature_value(f, x, y, reward_here);

    float logits[ACTION_DIM];
    #pragma unroll
    for (uint32_t a = 0; a < ACTION_DIM; ++a) {
      float sum = 0.0f;
      #pragma unroll
      for (uint32_t f = 0; f < FEATURE_DIM; ++f) {
        sum += feats[f] * (float(weights_fixed[f * ACTION_DIM + a]) / WEIGHT_SCALE);
      }
      logits[a] = sum;
    }
    float max_logit = fmaxf(fmaxf(logits[0], logits[1]), fmaxf(logits[2], logits[3]));
    float probs[ACTION_DIM];
    float prob_sum = 0.0f;
    #pragma unroll
    for (uint32_t a = 0; a < ACTION_DIM; ++a) { probs[a] = expf(logits[a] - max_logit); prob_sum += probs[a]; }
    #pragma unroll
    for (uint32_t a = 0; a < ACTION_DIM; ++a) probs[a] /= prob_sum;

    float r = rand_f01(&seed);
    uint32_t action = 3;
    float cdf = 0.0f;
    #pragma unroll
    for (uint32_t a = 0; a < ACTION_DIM; ++a) {
      cdf += probs[a];
      if (r <= cdf) { action = a; break; }
    }
    #pragma unroll
    for (uint32_t f = 0; f < FEATURE_DIM; ++f) {
      #pragma unroll
      for (uint32_t a = 0; a < ACTION_DIM; ++a) {
        float chosen = (a == action) ? 1.0f : 0.0f;
        grad[f * ACTION_DIM + a] += feats[f] * (chosen - probs[a]);
      }
    }

    if (action == 0) y = (y + 63u) & 63u;
    else if (action == 1) y = (y + 1u) & 63u;
    else if (action == 2) x = (x + 63u) & 63u;
    else x = (x + 1u) & 63u;

    float reward = rewards[y * GRID_W + x];
    total_reward += reward;
    steps++;
    if (is_goal(x, y)) { hit_goal = 1; break; }
  }

  #pragma unroll
  for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) {
    int delta = clamp_i32(LEARNING_RATE * total_reward * grad[i] * WEIGHT_SCALE, -2048.0f, 2048.0f);
    atomicAdd(&weights_fixed[i], delta);
  }
  atomicAdd(&stats[0], 1u);
  atomicAdd(&stats[1], hit_goal);
  atomicAdd(&stats[2], uint32_t(rintf((total_reward + 512.0f) * 1000.0f)));
  atomicAdd(&stats[3], steps);
}

static uint32_t parse_u32(const std::string& s) { return static_cast<uint32_t>(std::stoul(s, nullptr, 0)); }
static uint32_t xorshift(uint32_t& seed) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }

static std::vector<float> build_reward_map(uint32_t seed) {
  std::vector<float> rewards(TILE_COUNT);
  for (uint32_t y = 0; y < GRID_H; ++y) {
    for (uint32_t x = 0; x < GRID_W; ++x) {
      bool goal = x >= GOAL_X0 && x < GOAL_X0 + GOAL_SIZE && y >= GOAL_Y0 && y < GOAL_Y0 + GOAL_SIZE;
      rewards[y * GRID_W + x] = goal ? 64.0f : ((xorshift(seed) % 100 < 12) ? -3.0f : -1.0f);
    }
  }
  return rewards;
}

static std::vector<int> initial_weights_fixed() {
  std::vector<int> weights(WEIGHT_COUNT, 0);
  weights[2 * ACTION_DIM + 0] = int(-0.05f * WEIGHT_SCALE);
  weights[2 * ACTION_DIM + 1] = int(0.05f * WEIGHT_SCALE);
  weights[3 * ACTION_DIM + 2] = int(-0.05f * WEIGHT_SCALE);
  weights[3 * ACTION_DIM + 3] = int(0.05f * WEIGHT_SCALE);
  return weights;
}

static ResultMsg run_chunk_cuda(
    const std::vector<float>& rewards,
    const std::vector<int>& initialWeights,
    const WorkMsg& work,
    uint32_t blockSize)
{
  ResultMsg result{};
  result.offset = work.offset;
  result.count = work.count;
  if (work.count == 0) return result;

  float* d_rewards = nullptr;
  int* d_weights = nullptr;
  uint32_t* d_stats = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rewards, rewards.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, initialWeights.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_stats, sizeof(result.stats)));
  CUDA_CHECK(cudaMemcpy(d_rewards, rewards.data(), rewards.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, initialWeights.data(), initialWeights.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(result.stats)));

  uint32_t grid = (work.count + blockSize - 1) / blockSize;
  torus_grid_rl_kernel<<<grid, blockSize>>>(d_rewards, d_weights, d_stats, work.count, work.seed);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(result.stats, d_stats, sizeof(result.stats), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.weights, d_weights, sizeof(result.weights), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_rewards));
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_stats));
  return result;
}

static void usage(const char* argv0) {
  std::cerr
    << "Usage: mpirun -np N " << argv0 << " [options]\n"
    << "  --totalTrajectories=N  Total trajectories (default: 65536)\n"
    << "  --chunkSize=N          Trajectories per MPI work message (default: 1024)\n"
    << "  --blockSize=N          CUDA block size (default: 128)\n"
    << "  --seed=N               Trajectory seed (default: 0xabcdef01)\n"
    << "  --environmentSeed=N    Reward-map seed (default: 0x5eed1234)\n";
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  uint32_t totalTrajectories = 65536, chunkSize = 1024, blockSize = 128;
  uint32_t seed = 0xabcdef01u, environmentSeed = 0x5eed1234u;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
    auto eq = arg.find('=');
    std::string key = eq == std::string::npos ? arg : arg.substr(0, eq);
    std::string val = eq == std::string::npos ? std::string() : arg.substr(eq + 1);
    if (key == "--totalTrajectories") totalTrajectories = parse_u32(val);
    else if (key == "--chunkSize") chunkSize = parse_u32(val);
    else if (key == "--blockSize") blockSize = parse_u32(val);
    else if (key == "--seed") seed = parse_u32(val);
    else if (key == "--environmentSeed") environmentSeed = parse_u32(val);
  }
  if (nproc < 2) {
    if (!rank) std::cerr << "This master-slave program requires at least 2 ranks.\n";
    MPI_Finalize();
    return 2;
  }
  if (totalTrajectories == 0 || chunkSize == 0 || blockSize == 0 || blockSize > 1024) {
    if (!rank) std::cerr << "Invalid totalTrajectories/chunkSize/blockSize\n";
    MPI_Finalize();
    return 2;
  }

  auto rewards = build_reward_map(environmentSeed);
  auto initialWeights = initial_weights_fixed();

  constexpr int DATA_TAG = 1, RESULT_TAG = 2, FINISH_TAG = 3;
  if (rank == 0) {
    uint32_t totalChunks = (totalTrajectories + chunkSize - 1) / chunkSize;
    uint32_t nextOffset = 0, sent = 0, received = 0;
    uint64_t totalStats[4] = {};
    std::vector<double> weightedWeights(WEIGHT_COUNT, 0.0);
    std::vector<uint8_t> workerFinished(nproc, 0);

    const auto wall0 = std::chrono::steady_clock::now();
    for (int dst = 1; dst < nproc && nextOffset < totalTrajectories; ++dst) {
      WorkMsg msg{ nextOffset, std::min(chunkSize, totalTrajectories - nextOffset), seed + nextOffset * 2654435761u };
      MPI_Send(&msg, sizeof(msg), MPI_BYTE, dst, DATA_TAG, MPI_COMM_WORLD);
      nextOffset += msg.count;
      sent++;
    }

    while (received < totalChunks) {
      ResultMsg result{};
      MPI_Status status;
      MPI_Recv(&result, sizeof(result), MPI_BYTE, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
      received++;
      for (int i = 0; i < 4; ++i) totalStats[i] += result.stats[i];
      for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) {
        weightedWeights[i] += double(result.weights[i]) * double(result.stats[0]);
      }

      if (nextOffset < totalTrajectories) {
        WorkMsg msg{ nextOffset, std::min(chunkSize, totalTrajectories - nextOffset), seed + nextOffset * 2654435761u };
        MPI_Send(&msg, sizeof(msg), MPI_BYTE, status.MPI_SOURCE, DATA_TAG, MPI_COMM_WORLD);
        nextOffset += msg.count;
        sent++;
      } else {
        WorkMsg msg{};
        MPI_Send(&msg, sizeof(msg), MPI_BYTE, status.MPI_SOURCE, FINISH_TAG, MPI_COMM_WORLD);
        workerFinished[status.MPI_SOURCE] = 1;
      }
    }
    for (int dst = 1; dst < nproc; ++dst) {
      if (workerFinished[dst]) continue;
      WorkMsg msg{};
      MPI_Send(&msg, sizeof(msg), MPI_BYTE, dst, FINISH_TAG, MPI_COMM_WORLD);
      workerFinished[dst] = 1;
    }
    const auto wall1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    double totalReward = double(totalStats[2]) / 1000.0 - double(totalStats[0]) * 512.0;

    std::cout << std::fixed << std::setprecision(6)
              << "trajectories=" << totalStats[0]
              << ",chunks=" << totalChunks
              << ",sent=" << sent
              << ",goals=" << totalStats[1]
              << ",goalRate=" << (totalStats[0] ? double(totalStats[1]) / totalStats[0] : 0.0)
              << ",meanReward=" << (totalStats[0] ? totalReward / totalStats[0] : 0.0)
              << ",meanSteps=" << (totalStats[0] ? double(totalStats[3]) / totalStats[0] : 0.0)
              << ",blockSize=" << blockSize
              << ",chunkSize=" << chunkSize
              << ",nproc=" << nproc
              << ",wall_ms=" << wall_ms
              << "\nweights=";
    for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) {
      if (i) std::cout << ",";
      std::cout << ((totalStats[0] ? weightedWeights[i] / double(totalStats[0]) : 0.0) / WEIGHT_SCALE);
    }
    std::cout << "\n";
  } else {
    while (true) {
      WorkMsg work{};
      MPI_Status status;
      MPI_Recv(&work, sizeof(work), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == FINISH_TAG) break;
      ResultMsg result = run_chunk_cuda(rewards, initialWeights, work, blockSize);
      MPI_Send(&result, sizeof(result), MPI_BYTE, 0, RESULT_TAG, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;
}
