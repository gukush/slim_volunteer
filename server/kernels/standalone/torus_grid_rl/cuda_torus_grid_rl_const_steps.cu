#include <cuda_runtime.h>

#include <algorithm>
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

static constexpr uint32_t GRID_W = 64;
static constexpr uint32_t GRID_H = 64;
static constexpr uint32_t TILE_COUNT = GRID_W * GRID_H;
static constexpr uint32_t FEATURE_DIM = 6;
static constexpr uint32_t ACTION_DIM = 4;
static constexpr uint32_t WEIGHT_COUNT = FEATURE_DIM * ACTION_DIM;
static constexpr uint32_t GOAL_X0 = 28;
static constexpr uint32_t GOAL_Y0 = 28;
static constexpr uint32_t GOAL_SIZE = 8;
static constexpr float WEIGHT_SCALE = 65536.0f;
static constexpr float LEARNING_RATE = 0.00005f;

__device__ __forceinline__ uint32_t rand_u32(uint32_t* s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
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
  v = fminf(fmaxf(v, lo), hi);
  return int(rintf(v));
}

template<uint32_t MAX_STEPS>
__global__ void torus_grid_rl_kernel_const_steps(
    const float* rewards,
    int* weights_fixed,
    uint32_t* stats,
    uint32_t trajectories,
    uint32_t seed_base)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = idx < trajectories;
  uint32_t safe_idx = active ? idx : (trajectories - 1u);

  __shared__ int block_weight_delta[WEIGHT_COUNT];
  __shared__ float block_weight_value[WEIGHT_COUNT];
  __shared__ uint32_t block_stats[4];
  if (threadIdx.x < WEIGHT_COUNT) {
    block_weight_delta[threadIdx.x] = 0;
    block_weight_value[threadIdx.x] = float(weights_fixed[threadIdx.x]) / WEIGHT_SCALE;
  }
  if (threadIdx.x < 4) block_stats[threadIdx.x] = 0;
  __syncthreads();

  uint32_t seed = seed_base + safe_idx * 747796405u + 2891336453u;
  for (uint32_t i = 0; i < 4; ++i) rand_u32(&seed);

  uint32_t x = rand_u32(&seed) & 63u;
  uint32_t y = rand_u32(&seed) & 63u;
  if (is_goal(x, y)) {
    x = (x + 17u) & 63u;
    y = (y + 29u) & 63u;
  }

  float grad[WEIGHT_COUNT];
  #pragma unroll
  for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) grad[i] = 0.0f;

  float total_reward = 0.0f;
  uint32_t steps = 0;
  uint32_t hit_goal = 0;

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
        sum += feats[f] * block_weight_value[f * ACTION_DIM + a];
      }
      logits[a] = sum;
    }

    float max_logit = fmaxf(fmaxf(logits[0], logits[1]), fmaxf(logits[2], logits[3]));
    float probs[ACTION_DIM];
    float prob_sum = 0.0f;
    #pragma unroll
    for (uint32_t a = 0; a < ACTION_DIM; ++a) {
      probs[a] = expf(logits[a] - max_logit);
      prob_sum += probs[a];
    }
    #pragma unroll
    for (uint32_t a = 0; a < ACTION_DIM; ++a) probs[a] /= prob_sum;

    float r = rand_f01(&seed);
    uint32_t action = 3;
    float cdf = 0.0f;
    #pragma unroll
    for (uint32_t a = 0; a < ACTION_DIM; ++a) {
      cdf += probs[a];
      if (r <= cdf) {
        action = a;
        break;
      }
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
    if (is_goal(x, y)) {
      hit_goal = 1;
      break;
    }
  }

  #pragma unroll
  for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) {
    int delta = clamp_i32(LEARNING_RATE * total_reward * grad[i] * WEIGHT_SCALE, -2048.0f, 2048.0f);
    atomicAdd(&block_weight_delta[i], active ? delta : 0);
  }

  atomicAdd(&block_stats[0], active ? 1u : 0u);
  atomicAdd(&block_stats[1], active ? hit_goal : 0u);
  atomicAdd(&block_stats[2], active ? uint32_t(rintf((total_reward + 512.0f) * 1000.0f)) : 0u);
  atomicAdd(&block_stats[3], active ? steps : 0u);
  __syncthreads();

  if (threadIdx.x < WEIGHT_COUNT) {
    atomicAdd(&weights_fixed[threadIdx.x], block_weight_delta[threadIdx.x]);
  }
  if (threadIdx.x < 4) {
    atomicAdd(&stats[threadIdx.x], block_stats[threadIdx.x]);
  }
}

static void launch_torus_grid_rl_kernel_const_steps(
    const float* d_rewards,
    int* d_weights,
    uint32_t* d_stats,
    uint32_t count,
    uint32_t seed_base,
    uint32_t maxSteps,
    uint32_t grid,
    uint32_t blockSize)
{
  switch (maxSteps) {
    case 1:
      torus_grid_rl_kernel_const_steps<1><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 2:
      torus_grid_rl_kernel_const_steps<2><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 4:
      torus_grid_rl_kernel_const_steps<4><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 8:
      torus_grid_rl_kernel_const_steps<8><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 16:
      torus_grid_rl_kernel_const_steps<16><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 32:
      torus_grid_rl_kernel_const_steps<32><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 64:
      torus_grid_rl_kernel_const_steps<64><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 128:
      torus_grid_rl_kernel_const_steps<128><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 256:
      torus_grid_rl_kernel_const_steps<256><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 512:
      torus_grid_rl_kernel_const_steps<512><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    case 1024:
      torus_grid_rl_kernel_const_steps<1024><<<grid, blockSize>>>(d_rewards, d_weights, d_stats, count, seed_base);
      break;
    default:
      std::cerr << "Unsupported compile-time maxSteps for const-steps variant: " << maxSteps
                << " (supported: 1,2,4,8,16,32,64,128,256,512,1024)\n";
      std::exit(2);
  }
}

static uint32_t parse_u32(const std::string& s) {
  return static_cast<uint32_t>(std::stoul(s, nullptr, 0));
}

static uint32_t xorshift(uint32_t& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

static std::vector<float> build_reward_map(uint32_t seed) {
  std::vector<float> rewards(TILE_COUNT);
  for (uint32_t y = 0; y < GRID_H; ++y) {
    for (uint32_t x = 0; x < GRID_W; ++x) {
      bool goal = x >= GOAL_X0 && x < GOAL_X0 + GOAL_SIZE && y >= GOAL_Y0 && y < GOAL_Y0 + GOAL_SIZE;
      if (goal) rewards[y * GRID_W + x] = 64.0f;
      else rewards[y * GRID_W + x] = (xorshift(seed) % 100 < 12) ? -3.0f : -1.0f;
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

static void usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [options]\n"
    << "  --totalTrajectories=N  Number of trajectories (default: 65536)\n"
    << "  --chunkSize=N          Trajectories per chunk-local network (default: 4096)\n"
    << "  --blockSize=N          CUDA block size (default: 128)\n"
    << "  --maxSteps=N           Max steps per trajectory (default: 128)\n"
    << "  --seed=N               Trajectory seed (default: 0xabcdef01)\n"
    << "  --environmentSeed=N    Reward-map seed (default: 0x5eed1234)\n";
}

int main(int argc, char** argv) {
  uint32_t totalTrajectories = 65536;
  uint32_t chunkSize = 4096;
  uint32_t blockSize = 128;
  uint32_t maxSteps = 128;
  uint32_t seed = 0xabcdef01u;
  uint32_t environmentSeed = 0x5eed1234u;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
    auto eq = arg.find('=');
    std::string key = eq == std::string::npos ? arg : arg.substr(0, eq);
    std::string val = eq == std::string::npos ? std::string() : arg.substr(eq + 1);
    if (key == "--totalTrajectories") totalTrajectories = parse_u32(val);
    else if (key == "--chunkSize") chunkSize = parse_u32(val);
    else if (key == "--blockSize") blockSize = parse_u32(val);
    else if (key == "--maxSteps") maxSteps = parse_u32(val);
    else if (key == "--seed") seed = parse_u32(val);
    else if (key == "--environmentSeed") environmentSeed = parse_u32(val);
    else { std::cerr << "Unknown option: " << arg << "\n"; usage(argv[0]); return 2; }
  }
  if (totalTrajectories == 0 || chunkSize == 0 || blockSize == 0 || blockSize > 1024) {
    std::cerr << "totalTrajectories/chunkSize must be positive and blockSize must be in [1, 1024]\n";
    return 2;
  }
  if (maxSteps == 0 || maxSteps > 1024) {
    std::cerr << "maxSteps must be in [1, 1024]\n";
    return 2;
  }
  auto rewards = build_reward_map(environmentSeed);
  const auto initialWeights = initial_weights_fixed();
  uint64_t totalStats[4] = {};
  std::vector<double> weightedWeights(WEIGHT_COUNT, 0.0);

  float* d_rewards = nullptr;
  int* d_weights = nullptr;
  uint32_t* d_stats = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rewards, rewards.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, initialWeights.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_stats, 4 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_rewards, rewards.data(), rewards.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  const auto epoch0 = std::chrono::system_clock::now();
  const auto wall0 = std::chrono::steady_clock::now();

  float kernel_ms = 0.0f;
  uint32_t chunks = 0;
  for (uint32_t offset = 0; offset < totalTrajectories; offset += chunkSize) {
    uint32_t count = std::min(chunkSize, totalTrajectories - offset);
    CUDA_CHECK(cudaMemcpy(d_weights, initialWeights.data(), initialWeights.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_stats, 0, 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaEventRecord(ev0));
    uint32_t grid = (count + blockSize - 1) / blockSize;
    launch_torus_grid_rl_kernel_const_steps(d_rewards, d_weights, d_stats, count, seed + offset * 2654435761u, maxSteps, grid, blockSize);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float chunk_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&chunk_kernel_ms, ev0, ev1));
    kernel_ms += chunk_kernel_ms;

    uint32_t h_stats[4] = {};
    std::vector<int> chunkWeights(WEIGHT_COUNT);
    CUDA_CHECK(cudaMemcpy(h_stats, d_stats, sizeof(h_stats), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(chunkWeights.data(), d_weights, chunkWeights.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < 4; ++i) totalStats[i] += h_stats[i];
    for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) {
      weightedWeights[i] += double(chunkWeights[i]) * double(h_stats[0]);
    }
    chunks++;
  }
  const auto wall1 = std::chrono::steady_clock::now();
  const auto epoch1 = std::chrono::system_clock::now();

  CUDA_CHECK(cudaFree(d_rewards));
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_stats));
  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));

  double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
  int64_t start_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch0.time_since_epoch()).count();
  int64_t end_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch1.time_since_epoch()).count();
  double totalReward = double(totalStats[2]) / 1000.0 - double(totalStats[0]) * 512.0;
  std::cout << std::fixed << std::setprecision(6)
            << "trajectories=" << totalStats[0]
            << ",chunks=" << chunks
            << ",goals=" << totalStats[1]
            << ",goalRate=" << (totalStats[0] ? double(totalStats[1]) / totalStats[0] : 0.0)
            << ",meanReward=" << (totalStats[0] ? totalReward / totalStats[0] : 0.0)
            << ",meanSteps=" << (totalStats[0] ? double(totalStats[3]) / totalStats[0] : 0.0)
            << ",chunkSize=" << chunkSize
            << ",blockSize=" << blockSize
            << ",wall_ms=" << wall_ms
            << ",start_epoch_ms=" << start_epoch_ms
            << ",end_epoch_ms=" << end_epoch_ms
            << ",kernel_ms=" << kernel_ms
            << "\nweights=";
  for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) {
    if (i) std::cout << ",";
    std::cout << ((totalStats[0] ? weightedWeights[i] / double(totalStats[0]) : 0.0) / WEIGHT_SCALE);
  }
  std::cout << "\n";
  return 0;
}
