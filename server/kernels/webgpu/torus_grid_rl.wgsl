const GRID_W: u32 = 64u;
const GRID_H: u32 = 64u;
const TILE_COUNT: u32 = 4096u;
const FEATURE_DIM: u32 = 6u;
const ACTION_DIM: u32 = 4u;
const WEIGHT_COUNT: u32 = 24u;
const MAX_STEPS: u32 = {{MAX_STEPS}}u;
const GOAL_X0: u32 = 28u;
const GOAL_Y0: u32 = 28u;
const GOAL_SIZE: u32 = 8u;
const MAGIC_F32: f32 = 20240529.0;
const WEIGHT_SCALE: f32 = 65536.0;
const LEARNING_RATE: f32 = 0.00005;
const WORKGROUP_SIZE: u32 = 128u;

struct Config {
  trajectories: u32,
  workgroup_size: u32,
  seed: u32,
  environment_seed: u32,
  dispatch_x: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> rewards: array<f32>;
@group(0) @binding(1) var<storage, read_write> weights_fixed: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> stats_fixed: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> config: Config;

var<workgroup> wg_weights_delta: array<atomic<i32>, 24>;
var<workgroup> wg_weight_values: array<f32, 24>;
var<workgroup> wg_stats: array<atomic<u32>, 4>;

fn rand_u32(s: ptr<function, u32>) -> u32 {
  var x = *s;
  x = x ^ (x << 13u);
  x = x ^ (x >> 17u);
  x = x ^ (x << 5u);
  *s = x;
  return x;
}

fn rand_f01(s: ptr<function, u32>) -> f32 {
  return f32(rand_u32(s)) / 4294967295.0;
}

fn torus_delta(coord: u32, goal_coord: u32) -> f32 {
  var d = i32(goal_coord) - i32(coord);
  if (d > 32) {
    d = d - 64;
  }
  if (d < -32) {
    d = d + 64;
  }
  return f32(d) / 32.0;
}

fn is_goal(x: u32, y: u32) -> bool {
  return x >= GOAL_X0 && x < (GOAL_X0 + GOAL_SIZE) && y >= GOAL_Y0 && y < (GOAL_Y0 + GOAL_SIZE);
}

fn feature_value(feature: u32, x: u32, y: u32, reward_here: f32) -> f32 {
  if (feature == 0u) {
    return f32(x) / 63.0;
  }
  if (feature == 1u) {
    return f32(y) / 63.0;
  }
  if (feature == 2u) {
    return torus_delta(x, GOAL_X0 + GOAL_SIZE / 2u);
  }
  if (feature == 3u) {
    return torus_delta(y, GOAL_Y0 + GOAL_SIZE / 2u);
  }
  if (feature == 4u) {
    return reward_here / 64.0;
  }
  return 1.0;
}

fn clamp_i32(v: f32, lo: f32, hi: f32) -> i32 {
  return i32(round(clamp(v, lo, hi)));
}

@compute @workgroup_size(128)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let idx = gid.y * config.dispatch_x * WORKGROUP_SIZE + gid.x;
  let active = idx < config.trajectories;

  if (lid.x < WEIGHT_COUNT) {
    atomicStore(&wg_weights_delta[lid.x], 0);
    wg_weight_values[lid.x] = f32(atomicLoad(&weights_fixed[lid.x])) / WEIGHT_SCALE;
  }
  if (lid.x < 4u) {
    atomicStore(&wg_stats[lid.x], 0u);
  }
  workgroupBarrier();

  if (active) {
    var seed = config.seed + idx * 747796405u + 2891336453u;
    for (var warm = 0u; warm < 4u; warm = warm + 1u) {
      let ignored = rand_u32(&seed);
    }

    var x = rand_u32(&seed) & 63u;
    var y = rand_u32(&seed) & 63u;
    if (is_goal(x, y)) {
      x = (x + 17u) & 63u;
      y = (y + 29u) & 63u;
    }

    var grad: array<f32, 24>;
    for (var i = 0u; i < WEIGHT_COUNT; i = i + 1u) {
      grad[i] = 0.0;
    }

    var total_reward = 0.0;
    var steps = 0u;
    var hit_goal = 0u;

    for (var step = 0u; step < MAX_STEPS; step = step + 1u) {
      let tile = y * GRID_W + x;
      let reward_here = rewards[tile];

      var feats: array<f32, 6>;
      for (var f = 0u; f < FEATURE_DIM; f = f + 1u) {
        feats[f] = feature_value(f, x, y, reward_here);
      }

      var logits: array<f32, 4>;
      for (var a = 0u; a < ACTION_DIM; a = a + 1u) {
        var sum = 0.0;
        for (var f = 0u; f < FEATURE_DIM; f = f + 1u) {
          sum = sum + feats[f] * wg_weight_values[f * ACTION_DIM + a];
        }
        logits[a] = sum;
      }

      let max_logit = max(max(logits[0], logits[1]), max(logits[2], logits[3]));
      var probs: array<f32, 4>;
      var prob_sum = 0.0;
      for (var a = 0u; a < ACTION_DIM; a = a + 1u) {
        probs[a] = exp(logits[a] - max_logit);
        prob_sum = prob_sum + probs[a];
      }
      for (var a = 0u; a < ACTION_DIM; a = a + 1u) {
        probs[a] = probs[a] / prob_sum;
      }

      let r = rand_f01(&seed);
      var action = 3u;
      var cdf = 0.0;
      for (var a = 0u; a < ACTION_DIM; a = a + 1u) {
        cdf = cdf + probs[a];
        if (r <= cdf) {
          action = a;
          break;
        }
      }

      for (var f = 0u; f < FEATURE_DIM; f = f + 1u) {
        for (var a = 0u; a < ACTION_DIM; a = a + 1u) {
          let chosen = select(0.0, 1.0, a == action);
          grad[f * ACTION_DIM + a] = grad[f * ACTION_DIM + a] + feats[f] * (chosen - probs[a]);
        }
      }

      if (action == 0u) {
        y = (y + 63u) & 63u;
      } else if (action == 1u) {
        y = (y + 1u) & 63u;
      } else if (action == 2u) {
        x = (x + 63u) & 63u;
      } else {
        x = (x + 1u) & 63u;
      }

      let reward = rewards[y * GRID_W + x];
      total_reward = total_reward + reward;
      steps = steps + 1u;
      if (is_goal(x, y)) {
        hit_goal = 1u;
        break;
      }
    }

    for (var i = 0u; i < WEIGHT_COUNT; i = i + 1u) {
      let delta = clamp_i32(LEARNING_RATE * total_reward * grad[i] * WEIGHT_SCALE, -2048.0, 2048.0);
      atomicAdd(&wg_weights_delta[i], delta);
    }

    atomicAdd(&wg_stats[0], 1u);
    atomicAdd(&wg_stats[1], hit_goal);
    atomicAdd(&wg_stats[2], u32(round((total_reward + 512.0) * 1000.0)));
    atomicAdd(&wg_stats[3], steps);
  }

  workgroupBarrier();

  if (lid.x < WEIGHT_COUNT) {
    atomicAdd(&weights_fixed[lid.x], atomicLoad(&wg_weights_delta[lid.x]));
  }
  if (lid.x < 4u) {
    atomicAdd(&stats_fixed[lid.x], atomicLoad(&wg_stats[lid.x]));
  }
}
