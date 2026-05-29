import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'torus-grid-rl';
export const name = 'Torus Grid RL Policy Gradient (WebGPU)';

const GRID_W = 64;
const GRID_H = 64;
const TILE_COUNT = GRID_W * GRID_H;
const FEATURE_DIM = 6;
const ACTION_DIM = 4;
const WEIGHT_COUNT = FEATURE_DIM * ACTION_DIM;
const MAX_STEPS = 128;
const GOAL_X0 = 28;
const GOAL_Y0 = 28;
const GOAL_SIZE = 8;
const MAGIC = 20240529;

function parseU32(value, name, fallback) {
  const raw = value ?? fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0 || n > 0xffffffff) {
    throw new Error(`${name} must be a uint32 integer`);
  }
  return n >>> 0;
}

function buildRewardMap(seedValue) {
  let seed = (seedValue || 0x12345678) >>> 0;
  function randU32() {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  }

  const rewards = new Float32Array(TILE_COUNT);
  for (let y = 0; y < GRID_H; y++) {
    for (let x = 0; x < GRID_W; x++) {
      const inGoal = x >= GOAL_X0 && x < GOAL_X0 + GOAL_SIZE && y >= GOAL_Y0 && y < GOAL_Y0 + GOAL_SIZE;
      if (inGoal) {
        rewards[y * GRID_W + x] = 64.0;
      } else {
        const r = randU32() % 100;
        rewards[y * GRID_W + x] = r < 12 ? -3.0 : -1.0;
      }
    }
  }
  return rewards;
}

function initialWeights() {
  const weights = new Float32Array(WEIGHT_COUNT);
  // A tiny hand-written prior: prefer actions that reduce torus distance to the goal.
  // Features 2 and 3 are signed dx/dy toward the goal center.
  weights[2 * ACTION_DIM + 0] = -0.05; // up
  weights[2 * ACTION_DIM + 1] = 0.05;  // down
  weights[3 * ACTION_DIM + 2] = -0.05; // left
  weights[3 * ACTION_DIM + 3] = 0.05;  // right
  return weights;
}

function readResultF32(result) {
  if (result instanceof ArrayBuffer) return new Float32Array(result);
  if (ArrayBuffer.isView(result)) return new Float32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (Buffer.isBuffer(result)) return new Float32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
    const u8 = Uint8Array.from(result.data);
    return new Float32Array(u8.buffer, u8.byteOffset, Math.floor(u8.byteLength / 4));
  }
  throw new Error('Unsupported torus-grid-rl result buffer type');
}

function resolveState(taskDir, config, inputArgs) {
  const environmentSeed = parseU32(inputArgs.environmentSeed ?? config.environmentSeed, 'environmentSeed', 0x5eed1234);
  const envPath = path.join(taskDir, 'environment.json');
  const rewardMap = buildRewardMap(environmentSeed);
  if (!fs.existsSync(envPath)) {
    fs.writeFileSync(envPath, JSON.stringify({
      gridW: GRID_W,
      gridH: GRID_H,
      maxSteps: MAX_STEPS,
      goal: { x0: GOAL_X0, y0: GOAL_Y0, size: GOAL_SIZE },
      environmentSeed,
      rewards: Array.from(rewardMap),
    }, null, 2));
  }

  const policyPath = inputArgs.policyPath ?? config.policyPath ?? path.join(taskDir, 'policy_state.json');
  let weights = initialWeights();
  if (fs.existsSync(policyPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(policyPath, 'utf8'));
      if (Array.isArray(data.weights) && data.weights.length === WEIGHT_COUNT) {
        weights = Float32Array.from(data.weights);
      }
    } catch (error) {
      logger.warn('Ignoring unreadable torus-grid-rl policy state', { policyPath, error: error.message });
    }
  }
  return { rewardMap, weights, policyPath, environmentSeed };
}

export function getClientExecutorInfo(config) {
  const framework = (config?.framework || 'webgpu').toLowerCase();
  if (framework !== 'webgpu') {
    throw new Error('Unsupported framework for torus-grid-rl: ' + framework);
  }
  return {
    framework: 'webgpu',
    path: 'executors/webgpu-torus-grid-rl.client.js',
    kernels: ['kernels/webgpu/torus_grid_rl.wgsl'],
    schema: { output: 'Float32Array' },
  };
}

export function buildChunker({ taskId, taskDir, config, inputArgs }) {
  const totalTrajectories = parseU32(inputArgs.totalTrajectories ?? config.totalTrajectories, 'totalTrajectories', 65536);
  const chunkTrajectories = parseU32(inputArgs.chunkTrajectories ?? inputArgs.chunkSize ?? config.chunkTrajectories ?? config.chunkSize, 'chunkTrajectories', 4096);
  const workgroupSize = parseU32(inputArgs.workgroupSize ?? config.workgroupSize, 'workgroupSize', 128);
  if (totalTrajectories === 0) throw new Error('totalTrajectories must be positive');
  if (chunkTrajectories === 0) throw new Error('chunkTrajectories must be positive');
  if (workgroupSize === 0 || workgroupSize > 1024) throw new Error('workgroupSize must be in [1, 1024]');

  const { rewardMap, weights, environmentSeed } = resolveState(taskDir, config, inputArgs);
  const totalChunks = Math.ceil(totalTrajectories / chunkTrajectories);
  const baseSeed = parseU32(inputArgs.seed ?? config.seed, 'seed', 0xabcdef01);

  logger.info(`Torus Grid RL: ${totalTrajectories} trajectories, ${totalChunks} chunks, chunkTrajectories=${chunkTrajectories}, workgroupSize=${workgroupSize}`);

  return {
    async *stream() {
      let offset = 0;
      for (let chunkIndex = 0; offset < totalTrajectories; chunkIndex++) {
        const count = Math.min(chunkTrajectories, totalTrajectories - offset);
        const params = new Uint32Array([
          count >>> 0,
          workgroupSize >>> 0,
          (baseSeed + chunkIndex * 747796405) >>> 0,
          environmentSeed >>> 0,
        ]);
        yield {
          id: uuidv4(),
          payload: {
            rewardMap: rewardMap.buffer.slice(rewardMap.byteOffset, rewardMap.byteOffset + rewardMap.byteLength),
            weights: weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength),
            params: params.buffer.slice(params.byteOffset, params.byteOffset + params.byteLength),
          },
          meta: {
            chunkIndex,
            offset,
            trajectories: count,
            workgroupSize,
            weightCount: WEIGHT_COUNT,
          },
          tCreate: Date.now(),
        };
        offset += count;
      }
      logger.info('Torus Grid RL chunker done');
    },
  };
}

export function buildAssembler({ taskId, taskDir, config, inputArgs }) {
  const outPath = path.join(taskDir, 'output.json');
  const { policyPath } = resolveState(taskDir, config, inputArgs);
  let chunksProcessed = 0;
  let totalTrajectories = 0;
  let totalGoals = 0;
  let totalReward = 0;
  let totalSteps = 0;
  const weightSums = new Float64Array(WEIGHT_COUNT);

  return {
    integrate({ result, meta }) {
      const floats = readResultF32(result);
      if (floats.length < 6 + WEIGHT_COUNT) {
        throw new Error(`torus-grid-rl result too short: ${floats.length}`);
      }
      if (Math.abs(Number(floats[0]) - MAGIC) > 2) {
        throw new Error(`torus-grid-rl bad result magic: ${floats[0]}`);
      }
      const trajectories = Number(floats[1]);
      const goals = Number(floats[2]);
      const reward = Number(floats[3]);
      const steps = Number(floats[4]);
      chunksProcessed++;
      totalTrajectories += trajectories;
      totalGoals += goals;
      totalReward += reward;
      totalSteps += steps;
      for (let i = 0; i < WEIGHT_COUNT; i++) {
        weightSums[i] += Number(floats[6 + i]) * trajectories;
      }
      if (trajectories !== Number(meta.trajectories || trajectories)) {
        logger.warn('Torus Grid RL result trajectory count differs from metadata', { trajectories, meta: meta.trajectories });
      }
    },
    finalize() {
      const averagedWeights = Array.from(weightSums, (v) => totalTrajectories > 0 ? v / totalTrajectories : 0);
      const summary = {
        taskId,
        chunksProcessed,
        totalTrajectories,
        totalGoals,
        goalRate: totalTrajectories > 0 ? totalGoals / totalTrajectories : 0,
        totalReward,
        meanReward: totalTrajectories > 0 ? totalReward / totalTrajectories : 0,
        meanSteps: totalTrajectories > 0 ? totalSteps / totalTrajectories : 0,
        policyPath,
        weights: averagedWeights,
        constants: {
          gridW: GRID_W,
          gridH: GRID_H,
          maxSteps: MAX_STEPS,
          featureDim: FEATURE_DIM,
          actionDim: ACTION_DIM,
          weightCount: WEIGHT_COUNT,
          goal: { x0: GOAL_X0, y0: GOAL_Y0, size: GOAL_SIZE },
        },
        completedAt: new Date().toISOString(),
      };
      fs.writeFileSync(policyPath, JSON.stringify({
        weights: averagedWeights,
        featureDim: FEATURE_DIM,
        actionDim: ACTION_DIM,
        updatedAt: summary.completedAt,
      }, null, 2));
      fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));
      return { outPath, totalTrajectories, totalGoals, meanReward: summary.meanReward };
    },
  };
}

export function getTotalChunks(config, inputArgs) {
  const totalTrajectories = parseU32(inputArgs.totalTrajectories ?? config.totalTrajectories, 'totalTrajectories', 65536);
  const chunkTrajectories = parseU32(inputArgs.chunkTrajectories ?? inputArgs.chunkSize ?? config.chunkTrajectories ?? config.chunkSize, 'chunkTrajectories', 4096);
  return Math.ceil(totalTrajectories / chunkTrajectories);
}
