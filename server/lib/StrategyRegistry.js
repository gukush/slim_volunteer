// lib/StrategyRegistry.js
import * as blockMatmul from '../strategies/block-matmul.js';
import * as blockMatmulFlex from '../strategies/block-matmul-flex.js';
import * as nativeBlockMatmul from '../strategies/native-block-matmul.js';
import * as ecmStage1 from '../strategies/ecm-stage1.js';
import * as distributedSort from '../strategies/distributed-sort.js';
import * as multiHeadAttention from '../strategies/multi-head-attention.js';
import * as exeBlockMatmulFlex from '../strategies/exe-block-matmul-flex.js';
import * as nativeEcmStage1 from '../strategies/native-ecm-stage1.js';
import * as nativeDistributedSort from '../strategies/native-distributed-sort.js';
import * as nativeMultiHeadAttention from '../strategies/native-multi-head-attention.js';
import * as exeMultiHeadAttention from '../strategies/exe-multi-head-attention.js';
import * as exeCpuQuicksort from '../strategies/exe-cpu-quicksort.js';
import * as exeEcmStage1 from '../strategies/exe-ecm-stage1.js';
import * as exeDistributedSort from '../strategies/exe-distributed-sort.js';


const strategies = new Map();

function register(strategy) {
  strategies.set(strategy.id, strategy);
}

register(blockMatmul);
register(blockMatmulFlex);
register(nativeBlockMatmul);
register(ecmStage1);
register(distributedSort);
register(multiHeadAttention);
register(exeBlockMatmulFlex);
register(nativeEcmStage1);
register(nativeDistributedSort);
register(nativeMultiHeadAttention);
register(exeMultiHeadAttention);
register(exeCpuQuicksort);
register(exeEcmStage1);
register(exeDistributedSort);

export function listStrategies() {
  return Array.from(strategies.keys());
}

export function getStrategy(id) {
  const strategy = strategies.get(id);
  if (!strategy) {
    throw new Error(`Unknown strategy: ${id}`);
  }
  return strategy;
}

export { strategies };
