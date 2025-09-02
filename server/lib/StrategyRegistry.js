// lib/StrategyRegistry.js
import * as blockMatmul from '../strategies/block-matmul.js';
import * as blockMatmulFlex from '../strategies/block-matmul-flex.js';
import * as nativeBlockMatmul from '../strategies/native-block-matmul.js';
import * as ecmStage1 from '../strategies/ecm-stage1.js';

const strategies = new Map();

function register(strategy) {
  strategies.set(strategy.id, strategy);
}

register(blockMatmul);
register(blockMatmulFlex);
register(nativeBlockMatmul);  // Add the new native strategy
register(ecmStage1);

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