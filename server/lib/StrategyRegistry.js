import * as BlockMatmul from '../strategies/block-matmul.js';
import * as ECMStage1 from '../strategies/ecm-stage1.js';
import * as BlockMatmulFlex from '../strategies/block-matmul-flex.js';
import * as DistributedSort from '../strategies/distributed-sort.js';

const registry = new Map([
  [BlockMatmul.id, BlockMatmul],
  [ECMStage1.id, ECMStage1],
  [BlockMatmulFlex.id, BlockMatmulFlex],
  [DistributedSort.id, DistributedSort],
]);

export function listStrategies(){ return Array.from(registry.values()).map(s=>({id:s.id, name:s.name})); }
export function getStrategy(id){
  const s = registry.get(id);
  if(!s) throw new Error(`Unknown strategy: ${id}`);
  return s;
}
