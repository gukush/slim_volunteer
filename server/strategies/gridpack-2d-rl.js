import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'gridpack-2d-rl';
export const name = 'GridPack-2D RL Floor Planning (WebGPU ES)';

const MAGIC = 0x47524C50; // "GRPL"
const MAX_BLOCKS = 16;
const MAX_GRID_H = 32;

// ── WGSL Template Generator (Option C: runtime shader generation) ───

function generateWGSL({ stateDim, hiddenDim, actionDim }) {
  const regionCount = Math.round(Math.sqrt(actionDim));
  const thetaSize = stateDim * hiddenDim + hiddenDim + hiddenDim * actionDim + actionDim;
  const l1WeightsEnd = stateDim * hiddenDim;
  const l1BiasEnd = l1WeightsEnd + hiddenDim;
  const l2WeightsEnd = l1BiasEnd + hiddenDim * actionDim;
  const l2BiasEnd = l2WeightsEnd + actionDim;

  return `const MAX_BLOCKS: u32 = ${MAX_BLOCKS}u;
const MAX_GRID_H: u32 = ${MAX_GRID_H}u;
const STATE_DIM: u32 = ${stateDim}u;
const HIDDEN_DIM: u32 = ${hiddenDim}u;
const ACTION_DIM: u32 = ${actionDim}u;
const THETA_SIZE: u32 = ${thetaSize}u;
const MAGIC: u32 = ${MAGIC}u;
const REGION_COUNT: u32 = ${regionCount}u;

struct Block { w: u32, h: u32, };

struct Params {
    gridW: u32, gridH: u32, numBlocks: u32, numRollouts: u32,
    seed: u32, maxAttempts: u32, allowRotation: u32, sigma: f32,
    numThreads: u32, thetaVersion: u32,
    epsilonSeed: u32, mode: u32,
};

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> blocks: array<Block>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> theta: array<f32>;
@group(0) @binding(4) var<storage, read> epsilon: array<f32>;

fn randU32(s: ptr<function, u32>) -> u32 {
    var x = *s; x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u; *s = x; return x;
}
fn randF01(s: ptr<function, u32>) -> f32 { return f32(randU32(s)) / 4294967295.0; }
fn randN(s: ptr<function, u32>) -> f32 {
    let u1 = max(randF01(s), 0.0001);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * randF01(s));
}

fn rowMask(x: u32, w: u32) -> u32 {
    if (w >= 32u) { return 0xFFFFFFFFu << x; }
    return ((1u << w) - 1u) << x;
}
fn canPlace(g: ptr<function, array<u32, MAX_GRID_H>>, x: u32, y: u32, w: u32, h: u32, gw: u32, gh: u32) -> bool {
    if (x+w > gw || y+h > gh) { return false; }
    let m = rowMask(x,w);
    for (var r=y; r<y+h; r=r+1u) { if (((*g)[r] & m) != 0u) { return false; } }
    return true;
}
fn place(g: ptr<function, array<u32, MAX_GRID_H>>, x: u32, y: u32, w: u32, h: u32) {
    let m = rowMask(x,w);
    for (var r=y; r<y+h; r=r+1u) { (*g)[r] = (*g)[r] | m; }
}

fn mlpForward(st: array<f32,${stateDim}>, useEps: bool, sigma: f32) -> array<f32,${actionDim}> {
    var hidden: array<f32,${hiddenDim}>;
    for (var i=0u; i<${hiddenDim}u; i=i+1u) {
        var sum = theta[${l1BiasEnd - hiddenDim}u + i];
        if (useEps) { sum = sum + sigma * epsilon[${l1BiasEnd - hiddenDim}u + i]; }
        for (var j=0u; j<${stateDim}u; j=j+1u) {
            var w = theta[i * ${stateDim}u + j];
            if (useEps) { w = w + sigma * epsilon[i * ${stateDim}u + j]; }
            sum = sum + st[j] * w;
        }
        hidden[i] = max(sum, 0.0);
    }
    var logits: array<f32,${actionDim}>;
    for (var i=0u; i<${actionDim}u; i=i+1u) {
        var sum = theta[${l2BiasEnd - actionDim}u + i];
        if (useEps) { sum = sum + sigma * epsilon[${l2BiasEnd - actionDim}u + i]; }
        for (var j=0u; j<${hiddenDim}u; j=j+1u) {
            var w = theta[${l1BiasEnd}u + i * ${hiddenDim}u + j];
            if (useEps) { w = w + sigma * epsilon[${l1BiasEnd}u + i * ${hiddenDim}u + j]; }
            sum = sum + hidden[j] * w;
        }
        logits[i] = sum;
    }
    return logits;
}

fn sampleAction(logits: array<f32,${actionDim}>, s: ptr<function, u32>, gw: u32, gh: u32) -> vec2<u32> {
    var mx = logits[0];
    for (var i=1u; i<${actionDim}u; i=i+1u) { if (logits[i] > mx) { mx = logits[i]; } }
    var expSum = 0.0;
    var probs: array<f32,${actionDim}>;
    for (var i=0u; i<${actionDim}u; i=i+1u) { let e = exp(logits[i]-mx); probs[i]=e; expSum=expSum+e; }
    let r = randF01(s) * expSum;
    var c = 0.0; var a = 0u;
    for (var i=0u; i<${actionDim}u; i=i+1u) { c=c+probs[i]; if (c>=r) { a=i; break; } }
    let rx = a % ${regionCount}u;
    let ry = a / ${regionCount}u;
    let rw = max(1u, gw / ${regionCount}u);
    let rh = max(1u, gh / ${regionCount}u);
    let px = rx*rw + u32(randF01(s)*f32(rw));
    let py = ry*rh + u32(randF01(s)*f32(rh));
    return vec2<u32>(min(px,gw-1u), min(py,gh-1u));
}

fn extractFeatures(g: ptr<function, array<u32, MAX_GRID_H>>, gw: u32, gh: u32, bw: u32, bh: u32, left: u32, total: u32) -> array<f32,${stateDim}> {
    var occ = 0u;
    for (var y=0u; y<gh; y=y+1u) { occ = occ + countOneBits((*g)[y]); }
    var mix = gw;
    var miy = gh;
    var mxx = 0u;
    var myy = 0u;
    for (var y=0u; y<gh; y=y+1u) {
        let row = (*g)[y]; if (row==0u) { continue; }
        if (y<miy) { miy=y; } if (y>myy) { myy=y; }
        for (var x=0u; x<gw; x=x+1u) { if ((row&(1u<<x))!=0u) { if (x<mix){mix=x;} if (x>mxx){mxx=x;} } }
    }
    var bx = gw; var by = gh;
    if (mxx>=mix) { bx = mxx-mix+1u; by = myy-miy+1u; }
    return array<f32,${stateDim}>(
        f32(bw)/f32(gw), f32(bh)/f32(gh), f32(occ)/f32(gw*gh),
        f32(bx)/f32(gw), f32(by)/f32(gh), f32(left)/f32(total),
        f32(gw)/f32(gh), 1.0
    );
}

fn rollout(seed: u32, gw: u32, gh: u32, nb: u32, ma: u32, ar: u32, useEps: bool, sigma: f32) -> f32 {
    var rng = seed;
    var g: array<u32, MAX_GRID_H>;
    for (var i=0u; i<gh; i=i+1u) { g[i]=0u; }
    var valid = true;
    for (var b=0u; b<nb; b=b+1u) {
        var bw = blocks[b].w; var bh = blocks[b].h;
        if (bw>gw || bh>gh) { valid=false; break; }
        if (ar!=0u && randU32(&rng)%2u==1u) { let t=bw; bw=bh; bh=t; if (bw>gw||bh>gh) { let t2=bw; bw=bh; bh=t2; } }
        let feat = extractFeatures(&g, gw, gh, bw, bh, nb-b, nb);
        let logits = mlpForward(feat, useEps, sigma);
        var placed = false;
        for (var a=0u; a<ma; a=a+1u) {
            let pos = sampleAction(logits, &rng, gw, gh);
            if (canPlace(&g, pos.x, pos.y, bw, bh, gw, gh)) { place(&g, pos.x, pos.y, bw, bh); placed=true; break; }
        }
        if (!placed) { valid=false; break; }
    }
    if (!valid) { return -1.0; }
    var mix = gw;
    var miy = gh;
    var mxx = 0u;
    var myy = 0u;
    var ba = 0.0;
    for (var y=0u; y<gh; y=y+1u) {
        let row=g[y]; if (row==0u) { continue; }
        for (var x=0u; x<gw; x=x+1u) { if ((row&(1u<<x))!=0u) { ba=ba+1.0; if (x<mix){mix=x;} if (y<miy){miy=y;} if (x>mxx){mxx=x;} if (y>myy){myy=y;} } }
    }
    let bx=mxx-mix+1u; let by=myy-miy+1u; let barea=f32(bx*by);
    return select(1.0, 1.0+ba/barea, barea>0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.numThreads) { return; }

    let gw = params.gridW; let gh = params.gridH;
    let nb = params.numBlocks; let nr = params.numRollouts;
    let ma = params.maxAttempts; let ar = params.allowRotation;
    let sigma = params.sigma;
    let useEps = (params.mode == 1u);

    var sumR = 0.0; var validN = 0u;
    var rng = params.seed + idx * 7919u + 104729u;
    for (var i=0u; i<4u; i=i+1u) { randU32(&rng); }

    for (var r=0u; r<nr; r=r+1u) {
        let rs = randU32(&rng);
        let reward = rollout(rs, gw, gh, nb, ma, ar, useEps, sigma);
        if (reward > 0.0) { sumR = sumR + reward; validN = validN + 1u; }
    }

    let meanR = select(-1.0, sumR / f32(validN), validN > 0u);

    let base = idx * 6u;
    out[base + 0u] = f32(MAGIC);
    out[base + 1u] = meanR;
    out[base + 2u] = f32(validN);
    out[base + 3u] = f32(nr);
    out[base + 4u] = f32(params.mode);
    out[base + 5u] = 0.0;
}`;
}

function computeThetaSize(stateDim, hiddenDim, actionDim) {
  return stateDim * hiddenDim + hiddenDim + hiddenDim * actionDim + actionDim;
}

function getArchitecture(config, inputArgs) {
  const hiddenDim = Number(inputArgs?.mlpHiddenDim ?? config?.mlpHiddenDim ?? 32);
  const actionRegions = Number(inputArgs?.actionRegions ?? config?.actionRegions ?? 8);
  const actionDim = actionRegions * actionRegions;
  const stateDim = 8; // fixed for now

  // Validate
  if (![4, 16, 32, 64, 128].includes(hiddenDim)) {
    throw new Error(`mlpHiddenDim must be one of [4, 16, 32, 64, 128], got ${hiddenDim}`);
  }
  if (![2, 4, 8, 16].includes(actionRegions)) {
    throw new Error(`actionRegions must be one of [2, 4, 8, 16], got ${actionRegions}`);
  }

  return { stateDim, hiddenDim, actionDim, actionRegions };
}

// ── PolicyState: ES parameter management ───────────────────────────

export class PolicyState {
  constructor(thetaPath = null, arch = null) {
    this.arch = arch || { stateDim: 8, hiddenDim: 32, actionDim: 64 };
    this.thetaSize = computeThetaSize(this.arch.stateDim, this.arch.hiddenDim, this.arch.actionDim);

    if (thetaPath && fs.existsSync(thetaPath)) {
      const data = JSON.parse(fs.readFileSync(thetaPath, 'utf-8'));
      // Validate architecture matches
      if (data.arch) {
        this.arch = data.arch;
        this.thetaSize = computeThetaSize(this.arch.stateDim, this.arch.hiddenDim, this.arch.actionDim);
      }
      this.theta = new Float32Array(data.theta);
      this.version = data.version || 0;
      this.round = data.round || 0;
      logger.info('Loaded policy state from', thetaPath, 'version', this.version, 'arch', this.arch);
    } else {
      this.theta = PolicyState.xavierInit(this.thetaSize, this.arch.stateDim, this.arch.hiddenDim);
      this.version = 0;
      this.round = 0;
      logger.info('Initialized new policy state', 'arch', this.arch, 'thetaSize', this.thetaSize);
    }
    this.epsilonSeed = Math.floor(Math.random() * 0x7FFFFFFF);
    this.sigma = 0.01;
    this.lr = 0.001;
  }

  static xavierInit(n, fanIn, fanOut) {
    const arr = new Float32Array(n);
    const scale = Math.sqrt(2.0 / (fanIn + fanOut));
    for (let i = 0; i < n; i++) {
      arr[i] = (Math.random() - 0.5) * 2.0 * scale;
    }
    return arr;
  }

  generateEpsilon() {
    let seed = this.epsilonSeed >>> 0;
    const eps = new Float32Array(this.thetaSize);
    function randU32() {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      return seed >>> 0;
    }
    function randF01() { return randU32() / 4294967295.0; }
    function randN() {
      const u1 = Math.max(randF01(), 0.0001);
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(6.28318530718 * randF01());
    }
    for (let i = 0; i < this.thetaSize; i++) {
      eps[i] = randN();
    }
    return eps;
  }

  update(grad) {
    for (let i = 0; i < this.thetaSize; i++) {
      this.theta[i] += this.lr * grad[i];
    }
    this.version += 1;
    this.round += 1;
    this.epsilonSeed = Math.floor(Math.random() * 0x7FFFFFFF);
  }

  save(outPath) {
    const data = {
      theta: Array.from(this.theta),
      version: this.version,
      round: this.round,
      sigma: this.sigma,
      lr: this.lr,
      arch: this.arch,
      thetaSize: this.thetaSize,
      savedAt: new Date().toISOString(),
    };
    fs.writeFileSync(outPath, JSON.stringify(data, null, 2));
  }
}

// ── Strategy exports ──────────────────────────────────────────────

export function getClientExecutorInfo(config, inputArgs) {
  const framework = (config?.framework || 'webgpu').toLowerCase();
  if (framework !== 'webgpu') {
    throw new Error('Unsupported framework for gridpack-2d-rl: ' + framework);
  }

  const arch = getArchitecture(config, inputArgs);
  const wgslCode = generateWGSL(arch);

  return {
    framework: 'webgpu',
    path: 'executors/webgpu-gridpack-2d-rl.client.js',
    kernels: [{ name: 'gridpack_2d_rl_generated.wgsl', content: wgslCode }],
    schema: { output: 'Float32Array' },
  };
}

function parseU32(value, name, fallback) {
  const n = Number(value);
  if (Number.isNaN(n) || !Number.isInteger(n) || n < 0) {
    if (fallback !== undefined) return fallback;
    throw new Error(`${name} must be a non-negative integer`);
  }
  return n >>> 0;
}

function generateBlocks(numBlocks, minSize, maxSize, seed) {
  const blocks = [];
  let rng = (seed || 12345) >>> 0;
  function xorshift() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (rng >>> 0) % 0x7FFFFFFF;
  }
  const range = maxSize - minSize + 1;
  for (let i = 0; i < numBlocks; i++) {
    const w = minSize + (xorshift() % range);
    const h = minSize + (xorshift() % range);
    blocks.push({ w, h, id: i });
  }
  return blocks;
}

function ensureFitsGrid(blocks, gridW, gridH) {
  const maxArea = gridW * gridH;
  let totalArea = blocks.reduce((s, b) => s + b.w * b.h, 0);
  if (totalArea > maxArea) {
    const scale = Math.sqrt(maxArea / (totalArea * 1.2));
    for (const b of blocks) {
      b.w = Math.max(1, Math.floor(b.w * scale));
      b.h = Math.max(1, Math.floor(b.h * scale));
    }
  }
  for (const b of blocks) {
    if (b.w > gridW) b.w = gridW;
    if (b.h > gridH) b.h = gridH;
  }
  return blocks;
}

export function buildChunker({ taskId, taskDir, K, config, inputArgs }) {
  const gridW = parseU32(inputArgs.gridW ?? config.gridW, 'gridW', 16);
  const gridH = parseU32(inputArgs.gridH ?? config.gridH, 'gridH', 16);
  if (gridW > MAX_GRID_H || gridH > MAX_GRID_H) {
    throw new Error('gridpack-2d-rl supports grid dimensions up to 32x32');
  }

  const numBlocks = parseU32(inputArgs.numBlocks ?? config.numBlocks, 'numBlocks', 6);
  if (numBlocks > MAX_BLOCKS) {
    throw new Error('gridpack-2d-rl supports up to 16 blocks');
  }

  const numProblems = parseU32(inputArgs.numProblems ?? config.numProblems, 'numProblems', 10);
  const minBlockSize = parseU32(inputArgs.minBlockSize ?? config.minBlockSize, 'minBlockSize', 2);
  const maxBlockSize = parseU32(inputArgs.maxBlockSize ?? config.maxBlockSize, 'maxBlockSize', 6);
  const rolloutsPerThread = parseU32(inputArgs.rolloutsPerThread ?? config.rolloutsPerThread, 'rolloutsPerThread', 64);
  const maxAttempts = parseU32(inputArgs.maxAttempts ?? config.maxAttempts, 'maxAttempts', 50);
  const allowRotation = parseU32(inputArgs.allowRotation ?? config.allowRotation, 'allowRotation', 1);
  const numThreads = parseU32(inputArgs.numThreads ?? config.numThreads, 'numThreads', 256);

  // MLP architecture (pass through to config for executor info)
  const arch = getArchitecture(config, inputArgs);

  // Load or initialize policy state
  const policyPath = inputArgs.policyPath ?? config.policyPath ?? path.join(taskDir, 'policy_state.json');
  const policy = new PolicyState(fs.existsSync(policyPath) ? policyPath : null, arch);
  policy.sigma = inputArgs.sigma ?? config.sigma ?? 0.01;
  policy.lr = inputArgs.lr ?? config.lr ?? 0.001;

  // Verify architecture matches loaded policy
  if (policy.arch.hiddenDim !== arch.hiddenDim || policy.arch.actionDim !== arch.actionDim) {
    logger.warn('Policy architecture mismatch, reinitializing',
      { policy: policy.arch, requested: arch });
    policy.arch = arch;
    policy.thetaSize = computeThetaSize(arch.stateDim, arch.hiddenDim, arch.actionDim);
    policy.theta = PolicyState.xavierInit(policy.thetaSize, arch.stateDim, arch.hiddenDim);
  }

  // Generate one epsilon vector for this round
  const epsilon = policy.generateEpsilon();
  const epsilonSeed = policy.epsilonSeed;
  const thetaVersion = policy.version;

  // Save policy state at start of round
  policy.save(path.join(taskDir, `policy_v${policy.version}_start.json`));

  const totalChunks = numProblems * 2;
  logger.info(`ES Round ${policy.round}: ${numProblems} problems, grid=${gridW}x${gridH}, blocks=${numBlocks}, theta_v${thetaVersion}, arch=${arch.hiddenDim}h/${arch.actionDim}a, epsilonSeed=${epsilonSeed}`);

  const roundState = {
    policyPath,
    thetaVersion,
    epsilonSeed,
    sigma: policy.sigma,
    lr: policy.lr,
    arch,
    thetaSize: policy.thetaSize,
    numProblems,
    problemRewardsBase: new Array(numProblems).fill(null),
    problemRewardsPerturbed: new Array(numProblems).fill(null),
    problemValidBase: new Array(numProblems).fill(0),
    problemValidPerturbed: new Array(numProblems).fill(0),
  };

  return {
    async *stream() {
      for (let problemIdx = 0; problemIdx < numProblems; problemIdx++) {
        const seed = (123456789 + problemIdx * 747796405 + taskId.split('').reduce((a, c) => a + c.charCodeAt(0), 0)) >>> 0;
        let blocks = generateBlocks(numBlocks, minBlockSize, maxBlockSize, seed);
        blocks = ensureFitsGrid(blocks, gridW, gridH);
        const blocksFlat = blocks.map(b => [b.w, b.h, b.id]).flat();

        const payloadBase = {
          blocks: blocksFlat,
          theta: policy.theta.buffer.slice(policy.theta.byteOffset, policy.theta.byteOffset + policy.theta.byteLength),
          epsilon: epsilon.buffer.slice(epsilon.byteOffset, epsilon.byteOffset + epsilon.byteLength),
          thetaVersion,
          epsilonSeed,
          numThreads,
          arch,
          mode: 0,
          params: [gridW, gridH, numBlocks, rolloutsPerThread, seed, maxAttempts, allowRotation, policy.sigma, numThreads, thetaVersion, epsilonSeed, 0],
        };

        const payloadPerturbed = {
          ...payloadBase,
          mode: 1,
          params: [gridW, gridH, numBlocks, rolloutsPerThread, seed + 1, maxAttempts, allowRotation, policy.sigma, numThreads, thetaVersion, epsilonSeed, 1],
        };

        yield {
          id: uuidv4(),
          payload: payloadBase,
          meta: { problemIdx, gridW, gridH, numBlocks, seed, blocks, numThreads, rolloutsPerThread, mode: 0, roundState },
          tCreate: Date.now(),
        };

        yield {
          id: uuidv4(),
          payload: payloadPerturbed,
          meta: { problemIdx, gridW, gridH, numBlocks, seed, blocks, numThreads, rolloutsPerThread, mode: 1, roundState },
          tCreate: Date.now(),
        };
      }
      logger.info('GridPack-2D RL chunker done');
    },
  };
}

function readResultF32(result) {
  if (result instanceof ArrayBuffer) return new Float32Array(result);
  if (ArrayBuffer.isView(result)) return new Float32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (Buffer.isBuffer(result)) return new Float32Array(result.buffer, result.byteOffset, Math.floor(result.byteLength / 4));
  if (result && result.type === 'Buffer' && Array.isArray(result.data)) {
    const u8 = Uint8Array.from(result.data);
    return new Float32Array(u8.buffer, u8.byteOffset, Math.floor(u8.byteLength / 4));
  }
  throw new Error('Unsupported gridpack-2d-rl result buffer type');
}

export function buildAssembler({ taskId, taskDir, config, inputArgs }) {
  const outPath = path.join(taskDir, 'output.json');
  const policyPath = inputArgs.policyPath ?? config.policyPath ?? path.join(taskDir, 'policy_state.json');

  let roundState = null;
  let chunksProcessed = 0;
  const pairs = new Map();

  return {
    integrate({ result, meta }) {
      chunksProcessed++;
      const floats = readResultF32(result);
      if (floats.length < 6) {
        logger.warn('GridPack-2D RL result too short', { length: floats.length });
        return;
      }

      if (!roundState && meta.roundState) {
        roundState = meta.roundState;
      }

      const meanReward = floats[1];
      const validCount = floats[2];
      const mode = floats[4];
      const problemIdx = meta.problemIdx;

      if (!pairs.has(problemIdx)) {
        pairs.set(problemIdx, {});
      }
      const pair = pairs.get(problemIdx);
      if (mode === 0) {
        pair.base = { meanReward, validCount };
      } else {
        pair.perturbed = { meanReward, validCount };
      }

      if (roundState) {
        if (mode === 0) {
          roundState.problemRewardsBase[problemIdx] = meanReward;
          roundState.problemValidBase[problemIdx] = validCount;
        } else {
          roundState.problemRewardsPerturbed[problemIdx] = meanReward;
          roundState.problemValidPerturbed[problemIdx] = validCount;
        }
      }
    },
    finalize() {
      let sumDelta = 0.0;
      let validPairs = 0;
      let sumBase = 0.0;
      let sumPerturbed = 0.0;

      for (const [problemIdx, pair] of pairs) {
        if (pair.base && pair.perturbed && pair.base.validCount > 0 && pair.perturbed.validCount > 0) {
          const delta = pair.perturbed.meanReward - pair.base.meanReward;
          sumDelta += delta;
          sumBase += pair.base.meanReward;
          sumPerturbed += pair.perturbed.meanReward;
          validPairs++;
        }
      }

      const meanDelta = validPairs > 0 ? sumDelta / validPairs : 0;
      const meanBase = validPairs > 0 ? sumBase / validPairs : 0;
      const meanPerturbed = validPairs > 0 ? sumPerturbed / validPairs : 0;

      let policyUpdated = false;
      if (roundState) {
        try {
          const policy = new PolicyState(policyPath, roundState.arch);
          // Reconstruct epsilon from seed
          const eps = new Float32Array(policy.thetaSize);
          let seed = roundState.epsilonSeed >>> 0;
          function randU32() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            return seed >>> 0;
          }
          function randF01() { return randU32() / 4294967295.0; }
          function randN() {
            const u1 = Math.max(randF01(), 0.0001);
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(6.28318530718 * randF01());
          }
          for (let i = 0; i < policy.thetaSize; i++) eps[i] = randN();

          const grad = new Float32Array(policy.thetaSize);
          const scale = meanDelta / roundState.sigma;
          for (let i = 0; i < policy.thetaSize; i++) {
            grad[i] = scale * eps[i];
          }

          policy.update(grad);
          policy.save(policyPath);
          policyUpdated = true;
          logger.info(`Policy updated: v${policy.version}, delta=${meanDelta.toFixed(6)}, base=${meanBase.toFixed(4)}, perturbed=${meanPerturbed.toFixed(4)}, thetaSize=${policy.thetaSize}`);
        } catch (e) {
          logger.error('Policy update failed:', e);
        }
      }

      const summary = {
        taskId,
        validPairs,
        totalPairs: pairs.size,
        meanDelta,
        meanBase,
        meanPerturbed,
        chunksProcessed,
        policyUpdated,
        thetaVersion: roundState?.thetaVersion ?? null,
        arch: roundState?.arch ?? null,
        completedAt: new Date().toISOString(),
      };

      fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));
      return { outPath, validPairs, meanDelta, policyUpdated };
    },
  };
}

export function getTotalChunks(config, inputArgs) {
  const numProblems = parseU32(inputArgs.numProblems ?? config.numProblems, 'numProblems', 10);
  return numProblems * 2;
}
