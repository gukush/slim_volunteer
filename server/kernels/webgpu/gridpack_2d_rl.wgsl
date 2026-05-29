const MAX_BLOCKS: u32 = 16u;
const MAX_GRID_H: u32 = 32u;
const THETA_SIZE: u32 = 2400u;
const MAGIC: u32 = 0x47524C50u;

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

fn mlpForward(st: array<f32,8>, useEps: bool, sigma: f32) -> array<f32,64> {
    var hidden: array<f32,32>;
    for (var i=0u; i<32u; i=i+1u) {
        var sum = theta[256u+i];
        if (useEps) { sum = sum + sigma * epsilon[256u+i]; }
        for (var j=0u; j<8u; j=j+1u) {
            var w = theta[i*8u+j];
            if (useEps) { w = w + sigma * epsilon[i*8u+j]; }
            sum = sum + st[j] * w;
        }
        hidden[i] = max(sum, 0.0);
    }
    var logits: array<f32,64>;
    for (var i=0u; i<64u; i=i+1u) {
        var sum = theta[2336u+i];
        if (useEps) { sum = sum + sigma * epsilon[2336u+i]; }
        for (var j=0u; j<32u; j=j+1u) {
            var w = theta[288u+i*32u+j];
            if (useEps) { w = w + sigma * epsilon[288u+i*32u+j]; }
            sum = sum + hidden[j] * w;
        }
        logits[i] = sum;
    }
    return logits;
}

fn sampleAction(logits: array<f32,64>, s: ptr<function, u32>, gw: u32, gh: u32) -> vec2<u32> {
    var mx = logits[0];
    for (var i=1u; i<64u; i=i+1u) { if (logits[i] > mx) { mx = logits[i]; } }
    var expSum = 0.0;
    var probs: array<f32,64>;
    for (var i=0u; i<64u; i=i+1u) { let e = exp(logits[i]-mx); probs[i]=e; expSum=expSum+e; }
    let r = randF01(s) * expSum;
    var c = 0.0; var a = 0u;
    for (var i=0u; i<64u; i=i+1u) { c=c+probs[i]; if (c>=r) { a=i; break; } }
    let rx = a % 8u; let ry = a / 8u;
    let rw = max(1u, gw/8u); let rh = max(1u, gh/8u);
    let px = rx*rw + u32(randF01(s)*f32(rw));
    let py = ry*rh + u32(randF01(s)*f32(rh));
    return vec2<u32>(min(px,gw-1u), min(py,gh-1u));
}

fn extractFeatures(g: ptr<function, array<u32, MAX_GRID_H>>, gw: u32, gh: u32, bw: u32, bh: u32, left: u32, total: u32) -> array<f32,8> {
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
    return array<f32,8>(
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
}
