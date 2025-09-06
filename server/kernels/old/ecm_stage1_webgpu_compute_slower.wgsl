// ecm_stage1_webgpu_fast_uniform.wgsl
// ECM Stage 1 (Montgomery curves), single-thread-per-curve.
// Uniform-control-flow safe (no early returns before barriers).
// 256-bit bigint via 8x32 limbs, CIOS Montgomery, binary ladder + pow2 shortcut.
// Workgroup tiling for prime-powers to cut global traffic.

const LIMBS : u32 = 8u;
const TILE  : u32 = 512u;       // prime-powers tile
const WG    : u32 = 64u;        // workgroup_size

struct U256 { limbs: array<u32, 8>, };
struct CurveIn { A24: U256, X1: U256, };
struct CurveOut { result: U256, status: u32, _p0:u32, _p1:u32, _p2:u32, };
struct PointXZ { X: U256, Z: U256, };

struct Consts {
  N:        U256,
  R2:       U256,
  mont_one: U256,   // R mod N
  n0inv32:  u32,
  _c0:u32, _c1:u32, _c2:u32,
};
struct Params {
  pp_count:    u32,
  num_curves:  u32,
  compute_gcd: u32, // 0: return Z (std), 1: final GCD
  _pad:        u32,
};
struct Packed {
  consts: Consts,
  primes: array<u32>,
};

@group(0) @binding(0) var<uniform>              params: Params;
@group(0) @binding(1) var<storage, read>        packed: Packed;
@group(0) @binding(2) var<storage, read>        curves: array<CurveIn>;
@group(0) @binding(3) var<storage, read_write>  outBuf: array<CurveOut>;

var<workgroup> ppTile : array<u32, TILE>;

// -------- 32x32->64 via 16-bit split (portable) --------
fn mul32x32_64(a: u32, b: u32) -> vec2<u32> {
  let a0=a & 0xFFFFu; let a1=a>>16u; let b0=b & 0xFFFFu; let b1=b>>16u;
  var p00=a0*b0; var p01=a0*b1; var p10=a1*b0; var p11=a1*b1;
  var mid=p01+p10;
  var lo=(p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
  var carry=(p00 >> 16u) + (mid >> 16u);
  var hi=p11 + carry;
  return vec2<u32>(lo, hi);
}

// -------- small helpers --------
fn addc(a:u32, b:u32, cin:u32) -> vec2<u32> {
  let s=a+b; let c1=select(0u,1u,s<a);
  let s2=s+cin; let c2=select(0u,1u,s2<cin);
  return vec2<u32>(s2, c1+c2);
}
fn subb(a:u32, b:u32, bin:u32) -> vec2<u32> {
  let d=a-b; let b1=select(0u,1u,a<b);
  let d2=d-bin; let b2=select(0u,1u,d<bin);
  return vec2<u32>(d2, select(0u,1u,(b1|b2)!=0u));
}
fn set_zero() -> U256 { var r:U256; for (var i=0u;i<LIMBS;i++){ r.limbs[i]=0u; } return r; }
fn set_one() -> U256 { var r=set_zero(); r.limbs[0]=1u; return r; }
fn is_zero(a:U256)->bool{ var x:u32=0u; for(var i=0u;i<LIMBS;i++){ x|=a.limbs[i]; } return x==0u; }
fn cmp(a:U256,b:U256)->i32{ for(var i:i32=7;i>=0;i--){let ai=a.limbs[u32(i)]; let bi=b.limbs[u32(i)]; if(ai<bi){return -1;} if(ai>bi){return 1;} } return 0; }
fn add_u256(a:U256,b:U256)->U256{ var r:U256; var c:u32=0u; for(var i=0u;i<LIMBS;i++){ let t=addc(a.limbs[i],b.limbs[i],c); r.limbs[i]=t.x; c=t.y; } return r; }
fn sub_u256(a:U256,b:U256)->U256{ var r:U256; var br:u32=0u; for(var i=0u;i<LIMBS;i++){ let t=subb(a.limbs[i],b.limbs[i],br); r.limbs[i]=t.x; br=t.y; } return r; }

// conditional subtract N if r >= N
fn cond_sub_N(a:U256, N:U256) -> U256 {
  let c = cmp(a, N);
  if (c >= 0) { return sub_u256(a, N); }
  return a;
}

// -------- Montgomery (CIOS, L=8) --------
fn mont_mul(a:U256, b:U256, N:U256, n0inv32:u32) -> U256 {
  var t: array<u32, 9>; for (var i=0u;i<9u;i++){ t[i]=0u; }

  for (var i=0u;i<LIMBS;i++) {
    // t += a_i * b
    var carry:u32=0u;
    {
      var j: u32 = 0u;
      loop {
        if (j >= LIMBS) { break; }
        let prod = mul32x32_64(a.limbs[i], b.limbs[j]);
        let s1 = addc(t[j], prod.x, 0u);
        let s2 = addc(s1.x, carry, 0u);
        t[j] = s2.x;
        carry = prod.y + s1.y + s2.y;
        j = j + 1u;
      }
    }
    t[8] = t[8] + carry;

    // m = t0 * n0inv32 mod 2^32
    let m = t[0] * n0inv32;

    // t += m * N
    carry = 0u;
    {
      var j: u32 = 0u;
      loop {
        if (j >= LIMBS) { break; }
        let prod = mul32x32_64(m, N.limbs[j]);
        let s1 = addc(t[j], prod.x, 0u);
        let s2 = addc(s1.x, carry, 0u);
        t[j] = s2.x;
        carry = prod.y + s1.y + s2.y;
        j = j + 1u;
      }
    }
    t[8] = t[8] + carry;

    // shift t right by one limb
    for (var k=0u;k<8u;k++){ t[k]=t[k+1u]; }
    t[8]=0u;
  }

  var r:U256; for (var i=0u;i<LIMBS;i++){ r.limbs[i]=t[i]; }
  return cond_sub_N(r, N);
}
fn mont_sqr(a:U256, C:Consts)->U256 { return mont_mul(a,a,C.N,C.n0inv32); }
fn mont_add(a:U256,b:U256,C:Consts)->U256{ return cond_sub_N(add_u256(a,b), C.N); }
fn mont_sub(a:U256,b:U256,C:Consts)->U256{
  let c = cmp(a,b);
  if (c < 0) { return sub_u256(add_u256(a, C.N), b); }
  return sub_u256(a,b);
}
fn to_mont(a:U256,C:Consts)->U256{ return mont_mul(a, C.R2, C.N, C.n0inv32); }
// from_mont(x) = x * 1 * R^{-1} (use literal 1 for fewer nonzero limbs)
fn from_mont(a:U256,C:Consts)->U256{ return mont_mul(a, set_one(), C.N, C.n0inv32); }

// -------- Montgomery x-only ops --------
fn xDBL(R:PointXZ, A24:U256, C:Consts)->PointXZ {
  var t1 = mont_add(R.X, R.Z, C);
  var t2 = mont_sub(R.X, R.Z, C);
  t1 = mont_sqr(t1, C);
  t2 = mont_sqr(t2, C);
  let t3 = mont_sub(t1, t2, C);
  let X2 = mont_mul(t1, t2, C.N, C.n0inv32);
  var t4 = mont_mul(t3, A24, C.N, C.n0inv32);
  t4 = mont_add(t4, t2, C);
  let Z2 = mont_mul(t3, t4, C.N, C.n0inv32);
  return PointXZ(X2, Z2);
}
fn xADD(P:PointXZ, Q:PointXZ, Diff:PointXZ, C:Consts)->PointXZ {
  var t1 = mont_add(P.X, P.Z, C);
  var t2 = mont_sub(P.X, P.Z, C);
  var t3 = mont_add(Q.X, Q.Z, C);
  var t4 = mont_sub(Q.X, Q.Z, C);
  let t5 = mont_mul(t1, t4, C.N, C.n0inv32);
  let t6 = mont_mul(t2, t3, C.N, C.n0inv32);
  t1 = mont_add(t5, t6, C);
  t2 = mont_sub(t5, t6, C);
  t1 = mont_sqr(t1, C);
  t2 = mont_sqr(t2, C);
  let X3 = mont_mul(t1, Diff.Z, C.N, C.n0inv32);
  let Z3 = mont_mul(t2, Diff.X, C.N, C.n0inv32);
  return PointXZ(X3, Z3);
}
fn cswap(a: ptr<function, PointXZ>, b: ptr<function, PointXZ>, bit: u32) {
  let mask = 0u - (bit & 1u);
  for (var i=0u;i<LIMBS;i++){
    let tx = ((*a).X.limbs[i] ^ (*b).X.limbs[i]) & mask;
    (*a).X.limbs[i] ^= tx; (*b).X.limbs[i] ^= tx;
    let tz = ((*a).Z.limbs[i] ^ (*b).Z.limbs[i]) & mask;
    (*a).Z.limbs[i] ^= tz; (*b).Z.limbs[i] ^= tz;
  }
}
fn ladder(P:PointXZ, k:u32, A24:U256, C:Consts)->PointXZ {
  var R0 = PointXZ(C.mont_one, set_zero()); // (1:0)
  var R1 = P;
  var started = false;
  for (var i:i32=31; i>=0; i--) {
    let bit = (k >> u32(i)) & 1u;
    if (!started && bit==0u) { continue; }
    started = true;
    cswap(&R0, &R1, 1u - bit);
    let T0 = xADD(R0, R1, P, C);
    let T1 = xDBL(R1, A24, C);
    R0 = T0; R1 = T1;
  }
  return R0;
}

// Fast path: multiply by 2^e = e repeated doublings
fn mul_pow2(R:PointXZ, e:u32, A24:U256, C:Consts)->PointXZ {
  var Q = R; var i:u32 = 0u;
  loop { if (i >= e) { break; } Q = xDBL(Q, A24, C); i = i + 1u; }
  return Q;
}
fn is_pow2(x:u32)->bool { return (x != 0u) && ((x & (x - 1u)) == 0u); }
fn log2_u32(x:u32)->u32 { var k:u32=0u; var y=x; while ((y>>1u)!=0u){ y>>=1u; k+=1u; } return k; }

// -------- shifts + GCD (Stein) --------
fn ctz32(x:u32)->u32{ var n:u32=0u; if(x==0u){return 32u;} var y=x; while((y&1u)==0u){ y>>=1u; n+=1u; } return n; }
fn ctz_u256(a:U256)->u32{ var tz:u32=0u; for(var i=0u;i<LIMBS;i++){ let w=a.limbs[i]; if(w==0u){ tz+=32u; continue; } tz+=ctz32(w); break; } return tz; }
fn rshiftk(a:U256,k:u32)->U256{
  if(k==0u){return a;} var r=set_zero(); let limb=k/32u; let bits=k%32u;
  for(var i=0u;i<LIMBS;i++){
    var val:u32=0u;
    if(i+limb<LIMBS){
      val=a.limbs[i+limb];
      if(bits!=0u){
        let cond=(i+limb+1u)<LIMBS; let hi=select(0u, a.limbs[i+limb+1u], cond);
        val=(val>>bits) | (hi<<(32u-bits));
      }
    }
    r.limbs[i]=val;
  }
  return r;
}
fn lshiftk(a:U256,k:u32)->U256{
  if(k==0u){return a;} var r=set_zero(); let limb=k/32u; let bits=k%32u;
  for(var ii:i32=7; ii>=0; ii--){
    var val:u32=0u;
    if(ii>=i32(limb)){
      val=a.limbs[u32(ii)-limb];
      if(bits!=0u){
        let cond=(ii>=i32(limb)+1); let lo=select(0u, a.limbs[u32(ii-1)-limb], cond);
        val=(val<<bits) | (lo>>(32u-bits));
      }
    }
    r.limbs[u32(ii)]=val;
  }
  return r;
}
fn gcd_u256(a_in:U256,b_in:U256)->U256{
  var a=a_in; var b=b_in;
  if(is_zero(a)){return b;} if(is_zero(b)){return a;}
  let shift=min(ctz_u256(a), ctz_u256(b));
  a=rshiftk(a, ctz_u256(a));
  loop {
    b=rshiftk(b, ctz_u256(b));
    if (cmp(a,b)>0) { let t=a; a=b; b=t; }
    b=sub_u256(b,a);
    if (is_zero(b)) { break; }
  }
  return lshiftk(a, shift);
}

// -------- main --------
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>)
{
  let idx = gid.x;
  let isActive = idx < params.num_curves;   // NO early return; keep barriers uniform

  let C   = packed.consts;

  // Safe per-thread input fetch (masked)
  var inC : CurveIn;
  if (isActive) {
    inC = curves[idx];
  } else {
    inC = CurveIn(set_zero(), set_zero());
  }

  // Do conversions and math for all threads (inactive threads operate on zeros)
  var A24m = to_mont(inC.A24, C);
  var X1m  = to_mont(inC.X1,  C);
  var P    = PointXZ(X1m, C.mont_one);
  var R    = P;

  var start:u32 = 0u;
  loop {
    if (start >= params.pp_count) { break; }
    let remaining = params.pp_count - start;
    let tileCount = select(remaining, TILE, remaining > TILE);

    // Cooperative tile load of prime powers (uniform barrier)
    var t = lid.x;
    loop { if (t >= tileCount) { break; } ppTile[t] = packed.primes[start + t]; t += WG; }
    workgroupBarrier();

    for (var j=0u; j<tileCount; j++) {
      let s = ppTile[j];
      if (s <= 1u) { continue; }
      if (is_pow2(s)) {
        R = mul_pow2(R, log2_u32(s), A24m, C);
      } else {
        R = ladder(R, s, A24m, C);
      }
    }

    workgroupBarrier(); // uniform
    start += tileCount;
  }

  // Only active threads write results
  if (isActive) {
    var result:U256; var status:u32 = 0u;
    if (params.compute_gcd == 1u) {
      let Zstd = from_mont(R.Z, C);
      let g    = gcd_u256(Zstd, C.N);
      result   = g;
      let one  = set_one();
      status = 1u;
      if (!is_zero(g) && (cmp(g, C.N) < 0) && (cmp(g, one) > 0)) { status = 2u; }
    } else {
      result = from_mont(R.Z, C);
      status = 0u;
    }
    outBuf[idx].result = result;
    outBuf[idx].status = status;
  }
}
