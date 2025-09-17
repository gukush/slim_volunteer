// ecm_stage1_cuda.cu
// CUDA port of WGSL ECM Stage 1 (Version 3 - resumable)
// Now with in-app prime powers generation (highest p^k <= B1),
// start confirmation, CUDA event timings, chrono end-to-end timing,
// and per-run CSV timing dump.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <ctime>

// CUDA runtime API
#include <cuda_runtime.h>
#ifdef _WIN32
  #include <device_launch_parameters.h>
#endif

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while(0)
#endif

// --------------------------- Types ---------------------------
struct U256 { uint32_t limbs[8]; };
struct PointXZ { U256 X, Z; };
struct InvResult { bool ok; U256 val; };
struct CurveResult { bool ok; U256 A24m; U256 X1m; };

// --------------------------- Small helpers ---------------------------
static inline __host__ __device__ U256 set_zero() {
  U256 r;
  #pragma unroll
  for (int i=0;i<8;i++) r.limbs[i]=0u;
  return r;
}
static inline __host__ __device__ U256 set_one() {
  U256 r=set_zero(); r.limbs[0]=1u; return r;
}
static inline __host__ __device__ U256 u256_from_u32(uint32_t x){
  U256 r=set_zero(); r.limbs[0]=x; return r;
}
static inline __host__ __device__ bool is_zero(const U256& a){
  uint32_t x=0u;
  #pragma unroll
  for (int i=0;i<8;i++) x |= a.limbs[i];
  return x==0u;
}
static inline __host__ __device__ int cmp(const U256& a, const U256& b){
  for (int i=7;i>=0;i--){
    uint32_t ai=a.limbs[i], bi=b.limbs[i];
    if (ai<bi) return -1;
    if (ai>bi) return 1;
  }
  return 0;
}
static inline __host__ __device__ bool is_even(const U256& a){ return (a.limbs[0] & 1u)==0u; }
static inline __host__ __device__ U256 rshift1(const U256& a){
  U256 r; uint32_t carry=0u;
  for (int i=7;i>=0;i--){
    uint32_t w=a.limbs[i];
    r.limbs[i]=(w>>1) | (carry<<31);
    carry = (w & 1u);
  }
  return r;
}
static inline __host__ __device__ void addc(uint32_t a, uint32_t b, uint32_t cin, uint32_t& sum, uint32_t& cout){
  uint32_t s=a+b;
  uint32_t c1 = (s<a);
  uint32_t s2=s+cin;
  uint32_t c2=(s2<cin);
  sum=s2; cout=c1+c2;
}
static inline __host__ __device__ void subb(uint32_t a, uint32_t b, uint32_t bin, uint32_t& diff, uint32_t& bout){
  uint32_t d=a-b;
  uint32_t b1=(a<b);
  uint32_t d2=d-bin;
  uint32_t b2=(d<bin);
  diff=d2; bout=((b1|b2)!=0u);
}
static inline __host__ __device__ U256 add_u256(const U256& a, const U256& b){
  U256 r; uint32_t c=0u;
  for (int i=0;i<8;i++){ uint32_t s,co; addc(a.limbs[i], b.limbs[i], c, s, co); r.limbs[i]=s; c=co; }
  return r;
}
static inline __host__ __device__ U256 sub_u256(const U256& a, const U256& b){
  U256 r; uint32_t br=0u;
  for (int i=0;i<8;i++){ uint32_t d,bo; subb(a.limbs[i], b.limbs[i], br, d, bo); r.limbs[i]=d; br=bo; }
  return r;
}
static inline __host__ __device__ U256 cond_sub_N(const U256& a, const U256& N){
  return (cmp(a,N)>=0) ? sub_u256(a,N) : a;
}
static inline __host__ __device__ U256 add_mod(const U256& a, const U256& b, const U256& N){
  return cond_sub_N(add_u256(a,b), N);
}
static inline __host__ __device__ U256 sub_mod(const U256& a, const U256& b, const U256& N){
  if (cmp(a,b)>=0) return sub_u256(a,b);
  U256 diff=sub_u256(b,a);
  return sub_u256(N,diff);
}

// 32x32 -> 64 via 16-bit split (WGSL-compatible)
static inline __host__ __device__ void mul32x32_64(uint32_t a, uint32_t b, uint32_t& lo, uint32_t& hi){
  uint32_t a0=a & 0xFFFFu, a1=a>>16;
  uint32_t b0=b & 0xFFFFu, b1=b>>16;
  uint32_t p00=a0*b0, p01=a0*b1, p10=a1*b0, p11=a1*b1;
  uint32_t mid = p01 + p10;
  lo  = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16);
  uint32_t carry = (p00>>16) + (mid>>16);
  hi = p11 + carry;
}

// --------------------------- Montgomery math (CIOS) ---------------------------
static inline __host__ __device__ U256 mont_mul(const U256& a, const U256& b, const U256& N, uint32_t n0inv32){
  uint32_t t[9];
  #pragma unroll
  for (int i=0;i<9;i++) t[i]=0u;

  for (int i=0;i<8;i++){
    uint32_t carry=0u;
    for (int j=0;j<8;j++){
      uint32_t plo,phi; mul32x32_64(a.limbs[i], b.limbs[j], plo, phi);
      uint32_t s1, c1; addc(t[j], plo, 0u, s1, c1);
      uint32_t s2, c2; addc(s1, carry, 0u, s2, c2);
      t[j]=s2;
      carry = phi + c1 + c2;
    }
    t[8] = t[8] + carry;

    uint32_t m = t[0] * n0inv32;

    carry=0u;
    for (int j=0;j<8;j++){
      uint32_t plo,phi; mul32x32_64(m, N.limbs[j], plo, phi);
      uint32_t s1,c1; addc(t[j], plo, 0u, s1, c1);
      uint32_t s2,c2; addc(s1, carry, 0u, s2, c2);
      t[j]=s2;
      carry = phi + c1 + c2;
    }
    t[8] = t[8] + carry;

    for (int k=0;k<8;k++) t[k]=t[k+1];
    t[8]=0u;
  }
  U256 r; for (int i=0;i<8;i++) r.limbs[i]=t[i];
  return cond_sub_N(r, N);
}
static inline __host__ __device__ U256 mont_add(const U256& a, const U256& b, const U256& N){ return cond_sub_N(add_u256(a,b), N); }
static inline __host__ __device__ U256 mont_sub(const U256& a, const U256& b, const U256& N){
  if (cmp(a,b)>=0) return sub_u256(a,b);
  U256 diff=sub_u256(b,a);
  return sub_u256(N,diff);
}
static inline __host__ __device__ U256 mont_sqr(const U256& a, const U256& N, uint32_t n0inv32){ return mont_mul(a,a,N,n0inv32); }
static inline __host__ __device__ U256 to_mont(const U256& a, const U256& R2, const U256& N, uint32_t n0inv32){ return mont_mul(a,R2,N,n0inv32); }
static inline __host__ __device__ U256 from_mont(const U256& a, const U256& N, uint32_t n0inv32){ return mont_mul(a,set_one(),N,n0inv32); }

// --------------------------- X-only ops ---------------------------
static inline __host__ __device__ PointXZ xDBL(const PointXZ& P, const U256& A24, const U256& N, uint32_t n0inv32){
  U256 t1 = mont_add(P.X, P.Z, N);
  U256 t2 = mont_sub(P.X, P.Z, N);
  U256 t3 = mont_sqr(t1, N, n0inv32);
  U256 t4 = mont_sqr(t2, N, n0inv32);
  U256 t5 = mont_sub(t3, t4, N);
  U256 t6 = mont_mul(A24, t5, N, n0inv32);
  U256 Z_mult = mont_add(t3, t6, N);
  U256 X2 = mont_mul(t3, t4, N, n0inv32);
  U256 Z2 = mont_mul(t5, Z_mult, N, n0inv32);
  return PointXZ{X2, Z2};
}
static inline __host__ __device__ PointXZ xADD(const PointXZ& P, const PointXZ& Q, const PointXZ& Diff, const U256& N, uint32_t n0inv32){
  U256 t1 = mont_add(P.X, P.Z, N);
  U256 t2 = mont_sub(P.X, P.Z, N);
  U256 t3 = mont_add(Q.X, Q.Z, N);
  U256 t4 = mont_sub(Q.X, Q.Z, N);
  U256 t5 = mont_mul(t1, t4, N, n0inv32);
  U256 t6 = mont_mul(t2, t3, N, n0inv32);
  U256 t1n = mont_add(t5, t6, N);
  U256 t2n = mont_sub(t5, t6, N);
  U256 X3 = mont_mul(mont_sqr(t1n, N, n0inv32), Diff.Z, N, n0inv32);
  U256 Z3 = mont_mul(mont_sqr(t2n, N, n0inv32), Diff.X, N, n0inv32);
  return PointXZ{X3, Z3};
}
static inline __host__ __device__ void cswap(PointXZ& a, PointXZ& b, uint32_t bit){
  uint32_t mask = 0u - (bit & 1u);
  #pragma unroll
  for (int i=0;i<8;i++){
    uint32_t tx = (a.X.limbs[i]^b.X.limbs[i]) & mask; a.X.limbs[i]^=tx; b.X.limbs[i]^=tx;
    uint32_t tz = (a.Z.limbs[i]^b.Z.limbs[i]) & mask; a.Z.limbs[i]^=tz; b.Z.limbs[i]^=tz;
  }
}
static inline __host__ __device__ PointXZ ladder(const PointXZ& P, uint32_t k, const U256& A24, const U256& N, uint32_t n0inv32, const U256& mont_one){
  PointXZ R0{mont_one, set_zero()};
  PointXZ R1=P;
  bool started=false;
  for (int i=31;i>=0;i--){
    uint32_t bit = (k>>i) & 1u;
    if (!started && bit==0u) continue;
    started=true;
    cswap(R0,R1, 1u - bit);
    PointXZ T0 = xADD(R0, R1, P, N, n0inv32);
    PointXZ T1 = xDBL(R1, A24, N, n0inv32);
    R0=T0; R1=T1;
  }
  return R0;
}

// --------------------------- RNG ---------------------------
static inline __host__ __device__ uint32_t lcg32(uint32_t& state){
  uint32_t ns = state * 1103515245u + 12345u;
  state = ns; return ns;
}
static inline __host__ __device__ U256 next_sigma(const U256& /*N*/, uint32_t& lcg_state){
  U256 acc;
  #pragma unroll
  for (int i=0;i<8;i++) acc.limbs[i]=lcg32(lcg_state);
  if (is_zero(acc)) acc.limbs[0]=6u;
  if (cmp(acc,set_one())==0) acc.limbs[0]=6u;
  return acc;
}

// --------------------------- Binary modular inverse ---------------------------
// (forward declared above but definition already present; keeping here is fine)
// static inline __host__ __device__ U256 add_u256(const U256&,const U256&);

static inline __host__ __device__ InvResult mod_inverse(U256 a_in, const U256& N){
  if (is_zero(a_in)) return {false,set_zero()};
  for (int k=0;k<2 && cmp(a_in,N)>=0;k++) a_in=sub_u256(a_in,N);
  if (is_zero(a_in)) return {false,set_zero()};

  U256 u=a_in, v=N, x1=set_one(), x2=set_zero();
  for (uint32_t iter=0; iter<20000u; iter++){
    if (cmp(u,set_one())==0) return {true,x1};
    if (cmp(v,set_one())==0) return {true,x2};
    if (is_zero(u) || is_zero(v)) break;

    while (is_even(u)){
      u = rshift1(u);
      if (is_even(x1)) x1 = rshift1(x1);
      else             x1 = rshift1(add_u256(x1, N));
    }
    while (is_even(v)){
      v = rshift1(v);
      if (is_even(x2)) x2 = rshift1(x2);
      else             x2 = rshift1(add_u256(x2, N));
    }
    if (cmp(u,v)>=0){ u=sub_u256(u,v); x1=sub_mod(x1,x2,N); }
    else            { v=sub_u256(v,u); x2=sub_mod(x2,x1,N); }
  }
  return {false,set_zero()};
}

// --------------------------- Curve generation (Suyama) ---------------------------
static inline __host__ __device__ CurveResult generate_curve(const U256& sigma, const U256& N, const U256& R2, uint32_t n0inv32){
  U256 sigma_m = to_mont(sigma, R2, N, n0inv32);
  U256 five_m  = to_mont(u256_from_u32(5u),  R2, N, n0inv32);
  U256 four_m  = to_mont(u256_from_u32(4u),  R2, N, n0inv32);
  U256 three_m = to_mont(u256_from_u32(3u),  R2, N, n0inv32);
  U256 two_m   = to_mont(u256_from_u32(2u),  R2, N, n0inv32);

  U256 sigma_sq_m = mont_sqr(sigma_m, N, n0inv32);
  U256 u_m = mont_sub(sigma_sq_m, five_m, N);
  U256 v_m = mont_mul(four_m, sigma_m, N, n0inv32);

  U256 sigma_std = from_mont(sigma_m, N, n0inv32);
  U256 u_std     = from_mont(u_m,     N, n0inv32);
  U256 v_std     = from_mont(v_m,     N, n0inv32);

  U256 one=set_one(), two=u256_from_u32(2u);
  U256 N_minus_1=sub_u256(N, one);
  U256 N_minus_2=sub_u256(N, two);

  if (is_zero(sigma_std) || cmp(sigma_std,one)==0 || cmp(sigma_std,N_minus_1)==0 ||
      cmp(sigma_std,two)==0 || cmp(sigma_std,N_minus_2)==0)
    return {false, set_zero(), set_zero()};

  InvResult inv_u=mod_inverse(u_std,N);
  InvResult inv_v=mod_inverse(v_std,N);
  if (!inv_u.ok || !inv_v.ok) return {false, set_zero(), set_zero()};

  U256 u_sq_m = mont_sqr(u_m,N,n0inv32);
  U256 v_sq_m = mont_sqr(v_m,N,n0inv32);
  U256 u2_v2_m = mont_sub(u_sq_m, v_sq_m, N);
  U256 u2_v2_sq_m = mont_sqr(u2_v2_m, N, n0inv32);
  U256 u2_v2_cubed_m = mont_mul(u2_v2_m, u2_v2_sq_m, N, n0inv32);

  U256 four_uv_m = mont_mul(four_m, mont_mul(u_m, v_m, N, n0inv32), N, n0inv32);
  U256 four_uv_sq_m = mont_sqr(four_uv_m, N, n0inv32);

  U256 four_uv_sq_std = from_mont(four_uv_sq_m, N, n0inv32);
  InvResult inv_four_uv_sq = mod_inverse(four_uv_sq_std, N);
  if (!inv_four_uv_sq.ok) return {false, set_zero(), set_zero()};
  U256 inv_four_uv_sq_m = to_mont(inv_four_uv_sq.val, R2, N, n0inv32);

  U256 X1m = mont_mul(u2_v2_cubed_m, inv_four_uv_sq_m, N, n0inv32);

  U256 vm_u_m = mont_sub(v_m, u_m, N);
  U256 vm_u_sq_m = mont_sqr(vm_u_m, N, n0inv32);
  U256 vm_u_cubed_m = mont_mul(vm_u_m, vm_u_sq_m, N, n0inv32);

  U256 three_u_m = mont_mul(three_m, u_m, N, n0inv32);
  U256 three_u_plus_v_m = mont_add(three_u_m, v_m, N);

  U256 u_cubed_m = mont_mul(u_m, u_sq_m, N, n0inv32);
  U256 four_u3_v_m = mont_mul(four_m, mont_mul(u_cubed_m, v_m, N, n0inv32), N, n0inv32);

  U256 four_u3_v_std = from_mont(four_u3_v_m, N, n0inv32);
  InvResult inv_four_u3_v = mod_inverse(four_u3_v_std, N);
  if (!inv_four_u3_v.ok) return {false, set_zero(), set_zero()};
  U256 inv_four_u3_v_m = to_mont(inv_four_u3_v.val, R2, N, n0inv32);

  U256 numerator_m = mont_mul(vm_u_cubed_m, three_u_plus_v_m, N, n0inv32);
  U256 A_plus_2_m = mont_mul(numerator_m, inv_four_u3_v_m, N, n0inv32);

  InvResult inv4 = mod_inverse(u256_from_u32(4u), N);
  if (!inv4.ok) return {false, set_zero(), set_zero()};
  U256 inv4_m = to_mont(inv4.val, R2, N, n0inv32);
  U256 A24m = mont_mul(A_plus_2_m, inv4_m, N, n0inv32);

  return {true, A24m, X1m};
}

// --------------------------- GCD (binary, odd N) ---------------------------
static inline __host__ __device__ U256 gcd_binary_u256_oddN(U256 a, U256 b){
  if (is_zero(a)) return b;
  while (is_even(a)) a=rshift1(a);
  for(;;){
    if (is_zero(b)) return a;
    while (is_even(b)) b=rshift1(b);
    if (cmp(a,b)>0){ U256 t=a; a=b; b=t; }
    b=sub_u256(b,a);
  }
}

// --------------------------- Kernel ---------------------------
__global__ void ecm_stage1_kernel(
  uint32_t n_curves,
  U256 N, U256 R2, U256 mont_one, uint32_t n0inv32,
  const uint32_t* __restrict__ pp, uint32_t pp_count, uint32_t pp_start, uint32_t pp_len,
  uint32_t seed_lo, uint32_t seed_hi, uint32_t base_curve, uint32_t flags,
  uint32_t* __restrict__ X_state, uint32_t* __restrict__ Z_state, uint32_t* __restrict__ A24_state,
  uint32_t* __restrict__ sigma_state, uint32_t* __restrict__ curveok_state,
  uint32_t* __restrict__ out_words
){
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_curves) return;

  uint32_t out_base = idx * 12u;
  PointXZ R; U256 A24m; uint32_t sigma_val=0u; bool curve_ok=false;

  uint32_t pp_end = pp_start + pp_len;
  if (pp_end > pp_count) pp_end = pp_count;

  if (pp_start == 0u){
    uint32_t global_curve_idx = base_curve + idx;
    uint32_t rng = seed_lo ^ global_curve_idx ^ 0x12345678u;
    uint32_t dummy_y = seed_hi ^ (global_curve_idx * 0x9E3779B9u) ^ 0x87654321u;
    if (rng==0u && dummy_y==0u){
      rng=0x12345678u+global_curve_idx;
      dummy_y=0x87654321u+global_curve_idx;
    }
    for (uint32_t tries=0; tries<4u && !curve_ok; ++tries){
      U256 sigma = next_sigma(N, rng);
      sigma_val = sigma.limbs[0];
      CurveResult cr = generate_curve(sigma, N, R2, n0inv32);
      if (cr.ok){ A24m=cr.A24m; R={cr.X1m, mont_one}; curve_ok=true; }
    }
    if (!curve_ok){
      #pragma unroll
      for (int i=0;i<8;i++) out_words[out_base+i]=0u;
      out_words[out_base+8]=3u;
      return;
    }
  } else {
    #pragma unroll
    for (int i=0;i<8;i++){
      R.X.limbs[i]=X_state[idx*8u + i];
      R.Z.limbs[i]=Z_state[idx*8u + i];
      A24m.limbs[i]=A24_state[idx*8u + i];
    }
    sigma_val = sigma_state[idx];
    curve_ok = (curveok_state[idx]==1u);
    if(!curve_ok){
      #pragma unroll
      for (int i=0;i<8;i++) out_words[out_base+i]=0u;
      out_words[out_base+8]=3u;
      return;
    }
  }

  for (uint32_t i=pp_start; i<pp_end; ++i){
    uint32_t e=pp[i];
    if (e>1u) R = ladder(R, e, A24m, N, n0inv32, mont_one);
  }

  if (pp_end < pp_count){
    #pragma unroll
    for (int i=0;i<8;i++){
      X_state[idx*8u + i] = R.X.limbs[i];
      Z_state[idx*8u + i] = R.Z.limbs[i];
      A24_state[idx*8u + i] = A24m.limbs[i];
    }
    sigma_state[idx] = sigma_val;
    curveok_state[idx]=1u;
    out_words[out_base+8]=0u; // needs_more
  } else {
    U256 result=set_zero(); uint32_t status=1u;
    if ((flags & 1u)!=0u){
      U256 Zstd = from_mont(R.Z, N, n0inv32);
      U256 g = gcd_binary_u256_oddN(Zstd, N);
      result = g;
      U256 one=set_one();
      status = (!is_zero(g) && (cmp(g,N)<0) && (cmp(g,one)>0)) ? 2u : 1u;
    } else {
      result = from_mont(R.Z, N, n0inv32);
      status=1u;
    }
    #pragma unroll
    for (int i=0;i<8;i++) out_words[out_base+i]=result.limbs[i];
    out_words[out_base+8]=status;
  }
}

// ---------------- Host utilities ----------------
static bool parse_hex_u256(const std::string& s, U256& out){
  std::string t=s;
  if (t.rfind("0x",0)==0 || t.rfind("0X",0)==0) t=t.substr(2);
  if (t.size()>64) return false;
  if (t.size()<64) t = std::string(64 - t.size(), '0') + t;
  for (int i=0;i<8;i++){
    uint64_t limb=0;
    for (int b=0;b<8;b++){
      char c=t[64-1-(i*8+b)];
      int v = (c>='0'&&c<='9') ? (c-'0')
            : (c>='a'&&c<='f') ? (10+c-'a')
            : (c>='A'&&c<='F') ? (10+c-'A') : -1;
      if (v<0) return false;
      limb |= (uint64_t)v << (4*b);
    }
    out.limbs[i]=(uint32_t)limb;
  }
  return true;
}
static std::string to_hex_le(const U256& x){
  std::ostringstream oss; oss<<"0x";
  for (int i=7;i>=0;i--) oss<<std::hex<<std::setw(8)<<std::setfill('0')<<std::nouppercase<<x.limbs[i];
  return oss.str();
}
static uint32_t mont_n0inv32(uint32_t n0){ uint32_t inv=1u; for (int i=0;i<5;i++) inv *= (2u - n0*inv); return (uint32_t)(0u - inv); }
static U256 two_pow_k_mod_N(uint32_t k, const U256& N){
  U256 x=set_one();
  for (uint32_t i=0;i<k;i++){ x=add_u256(x,x); if(cmp(x,N)>=0) x=sub_u256(x,N); }
  return x;
}

// ---- Prime powers file ----
static std::vector<uint32_t> read_pp_file(const std::string& path){
  std::ifstream fin(path);
  if (!fin) { std::cerr<<"Failed to open pp file: "<<path<<"\n"; std::exit(1); }
  std::vector<uint32_t> v; v.reserve(1<<20);
  std::string line;
  while (std::getline(fin, line)){
    if (line.empty()) continue;
    uint64_t val=0;
    if (line.rfind("0x",0)==0 || line.rfind("0X",0)==0) val=std::stoull(line, nullptr, 16);
    else val=std::stoull(line, nullptr, 10);
    v.push_back((uint32_t)val);
  }
  return v;
}

// ---- Prime powers generator (highest p^k <= B1) ----
static std::vector<uint32_t> generate_prime_powers(uint32_t B1){
  std::vector<uint8_t> sieve(B1 + 1, 1);
  // set 0 and 1 safely without pointless comparisons
  sieve[0] = 0;
  if (B1 >= 1u) sieve[1] = 0;

  std::vector<uint32_t> powers; powers.reserve(B1/10 + 10);
  for (uint32_t p=2; p<=B1; ++p){
    if (!sieve[p]) continue;
    if ((uint64_t)p*p <= B1){
      for (uint32_t i=p*p; i<=B1; i+=p) sieve[i]=0;
    }
    uint64_t pk=p;
    while (pk * (uint64_t)p <= B1) pk *= p;
    powers.push_back((uint32_t)pk);
  }
  return powers; // ascending p, emitting p^k for each prime
}

// ---------------- CLI ----------------
struct Args{
  U256 N; uint32_t n_curves=0;
  std::string pp_file; uint32_t B1=0;
  uint32_t seed_lo=0, seed_hi=0, base_curve=0, flags=1;
  uint32_t pp_start=0, pp_len=0;
  std::string resume_in, resume_out;
  int threads_per_block=64;
};
static void usage_and_exit(const char* prog){
  std::cerr
    << "Usage: " << prog << " --modulus 0x<64hex> --n-curves N (--pp-file file | --b1 B1)\n"
    << "  [--seed-lo u32] [--seed-hi u32] [--base-curve u32] [--flags u32]\n"
    << "  [--pp-start u32] [--pp-len u32]\n"
    << "  [--resume-in file] [--resume-out file]\n"
    << "  [--threads-per-block 64]\n";
  std::exit(1);
}
static Args parse_args(int argc, char** argv){
  Args a; bool haveN=false, haveNC=false;
  for (int i=1;i<argc;i++){
    std::string k=argv[i];
    auto need = [&](int more){ if (i+more>=argc) usage_and_exit(argv[0]); };
    if (k=="--modulus"){ need(1); haveN=true; if(!parse_hex_u256(argv[++i], a.N)) usage_and_exit(argv[0]); }
    else if (k=="--n-curves"){ need(1); haveNC=true; a.n_curves=std::stoul(argv[++i]); }
    else if (k=="--pp-file"){ need(1); a.pp_file=argv[++i]; }
    else if (k=="--b1" || k=="--B1"){ need(1); a.B1=(uint32_t)std::stoul(argv[++i]); }
    else if (k=="--seed-lo"){ need(1); a.seed_lo=std::stoul(argv[++i]); }
    else if (k=="--seed-hi"){ need(1); a.seed_hi=std::stoul(argv[++i]); }
    else if (k=="--base-curve"){ need(1); a.base_curve=std::stoul(argv[++i]); }
    else if (k=="--flags"){ need(1); a.flags=std::stoul(argv[++i]); }
    else if (k=="--pp-start"){ need(1); a.pp_start=std::stoul(argv[++i]); }
    else if (k=="--pp-len"){ need(1); a.pp_len=std::stoul(argv[++i]); }
    else if (k=="--resume-in"){ need(1); a.resume_in=argv[++i]; }
    else if (k=="--resume-out"){ need(1); a.resume_out=argv[++i]; }
    else if (k=="--threads-per-block"){ need(1); a.threads_per_block=std::stoi(argv[++i]); }
    else usage_and_exit(argv[0]);
  }
  if (!haveN || !haveNC) usage_and_exit(argv[0]);
  if (a.pp_file.empty() && a.B1==0){ std::cerr<<"Provide either --pp-file or --b1\n"; std::exit(1); }
  if ((a.pp_len==0u) && (a.pp_start!=0u)) { std::cerr<<"If --pp-start is set, --pp-len must be >0\n"; std::exit(1); }
  return a;
}

// ---------------- MAIN ----------------
int main(int argc, char** argv){
  Args args = parse_args(argc, argv);

  if ((args.N.limbs[0] & 1u)==0u){
    std::cerr<<"Modulus N must be odd.\n"; return 1;
  }
  uint32_t n0inv32 = mont_n0inv32(args.N.limbs[0]);
  U256 mont_one = two_pow_k_mod_N(256, args.N);
  U256 R2       = two_pow_k_mod_N(512, args.N);

  // Build prime powers
  std::vector<uint32_t> pp = args.pp_file.empty() ? generate_prime_powers(args.B1) : read_pp_file(args.pp_file);
  if (pp.empty()){ std::cerr<<"Prime powers are empty.\n"; return 1; }
  uint32_t pp_count = (uint32_t)pp.size();
  uint32_t pp_start = args.pp_start;
  uint32_t pp_len   = (args.pp_len==0u) ? pp_count : args.pp_len;
  if (pp_start > pp_count) { std::cerr<<"pp_start beyond pp_count\n"; return 1; }
  if (pp_start + pp_len > pp_count) pp_len = pp_count - pp_start;

  // Optional resume
  size_t n=args.n_curves;
  std::vector<uint32_t> X_h(n*8,0), Z_h(n*8,0), A24_h(n*8,0), sigma_h(n,0), ok_h(n,0);
  auto read_state = [&](const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;
    f.read((char*)X_h.data(), n*8*4);
    f.read((char*)Z_h.data(), n*8*4);
    f.read((char*)A24_h.data(), n*8*4);
    f.read((char*)sigma_h.data(), n*4);
    f.read((char*)ok_h.data(), n*4);
    return (bool)f;
  };
  if (!args.resume_in.empty()){
    if (!read_state(args.resume_in)){ std::cerr<<"Could not read resume state; starting fresh.\n"; }
  }

  // --- WAIT FOR USER INPUT BEFORE START ---
  std::cout << "ECM Stage 1 (CUDA)\n"
            << "  n_curves: " << n << "\n"
            << "  modulus : " << to_hex_le(args.N) << "\n"
            << "  pp_count: " << pp_count << "  window: [" << pp_start << ", " << (pp_start+pp_len) << ")\n"
            << "  seeds   : lo=" << args.seed_lo << " hi=" << args.seed_hi << "\n"
            << "Press ENTER to start..." << std::flush;
  std::string _line; std::getline(std::cin, _line);

  // --- Timing setup ---
  using clk = std::chrono::steady_clock;
  auto t_begin = clk::now();

  cudaEvent_t eH2DStart, eH2DStop, eKStart, eKStop, eD2HStart, eD2HStop;
  CHECK_CUDA(cudaEventCreate(&eH2DStart));
  CHECK_CUDA(cudaEventCreate(&eH2DStop));
  CHECK_CUDA(cudaEventCreate(&eKStart));
  CHECK_CUDA(cudaEventCreate(&eKStop));
  CHECK_CUDA(cudaEventCreate(&eD2HStart));
  CHECK_CUDA(cudaEventCreate(&eD2HStop));

  float msH2D = 0.0f, msKernel = 0.0f, msD2H = 0.0f;

  // Device buffers
  uint32_t *d_pp=nullptr,*dX=nullptr,*dZ=nullptr,*dA24=nullptr,*dSigma=nullptr,*dOk=nullptr,*dOut=nullptr;
  CHECK_CUDA(cudaMalloc(&d_pp, pp_count*sizeof(uint32_t)));

  // H2D timing start
  cudaEventRecord(eH2DStart);

  CHECK_CUDA(cudaMemcpy(d_pp, pp.data(), pp_count*sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dX, n*8*4));
  CHECK_CUDA(cudaMalloc(&dZ, n*8*4));
  CHECK_CUDA(cudaMalloc(&dA24, n*8*4));
  CHECK_CUDA(cudaMalloc(&dSigma, n*4));
  CHECK_CUDA(cudaMalloc(&dOk, n*4));
  CHECK_CUDA(cudaMalloc(&dOut, n*12*4));

  CHECK_CUDA(cudaMemcpy(dX, X_h.data(), n*8*4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dZ, Z_h.data(), n*8*4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA24, A24_h.data(), n*8*4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dSigma, sigma_h.data(), n*4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dOk, ok_h.data(), n*4, cudaMemcpyHostToDevice));

  // H2D timing stop
  cudaEventRecord(eH2DStop);
  CHECK_CUDA(cudaEventSynchronize(eH2DStop));
  CHECK_CUDA(cudaEventElapsedTime(&msH2D, eH2DStart, eH2DStop));

  // Launch
  int TPB = args.threads_per_block;
  int blocks = (int)((n + TPB - 1) / TPB);

  // Kernel timing start
  cudaEventRecord(eKStart);
  ecm_stage1_kernel<<<blocks, TPB>>>(
    (uint32_t)n,
    args.N, R2, mont_one, n0inv32,
    d_pp, pp_count, pp_start, pp_len,
    args.seed_lo, args.seed_hi, args.base_curve, args.flags,
    dX, dZ, dA24, dSigma, dOk, dOut
  );
  cudaEventRecord(eKStop);
  CHECK_CUDA(cudaEventSynchronize(eKStop));
  CHECK_CUDA(cudaEventElapsedTime(&msKernel, eKStart, eKStop));
  CHECK_CUDA(cudaPeekAtLastError());

  // D2H timing start
  cudaEventRecord(eD2HStart);

  // Read back
  std::vector<uint32_t> out_h(n*12,0);
  CHECK_CUDA(cudaMemcpy(out_h.data(), dOut, n*12*4, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(X_h.data(), dX, n*8*4, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(Z_h.data(), dZ, n*8*4, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(A24_h.data(), dA24, n*8*4, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(sigma_h.data(), dSigma, n*4, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ok_h.data(), dOk, n*4, cudaMemcpyDeviceToHost));

  // D2H timing stop
  cudaEventRecord(eD2HStop);
  CHECK_CUDA(cudaEventSynchronize(eD2HStop));
  CHECK_CUDA(cudaEventElapsedTime(&msD2H, eD2HStart, eD2HStop));

  auto t_end = clk::now();
  double msTotal = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count() / 1000.0;

  std::cout << std::fixed << std::setprecision(3)
            << "Timing (ms) - H2D: " << msH2D
            << "  Kernel: " << msKernel
            << "  D2H: " << msD2H
            << "  End-to-end: " << msTotal << "\n";

  // Print results
  for (size_t i=0;i<n;i++){
    U256 res; for (int j=0;j<8;j++) res.limbs[j]=out_h[i*12 + j];
    uint32_t status = out_h[i*12 + 8];
    std::cout << "curve " << i
              << "  status " << status
              << "  result " << to_hex_le(res)
              << "\n";
  }

  // Save resume if needed
  bool any_needs=false;
  for (size_t i=0;i<n;i++){
    if (out_h[i*12 + 8]==0u){ any_needs=true; break; }
  }
  if (!args.resume_out.empty() && (any_needs || !args.resume_in.empty())){
    std::ofstream f(args.resume_out, std::ios::binary);
    if(f){
      f.write((const char*)X_h.data(), n*8*4);
      f.write((const char*)Z_h.data(), n*8*4);
      f.write((const char*)A24_h.data(), n*8*4);
      f.write((const char*)sigma_h.data(), n*4);
      f.write((const char*)ok_h.data(), n*4);
    }
  }

  // --- Dump timings to a unique CSV per run ---
  {
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
  #ifdef _WIN32
    localtime_s(&tm, &tt);
  #else
    localtime_r(&tt, &tm);
  #endif
    auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::ostringstream ts;
    ts << (tm.tm_year + 1900)
       << std::setw(2) << std::setfill('0') << (tm.tm_mon + 1)
       << std::setw(2) << std::setfill('0') << tm.tm_mday
       << "_"
       << std::setw(2) << std::setfill('0') << tm.tm_hour
       << std::setw(2) << std::setfill('0') << tm.tm_min
       << std::setw(2) << std::setfill('0') << tm.tm_sec
       << "_" << (epoch_ms % 1000);

    std::string pp_source = args.pp_file.empty() ? (std::string("B1=") + std::to_string(args.B1)) : args.pp_file;

    cudaDeviceProp prop{}; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);

    std::ostringstream csvname;
    csvname << "ecm_timing_" << ts.str() << ".csv";

    std::ofstream csv(csvname.str());
    if (csv) {
      csv << "timestamp,device,n_curves,threads_per_block,blocks,pp_source,pp_count,pp_start,pp_len,seed_lo,seed_hi,base_curve,flags,H2D_ms,Kernel_ms,D2H_ms,EndToEnd_ms\n";
      csv << std::quoted(ts.str()) << ","
          << std::quoted(prop.name) << ","
          << n << ","
          << TPB << ","
          << blocks << ","
          << std::quoted(pp_source) << ","
          << pp_count << ","
          << pp_start << ","
          << pp_len << ","
          << args.seed_lo << ","
          << args.seed_hi << ","
          << args.base_curve << ","
          << args.flags << ","
          << msH2D << ","
          << msKernel << ","
          << msD2H << ","
          << msTotal << "\n";
      csv.close();
      std::cout << "Timing CSV: " << csvname.str() << "\n";
    } else {
      std::cerr << "Failed to write timing CSV" << "\n";
    }
  }

  // Destroy CUDA events
  cudaEventDestroy(eH2DStart);
  cudaEventDestroy(eH2DStop);
  cudaEventDestroy(eKStart);
  cudaEventDestroy(eKStop);
  cudaEventDestroy(eD2HStart);
  cudaEventDestroy(eD2HStop);

  // Cleanup
  cudaFree(d_pp); cudaFree(dX); cudaFree(dZ); cudaFree(dA24); cudaFree(dSigma); cudaFree(dOk); cudaFree(dOut);
  return 0;
}
