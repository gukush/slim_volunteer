// native_cuda_ecm_stage1.cu
// Build: nvcc -O3 -std=c++17 -arch=sm_70 -o exe-ecm-stage1 native_cuda_ecm_stage1.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

static void die(const char* msg){ std::fprintf(stderr, "%s\n", msg); std::exit(1); }
static void read_all_stdin(std::vector<uint8_t>& out){
    const size_t CH = 1<<20;
    std::vector<uint8_t> buf; buf.reserve(CH);
    uint8_t tmp[CH];
    size_t n;
    while((n = std::fread(tmp,1,CH,stdin))>0){
        buf.insert(buf.end(), tmp, tmp+n);
    }
    if(std::ferror(stdin)) die("stdin read error");
    out.swap(buf);
}
static void write_exact(const void* src, size_t n){
    const uint8_t* p = (const uint8_t*)src; size_t put=0;
    while(put<n){
        size_t w = std::fwrite(p+put,1,n-put,stdout);
        if(w==0) die("stdout write error");
        put+=w;
    }
}

// -------------------- Shared layout constants (ECM v3) --------------------
#define LIMBS32 8u
#define HEADER_WORDS_V3 12u
#define CONST_WORDS     (8u*3u + 4u)   // N(8)+R2(8)+mont_one(8)+n0inv32(1)+pad(3)
#define OUT_WORDS_PER   (8u + 1u + 3u) // result(8)+status(1)+pad(3)
#define STATE_WORDS_PER (8u + 8u + 8u + 2u) // X(8)+Z(8)+A24(8)+(sigma,curve_ok)

struct Header32 {
    uint32_t magic, version, rsv0, rsv1;
    uint32_t pp_count, n_curves, seed_lo, seed_hi;
    uint32_t base_curve, flags, pp_start, pp_len;
};

__device__ __forceinline__ uint32_t constOffset(){ return 12u; }
__device__ __forceinline__ uint32_t ppOffset(){ return constOffset() + (8u*3u + 4u); }
__device__ __forceinline__ uint32_t outOffset(const Header32& h){ return ppOffset() + h.pp_count; }
__device__ __forceinline__ uint32_t stateOffset(const Header32& h){
    return outOffset(h) + h.n_curves * OUT_WORDS_PER;
}

// -------------------- 256-bit types (8 x u32) --------------------
struct U256 { uint32_t w[LIMBS32]; };

__device__ __forceinline__ void set_zero(U256& a){
    #pragma unroll
    for(int i=0;i<8;i++) a.w[i]=0u;
}
__device__ __forceinline__ void set_one (U256& a){ set_zero(a); a.w[0]=1u; }
__device__ __forceinline__ int  cmp(const U256& a, const U256& b){
    for(int i=7;i>=0;i--){ if(a.w[i]<b.w[i]) return -1; if(a.w[i]>b.w[i]) return 1; } return 0;
}
__device__ __forceinline__ bool is_zero(const U256& a){
    uint32_t x=0u;
    #pragma unroll
    for(int i=0;i<8;i++) x|=a.w[i];
    return x==0u;
}
__device__ __forceinline__ bool is_even(const U256& a){ return (a.w[0] & 1u)==0u; }

__device__ __forceinline__ void add_u256(U256& r, const U256& a, const U256& b){
    uint32_t c=0u;
    #pragma unroll
    for(int i=0;i<8;i++){
        uint64_t s = (uint64_t)a.w[i] + b.w[i] + c;
        r.w[i] = (uint32_t)s;
        c = (uint32_t)(s >> 32);
    }
}
__device__ __forceinline__ void sub_u256(U256& r, const U256& a, const U256& b){
    uint32_t br=0u;
    #pragma unroll
    for(int i=0;i<8;i++){
        uint64_t aa = a.w[i], bb = b.w[i];
        uint64_t d = aa - bb - br;
        r.w[i] = (uint32_t)d;
        br = ((aa < bb) || (aa==bb && br));
    }
}
__device__ __forceinline__ void cond_sub_N(U256& a, const U256& N){
    // if a >= N then a -= N
    int ge = (cmp(a,N) >= 0);
    if(!ge) return;
    U256 t; sub_u256(t,a,N); a = t;
}
__device__ __forceinline__ void add_mod(U256& r, const U256& a, const U256& b, const U256& N){
    add_u256(r,a,b); cond_sub_N(r,N);
}
__device__ __forceinline__ void sub_mod(U256& r, const U256& a, const U256& b, const U256& N){
    if(cmp(a,b) >= 0){ sub_u256(r,a,b); }
    else {
        U256 d; sub_u256(d,b,a);
        sub_u256(r,N,d);
    }
}

// 32x32->64 by native 64-bit
__device__ __forceinline__ void mul32(uint32_t a, uint32_t b, uint32_t& lo, uint32_t& hi){
    uint64_t p = (uint64_t)a * b;
    lo = (uint32_t)p;
    hi = (uint32_t)(p>>32);
}

// Montgomery CIOS (word=32, L=8)
__device__ void mont_mul(U256& r, const U256& a, const U256& b, const U256& N, uint32_t n0inv32){
    uint32_t t[9];
    #pragma unroll
    for(int i=0;i<9;i++) t[i]=0u;

    #pragma unroll
    for(int i=0;i<8;i++){
        uint32_t carry=0u;
        // t += a_i * b
        #pragma unroll
        for(int j=0;j<8;j++){
            uint32_t lo, hi; mul32(a.w[i], b.w[j], lo, hi);
            uint64_t s = (uint64_t)t[j] + lo + carry;
            t[j] = (uint32_t)s;
            carry = (uint32_t)((s>>32) + hi);
        }
        t[8] = t[8] + carry;

        uint32_t m = t[0] * n0inv32;

        carry = 0u;
        #pragma unroll
        for(int j=0;j<8;j++){
            uint32_t lo, hi; mul32(m, N.w[j], lo, hi);
            uint64_t s = (uint64_t)t[j] + lo + carry;
            t[j] = (uint32_t)s;
            carry = (uint32_t)((s>>32) + hi);
        }
        t[8] = t[8] + carry;

        // shift down one limb
        #pragma unroll
        for(int k=0;k<8;k++) t[k] = t[k+1];
        t[8] = 0u;
    }
    #pragma unroll
    for(int i=0;i<8;i++) r.w[i] = t[i];
    cond_sub_N(r,N);
}
__device__ __forceinline__ void mont_sqr(U256& r, const U256& a, const U256& N, uint32_t n0){ mont_mul(r,a,a,N,n0); }

__device__ __forceinline__ void to_mont(U256& r, const U256& a, const U256& R2, const U256& N, uint32_t n0){ mont_mul(r,a,R2,N,n0); }
__device__ __forceinline__ void from_mont(U256& r, const U256& a, const U256& oneM, const U256& N, uint32_t n0){ mont_mul(r,a,oneM,N,n0); }

// helpers
__device__ __forceinline__ U256 u256_from_u32(uint32_t x){ U256 r; set_zero(r); r.w[0]=x; return r; }

// X-only ops
struct PointXZ { U256 X, Z; };

__device__ void mont_add_point(U256& r, const U256& a, const U256& b, const U256& N){ add_mod(r,a,b,N); }
__device__ void mont_sub_point(U256& r, const U256& a, const U256& b, const U256& N){ sub_mod(r,a,b,N); }

__device__ void xDBL(PointXZ& R2, const PointXZ& R, const U256& A24, const U256& N, uint32_t n0){
    U256 t1,t2,t3,t4;
    mont_add_point(t1, R.X, R.Z, N);
    mont_sub_point(t2, R.X, R.Z, N);
    mont_sqr(t1,t1,N,n0);
    mont_sqr(t2,t2,N,n0);
    mont_sub_point(t3,t1,t2,N);
    mont_mul(R2.X, t1, t2, N, n0);
    mont_mul(t4, t3, A24, N, n0);
    mont_add_point(t4, t4, t2, N);
    mont_mul(R2.Z, t3, t4, N, n0);
}

__device__ void xADD(PointXZ& R3, const PointXZ& P, const PointXZ& Q, const PointXZ& Diff, const U256& N, uint32_t n0){
    U256 t1,t2,t3,t4,t5,t6;
    mont_add_point(t1, P.X, P.Z, N);
    mont_sub_point(t2, P.X, P.Z, N);
    mont_add_point(t3, Q.X, Q.Z, N);
    mont_sub_point(t4, Q.X, Q.Z, N);
    mont_mul(t5, t1, t4, N, n0);
    mont_mul(t6, t2, t3, N, n0);
    mont_add_point(t1, t5, t6, N);
    mont_sub_point(t2, t5, t6, N);
    mont_sqr(t1, t1, N, n0);
    mont_sqr(t2, t2, N, n0);
    mont_mul(R3.X, t1, Diff.Z, N, n0);
    mont_mul(R3.Z, t2, Diff.X, N, n0);
}

__device__ void cswap(PointXZ& A, PointXZ& B, uint32_t bit){
    uint32_t mask = 0u - (bit & 1u);
    #pragma unroll
    for(int i=0;i<8;i++){
        uint32_t tx = (A.X.w[i] ^ B.X.w[i]) & mask;
        A.X.w[i]^=tx; B.X.w[i]^=tx;
        uint32_t tz = (A.Z.w[i] ^ B.Z.w[i]) & mask;
        A.Z.w[i]^=tz; B.Z.w[i]^=tz;
    }
}

__device__ void ladder(PointXZ& R0, const PointXZ& P, uint32_t k, const U256& A24, const U256& N, uint32_t n0, const U256& oneM){
    PointXZ R1 = P;
    R0.X = oneM; set_zero(R0.Z);
    int started = 0;
    for(int i=31;i>=0;i--){
        uint32_t bit = (k>>i)&1u;
        if(!started && bit==0u) continue;
        started = 1;
        cswap(R0, R1, 1u - bit);
        PointXZ t0,t1;
        xADD(t0, R0, R1, P, N, n0);
        xDBL(t1, R1, A24, N, n0);
        R0 = t0; R1 = t1;
    }
}

// RNG + simple helpers (LCG32)
__device__ __forceinline__ uint32_t lcg32(uint32_t& s){ s = s*1103515245u + 12345u; return s; }
__device__ U256 next_sigma(const U256& /*N*/, uint32_t& state){
    U256 sigma;
    #pragma unroll
    for(int i=0;i<8;i++) sigma.w[i] = lcg32(state);
    if(is_zero(sigma)){ set_one(sigma); sigma.w[0]=6u; }
    if(sigma.w[0]==1u){ sigma.w[0]=6u; }
    return sigma;
}

// rshift1 and helpers
__device__ U256 rshift1(const U256& a){
    U256 r; uint32_t c=0u;
    for(int i=7;i>=0;i--){
        uint32_t w=a.w[i];
        r.w[i] = (w>>1) | (c<<31); c = w & 1u;
        if(i==0) break;
    }
    return r;
}
__device__ __forceinline__ U256 add_raw(const U256& a, const U256& b){ U256 r; add_u256(r,a,b); return r; }
__device__ __forceinline__ U256 sub_raw(const U256& a, const U256& b){ U256 r; sub_u256(r,a,b); return r; }

// Binary inverse (non-Montgomery domain)
__device__ struct InvResult { int ok; U256 v; };

__device__ InvResult mod_inverse(U256 a, const U256& N){
    if(is_zero(a)) return {0, {}};
    // reduce a (a < 2N)
    for(int k=0;k<2 && cmp(a,N)>=0;k++) { U256 t; sub_u256(t,a,N); a=t; }
    if(is_zero(a)) return {0, {}};

    U256 u=a, v=N, x1, x2; set_one(x1); set_zero(x2);

    for(int it=0; it<20000; it++){
        U256 one; set_one(one);
        if(cmp(u,one)==0) return {1,x1};
        if(cmp(v,one)==0) return {1,x2};
        if(is_zero(u) || is_zero(v)) break;

        while(is_even(u)){
            u = rshift1(u);
            if(is_even(x1)) x1 = rshift1(x1);
            else            x1 = rshift1(add_raw(x1,N));
        }
        while(is_even(v)){
            v = rshift1(v);
            if(is_even(x2)) x2 = rshift1(x2);
            else            x2 = rshift1(add_raw(x2,N));
        }
        if(cmp(u,v) >= 0){
            U256 t; sub_u256(t,u,v); u=t;
            U256 xm; sub_mod(xm,x1,x2,N); x1=xm;
        } else {
            U256 t; sub_u256(t,v,u); v=t;
            U256 xm; sub_mod(xm,x2,x1,N); x2=xm;
        }
    }
    InvResult bad={0,{}}; set_zero(bad.v); return bad;
}

// Curve generation (Suyama), closely following WGSL
__device__ struct CurveGen { int ok; U256 A24m; U256 X1m; };

__device__ CurveGen generate_curve(const U256& sigma, const U256& N, const U256& R2, uint32_t n0, const U256& oneM){
    // Convert some constants to Montgomery
    U256 five= u256_from_u32(5u), four=u256_from_u32(4u), three=u256_from_u32(3u), two=u256_from_u32(2u);
    U256 five_m, four_m, three_m, two_m, sigma_m;
    to_mont(five_m, five, R2, N, n0);
    to_mont(four_m, four, R2, N, n0);
    to_mont(three_m,three,R2,N,n0);
    to_mont(two_m,  two, R2, N, n0);
    to_mont(sigma_m, sigma, R2, N, n0);

    U256 sigma_sq_m; mont_sqr(sigma_sq_m, sigma_m, N, n0);
    U256 u_m; sub_mod(u_m, sigma_sq_m, five_m, N);
    U256 v_m; mont_mul(v_m, four_m, sigma_m, N, n0);

    U256 sigma_std; from_mont(sigma_std, sigma_m, oneM, N, n0);
    U256 one; set_one(one);
    U256 twoStd = two;
    U256 Nm1; sub_u256(Nm1, N, one);
    U256 Nm2; sub_u256(Nm2, N, twoStd);
    if(is_zero(sigma_std) || cmp(sigma_std,one)==0 || cmp(sigma_std,Nm1)==0 || cmp(sigma_std,twoStd)==0 || cmp(sigma_std,Nm2)==0){
        CurveGen bad={0,{},{}};
        return bad;
    }

    // inverses in standard domain
    U256 u_std, v_std; from_mont(u_std, u_m, oneM, N, n0); from_mont(v_std, v_m, oneM, N, n0);
    InvResult inv_u = mod_inverse(u_std, N);
    InvResult inv_v = mod_inverse(v_std, N);
    if(!inv_u.ok || !inv_v.ok){ CurveGen bad={0,{},{} }; return bad; }

    // X1 = (u^2 - v^2)^3 / (4 u v)^2
    U256 u_sq_m; mont_sqr(u_sq_m, u_m, N, n0);
    U256 v_sq_m; mont_sqr(v_sq_m, v_m, N, n0);
    U256 u2_v2_m; sub_mod(u2_v2_m, u_sq_m, v_sq_m, N);
    U256 u2_v2_sq_m; mont_sqr(u2_v2_sq_m, u2_v2_m, N, n0);
    U256 u2_v2_cu_m; mont_mul(u2_v2_cu_m, u2_v2_m, u2_v2_sq_m, N, n0);
    U256 four_uv_m; U256 tmp; mont_mul(tmp, u_m, v_m, N, n0); mont_mul(four_uv_m, four_m, tmp, N, n0);
    U256 four_uv_sq_m; mont_sqr(four_uv_sq_m, four_uv_m, N, n0);

    U256 four_uv_sq_std; from_mont(four_uv_sq_std, four_uv_sq_m, oneM, N, n0);
    InvResult inv_4uv2 = mod_inverse(four_uv_sq_std, N); if(!inv_4uv2.ok){ CurveGen bad={0,{},{} }; return bad; }
    U256 inv_4uv2_m; to_mont(inv_4uv2_m, inv_4uv2.v, R2, N, n0);
    U256 X1m; mont_mul(X1m, u2_v2_cu_m, inv_4uv2_m, N, n0);

    // A24
    U256 vm_u_m; sub_mod(vm_u_m, v_m, u_m, N);
    U256 vm_u_sq_m; mont_sqr(vm_u_sq_m, vm_u_m, N, n0);
    U256 vm_u_cu_m; mont_mul(vm_u_cu_m, vm_u_m, vm_u_sq_m, N, n0);
    U256 three_u_m; mont_mul(three_u_m, three_m, u_m, N, n0);
    U256 three_u_plus_v_m; add_mod(three_u_plus_v_m, three_u_m, v_m, N);

    U256 u_cu_m; mont_mul(u_cu_m, u_m, u_sq_m, N, n0);
    U256 four_u3_v_m; mont_mul(tmp, u_cu_m, v_m, N, n0); mont_mul(four_u3_v_m, four_m, tmp, N, n0);
    U256 four_u3_v_std; from_mont(four_u3_v_std, four_u3_v_m, oneM, N, n0);
    InvResult inv_4u3v = mod_inverse(four_u3_v_std, N); if(!inv_4u3v.ok){ CurveGen bad={0,{},{} }; return bad; }
    U256 inv_4u3v_m; to_mont(inv_4u3v_m, inv_4u3v.v, R2, N, n0);

    U256 numerator_m; mont_mul(numerator_m, vm_u_cu_m, three_u_plus_v_m, N, n0);
    U256 A_plus_2_m; mont_mul(A_plus_2_m, numerator_m, inv_4u3v_m, N, n0);

    // A24 = (A+2)/4
    InvResult inv4 = mod_inverse(u256_from_u32(4u), N);
    U256 inv4_m; to_mont(inv4_m, inv4.v, R2, N, n0);
    U256 A24m; mont_mul(A24m, A_plus_2_m, inv4_m, N, n0);

    CurveGen ok = {1, A24m, X1m};
    return ok;
}

// GCD (binary)
__device__ int ctz32(uint32_t x){ return __ffs(x) - 1; }

__device__ int ctz256(const U256& a){
    for(int i=0;i<8;i++){
        if(a.w[i]) return i*32 + ctz32(a.w[i]);
    }
    return 256;
}
__device__ void rshiftk(U256& a, int k){
    if(k<=0) return;
    int limb = k/32, bits = k%32;
    if(limb){
        for(int i=0;i<8;i++) a.w[i] = (i+limb<8) ? a.w[i+limb] : 0u;
    }
    if(bits){
        uint32_t carry=0u;
        for(int i=7;i>=0;i--){
            uint32_t nc = a.w[i] << (32-bits);
            a.w[i] = (a.w[i] >> bits) | carry;
            carry = nc;
            if(i==0) break;
        }
    }
}
__device__ void lshiftk(U256& a, int k){
    if(k<=0) return;
    int limb = k/32, bits = k%32;
    if(limb){
        for(int i=7;i>=0;i--) a.w[i] = (i>=limb)? a.w[i-limb] : 0u;
    }
    if(bits){
        uint32_t carry=0u;
        for(int i=0;i<8;i++){
            uint32_t nc = a.w[i] >> (32-bits);
            a.w[i] = (a.w[i] << bits) | carry;
            carry = nc;
        }
    }
}
__device__ U256 gcd_u256(U256 a, U256 b){
    if(is_zero(a)) return b;
    if(is_zero(b)) return a;
    int s = min(ctz256(a), ctz256(b));
    rshiftk(a, ctz256(a));
    do{
        rshiftk(b, ctz256(b));
        if(cmp(a,b)>0){ U256 t=a; a=b; b=t; }
        U256 t; sub_u256(t,b,a); b=t;
    }while(!is_zero(b));
    lshiftk(a,s); return a;
}

// Load/save little-endian words from global io
__device__ __forceinline__ Header32 load_header(const uint32_t* io){
    Header32 h;
    #pragma unroll
    for(int i=0;i<12;i++) ((uint32_t*)&h)[i] = io[i];
    return h;
}
__device__ __forceinline__ void store_header(uint32_t* io, const Header32& h){
    #pragma unroll
    for(int i=0;i<12;i++) ((uint32_t*)&h)[i] = ((const uint32_t*)&h)[i];
}

// Load constants
__device__ void load_consts(const uint32_t* io, U256& N, U256& R2, U256& oneM, uint32_t& n0){
    uint32_t off = constOffset();
    #pragma unroll
    for(int i=0;i<8;i++) N.w[i]    = io[off+i]; off+=8;
    #pragma unroll
    for(int i=0;i<8;i++) R2.w[i]   = io[constOffset()+8 + i];
    #pragma unroll
    for(int i=0;i<8;i++) oneM.w[i] = io[constOffset()+16 + i];
    n0 = io[constOffset()+24];
}

// State helpers
__device__ void load_state(uint32_t* io, uint32_t idx, const Header32& h, U256& X, U256& Z, U256& A24, uint32_t& sigma, uint32_t& curve_ok){
    uint32_t base = stateOffset(h) + idx * STATE_WORDS_PER;
    #pragma unroll
    for(int i=0;i<8;i++) X.w[i] = io[base + i];
    #pragma unroll
    for(int i=0;i<8;i++) Z.w[i] = io[base + 8 + i];
    #pragma unroll
    for(int i=0;i<8;i++) A24.w[i]=io[base + 16 + i];
    sigma    = io[base + 24];
    curve_ok = io[base + 25];
}
__device__ void save_state(uint32_t* io, uint32_t idx, const Header32& h, const U256& X, const U256& Z, const U256& A24, uint32_t sigma, uint32_t ok){
    uint32_t base = stateOffset(h) + idx * STATE_WORDS_PER;
    #pragma unroll
    for(int i=0;i<8;i++) io[base + i]       = X.w[i];
    #pragma unroll
    for(int i=0;i<8;i++) io[base + 8 + i]   = Z.w[i];
    #pragma unroll
    for(int i=0;i<8;i++) io[base + 16 + i]  = A24.w[i];
    io[base + 24] = sigma;
    io[base + 25] = ok;
}

// -------------------- Kernel: one pass over [pp_start, pp_start+pp_len) --------------------
__global__ void ecm_stage1_v3(uint32_t* io){
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    Header32 h = load_header(io);
    if(tid >= h.n_curves) return;

    U256 N,R2,oneM; uint32_t n0;
    load_consts(io, N,R2,oneM, n0);

    uint32_t out_base = outOffset(h) + tid * OUT_WORDS_PER;
    uint32_t pp_off   = ppOffset();
    uint32_t pp_end   = min(h.pp_start + h.pp_len, h.pp_count);

    // Fresh/start or resume?
    U256 A24m, X, Z;
    uint32_t sigma_val=0u, curve_ok=0u;

    if(h.pp_start == 0u){
        // Make curve (few tries)
        uint32_t s = (h.seed_lo ^ (h.base_curve + tid) ^ 0x12345678u);
        if(s==0u) s = 0x12345678u + tid;
        int ok=0;
        for(int tries=0; tries<4 && !ok; ++tries){
            U256 sigma = next_sigma(N, s);
            sigma_val = sigma.w[0];
            CurveGen cg = generate_curve(sigma, N, R2, n0, oneM);
            if(cg.ok){
                A24m = cg.A24m;
                X = cg.X1m;
                Z = oneM;
                ok=1; curve_ok=1u;
            }
        }
        if(!curve_ok){
            // bad curve
            #pragma unroll
            for(int i=0;i<8;i++) io[out_base+i]=0u;
            io[out_base+8]=3u; // bad
            return;
        }
    } else {
        load_state(io, tid, h, X, Z, A24m, sigma_val, curve_ok);
        if(!curve_ok){
            #pragma unroll
            for(int i=0;i<8;i++) io[out_base+i]=0u;
            io[out_base+8]=3u;
            return;
        }
    }

    // Process prime powers in the current window
    PointXZ R{X,Z}, T;
    for(uint32_t i=h.pp_start; i<pp_end; ++i){
        uint32_t s = io[pp_off + i];
        if(s>1u){
            ladder(T, R, s, A24m, N, n0, oneM);
            R = T;
        }
    }

    // If not finished, save state and mark needs_more
    if(pp_end < h.pp_count){
        save_state(io, tid, h, R.X, R.Z, A24m, sigma_val, 1u);
        io[out_base + 8] = 0u; // needs_more
        return;
    }

    // Finished stage 1: return gcd(Z, N) if flags&1, else Z
    U256 result; set_zero(result);
    uint32_t status = 1u; // default no factor

    if((h.flags & 1u) != 0u){
        U256 Zstd; from_mont(Zstd, R.Z, oneM, N, n0);
        U256 g = gcd_u256(Zstd, N);
        result = g;
        U256 one; set_one(one);
        if(!is_zero(g) && cmp(g,N) < 0 && cmp(g,one) > 0){
            status = 2u; // factor found
        } else {
            status = 1u; // no factor
        }
    } else {
        from_mont(result, R.Z, oneM, N, n0);
        status = 1u;
    }

    #pragma unroll
    for(int i=0;i<8;i++) io[out_base+i] = result.w[i];
    io[out_base+8] = status;
}

// -------------------- Host: resumable loop, full-buffer IO --------------------
int main(){
    // Read entire IO buffer (header + consts + pp + outputs + state)
    std::vector<uint8_t> buf;
    read_all_stdin(buf);
    if(buf.size() < HEADER_WORDS_V3*4) die("buffer too small for header");

    // Compute geometry from header
    const uint32_t* u32 = reinterpret_cast<const uint32_t*>(buf.data());
    Header32 h;
    std::memcpy(&h, u32, sizeof(Header32));
    if(h.version < 3u){
        // upgrade header in-place
        ((uint32_t*)buf.data())[1]  = 3u;
        ((uint32_t*)buf.data())[10] = 0u; // pp_start
        ((uint32_t*)buf.data())[11] = 0u; // pp_len
        std::memcpy(&h, buf.data(), sizeof(Header32));
    }

    const uint32_t outputOffset = outOffset(h);
    const uint32_t stateOff     = stateOffset(h);
    const uint64_t totalWords   = (uint64_t)stateOff + (uint64_t)h.n_curves * STATE_WORDS_PER;
    const uint64_t needBytes    = totalWords * 4ull;
    if(buf.size() < needBytes){
        buf.resize(needBytes, 0u);
    }

    // Device buffer
    uint32_t* d_io = nullptr;
    cudaMalloc(&d_io, buf.size());
    cudaMemcpy(d_io, buf.data(), buf.size(), cudaMemcpyHostToDevice);

    // Loop windows (stay near ~500ms/pass like your JS)
    const double TARGET_MS = 500.0;
    uint32_t pp_start = 0;
    uint32_t pp_len   = h.pp_count > 1500 ? 1500 : h.pp_count;
    int pass = 0;

    // Infer grid (WG size 64)
    const uint32_t WG = 64;
    uint32_t blocks = (h.n_curves + WG - 1)/WG;
    dim3 grid(blocks,1,1), block(WG,1,1);

    while(pp_start < h.pp_count){
        uint32_t this_len = (pp_len > (h.pp_count-pp_start)) ? (h.pp_count-pp_start) : pp_len;

        // Patch header[pp_start, pp_len] on device
        ((uint32_t*)buf.data())[10] = pp_start;
        ((uint32_t*)buf.data())[11] = this_len;
        cudaMemcpy(d_io, buf.data(), HEADER_WORDS_V3*4, cudaMemcpyHostToDevice);

        // Launch one pass
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        ecm_stage1_v3<<<grid, block>>>(d_io);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms=0.f; cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0); cudaEventDestroy(t1);

        cudaError_t err = cudaDeviceSynchronize();
        if(err != cudaSuccess){ std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err)); return 2; }

        // Adaptive window
        if(ms < TARGET_MS/2 && pp_len < 10000)       pp_len = min(10000u, pp_len*2);
        else if(ms > TARGET_MS*1.5 && pp_len > 100)  pp_len = max(100u, (uint32_t)((double)pp_len * TARGET_MS / (ms+1e-3)));

        pp_start += this_len;
        pass++;
    }

    // Copy back full IO buffer and write to stdout (assembler expects header+consts+pp+outputs)
    cudaMemcpy(buf.data(), d_io, buf.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_io);

    // Only send until end of output section (no need to stream state back)
    const uint64_t resultWords = outOffset(h) + (uint64_t)h.n_curves * OUT_WORDS_PER;
    const uint64_t resultBytes = resultWords * 4ull;
    write_exact(buf.data(), resultBytes);
    return 0;
}
