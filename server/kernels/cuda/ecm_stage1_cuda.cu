// kernels/ecm_stage1_cuda.cu  (ECM Stage 1 - Version 3, resumable, CUDA)

extern "C" {

#include <stdint.h>

// --------------------------- Types ---------------------------
struct U256 { uint32_t limbs[8]; };

struct PointXZ {
    U256 X;
    U256 Z;
};

struct InvResult {
    bool ok;
    U256 val;
};

struct CurveResult {
    bool ok;
    U256 A24m; // Montgomery domain
    U256 X1m;  // Montgomery domain
};

struct Header {
    uint32_t magic;      // "ECM1"
    uint32_t version;    // 3
    uint32_t rsv0;
    uint32_t rsv1;
    uint32_t pp_count;
    uint32_t n_curves;
    uint32_t seed_lo;
    uint32_t seed_hi;
    uint32_t base_curve;
    uint32_t flags;
    uint32_t pp_start;   // window start
    uint32_t pp_len;     // window size
};

// Layout helpers (words)
__device__ __forceinline__ uint32_t constOffset() { return 12u; }
__device__ __forceinline__ uint32_t ppOffset()    { return constOffset() + (8u*3u + 4u); }
__device__ __forceinline__ uint32_t outOffset(const Header& h) { return ppOffset() + h.pp_count; }
__device__ __forceinline__ uint32_t stateOffset(const Header& h) {
    // after output section (per curve: result(8) + status(1) + pad(3))
    return outOffset(h) + h.n_curves * (8u + 1u + 3u);
}

static const uint32_t LIMBS = 8u;
static const uint32_t STATE_WORDS_PER_CURVE = 8u + 8u + 8u + 2u; // X + Z + A24 + (sigma, curve_ok)

// --------------------------- Small helpers ---------------------------
__device__ __forceinline__ U256 set_zero(){ U256 r; #pragma unroll for(int i=0;i<8;i++) r.limbs[i]=0u; return r; }
__device__ __forceinline__ U256 set_one(){ U256 r = set_zero(); r.limbs[0]=1u; return r; }
__device__ __forceinline__ U256 u256_from_u32(uint32_t x){ U256 r=set_zero(); r.limbs[0]=x; return r; }

__device__ __forceinline__ bool is_zero(const U256& a){
    uint32_t x=0u; #pragma unroll for (int i=0;i<8;i++) x |= a.limbs[i]; return x==0u;
}

__device__ __forceinline__ int cmp(const U256& a, const U256& b){
    for (int i=7; i>=0; --i){
        uint32_t ai=a.limbs[i], bi=b.limbs[i];
        if (ai<bi) return -1;
        if (ai>bi) return  1;
    }
    return 0;
}

__device__ __forceinline__ bool is_even(const U256& a){ return (a.limbs[0] & 1u) == 0u; }

__device__ __forceinline__ U256 rshift1(const U256& a){
    U256 r; uint32_t carry=0u;
    for (int i=7; i>=0; --i){
        uint32_t w=a.limbs[i];
        r.limbs[i] = (w>>1) | (carry<<31);
        carry = w & 1u;
    }
    return r;
}

__device__ __forceinline__ uint32_t selu32(uint32_t a, uint32_t b, bool c){ return c? b : a; }

__device__ __forceinline__ void addc(uint32_t a, uint32_t b, uint32_t cin, uint32_t& sum, uint32_t& cout){
    uint32_t s  = a + b;
    uint32_t c1 = (s < a) ? 1u : 0u;
    uint32_t s2 = s + cin;
    uint32_t c2 = (s2 < cin) ? 1u : 0u;
    sum = s2; cout = c1 + c2;
}

__device__ __forceinline__ void subb(uint32_t a, uint32_t b, uint32_t bin, uint32_t& diff, uint32_t& bout){
    uint32_t d  = a - b;
    uint32_t b1 = (a < b) ? 1u : 0u;
    uint32_t d2 = d - bin;
    uint32_t b2 = (d < bin) ? 1u : 0u;
    diff = d2; bout = ((b1|b2)!=0u) ? 1u : 0u;
}

__device__ __forceinline__ U256 add_u256(const U256& a, const U256& b){
    U256 r; uint32_t c=0u;
    #pragma unroll
    for (int i=0;i<8;i++){ uint32_t s,cout; addc(a.limbs[i], b.limbs[i], c, s, cout); r.limbs[i]=s; c=cout; }
    return r;
}

__device__ __forceinline__ U256 sub_u256(const U256& a, const U256& b){
    U256 r; uint32_t br=0u;
    #pragma unroll
    for (int i=0;i<8;i++){ uint32_t d,bout; subb(a.limbs[i], b.limbs[i], br, d, bout); r.limbs[i]=d; br=bout; }
    return r;
}

__device__ __forceinline__ U256 cond_sub_N(const U256& a, const U256& N){
    return (cmp(a,N) >= 0) ? sub_u256(a,N) : a;
}

__device__ __forceinline__ U256 add_mod(const U256& a, const U256& b, const U256& N){
    return cond_sub_N(add_u256(a,b), N);
}

__device__ __forceinline__ U256 sub_mod(const U256& a, const U256& b, const U256& N){
    if (cmp(a,b) >= 0) return sub_u256(a,b);
    U256 diff = sub_u256(b,a);
    return sub_u256(N, diff);
}

// 32x32 -> (lo,hi) via 16-bit split (to mirror WGSL exactness)
__device__ __forceinline__ void mul32x32_64(uint32_t a, uint32_t b, uint32_t& lo, uint32_t& hi){
    uint32_t a0 = a & 0xFFFFu, a1 = a >> 16;
    uint32_t b0 = b & 0xFFFFu, b1 = b >> 16;
    uint32_t p00 = a0*b0;
    uint32_t p01 = a0*b1;
    uint32_t p10 = a1*b0;
    uint32_t p11 = a1*b1;
    uint32_t mid = p01 + p10;
    lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu)<<16);
    uint32_t carry = (p00>>16) + (mid>>16);
    hi = p11 + carry;
}

// --------------------------- Montgomery math (CIOS) ---------------------------
__device__ __forceinline__ U256 mont_mul(const U256& a, const U256& b, const U256& N, uint32_t n0inv32){
    uint32_t t[9]; #pragma unroll for (int i=0;i<9;i++) t[i]=0u;

    for (int i=0;i<8;i++){
        uint32_t carry=0u;
        for (int j=0;j<8;j++){
            uint32_t plo, phi; mul32x32_64(a.limbs[i], b.limbs[j], plo, phi);
            uint32_t s1, c1; addc(t[j], plo, 0u, s1, c1);
            uint32_t s2, c2; addc(s1, carry, 0u, s2, c2);
            t[j] = s2;
            carry = phi + c1 + c2;
        }
        t[8] = t[8] + carry;

        uint32_t m = t[0] * n0inv32;

        carry = 0u;
        for (int j=0;j<8;j++){
            uint32_t plo, phi; mul32x32_64(m, N.limbs[j], plo, phi);
            uint32_t s1, c1; addc(t[j], plo, 0u, s1, c1);
            uint32_t s2, c2; addc(s1, carry, 0u, s2, c2);
            t[j] = s2;
            carry = phi + c1 + c2;
        }
        t[8] = t[8] + carry;

        #pragma unroll
        for (int k=0;k<8;k++) t[k] = t[k+1];
        t[8] = 0u;
    }

    U256 r; #pragma unroll for (int i=0;i<8;i++) r.limbs[i]=t[i];
    return cond_sub_N(r, N);
}

__device__ __forceinline__ U256 mont_add(const U256& a, const U256& b, const U256& N){ return cond_sub_N(add_u256(a,b), N); }
__device__ __forceinline__ U256 mont_sub(const U256& a, const U256& b, const U256& N){
    if (cmp(a,b) >= 0) return sub_u256(a,b);
    U256 diff = sub_u256(b,a);
    return sub_u256(N, diff);
}

__device__ __forceinline__ U256 mont_sqr(const U256& a, const U256& N, uint32_t n0inv32){ return mont_mul(a,a,N,n0inv32); }
__device__ __forceinline__ U256 to_mont(const U256& a, const U256& R2, const U256& N, uint32_t n0inv32){ return mont_mul(a,R2,N,n0inv32); }
__device__ __forceinline__ U256 from_mont(const U256& a, const U256& N, uint32_t n0inv32){ return mont_mul(a, set_one(), N, n0inv32); }

// --------------------------- X-only ops ---------------------------
__device__ __forceinline__ PointXZ xDBL(const PointXZ& P, const U256& A24, const U256& N, uint32_t n0inv32){
    U256 t1 = mont_add(P.X, P.Z, N);
    U256 t2 = mont_sub(P.X, P.Z, N);
    U256 t3 = mont_sqr(t1, N, n0inv32);
    U256 t4 = mont_sqr(t2, N, n0inv32);
    U256 t5 = mont_sub(t3, t4, N);
    U256 t6 = mont_mul(A24, t5, N, n0inv32);
    U256 Z_mult = mont_add(t3, t6, N);
    PointXZ R;
    R.X = mont_mul(t3, t4, N, n0inv32);
    R.Z = mont_mul(t5, Z_mult, N, n0inv32);
    return R;
}

__device__ __forceinline__ PointXZ xADD(const PointXZ& P, const PointXZ& Q, const PointXZ& Diff, const U256& N, uint32_t n0inv32){
    U256 t1 = mont_add(P.X, P.Z, N);
    U256 t2 = mont_sub(P.X, P.Z, N);
    U256 t3 = mont_add(Q.X, Q.Z, N);
    U256 t4 = mont_sub(Q.X, Q.Z, N);
    U256 t5 = mont_mul(t1, t4, N, n0inv32);
    U256 t6 = mont_mul(t2, t3, N, n0inv32);
    U256 t1n = mont_add(t5, t6, N);
    U256 t2n = mont_sub(t5, t6, N);
    PointXZ R;
    R.X = mont_mul(mont_sqr(t1n, N, n0inv32), Diff.Z, N, n0inv32);
    R.Z = mont_mul(mont_sqr(t2n, N, n0inv32), Diff.X, N, n0inv32);
    return R;
}

__device__ __forceinline__ void cswap(PointXZ& a, PointXZ& b, uint32_t bit){
    uint32_t mask = 0u - (bit & 1u);
    #pragma unroll
    for (int i=0;i<8;i++){
        uint32_t tx = (a.X.limbs[i] ^ b.X.limbs[i]) & mask;
        a.X.limbs[i] ^= tx; b.X.limbs[i] ^= tx;
        uint32_t tz = (a.Z.limbs[i] ^ b.Z.limbs[i]) & mask;
        a.Z.limbs[i] ^= tz; b.Z.limbs[i] ^= tz;
    }
}

__device__ __forceinline__ PointXZ ladder(const PointXZ& P, uint32_t k, const U256& A24, const U256& N, uint32_t n0inv32, const U256& mont_one){
    PointXZ R0; R0.X = mont_one; R0.Z = set_zero();
    PointXZ R1 = P;
    bool started = false;

    for (int i=31; i>=0; --i){
        uint32_t bit = (k >> (uint32_t)i) & 1u;
        if (!started && bit==0u) continue;
        started = true;

        cswap(R0, R1, 1u - bit);
        PointXZ T0 = xADD(R0, R1, P, N, n0inv32);
        PointXZ T1 = xDBL(R1, A24, N, n0inv32);
        R0 = T0;
        R1 = T1;
    }
    return R0;
}

// --------------------------- RNG (LCG) ---------------------------
__device__ __forceinline__ uint32_t lcg32(uint32_t& state){
    state = state * 1103515245u + 12345u;
    return state;
}

__device__ __forceinline__ U256 next_sigma(const U256& /*N*/, uint32_t& lcg_state){
    U256 acc;
    #pragma unroll
    for (int i=0;i<8;i++) acc.limbs[i] = lcg32(lcg_state);

    U256 sigma = acc;
    if (is_zero(sigma)) sigma.limbs[0] = 6u;
    U256 one = set_one();
    if (cmp(sigma, one)==0) sigma.limbs[0]=6u;
    return sigma;
}

// --------------------------- Binary modular inverse ---------------------------
__device__ __forceinline__ InvResult mod_inverse(U256 a, const U256& N){
    if (is_zero(a)) return {false, set_zero()};
    // reduce a mod N (lightly)
    for (int k=0; k<2 && cmp(a,N) >= 0; ++k) a = sub_u256(a,N);
    if (is_zero(a)) return {false, set_zero()};

    U256 u=a, v=N, x1=set_one(), x2=set_zero();

    for (uint32_t it=0; it<20000u; ++it){
        if (cmp(u, set_one())==0) return {true, x1};
        if (cmp(v, set_one())==0) return {true, x2};
        if (is_zero(u) || is_zero(v)) break;

        while (is_even(u)){
            u = rshift1(u);
            x1 = is_even(x1) ? rshift1(x1) : rshift1(add_u256(x1, N));
        }
        while (is_even(v)){
            v = rshift1(v);
            x2 = is_even(x2) ? rshift1(x2) : rshift1(add_u256(x2, N));
        }
        if (cmp(u,v) >= 0){ u = sub_u256(u,v); x1 = sub_mod(x1, x2, N); }
        else              { v = sub_u256(v,u); x2 = sub_mod(x2, x1, N); }
    }
    return {false, set_zero()};
}

// --------------------------- Curve generation (Suyama) ---------------------------
__device__ __forceinline__ CurveResult generate_curve(const U256& sigma, const U256& N, const U256& R2, uint32_t n0inv32){
    U256 sigma_m = to_mont(sigma, R2, N, n0inv32);
    U256 five_m  = to_mont(u256_from_u32(5u),  R2, N, n0inv32);
    U256 four_m  = to_mont(u256_from_u32(4u),  R2, N, n0inv32);
    U256 three_m = to_mont(u256_from_u32(3u),  R2, N, n0inv32);
    U256 two_m   = to_mont(u256_from_u32(2u),  R2, N, n0inv32);

    U256 sigma_sq_m = mont_sqr(sigma_m, N, n0inv32);
    U256 u_m = mont_sub(sigma_sq_m, five_m, N);
    U256 v_m = mont_mul(four_m, sigma_m, N, n0inv32);

    U256 sigma_std = from_mont(sigma_m, N, n0inv32);
    U256 u_std = from_mont(u_m, N, n0inv32);
    U256 v_std = from_mont(v_m, N, n0inv32);

    U256 one = set_one();
    U256 two = u256_from_u32(2u);
    U256 N_minus_1 = sub_u256(N, one);
    U256 N_minus_2 = sub_u256(N, two);

    if (is_zero(sigma_std) || cmp(sigma_std, one)==0 || cmp(sigma_std, N_minus_1)==0 ||
        cmp(sigma_std, two)==0 || cmp(sigma_std, N_minus_2)==0){
        return {false, set_zero(), set_zero()};
    }

    InvResult inv_u = mod_inverse(u_std, N);
    InvResult inv_v = mod_inverse(v_std, N);
    if (!inv_u.ok || !inv_v.ok) return {false, set_zero(), set_zero()};

    U256 u_sq_m = mont_sqr(u_m, N, n0inv32);
    U256 v_sq_m = mont_sqr(v_m, N, n0inv32);
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
    U256 inv4_m = to_mont(inv4.val, R2, N, n0inv32);
    U256 A24m = mont_mul(A_plus_2_m, inv4_m, N, n0inv32);

    CurveResult cr; cr.ok=true; cr.A24m=A24m; cr.X1m=X1m; return cr;
}

// --------------------------- GCD (binary, odd N) ---------------------------
__device__ __forceinline__ U256 gcd_binary_u256_oddN(U256 a, U256 b /*N*/){
    if (is_zero(a)) return b;
    while (is_even(a)) a = rshift1(a);
    while (true){
        if (is_zero(b)) return a;
        while (is_even(b)) b = rshift1(b);
        if (cmp(a,b)>0){ U256 t=a; a=b; b=t; }
        b = sub_u256(b,a);
    }
}

// --------------------------- State helpers ---------------------------
__device__ __forceinline__ void load_state(uint32_t idx, const Header& h, const uint32_t* io, /*out*/U256& X, U256& Z, U256& A24, uint32_t& sigma, uint32_t& curve_ok){
    uint32_t base = stateOffset(h) + idx * STATE_WORDS_PER_CURVE;
    #pragma unroll
    for (int i=0;i<8;i++) X.limbs[i]    = io[base + i];
    #pragma unroll
    for (int i=0;i<8;i++) Z.limbs[i]    = io[base + 8u + i];
    #pragma unroll
    for (int i=0;i<8;i++) A24.limbs[i]  = io[base + 16u + i];
    sigma    = io[base + 24u];
    curve_ok = io[base + 25u];
}

__device__ __forceinline__ void save_state(uint32_t idx, const Header& h, uint32_t* io, const U256& X, const U256& Z, const U256& A24, uint32_t sigma, uint32_t curve_ok){
    uint32_t base = stateOffset(h) + idx * STATE_WORDS_PER_CURVE;
    #pragma unroll
    for (int i=0;i<8;i++) io[base + i]       = X.limbs[i];
    #pragma unroll
    for (int i=0;i<8;i++) io[base + 8u + i]  = Z.limbs[i];
    #pragma unroll
    for (int i=0;i<8;i++) io[base + 16u + i] = A24.limbs[i];
    io[base + 24u] = sigma;
    io[base + 25u] = curve_ok;
}

// --------------------------- Header loader ---------------------------
__device__ __forceinline__ Header getHeader(const uint32_t* w){
    Header h;
    h.magic      = w[0];  h.version    = w[1];  h.rsv0 = w[2];  h.rsv1 = w[3];
    h.pp_count   = w[4];  h.n_curves   = w[5];  h.seed_lo = w[6]; h.seed_hi = w[7];
    h.base_curve = w[8];  h.flags      = w[9];  h.pp_start = w[10]; h.pp_len = w[11];
    return h;
}

// --------------------------- Kernel entry ---------------------------
__global__ void execute_task(const uint32_t* __restrict__ in_words,
                              uint32_t*       __restrict__ out_words)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Read header from input (authoritative)
    Header h = getHeader(in_words);
    if (idx >= h.n_curves) return;

    const uint32_t CONST_WORDS = 8u*3u + 4u; // N(8)+R2(8)+mont_one(8)+n0inv32(1)+pad(3)
    const uint32_t CURVE_OUT_WORDS_PER = 8u + 1u + 3u;

    const uint32_t off_consts = constOffset();
    const uint32_t off_pp     = ppOffset();
    const uint32_t off_out    = outOffset(h);
    const uint32_t off_state  = stateOffset(h);
    const uint32_t total_words = off_state + h.n_curves * STATE_WORDS_PER_CURVE;

    // Make output buffer self-contained: copy header + constants + prime powers (once)
    if (idx == 0){
        const uint32_t prefix_words = off_pp + h.pp_count;
        for (uint32_t i=0;i<prefix_words;i++) out_words[i] = in_words[i];
        // (State section will be populated by threads saving their own state when needed.)
    }

    // Load constants (from input)
    uint32_t off = off_consts;
    U256 N, R2, mont_one;
    #pragma unroll
    for (int i=0;i<8;i++) N.limbs[i] = in_words[off+i];     off += 8u;
    #pragma unroll
    for (int i=0;i<8;i++) R2.limbs[i] = in_words[off+i];    off += 8u;
    #pragma unroll
    for (int i=0;i<8;i++) mont_one.limbs[i] = in_words[off+i]; off += 8u;
    uint32_t n0inv32 = in_words[off]; off += 4u; // +3 pad ignored

    const uint32_t pp_off = off_pp;
    const uint32_t out_base = off_out + idx * CURVE_OUT_WORDS_PER;

    PointXZ R;
    U256 A24m;
    uint32_t sigma_val = 0u;
    bool curve_ok = false;

    if (h.pp_start == 0u){
        // Fresh start: generate curve (deterministic per curve)
        uint32_t global_curve_idx = h.base_curve + idx;
        uint32_t rng_x = (h.seed_lo ^ global_curve_idx ^ 0x12345678u);
        uint32_t rng_y = (h.seed_hi ^ (global_curve_idx * 0x9E3779B9u) ^ 0x87654321u);
        // keep interface identical to WGSL (y unused but kept for determinism)
        if (rng_x == 0u && rng_y == 0u){
            rng_x = 0x12345678u + global_curve_idx;
            rng_y = 0x87654321u + global_curve_idx;
        }

        for (uint32_t tries=0u; tries<4u && !curve_ok; ++tries){
            U256 sigma = next_sigma(N, rng_x);
            sigma_val = sigma.limbs[0];
            CurveResult cr = generate_curve(sigma, N, R2, n0inv32);
            if (cr.ok){
                A24m = cr.A24m;
                R.X  = cr.X1m;
                R.Z  = mont_one;
                curve_ok = true;
            }
        }

        if (!curve_ok){
            // bad curve: write zeros + status=3
            #pragma unroll
            for (int i=0;i<8;i++) out_words[out_base+i] = 0u;
            out_words[out_base+8u] = 3u;
            return;
        }
    } else {
        // Resume: load saved state from INPUT (previous pass returned buffer)
        U256 X,Z; U256 A24;
        uint32_t sigma_saved=0u, curve_ok_u32=0u;
        load_state(idx, h, in_words, X, Z, A24, sigma_saved, curve_ok_u32);
        R.X = X; R.Z = Z; A24m = A24;
        sigma_val = sigma_saved;
        curve_ok = (curve_ok_u32 == 1u);
        if (!curve_ok){
            #pragma unroll
            for (int i=0;i<8;i++) out_words[out_base+i] = 0u;
            out_words[out_base+8u] = 3u;
            return;
        }
    }

    // Process prime-power window
    uint32_t end_pp = (h.pp_start + h.pp_len < h.pp_count) ? (h.pp_start + h.pp_len) : h.pp_count;
    for (uint32_t i = h.pp_start; i < end_pp; ++i){
        uint32_t pp = in_words[pp_off + i];
        if (pp > 1u){
            R = ladder(R, pp, A24m, N, n0inv32, mont_one);
        }
    }

    if (end_pp < h.pp_count){
        // Not done; save state for resume and mark status=0
        save_state(idx, h, out_words, R.X, R.Z, A24m, sigma_val, 1u);
        out_words[out_base+8u] = 0u;
    } else {
        // Done; compute final result
        U256 result = set_zero();
        uint32_t status = 1u; // default: no factor

        if ((h.flags & 1u) != 0u){
            U256 Zstd = from_mont(R.Z, N, n0inv32);
            U256 g = gcd_binary_u256_oddN(Zstd, N);
            result = g;
            U256 one = set_one();
            if (!is_zero(g) && (cmp(g, N) < 0) && (cmp(g, one) > 0)) status = 2u;
            else status = 1u;
        } else {
            result = from_mont(R.Z, N, n0inv32);
            status = 1u;
        }

        // Write final result (8 limbs) + status
        #pragma unroll
        for (int i=0;i<8;i++) out_words[out_base+i] = result.limbs[i];
        out_words[out_base+8u] = status;
    }
}

} // extern "C"
