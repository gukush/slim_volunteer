// ecm_stage1_kernel.cu - CUDA kernel for ECM Stage 1
// All device code in this single file for JIT compilation

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------- Types & Structures ---------------------------
struct U256 {
    unsigned int limbs[8];
};

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
    U256 A24m;
    U256 X1m;
};

struct Header {
    unsigned int magic;      // "ECM1"
    unsigned int version;    // 2
    unsigned int rsv0;
    unsigned int rsv1;
    unsigned int pp_count;
    unsigned int n_curves;
    unsigned int seed_lo;
    unsigned int seed_hi;
    unsigned int base_curve;
    unsigned int flags;
    unsigned int rsv2;
    unsigned int rsv3;
};

// --------------------------- Device Functions ---------------------------
__device__ inline U256 set_zero() {
    U256 r;
    #pragma unroll
    for (int i = 0; i < 8; i++) r.limbs[i] = 0;
    return r;
}

__device__ inline U256 set_one() {
    U256 r = set_zero();
    r.limbs[0] = 1;
    return r;
}

__device__ inline U256 u256_from_u32(unsigned int x) {
    U256 r = set_zero();
    r.limbs[0] = x;
    return r;
}

__device__ inline bool is_zero(const U256& a) {
    unsigned int x = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) x |= a.limbs[i];
    return x == 0;
}

__device__ inline int cmp(const U256& a, const U256& b) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

__device__ inline bool is_even(const U256& a) {
    return (a.limbs[0] & 1) == 0;
}

__device__ inline U256 rshift1(const U256& a) {
    U256 r;
    unsigned int carry = 0;
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        unsigned int w = a.limbs[i];
        r.limbs[i] = (w >> 1) | (carry << 31);
        carry = w & 1;
    }
    return r;
}

__device__ inline void addc(unsigned int a, unsigned int b, unsigned int cin,
                            unsigned int* out, unsigned int* cout) {
    unsigned long long sum = (unsigned long long)a + b + cin;
    *out = (unsigned int)sum;
    *cout = (unsigned int)(sum >> 32);
}

__device__ inline void subb(unsigned int a, unsigned int b, unsigned int bin,
                            unsigned int* out, unsigned int* bout) {
    unsigned long long diff = (unsigned long long)a - b - bin;
    *out = (unsigned int)diff;
    *bout = (diff >> 32) & 1;
}

__device__ inline U256 add_u256(const U256& a, const U256& b) {
    U256 r;
    unsigned int c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        addc(a.limbs[i], b.limbs[i], c, &r.limbs[i], &c);
    }
    return r;
}

__device__ inline U256 sub_u256(const U256& a, const U256& b) {
    U256 r;
    unsigned int br = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        subb(a.limbs[i], b.limbs[i], br, &r.limbs[i], &br);
    }
    return r;
}

__device__ inline U256 cond_sub_N(const U256& a, const U256& N) {
    if (cmp(a, N) >= 0) return sub_u256(a, N);
    return a;
}

__device__ inline U256 add_mod(const U256& a, const U256& b, const U256& N) {
    return cond_sub_N(add_u256(a, b), N);
}

__device__ inline U256 sub_mod(const U256& a, const U256& b, const U256& N) {
    if (cmp(a, b) >= 0) return sub_u256(a, b);
    U256 diff = sub_u256(b, a);
    return sub_u256(N, diff);
}

__device__ inline void mul32x32_64(unsigned int a, unsigned int b,
                                   unsigned int* lo, unsigned int* hi) {
    unsigned long long prod = (unsigned long long)a * b;
    *lo = (unsigned int)prod;
    *hi = (unsigned int)(prod >> 32);
}

// Montgomery multiplication (CIOS method)
__device__ U256 mont_mul(const U256& a, const U256& b, const U256& N, unsigned int n0inv32) {
    unsigned int t[9] = {0};
    
    for (int i = 0; i < 8; i++) {
        unsigned int carry = 0;
        for (int j = 0; j < 8; j++) {
            unsigned int lo, hi;
            mul32x32_64(a.limbs[i], b.limbs[j], &lo, &hi);
            unsigned int s1, c1;
            addc(t[j], lo, 0, &s1, &c1);
            unsigned int s2, c2;
            addc(s1, carry, 0, &s2, &c2);
            t[j] = s2;
            carry = hi + c1 + c2;
        }
        t[8] += carry;
        
        unsigned int m = t[0] * n0inv32;
        
        carry = 0;
        for (int j = 0; j < 8; j++) {
            unsigned int lo, hi;
            mul32x32_64(m, N.limbs[j], &lo, &hi);
            unsigned int s1, c1;
            addc(t[j], lo, 0, &s1, &c1);
            unsigned int s2, c2;
            addc(s1, carry, 0, &s2, &c2);
            t[j] = s2;
            carry = hi + c1 + c2;
        }
        t[8] += carry;
        
        for (int k = 0; k < 8; k++) t[k] = t[k+1];
        t[8] = 0;
    }
    
    U256 r;
    for (int i = 0; i < 8; i++) r.limbs[i] = t[i];
    return cond_sub_N(r, N);
}

__device__ inline U256 mont_add(const U256& a, const U256& b, const U256& N) {
    return cond_sub_N(add_u256(a, b), N);
}

__device__ inline U256 mont_sub(const U256& a, const U256& b, const U256& N) {
    if (cmp(a, b) >= 0) return sub_u256(a, b);
    U256 diff = sub_u256(b, a);
    return sub_u256(N, diff);
}

__device__ inline U256 mont_sqr(const U256& a, const U256& N, unsigned int n0inv32) {
    return mont_mul(a, a, N, n0inv32);
}

__device__ inline U256 to_mont(const U256& a, const U256& R2, const U256& N, unsigned int n0inv32) {
    return mont_mul(a, R2, N, n0inv32);
}

__device__ inline U256 from_mont(const U256& a, const U256& N, unsigned int n0inv32) {
    return mont_mul(a, set_one(), N, n0inv32);
}

// X-only point doubling
__device__ PointXZ xDBL(const PointXZ& P, const U256& A24, const U256& N, unsigned int n0inv32) {
    U256 t1 = mont_add(P.X, P.Z, N);
    U256 t2 = mont_sub(P.X, P.Z, N);
    U256 t3 = mont_sqr(t1, N, n0inv32);
    U256 t4 = mont_sqr(t2, N, n0inv32);
    U256 t5 = mont_sub(t3, t4, N);
    U256 t6 = mont_mul(A24, t5, N, n0inv32);
    U256 Z_mult = mont_add(t3, t6, N);
    U256 X2 = mont_mul(t3, t4, N, n0inv32);
    U256 Z2 = mont_mul(t5, Z_mult, N, n0inv32);
    PointXZ result;
    result.X = X2;
    result.Z = Z2;
    return result;
}

// X-only point addition
__device__ PointXZ xADD(const PointXZ& P, const PointXZ& Q, const PointXZ& Diff,
                        const U256& N, unsigned int n0inv32) {
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
    PointXZ result;
    result.X = X3;
    result.Z = Z3;
    return result;
}

__device__ void cswap(PointXZ* a, PointXZ* b, unsigned int bit) {
    unsigned int mask = (0u - (bit & 1u));
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int tx = (a->X.limbs[i] ^ b->X.limbs[i]) & mask;
        a->X.limbs[i] ^= tx;
        b->X.limbs[i] ^= tx;
        unsigned int tz = (a->Z.limbs[i] ^ b->Z.limbs[i]) & mask;
        a->Z.limbs[i] ^= tz;
        b->Z.limbs[i] ^= tz;
    }
}

// Montgomery ladder
__device__ PointXZ ladder(const PointXZ& P, unsigned int k, const U256& A24,
                         const U256& N, unsigned int n0inv32, const U256& mont_one) {
    PointXZ R0;
    R0.X = mont_one;
    R0.Z = set_zero();
    PointXZ R1 = P;
    
    unsigned int prev = 0;
    bool started = false;
    
    for (int i = 31; i >= 0; i--) {
        unsigned int b = (k >> i) & 1;
        if (!started && b == 0) continue;
        started = true;
        
        cswap(&R0, &R1, b ^ prev);
        prev = b;
        
        PointXZ R0n = xADD(R0, R1, P, N, n0inv32);
        PointXZ R1n = xDBL(R1, A24, N, n0inv32);
        R0 = R0n;
        R1 = R1n;
    }
    cswap(&R0, &R1, prev);
    return R0;
}

// 64-bit LCG RNG using 32-bit operations
__device__ void lcg64_u32(unsigned int* state_lo, unsigned int* state_hi) {
    const unsigned int A_LO = 0x4C957F2Du;
    const unsigned int A_HI = 0x5851F42Du;
    const unsigned int C_LO = 0xF767814Fu;
    const unsigned int C_HI = 0x14057B7Eu;
    
    unsigned int s_lo = *state_lo;
    unsigned int s_hi = *state_hi;
    
    unsigned int res_lo, res_hi;
    mul32x32_64(s_lo, A_LO, &res_lo, &res_hi);
    
    unsigned int u_lo, u_hi;
    mul32x32_64(s_lo, A_HI, &u_lo, &u_hi);
    res_hi += u_lo;
    
    unsigned int v_lo, v_hi;
    mul32x32_64(s_hi, A_LO, &v_lo, &v_hi);
    res_hi += v_lo;
    
    unsigned long long sum = (unsigned long long)res_lo + C_LO;
    unsigned int new_lo = (unsigned int)sum;
    unsigned int carry0 = (unsigned int)(sum >> 32);
    unsigned int new_hi = res_hi + C_HI + carry0;
    
    *state_lo = new_lo;
    *state_hi = new_hi;
}

__device__ U256 next_sigma(const U256& N, unsigned int* state_lo, unsigned int* state_hi) {
    U256 acc;
    for (int i = 0; i < 4; i++) {
        lcg64_u32(state_lo, state_hi);
        acc.limbs[2*i] = *state_lo;
        acc.limbs[2*i+1] = *state_hi;
    }
    
    U256 sigma = acc;
    if (is_zero(sigma)) sigma.limbs[0] = 6;
    U256 one = set_one();
    if (cmp(sigma, one) == 0) sigma.limbs[0] = 6;
    return sigma;
}

// Binary modular inverse
__device__ InvResult mod_inverse(const U256& a_in, const U256& N) {
    InvResult result;
    result.ok = false;
    result.val = set_zero();
    
    if (is_zero(a_in)) return result;
    
    U256 a = a_in;
    for (int k = 0; k < 2 && cmp(a, N) >= 0; k++) {
        a = sub_u256(a, N);
    }
    if (is_zero(a)) return result;
    
    U256 u = a;
    U256 v = N;
    U256 x1 = set_one();
    U256 x2 = set_zero();
    
    for (int iter = 0; iter < 20000; iter++) {
        if (cmp(u, set_one()) == 0) {
            result.ok = true;
            result.val = x1;
            return result;
        }
        if (cmp(v, set_one()) == 0) {
            result.ok = true;
            result.val = x2;
            return result;
        }
        if (is_zero(u) || is_zero(v)) break;
        
        while (is_even(u)) {
            u = rshift1(u);
            if (is_even(x1)) {
                x1 = rshift1(x1);
            } else {
                x1 = rshift1(add_u256(x1, N));
            }
        }
        
        while (is_even(v)) {
            v = rshift1(v);
            if (is_even(x2)) {
                x2 = rshift1(x2);
            } else {
                x2 = rshift1(add_u256(x2, N));
            }
        }
        
        if (cmp(u, v) >= 0) {
            u = sub_u256(u, v);
            x1 = sub_mod(x1, x2, N);
        } else {
            v = sub_u256(v, u);
            x2 = sub_mod(x2, x1, N);
        }
    }
    
    return result;
}

// Suyama curve generation
__device__ CurveResult generate_curve(const U256& sigma, const U256& N, const U256& R2,
                                      unsigned int n0inv32) {
    CurveResult result;
    result.ok = false;
    result.A24m = set_zero();
    result.X1m = set_zero();
    
    U256 sigma_m = to_mont(sigma, R2, N, n0inv32);
    U256 five_m = to_mont(u256_from_u32(5), R2, N, n0inv32);
    U256 four_m = to_mont(u256_from_u32(4), R2, N, n0inv32);
    U256 three_m = to_mont(u256_from_u32(3), R2, N, n0inv32);
    U256 sixteen_m = to_mont(u256_from_u32(16), R2, N, n0inv32);
    
    U256 sigma_sq_m = mont_sqr(sigma_m, N, n0inv32);
    U256 u_m = mont_sub(sigma_sq_m, five_m, N);
    U256 v_m = mont_mul(four_m, sigma_m, N, n0inv32);
    
    U256 u_std = from_mont(u_m, N, n0inv32);
    U256 v_std = from_mont(v_m, N, n0inv32);
    InvResult inv_u = mod_inverse(u_std, N);
    InvResult inv_v = mod_inverse(v_std, N);
    
    if (!inv_u.ok || !inv_v.ok) return result;
    
    U256 u_sq_m = mont_sqr(u_m, N, n0inv32);
    U256 u_cubed_m = mont_mul(u_m, u_sq_m, N, n0inv32);
    
    U256 v_sq_m = mont_sqr(v_m, N, n0inv32);
    U256 v_cubed_m = mont_mul(v_m, v_sq_m, N, n0inv32);
    
    U256 v3_std = from_mont(v_cubed_m, N, n0inv32);
    InvResult inv_v3 = mod_inverse(v3_std, N);
    if (!inv_v3.ok) return result;
    
    U256 inv_v3_m = to_mont(inv_v3.val, R2, N, n0inv32);
    U256 X1m = mont_mul(u_cubed_m, inv_v3_m, N, n0inv32);
    
    U256 vm_u_m = mont_sub(v_m, u_m, N);
    U256 vm_u_sq_m = mont_sqr(vm_u_m, N, n0inv32);
    U256 vm_u_cubed_m = mont_mul(vm_u_m, vm_u_sq_m, N, n0inv32);
    
    U256 three_u_m = mont_mul(three_m, u_m, N, n0inv32);
    U256 three_u_plus_v_m = mont_add(three_u_m, v_m, N);
    
    U256 numerator_m = mont_mul(vm_u_cubed_m, three_u_plus_v_m, N, n0inv32);
    U256 denom_m = mont_mul(sixteen_m, u_cubed_m, N, n0inv32);
    
    U256 denom_std = from_mont(denom_m, N, n0inv32);
    InvResult inv_denom = mod_inverse(denom_std, N);
    if (!inv_denom.ok) return result;
    
    U256 inv_denom_m = to_mont(inv_denom.val, R2, N, n0inv32);
    U256 A24m = mont_mul(numerator_m, inv_denom_m, N, n0inv32);
    
    result.ok = true;
    result.A24m = A24m;
    result.X1m = X1m;
    return result;
}

// Binary GCD (optimized for odd N)
__device__ U256 gcd_binary_u256_oddN(const U256& a_in, const U256& N_odd) {
    U256 a = a_in;
    U256 b = N_odd;
    
    if (is_zero(a)) return b;
    
    while (is_even(a)) a = rshift1(a);
    
    while (true) {
        if (is_zero(b)) return a;
        while (is_even(b)) b = rshift1(b);
        if (cmp(a, b) > 0) {
            U256 t = a;
            a = b;
            b = t;
        }
        b = sub_u256(b, a);
    }
}

// --------------------------- Main Kernel ---------------------------
__global__ void ecm_stage1_kernel(unsigned int* io) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load header
    Header h;
    h.magic = io[0];
    h.version = io[1];
    h.rsv0 = io[2];
    h.rsv1 = io[3];
    h.pp_count = io[4];
    h.n_curves = io[5];
    h.seed_lo = io[6];
    h.seed_hi = io[7];
    h.base_curve = io[8];
    h.flags = io[9];
    h.rsv2 = io[10];
    h.rsv3 = io[11];
    
    if (idx >= h.n_curves) return;
    
    // Load constants
    unsigned int off = 12;
    U256 N, R2, mont_one;
    for (int i = 0; i < 8; i++) N.limbs[i] = io[off + i];
    off += 8;
    for (int i = 0; i < 8; i++) R2.limbs[i] = io[off + i];
    off += 8;
    for (int i = 0; i < 8; i++) mont_one.limbs[i] = io[off + i];
    off += 8;
    unsigned int n0inv32 = io[off];
    off += 4;
    
    unsigned int pp_off = 12 + (8*3 + 4);
    unsigned int out_base = pp_off + h.pp_count + idx * 12;
    
    // Initialize RNG
    unsigned int rng_lo = h.seed_lo ^ idx ^ h.base_curve;
    unsigned int rng_hi = h.seed_hi ^ (idx * 0x9E3779B9u);
    
    // Generate curve
    U256 A24m, X1m;
    bool curve_ok = false;
    
    for (int tries = 0; tries < 4 && !curve_ok; tries++) {
        U256 sigma = next_sigma(N, &rng_lo, &rng_hi);
        CurveResult cr = generate_curve(sigma, N, R2, n0inv32);
        if (cr.ok) {
            A24m = cr.A24m;
            X1m = cr.X1m;
            curve_ok = true;
        }
    }
    
    if (!curve_ok) {
        for (int i = 0; i < 8; i++) io[out_base + i] = 0;
        io[out_base + 8] = 3; // status: bad curve
        return;
    }
    
    // Stage 1 ladder
    PointXZ P;
    P.X = X1m;
    P.Z = mont_one;
    PointXZ R = P;
    
    for (unsigned int i = 0; i < h.pp_count; i++) {
        unsigned int pp = io[pp_off + i];
        if (pp <= 1) continue;
        R = ladder(R, pp, A24m, N, n0inv32, mont_one);
    }
    
    // Output
    U256 result = set_zero();
    unsigned int status = 1;
    
    if ((h.flags & 1) != 0) {
        U256 Zstd = from_mont(R.Z, N, n0inv32);
        U256 g = gcd_binary_u256_oddN(Zstd, N);
        result = g;
        
        U256 one = set_one();
        if (!is_zero(g) && cmp(g, N) < 0 && cmp(g, one) > 0) {
            status = 2; // factor found
        } else {
            status = 1; // no factor
        }
    } else {
        result = from_mont(R.Z, N, n0inv32);
        status = 1;
    }
    
    for (int i = 0; i < 8; i++) io[out_base + i] = result.limbs[i];
    io[out_base + 8] = status;
}

#ifdef __cplusplus
}
#endif