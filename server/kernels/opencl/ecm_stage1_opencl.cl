// ecm_stage1_kernel.cl - OpenCL kernel for ECM Stage 1
// All device code in this single file for JIT compilation

// --------------------------- Types & Structures ---------------------------
typedef struct {
    uint limbs[8];
} U256;

typedef struct {
    U256 X;
    U256 Z;
} PointXZ;

typedef struct {
    bool ok;
    U256 val;
} InvResult;

typedef struct {
    bool ok;
    U256 A24m;
    U256 X1m;
} CurveResult;

typedef struct {
    uint magic;      // "ECM1"
    uint version;    // 2
    uint rsv0;
    uint rsv1;
    uint pp_count;
    uint n_curves;
    uint seed_lo;
    uint seed_hi;
    uint base_curve;
    uint flags;
    uint rsv2;
    uint rsv3;
} Header;

// --------------------------- Device Functions ---------------------------
inline U256 set_zero() {
    U256 r;
    for (int i = 0; i < 8; i++) r.limbs[i] = 0;
    return r;
}

inline U256 set_one() {
    U256 r = set_zero();
    r.limbs[0] = 1;
    return r;
}

inline U256 u256_from_u32(uint x) {
    U256 r = set_zero();
    r.limbs[0] = x;
    return r;
}

inline bool is_zero(U256 a) {
    uint x = 0;
    for (int i = 0; i < 8; i++) x |= a.limbs[i];
    return x == 0;
}

inline int cmp(U256 a, U256 b) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

inline bool is_even(U256 a) {
    return (a.limbs[0] & 1) == 0;
}

inline U256 rshift1(U256 a) {
    U256 r;
    uint carry = 0;
    for (int i = 7; i >= 0; i--) {
        uint w = a.limbs[i];
        r.limbs[i] = (w >> 1) | (carry << 31);
        carry = w & 1;
    }
    return r;
}

inline uint2 addc(uint a, uint b, uint cin) {
    ulong sum = (ulong)a + b + cin;
    return (uint2)((uint)sum, (uint)(sum >> 32));
}

inline uint2 subb(uint a, uint b, uint bin) {
    ulong diff = (ulong)a - b - bin;
    return (uint2)((uint)diff, (diff >> 32) & 1);
}

inline U256 add_u256(U256 a, U256 b) {
    U256 r;
    uint c = 0;
    for (int i = 0; i < 8; i++) {
        uint2 ac = addc(a.limbs[i], b.limbs[i], c);
        r.limbs[i] = ac.x;
        c = ac.y;
    }
    return r;
}

inline U256 sub_u256(U256 a, U256 b) {
    U256 r;
    uint br = 0;
    for (int i = 0; i < 8; i++) {
        uint2 sb = subb(a.limbs[i], b.limbs[i], br);
        r.limbs[i] = sb.x;
        br = sb.y;
    }
    return r;
}

inline U256 cond_sub_N(U256 a, U256 N) {
    if (cmp(a, N) >= 0) return sub_u256(a, N);
    return a;
}

inline U256 add_mod(U256 a, U256 b, U256 N) {
    return cond_sub_N(add_u256(a, b), N);
}

inline U256 sub_mod(U256 a, U256 b, U256 N) {
    if (cmp(a, b) >= 0) return sub_u256(a, b);
    U256 diff = sub_u256(b, a);
    return sub_u256(N, diff);
}

inline uint2 mul32x32_64(uint a, uint b) {
    ulong prod = (ulong)a * b;
    return (uint2)((uint)prod, (uint)(prod >> 32));
}

// Montgomery multiplication (CIOS method)
U256 mont_mul(U256 a, U256 b, U256 N, uint n0inv32) {
    uint t[9];
    for (int i = 0; i < 9; i++) t[i] = 0;
    
    for (int i = 0; i < 8; i++) {
        uint carry = 0;
        for (int j = 0; j < 8; j++) {
            uint2 prod = mul32x32_64(a.limbs[i], b.limbs[j]);
            uint2 s1 = addc(t[j], prod.x, 0);
            uint2 s2 = addc(s1.x, carry, 0);
            t[j] = s2.x;
            carry = prod.y + s1.y + s2.y;
        }
        t[8] += carry;
        
        uint m = t[0] * n0inv32;
        
        carry = 0;
        for (int j = 0; j < 8; j++) {
            uint2 prod = mul32x32_64(m, N.limbs[j]);
            uint2 s1 = addc(t[j], prod.x, 0);
            uint2 s2 = addc(s1.x, carry, 0);
            t[j] = s2.x;
            carry = prod.y + s1.y + s2.y;
        }
        t[8] += carry;
        
        for (int k = 0; k < 8; k++) t[k] = t[k+1];
        t[8] = 0;
    }
    
    U256 r;
    for (int i = 0; i < 8; i++) r.limbs[i] = t[i];
    return cond_sub_N(r, N);
}

inline U256 mont_add(U256 a, U256 b, U256 N) {
    return cond_sub_N(add_u256(a, b), N);
}

inline U256 mont_sub(U256 a, U256 b, U256 N) {
    if (cmp(a, b) >= 0) return sub_u256(a, b);
    U256 diff = sub_u256(b, a);
    return sub_u256(N, diff);
}

inline U256 mont_sqr(U256 a, U256 N, uint n0inv32) {
    return mont_mul(a, a, N, n0inv32);
}

inline U256 to_mont(U256 a, U256 R2, U256 N, uint n0inv32) {
    return mont_mul(a, R2, N, n0inv32);
}

inline U256 from_mont(U256 a, U256 N, uint n0inv32) {
    return mont_mul(a, set_one(), N, n0inv32);
}

// X-only point doubling
PointXZ xDBL(PointXZ P, U256 A24, U256 N, uint n0inv32) {
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
PointXZ xADD(PointXZ P, PointXZ Q, PointXZ Diff, U256 N, uint n0inv32) {
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

void cswap(PointXZ* a, PointXZ* b, uint bit) {
    uint mask = (0u - (bit & 1u));
    for (int i = 0; i < 8; i++) {
        uint tx = (a->X.limbs[i] ^ b->X.limbs[i]) & mask;
        a->X.limbs[i] ^= tx;
        b->X.limbs[i] ^= tx;
        uint tz = (a->Z.limbs[i] ^ b->Z.limbs[i]) & mask;
        a->Z.limbs[i] ^= tz;
        b->Z.limbs[i] ^= tz;
    }
}

// Montgomery ladder
PointXZ ladder(PointXZ P, uint k, U256 A24, U256 N, uint n0inv32, U256 mont_one) {
    PointXZ R0;
    R0.X = mont_one;
    R0.Z = set_zero();
    PointXZ R1 = P;
    
    uint prev = 0;
    bool started = false;
    
    for (int i = 31; i >= 0; i--) {
        uint b = (k >> i) & 1;
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
uint2 lcg64_u32(uint2 state) {
    const uint A_LO = 0x4C957F2Du;
    const uint A_HI = 0x5851F42Du;
    const uint C_LO = 0xF767814Fu;
    const uint C_HI = 0x14057B7Eu;
    
    uint s_lo = state.x;
    uint s_hi = state.y;
    
    uint2 res = mul32x32_64(s_lo, A_LO);
    uint res_lo = res.x;
    uint res_hi = res.y;
    
    uint2 u = mul32x32_64(s_lo, A_HI);
    res_hi += u.x;
    
    uint2 v = mul32x32_64(s_hi, A_LO);
    res_hi += v.x;
    
    ulong sum = (ulong)res_lo + C_LO;
    uint new_lo = (uint)sum;
    uint carry0 = (uint)(sum >> 32);
    uint new_hi = res_hi + C_HI + carry0;
    
    return (uint2)(new_lo, new_hi);
}

U256 next_sigma(U256 N, uint2* state) {
    U256 acc;
    for (int i = 0; i < 4; i++) {
        *state = lcg64_u32(*state);
        acc.limbs[2*i] = state->x;
        acc.limbs[2*i+1] = state->y;
    }
    
    U256 sigma = acc;
    if (is_zero(sigma)) sigma.limbs[0] = 6;
    U256 one = set_one();
    if (cmp(sigma, one) == 0) sigma.limbs[0] = 6;
    return sigma;
}

// Binary modular inverse
InvResult mod_inverse(U256 a_in, U256 N) {
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
CurveResult generate_curve(U256 sigma, U256 N, U256 R2, uint n0inv32) {
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
U256 gcd_binary_u256_oddN(U256 a_in, U256 N_odd) {
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
__kernel void ecm_stage1_kernel(__global uint* io) {
    uint idx = get_global_id(0);
    
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
    uint off = 12;
    U256 N, R2, mont_one;
    for (int i = 0; i < 8; i++) N.limbs[i] = io[off + i];
    off += 8;
    for (int i = 0; i < 8; i++) R2.limbs[i] = io[off + i];
    off += 8;
    for (int i = 0; i < 8; i++) mont_one.limbs[i] = io[off + i];
    off += 8;
    uint n0inv32 = io[off];
    off += 4;
    
    uint pp_off = 12 + (8*3 + 4);
    uint out_base = pp_off + h.pp_count + idx * 12;
    
    // Initialize RNG
    uint2 rng;
    rng.x = h.seed_lo ^ idx ^ h.base_curve;
    rng.y = h.seed_hi ^ (idx * 0x9E3779B9u);
    
    // Generate curve
    U256 A24m, X1m;
    bool curve_ok = false;
    
    for (int tries = 0; tries < 4 && !curve_ok; tries++) {
        U256 sigma = next_sigma(N, &rng);
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
    
    for (uint i = 0; i < h.pp_count; i++) {
        uint pp = io[pp_off + i];
        if (pp <= 1) continue;
        R = ladder(R, pp, A24m, N, n0inv32, mont_one);
    }
    
    // Output
    U256 result = set_zero();
    uint status = 1;
    
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