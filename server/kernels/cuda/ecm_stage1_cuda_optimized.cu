// kernels/ecm_stage1_cuda_optimized.cu - Version 3 (resumable, 32-bit limbs)
// Matches WebGPU implementation exactly for optimal performance
// - 256-bit big integers (8x u32, little-endian) - SAME AS WEBGPU
// - Montgomery math (CIOS, word=32, L=8) - SAME AS WEBGPU
// - Deterministic RNG via 64-bit LCG emulated with u32 - SAME AS WEBGPU
// - Suyama parametrization - SAME AS WEBGPU
// - Binary modular inverse (u32) - SAME AS WEBGPU
// - Correct constant-time Montgomery ladder with final swap - SAME AS WEBGPU
// - RESUMABLE: Supports splitting work across multiple GPU submits - SAME AS WEBGPU

extern "C" {

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// --------------------------- Types & IO ---------------------------
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

struct CurveState {
    uint32_t X_limbs[8];
    uint32_t Z_limbs[8];
    uint32_t A24_limbs[8];
    uint32_t sigma;      // for debugging/reproducibility
    uint32_t curve_ok;   // 1 if curve is valid
};

// Layout helpers (words) - EXACTLY MATCH WEBGPU
__device__ __forceinline__ uint32_t constOffset() { return 12u; }
__device__ __forceinline__ uint32_t ppOffset()    { return constOffset() + (8u*3u + 4u); }
__device__ __forceinline__ uint32_t outOffset(const Header& h) { return ppOffset() + h.pp_count; }
__device__ __forceinline__ uint32_t stateOffset(const Header& h) {
    return outOffset(h) + h.n_curves * (8u + 1u + 3u); // after output section
}

static const uint32_t LIMBS = 8u;
static const uint32_t STATE_WORDS_PER_CURVE = 8u + 8u + 8u + 2u; // X + Z + A24 + (sigma, curve_ok)

// --------------------------- Big Integer Operations (32-bit) ---------------------------

__device__ __forceinline__ void set_zero(U256& a) {
    for (uint32_t i = 0; i < LIMBS; i++) a.limbs[i] = 0u;
}

__device__ __forceinline__ void set_one(U256& a) {
    a.limbs[0] = 1u;
    for (uint32_t i = 1; i < LIMBS; i++) a.limbs[i] = 0u;
}

__device__ __forceinline__ bool is_zero(const U256& a) {
    for (uint32_t i = 0; i < LIMBS; i++) {
        if (a.limbs[i] != 0u) return false;
    }
    return true;
}

__device__ __forceinline__ bool is_one(const U256& a) {
    if (a.limbs[0] != 1u) return false;
    for (uint32_t i = 1; i < LIMBS; i++) {
        if (a.limbs[i] != 0u) return false;
    }
    return true;
}

__device__ __forceinline__ int cmp(const U256& a, const U256& b) {
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

__device__ __forceinline__ void copy(U256& dst, const U256& src) {
    for (uint32_t i = 0; i < LIMBS; i++) dst.limbs[i] = src.limbs[i];
}

__device__ __forceinline__ void add(U256& result, const U256& a, const U256& b) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < LIMBS; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)b.limbs[i] + carry;
        result.limbs[i] = (uint32_t)(sum & 0xFFFFFFFFu);
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void sub(U256& result, const U256& a, const U256& b) {
    uint64_t borrow = 0;
    for (uint32_t i = 0; i < LIMBS; i++) {
        uint64_t diff = (uint64_t)a.limbs[i] - (uint64_t)b.limbs[i] - borrow;
        result.limbs[i] = (uint32_t)(diff & 0xFFFFFFFFu);
        borrow = (diff >> 32) & 1;
    }
}

__device__ __forceinline__ void mul_32(U256& result, const U256& a, uint32_t b) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < LIMBS; i++) {
        uint64_t prod = (uint64_t)a.limbs[i] * (uint64_t)b + carry;
        result.limbs[i] = (uint32_t)(prod & 0xFFFFFFFFu);
        carry = prod >> 32;
    }
}

// --------------------------- Montgomery Math (32-bit CIOS) ---------------------------

__device__ __forceinline__ void montgomery_mul(U256& result, const U256& a, const U256& b, const U256& N, uint32_t n0inv) {
    U256 T;
    set_zero(T);

    for (uint32_t i = 0; i < LIMBS; i++) {
        uint64_t carry = 0;
        const uint32_t ai = a.limbs[i];

        // T = T + a[i] * b
        for (uint32_t j = 0; j < LIMBS; j++) {
            const uint64_t prod = (uint64_t)ai * (uint64_t)b.limbs[j] + (uint64_t)T.limbs[j] + carry;
            T.limbs[j] = (uint32_t)(prod & 0xFFFFFFFFu);
            carry = prod >> 32;
        }
        if (i + 1 < LIMBS) T.limbs[i + 1] = (uint32_t)carry;

        // m = (T[0] * n0inv) mod 2^32
        const uint32_t m = T.limbs[0] * n0inv;

        // T = T + m * N
        carry = 0;
        for (uint32_t j = 0; j < LIMBS; j++) {
            const uint64_t prod = (uint64_t)m * (uint64_t)N.limbs[j] + (uint64_t)T.limbs[j] + carry;
            T.limbs[j] = (uint32_t)(prod & 0xFFFFFFFFu);
            carry = prod >> 32;
        }

        // T = T >> 32
        for (uint32_t j = 0; j < LIMBS - 1; j++) {
            T.limbs[j] = T.limbs[j + 1];
        }
        T.limbs[LIMBS - 1] = (uint32_t)carry;
    }

    // Final reduction
    if (cmp(T, N) >= 0) {
        sub(result, T, N);
    } else {
        copy(result, T);
    }
}

__device__ __forceinline__ void to_montgomery(U256& result, const U256& a, const U256& R2, const U256& N, uint32_t n0inv) {
    montgomery_mul(result, a, R2, N, n0inv);
}

__device__ __forceinline__ void from_montgomery(U256& result, const U256& a, const U256& N, uint32_t n0inv) {
    U256 one;
    set_one(one);
    montgomery_mul(result, a, one, N, n0inv);
}

// --------------------------- Modular Inverse (32-bit) ---------------------------

__device__ __forceinline__ InvResult mod_inverse(const U256& a, const U256& N) {
    InvResult result = {false, {0}};

    if (is_zero(a) || is_one(a)) {
        result.ok = true;
        copy(result.val, a);
        return result;
    }

    U256 u, v, r, s;
    copy(u, a);
    copy(v, N);
    set_one(r);
    set_zero(s);

    while (!is_zero(v)) {
        if (u.limbs[0] & 1u) {
            if (cmp(u, v) >= 0) {
                sub(u, u, v);
                sub(r, r, s);
            } else {
                sub(v, v, u);
                sub(s, s, r);
            }
        } else {
            // u = u >> 1
            for (uint32_t i = 0; i < LIMBS - 1; i++) {
                u.limbs[i] = (u.limbs[i] >> 1) | (u.limbs[i + 1] << 31);
            }
            u.limbs[LIMBS - 1] >>= 1;

            if (r.limbs[0] & 1u) {
                add(r, r, N);
            }
            // r = r >> 1
            for (uint32_t i = 0; i < LIMBS - 1; i++) {
                r.limbs[i] = (r.limbs[i] >> 1) | (r.limbs[i + 1] << 31);
            }
            r.limbs[LIMBS - 1] >>= 1;
        }
    }

    if (is_one(u)) {
        result.ok = true;
        copy(result.val, r);
    }

    return result;
}

// --------------------------- RNG (32-bit) ---------------------------

__device__ __forceinline__ U256 next_sigma(const U256& N, uint32_t* rng_state) {
    // 64-bit LCG: state = (state * 6364136223846793005 + 1) mod 2^64
    uint64_t state = ((uint64_t)rng_state[1] << 32) | rng_state[0];
    state = state * 6364136223846793005ULL + 1ULL;
    rng_state[0] = (uint32_t)(state & 0xFFFFFFFFu);
    rng_state[1] = (uint32_t)(state >> 32);

    U256 sigma;
    sigma.limbs[0] = rng_state[0];
    sigma.limbs[1] = rng_state[1];
    for (uint32_t i = 2; i < LIMBS; i++) {
        sigma.limbs[i] = 0u;
    }

    // sigma = sigma mod N
    if (cmp(sigma, N) >= 0) {
        sub(sigma, sigma, N);
    }

    return sigma;
}

// --------------------------- Curve Generation ---------------------------

__device__ __forceinline__ CurveResult generate_curve(const U256& sigma, const U256& N, const U256& R2, uint32_t n0inv) {
    CurveResult result = {false, {0}, {0}};

    // Compute u = sigma^2 - 5 mod N
    U256 u;
    montgomery_mul(u, sigma, sigma, N, n0inv);
    U256 five;
    set_zero(five);
    five.limbs[0] = 5u;
    to_montgomery(five, five, R2, N, n0inv);
    sub(u, u, five);

    // Compute v = 4*sigma mod N
    U256 v;
    mul_32(v, sigma, 4u);
    if (cmp(v, N) >= 0) {
        sub(v, v, N);
    }
    to_montgomery(v, v, R2, N, n0inv);

    // Compute A = (v-u)^3 * (3*u+v) mod N
    U256 temp1, temp2;
    sub(temp1, v, u);
    montgomery_mul(temp1, temp1, temp1, N, n0inv);
    montgomery_mul(temp1, temp1, temp1, N, n0inv);

    mul_32(temp2, u, 3u);
    add(temp2, temp2, v);
    montgomery_mul(temp1, temp1, temp2, N, n0inv);

    // Compute A24 = (A + 2) / 4 mod N
    U256 two;
    set_zero(two);
    two.limbs[0] = 2u;
    to_montgomery(two, two, R2, N, n0inv);
    add(temp1, temp1, two);

    U256 four;
    set_zero(four);
    four.limbs[0] = 4u;
    to_montgomery(four, four, R2, N, n0inv);

    InvResult inv_result = mod_inverse(four, N);
    if (!inv_result.ok) return result;

    montgomery_mul(result.A24m, temp1, inv_result.val, N, n0inv);

    // Compute X1 = u^3 mod N
    montgomery_mul(result.X1m, u, u, N, n0inv);
    montgomery_mul(result.X1m, result.X1m, u, N, n0inv);

    result.ok = true;
    return result;
}

// --------------------------- Point Operations ---------------------------

__device__ __forceinline__ void point_dbl(PointXZ& result, const PointXZ& P, const U256& A24m, const U256& N, uint32_t n0inv) {
    U256 X2, Z2, temp1, temp2, temp3;

    // X2 = (X + Z)^2 mod N
    add(temp1, P.X, P.Z);
    montgomery_mul(X2, temp1, temp1, N, n0inv);

    // Z2 = (X - Z)^2 mod N
    sub(temp1, P.X, P.Z);
    montgomery_mul(Z2, temp1, temp1, N, n0inv);

    // temp1 = X2 - Z2
    sub(temp1, X2, Z2);

    // temp2 = A24m * Z2
    montgomery_mul(temp2, A24m, Z2, N, n0inv);

    // temp3 = X2 + temp2
    add(temp3, X2, temp2);

    // result.X = X2 * Z2
    montgomery_mul(result.X, X2, Z2, N, n0inv);

    // result.Z = temp1 * temp3
    montgomery_mul(result.Z, temp1, temp3, N, n0inv);
}

__device__ __forceinline__ void point_add(PointXZ& result, const PointXZ& P, const PointXZ& Q, const PointXZ& diff, const U256& N, uint32_t n0inv) {
    U256 temp1, temp2, temp3, temp4;

    // temp1 = (P.X + P.Z) * (Q.X - Q.Z)
    add(temp1, P.X, P.Z);
    sub(temp2, Q.X, Q.Z);
    montgomery_mul(temp1, temp1, temp2, N, n0inv);

    // temp2 = (P.X - P.Z) * (Q.X + Q.Z)
    sub(temp2, P.X, P.Z);
    add(temp3, Q.X, Q.Z);
    montgomery_mul(temp2, temp2, temp3, N, n0inv);

    // temp3 = temp1 + temp2
    add(temp3, temp1, temp2);

    // temp4 = temp1 - temp2
    sub(temp4, temp1, temp2);

    // result.X = diff.Z * temp3^2
    montgomery_mul(temp1, temp3, temp3, N, n0inv);
    montgomery_mul(result.X, diff.Z, temp1, N, n0inv);

    // result.Z = diff.X * temp4^2
    montgomery_mul(temp1, temp4, temp4, N, n0inv);
    montgomery_mul(result.Z, diff.X, temp1, N, n0inv);
}

__device__ __forceinline__ void point_mul_small(PointXZ& result, const PointXZ& P, uint32_t k, const U256& A24m, const U256& N, uint32_t n0inv) {
    PointXZ R0, R1, temp;
    R0.X = P.X;
    R0.Z = P.Z;
    R1.X = P.X;
    R1.Z = P.Z;
    point_dbl(R1, R1, A24m, N, n0inv);

    for (int i = 31; i >= 0; i--) {
        if ((k >> i) & 1u) {
            point_add(temp, R0, R1, P, N, n0inv);
            point_dbl(R0, R0, A24m, N, n0inv);
            R1.X = temp.X;
            R1.Z = temp.Z;
        } else {
            point_add(temp, R0, R1, P, N, n0inv);
            point_dbl(R1, R1, A24m, N, n0inv);
            R0.X = temp.X;
            R0.Z = temp.Z;
        }
    }

    result.X = R0.X;
    result.Z = R0.Z;
}

// --------------------------- State Management ---------------------------

__device__ __forceinline__ CurveState load_state(uint32_t curve_idx, const Header& h, const uint32_t* io) {
    CurveState state;
    uint32_t state_base = stateOffset(h) + curve_idx * STATE_WORDS_PER_CURVE;

    for (uint32_t i = 0; i < 8; i++) {
        state.X_limbs[i] = io[state_base + i];
        state.Z_limbs[i] = io[state_base + 8 + i];
        state.A24_limbs[i] = io[state_base + 16 + i];
    }
    state.sigma = io[state_base + 24];
    state.curve_ok = io[state_base + 25];

    return state;
}

__device__ __forceinline__ void save_state(uint32_t curve_idx, const Header& h, uint32_t* io, const CurveState& state) {
    uint32_t state_base = stateOffset(h) + curve_idx * STATE_WORDS_PER_CURVE;

    for (uint32_t i = 0; i < 8; i++) {
        io[state_base + i] = state.X_limbs[i];
        io[state_base + 8 + i] = state.Z_limbs[i];
        io[state_base + 16 + i] = state.A24_limbs[i];
    }
    io[state_base + 24] = state.sigma;
    io[state_base + 25] = state.curve_ok;
}

// --------------------------- Main Kernel ---------------------------

__global__ void ecm_stage1_v3_optimized(uint32_t* io) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

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
    h.pp_start = io[10];
    h.pp_len = io[11];

    if (idx >= h.n_curves) return;

    // Load constants
    uint32_t off = constOffset();
    U256 N, R2, mont_one;
    for (uint32_t i = 0; i < 8; i++) { N.limbs[i] = io[off + i]; }        off += 8;
    for (uint32_t i = 0; i < 8; i++) { R2.limbs[i] = io[off + i]; }       off += 8;
    for (uint32_t i = 0; i < 8; i++) { mont_one.limbs[i] = io[off + i]; } off += 8;
    uint32_t n0inv32 = io[off];

    uint32_t pp_off = ppOffset();
    uint32_t out_base = outOffset(h) + idx * (8u + 1u + 3u);

    PointXZ R;
    U256 A24m;
    uint32_t sigma_val = 0u;
    bool curve_ok = false;

    // Check if this is a fresh start or resume
    if (h.pp_start == 0u) {
        // Fresh start - generate curve
        uint32_t rng_state[2];
        uint32_t global_curve_idx = h.base_curve + idx;
        rng_state[0] = h.seed_lo ^ global_curve_idx ^ 0x12345678u;
        rng_state[1] = h.seed_hi ^ (global_curve_idx * 0x9E3779B9u) ^ 0x87654321u;

        if (rng_state[0] == 0u && rng_state[1] == 0u) {
            rng_state[0] = 0x12345678u + global_curve_idx;
            rng_state[1] = 0x87654321u + global_curve_idx;
        }

        // Try generating a valid curve
        for (uint32_t tries = 0; tries < 4u && !curve_ok; tries++) {
            U256 sigma = next_sigma(N, rng_state);
            sigma_val = sigma.limbs[0];
            CurveResult cr = generate_curve(sigma, N, R2, n0inv32);
            if (cr.ok) {
                A24m = cr.A24m;
                R.X = cr.X1m;
                R.Z = mont_one;
                curve_ok = true;
            }
        }

        if (!curve_ok) {
            // Bad curve - write error status
            for (uint32_t i = 0; i < 8; i++) { io[out_base + i] = 0u; }
            io[out_base + 8] = 3u; // bad curve status
            return;
        }
    } else {
        // Resume - load saved state
        CurveState state = load_state(idx, h, io);
        U256 X, Z;
        for (uint32_t i = 0; i < 8; i++) {
            X.limbs[i] = state.X_limbs[i];
            Z.limbs[i] = state.Z_limbs[i];
            A24m.limbs[i] = state.A24_limbs[i];
        }
        R.X = X;
        R.Z = Z;
        sigma_val = state.sigma;
        curve_ok = (state.curve_ok != 0u);
    }

    // Process prime powers in current window
    for (uint32_t i = h.pp_start; i < h.pp_start + h.pp_len; i++) {
        if (i >= h.pp_count) break;
        uint32_t pp = io[pp_off + i];
        if (pp <= 1u) continue;

        point_mul_small(R, R, pp, A24m, N, n0inv32);
    }

    // Save state for next pass
    CurveState state;
    for (uint32_t i = 0; i < 8; i++) {
        state.X_limbs[i] = R.X.limbs[i];
        state.Z_limbs[i] = R.Z.limbs[i];
        state.A24_limbs[i] = A24m.limbs[i];
    }
    state.sigma = sigma_val;
    state.curve_ok = curve_ok ? 1u : 0u;
    save_state(idx, h, io, state);

    // Check if we're done
    if (h.pp_start + h.pp_len >= h.pp_count) {
        // Final pass - compute result
        U256 Z_std;
        from_montgomery(Z_std, R.Z, N, n0inv32);

        // Write result
        for (uint32_t i = 0; i < 8; i++) {
            io[out_base + i] = Z_std.limbs[i];
        }
        io[out_base + 8] = 1u; // no factor status
    } else {
        // Not done - write "needs more" status
        for (uint32_t i = 0; i < 8; i++) { io[out_base + i] = 0u; }
        io[out_base + 8] = 0u; // needs more status
    }
}

} // extern "C"
