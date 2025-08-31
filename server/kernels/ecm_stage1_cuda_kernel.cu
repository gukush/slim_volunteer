// ecm_stage1_cuda.cu
//
// ECM Stage 1 (Montgomery curves, x-only arithmetic) on CUDA
// - 256-bit big integers using 4x64-bit limbs (little-endian limbs)
// - Montgomery multiplication (CIOS) with precomputed n0inv and R^2 mod N
// - Each thread processes exactly one curve (one work-unit): computes Q = [k]P with k = ∏ p^e (p^e ≤ B1)
//   then returns either g = gcd(Z(Q), N) or Z(Q) depending on "compute_gcd" flag.
//
// This file is self-contained (header-only style). You can #include it or compile standalone by providing a small host harness.
// The host must prepare per-curve A24 and X1 in normal (non-Montgomery) representation.
// It must also precompute Montgomery constants for the modulus N: n0inv = -N^{-1} mod 2^64, and R2 = (2^(64*4))^2 mod N.
// For performance, broadcast N, R2, mont_one, and prime power list via __constant__ memory where appropriate.
//
// Author: ChatGPT (GPT-5 Thinking)
// License: MIT
//

#include <cuda.h>
#include <stdint.h>

#ifndef ECM_INLINE
#define ECM_INLINE __forceinline__ __device__
#endif

// ---- Configuration ----
#define LIMBS 4 // 4 x 64-bit limbs = 256 bits

// ---- Types ----
struct u256 {
    uint64_t v[LIMBS]; // little-endian limbs: v[0] least significant
};

struct EcmConsts {
    u256 N;       // modulus
    u256 R2;      // R^2 mod N
    u256 mont_one;// R mod N
    uint64_t n0inv; // -N^{-1} mod 2^64
    int compute_gcd; // 1 => compute gcd(Z, N) on device; 0 => return Z only
};

// Per-curve inputs
struct CurveIn {
    u256 A24;  // (A+2)/4 mod N (normal domain; kernel will convert to Montgomery)
    u256 X1;   // x-coordinate of base point P (normal domain; kernel will convert to Montgomery)
};

// Per-curve outputs
struct CurveOut {
    u256 result; // if compute_gcd==1 => gcd(Z, N); else => Z of final point (Montgomery domain)
    uint8_t status; // 0 = OK; 1 = trivial gcd 1 or N (no factor); 2 = non-trivial factor found; 3 = invalid input
};

// ---- Helpers: basic limb ops ----

ECM_INLINE bool is_zero(const u256 &a) {
    return (a.v[0]|a.v[1]|a.v[2]|a.v[3]) == 0ull;
}

ECM_INLINE int cmp_u256(const u256 &a, const u256 &b) {
    for (int i=LIMBS-1;i>=0;--i) {
        if (a.v[i] < b.v[i]) return -1;
        if (a.v[i] > b.v[i]) return 1;
    }
    return 0;
}

ECM_INLINE void set_zero(u256 &r) {
    #pragma unroll
    for (int i=0;i<LIMBS;++i) r.v[i]=0ull;
}

ECM_INLINE void set_one(u256 &r) {
    r.v[0]=1ull; r.v[1]=r.v[2]=r.v[3]=0ull;
}

ECM_INLINE void copy_u256(u256 &r, const u256 &a) {
    #pragma unroll
    for (int i=0;i<LIMBS;++i) r.v[i]=a.v[i];
}

ECM_INLINE uint64_t add_cc(uint64_t a, uint64_t b, uint64_t &carry) {
    uint64_t s = a + b;
    uint64_t c1 = (s < a);
    s += carry;
    uint64_t c2 = (s < carry);
    carry = c1 | c2;
    return s;
}

ECM_INLINE uint64_t sub_bb(uint64_t a, uint64_t b, uint64_t &borrow) {
    uint64_t d = a - b;
    uint64_t b1 = (a < b);
    uint64_t d2 = d - borrow;
    uint64_t b2 = (d < borrow);
    borrow = b1 | b2;
    return d2;
}

ECM_INLINE void add_u256(u256 &r, const u256 &a, const u256 &b) {
    uint64_t c=0;
    #pragma unroll
    for (int i=0;i<LIMBS;++i) r.v[i] = add_cc(a.v[i], b.v[i], c);
}

ECM_INLINE void sub_u256(u256 &r, const u256 &a, const u256 &b) {
    uint64_t brr=0;
    #pragma unroll
    for (int i=0;i<LIMBS;++i) r.v[i] = sub_bb(a.v[i], b.v[i], brr);
}

ECM_INLINE void rshift1(u256 &r) {
    uint64_t c=0;
    for (int i=LIMBS-1;i>=0;--i) {
        uint64_t newc = r.v[i] << 63;
        r.v[i] = (r.v[i] >> 1) | c;
        c = newc;
        if (i==0) break;
    }
}

ECM_INLINE void lshift1(u256 &r) {
    uint64_t c=0;
    for (int i=0;i<LIMBS;++i) {
        uint64_t newc = r.v[i] >> 63;
        r.v[i] = (r.v[i] << 1) | c;
        c = newc;
    }
}

// ---- Montgomery arithmetic (CIOS) ----
// All inputs/outputs are reduced modulo N and in Montgomery domain unless stated otherwise.

ECM_INLINE void cond_sub(u256 &a, const u256 &N) {
    // if a >= N => a -= N
    uint64_t borrow=0;
    u256 t;
    #pragma unroll
    for (int i=0;i<LIMBS;++i) t.v[i] = sub_bb(a.v[i], N.v[i], borrow);
    // if no borrow, t is result; else keep a
    uint64_t mask = (uint64_t)0 - (uint64_t)(1 - borrow); // 0x..FF if borrow==0 else 0
    #pragma unroll
    for (int i=0;i<LIMBS;++i) a.v[i] = (t.v[i] & mask) | (a.v[i] & ~mask);
}

ECM_INLINE void mont_mul(u256 &r, const u256 &a, const u256 &b, const EcmConsts &C) {
    // t has LIMBS+1 limbs
    uint64_t t[LIMBS+1];
    #pragma unroll
    for (int i=0;i<=LIMBS;++i) t[i]=0ull;

    #pragma unroll
    for (int i=0;i<LIMBS;++i) {
        // t += a_i * b
        uint64_t carry=0;
        #pragma unroll
        for (int j=0;j<LIMBS;++j) {
            unsigned long long lo = a.v[i] * b.v[j];
            unsigned long long hi = __umul64hi(a.v[i], b.v[j]);
            unsigned long long tmp = t[j] + lo;
            unsigned long long c1 = (tmp < t[j]);
            tmp += carry;
            unsigned long long c2 = (tmp < carry);
            t[j] = tmp;
            unsigned long long sum_hi = hi + c1 + c2;
            carry = sum_hi;
        }
        t[LIMBS] += carry;

        // m = t0 * n0inv mod 2^64
        uint64_t m = t[0] * C.n0inv;

        // t += m * N
        carry=0;
        #pragma unroll
        for (int j=0;j<LIMBS;++j) {
            unsigned long long lo = m * C.N.v[j];
            unsigned long long hi = __umul64hi(m, C.N.v[j]);
            unsigned long long tmp = t[j] + lo;
            unsigned long long c1 = (tmp < t[j]);
            tmp += carry;
            unsigned long long c2 = (tmp < carry);
            t[j] = tmp;
            unsigned long long sum_hi = hi + c1 + c2;
            carry = sum_hi;
        }
        unsigned long long tmp2 = t[LIMBS] + carry;
        t[LIMBS] = tmp2;

        // shift t right by one limb (divide by 2^64)
        #pragma unroll
        for (int k=0;k<LIMBS;++k) t[k] = t[k+1];
        t[LIMBS]=0ull;
    }

    #pragma unroll
    for (int i=0;i<LIMBS;++i) r.v[i] = t[i];

    cond_sub(r, C.N);
}

ECM_INLINE void to_mont(u256 &r, const u256 &a, const EcmConsts &C) {
    mont_mul(r, a, C.R2, C);
}
ECM_INLINE void from_mont(u256 &r, const u256 &a, const EcmConsts &C) {
    u256 one = C.mont_one; // actually already R mod N
    // Convert by multiplying by 1 (Montgomery form of 1 is mont_one). To get standard form, multiply by 1 (which equals 1*R^{-1} mod N).
    // Here, 'a' is in Montgomery domain; mont_mul(a, 1) returns a*1*R^{-1} = a (standard).
    mont_mul(r, a, one, C);
}

ECM_INLINE void mont_add(u256 &r, const u256 &a, const u256 &b, const EcmConsts &C) {
    add_u256(r, a, b);
    cond_sub(r, C.N);
}
ECM_INLINE void mont_sub(u256 &r, const u256 &a, const u256 &b, const EcmConsts &C) {
    u256 t;
    sub_u256(t, a, b);
    // if borrow happened, add N back
    // Detect borrow by recomputing a < b
    if (cmp_u256(a,b) < 0) {
        add_u256(r, t, C.N);
    } else {
        copy_u256(r, t);
    }
}
ECM_INLINE void mont_sqr(u256 &r, const u256 &a, const EcmConsts &C) { mont_mul(r,a,a,C); }

// ---- Montgomery curve x-only ops ----
// Using the classic formulas (Montgomery, 1987):
// xDBL: (X2:Z2) = 2*(X1:Z1)
// xADD: (X3:Z3) = (X1:Z1) + (X2:Z2), given difference (Xdiff:Zdiff) = (X1 - X2)

struct PointXZ { u256 X; u256 Z; };

ECM_INLINE void xDBL(PointXZ &R2, const PointXZ &R, const u256 &A24, const EcmConsts &C) {
    u256 t1,t2,t3,t4;
    mont_add(t1, R.X, R.Z, C);   // t1 = X+Z
    mont_sub(t2, R.X, R.Z, C);   // t2 = X-Z
    mont_sqr(t1, t1, C);         // t1 = (X+Z)^2
    mont_sqr(t2, t2, C);         // t2 = (X-Z)^2
    mont_sub(t3, t1, t2, C);     // t3 = t1 - t2
    mont_mul(R2.X, t1, t2, C);   // X2 = t1 * t2
    mont_mul(t4, t3, A24, C);    // t4 = A24 * t3
    mont_add(t4, t4, t2, C);     // t4 = t4 + t2
    mont_mul(R2.Z, t3, t4, C);   // Z2 = t3 * t4
}

ECM_INLINE void xADD(PointXZ &R3, const PointXZ &P, const PointXZ &Q, const PointXZ &Diff, const EcmConsts &C) {
    // R3 = P + Q, given Diff = P - Q
    u256 t1,t2,t3,t4,t5,t6;
    mont_add(t1, P.X, P.Z, C);    // t1 = Xp + Zp
    mont_sub(t2, P.X, P.Z, C);    // t2 = Xp - Zp
    mont_add(t3, Q.X, Q.Z, C);    // t3 = Xq + Zq
    mont_sub(t4, Q.X, Q.Z, C);    // t4 = Xq - Zq
    mont_mul(t5, t1, t4, C);      // t5 = t1 * t4
    mont_mul(t6, t2, t3, C);      // t6 = t2 * t3
    mont_add(t1, t5, t6, C);      // t1 = t5 + t6
    mont_sub(t2, t5, t6, C);      // t2 = t5 - t6
    mont_sqr(t1, t1, C);          // t1 = (t5 + t6)^2
    mont_sqr(t2, t2, C);          // t2 = (t5 - t6)^2
    mont_mul(R3.X, t1, Diff.Z, C);// X3 = Zdiff * t1
    mont_mul(R3.Z, t2, Diff.X, C);// Z3 = Xdiff * t2
}

// Constant-time conditional swap of two points, controlled by bit (0 or 1).
ECM_INLINE void cswap(PointXZ &A, PointXZ &B, uint64_t bit) {
    uint64_t mask = (uint64_t)0 - (bit & 1ull);
    #pragma unroll
    for (int i=0;i<LIMBS;++i) {
        uint64_t tmp = mask & (A.X.v[i] ^ B.X.v[i]); A.X.v[i] ^= tmp; B.X.v[i] ^= tmp;
    }
    #pragma unroll
    for (int i=0;i<LIMBS;++i) {
        uint64_t tmp = mask & (A.Z.v[i] ^ B.Z.v[i]); A.Z.v[i] ^= tmp; B.Z.v[i] ^= tmp;
    }
}

// Montgomery ladder: compute R0 = [k]P input as (XP:ZP) in Montgomery domain.
// This version uses constant-time cswap and the "difference = P" invariant.
ECM_INLINE void mont_ladder(PointXZ &R0, const PointXZ &Pin, uint32_t k, const u256 &A24, const EcmConsts &C) {
    PointXZ R1 = Pin;
    // R0 = (1:0)
    copy_u256(R0.X, C.mont_one);
    set_zero(R0.Z);

    // Highest bit (31..0)
    int started = 0;
    for (int i=31;i>=0;--i) {
        uint32_t bit = (k >> i) & 1u;
        if (!started && bit == 0) continue; // skip leading zeros
        started = 1;

        // Conditional swap to keep invariant R1 - R0 = P
        cswap(R0, R1, 1ull - (uint64_t)bit);

        // After swap: if bit==1 then (R0,R1) unchanged; if bit==0 then swapped
        // Now do: R0 = xADD(R0,R1,P), R1 = xDBL(R1)
        PointXZ t0, t1;
        xADD(t0, R0, R1, Pin, C);
        xDBL(t1, R1, A24, C);
        R0 = t0;
        R1 = t1;
    }
    // One more swap to undo last if loop never started
    if (!started) {
        // k==0: return infinity (1:0)
        // already set
    }
}

// Multiply by an unsigned small scalar m (<= B1) using the ladder above.
ECM_INLINE void mul_small(PointXZ &R, const PointXZ &P, uint32_t m, const u256 &A24, const EcmConsts &C) {
    mont_ladder(R, P, m, A24, C);
}

// ---- GCD (binary GCD) ----

ECM_INLINE int ctz_u64(uint64_t x) {
    if (x==0ull) return 64;
    return __ffsll((long long)x) - 1; // count trailing zeros
}

ECM_INLINE int ctz_u256(const u256 &a) {
    int tz = 0;
    for (int i=0;i<LIMBS;++i) {
        if (a.v[i]==0ull) { tz += 64; continue; }
        tz += ctz_u64(a.v[i]);
        break;
    }
    return tz;
}

ECM_INLINE void rshiftk(u256 &a, int k) {
    if (k <= 0) return;
    int limb = k / 64;
    int bits = k % 64;
    if (limb) {
        for (int i=0;i<LIMBS;i++) {
            a.v[i] = (limb + i < LIMBS) ? a.v[limb+i] : 0ull;
        }
    }
    if (bits) {
        uint64_t carry = 0;
        for (int i=LIMBS-1;i>=0;--i) {
            uint64_t newcarry = a.v[i] << (64 - bits);
            a.v[i] = (a.v[i] >> bits) | carry;
            carry = newcarry;
            if (i==0) break;
        }
    }
}

ECM_INLINE void lshiftk(u256 &a, int k) {
    if (k <= 0) return;
    int limb = k / 64;
    int bits = k % 64;
    if (limb) {
        for (int i=LIMBS-1;i>=0;--i) {
            a.v[i] = (i >= limb) ? a.v[i-limb] : 0ull;
        }
    }
    if (bits) {
        uint64_t carry = 0;
        for (int i=0;i<LIMBS;++i) {
            uint64_t newcarry = a.v[i] >> (64 - bits);
            a.v[i] = (a.v[i] << bits) | carry;
            carry = newcarry;
        }
    }
}

ECM_INLINE u256 gcd_u256(u256 a, u256 b) {
    if (is_zero(a)) return b;
    if (is_zero(b)) return a;

    int shift = min(ctz_u256(a), ctz_u256(b));
    rshiftk(a, ctz_u256(a));
    do {
        rshiftk(b, ctz_u256(b));
        if (cmp_u256(a,b) > 0) { u256 t=a; a=b; b=t; }
        // b = b - a
        u256 t;
        sub_u256(t, b, a);
        b = t;
    } while (!is_zero(b));
    lshiftk(a, shift);
    return a;
}

// ---- Kernel ----
// Grid: (num_curves + blockDim.x - 1) / blockDim.x blocks, blockDim multiple of warp size recommended.

extern "C" __global__
void ecm_stage1_kernel(
    const EcmConsts C,
    const CurveIn * __restrict__ curves, // length = num_curves
    const uint32_t * __restrict__ prime_powers, // length = pp_count
    int pp_count,
    CurveOut * __restrict__ out, // length = num_curves
    int num_curves
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_curves) return;

    CurveOut res;
    res.status = 0;
    set_zero(res.result);

    CurveIn in = curves[tid];

    // Convert inputs to Montgomery
    u256 A24m, X1m;
    to_mont(A24m, in.A24, C);
    to_mont(X1m, in.X1, C);

    // Base point P = (X1m : 1)
    PointXZ P; P.X = X1m; P.Z = C.mont_one;

    // R accumulates the successive multiplications: start at P
    PointXZ R = P;
    PointXZ T;

    // For each prime power s, compute R = [s]R
    for (int i=0;i<pp_count;++i) {
        uint32_t s = prime_powers[i];
        // Skip 1
        if (s <= 1u) continue;
        mul_small(T, R, s, A24m, C); // T = [s]R
        R = T;
    }

    // After Stage 1, compute g = gcd(Z(R), N), or return Z(R)
    if (C.compute_gcd) {
        u256 Zstd;
        from_mont(Zstd, R.Z, C); // convert Z out of Montgomery before gcd
        u256 g = gcd_u256(Zstd, C.N);
        res.result = g;
        // status: 2 if 1<g<N, else 1 (no factor) or 3 if invalid
        if (!is_zero(g) && cmp_u256(g, C.N) < 0) {
            // check if g != 1
            u256 one; set_one(one);
            if (cmp_u256(g, one) > 0) res.status = 2; else res.status = 1;
        } else {
            res.status = 1;
        }
    } else {
        // Return Z in standard (non-Montgomery) domain to keep host simple
        u256 Zstd; from_mont(Zstd, R.Z, C);
        res.result = Zstd;
        res.status = 0;
    }

    out[tid] = res;
}

// ---- Notes ----
// * Host responsibilities:
//   - Precompute n0inv = -N^{-1} mod 2^64 and R2 = (2^(64*4))^2 mod N, mont_one = (2^(64*4)) mod N.
//   - Build prime_powers as the list { p^e : p prime, e = floor(log_p(B1)), p^e ≤ B1 }.
//   - Generate curves: choose random sigma (server-side), then compute (A24, X1) via Suyama parameterization (mod N).
//     If any modular inverse fails during curve generation, you've found a gcd; report and skip sending that curve to GPU.
// * Each thread is independent; choose blocks/threads to saturate your GPU.
// * You can pack EcmConsts into __constant__ memory for speed if N is fixed per job.
// * Safety: this code is constant-time-ish for add/mul; the ladder uses cswap to mitigate divergence.
// * Limitations: only 256-bit moduli (4x64). Extend by increasing LIMBS and adding 64x64 to 128 helpers as needed.
