#!/usr/bin/env python3
"""
Generate benchmark semiprimes for Pollard p-1.
Each N = p * q where p-1 is guaranteed to be B1-smooth for a specific bound,
ensuring the algorithm succeeds (finds p) right before the bound is exhausted.
"""

import random
import argparse

try:
    import sympy
except ImportError:
    print("This script requires sympy. Install it with: pip install sympy")
    raise


def generate_benchmark_batch(count=20, B1=100000, min_bits=200, max_bits=256, seed=None):
    """
    Generate 'count' semiprimes N = p * q where:
      - p-1 is B1-smooth (so Pollard p-1 with bound B1 will find p)
      - q is a large prime so N is roughly 'max_bits' bits
      - The largest prime factor of p-1 is close to B1 (so it won't succeed too early)
    """
    if seed is not None:
        random.seed(seed)

    batch = []
    high_primes = list(sympy.primerange(B1 // 2, B1))
    low_primes = list(sympy.primerange(2, min(1000, B1 // 2)))

    attempts = 0
    while len(batch) < count and attempts < count * 1000:
        attempts += 1
        # 1. Build a B1-smooth number for p-1
        # Force one large prime near B1 so success happens late in the loop
        p_minus_1 = 1
        if high_primes:
            p_minus_1 *= random.choice(high_primes)

        # Pad with small primes until p is ~128 bits (half of target N bits)
        target_p_bits = (max_bits - 16) // 2
        while p_minus_1.bit_length() < target_p_bits:
            p_minus_1 *= random.choice(low_primes)

        p = p_minus_1 + 1
        # Ensure p is actually prime
        while not sympy.isprime(p):
            p_minus_1 *= random.choice([2, 3, 5])
            p = p_minus_1 + 1
            # Prevent infinite growth
            if p_minus_1.bit_length() > target_p_bits + 8:
                break

        if not sympy.isprime(p):
            continue

        # 2. Generate a random large prime q so that N has the desired bit length
        q_min = max(2 ** (max_bits - p.bit_length() - 1), 2 ** 127)
        q_max = 2 ** (max_bits - p.bit_length() + 1)
        try:
            q = sympy.randprime(q_min, q_max)
        except ValueError:
            continue

        N = p * q
        if N.bit_length() < min_bits or N.bit_length() > max_bits:
            continue

        # Verify p-1 is B1-smooth
        largest_factor = max(sympy.factorint(p_minus_1).keys())
        if largest_factor > B1:
            continue

        batch.append((hex(N), p, largest_factor))

    if len(batch) < count:
        print(f"WARNING: only generated {len(batch)}/{count} numbers (attempts={attempts})")

    return batch


def main():
    parser = argparse.ArgumentParser(description="Generate Pollard p-1 benchmark semiprimes")
    parser.add_argument("--count", type=int, default=20, help="Number of semiprimes to generate")
    parser.add_argument("--B1", type=int, default=100000, help="Stage-1 bound")
    parser.add_argument("--min-bits", type=int, default=200, help="Minimum bit length of N")
    parser.add_argument("--max-bits", type=int, default=256, help="Maximum bit length of N")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output file (one N per line)")
    args = parser.parse_args()

    batch = generate_benchmark_batch(args.count, args.B1, args.min_bits, args.max_bits, seed=args.seed)

    print(f"Generated {len(batch)} semiprimes for B1={args.B1}, bits={args.min_bits}-{args.max_bits}")
    print()
    ns = []
    for i, (n_hex, p, largest_factor) in enumerate(batch):
        print(f"[{i}] N={n_hex}  (factor p={p}, largest p-1 factor={largest_factor})")
        ns.append(n_hex)

    if args.output:
        with open(args.output, 'w') as f:
            for n_hex in ns:
                f.write(n_hex + '\n')
        print(f"\nWrote {len(ns)} numbers to {args.output}")

    print()
    print("CUDA command:")
    print(f"./cuda_pollard_pminus1 --batchFile=numbers.txt --B1={args.B1} --blockSize=256")
    print()
    print("WebGPU JSON inputArgs:")
    print(f'{{ "batchFile": "numbers.txt", "B1": {args.B1}, "chunkSize": 1024 }}')


if __name__ == "__main__":
    main()
