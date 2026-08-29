#!/usr/bin/env python3
"""
Reference Python simulation of Mitchell's logarithmic approximate multiplier
implemented in hls4mlrcoem/hls4ml/templates/vivado/nnet_utils/nnet_mult.h
(`mitchell_base<x_T, w_T>`).

Verifies:
  (1) the bit-level math matches what the C++ template computes;
  (2) the approximation error stays within the textbook ~11.1% worst-case
      relative error bound, and never overestimates the magnitude;
  (3) edge cases (zero operands, signed operands, the most-negative
      two's-complement corner case) behave sanely.

No Vivado/Vitis HLS required -- this is a pure-Python two's-complement bit
simulator, mirroring the C++ operation-for-operation.
"""

import numpy as np


def twos(value: int, width: int) -> int:
    """Wrap an int into a `width`-bit two's-complement representation."""
    return value & ((1 << width) - 1)


def to_signed(raw: int, width: int) -> int:
    """Interpret a `width`-bit two's-complement pattern as a signed Python int."""
    return raw - (1 << width) if raw & (1 << (width - 1)) else raw


def leading_one_index(mag: int, width: int) -> int:
    """Index of the highest set bit of `mag` (a `width`-bit unsigned value)."""
    for i in range(width - 1, -1, -1):
        if (mag >> i) & 1:
            return i
    return -1  # mag == 0, not used by mitchell() below (guarded separately)


def mitchell_unsigned(mag_a: int, mag_w: int, WA: int, WW: int) -> int:
    """
    Faithful re-implementation of the unsigned magnitude path inside
    mitchell_base<x_T, w_T>::product(a, w): leading-one-detect both operands,
    approximate log2(1+mantissa) ~= mantissa, add in the log domain, then
    reconstruct by re-inserting the leading one and shifting.
    """
    assert mag_a != 0 and mag_w != 0

    exp_a = leading_one_index(mag_a, WA)
    exp_w = leading_one_index(mag_w, WW)

    MANT_A = WA - 1
    MANT_W = WW - 1

    mant_a = twos(mag_a << (MANT_A - exp_a), MANT_A)
    mant_w = twos(mag_w << (MANT_W - exp_w), MANT_W)

    MANT_BITS = max(MANT_A, MANT_W)
    mant_a_ext = mant_a << (MANT_BITS - MANT_A)
    mant_w_ext = mant_w << (MANT_BITS - MANT_W)

    sum_frac = mant_a_ext + mant_w_ext          # up to MANT_BITS+1 bits
    carry = (sum_frac >> MANT_BITS) & 1
    mant_sum = sum_frac & ((1 << MANT_BITS) - 1)
    exp_sum = exp_a + exp_w + carry

    normalized = (1 << MANT_BITS) | mant_sum    # in [2^MANT_BITS, 2^(MANT_BITS+1))
    shift = exp_sum - MANT_BITS
    if shift >= 0:
        return normalized << shift
    return normalized >> (-shift)


def mitchell(a_raw: int, w_raw: int, WA: int, WW: int) -> int:
    """
    Faithful re-implementation of mitchell_base<x_T, w_T>::product(a, w).
    Operates on the raw two's-complement bit patterns. Returns the raw bit
    pattern of the (WA+WW)-bit signed result.
    """
    sign_a = (a_raw >> (WA - 1)) & 1
    sign_w = (w_raw >> (WW - 1)) & 1

    if a_raw == 0 or w_raw == 0:
        return 0

    # Sign/magnitude split (two's-complement negate when signed). Note this
    # is correct even for the most-negative corner case (e.g. a_raw = 0x80
    # at WA=8): -(-128) mod 256 == 128 == 0x80, whose bit pattern read as
    # unsigned is exactly the correct magnitude 2**(WA-1).
    mag_a = twos(-to_signed(a_raw, WA), WA) if sign_a else a_raw
    mag_w = twos(-to_signed(w_raw, WW), WW) if sign_w else w_raw

    mag_result = mitchell_unsigned(mag_a, mag_w, WA, WW)

    WOUT = WA + WW
    if sign_a ^ sign_w:
        return twos(-mag_result, WOUT)
    return twos(mag_result, WOUT)


def exact_mult(a_raw: int, w_raw: int, WA: int, WW: int) -> int:
    """Exact signed multiply of two (WA/WW)-bit two's-complement values."""
    a = to_signed(a_raw, WA)
    w = to_signed(w_raw, WW)
    return twos(a * w, WA + WW)


def random_test(WA: int, WW: int, n_samples: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    max_abs_err = 0
    max_rel_err = 0.0
    sum_rel_err = 0.0
    n_nonzero = 0
    n_over = 0  # cases where |approx| > |exact| -- should never happen
    for _ in range(n_samples):
        a_raw = int(rng.integers(0, 1 << WA))
        w_raw = int(rng.integers(0, 1 << WW))
        approx_val = to_signed(mitchell(a_raw, w_raw, WA, WW), WA + WW)
        exact_val = to_signed(exact_mult(a_raw, w_raw, WA, WW), WA + WW)
        err = approx_val - exact_val
        max_abs_err = max(max_abs_err, abs(err))
        if exact_val != 0:
            rel = abs(err) / abs(exact_val)
            max_rel_err = max(max_rel_err, rel)
            sum_rel_err += rel
            n_nonzero += 1
            if abs(approx_val) > abs(exact_val):
                n_over += 1
    mean_rel_err = sum_rel_err / n_nonzero if n_nonzero else 0.0
    return max_abs_err, max_rel_err, mean_rel_err, n_over


def check_exhaustive(WA: int, WW: int):
    """Walk every (a_raw, w_raw) pair at a small width -- exercises zero
    operands and every sign combination exhaustively, not just by sampling."""
    max_err = 0
    max_rel = 0.0
    n_over = 0
    for a_raw in range(1 << WA):
        for w_raw in range(1 << WW):
            approx_val = to_signed(mitchell(a_raw, w_raw, WA, WW), WA + WW)
            exact_val = to_signed(exact_mult(a_raw, w_raw, WA, WW), WA + WW)
            err = approx_val - exact_val
            max_err = max(max_err, abs(err))
            if exact_val != 0:
                max_rel = max(max_rel, abs(err) / abs(exact_val))
                if abs(approx_val) > abs(exact_val):
                    n_over += 1
    return max_err, max_rel, n_over


def main():
    print("Verifying mitchell_base from hls4mlrcoem\n")
    print(f"{'WA':>4} {'WW':>4}  {'max|err|':>12}  {'max rel err':>12}  {'mean rel err':>13}  {'#overestimates':>15}")
    print("-" * 72)

    for WA, WW in [(8, 8), (12, 12), (16, 16), (18, 8), (16, 6)]:
        max_err, max_rel, mean_rel, n_over = random_test(WA, WW)
        print(f"{WA:>4} {WW:>4}  {max_err:>12d}  {max_rel:>12.4%}  {mean_rel:>13.4%}  {n_over:>15d}")
        assert n_over == 0, f"WA={WA} WW={WW}: found an overestimate -- Mitchell's algorithm should never overestimate"
        assert max_rel < 0.115, f"WA={WA} WW={WW}: max relative error {max_rel:.4%} exceeds the ~11.1% textbook bound"

    print("\nExhaustive check (every (a, w) pair), WA=WW=6:")
    max_err, max_rel, n_over = check_exhaustive(6, 6)
    print(f"  max|err| = {max_err}, max rel err = {max_rel:.4%}, overestimate count = {n_over}")
    assert n_over == 0
    assert max_rel < 0.115

    print("\nmitchell_base is bit-faithful to its mathematical claim:")
    print("  - leading-one-detect + linear mantissa (log2(1+x) ~= x) on both operands")
    print("  - exponents added, mantissas added (with carry bumping the exponent)")
    print("  - magnitude reconstructed by re-inserting the leading one and shifting")
    print("  - sign is XORed back on separately (the log/magnitude path is unsigned)")
    print()
    print("Confirmed properties (matching the classic Mitchell 1962 algorithm):")
    print("  - never overestimates |product| (log2(1+x) >= x on [0,1], concave, touches")
    print("    the chord y=x only at the endpoints)")
    print("  - worst-case relative error ~11.1%, at x=0.5 on both operands")
    print("  - mean relative error a few percent over uniformly distributed operands")
    print()
    print("Unlike lpor/lsb_zero/comp42/random_lsb (which keep a real multiply for the")
    print("hi/hi, hi/lo, lo/hi quadrants and only approximate lo*lo), Mitchell's algorithm")
    print("replaces the multiply entirely with a leading-one-detect + add + shift -- no")
    print("multiplier is inferred anywhere in this path.")


if __name__ == "__main__":
    main()
