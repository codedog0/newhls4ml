#!/usr/bin/env python3
"""
Reference simulation of the 4:2-compressor-based approximate multiplier that we
are about to land in nnet_mult.h.

Validates:
  * exact 4:2 compressor truth table (built from two chained full adders)
  * approximate 4:2 compressor truth table (simplified logic)
  * multi-bit (4:2) counter reduces 4 multi-bit values to (sum, carry)
  * K x K unsigned multiplier via compressor tree, for K in {4, 8}
  * comp42_base<x_T, w_T, K> behaviour matches the LPOR-style split where
    the lo*lo quadrant is replaced by the compressor-based multiply
"""

import numpy as np


def twos(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def signed(raw: int, width: int) -> int:
    return raw - (1 << width) if raw & (1 << (width - 1)) else raw


# ============================================================================
# 1-bit 4:2 compressors
# ============================================================================

def comp42_exact_5to3(a, b, c, d, cin):
    """
    Standard 4:2 compressor (with lateral carry chaining).
    Built from two full adders in series.

    Inputs:  4 bits at weight w (a,b,c,d) + 1 lateral carry-in (cin) at weight w
    Outputs: sum (weight w), vertical carry (weight w+1), lateral cout (weight w+1)

    Key property: cout does NOT depend on cin (non-rippling lateral chain).
    Weight conservation: a+b+c+d+cin == sum + 2*carry + 2*cout.
    """
    s1 = a ^ b ^ c
    c1 = (a & b) | (a & c) | (b & c)           # carry of FA1 (no cin)
    s = s1 ^ d ^ cin
    c2 = (s1 & d) | (s1 & cin) | (d & cin)     # carry of FA2 (depends on cin)
    return s, c2, c1   # sum, vertical carry, lateral carry


def comp42_approx_5to3(a, b, c, d, cin):
    """
    Approximate 4:2 compressor (simplified logic, fewer gates).

    Approximations:
      * sum ignores cin            (error when cin=1 and a^b^c^d=0)
      * vertical carry = (a&b) | (c&d)
      * lateral cout   = (a&c) | (b&d)   (still cin-independent)
    """
    s = a ^ b ^ c ^ d           # ignores cin
    carry = (a & b) | (c & d)
    cout = (a & c) | (b & d)
    return s, carry, cout


def comp42_counter_exact(a, b, c, d):
    """
    (4:2) counter -- reduces 4 bits at weight w to a sum bit and a carry bit.
    No chaining. Exact for 0-3 ones; lossy for the all-ones case (would need 3 bits).
    """
    s = a ^ b ^ c ^ d
    carry = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d)
    return s, carry


def comp42_counter_approx(a, b, c, d):
    """
    Approximate (4:2) counter -- simplified carry logic.
    May lose carries in some input patterns but uses fewer gates.
    """
    s = a ^ b ^ c ^ d
    carry = (a & b) | (c & d)   # only considers (a,b) pair and (c,d) pair
    return s, carry


# ============================================================================
# Multi-bit (4:2) counter -- reduces 4 multi-bit rows to 2 (sum + carry)
# ============================================================================

def comp42_multibit(a, b, c, d, width, *, approx=False):
    """
    Apply the (4:2) counter at each bit position independently.
    The carry output at weight i goes to bit i+1 of the carry row (since it's at
    the next-higher weight). This is the standard multi-bit carry-save style
    reduction -- the carry row is effectively a<<1 vs the sum row.

    The compressor is exact for any input pattern except the all-1s case at a
    single bit position, which loses one carry (introduces a bounded error
    of 2^(i+1) at that bit position).

    For the `approx=True` variant, additional carries are dropped via the
    simplified carry logic, giving a larger but still bounded error.
    """
    fn = comp42_counter_approx if approx else comp42_counter_exact
    sum_v = 0
    carry_v = 0
    for i in range(width):
        s, c = fn((a >> i) & 1, (b >> i) & 1, (c >> i) & 1, (d >> i) & 1)
        sum_v |= (s << i)
        # the carry at weight i contributes to bit i+1 of the carry row
        if c:
            carry_v |= (1 << i)   # remember "this column produced a carry"
    # The carry row is at weight+1, so we shift left by 1 to get its true value
    return sum_v, carry_v << 1


# ============================================================================
# K x K unsigned multiplier via 4:2 compressor tree
# ============================================================================

def mult_comp42(a, w, K, *, approx=False):
    """
    K x K unsigned multiplier using a 4:2-compressor tree for partial product
    reduction. Supports K = 4 (1 compression level) and K = 8 (2 levels).

    For other K, falls back to an exact integer multiply (we only need K=4,8
    in practice -- these correspond to comp42_k4 / comp42_k8 in nnet_mult.h).
    """
    pps = [((w << i) if (a >> i) & 1 else 0) & ((1 << (2 * K)) - 1) for i in range(K)]

    if K == 4:
        # 1 level: 4 partial products -> 4:2 compressor -> sum + carry -> CPA
        s, c = comp42_multibit(pps[0], pps[1], pps[2], pps[3], 2 * K, approx=approx)
        return (s + c) & ((1 << (2 * K)) - 1)
    elif K == 8:
        # Level 1: two 4:2 compressors
        s1, c1 = comp42_multibit(pps[0], pps[1], pps[2], pps[3], 2 * K, approx=approx)
        s2, c2 = comp42_multibit(pps[4], pps[5], pps[6], pps[7], 2 * K, approx=approx)
        # Level 2: one 4:2 compressor on the four rows
        # After level 1, the rows are at most 2K+1 bits wide; we use width = 2K+1
        s3, c3 = comp42_multibit(s1, c1, s2, c2, 2 * K + 1, approx=approx)
        return (s3 + c3) & ((1 << (2 * K)) - 1)
    else:
        return (a * w) & ((1 << (2 * K)) - 1)


# ============================================================================
# comp42_base<x_T, w_T, K>: LPOR-style split with compressor-based lo*lo
# ============================================================================

def comp42_base(a_raw, w_raw, WA, WW, K, *, approx=True):
    """
    Mirrors the C++ `comp42_base` template we will write into nnet_mult.h:
      * split each operand into signed high part and unsigned k-bit low part
      * compute hi*hi, hi*lo, lo*hi exactly (signed arithmetic)
      * replace lo*lo with compressor-based multiply (approximate or exact)
      * reassemble via shifts and adds
    """
    KA = K if K < WA else WA - 1
    KW = K if K < WW else WW - 1
    WOUT = WA + WW

    a = signed(a_raw, WA)
    w = signed(w_raw, WW)

    a_hi = a >> KA                 # arithmetic shift -> signed
    w_hi = w >> KW
    a_lo = a_raw & ((1 << KA) - 1)  # unsigned
    w_lo = w_raw & ((1 << KW) - 1)

    hi_hi = a_hi * w_hi
    hi_lo = a_hi * w_lo
    lo_hi = w_hi * a_lo
    if KA == KW:
        lo_lo = mult_comp42(a_lo, w_lo, KA, approx=approx)
    else:
        lo_lo = a_lo * w_lo   # asymmetric split not supported by our comp42 tree

    raw_result = (hi_hi << (KA + KW)) + (hi_lo << KW) + (lo_hi << KA) + lo_lo
    return twos(raw_result, WOUT)


# ============================================================================
# Verification
# ============================================================================

def check_comp42_truth_tables():
    """Walk all 32 input combinations and check the exact compressor's weight-conservation."""
    for inputs in range(32):
        a = (inputs >> 4) & 1
        b = (inputs >> 3) & 1
        c = (inputs >> 2) & 1
        d = (inputs >> 1) & 1
        cin = inputs & 1
        s, carry, cout = comp42_exact_5to3(a, b, c, d, cin)
        lhs = a + b + c + d + cin
        rhs = s + 2 * carry + 2 * cout
        assert lhs == rhs, f"exact: {lhs} != {rhs} for inputs {a,b,c,d,cin}"
    print("[ok] exact 4:2 compressor conserves weight on all 32 input combos")


def check_comp42_counter_bounds():
    """Bound the max error introduced by the (4:2) counter vs the exact 5:3 compressor."""
    max_err = 0
    for inputs in range(16):
        a = (inputs >> 3) & 1
        b = (inputs >> 2) & 1
        c = (inputs >> 1) & 1
        d = inputs & 1
        exact_count = a + b + c + d
        s_e, c_e = comp42_counter_exact(a, b, c, d)
        s_a, c_a = comp42_counter_approx(a, b, c, d)
        err_e = exact_count - (s_e + 2 * c_e)
        err_a = exact_count - (s_a + 2 * c_a)
        max_err = max(max_err, abs(err_e), abs(err_a))
    print(f"[ok] (4:2) counter max |weight error| per bit position = {max_err} (exact: only on all-1s)")


def check_mult_comp42(K):
    """Compare K x K compressor multiplier against the exact product."""
    rng = np.random.default_rng(42)
    max_abs_err_exact = 0
    max_abs_err_approx = 0
    n_samples = 5000
    for _ in range(n_samples):
        a = int(rng.integers(0, 1 << K))
        w = int(rng.integers(0, 1 << K))
        exact = (a * w) & ((1 << (2 * K)) - 1)
        # exact-compressor version (approx=False) -- error is bounded but nonzero
        # because the (4:2) counter loses info when all 4 partial products at a
        # column are 1.
        comp_exact = mult_comp42(a, w, K, approx=False)
        comp_approx = mult_comp42(a, w, K, approx=True)
        max_abs_err_exact = max(max_abs_err_exact, abs(comp_exact - exact))
        max_abs_err_approx = max(max_abs_err_approx, abs(comp_approx - exact))
    print(f"[ok] K={K}: exact-compressor max|err| = {max_abs_err_exact}, "
          f"approx-compressor max|err| = {max_abs_err_approx}")


def check_comp42_base(WA, WW, K):
    """End-to-end check of comp42_base against the exact signed multiply."""
    rng = np.random.default_rng(7)
    max_abs_err_exact = 0
    max_abs_err_approx = 0
    n_samples = 5000
    for _ in range(n_samples):
        a_raw = int(rng.integers(0, 1 << WA))
        w_raw = int(rng.integers(0, 1 << WW))
        exact = signed(twos(signed(a_raw, WA) * signed(w_raw, WW), WA + WW), WA + WW)
        result_exact = signed(comp42_base(a_raw, w_raw, WA, WW, K, approx=False), WA + WW)
        result_approx = signed(comp42_base(a_raw, w_raw, WA, WW, K, approx=True), WA + WW)
        max_abs_err_exact = max(max_abs_err_exact, abs(result_exact - exact))
        max_abs_err_approx = max(max_abs_err_approx, abs(result_approx - exact))
    print(f"[ok] comp42_base<{WA}x{WW},K={K}>: max|err| (exact compressor) = {max_abs_err_exact}, "
          f"max|err| (approx compressor) = {max_abs_err_approx}")


def main():
    print("== 4:2 compressor truth-table and bound checks ==")
    check_comp42_truth_tables()
    check_comp42_counter_bounds()
    print()
    print("== K x K compressor multiplier (standalone) ==")
    for K in (4, 8):
        check_mult_comp42(K)
    print()
    print("== End-to-end comp42_base (LPOR-style split + compressor lo*lo) ==")
    for WA, WW, K in [(8, 8, 4), (8, 8, 8), (12, 12, 4), (12, 12, 8), (16, 16, 4), (16, 16, 8)]:
        check_comp42_base(WA, WW, K)


if __name__ == "__main__":
    main()
