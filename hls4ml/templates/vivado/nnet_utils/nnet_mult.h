#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include <iostream>
#include <math.h>

namespace nnet {

namespace product {

/* ---
 * different methods to perform the product of input and weight, depending on the
 * types of each.
 * --- */

class Product {};

template <class x_T, class w_T> class both_binary : public Product {
  public:
    static x_T product(x_T a, w_T w) {
        // specialisation for 1-bit weights and incoming data
        #pragma HLS INLINE
        return a == w;
    }
};

template <class x_T, class w_T> class weight_binary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 1-bit weights, arbitrary data
        #pragma HLS INLINE
        if (w == 0)
            return -a;
        else
            return a;
    }
};

template <class x_T, class w_T> class data_binary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-w) {
        // Specialisation for 1-bit data, arbitrary weight
        #pragma HLS INLINE
        if (a == 0)
            return -w;
        else
            return w;
    }
};

template <class x_T, class w_T> class weight_ternary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 2-bit weights, arbitrary data
        #pragma HLS INLINE
        if (w == 0)
            return 0;
        else if (w == -1)
            return -a;
        else
            return a; // if(w == 1)
    }
};

template <class x_T, class w_T> class mult : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
};

/* ---
 * Lower-Part-OR (LPOR) approximate multiplier.
 *
 * Splits each operand's raw two's-complement bit vector into a signed high
 * part and an unsigned k-bit low part. The three partial products involving
 * at least one high part are computed exactly; the k x k low*low partial
 * product (which contributes at most on the order of 2^(2k) raw units, i.e.
 * a bounded but non-negligible share of the result for small k) is replaced
 * by a bitwise OR of the two low parts, eliminating that quadrant's
 * multiply/carry logic. This is a bit-exact reimplementation of how
 * ap_fixed/ap_int operator* works internally (integer multiply of the raw
 * bits, result type just relabels the binary point), so it is valid for any
 * x_T/w_T pairing of ap_fixed or ap_int types.
 * --- */
template <class x_T, class w_T, int K> class lower_part_or_base : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        using r_T = decltype(a * w);
        static const int WA = x_T::width;
        static const int WW = w_T::width;
        static const int KA = (K < WA) ? K : WA - 1;
        static const int KW = (K < WW) ? K : WW - 1;
        static const int WOUT = WA + WW;

        ap_int<WA> raw_a;
        raw_a.range(WA - 1, 0) = a.range(WA - 1, 0);
        ap_int<WW> raw_w;
        raw_w.range(WW - 1, 0) = w.range(WW - 1, 0);

        // Natural (narrow) widths for the high parts -- keeping these at their
        // true bit-width, rather than pre-widening to WOUT, is what lets downstream
        // synthesis actually infer smaller multipliers for the un-approximated quadrants.
        ap_int<WA - KA> a_hi = raw_a >> KA; // arithmetic (signed) shift
        ap_int<WW - KW> w_hi = raw_w >> KW;
        ap_uint<KA> a_lo = raw_a.range(KA - 1, 0);
        ap_uint<KW> w_lo = raw_w.range(KW - 1, 0);

        ap_int<WOUT> hi_hi = a_hi * w_hi; // natural-width multiplies; widened only on assignment
        ap_int<WOUT> hi_lo = a_hi * w_lo;
        ap_int<WOUT> lo_hi = w_hi * a_lo;
        ap_int<WOUT> lo_lo_approx = ap_int<WOUT>(a_lo | w_lo); // <-- the approximation

        ap_int<WOUT> raw_result = (hi_hi << (KA + KW)) + (hi_lo << KW) + (lo_hi << KA) + lo_lo_approx;

        r_T result;
        result.range(WOUT - 1, 0) = raw_result.range(WOUT - 1, 0);
        return result;
    }
};

template <class x_T, class w_T> class lpor_k4 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return lower_part_or_base<x_T, w_T, 4>::product(a, w);
    }
};

template <class x_T, class w_T> class lpor_k8 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return lower_part_or_base<x_T, w_T, 8>::product(a, w);
    }
};

/* ---
 * LSB-zeroing (truncated) approximate multiplier.
 *
 * Simpler than lower_part_or_base: rather than reconstructing the low k x k
 * quadrant via a bitwise OR, this just ties the bottom k bits of both
 * operands to zero and performs the ordinary exact multiply on the
 * truncated values. No extra shift/add logic is introduced, so downstream
 * synthesis sees a plain multiply whose operands are provably zero in their
 * low k bits (which it can use to narrow the inferred multiplier).
 * --- */
template <class x_T, class w_T, int K> class lsb_zero_base : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        static const int WA = x_T::width;
        static const int WW = w_T::width;
        static const int KA = (K < WA) ? K : WA - 1;
        static const int KW = (K < WW) ? K : WW - 1;

        x_T a_trunc = a;
        w_T w_trunc = w;
        a_trunc.range(KA - 1, 0) = 0;
        w_trunc.range(KW - 1, 0) = 0;

        return a_trunc * w_trunc;
    }
};

template <class x_T, class w_T> class lsb_zero_k4 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return lsb_zero_base<x_T, w_T, 4>::product(a, w);
    }
};

template <class x_T, class w_T> class lsb_zero_k8 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return lsb_zero_base<x_T, w_T, 8>::product(a, w);
    }
};

/* ---
 * 4:2 Compressor-Based Approximate Multiplier
 *
 * A standard 4:2 compressor reduces four partial product bits (plus an optional
 * lateral carry-in) to a sum bit and a carry bit (plus a lateral carry-out for
 * chaining compressors in the same column). It is built from two full adders
 * in series -- the textbook (4:2) counter.
 *
 * We provide:
 *   - `comp42_exact_5to3`     : exact 5-input/3-output compressor (with lateral chain)
 *   - `comp42_approx_5to3`   : simplified logic, fewer gates, bounded error
 *   - `comp42_counter_exact` : 4-input/2-output counter (exact for 0..3 ones,
 *                              lossy for the all-ones case)
 *   - `comp42_counter_approx` : simplified carry logic, larger bounded error
 *
 * These are wired into a multi-bit `comp42_reduce4` that reduces 4 multi-bit
 * rows to (sum, carry) and a K x K `mult_comp42` partial-product-reduction
 * multiplier (K = 4 or 8).
 *
 * Finally, `compressor_4_2_base<x_T, w_T, K>` mirrors the LPOR split: the
 * three high quadrants (hi*hi, hi*lo, lo*hi) are computed exactly, and the
 * low*low quadrant is replaced by the compressor-based multiply above. The
 * shortcut aliases `comp42_k4` / `comp42_k8` are what hls4ml's
 * `mult_strategy` ChoiceAttribute consumes.
 * --- */
namespace compressor {

// Standard 4:2 compressor (exact, with lateral carry chaining).
//
// Reduces 4 same-weight bits (a,b,c,d) plus a lateral carry-in (cin) at the
// same weight to: a sum bit at the same weight, a vertical carry bit at the
// next-higher weight (carry), and a lateral carry-out at the next-higher
// weight (cout) that is intended for the next compressor's `cin` in the same
// column.
//
// Key property: `cout` does NOT depend on `cin`, so the lateral carry chain
// is non-rippling. This is what makes 4:2 compressors attractive as building
// blocks of partial-product-reduction trees.
//
// Weight conservation: a + b + c + d + cin == sum + 2*carry + 2*cout.
inline void comp42_exact_5to3(ap_uint<1> a, ap_uint<1> b, ap_uint<1> c, ap_uint<1> d,
                              ap_uint<1> cin, ap_uint<1> &sum, ap_uint<1> &carry,
                              ap_uint<1> &cout) {
    #pragma HLS INLINE
    // Stage 1: full adder on a, b, c
    ap_uint<1> s1 = a ^ b ^ c;
    ap_uint<1> c1 = (a & b) | (a & c) | (b & c); // majority(a,b,c); NO cin dependence
    // Stage 2: full adder on s1, d, cin
    sum = s1 ^ d ^ cin;
    ap_uint<1> c2 = (s1 & d) | (s1 & cin) | (d & cin); // majority(s1,d,cin)
    cout = c1;  // lateral carry-out (independent of cin)
    carry = c2; // vertical carry to next-higher weight column
}

// Approximate 4:2 compressor (simplified logic).
//
// Drops cin from the sum computation, simplifies the carry/cout functions to
// pure AND-OR form. Fewer gates / shorter critical path at the cost of a
// bounded error in some input combinations.
//
// cout is still cin-independent, so the lateral chain remains non-rippling.
inline void comp42_approx_5to3(ap_uint<1> a, ap_uint<1> b, ap_uint<1> c, ap_uint<1> d,
                                ap_uint<1> cin, ap_uint<1> &sum, ap_uint<1> &carry,
                                ap_uint<1> &cout) {
    #pragma HLS INLINE
    (void)cin; // explicitly ignored -- this is the approximation
    sum = a ^ b ^ c ^ d;             // ignores cin
    carry = (a & b) | (c & d);        // simplified vertical carry
    cout = (a & c) | (b & d);         // simplified lateral carry (cin-independent)
}

// (4:2) counter -- reduces 4 same-weight bits to a sum bit and a single
// carry bit (no lateral chain). Used inside the multi-bit compressor tree.
//
// The (4:2) counter is exact for input populations 0..3; it is lossy for the
// all-ones case (where the true value is 4 = 0b100, but the 2-bit output can
// only represent 0..3). In the multiplier context, this loss is bounded by
// 2^(i+1) per occurrence and is rare in practice.
inline void comp42_counter_exact(ap_uint<1> a, ap_uint<1> b, ap_uint<1> c, ap_uint<1> d,
                                 ap_uint<1> &sum, ap_uint<1> &carry) {
    #pragma HLS INLINE
    sum = a ^ b ^ c ^ d;
    carry = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d); // majority(a,b,c,d)
}

// Approximate (4:2) counter -- simplified carry logic that only considers
// the (a,b) and (c,d) pairs. Fewer gates but larger bounded error.
inline void comp42_counter_approx(ap_uint<1> a, ap_uint<1> b, ap_uint<1> c, ap_uint<1> d,
                                   ap_uint<1> &sum, ap_uint<1> &carry) {
    #pragma HLS INLINE
    sum = a ^ b ^ c ^ d;
    carry = (a & b) | (c & d);
}

// Multi-bit (4:2) counter: reduces 4 multi-bit rows of width W to two rows
// (sum and carry). The sum row has W bits; the carry row has W+1 bits
// (because the carry of bit W-1 lands at bit W).
//
// `approx=false` uses `comp42_counter_exact` (lossy only on all-1s columns).
// `approx=true`  uses `comp42_counter_approx` (larger bounded error).
template <int W, bool APPROX> inline void comp42_reduce4(ap_uint<W> a, ap_uint<W> b, ap_uint<W> c, ap_uint<W> d,
                                                        ap_uint<W> &sum, ap_uint<W + 1> &carry) {
    #pragma HLS INLINE
    sum = 0;
    carry = 0;
    for (int i = 0; i < W; i++) {
        #pragma HLS UNROLL
        ap_uint<1> ai = a[i], bi = b[i], ci = c[i], di = d[i];
        ap_uint<1> s_bit, c_bit;
        if (APPROX) {
            comp42_counter_approx(ai, bi, ci, di, s_bit, c_bit);
        } else {
            comp42_counter_exact(ai, bi, ci, di, s_bit, c_bit);
        }
        sum[i] = s_bit;
        // The carry output at weight i lands at bit i+1 of the carry row.
        if (c_bit)
            carry[i + 1] = 1;
    }
}

// K x K unsigned multiplier via 4:2-compressor partial-product reduction.
//
// Supports K=4 (one compression level: 4 partial products -> 1 compressor ->
// sum + carry -> CPA) and K=8 (two levels). For unsupported K the function
// falls back to an exact integer multiply.
template <int K, bool APPROX> inline ap_uint<2 * K> mult_comp42(ap_uint<K> a, ap_uint<K> w) {
    #pragma HLS INLINE
    // Generate K partial products, each 2K bits wide
    ap_uint<2 * K> pps[K];
    for (int i = 0; i < K; i++) {
        #pragma HLS UNROLL
        ap_uint<2 * K> pp = 0;
        if (a[i]) {
            // pp = w << i, zero-extended to 2K bits
            for (int j = 0; j < K; j++) {
                #pragma HLS UNROLL
                pp[i + j] = w[j];
            }
        }
        pps[i] = pp;
    }

    if (K == 4) {
        ap_uint<2 * K> sum;
        ap_uint<2 * K + 1> carry;
        comp42_reduce4<2 * K, APPROX>(pps[0], pps[1], pps[2], pps[3], sum, carry);
        // Final CPA (carry-propagate add). Truncate to 2K bits (the true result width).
        return ap_uint<2 * K>((sum + carry)(2 * K - 1, 0));
    } else if (K == 8) {
        // Level 1: two 4:2 compressors, each takes 4 partial products.
        ap_uint<2 * K> s1, s2;
        ap_uint<2 * K + 1> c1, c2;
        comp42_reduce4<2 * K, APPROX>(pps[0], pps[1], pps[2], pps[3], s1, c1);
        comp42_reduce4<2 * K, APPROX>(pps[4], pps[5], pps[6], pps[7], s2, c2);
        // Level 2: one 4:2 compressor on the 4 rows (s1, c1, s2, c2).
        // The widest of these is 2K+1 bits, so we widen the sum rows (zero-extend)
        // to 2K+1 and reduce at that width.
        ap_uint<2 * K + 1> s1e = s1; // zero-extend from 2K to 2K+1
        ap_uint<2 * K + 1> s2e = s2;
        ap_uint<2 * K + 1> sum_l2;
        ap_uint<2 * K + 2> carry_l2;
        comp42_reduce4<2 * K + 1, APPROX>(s1e, c1, s2e, c2, sum_l2, carry_l2);
        return ap_uint<2 * K>((sum_l2 + carry_l2)(2 * K - 1, 0));
    } else {
        // Unsupported K: fall back to exact (only K=4, K=8 are wired in practice).
        return ap_uint<2 * K>(a * w);
    }
}

} // namespace compressor

// Compressor-based approximate multiplier (LPOR-style split + comp42 lo*lo).
//
// Splits each operand into a signed high part and an unsigned k-bit low part.
// The three high partial products (hi*hi, hi*lo, lo*hi) are computed exactly
// with the natural (narrow-width) multiplies; the low*low quadrant is replaced
// by a 4:2-compressor-tree multiply that uses the approximate (4:2) counter
// variant -- introducing a bounded error whose magnitude is O(2^(2K)).
template <class x_T, class w_T, int K> class compressor_4_2_base : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        using r_T = decltype(a * w);
        static const int WA = x_T::width;
        static const int WW = w_T::width;
        static const int KA = (K < WA) ? K : WA - 1;
        static const int KW = (K < WW) ? K : WW - 1;
        static const int WOUT = WA + WW;

        ap_int<WA> raw_a;
        raw_a.range(WA - 1, 0) = a.range(WA - 1, 0);
        ap_int<WW> raw_w;
        raw_w.range(WW - 1, 0) = w.range(WW - 1, 0);

        ap_int<WA - KA> a_hi = raw_a >> KA; // arithmetic (signed) shift
        ap_int<WW - KW> w_hi = raw_w >> KW;
        ap_uint<KA> a_lo = raw_a.range(KA - 1, 0);
        ap_uint<KW> w_lo = raw_w.range(KW - 1, 0);

        ap_int<WOUT> hi_hi = a_hi * w_hi; // natural-width multiplies; widened only on assignment
        ap_int<WOUT> hi_lo = a_hi * w_lo;
        ap_int<WOUT> lo_hi = w_hi * a_lo;
        // Compressor-based lo*lo -- uses the approximate (4:2) counter tree
        // for symmetric splits (KA == KW == K), otherwise falls back to exact.
        ap_int<WOUT> lo_lo;
        if (KA == K && KW == K) {
            lo_lo = ap_int<WOUT>(compressor::mult_comp42<K, true>(a_lo, w_lo));
        } else {
            lo_lo = ap_int<WOUT>(a_lo * w_lo); // exact for asymmetric clamped splits
        }

        ap_int<WOUT> raw_result = (hi_hi << (KA + KW)) + (hi_lo << KW) + (lo_hi << KA) + lo_lo;

        r_T result;
        result.range(WOUT - 1, 0) = raw_result.range(WOUT - 1, 0);
        return result;
    }
};

template <class x_T, class w_T> class comp42_k4 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return compressor_4_2_base<x_T, w_T, 4>::product(a, w);
    }
};

template <class x_T, class w_T> class comp42_k8 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return compressor_4_2_base<x_T, w_T, 8>::product(a, w);
    }
};

/* ---
 * Random-LSB Approximate Multiplier
 *
 * Same LPOR-style split as `lower_part_or_base` and `compressor_4_2_base`:
 * each operand is divided into a signed high part and an unsigned k-bit low
 * part; the three high partial products (hi*hi, hi*lo, lo*hi) are computed
 * exactly. The lo*lo quadrant, however, is replaced by a *pseudo-random
 * k-bit pattern* generated by a 16-bit LFSR seeded with `SEED`.
 *
 * Why this saves LUTs:
 *   - The lo*lo multiplier (which would normally be a K x K multiplier taking
 *     up to K^2 LUTs) is replaced by a compile-time constant -- the random
 *     bit pattern is folded into the design at synthesis time, so the entire
 *     quadrant collapses to wires tied to VCC/GND.
 *   - HLS does not perform any arithmetic on the LSBs; the output bits in
 *     the lo*lo weight range are simply the (constant) random pattern.
 *
 * The pattern is deterministic per `SEED` (so each instance of the template
 * gets a stable, reproducible approximation). Different SEEDs give different
 * patterns, which is useful for spreading error across layers in a network.
 *
 * The approximation error is bounded by the max value of lo*lo, i.e.
 * (2^K - 1)^2 < 2^(2K), comparable to LPOR with the same K.
 * --- */
namespace random_lsb {

// 16-bit Galois LFSR with polynomial x^16 + x^14 + x^13 + x^11 + 1.
// Returns a `K`-bit pseudo-random pattern derived from `SEED`.
// Stateless (re-seeds from SEED on every call) so HLS can constant-fold
// the result at synthesis time.
template <unsigned SEED, int K> inline ap_uint<K> pattern() {
    #pragma HLS INLINE
    ap_uint<16> state = SEED == 0 ? 1 : SEED; // LFSR must be non-zero
    ap_uint<K> result = 0;
    for (int i = 0; i < K; i++) {
        #pragma HLS UNROLL
        ap_uint<1> lsb = state[0];
        state >>= 1;
        if (lsb) {
            state ^= 0xB400; // taps at x^16, x^14, x^13, x^11
        }
        result[i] = state[0];
    }
    return result;
}

} // namespace random_lsb

template <class x_T, class w_T, int K, unsigned SEED = 0x1234> class random_lsb_base : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        using r_T = decltype(a * w);
        static const int WA = x_T::width;
        static const int WW = w_T::width;
        static const int KA = (K < WA) ? K : WA - 1;
        static const int KW = (K < WW) ? K : WW - 1;
        static const int WOUT = WA + WW;

        ap_int<WA> raw_a;
        raw_a.range(WA - 1, 0) = a.range(WA - 1, 0);
        ap_int<WW> raw_w;
        raw_w.range(WW - 1, 0) = w.range(WW - 1, 0);

        ap_int<WA - KA> a_hi = raw_a >> KA; // arithmetic (signed) shift
        ap_int<WW - KW> w_hi = raw_w >> KW;
        ap_uint<KA> a_lo = raw_a.range(KA - 1, 0);
        ap_uint<KW> w_lo = raw_w.range(KW - 1, 0);

        ap_int<WOUT> hi_hi = a_hi * w_hi; // natural-width multiplies; widened only on assignment
        ap_int<WOUT> hi_lo = a_hi * w_lo;
        ap_int<WOUT> lo_hi = w_hi * a_lo;
        // The approximation: replace lo*lo with a pseudo-random pattern.
        // No arithmetic is performed -- the pattern is a compile-time
        // constant derived from SEED, which HLS can fold into the design
        // (the lo*lo multiplier is replaced by wires tied to VCC/GND).
        ap_int<WOUT> lo_lo;
        if (KA == K && KW == K) {
            lo_lo = ap_int<WOUT>(random_lsb::pattern<SEED, 2 * K>());
        } else {
            // Asymmetric clamped split: fall back to the natural-width pattern.
            lo_lo = ap_int<WOUT>(random_lsb::pattern<SEED, KA + KW>());
        }

        ap_int<WOUT> raw_result = (hi_hi << (KA + KW)) + (hi_lo << KW) + (lo_hi << KA) + lo_lo;

        r_T result;
        result.range(WOUT - 1, 0) = raw_result.range(WOUT - 1, 0);
        return result;
    }
};

template <class x_T, class w_T> class random_lsb_k4 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return random_lsb_base<x_T, w_T, 4, 0x1234>::product(a, w);
    }
};

template <class x_T, class w_T> class random_lsb_k8 : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return random_lsb_base<x_T, w_T, 8, 0x5678>::product(a, w);
    }
};

/* ---
 * Mitchell's Logarithmic Approximate Multiplier
 *
 * Unlike the LPOR / lsb_zero / comp42 / random_lsb approximators above (which
 * all keep a real hi/hi, hi/lo, lo/hi partial-product multiply and only
 * approximate the low k x k quadrant), Mitchell's algorithm replaces the
 * multiply entirely with an add:
 *
 *   1. Take the sign/magnitude of each operand. log2(magnitude) is
 *      approximated as `exponent + mantissa`, where `exponent` is the index
 *      of the magnitude's leading (most-significant) set bit -- found via a
 *      leading-one detector (priority encoder) -- and `mantissa` is the
 *      value's remaining lower bits, left-justified into a fixed-point
 *      fraction in [0, 1). This is the classic linear approximation
 *      log2(1+x) ~= x for x in [0, 1).
 *   2. log2(product) ~= log2(|a|) + log2(|w|), computed as an exponent add
 *      plus a mantissa add (with carry-out bumping the exponent by one when
 *      the mantissas overflow past 1.0).
 *   3. The approximate magnitude is reconstructed from the summed
 *      exponent/mantissa by re-inserting the implicit leading one and
 *      shifting -- the same leading-one-detect-then-shift structure as step
 *      1, run in reverse.
 *   4. The original operand signs are XORed back onto the result.
 *
 * No multiplier is inferred anywhere in this path -- only a priority
 * encoder, a fixed-point add, and two variable (data-dependent) shifts, so
 * this is the one strategy in this file with a real chance of moving DSP
 * usage rather than just LUTs.
 *
 * Because log2(1+x) is concave and matches the chord y=x only at its
 * endpoints x=0 and x=1, log2(1+x) >= x everywhere in between: the
 * approximation always UNDERESTIMATES log2(magnitude), so the reconstructed
 * product is always <= the true magnitude in absolute value -- Mitchell's
 * algorithm never overestimates. The worst case is x=0.5 on both operands,
 * giving the textbook maximum relative error of ~11.1%; mean relative error
 * over uniformly distributed operands is a few percent. See
 * scripts/verify_mitchell.py for a bit-exact Python re-derivation of this
 * bound.
 *
 * Assumes WA = x_T::width >= 2 and WW = w_T::width >= 2 (true for every
 * hls4ml precision type used in practice); degenerate 1-bit operands are not
 * handled specially, same as this file's existing K-guard conventions.
 * --- */
template <class x_T, class w_T> class mitchell_base : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        using r_T = decltype(a * w);
        static const int WA = x_T::width;
        static const int WW = w_T::width;
        static const int WOUT = WA + WW;
        static const int MANT_A = WA - 1;
        static const int MANT_W = WW - 1;
        static const int MANT_BITS = (MANT_A > MANT_W) ? MANT_A : MANT_W;

        ap_int<WA> raw_a;
        raw_a.range(WA - 1, 0) = a.range(WA - 1, 0);
        ap_int<WW> raw_w;
        raw_w.range(WW - 1, 0) = w.range(WW - 1, 0);

        if (raw_a == 0 || raw_w == 0) {
            r_T zero_result = 0;
            return zero_result;
        }

        // Sign/magnitude split -- Mitchell's algorithm operates on unsigned
        // magnitudes via leading-one detection, so (unlike the LPOR-style
        // hi/lo split used by the other approximators) signedness has to be
        // pulled out explicitly up front and re-applied at the end.
        bool sign_a = raw_a[WA - 1];
        bool sign_w = raw_w[WW - 1];
        bool neg_result = sign_a ^ sign_w;

        ap_uint<WA> mag_a = sign_a ? ap_uint<WA>(-raw_a) : ap_uint<WA>(raw_a);
        ap_uint<WW> mag_w = sign_w ? ap_uint<WW>(-raw_w) : ap_uint<WW>(raw_w);

        // Leading-one detection (priority encoder): exp_* = index of the
        // highest set bit. mag_a/mag_w are both nonzero here (the raw_a/raw_w
        // == 0 case was already handled above), so exactly one of these
        // unrolled comparisons wins for each operand.
        int exp_a = 0;
        bool found_a = false;
        for (int i = WA - 1; i >= 0; i--) {
            #pragma HLS UNROLL
            if (!found_a && mag_a[i]) {
                exp_a = i;
                found_a = true;
            }
        }
        int exp_w = 0;
        bool found_w = false;
        for (int i = WW - 1; i >= 0; i--) {
            #pragma HLS UNROLL
            if (!found_w && mag_w[i]) {
                exp_w = i;
                found_w = true;
            }
        }

        // Left-justify the bits below the leading one into a fixed-width
        // fractional mantissa (variable/data-dependent shift -- the "shift"
        // half of Mitchell's leading-one-detect-then-shift structure).
        ap_uint<WA> mag_a_norm = ap_uint<WA>(mag_a) << (MANT_A - exp_a);
        ap_uint<MANT_A> mant_a = mag_a_norm.range(MANT_A - 1, 0);
        ap_uint<WW> mag_w_norm = ap_uint<WW>(mag_w) << (MANT_W - exp_w);
        ap_uint<MANT_W> mant_w = mag_w_norm.range(MANT_W - 1, 0);

        // Add the two mantissas at a common (widest) width, then add the
        // exponents -- this is the "multiply becomes add" step. A carry out
        // of the mantissa add means mant_a + mant_w >= 1.0, so it bumps the
        // combined exponent by one (matching normal log addition).
        ap_uint<MANT_BITS> mant_a_ext = ap_uint<MANT_BITS>(mant_a) << (MANT_BITS - MANT_A);
        ap_uint<MANT_BITS> mant_w_ext = ap_uint<MANT_BITS>(mant_w) << (MANT_BITS - MANT_W);
        ap_uint<MANT_BITS + 1> sum_frac = ap_uint<MANT_BITS + 1>(mant_a_ext) + ap_uint<MANT_BITS + 1>(mant_w_ext);
        bool carry = sum_frac[MANT_BITS];
        ap_uint<MANT_BITS> mant_sum = sum_frac.range(MANT_BITS - 1, 0);
        int exp_sum = exp_a + exp_w + (carry ? 1 : 0);

        // Reconstruct the magnitude: re-insert the implicit leading one,
        // then shift back out to its final position (the inverse of the
        // leading-one-detect-then-shift used to build the mantissas above).
        ap_uint<MANT_BITS + 1> normalized = (ap_uint<MANT_BITS + 1>(1) << MANT_BITS) | ap_uint<MANT_BITS + 1>(mant_sum);
        int shift = exp_sum - MANT_BITS;
        ap_uint<WOUT> result_mag;
        if (shift >= 0) {
            result_mag = ap_uint<WOUT>(normalized) << shift;
        } else {
            result_mag = ap_uint<WOUT>(normalized) >> (-shift);
        }

        ap_int<WOUT> raw_result = neg_result ? ap_int<WOUT>(-ap_int<WOUT>(result_mag)) : ap_int<WOUT>(result_mag);

        r_T result;
        result.range(WOUT - 1, 0) = raw_result.range(WOUT - 1, 0);
        return result;
    }
};

template <class x_T, class w_T> class mitchell : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        #pragma HLS INLINE
        return mitchell_base<x_T, w_T>::product(a, w);
    }
};

template <class x_T, class w_T> class weight_exponential : public Product {
  public:
    using r_T = ap_fixed<2 * (decltype(w_T::weight)::width + x_T::width), (decltype(w_T::weight)::width + x_T::width)>;
    static r_T product(x_T a, w_T w) {
        // Shift product for exponential weights
        #pragma HLS INLINE

        // Shift by the exponent. Negative weights shift right
        r_T y = static_cast<r_T>(a) << w.weight;

        // Negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<r_T>(-y);
    }
};

} // namespace product

template <class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value &&
                                   std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value,
                               ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>>::type
cast(typename CONFIG_T::accum_t x) {
    return static_cast<ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>>(x * 2 - CONFIG_T::n_in);
}

template <class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<
    std::is_same<data_T, ap_uint<1>>::value && !std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value, res_T>::type
cast(typename CONFIG_T::accum_t x) {
    return (res_T)x;
}

template <class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<(!std::is_same<data_T, ap_uint<1>>::value), res_T>::type cast(typename CONFIG_T::accum_t x) {
    return (res_T)x;
}

} // namespace nnet

#endif
