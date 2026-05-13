// Highway-driven XNOR-popcount matmul.
//
// A: (M, Kp), B: (N, Kp). Both packed; same pack width T (∈ {uint8,uint32,uint64}).
// K = logical last dim. Pad bits in the last word are zero.
//
// C(m, n) = 2 * popcount(~(A[m] ⊕ B[n])) − K_eff      ∈ [−K, K]
//
// Where K_eff = 2 * Kp * pw − K absorbs the (Kp*pw − K) extra pad-bit matches
// that XNOR turns into 1s. This is the same correction as the existing
// CPU kernel.
//
// Implementation strategy:
//   * Inner reduction is in-register: XOR → Not → PopulationCount → Add into
//     vector accumulator. No intermediate xnor buffer.
//   * Vector accumulator drained with ReduceSum at the end of each (m, n).
//   * Outer loop sliced by at::parallel_for in the caller.
//   * No allocations in this function.
//
// Single-row variant: XnorPopcountRow computes one row of C against B's
// (N, Kp) rows. parallel_for in the caller dispatches one row per work-item.

#pragma once

#include <hwy/highway.h>
#include "popcount.hpp"
#include <cstddef>
#include <cstdint>

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// XnorPopcountPair: XNOR-popcount inner product of two packed rows.
//
// Why uint64_t reinterpretation:
//   If T=uint8_t we would accumulate into ScalableTag<uint8_t>, whose
//   ReduceSum returns uint8_t.  On NEON (LANES=16) that wraps at 256, so
//   K ≥ 512 (Kp ≥ 64) produces wrong results.  Reinterpreting as uint64_t
//   words makes the accumulator ScalableTag<uint64_t> which returns uint64_t
//   — no overflow for any practical K.  XOR/XNOR is bitwise and commutes
//   with any re-chunking of the byte stream, so the bit count is identical.
template <typename T>
HWY_ATTR inline int32_t XnorPopcountPair(const T* HWY_RESTRICT a_row,
                                         const T* HWY_RESTRICT b_row,
                                         int64_t Kp, int32_t K_eff) {
    const size_t n_bytes = (size_t)Kp * sizeof(T);
    const uint64_t* a64  = reinterpret_cast<const uint64_t*>(a_row);
    const uint64_t* b64  = reinterpret_cast<const uint64_t*>(b_row);
    const size_t n_words = n_bytes / sizeof(uint64_t);

    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);

    size_t i = 0;
    auto acc = hn::Zero(d);
    for (; i + LANES <= n_words; i += LANES) {
        auto va = hn::LoadU(d, a64 + i);
        auto vb = hn::LoadU(d, b64 + i);
        acc = hn::Add(acc, hn::PopulationCount(hn::Not(hn::Xor(va, vb))));
    }
    uint64_t sum = hn::ReduceSum(d, acc);
    for (; i < n_words; ++i) {
        sum += (uint64_t)_brute_popcountll(~(a64[i] ^ b64[i]));
    }

    // Tail: up to 7 bytes not covered by uint64 words above.  Assemble into a
    // uint64, mask off the bits that don't exist in the byte stream, then
    // popcnt.  K_eff accounts for the pad bits within the last T-word, so we
    // must count those pad bits here too — i.e. no extra masking for pad bits,
    // only for the high bytes beyond tail that are absent from the stream.
    const size_t tail = n_bytes - n_words * sizeof(uint64_t);
    if (tail) {
        const uint8_t* ap = reinterpret_cast<const uint8_t*>(a_row) + n_words * 8;
        const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_row) + n_words * 8;
        uint64_t la = 0, lb = 0;
        for (size_t k = 0; k < tail; ++k) {
            la |= ((uint64_t)ap[k]) << (k * 8);
            lb |= ((uint64_t)bp[k]) << (k * 8);
        }
        const uint64_t mask = (tail < 8) ? ((uint64_t(1) << (tail * 8)) - 1)
                                         : ~uint64_t(0);
        sum += (uint64_t)_brute_popcountll(~(la ^ lb) & mask);
    }
    return 2 * (int32_t)sum - K_eff;
}

// One row of C: c_row[0..N) = XnorPopcount(a_row, b_rows[n], …) for each n.
template <typename T>
HWY_ATTR inline void XnorPopcountRow(const T* HWY_RESTRICT a_row,
                                     const T* HWY_RESTRICT b_rows,
                                     int64_t Kp, int64_t N, int32_t K_eff,
                                     int32_t* HWY_RESTRICT c_row) {
    for (int64_t n = 0; n < N; ++n) {
        c_row[n] = XnorPopcountPair<T>(a_row, b_rows + n * Kp, Kp, K_eff);
    }
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
