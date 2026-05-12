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

template <typename T>
HWY_ATTR inline int32_t XnorPopcountPair(const T* HWY_RESTRICT a_row,
                                         const T* HWY_RESTRICT b_row,
                                         int64_t Kp, int32_t K_eff) {
    const hn::ScalableTag<T> d;
    const size_t LANES = hn::Lanes(d);
    auto acc = hn::Zero(d);
    int64_t k = 0;
    for (; k + (int64_t)LANES <= Kp; k += (int64_t)LANES) {
        auto va = hn::LoadU(d, a_row + k);
        auto vb = hn::LoadU(d, b_row + k);
        auto x  = hn::Not(hn::Xor(va, vb));
        acc = hn::Add(acc, hn::PopulationCount(x));
    }
    uint64_t sum = (uint64_t)hn::ReduceSum(d, acc);
    for (; k < Kp; ++k) {
        const T x = (T)~(a_row[k] ^ b_row[k]);
        sum += (uint64_t)_brute_popcountll((uint64_t)x);
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
