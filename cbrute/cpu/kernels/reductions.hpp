// Highway-driven reductions for bit1 (packed) buffers.
//
//   AnyBitsSetWords  : OR-reduce; true iff any bit is 1.
//   AllBitsSetWords  : AND-reduce with last-word pad mask; true iff every
//                      logical bit is 1 (pad bits are forced to 1 before AND).
//   BuffersEqualWords: bitwise equality between two packed buffers.

#pragma once

#include <hwy/highway.h>
#include <cstddef>
#include <cstdint>

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR inline bool AnyBitsSetWords(const T* HWY_RESTRICT in, size_t n_words) {
    if (n_words == 0) return false;
    const hn::ScalableTag<T> d;
    const size_t LANES = hn::Lanes(d);
    auto acc = hn::Zero(d);
    size_t i = 0;
    for (; i + LANES <= n_words; i += LANES) {
        acc = hn::Or(acc, hn::LoadU(d, in + i));
    }
    if (!hn::AllTrue(d, hn::Eq(acc, hn::Zero(d)))) return true;
    for (; i < n_words; ++i) if (in[i]) return true;
    return false;
}

template <typename T>
HWY_ATTR inline bool AllBitsSetWords(const T* HWY_RESTRICT in, size_t n_words,
                                     T last_word_pad_mask) {
    // `last_word_pad_mask` has 1s in pad bit positions of the last word (so we
    // OR them in before AND-reducing). Zero if last word is fully used.
    if (n_words == 0) return true;
    const hn::ScalableTag<T> d;
    const size_t LANES = hn::Lanes(d);
    const T all_ones = (T)~(T)0;
    auto acc = hn::Set(d, all_ones);

    // Vectorize over the first (n_words - 1) words; handle the last word in
    // scalar so we can apply the pad mask without branching inside the SIMD
    // body.
    const size_t scan_n = n_words - 1;
    size_t i = 0;
    for (; i + LANES <= scan_n; i += LANES) {
        acc = hn::And(acc, hn::LoadU(d, in + i));
    }
    // Vector reduce: every lane must be all-ones, else there's a 0 bit.
    if (!hn::AllTrue(d, hn::Eq(acc, hn::Set(d, all_ones)))) return false;

    T scalar_acc = all_ones;
    for (; i < scan_n; ++i) scalar_acc &= in[i];
    scalar_acc &= (T)(in[n_words - 1] | last_word_pad_mask);
    return scalar_acc == all_ones;
}

template <typename T>
HWY_ATTR inline bool BuffersEqualWords(const T* HWY_RESTRICT a,
                                       const T* HWY_RESTRICT b, size_t n_words) {
    const hn::ScalableTag<T> d;
    const size_t LANES = hn::Lanes(d);
    size_t i = 0;
    for (; i + LANES <= n_words; i += LANES) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        if (!hn::AllTrue(d, hn::Eq(va, vb))) return false;
    }
    for (; i < n_words; ++i) if (a[i] != b[i]) return false;
    return true;
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
