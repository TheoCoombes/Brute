// Highway-driven popcount kernels.
//
// Per-element popcount:   one word in -> int32 count out
// Bulk popcount:          sum over a contiguous byte range -> uint64
//
// Both use Highway's PopulationCount (lane-wise) + ReduceSum. The bulk path is
// the canonical "in-register XNOR-popcount" building block.

#pragma once

#include <hwy/highway.h>
#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int _brute_popcountll(uint64_t x) { return (int)__popcnt64(x); }
#else
  static inline int _brute_popcountll(uint64_t x) { return __builtin_popcountll(x); }
#endif

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

//  Bulk popcount: total set-bit count over a contiguous byte range 
// Vector accumulator with ReduceSum at the end — single pass, no intermediate
// XOR/AND buffer required from the caller.
HWY_ATTR inline uint64_t TotalPopcountBytes(const void* HWY_RESTRICT bytes,
                                            size_t n_bytes) {
    if (n_bytes == 0) return 0;
    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);
    const uint64_t* in = reinterpret_cast<const uint64_t*>(bytes);

    const size_t n_words = n_bytes / sizeof(uint64_t);
    size_t i = 0;
    auto acc = hn::Zero(d);
    for (; i + LANES <= n_words; i += LANES) {
        acc = hn::Add(acc, hn::PopulationCount(hn::LoadU(d, in + i)));
    }
    uint64_t total = hn::ReduceSum(d, acc);
    for (; i < n_words; ++i) total += (uint64_t)_brute_popcountll(in[i]);

    // Trailing 1..7 bytes.
    const size_t tail = n_bytes - n_words * sizeof(uint64_t);
    if (tail) {
        uint64_t last = 0;
        const uint8_t* tail_p = reinterpret_cast<const uint8_t*>(bytes) +
                                n_words * sizeof(uint64_t);
        for (size_t b = 0; b < tail; ++b) last |= ((uint64_t)tail_p[b]) << (b * 8);
        total += (uint64_t)_brute_popcountll(last);
    }
    return total;
}

// Same, but XOR a then-b on the fly (used by hamming_distance over packed buffers).
HWY_ATTR inline uint64_t TotalPopcountXor(const void* HWY_RESTRICT a_bytes,
                                          const void* HWY_RESTRICT b_bytes,
                                          size_t n_bytes) {
    if (n_bytes == 0) return 0;
    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);
    const uint64_t* a = reinterpret_cast<const uint64_t*>(a_bytes);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(b_bytes);

    const size_t n_words = n_bytes / sizeof(uint64_t);
    size_t i = 0;
    auto acc = hn::Zero(d);
    for (; i + LANES <= n_words; i += LANES) {
        auto x = hn::Xor(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
        acc = hn::Add(acc, hn::PopulationCount(x));
    }
    uint64_t total = hn::ReduceSum(d, acc);
    for (; i < n_words; ++i) total += (uint64_t)_brute_popcountll(a[i] ^ b[i]);

    const size_t tail = n_bytes - n_words * sizeof(uint64_t);
    if (tail) {
        uint64_t la = 0, lb = 0;
        const uint8_t* ap = reinterpret_cast<const uint8_t*>(a_bytes) + n_words * sizeof(uint64_t);
        const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_bytes) + n_words * sizeof(uint64_t);
        for (size_t k = 0; k < tail; ++k) {
            la |= ((uint64_t)ap[k]) << (k * 8);
            lb |= ((uint64_t)bp[k]) << (k * 8);
        }
        total += (uint64_t)_brute_popcountll(la ^ lb);
    }
    return total;
}

//  Per-element popcount: input is T-typed words, output is int32 counts 
//
// We use scalar __builtin_popcountll per word. Hardware POPCNT / NEON CNT are
// single-cycle and the compiler vectorizes well; Highway's PopulationCount
// across mixed-width lanes (T-wide in, int32-wide out) needs PromoteTo/DemoteTo
// for every type which loses more than it gains here.
template <typename T>
HWY_ATTR void PopcountPerWord(const T* HWY_RESTRICT in, int32_t* HWY_RESTRICT out,
                              size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = (int32_t)_brute_popcountll((uint64_t)in[i]);
    }
}

//  Hamming weight check: returns true iff buffer has any set bit 
HWY_ATTR inline bool AnyBitsSetBytes(const void* HWY_RESTRICT bytes, size_t n_bytes) {
    if (n_bytes == 0) return false;
    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);
    const uint64_t* in = reinterpret_cast<const uint64_t*>(bytes);
    const size_t n_words = n_bytes / sizeof(uint64_t);
    size_t i = 0;
    auto acc = hn::Zero(d);
    for (; i + LANES <= n_words; i += LANES) {
        acc = hn::Or(acc, hn::LoadU(d, in + i));
    }
    if (!hn::AllTrue(d, hn::Eq(acc, hn::Zero(d)))) return true;
    for (; i < n_words; ++i) if (in[i]) return true;
    const size_t tail = n_bytes - n_words * sizeof(uint64_t);
    const uint8_t* tail_p = reinterpret_cast<const uint8_t*>(bytes) + n_words * sizeof(uint64_t);
    for (size_t b = 0; b < tail; ++b) if (tail_p[b]) return true;
    return false;
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
