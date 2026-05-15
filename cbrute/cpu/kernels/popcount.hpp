// Highway-driven popcount kernels.
//
// Per-element popcount:   one word in -> int32 count out
// Bulk popcount:          sum over a contiguous byte range -> uint64
//
// Bulk path uses Harley-Seal CSA for n_words ≥ 16 (typically 1.5× faster than
// hardware popcount because carries are done in cheap bitwise ops, popcount is
// only applied to the carry tree). For shorter buffers it falls back to
// Highway's PopulationCount + ReduceSum.
//
// On x86-64 with AVX-512 BITALG/VPOPCNTDQ we add a hand-written specialisation
// using `_mm512_popcnt_epi64` (8×64-bit popcounts in ~3 cycles, fully pipelined).

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

#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__)
  #define BRUTE_HAVE_VPOPCNTDQ 1
  #include <immintrin.h>
#else
  #define BRUTE_HAVE_VPOPCNTDQ 0
#endif

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ────────────────────────────────────────────────────────────────────────
// Harley-Seal CSA — accumulate bit counts in a small carry tree, popcount
// only the tree at the end.
//
// Adds 16 input words → 4 carry registers (one each for 1-bit, 2-bit, 4-bit,
// 8-bit weights). Each carry only needs to be popcounted once, divided by its
// weight, and summed. Net effect: ~1 popcount per 16 input words instead of
// 1 per word, while spending O(input/word) cheap AND/XOR ops to maintain
// carries. ~1.5× faster than hardware popcount for n_words ≥ 16 and is what
// every modern bulk-popcount library (libpopcnt, hwy/contrib/algo) uses.
// ────────────────────────────────────────────────────────────────────────
HWY_ATTR inline void csa64(uint64_t& h, uint64_t& l,
                            uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}

HWY_ATTR inline uint64_t HarleySealPopcountWords(const uint64_t* HWY_RESTRICT in,
                                                 size_t n_words) {
    // Process 16 words at a time. Each iteration produces four carry words
    // (ones, twos, fours, eights) which stand for sum-of-popcount weighted
    // by 1, 2, 4, 8. Total contribution: 1*ones + 2*twos + 4*fours + 8*eights
    // popcounts at the end.
    uint64_t total  = 0;
    uint64_t ones   = 0;
    uint64_t twos   = 0;
    uint64_t fours  = 0;
    uint64_t eights = 0;
    uint64_t sixteens_acc = 0;

    size_t i = 0;
    for (; i + 16 <= n_words; i += 16) {
        uint64_t twosA, twosB, foursA, foursB, eightsA, eightsB;

        csa64(twosA, ones, ones, in[i + 0], in[i + 1]);
        csa64(twosB, ones, ones, in[i + 2], in[i + 3]);
        csa64(foursA, twos, twos, twosA, twosB);
        csa64(twosA, ones, ones, in[i + 4], in[i + 5]);
        csa64(twosB, ones, ones, in[i + 6], in[i + 7]);
        csa64(foursB, twos, twos, twosA, twosB);
        csa64(eightsA, fours, fours, foursA, foursB);
        csa64(twosA, ones, ones, in[i + 8],  in[i + 9]);
        csa64(twosB, ones, ones, in[i + 10], in[i + 11]);
        csa64(foursA, twos, twos, twosA, twosB);
        csa64(twosA, ones, ones, in[i + 12], in[i + 13]);
        csa64(twosB, ones, ones, in[i + 14], in[i + 15]);
        csa64(foursB, twos, twos, twosA, twosB);
        csa64(eightsB, fours, fours, foursA, foursB);
        csa64(sixteens_acc, eights, eights, eightsA, eightsB);

        total += 16ull * (uint64_t)_brute_popcountll(sixteens_acc);
    }

    total += 8ull * (uint64_t)_brute_popcountll(eights);
    total += 4ull * (uint64_t)_brute_popcountll(fours);
    total += 2ull * (uint64_t)_brute_popcountll(twos);
    total += 1ull * (uint64_t)_brute_popcountll(ones);
    for (; i < n_words; ++i) {
        total += (uint64_t)_brute_popcountll(in[i]);
    }
    return total;
}

// Harley-Seal variant that XORs two streams on the fly (Hamming distance).
HWY_ATTR inline uint64_t HarleySealPopcountXorWords(const uint64_t* HWY_RESTRICT a,
                                                     const uint64_t* HWY_RESTRICT b,
                                                     size_t n_words) {
    uint64_t total  = 0;
    uint64_t ones   = 0;
    uint64_t twos   = 0;
    uint64_t fours  = 0;
    uint64_t eights = 0;
    uint64_t sixteens_acc = 0;

    size_t i = 0;
    for (; i + 16 <= n_words; i += 16) {
        uint64_t twosA, twosB, foursA, foursB, eightsA, eightsB;
        uint64_t x0  = a[i + 0]  ^ b[i + 0];
        uint64_t x1  = a[i + 1]  ^ b[i + 1];
        uint64_t x2  = a[i + 2]  ^ b[i + 2];
        uint64_t x3  = a[i + 3]  ^ b[i + 3];
        uint64_t x4  = a[i + 4]  ^ b[i + 4];
        uint64_t x5  = a[i + 5]  ^ b[i + 5];
        uint64_t x6  = a[i + 6]  ^ b[i + 6];
        uint64_t x7  = a[i + 7]  ^ b[i + 7];
        uint64_t x8  = a[i + 8]  ^ b[i + 8];
        uint64_t x9  = a[i + 9]  ^ b[i + 9];
        uint64_t x10 = a[i + 10] ^ b[i + 10];
        uint64_t x11 = a[i + 11] ^ b[i + 11];
        uint64_t x12 = a[i + 12] ^ b[i + 12];
        uint64_t x13 = a[i + 13] ^ b[i + 13];
        uint64_t x14 = a[i + 14] ^ b[i + 14];
        uint64_t x15 = a[i + 15] ^ b[i + 15];

        csa64(twosA, ones, ones, x0,  x1);
        csa64(twosB, ones, ones, x2,  x3);
        csa64(foursA, twos, twos, twosA, twosB);
        csa64(twosA, ones, ones, x4,  x5);
        csa64(twosB, ones, ones, x6,  x7);
        csa64(foursB, twos, twos, twosA, twosB);
        csa64(eightsA, fours, fours, foursA, foursB);
        csa64(twosA, ones, ones, x8,  x9);
        csa64(twosB, ones, ones, x10, x11);
        csa64(foursA, twos, twos, twosA, twosB);
        csa64(twosA, ones, ones, x12, x13);
        csa64(twosB, ones, ones, x14, x15);
        csa64(foursB, twos, twos, twosA, twosB);
        csa64(eightsB, fours, fours, foursA, foursB);
        csa64(sixteens_acc, eights, eights, eightsA, eightsB);

        total += 16ull * (uint64_t)_brute_popcountll(sixteens_acc);
    }

    total += 8ull * (uint64_t)_brute_popcountll(eights);
    total += 4ull * (uint64_t)_brute_popcountll(fours);
    total += 2ull * (uint64_t)_brute_popcountll(twos);
    total += 1ull * (uint64_t)_brute_popcountll(ones);
    for (; i < n_words; ++i) {
        total += (uint64_t)_brute_popcountll(a[i] ^ b[i]);
    }
    return total;
}

#if BRUTE_HAVE_VPOPCNTDQ
// AVX-512 explicit path: 8×uint64 popcounts in 3 cycles. Use only on Ice Lake+
// with `__AVX512VPOPCNTDQ__`. Highway *should* lower to this on `HWY_AVX3_DL`,
// but in practice many Highway/GCC combos fall back to the LUT path; this
// guarantees the tensor-core-equivalent throughput.
HWY_ATTR inline uint64_t Vpopcnt512Words(const uint64_t* HWY_RESTRICT in,
                                          size_t n_words) {
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= n_words; i += 8) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(in + i));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    uint64_t total = 0;
    alignas(64) uint64_t buf[8];
    _mm512_store_si512((__m512i*)buf, acc);
    for (int k = 0; k < 8; ++k) total += buf[k];
    for (; i < n_words; ++i) total += (uint64_t)_brute_popcountll(in[i]);
    return total;
}

HWY_ATTR inline uint64_t Vpopcnt512XorWords(const uint64_t* HWY_RESTRICT a,
                                             const uint64_t* HWY_RESTRICT b,
                                             size_t n_words) {
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i vb = _mm512_loadu_si512((const __m512i*)(b + i));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
    }
    uint64_t total = 0;
    alignas(64) uint64_t buf[8];
    _mm512_store_si512((__m512i*)buf, acc);
    for (int k = 0; k < 8; ++k) total += buf[k];
    for (; i < n_words; ++i) total += (uint64_t)_brute_popcountll(a[i] ^ b[i]);
    return total;
}
#endif

//  Bulk popcount: total set-bit count over a contiguous byte range
HWY_ATTR inline uint64_t TotalPopcountBytes(const void* HWY_RESTRICT bytes,
                                            size_t n_bytes) {
    if (n_bytes == 0) return 0;
    const uint64_t* in = reinterpret_cast<const uint64_t*>(bytes);
    const size_t n_words = n_bytes / sizeof(uint64_t);

    uint64_t total = 0;
#if BRUTE_HAVE_VPOPCNTDQ
    // VPOPCNTDQ is the fastest path on Ice Lake+; Harley-Seal compounds the
    // win for >256 words but VPOPCNTDQ alone is ≥ HW POPCNT throughput.
    if (n_words >= 8) {
        total = Vpopcnt512Words(in, n_words);
    } else {
        for (size_t i = 0; i < n_words; ++i) {
            total += (uint64_t)_brute_popcountll(in[i]);
        }
    }
#else
    if (n_words >= 16) {
        total = HarleySealPopcountWords(in, n_words);
    } else {
        // Highway vector path for short buffers (< 16 words = < 128 bytes).
        const hn::ScalableTag<uint64_t> d;
        const size_t LANES = hn::Lanes(d);
        size_t i = 0;
        auto acc = hn::Zero(d);
        for (; i + LANES <= n_words; i += LANES) {
            acc = hn::Add(acc, hn::PopulationCount(hn::LoadU(d, in + i)));
        }
        total = hn::ReduceSum(d, acc);
        for (; i < n_words; ++i) total += (uint64_t)_brute_popcountll(in[i]);
    }
#endif

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

// Same, but XOR a then b on the fly (used by hamming_distance over packed buffers).
HWY_ATTR inline uint64_t TotalPopcountXor(const void* HWY_RESTRICT a_bytes,
                                          const void* HWY_RESTRICT b_bytes,
                                          size_t n_bytes) {
    if (n_bytes == 0) return 0;
    const uint64_t* a = reinterpret_cast<const uint64_t*>(a_bytes);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(b_bytes);
    const size_t n_words = n_bytes / sizeof(uint64_t);

    uint64_t total = 0;
#if BRUTE_HAVE_VPOPCNTDQ
    if (n_words >= 8) {
        total = Vpopcnt512XorWords(a, b, n_words);
    } else {
        for (size_t i = 0; i < n_words; ++i) {
            total += (uint64_t)_brute_popcountll(a[i] ^ b[i]);
        }
    }
#else
    if (n_words >= 16) {
        total = HarleySealPopcountXorWords(a, b, n_words);
    } else {
        const hn::ScalableTag<uint64_t> d;
        const size_t LANES = hn::Lanes(d);
        size_t i = 0;
        auto acc = hn::Zero(d);
        for (; i + LANES <= n_words; i += LANES) {
            auto x = hn::Xor(hn::LoadU(d, a + i), hn::LoadU(d, b + i));
            acc = hn::Add(acc, hn::PopulationCount(x));
        }
        total = hn::ReduceSum(d, acc);
        for (; i < n_words; ++i) total += (uint64_t)_brute_popcountll(a[i] ^ b[i]);
    }
#endif

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
// Vectorised via Highway when T is a Highway-supported lane type (uint8,
// uint32, uint64). `bool` is not a valid Highway lane type — we fall back to
// the scalar path for the kBool dispatch case via `if constexpr`.
template <typename T>
HWY_ATTR void PopcountPerWord(const T* HWY_RESTRICT in, int32_t* HWY_RESTRICT out,
                              size_t n) {
    if constexpr (std::is_same_v<T, bool>) {
        for (size_t i = 0; i < n; ++i) out[i] = in[i] ? 1 : 0;
    } else {
        const hn::ScalableTag<T> d_in;
        const size_t LANES_IN = hn::Lanes(d_in);
        size_t i = 0;
        for (; i + LANES_IN <= n; i += LANES_IN) {
            auto v = hn::LoadU(d_in, in + i);
            auto p = hn::PopulationCount(v);
            // Lanewise extract → int32. Compiler folds into direct register
            // reads on NEON/AVX2 for small lane counts.
            for (size_t j = 0; j < LANES_IN; ++j) {
                out[i + j] = (int32_t)hn::ExtractLane(p, j);
            }
        }
        for (; i < n; ++i) {
            out[i] = (int32_t)_brute_popcountll((uint64_t)in[i]);
        }
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
