// Highway-driven XNOR-popcount matmul with register-blocked microkernel.
//
// A: (M, Kp), B: (N, Kp). Both packed; same pack width T (∈ {uint8,uint32,uint64}).
// K = logical last dim. Pad bits in the last word are zero.
//
// Output semantic (matches CPU/CUDA/MPS):
//   C(m, n) = K_logical − 2 * popcount_xor(A[m], B[n])
//           = 2 * popcount_xnor(A[m], B[n]) − K_eff
//   where K_eff = 2 * Kp * pw − K_logical.
//
// We compute popcount_xor (saves one `Not` per K-step) and apply `K - 2H` at
// the end. Pad bits XOR to zero, so they contribute 0 to H — no correction
// required besides the final `K - 2H`.
//
// Performance strategy:
//   * Reinterpret as uint64 stream (correctness: bitwise XOR commutes with
//     re-chunking; uint64 accumulator avoids any 8/16-bit overflow).
//   * Mr × Nr register-blocked microkernel (Mr=4, Nr=8 → 32 output cells per
//     inner-K iteration). 8 vector loads → 32 popcounts amortised. Quadruples
//     arithmetic intensity vs the legacy single-cell loop.
//   * Vector accumulators stay in registers across the K loop. ReduceSum drains
//     each accumulator only once at end.
//   * Software prefetching: pull B-tile cache lines ahead of the K cursor.
//   * Tail in M / N handled by smaller microkernels (4×4, 1×N, M×1).
//   * Outer parallelisation + N-tile cache blocking handled by the caller.

#pragma once

#include <hwy/highway.h>
#include "popcount.hpp"
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
  #define BRUTE_PREFETCH(p) __builtin_prefetch((const void*)(p))
#else
  #define BRUTE_PREFETCH(p) ((void)(p))
#endif

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ────────────────────────────────────────────────────────────────────────
// Single (m, n) cell — used for the M and N tails of the microkernel grid.
// XOR-popcount over packed uint64 words.
// ────────────────────────────────────────────────────────────────────────
template <typename T>
HWY_ATTR inline int32_t XorPopcountPair(const T* HWY_RESTRICT a_row,
                                        const T* HWY_RESTRICT b_row,
                                        int64_t Kp, int32_t K_logical) {
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
        acc = hn::Add(acc, hn::PopulationCount(hn::Xor(va, vb)));
    }
    uint64_t sum = hn::ReduceSum(d, acc);
    for (; i < n_words; ++i) {
        sum += (uint64_t)_brute_popcountll(a64[i] ^ b64[i]);
    }
    // Tail bytes (< 8) — assemble masked uint64.
    const size_t tail = n_bytes - n_words * sizeof(uint64_t);
    if (tail) {
        const uint8_t* ap = reinterpret_cast<const uint8_t*>(a_row) + n_words * 8;
        const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_row) + n_words * 8;
        uint64_t la = 0, lb = 0;
        for (size_t k = 0; k < tail; ++k) {
            la |= ((uint64_t)ap[k]) << (k * 8);
            lb |= ((uint64_t)bp[k]) << (k * 8);
        }
        sum += (uint64_t)_brute_popcountll(la ^ lb);
    }
    // C = K - 2*H. Pad bits XOR to 0, so H_padded == H_logical.
    return K_logical - 2 * (int32_t)sum;
}

// ────────────────────────────────────────────────────────────────────────
// 4×4 register-blocked microkernel (16 accumulators).
// Used as the inner-N tail for the 4×8 driver and as the baseline path
// for ABIs with only 16 vector registers (AVX2 / SSE / RVV-128).
//
// For each K-step:
//   load 4 A words + 4 B words = 8 loads
//   16 (XOR + popcount + add) operations into 16 accumulators
//
// Arithmetic intensity per byte loaded:
//   16 popcount-adds / (8 * sizeof(uint64) loads) = 2 ops/byte
// ────────────────────────────────────────────────────────────────────────
template <typename T>
HWY_ATTR inline void XorPopcountTile_4x4(const T* HWY_RESTRICT a_rows[4],
                                         const T* HWY_RESTRICT b_rows[4],
                                         int64_t Kp, int32_t K_logical,
                                         int32_t* HWY_RESTRICT C, int64_t ldC) {
    const size_t n_bytes = (size_t)Kp * sizeof(T);
    const size_t n_words = n_bytes / sizeof(uint64_t);

    const uint64_t* a0 = reinterpret_cast<const uint64_t*>(a_rows[0]);
    const uint64_t* a1 = reinterpret_cast<const uint64_t*>(a_rows[1]);
    const uint64_t* a2 = reinterpret_cast<const uint64_t*>(a_rows[2]);
    const uint64_t* a3 = reinterpret_cast<const uint64_t*>(a_rows[3]);
    const uint64_t* b0 = reinterpret_cast<const uint64_t*>(b_rows[0]);
    const uint64_t* b1 = reinterpret_cast<const uint64_t*>(b_rows[1]);
    const uint64_t* b2 = reinterpret_cast<const uint64_t*>(b_rows[2]);
    const uint64_t* b3 = reinterpret_cast<const uint64_t*>(b_rows[3]);

    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);

    auto acc00 = hn::Zero(d), acc01 = hn::Zero(d),
         acc02 = hn::Zero(d), acc03 = hn::Zero(d),
         acc10 = hn::Zero(d), acc11 = hn::Zero(d),
         acc12 = hn::Zero(d), acc13 = hn::Zero(d),
         acc20 = hn::Zero(d), acc21 = hn::Zero(d),
         acc22 = hn::Zero(d), acc23 = hn::Zero(d),
         acc30 = hn::Zero(d), acc31 = hn::Zero(d),
         acc32 = hn::Zero(d), acc33 = hn::Zero(d);

    // Apple Silicon's hardware prefetcher handles sequential streams; manual
    // prefetch instructions just consume issue slots. (Re-enable PF_AHEAD if
    // profiling shows L1 miss rate on x86 builds.)
    size_t i = 0;
    for (; i + LANES <= n_words; i += LANES) {
        auto va0 = hn::LoadU(d, a0 + i);
        auto va1 = hn::LoadU(d, a1 + i);
        auto va2 = hn::LoadU(d, a2 + i);
        auto va3 = hn::LoadU(d, a3 + i);
        auto vb0 = hn::LoadU(d, b0 + i);
        auto vb1 = hn::LoadU(d, b1 + i);
        auto vb2 = hn::LoadU(d, b2 + i);
        auto vb3 = hn::LoadU(d, b3 + i);

        acc00 = hn::Add(acc00, hn::PopulationCount(hn::Xor(va0, vb0)));
        acc01 = hn::Add(acc01, hn::PopulationCount(hn::Xor(va0, vb1)));
        acc02 = hn::Add(acc02, hn::PopulationCount(hn::Xor(va0, vb2)));
        acc03 = hn::Add(acc03, hn::PopulationCount(hn::Xor(va0, vb3)));
        acc10 = hn::Add(acc10, hn::PopulationCount(hn::Xor(va1, vb0)));
        acc11 = hn::Add(acc11, hn::PopulationCount(hn::Xor(va1, vb1)));
        acc12 = hn::Add(acc12, hn::PopulationCount(hn::Xor(va1, vb2)));
        acc13 = hn::Add(acc13, hn::PopulationCount(hn::Xor(va1, vb3)));
        acc20 = hn::Add(acc20, hn::PopulationCount(hn::Xor(va2, vb0)));
        acc21 = hn::Add(acc21, hn::PopulationCount(hn::Xor(va2, vb1)));
        acc22 = hn::Add(acc22, hn::PopulationCount(hn::Xor(va2, vb2)));
        acc23 = hn::Add(acc23, hn::PopulationCount(hn::Xor(va2, vb3)));
        acc30 = hn::Add(acc30, hn::PopulationCount(hn::Xor(va3, vb0)));
        acc31 = hn::Add(acc31, hn::PopulationCount(hn::Xor(va3, vb1)));
        acc32 = hn::Add(acc32, hn::PopulationCount(hn::Xor(va3, vb2)));
        acc33 = hn::Add(acc33, hn::PopulationCount(hn::Xor(va3, vb3)));
    }

    uint64_t s00 = hn::ReduceSum(d, acc00), s01 = hn::ReduceSum(d, acc01);
    uint64_t s02 = hn::ReduceSum(d, acc02), s03 = hn::ReduceSum(d, acc03);
    uint64_t s10 = hn::ReduceSum(d, acc10), s11 = hn::ReduceSum(d, acc11);
    uint64_t s12 = hn::ReduceSum(d, acc12), s13 = hn::ReduceSum(d, acc13);
    uint64_t s20 = hn::ReduceSum(d, acc20), s21 = hn::ReduceSum(d, acc21);
    uint64_t s22 = hn::ReduceSum(d, acc22), s23 = hn::ReduceSum(d, acc23);
    uint64_t s30 = hn::ReduceSum(d, acc30), s31 = hn::ReduceSum(d, acc31);
    uint64_t s32 = hn::ReduceSum(d, acc32), s33 = hn::ReduceSum(d, acc33);

    // Scalar tail words (< LANES words remaining).
    for (; i < n_words; ++i) {
        const uint64_t va0 = a0[i], va1 = a1[i], va2 = a2[i], va3 = a3[i];
        const uint64_t vb0 = b0[i], vb1 = b1[i], vb2 = b2[i], vb3 = b3[i];
        s00 += _brute_popcountll(va0 ^ vb0);
        s01 += _brute_popcountll(va0 ^ vb1);
        s02 += _brute_popcountll(va0 ^ vb2);
        s03 += _brute_popcountll(va0 ^ vb3);
        s10 += _brute_popcountll(va1 ^ vb0);
        s11 += _brute_popcountll(va1 ^ vb1);
        s12 += _brute_popcountll(va1 ^ vb2);
        s13 += _brute_popcountll(va1 ^ vb3);
        s20 += _brute_popcountll(va2 ^ vb0);
        s21 += _brute_popcountll(va2 ^ vb1);
        s22 += _brute_popcountll(va2 ^ vb2);
        s23 += _brute_popcountll(va2 ^ vb3);
        s30 += _brute_popcountll(va3 ^ vb0);
        s31 += _brute_popcountll(va3 ^ vb1);
        s32 += _brute_popcountll(va3 ^ vb2);
        s33 += _brute_popcountll(va3 ^ vb3);
    }
    // Trailing < 8 bytes: assemble masked uint64 from each row, popcount XOR.
    const size_t tail = n_bytes - n_words * sizeof(uint64_t);
    if (tail) {
        const uint8_t* ap0 = reinterpret_cast<const uint8_t*>(a_rows[0]) + n_words * 8;
        const uint8_t* ap1 = reinterpret_cast<const uint8_t*>(a_rows[1]) + n_words * 8;
        const uint8_t* ap2 = reinterpret_cast<const uint8_t*>(a_rows[2]) + n_words * 8;
        const uint8_t* ap3 = reinterpret_cast<const uint8_t*>(a_rows[3]) + n_words * 8;
        const uint8_t* bp0 = reinterpret_cast<const uint8_t*>(b_rows[0]) + n_words * 8;
        const uint8_t* bp1 = reinterpret_cast<const uint8_t*>(b_rows[1]) + n_words * 8;
        const uint8_t* bp2 = reinterpret_cast<const uint8_t*>(b_rows[2]) + n_words * 8;
        const uint8_t* bp3 = reinterpret_cast<const uint8_t*>(b_rows[3]) + n_words * 8;
        uint64_t la0 = 0, la1 = 0, la2 = 0, la3 = 0;
        uint64_t lb0 = 0, lb1 = 0, lb2 = 0, lb3 = 0;
        for (size_t k = 0; k < tail; ++k) {
            la0 |= ((uint64_t)ap0[k]) << (k * 8);
            la1 |= ((uint64_t)ap1[k]) << (k * 8);
            la2 |= ((uint64_t)ap2[k]) << (k * 8);
            la3 |= ((uint64_t)ap3[k]) << (k * 8);
            lb0 |= ((uint64_t)bp0[k]) << (k * 8);
            lb1 |= ((uint64_t)bp1[k]) << (k * 8);
            lb2 |= ((uint64_t)bp2[k]) << (k * 8);
            lb3 |= ((uint64_t)bp3[k]) << (k * 8);
        }
        s00 += _brute_popcountll(la0 ^ lb0);
        s01 += _brute_popcountll(la0 ^ lb1);
        s02 += _brute_popcountll(la0 ^ lb2);
        s03 += _brute_popcountll(la0 ^ lb3);
        s10 += _brute_popcountll(la1 ^ lb0);
        s11 += _brute_popcountll(la1 ^ lb1);
        s12 += _brute_popcountll(la1 ^ lb2);
        s13 += _brute_popcountll(la1 ^ lb3);
        s20 += _brute_popcountll(la2 ^ lb0);
        s21 += _brute_popcountll(la2 ^ lb1);
        s22 += _brute_popcountll(la2 ^ lb2);
        s23 += _brute_popcountll(la2 ^ lb3);
        s30 += _brute_popcountll(la3 ^ lb0);
        s31 += _brute_popcountll(la3 ^ lb1);
        s32 += _brute_popcountll(la3 ^ lb2);
        s33 += _brute_popcountll(la3 ^ lb3);
    }

    // C = K - 2H per cell.
    int32_t* HWY_RESTRICT row0 = C;
    int32_t* HWY_RESTRICT row1 = C + ldC;
    int32_t* HWY_RESTRICT row2 = C + 2 * ldC;
    int32_t* HWY_RESTRICT row3 = C + 3 * ldC;
    row0[0] = K_logical - 2 * (int32_t)s00;
    row0[1] = K_logical - 2 * (int32_t)s01;
    row0[2] = K_logical - 2 * (int32_t)s02;
    row0[3] = K_logical - 2 * (int32_t)s03;
    row1[0] = K_logical - 2 * (int32_t)s10;
    row1[1] = K_logical - 2 * (int32_t)s11;
    row1[2] = K_logical - 2 * (int32_t)s12;
    row1[3] = K_logical - 2 * (int32_t)s13;
    row2[0] = K_logical - 2 * (int32_t)s20;
    row2[1] = K_logical - 2 * (int32_t)s21;
    row2[2] = K_logical - 2 * (int32_t)s22;
    row2[3] = K_logical - 2 * (int32_t)s23;
    row3[0] = K_logical - 2 * (int32_t)s30;
    row3[1] = K_logical - 2 * (int32_t)s31;
    row3[2] = K_logical - 2 * (int32_t)s32;
    row3[3] = K_logical - 2 * (int32_t)s33;
}

// ────────────────────────────────────────────────────────────────────────
// 4×8 register-blocked microkernel (32 accumulators).
//
// Targets ISAs with 32 vector registers (AArch64 NEON, AVX-512). Doubles
// arithmetic intensity vs 4×4: 32 popcount-adds per K-step against only 12
// vector loads (4 A + 8 B).
//
// Per byte loaded: 32 ops / (12 × sizeof(uint64)) = 0.33 ops/byte (~33% over
// the 0.25 ops/byte of 4×4). This matters because Apple Silicon NEON popcount
// is throughput-bound on long-K workloads; we want the most outputs we can
// extract per K-tick.
// ────────────────────────────────────────────────────────────────────────
template <typename T>
HWY_ATTR inline void XorPopcountTile_4x8(const T* HWY_RESTRICT a_rows[4],
                                         const T* HWY_RESTRICT b_rows[8],
                                         int64_t Kp, int32_t K_logical,
                                         int32_t* HWY_RESTRICT C, int64_t ldC) {
    const size_t n_bytes = (size_t)Kp * sizeof(T);
    const size_t n_words = n_bytes / sizeof(uint64_t);

    const uint64_t* a0 = reinterpret_cast<const uint64_t*>(a_rows[0]);
    const uint64_t* a1 = reinterpret_cast<const uint64_t*>(a_rows[1]);
    const uint64_t* a2 = reinterpret_cast<const uint64_t*>(a_rows[2]);
    const uint64_t* a3 = reinterpret_cast<const uint64_t*>(a_rows[3]);
    const uint64_t* b0 = reinterpret_cast<const uint64_t*>(b_rows[0]);
    const uint64_t* b1 = reinterpret_cast<const uint64_t*>(b_rows[1]);
    const uint64_t* b2 = reinterpret_cast<const uint64_t*>(b_rows[2]);
    const uint64_t* b3 = reinterpret_cast<const uint64_t*>(b_rows[3]);
    const uint64_t* b4 = reinterpret_cast<const uint64_t*>(b_rows[4]);
    const uint64_t* b5 = reinterpret_cast<const uint64_t*>(b_rows[5]);
    const uint64_t* b6 = reinterpret_cast<const uint64_t*>(b_rows[6]);
    const uint64_t* b7 = reinterpret_cast<const uint64_t*>(b_rows[7]);

    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);

    auto a00 = hn::Zero(d), a01 = hn::Zero(d), a02 = hn::Zero(d), a03 = hn::Zero(d);
    auto a04 = hn::Zero(d), a05 = hn::Zero(d), a06 = hn::Zero(d), a07 = hn::Zero(d);
    auto a10 = hn::Zero(d), a11 = hn::Zero(d), a12 = hn::Zero(d), a13 = hn::Zero(d);
    auto a14 = hn::Zero(d), a15 = hn::Zero(d), a16 = hn::Zero(d), a17 = hn::Zero(d);
    auto a20 = hn::Zero(d), a21 = hn::Zero(d), a22 = hn::Zero(d), a23 = hn::Zero(d);
    auto a24 = hn::Zero(d), a25 = hn::Zero(d), a26 = hn::Zero(d), a27 = hn::Zero(d);
    auto a30 = hn::Zero(d), a31 = hn::Zero(d), a32 = hn::Zero(d), a33 = hn::Zero(d);
    auto a34 = hn::Zero(d), a35 = hn::Zero(d), a36 = hn::Zero(d), a37 = hn::Zero(d);

    size_t i = 0;
    for (; i + LANES <= n_words; i += LANES) {
        auto va0 = hn::LoadU(d, a0 + i);
        auto va1 = hn::LoadU(d, a1 + i);
        auto va2 = hn::LoadU(d, a2 + i);
        auto va3 = hn::LoadU(d, a3 + i);
        auto vb0 = hn::LoadU(d, b0 + i);
        auto vb1 = hn::LoadU(d, b1 + i);
        auto vb2 = hn::LoadU(d, b2 + i);
        auto vb3 = hn::LoadU(d, b3 + i);
        auto vb4 = hn::LoadU(d, b4 + i);
        auto vb5 = hn::LoadU(d, b5 + i);
        auto vb6 = hn::LoadU(d, b6 + i);
        auto vb7 = hn::LoadU(d, b7 + i);

        a00 = hn::Add(a00, hn::PopulationCount(hn::Xor(va0, vb0)));
        a01 = hn::Add(a01, hn::PopulationCount(hn::Xor(va0, vb1)));
        a02 = hn::Add(a02, hn::PopulationCount(hn::Xor(va0, vb2)));
        a03 = hn::Add(a03, hn::PopulationCount(hn::Xor(va0, vb3)));
        a04 = hn::Add(a04, hn::PopulationCount(hn::Xor(va0, vb4)));
        a05 = hn::Add(a05, hn::PopulationCount(hn::Xor(va0, vb5)));
        a06 = hn::Add(a06, hn::PopulationCount(hn::Xor(va0, vb6)));
        a07 = hn::Add(a07, hn::PopulationCount(hn::Xor(va0, vb7)));

        a10 = hn::Add(a10, hn::PopulationCount(hn::Xor(va1, vb0)));
        a11 = hn::Add(a11, hn::PopulationCount(hn::Xor(va1, vb1)));
        a12 = hn::Add(a12, hn::PopulationCount(hn::Xor(va1, vb2)));
        a13 = hn::Add(a13, hn::PopulationCount(hn::Xor(va1, vb3)));
        a14 = hn::Add(a14, hn::PopulationCount(hn::Xor(va1, vb4)));
        a15 = hn::Add(a15, hn::PopulationCount(hn::Xor(va1, vb5)));
        a16 = hn::Add(a16, hn::PopulationCount(hn::Xor(va1, vb6)));
        a17 = hn::Add(a17, hn::PopulationCount(hn::Xor(va1, vb7)));

        a20 = hn::Add(a20, hn::PopulationCount(hn::Xor(va2, vb0)));
        a21 = hn::Add(a21, hn::PopulationCount(hn::Xor(va2, vb1)));
        a22 = hn::Add(a22, hn::PopulationCount(hn::Xor(va2, vb2)));
        a23 = hn::Add(a23, hn::PopulationCount(hn::Xor(va2, vb3)));
        a24 = hn::Add(a24, hn::PopulationCount(hn::Xor(va2, vb4)));
        a25 = hn::Add(a25, hn::PopulationCount(hn::Xor(va2, vb5)));
        a26 = hn::Add(a26, hn::PopulationCount(hn::Xor(va2, vb6)));
        a27 = hn::Add(a27, hn::PopulationCount(hn::Xor(va2, vb7)));

        a30 = hn::Add(a30, hn::PopulationCount(hn::Xor(va3, vb0)));
        a31 = hn::Add(a31, hn::PopulationCount(hn::Xor(va3, vb1)));
        a32 = hn::Add(a32, hn::PopulationCount(hn::Xor(va3, vb2)));
        a33 = hn::Add(a33, hn::PopulationCount(hn::Xor(va3, vb3)));
        a34 = hn::Add(a34, hn::PopulationCount(hn::Xor(va3, vb4)));
        a35 = hn::Add(a35, hn::PopulationCount(hn::Xor(va3, vb5)));
        a36 = hn::Add(a36, hn::PopulationCount(hn::Xor(va3, vb6)));
        a37 = hn::Add(a37, hn::PopulationCount(hn::Xor(va3, vb7)));
    }

    uint64_t s00 = hn::ReduceSum(d, a00), s01 = hn::ReduceSum(d, a01);
    uint64_t s02 = hn::ReduceSum(d, a02), s03 = hn::ReduceSum(d, a03);
    uint64_t s04 = hn::ReduceSum(d, a04), s05 = hn::ReduceSum(d, a05);
    uint64_t s06 = hn::ReduceSum(d, a06), s07 = hn::ReduceSum(d, a07);
    uint64_t s10 = hn::ReduceSum(d, a10), s11 = hn::ReduceSum(d, a11);
    uint64_t s12 = hn::ReduceSum(d, a12), s13 = hn::ReduceSum(d, a13);
    uint64_t s14 = hn::ReduceSum(d, a14), s15 = hn::ReduceSum(d, a15);
    uint64_t s16 = hn::ReduceSum(d, a16), s17 = hn::ReduceSum(d, a17);
    uint64_t s20 = hn::ReduceSum(d, a20), s21 = hn::ReduceSum(d, a21);
    uint64_t s22 = hn::ReduceSum(d, a22), s23 = hn::ReduceSum(d, a23);
    uint64_t s24 = hn::ReduceSum(d, a24), s25 = hn::ReduceSum(d, a25);
    uint64_t s26 = hn::ReduceSum(d, a26), s27 = hn::ReduceSum(d, a27);
    uint64_t s30 = hn::ReduceSum(d, a30), s31 = hn::ReduceSum(d, a31);
    uint64_t s32 = hn::ReduceSum(d, a32), s33 = hn::ReduceSum(d, a33);
    uint64_t s34 = hn::ReduceSum(d, a34), s35 = hn::ReduceSum(d, a35);
    uint64_t s36 = hn::ReduceSum(d, a36), s37 = hn::ReduceSum(d, a37);

    // Scalar tail words (< LANES words remaining).
    for (; i < n_words; ++i) {
        const uint64_t va0 = a0[i], va1 = a1[i], va2 = a2[i], va3 = a3[i];
        const uint64_t vb0 = b0[i], vb1 = b1[i], vb2 = b2[i], vb3 = b3[i];
        const uint64_t vb4 = b4[i], vb5 = b5[i], vb6 = b6[i], vb7 = b7[i];
        s00 += _brute_popcountll(va0 ^ vb0); s01 += _brute_popcountll(va0 ^ vb1);
        s02 += _brute_popcountll(va0 ^ vb2); s03 += _brute_popcountll(va0 ^ vb3);
        s04 += _brute_popcountll(va0 ^ vb4); s05 += _brute_popcountll(va0 ^ vb5);
        s06 += _brute_popcountll(va0 ^ vb6); s07 += _brute_popcountll(va0 ^ vb7);
        s10 += _brute_popcountll(va1 ^ vb0); s11 += _brute_popcountll(va1 ^ vb1);
        s12 += _brute_popcountll(va1 ^ vb2); s13 += _brute_popcountll(va1 ^ vb3);
        s14 += _brute_popcountll(va1 ^ vb4); s15 += _brute_popcountll(va1 ^ vb5);
        s16 += _brute_popcountll(va1 ^ vb6); s17 += _brute_popcountll(va1 ^ vb7);
        s20 += _brute_popcountll(va2 ^ vb0); s21 += _brute_popcountll(va2 ^ vb1);
        s22 += _brute_popcountll(va2 ^ vb2); s23 += _brute_popcountll(va2 ^ vb3);
        s24 += _brute_popcountll(va2 ^ vb4); s25 += _brute_popcountll(va2 ^ vb5);
        s26 += _brute_popcountll(va2 ^ vb6); s27 += _brute_popcountll(va2 ^ vb7);
        s30 += _brute_popcountll(va3 ^ vb0); s31 += _brute_popcountll(va3 ^ vb1);
        s32 += _brute_popcountll(va3 ^ vb2); s33 += _brute_popcountll(va3 ^ vb3);
        s34 += _brute_popcountll(va3 ^ vb4); s35 += _brute_popcountll(va3 ^ vb5);
        s36 += _brute_popcountll(va3 ^ vb6); s37 += _brute_popcountll(va3 ^ vb7);
    }
    // Trailing < 8 bytes: assemble masked uint64 from each row, popcount XOR.
    const size_t tail = n_bytes - n_words * sizeof(uint64_t);
    if (tail) {
        auto load_tail = [&](const T* row) -> uint64_t {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(row) + n_words * 8;
            uint64_t v = 0;
            for (size_t k = 0; k < tail; ++k) v |= ((uint64_t)p[k]) << (k * 8);
            return v;
        };
        const uint64_t la0 = load_tail(a_rows[0]);
        const uint64_t la1 = load_tail(a_rows[1]);
        const uint64_t la2 = load_tail(a_rows[2]);
        const uint64_t la3 = load_tail(a_rows[3]);
        const uint64_t lb0 = load_tail(b_rows[0]);
        const uint64_t lb1 = load_tail(b_rows[1]);
        const uint64_t lb2 = load_tail(b_rows[2]);
        const uint64_t lb3 = load_tail(b_rows[3]);
        const uint64_t lb4 = load_tail(b_rows[4]);
        const uint64_t lb5 = load_tail(b_rows[5]);
        const uint64_t lb6 = load_tail(b_rows[6]);
        const uint64_t lb7 = load_tail(b_rows[7]);
        s00 += _brute_popcountll(la0 ^ lb0); s01 += _brute_popcountll(la0 ^ lb1);
        s02 += _brute_popcountll(la0 ^ lb2); s03 += _brute_popcountll(la0 ^ lb3);
        s04 += _brute_popcountll(la0 ^ lb4); s05 += _brute_popcountll(la0 ^ lb5);
        s06 += _brute_popcountll(la0 ^ lb6); s07 += _brute_popcountll(la0 ^ lb7);
        s10 += _brute_popcountll(la1 ^ lb0); s11 += _brute_popcountll(la1 ^ lb1);
        s12 += _brute_popcountll(la1 ^ lb2); s13 += _brute_popcountll(la1 ^ lb3);
        s14 += _brute_popcountll(la1 ^ lb4); s15 += _brute_popcountll(la1 ^ lb5);
        s16 += _brute_popcountll(la1 ^ lb6); s17 += _brute_popcountll(la1 ^ lb7);
        s20 += _brute_popcountll(la2 ^ lb0); s21 += _brute_popcountll(la2 ^ lb1);
        s22 += _brute_popcountll(la2 ^ lb2); s23 += _brute_popcountll(la2 ^ lb3);
        s24 += _brute_popcountll(la2 ^ lb4); s25 += _brute_popcountll(la2 ^ lb5);
        s26 += _brute_popcountll(la2 ^ lb6); s27 += _brute_popcountll(la2 ^ lb7);
        s30 += _brute_popcountll(la3 ^ lb0); s31 += _brute_popcountll(la3 ^ lb1);
        s32 += _brute_popcountll(la3 ^ lb2); s33 += _brute_popcountll(la3 ^ lb3);
        s34 += _brute_popcountll(la3 ^ lb4); s35 += _brute_popcountll(la3 ^ lb5);
        s36 += _brute_popcountll(la3 ^ lb6); s37 += _brute_popcountll(la3 ^ lb7);
    }

    int32_t* HWY_RESTRICT row0 = C;
    int32_t* HWY_RESTRICT row1 = C + ldC;
    int32_t* HWY_RESTRICT row2 = C + 2 * ldC;
    int32_t* HWY_RESTRICT row3 = C + 3 * ldC;
    row0[0] = K_logical - 2 * (int32_t)s00; row0[1] = K_logical - 2 * (int32_t)s01;
    row0[2] = K_logical - 2 * (int32_t)s02; row0[3] = K_logical - 2 * (int32_t)s03;
    row0[4] = K_logical - 2 * (int32_t)s04; row0[5] = K_logical - 2 * (int32_t)s05;
    row0[6] = K_logical - 2 * (int32_t)s06; row0[7] = K_logical - 2 * (int32_t)s07;
    row1[0] = K_logical - 2 * (int32_t)s10; row1[1] = K_logical - 2 * (int32_t)s11;
    row1[2] = K_logical - 2 * (int32_t)s12; row1[3] = K_logical - 2 * (int32_t)s13;
    row1[4] = K_logical - 2 * (int32_t)s14; row1[5] = K_logical - 2 * (int32_t)s15;
    row1[6] = K_logical - 2 * (int32_t)s16; row1[7] = K_logical - 2 * (int32_t)s17;
    row2[0] = K_logical - 2 * (int32_t)s20; row2[1] = K_logical - 2 * (int32_t)s21;
    row2[2] = K_logical - 2 * (int32_t)s22; row2[3] = K_logical - 2 * (int32_t)s23;
    row2[4] = K_logical - 2 * (int32_t)s24; row2[5] = K_logical - 2 * (int32_t)s25;
    row2[6] = K_logical - 2 * (int32_t)s26; row2[7] = K_logical - 2 * (int32_t)s27;
    row3[0] = K_logical - 2 * (int32_t)s30; row3[1] = K_logical - 2 * (int32_t)s31;
    row3[2] = K_logical - 2 * (int32_t)s32; row3[3] = K_logical - 2 * (int32_t)s33;
    row3[4] = K_logical - 2 * (int32_t)s34; row3[5] = K_logical - 2 * (int32_t)s35;
    row3[6] = K_logical - 2 * (int32_t)s36; row3[7] = K_logical - 2 * (int32_t)s37;
}

// ────────────────────────────────────────────────────────────────────────
// Driver: compute one row of C against an N-tile of B.
// Selects the 4×4 microkernel for groups of 4 A-rows × 4 B-rows; falls back
// to single-cell `XorPopcountPair` for tail rows/cols.
//
// Layout:
//   a_row0..a_row3 : Mr=4 contiguous A rows. Caller batches in 4s.
//   b_rows         : pointer to first B row in this N-tile.
//   N_tile         : number of B rows in the tile (≤ N).
//   c_block        : output rows for these 4 A-rows, ldC=N (full output stride).
// ────────────────────────────────────────────────────────────────────────
template <typename T>
HWY_ATTR inline void XorPopcountBlock_4xN(const T* HWY_RESTRICT a_row0,
                                          const T* HWY_RESTRICT a_row1,
                                          const T* HWY_RESTRICT a_row2,
                                          const T* HWY_RESTRICT a_row3,
                                          const T* HWY_RESTRICT b_rows,
                                          int64_t Kp, int64_t N_tile,
                                          int32_t K_logical,
                                          int32_t* HWY_RESTRICT c_block,
                                          int64_t ldC) {
    const T* a_arr[4] = {a_row0, a_row1, a_row2, a_row3};
    int64_t n = 0;
    // Wide pass: 4×8 microkernel (32 accumulators) on ISAs with 32 vector regs.
    for (; n + 8 <= N_tile; n += 8) {
        const T* b_arr[8] = {
            b_rows + (n + 0) * Kp, b_rows + (n + 1) * Kp,
            b_rows + (n + 2) * Kp, b_rows + (n + 3) * Kp,
            b_rows + (n + 4) * Kp, b_rows + (n + 5) * Kp,
            b_rows + (n + 6) * Kp, b_rows + (n + 7) * Kp,
        };
        XorPopcountTile_4x8<T>(a_arr, b_arr, Kp, K_logical, c_block + n, ldC);
    }
    for (; n + 4 <= N_tile; n += 4) {
        const T* b_arr[4] = {
            b_rows + (n + 0) * Kp,
            b_rows + (n + 1) * Kp,
            b_rows + (n + 2) * Kp,
            b_rows + (n + 3) * Kp,
        };
        XorPopcountTile_4x4<T>(a_arr, b_arr, Kp, K_logical, c_block + n, ldC);
    }
    // Tail N (< 4 columns) — single-cell loop, all 4 A-rows.
    for (; n < N_tile; ++n) {
        const T* b_row = b_rows + n * Kp;
        c_block[0 * ldC + n] = XorPopcountPair<T>(a_row0, b_row, Kp, K_logical);
        c_block[1 * ldC + n] = XorPopcountPair<T>(a_row1, b_row, Kp, K_logical);
        c_block[2 * ldC + n] = XorPopcountPair<T>(a_row2, b_row, Kp, K_logical);
        c_block[3 * ldC + n] = XorPopcountPair<T>(a_row3, b_row, Kp, K_logical);
    }
}

// Single-row driver (used for M tail).
template <typename T>
HWY_ATTR inline void XorPopcountRow(const T* HWY_RESTRICT a_row,
                                    const T* HWY_RESTRICT b_rows,
                                    int64_t Kp, int64_t N_tile, int32_t K_logical,
                                    int32_t* HWY_RESTRICT c_row) {
    for (int64_t n = 0; n < N_tile; ++n) {
        c_row[n] = XorPopcountPair<T>(a_row, b_rows + n * Kp, Kp, K_logical);
    }
}

// ────────────────────────────────────────────────────────────────────────
// Legacy XnorPopcountPair / XnorPopcountRow — kept for backward compatibility
// with any caller that still passes K_eff. New code should call the Xor*
// variants directly with K_logical.
// ────────────────────────────────────────────────────────────────────────
template <typename T>
HWY_ATTR inline int32_t XnorPopcountPair(const T* HWY_RESTRICT a_row,
                                         const T* HWY_RESTRICT b_row,
                                         int64_t Kp, int32_t K_eff) {
    // K_eff = 2*Kp*pw - K_logical → K_logical = 2*Kp*pw - K_eff.
    const int32_t K_logical = (int32_t)(2LL * (int64_t)Kp * (int64_t)(sizeof(T) * 8))
                               - K_eff;
    return XorPopcountPair<T>(a_row, b_row, Kp, K_logical);
}

template <typename T>
HWY_ATTR inline void XnorPopcountRow(const T* HWY_RESTRICT a_row,
                                     const T* HWY_RESTRICT b_rows,
                                     int64_t Kp, int64_t N, int32_t K_eff,
                                     int32_t* HWY_RESTRICT c_row) {
    const int32_t K_logical = (int32_t)(2LL * (int64_t)Kp * (int64_t)(sizeof(T) * 8))
                               - K_eff;
    XorPopcountRow<T>(a_row, b_rows, Kp, N, K_logical, c_row);
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
