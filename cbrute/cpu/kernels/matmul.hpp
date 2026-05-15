// Highway XNOR-popcount matmul for bit1 (uint64-packed) tensors.
//
//   A: (M, Kp) uint64 packed   B: (N, Kp) uint64 packed   →   C: (M, N) int32
//   Kp = ceil(K_logical / 64). Pad bits in the last word are zero.
//
//   C(m, n) = K_logical − 2 · popc_xor(A[m], B[n])
//
//   Pad bits XOR to zero so they contribute zero to popc_xor — no separate
//   correction step beyond the final `K - 2H`.
//
// Microkernel strategy: register-blocked Mr×Nr tile keeps vector accumulators
// hot across the K loop. ReduceSum drains each accumulator only once at end.
//   * 4×8 (32 accs, 12 loads / step) for ISAs with 32 vector regs (NEON, AVX-512)
//   * 4×4 (16 accs, 8 loads / step)  fallback for the inner N tail
//   * single-cell pair                for the inner N tail < 4 cols, M tail < 4 rows
//
// Outer parallelisation + N-tile cache blocking live in the driver.

#pragma once

#include <hwy/highway.h>
#include "popcount.hpp"
#include <cstddef>
#include <cstdint>

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

//  Single (m, n) cell — used for M and N tails of the microkernel grid.
HWY_ATTR inline int32_t XorPopcountPair(const uint64_t* HWY_RESTRICT a,
                                        const uint64_t* HWY_RESTRICT b,
                                        int64_t Kp, int32_t K_logical) {
    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);
    size_t i = 0;
    auto acc = hn::Zero(d);
    for (; i + LANES <= (size_t)Kp; i += LANES) {
        acc = hn::Add(acc, hn::PopulationCount(
            hn::Xor(hn::LoadU(d, a + i), hn::LoadU(d, b + i))));
    }
    uint64_t sum = hn::ReduceSum(d, acc);
    for (; i < (size_t)Kp; ++i) {
        sum += (uint64_t)_brute_popcountll(a[i] ^ b[i]);
    }
    return K_logical - 2 * (int32_t)sum;
}

//  4×4 register-blocked microkernel (16 accumulators).
//  Inner-N tail for the 4×8 driver / baseline path on 16-vreg ISAs.
HWY_ATTR inline void XorPopcountTile_4x4(const uint64_t* HWY_RESTRICT a_rows[4],
                                         const uint64_t* HWY_RESTRICT b_rows[4],
                                         int64_t Kp, int32_t K_logical,
                                         int32_t* HWY_RESTRICT C, int64_t ldC) {
    const uint64_t* a0 = a_rows[0]; const uint64_t* a1 = a_rows[1];
    const uint64_t* a2 = a_rows[2]; const uint64_t* a3 = a_rows[3];
    const uint64_t* b0 = b_rows[0]; const uint64_t* b1 = b_rows[1];
    const uint64_t* b2 = b_rows[2]; const uint64_t* b3 = b_rows[3];

    const hn::ScalableTag<uint64_t> d;
    const size_t LANES = hn::Lanes(d);

    auto acc00 = hn::Zero(d), acc01 = hn::Zero(d), acc02 = hn::Zero(d), acc03 = hn::Zero(d);
    auto acc10 = hn::Zero(d), acc11 = hn::Zero(d), acc12 = hn::Zero(d), acc13 = hn::Zero(d);
    auto acc20 = hn::Zero(d), acc21 = hn::Zero(d), acc22 = hn::Zero(d), acc23 = hn::Zero(d);
    auto acc30 = hn::Zero(d), acc31 = hn::Zero(d), acc32 = hn::Zero(d), acc33 = hn::Zero(d);

    size_t i = 0;
    for (; i + LANES <= (size_t)Kp; i += LANES) {
        const auto va0 = hn::LoadU(d, a0 + i);
        const auto va1 = hn::LoadU(d, a1 + i);
        const auto va2 = hn::LoadU(d, a2 + i);
        const auto va3 = hn::LoadU(d, a3 + i);
        const auto vb0 = hn::LoadU(d, b0 + i);
        const auto vb1 = hn::LoadU(d, b1 + i);
        const auto vb2 = hn::LoadU(d, b2 + i);
        const auto vb3 = hn::LoadU(d, b3 + i);
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

    // Scalar tail (< LANES words).
    for (; i < (size_t)Kp; ++i) {
        const uint64_t va0 = a0[i], va1 = a1[i], va2 = a2[i], va3 = a3[i];
        const uint64_t vb0 = b0[i], vb1 = b1[i], vb2 = b2[i], vb3 = b3[i];
        s00 += _brute_popcountll(va0 ^ vb0); s01 += _brute_popcountll(va0 ^ vb1);
        s02 += _brute_popcountll(va0 ^ vb2); s03 += _brute_popcountll(va0 ^ vb3);
        s10 += _brute_popcountll(va1 ^ vb0); s11 += _brute_popcountll(va1 ^ vb1);
        s12 += _brute_popcountll(va1 ^ vb2); s13 += _brute_popcountll(va1 ^ vb3);
        s20 += _brute_popcountll(va2 ^ vb0); s21 += _brute_popcountll(va2 ^ vb1);
        s22 += _brute_popcountll(va2 ^ vb2); s23 += _brute_popcountll(va2 ^ vb3);
        s30 += _brute_popcountll(va3 ^ vb0); s31 += _brute_popcountll(va3 ^ vb1);
        s32 += _brute_popcountll(va3 ^ vb2); s33 += _brute_popcountll(va3 ^ vb3);
    }

    int32_t* HWY_RESTRICT r0 = C;
    int32_t* HWY_RESTRICT r1 = C + ldC;
    int32_t* HWY_RESTRICT r2 = C + 2 * ldC;
    int32_t* HWY_RESTRICT r3 = C + 3 * ldC;
    r0[0] = K_logical - 2*(int32_t)s00; r0[1] = K_logical - 2*(int32_t)s01;
    r0[2] = K_logical - 2*(int32_t)s02; r0[3] = K_logical - 2*(int32_t)s03;
    r1[0] = K_logical - 2*(int32_t)s10; r1[1] = K_logical - 2*(int32_t)s11;
    r1[2] = K_logical - 2*(int32_t)s12; r1[3] = K_logical - 2*(int32_t)s13;
    r2[0] = K_logical - 2*(int32_t)s20; r2[1] = K_logical - 2*(int32_t)s21;
    r2[2] = K_logical - 2*(int32_t)s22; r2[3] = K_logical - 2*(int32_t)s23;
    r3[0] = K_logical - 2*(int32_t)s30; r3[1] = K_logical - 2*(int32_t)s31;
    r3[2] = K_logical - 2*(int32_t)s32; r3[3] = K_logical - 2*(int32_t)s33;
}

//  4×8 register-blocked microkernel (32 accumulators).
//  Primary path on ISAs with 32 vector regs (NEON, AVX-512). 4 A + 8 B loads
//  per K-step amortise across 32 popcount-adds, raising arithmetic intensity
//  to 0.33 ops/byte (vs. 0.25 ops/byte for 4×4).
HWY_ATTR inline void XorPopcountTile_4x8(const uint64_t* HWY_RESTRICT a_rows[4],
                                         const uint64_t* HWY_RESTRICT b_rows[8],
                                         int64_t Kp, int32_t K_logical,
                                         int32_t* HWY_RESTRICT C, int64_t ldC) {
    const uint64_t* a0 = a_rows[0]; const uint64_t* a1 = a_rows[1];
    const uint64_t* a2 = a_rows[2]; const uint64_t* a3 = a_rows[3];
    const uint64_t* b0 = b_rows[0]; const uint64_t* b1 = b_rows[1];
    const uint64_t* b2 = b_rows[2]; const uint64_t* b3 = b_rows[3];
    const uint64_t* b4 = b_rows[4]; const uint64_t* b5 = b_rows[5];
    const uint64_t* b6 = b_rows[6]; const uint64_t* b7 = b_rows[7];

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
    for (; i + LANES <= (size_t)Kp; i += LANES) {
        const auto va0 = hn::LoadU(d, a0 + i);
        const auto va1 = hn::LoadU(d, a1 + i);
        const auto va2 = hn::LoadU(d, a2 + i);
        const auto va3 = hn::LoadU(d, a3 + i);
        const auto vb0 = hn::LoadU(d, b0 + i);
        const auto vb1 = hn::LoadU(d, b1 + i);
        const auto vb2 = hn::LoadU(d, b2 + i);
        const auto vb3 = hn::LoadU(d, b3 + i);
        const auto vb4 = hn::LoadU(d, b4 + i);
        const auto vb5 = hn::LoadU(d, b5 + i);
        const auto vb6 = hn::LoadU(d, b6 + i);
        const auto vb7 = hn::LoadU(d, b7 + i);

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

    for (; i < (size_t)Kp; ++i) {
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

    int32_t* HWY_RESTRICT r0 = C;
    int32_t* HWY_RESTRICT r1 = C + ldC;
    int32_t* HWY_RESTRICT r2 = C + 2 * ldC;
    int32_t* HWY_RESTRICT r3 = C + 3 * ldC;
    r0[0]=K_logical-2*(int32_t)s00; r0[1]=K_logical-2*(int32_t)s01;
    r0[2]=K_logical-2*(int32_t)s02; r0[3]=K_logical-2*(int32_t)s03;
    r0[4]=K_logical-2*(int32_t)s04; r0[5]=K_logical-2*(int32_t)s05;
    r0[6]=K_logical-2*(int32_t)s06; r0[7]=K_logical-2*(int32_t)s07;
    r1[0]=K_logical-2*(int32_t)s10; r1[1]=K_logical-2*(int32_t)s11;
    r1[2]=K_logical-2*(int32_t)s12; r1[3]=K_logical-2*(int32_t)s13;
    r1[4]=K_logical-2*(int32_t)s14; r1[5]=K_logical-2*(int32_t)s15;
    r1[6]=K_logical-2*(int32_t)s16; r1[7]=K_logical-2*(int32_t)s17;
    r2[0]=K_logical-2*(int32_t)s20; r2[1]=K_logical-2*(int32_t)s21;
    r2[2]=K_logical-2*(int32_t)s22; r2[3]=K_logical-2*(int32_t)s23;
    r2[4]=K_logical-2*(int32_t)s24; r2[5]=K_logical-2*(int32_t)s25;
    r2[6]=K_logical-2*(int32_t)s26; r2[7]=K_logical-2*(int32_t)s27;
    r3[0]=K_logical-2*(int32_t)s30; r3[1]=K_logical-2*(int32_t)s31;
    r3[2]=K_logical-2*(int32_t)s32; r3[3]=K_logical-2*(int32_t)s33;
    r3[4]=K_logical-2*(int32_t)s34; r3[5]=K_logical-2*(int32_t)s35;
    r3[6]=K_logical-2*(int32_t)s36; r3[7]=K_logical-2*(int32_t)s37;
}

//  Driver: compute one (Mr=4) block of C against an N-tile of B.
//  Picks 4×8 → 4×4 → single-cell depending on remaining N.
HWY_ATTR inline void XorPopcountBlock_4xN(const uint64_t* HWY_RESTRICT a_row0,
                                          const uint64_t* HWY_RESTRICT a_row1,
                                          const uint64_t* HWY_RESTRICT a_row2,
                                          const uint64_t* HWY_RESTRICT a_row3,
                                          const uint64_t* HWY_RESTRICT b_rows,
                                          int64_t Kp, int64_t N_tile,
                                          int32_t K_logical,
                                          int32_t* HWY_RESTRICT c_block,
                                          int64_t ldC) {
    const uint64_t* a_arr[4] = {a_row0, a_row1, a_row2, a_row3};
    int64_t n = 0;
    for (; n + 8 <= N_tile; n += 8) {
        const uint64_t* b_arr[8] = {
            b_rows + (n + 0) * Kp, b_rows + (n + 1) * Kp,
            b_rows + (n + 2) * Kp, b_rows + (n + 3) * Kp,
            b_rows + (n + 4) * Kp, b_rows + (n + 5) * Kp,
            b_rows + (n + 6) * Kp, b_rows + (n + 7) * Kp,
        };
        XorPopcountTile_4x8(a_arr, b_arr, Kp, K_logical, c_block + n, ldC);
    }
    for (; n + 4 <= N_tile; n += 4) {
        const uint64_t* b_arr[4] = {
            b_rows + (n + 0) * Kp, b_rows + (n + 1) * Kp,
            b_rows + (n + 2) * Kp, b_rows + (n + 3) * Kp,
        };
        XorPopcountTile_4x4(a_arr, b_arr, Kp, K_logical, c_block + n, ldC);
    }
    for (; n < N_tile; ++n) {
        const uint64_t* b_row = b_rows + n * Kp;
        c_block[0 * ldC + n] = XorPopcountPair(a_row0, b_row, Kp, K_logical);
        c_block[1 * ldC + n] = XorPopcountPair(a_row1, b_row, Kp, K_logical);
        c_block[2 * ldC + n] = XorPopcountPair(a_row2, b_row, Kp, K_logical);
        c_block[3 * ldC + n] = XorPopcountPair(a_row3, b_row, Kp, K_logical);
    }
}

//  Single-row driver — used for M tail.
HWY_ATTR inline void XorPopcountRow(const uint64_t* HWY_RESTRICT a_row,
                                    const uint64_t* HWY_RESTRICT b_rows,
                                    int64_t Kp, int64_t N_tile, int32_t K_logical,
                                    int32_t* HWY_RESTRICT c_row) {
    for (int64_t n = 0; n < N_tile; ++n) {
        c_row[n] = XorPopcountPair(a_row, b_rows + n * Kp, Kp, K_logical);
    }
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
