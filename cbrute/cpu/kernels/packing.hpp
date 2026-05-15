// Highway-driven bit-packing / bit-unpacking kernels.
//
// Internal byte layout: bits are packed LSB-first within each byte; a packed
// row of N logical bits consumes ceil(N/8) bytes. Wider pack widths (uint32,
// uint64) just round the row byte count up to a multiple of 4 or 8 and zero-
// pad. The bit content within the used bytes is identical, regardless of pw.
//
// Hot paths:
//   * pack_bool : on AVX-512BW we use VPMOVB2M (byte-MSB → mask register) for
//                 64 bool bytes → 8 packed bytes per iteration. Otherwise the
//                 per-byte Highway BitsFromMask path remains.
//   * unpack_pm1: 256-entry LUT mapping each input byte → 8 floats (+1.0 / −1.0).
//                 LUT is 8 KB, fits in L1; one load + one store per byte.
//   * unpack_bool: 256-entry LUT mapping each input byte → 8 uint8 (0 / 1).
//                  LUT is 2 KB, fits in L1.

#pragma once

#include <hwy/highway.h>
#include <cstddef>
#include <cstdint>

#if defined(__AVX512BW__) && defined(__AVX512F__)
  #define BRUTE_HAVE_AVX512BW 1
  #include <immintrin.h>
#else
  #define BRUTE_HAVE_AVX512BW 0
#endif

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ────────────────────────────────────────────────────────────────────────
// Lookup tables. Generated at compile time so the kernels stay header-only
// and the LUT is an immediate const data section (no runtime init).
// ────────────────────────────────────────────────────────────────────────
namespace detail {

constexpr uint64_t make_bool_lut_entry(uint8_t byte) {
    // Each output uint64 holds 8 bytes of value 0/1 (little-endian).
    uint64_t out = 0;
    for (int b = 0; b < 8; ++b) {
        if ((byte >> b) & 1) out |= (uint64_t)1 << (b * 8);
    }
    return out;
}

struct BoolLUT {
    uint64_t v[256];
    constexpr BoolLUT() : v{} {
        for (int i = 0; i < 256; ++i) v[i] = make_bool_lut_entry((uint8_t)i);
    }
};
inline constexpr BoolLUT kBoolLUT{};

// pm1 LUT: 8 floats per entry. We store the float bit pattern as uint32 so
// the table is constexpr-friendly and we can splice via memcpy.
struct Pm1LUT {
    uint32_t v[256][8];   // each entry: 8 little-endian float bit patterns
    constexpr Pm1LUT() : v{} {
        constexpr uint32_t POS = 0x3F800000u;   // bits of +1.0f
        constexpr uint32_t NEG = 0xBF800000u;   // bits of -1.0f
        for (int i = 0; i < 256; ++i) {
            for (int b = 0; b < 8; ++b) {
                v[i][b] = ((i >> b) & 1) ? POS : NEG;
            }
        }
    }
};
inline constexpr Pm1LUT kPm1LUT{};

}  // namespace detail

//  Pack bool bytes (0/1) → packed bytes (LSB-first)
// Writes ceil(n_bits / 8) bytes. Pad bits (in the trailing byte beyond n_bits)
// are zero.
HWY_ATTR inline void PackBoolBytesToBits(const uint8_t* HWY_RESTRICT in_bool,
                                         uint8_t* HWY_RESTRICT out_bytes,
                                         size_t n_bits) {
    const size_t n_full = n_bits / 8;
    size_t j = 0;

#if BRUTE_HAVE_AVX512BW
    // 64 bool bytes → 64-bit mask → 8 output bytes per iteration.
    // VPMOVB2M can't be used directly because bool bytes are 0/1 (not 0/0xFF),
    // so we compare against zero with VPCMPNEQB to get a mask.
    const __m512i zero = _mm512_setzero_si512();
    for (; j + 8 <= n_full; j += 8) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(in_bool + j * 8));
        __mmask64 m = _mm512_cmpneq_epi8_mask(v, zero);
        _store_mask64((__mmask64*)(out_bytes + j), m);
    }
#endif

    // Highway 8-byte-at-a-time fallback / tail.
    {
        const hn::CappedTag<uint8_t, 8> d;
        const size_t N = hn::Lanes(d);
        const auto zero = hn::Zero(d);
        if (N == 8) {
            for (; j < n_full; ++j) {
                auto v = hn::LoadU(d, in_bool + j * 8);
                auto m = hn::Ne(v, zero);
                out_bytes[j] = (uint8_t)hn::BitsFromMask(d, m);
            }
        } else {
            for (; j < n_full; ++j) {
                uint8_t byte = 0;
                for (size_t s = 0; s < 8; s += N) {
                    auto v = hn::LoadU(d, in_bool + j * 8 + s);
                    auto m = hn::Ne(v, zero);
                    byte |= (uint8_t)(hn::BitsFromMask(d, m) << s);
                }
                out_bytes[j] = byte;
            }
        }
    }

    const size_t leftover = n_bits - n_full * 8;
    if (leftover) {
        uint8_t byte = 0;
        for (size_t b = 0; b < leftover; ++b) {
            if (in_bool[n_full * 8 + b]) byte |= (uint8_t)(1u << b);
        }
        out_bytes[n_full] = byte;
    }
}

//  Unpack packed bytes → float32 (+1.0 / −1.0)
// 256-entry LUT: each input byte indexes 8 pre-computed float bit-patterns.
// Compiler emits a single 32-byte aligned load per byte. The LUT is 8 KB so
// it lives in L1 across calls.
HWY_ATTR inline void UnpackBitsToPm1(const uint8_t* HWY_RESTRICT in_bytes,
                                     float* HWY_RESTRICT out, size_t n_bits) {
    const size_t n_full = n_bits / 8;
    for (size_t j = 0; j < n_full; ++j) {
        // memcpy is the standards-conformant way to spell "treat 32 bytes of
        // table as a vector store" — the compiler reduces it to a single
        // movups/vmovdqu.
        __builtin_memcpy(out + j * 8, &detail::kPm1LUT.v[in_bytes[j]][0],
                         sizeof(uint32_t) * 8);
    }
    const size_t leftover = n_bits - n_full * 8;
    if (leftover) {
        const uint8_t byte = in_bytes[n_full];
        for (size_t b = 0; b < leftover; ++b) {
            out[n_full * 8 + b] = ((byte >> b) & 1) ? 1.0f : -1.0f;
        }
    }
}

//  Unpack packed bytes → bool bytes (0/1)
// 256-entry LUT: each input byte indexes 8 bytes of value 0 or 1, packed into
// a uint64. Compiler emits a single 8-byte load + 8-byte store per byte.
HWY_ATTR inline void UnpackBitsToBoolBytes(const uint8_t* HWY_RESTRICT in_bytes,
                                           uint8_t* HWY_RESTRICT out_bool,
                                           size_t n_bits) {
    const size_t n_full = n_bits / 8;
    for (size_t j = 0; j < n_full; ++j) {
        const uint64_t entry = detail::kBoolLUT.v[in_bytes[j]];
        __builtin_memcpy(out_bool + j * 8, &entry, sizeof(uint64_t));
    }
    const size_t leftover = n_bits - n_full * 8;
    if (leftover) {
        const uint8_t byte = in_bytes[n_full];
        for (size_t b = 0; b < leftover; ++b) {
            out_bool[n_full * 8 + b] = (byte >> b) & 1;
        }
    }
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
