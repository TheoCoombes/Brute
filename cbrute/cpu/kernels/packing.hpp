// Highway-driven bit-packing / bit-unpacking kernels.
//
// Internal byte layout: bits are packed LSB-first within each byte; a packed
// row of N logical bits consumes ceil(N/8) bytes. Wider pack widths (uint32,
// uint64) just round the row byte count up to a multiple of 4 or 8 and zero-
// pad. The bit content within the used bytes is identical, regardless of pw.

#pragma once

#include <hwy/highway.h>
#include <cstddef>
#include <cstdint>

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ── Pack bool bytes (0/1) → packed bytes (LSB-first) ─────────────────────────
// Writes ceil(n_bits / 8) bytes. Pad bits (in the trailing byte beyond n_bits)
// are zero.
HWY_ATTR inline void PackBoolBytesToBits(const uint8_t* HWY_RESTRICT in_bool,
                                         uint8_t* HWY_RESTRICT out_bytes,
                                         size_t n_bits) {
    const hn::CappedTag<uint8_t, 8> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    const size_t n_full = n_bits / 8;

    if (N == 8) {
        for (size_t j = 0; j < n_full; ++j) {
            auto v = hn::LoadU(d, in_bool + j * 8);
            auto m = hn::Ne(v, zero);
            out_bytes[j] = (uint8_t)hn::BitsFromMask(d, m);
        }
    } else {
        // Narrow-vector fallback: consume N bytes at a time, OR partial bits.
        for (size_t j = 0; j < n_full; ++j) {
            uint8_t byte = 0;
            for (size_t s = 0; s < 8; s += N) {
                auto v = hn::LoadU(d, in_bool + j * 8 + s);
                auto m = hn::Ne(v, zero);
                byte |= (uint8_t)(hn::BitsFromMask(d, m) << s);
            }
            out_bytes[j] = byte;
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

// ── Pack float32 (>0) → packed bytes ──────────────────────────────────────────
// Bit = 1 iff input > 0.0f.
HWY_ATTR inline void PackFloatToBits(const float* HWY_RESTRICT in_f,
                                     uint8_t* HWY_RESTRICT out_bytes,
                                     size_t n_bits) {
    const hn::CappedTag<float, 8> df;
    const size_t N = hn::Lanes(df);
    const auto zero = hn::Zero(df);
    const size_t n_full = n_bits / 8;

    if (N == 8) {
        for (size_t j = 0; j < n_full; ++j) {
            auto v = hn::LoadU(df, in_f + j * 8);
            auto m = hn::Gt(v, zero);
            out_bytes[j] = (uint8_t)hn::BitsFromMask(df, m);
        }
    } else {
        for (size_t j = 0; j < n_full; ++j) {
            uint8_t byte = 0;
            for (size_t s = 0; s < 8; s += N) {
                auto v = hn::LoadU(df, in_f + j * 8 + s);
                auto m = hn::Gt(v, zero);
                byte |= (uint8_t)(hn::BitsFromMask(df, m) << s);
            }
            out_bytes[j] = byte;
        }
    }
    const size_t leftover = n_bits - n_full * 8;
    if (leftover) {
        uint8_t byte = 0;
        for (size_t b = 0; b < leftover; ++b) {
            if (in_f[n_full * 8 + b] > 0.f) byte |= (uint8_t)(1u << b);
        }
        out_bytes[n_full] = byte;
    }
}

// ── Unpack packed bytes → float32 (+1.0 / −1.0) ──────────────────────────────
HWY_ATTR inline void UnpackBitsToPm1(const uint8_t* HWY_RESTRICT in_bytes,
                                     float* HWY_RESTRICT out, size_t n_bits) {
    const hn::CappedTag<float, 8> df;
    const size_t N = hn::Lanes(df);
    const auto pos = hn::Set(df, 1.0f);
    const auto neg = hn::Set(df, -1.0f);
    const size_t n_full = n_bits / 8;

    if (N == 8) {
        for (size_t j = 0; j < n_full; ++j) {
            auto m = hn::LoadMaskBits(df, in_bytes + j);
            hn::StoreU(hn::IfThenElse(m, pos, neg), df, out + j * 8);
        }
    } else {
        for (size_t j = 0; j < n_full; ++j) {
            const uint8_t byte = in_bytes[j];
            for (int b = 0; b < 8; ++b) {
                out[j * 8 + b] = ((byte >> b) & 1) ? 1.0f : -1.0f;
            }
        }
    }
    const size_t leftover = n_bits - n_full * 8;
    if (leftover) {
        const uint8_t byte = in_bytes[n_full];
        for (size_t b = 0; b < leftover; ++b) {
            out[n_full * 8 + b] = ((byte >> b) & 1) ? 1.0f : -1.0f;
        }
    }
}

// ── Unpack packed bytes → bool bytes (0/1) ───────────────────────────────────
HWY_ATTR inline void UnpackBitsToBoolBytes(const uint8_t* HWY_RESTRICT in_bytes,
                                           uint8_t* HWY_RESTRICT out_bool,
                                           size_t n_bits) {
    const hn::CappedTag<uint8_t, 8> d8;
    const size_t N = hn::Lanes(d8);
    const auto one = hn::Set(d8, (uint8_t)1);
    const size_t n_full = n_bits / 8;

    if (N == 8) {
        for (size_t j = 0; j < n_full; ++j) {
            auto m = hn::LoadMaskBits(d8, in_bytes + j);
            hn::StoreU(hn::IfThenElseZero(m, one), d8, out_bool + j * 8);
        }
    } else {
        for (size_t j = 0; j < n_full; ++j) {
            const uint8_t byte = in_bytes[j];
            for (int b = 0; b < 8; ++b) out_bool[j * 8 + b] = (byte >> b) & 1;
        }
    }
    const size_t leftover = n_bits - n_full * 8;
    if (leftover) {
        const uint8_t byte = in_bytes[n_full];
        for (size_t b = 0; b < leftover; ++b) out_bool[n_full * 8 + b] = (byte >> b) & 1;
    }
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
