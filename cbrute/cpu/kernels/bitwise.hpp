// Highway-driven elementwise bitwise + popcount kernels.
//
// All kernels operate on raw pointers + word counts. Wrapping into at::Tensor,
// dispatch by dtype, and at::parallel_for slicing happen in ops_cpu.cpp.
//
// No allocations inside any kernel. No std::vector. Scalar tails handled inline.

#pragma once

#include <hwy/highway.h>
#include <cstddef>
#include <cstdint>

HWY_BEFORE_NAMESPACE();
namespace cbrute { namespace cpu { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ── Word-level bitwise ─────────────────────────────────────────────────────────
// Templated on the storage word type T ∈ {uint8_t, uint32_t, uint64_t}.

template <typename T>
HWY_ATTR void XorWords(const T* HWY_RESTRICT a, const T* HWY_RESTRICT b,
                       T* HWY_RESTRICT out, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::Xor(hn::LoadU(d, a + i), hn::LoadU(d, b + i)),
                   d, out + i);
    }
    for (; i < n; ++i) out[i] = a[i] ^ b[i];
}

template <typename T>
HWY_ATTR void AndWords(const T* HWY_RESTRICT a, const T* HWY_RESTRICT b,
                       T* HWY_RESTRICT out, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::And(hn::LoadU(d, a + i), hn::LoadU(d, b + i)),
                   d, out + i);
    }
    for (; i < n; ++i) out[i] = a[i] & b[i];
}

template <typename T>
HWY_ATTR void OrWords(const T* HWY_RESTRICT a, const T* HWY_RESTRICT b,
                      T* HWY_RESTRICT out, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::Or(hn::LoadU(d, a + i), hn::LoadU(d, b + i)),
                   d, out + i);
    }
    for (; i < n; ++i) out[i] = a[i] | b[i];
}

template <typename T>
HWY_ATTR void NotWords(const T* HWY_RESTRICT a, T* HWY_RESTRICT out, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::Not(hn::LoadU(d, a + i)), d, out + i);
    }
    for (; i < n; ++i) out[i] = ~a[i];
}

// XNOR (Not(Xor(a,b))) — fundamental for bit1 equality.
template <typename T>
HWY_ATTR void XnorWords(const T* HWY_RESTRICT a, const T* HWY_RESTRICT b,
                        T* HWY_RESTRICT out, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N) {
        hn::StoreU(hn::Not(hn::Xor(hn::LoadU(d, a + i), hn::LoadU(d, b + i))),
                   d, out + i);
    }
    for (; i < n; ++i) out[i] = ~(a[i] ^ b[i]);
}

// where: (cond & yes) | (~cond & no) — Highway exposes this directly.
template <typename T>
HWY_ATTR void BitwiseIfThenElseWords(const T* HWY_RESTRICT cond,
                                     const T* HWY_RESTRICT yes,
                                     const T* HWY_RESTRICT no,
                                     T* HWY_RESTRICT out, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;
    for (; i + N <= n; i += N) {
        auto vc = hn::LoadU(d, cond + i);
        auto vy = hn::LoadU(d, yes  + i);
        auto vn = hn::LoadU(d, no   + i);
        hn::StoreU(hn::BitwiseIfThenElse(vc, vy, vn), d, out + i);
    }
    for (; i < n; ++i) out[i] = (cond[i] & yes[i]) | (~cond[i] & no[i]);
}

}}}  // namespace cbrute::cpu::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
