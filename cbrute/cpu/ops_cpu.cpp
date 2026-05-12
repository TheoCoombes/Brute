#include "ops_cpu.h"
#include "fastlibpopcnt.h"
#include <ATen/Parallel.h>
#include <climits>

namespace cbrute { namespace cpu {

namespace {

at::ScalarType pack_scalar_type(int64_t pw) {
    if (pw == 8)  return at::kByte;
    if (pw == 32) return at::kInt;
    if (pw == 64) return at::kLong;
    TORCH_CHECK(false, "pack_width must be 8, 32, or 64, got ", pw);
}

} // anon

// ──────────────────────────────────────────────────────────
// pack_bits
// Converts a float tensor to a packed-bit integer tensor.
// Bit = 1 when input element > 0, else 0. Packs along last dim (LSB first).
// Padding bits beyond the logical size are set to 0.
// ──────────────────────────────────────────────────────────
at::Tensor pack_bits(const at::Tensor& input, int64_t pw) {
    TORCH_CHECK(input.dim() >= 1, "pack_bits: input must have >=1 dim");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64, "pack_bits: pack_width must be 8, 32, or 64");

    auto inp   = input.contiguous().to(at::kFloat);
    int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, pack_scalar_type(pw));
    }

    int64_t pd    = (ld + pw - 1) / pw;
    int64_t batch = inp.numel() / ld;
    out_shape.back() = pd;

    auto output = at::zeros(out_shape, pack_scalar_type(pw));
    const float* in_ptr = inp.data_ptr<float>();

    if (pw == 8) {
        uint8_t* out = output.data_ptr<uint8_t>();
        at::parallel_for(0, batch, 1, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                for (int64_t j = 0; j < pd; j++) {
                    uint8_t w = 0;
                    for (int b = 0; b < 8; b++) {
                        int64_t k = j * 8 + b;
                        if (k < ld && in_ptr[i * ld + k] > 0.f) w |= (uint8_t)(1u << b);
                    }
                    out[i * pd + j] = w;
                }
        });
    } else if (pw == 32) {
        int32_t* out = output.data_ptr<int32_t>();
        at::parallel_for(0, batch, 1, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                for (int64_t j = 0; j < pd; j++) {
                    uint32_t w = 0;
                    for (int b = 0; b < 32; b++) {
                        int64_t k = j * 32 + b;
                        if (k < ld && in_ptr[i * ld + k] > 0.f) w |= (1u << b);
                    }
                    out[i * pd + j] = (int32_t)w;
                }
        });
    } else {
        int64_t* out = output.data_ptr<int64_t>();
        at::parallel_for(0, batch, 1, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                for (int64_t j = 0; j < pd; j++) {
                    uint64_t w = 0;
                    for (int b = 0; b < 64; b++) {
                        int64_t k = j * 64 + b;
                        if (k < ld && in_ptr[i * ld + k] > 0.f) w |= (1ull << b);
                    }
                    out[i * pd + j] = (int64_t)w;
                }
        });
    }
    return output;
}

// ──────────────────────────────────────────────────────────
// unpack_bits
// Inverse of pack_bits. Returns float32 with values +1 / -1.
// ──────────────────────────────────────────────────────────
at::Tensor unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw) {
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64, "unpack_bits: pack_width must be 8, 32, or 64");

    auto p        = packed.contiguous();
    int64_t ll    = logical_shape.back();

    if (ll == 0 || p.numel() == 0) {
        return at::empty(logical_shape.vec(), at::kFloat);
    }

    int64_t pl    = p.size(-1);
    int64_t batch = p.numel() / pl;

    auto output = at::empty(logical_shape.vec(), at::kFloat);
    float* out  = output.data_ptr<float>();

    if (pw == 8) {
        const uint8_t* in = p.data_ptr<uint8_t>();
        at::parallel_for(0, batch, 1, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                for (int64_t j = 0; j < pl; j++) {
                    uint8_t w = in[i * pl + j];
                    for (int b = 0; b < 8; b++) {
                        int64_t k = j * 8 + b;
                        if (k < ll) out[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
                    }
                }
        });
    } else if (pw == 32) {
        const int32_t* in = p.data_ptr<int32_t>();
        at::parallel_for(0, batch, 1, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                for (int64_t j = 0; j < pl; j++) {
                    uint32_t w = (uint32_t)in[i * pl + j];
                    for (int b = 0; b < 32; b++) {
                        int64_t k = j * 32 + b;
                        if (k < ll) out[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
                    }
                }
        });
    } else {
        const int64_t* in = p.data_ptr<int64_t>();
        at::parallel_for(0, batch, 1, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                for (int64_t j = 0; j < pl; j++) {
                    uint64_t w = (uint64_t)in[i * pl + j];
                    for (int b = 0; b < 64; b++) {
                        int64_t k = j * 64 + b;
                        if (k < ll) out[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
                    }
                }
        });
    }
    return output;
}

// ──────────────────────────────────────────────────────────
// xnor_popcount_matmul
// A: (M, Kp), B: (N, Kp) — both packed. K = logical last dim.
// C(m,n) = 2 * popcount(~(A[m] ^ B[n])) - K  ∈ [-K, K]
//
// Uses libpopcnt on the full XNOR row for bulk SIMD popcount
// (AVX-512 VPOPCNTDQ / AVX2 / NEON / scalar fallback).
//
// Padding-bit correction: zero-padded tail bits in both operands
// XNOR to 1, inflating the raw count by (Kp*pw - K) per row.
// K_eff = 2*Kp*pw - K absorbs this: 2*acc - K_eff = 2*true_matches - K.
// ──────────────────────────────────────────────────────────
at::Tensor xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K, int64_t pw) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "xnor_popcount_matmul: inputs must be 2-D");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64, "pack_width must be 8, 32, or 64");
    int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    TORCH_CHECK(Kp == B.size(1), "xnor_popcount_matmul: packed K mismatch");

    auto C     = at::zeros({M, N}, at::kInt);
    int32_t* c = C.data_ptr<int32_t>();
    auto Ac = A.contiguous(), Bc = B.contiguous();

    size_t   row_bytes = static_cast<size_t>(Kp) * (pw / 8);
    int32_t  K_eff     = (int32_t)(2LL * Kp * pw - K);

    if (pw == 64) {
        const uint64_t* a = reinterpret_cast<const uint64_t*>(Ac.data_ptr<int64_t>());
        const uint64_t* b = reinterpret_cast<const uint64_t*>(Bc.data_ptr<int64_t>());
        at::parallel_for(0, M, 1, [&](int64_t ms, int64_t me) {
            std::vector<uint64_t> xnor_buf(Kp);
            for (int64_t m = ms; m < me; m++)
                for (int64_t n = 0; n < N; n++) {
                    for (int64_t k = 0; k < Kp; k++)
                        xnor_buf[k] = ~(a[m*Kp+k] ^ b[n*Kp+k]);
                    c[m*N+n] = 2*(int32_t)popcnt(xnor_buf.data(), row_bytes) - K_eff;
                }
        });
    } else if (pw == 32) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(Ac.data_ptr<int32_t>());
        const uint32_t* b = reinterpret_cast<const uint32_t*>(Bc.data_ptr<int32_t>());
        at::parallel_for(0, M, 1, [&](int64_t ms, int64_t me) {
            std::vector<uint32_t> xnor_buf(Kp);
            for (int64_t m = ms; m < me; m++)
                for (int64_t n = 0; n < N; n++) {
                    for (int64_t k = 0; k < Kp; k++)
                        xnor_buf[k] = ~(a[m*Kp+k] ^ b[n*Kp+k]);
                    c[m*N+n] = 2*(int32_t)popcnt(xnor_buf.data(), row_bytes) - K_eff;
                }
        });
    } else { // pw == 8
        const uint8_t* a = Ac.data_ptr<uint8_t>();
        const uint8_t* b = Bc.data_ptr<uint8_t>();
        at::parallel_for(0, M, 1, [&](int64_t ms, int64_t me) {
            std::vector<uint8_t> xnor_buf(Kp);
            for (int64_t m = ms; m < me; m++)
                for (int64_t n = 0; n < N; n++) {
                    for (int64_t k = 0; k < Kp; k++)
                        xnor_buf[k] = ~(a[m*Kp+k] ^ b[n*Kp+k]);
                    c[m*N+n] = 2*(int32_t)popcnt(xnor_buf.data(), row_bytes) - K_eff;
                }
        });
    }
    return C;
}

// ──────────────────────────────────────────────────────────
// popcount — per-element popcount of each packed word into int32.
// Uses popcnt64 (libpopcnt: hardware POPCNT / builtin / bitwise fallback)
// with at::parallel_for for threading. Grain size keeps single-word
// tensors on the calling thread (no parallel overhead).
// ──────────────────────────────────────────────────────────
at::Tensor popcount(const at::Tensor& packed) {
    auto p   = packed.contiguous();
    auto out = at::empty(p.sizes(), at::kInt);
    int64_t n = p.numel();
    int32_t* o = out.data_ptr<int32_t>();

    // Below this many elements, run on the calling thread.
    constexpr int64_t GRAIN = 2048;

    switch (p.scalar_type()) {
    case at::kByte: {
        const uint8_t* in = p.data_ptr<uint8_t>();
        at::parallel_for(0, n, GRAIN, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                o[i] = (int32_t)popcnt64((uint64_t)in[i]);
        });
        break;
    }
    case at::kInt: {
        const int32_t* in = p.data_ptr<int32_t>();
        at::parallel_for(0, n, GRAIN, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                o[i] = (int32_t)popcnt64((uint64_t)(uint32_t)in[i]);
        });
        break;
    }
    default: { // kLong
        const int64_t* in = p.data_ptr<int64_t>();
        at::parallel_for(0, n, GRAIN, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; i++)
                o[i] = (int32_t)popcnt64((uint64_t)in[i]);
        });
        break;
    }
    }
    return out;
}

// ──────────────────────────────────────────────────────────
// packed_popcount — total count of 1-bits in the entire buffer.
// Uses libpopcnt on the full contiguous byte range (SIMD-optimal:
// exploits AVX-512 VPOPCNTDQ / AVX2 / NEON depending on host CPU).
// Padding bits in the packed buffer are 0 (set during pack_bits),
// so they contribute 0 to the total — no correction needed.
// ──────────────────────────────────────────────────────────
at::Tensor packed_popcount(const at::Tensor& packed) {
    auto p = packed.contiguous();
    uint64_t total = popcnt(p.data_ptr(), static_cast<uint64_t>(p.nbytes()));
    return at::scalar_tensor((int64_t)total, at::kLong);
}

at::Tensor hamming_distance(const at::Tensor& A, const at::Tensor& B) {
    return popcount(at::bitwise_xor(A, B));
}

at::Tensor bitwise_and(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_and(A, B); }
at::Tensor bitwise_or (const at::Tensor& A, const at::Tensor& B) { return at::bitwise_or(A, B);  }
at::Tensor bitwise_xor(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_xor(A, B); }
at::Tensor bitwise_not(const at::Tensor& A)                       { return at::bitwise_not(A);    }

at::Tensor& randomize_bits(at::Tensor& out) {
    out.random_();
    return out;
}

}} // cbrute::cpu
