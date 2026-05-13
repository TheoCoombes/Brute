// brute CPU backend — driver layer.
//
// Public ops registered as TORCH_LIBRARY_IMPL handlers (see ext.cpp).
// All heavy lifting is delegated to the Highway kernels in `kernels/`; this
// file does Tensor validation, output allocation (always once, outside the
// parallel loop), at::parallel_for slicing of the outer dimension, and dtype
// dispatch.
//
// Invariants:
//   * Zero heap allocations inside any parallel_for body.
//   * No std::vector / std::shared_ptr in hot paths.
//   * All kernels see contiguous, type-stable raw pointers.
//   * Output tensors are zero-initialized when padding bytes/bits must be 0;
//     otherwise allocated with at::empty.

#include "ops_cpu.h"
#include "kernels/bitwise.hpp"
#include "kernels/packing.hpp"
#include "kernels/popcount.hpp"
#include "kernels/reductions.hpp"
#include "kernels/matmul.hpp"

#include <ATen/Parallel.h>
#include <ATen/Dispatch.h>
#include <type_traits>

namespace hnk = cbrute::cpu::HWY_NAMESPACE;

namespace cbrute { namespace cpu {

namespace {

at::ScalarType pack_scalar_type(int64_t pw) {
    switch (pw) {
        case 8:  return at::kByte;
        case 32: return at::kInt;
        case 64: return at::kLong;
        default: TORCH_CHECK(false, "pack_width must be 8, 32, or 64; got ", pw);
    }
}

// Lower bytes of any contiguous integer/bool tensor as a flat uint8 stream.
// Used wherever we treat the packed buffer as raw bytes.
inline uint8_t*       byte_ptr(at::Tensor& t)       { return static_cast<uint8_t*>(t.data_ptr()); }
inline const uint8_t* byte_ptr(const at::Tensor& t) { return static_cast<const uint8_t*>(t.data_ptr()); }

// Grain size for at::parallel_for: roughly "min work per thread" in elements.
// Tuned so that single-element / tiny-tensor cases stay on the calling thread.
constexpr int64_t POPCOUNT_GRAIN = 4096;
constexpr int64_t ROW_GRAIN      = 1;       // per-row workloads (matmul, pack)

} // anon

// 
// pack_bits — float input, threshold (> 0.f) → packed bits.
// Legacy path used by `unpack_pm1`-style round-trips and any caller that
// already has float storage.
// 
at::Tensor pack_bits(const at::Tensor& input, int64_t pw) {
    TORCH_CHECK(input.dim() >= 1, "pack_bits: input must have >= 1 dim");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "pack_bits: pack_width must be 8, 32, or 64");

    const auto inp = input.contiguous().to(at::kFloat);
    const int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, pack_scalar_type(pw));
    }

    const int64_t pd_words  = (ld + pw - 1) / pw;
    const int64_t row_bytes = pd_words * (pw / 8);
    const int64_t batch     = inp.numel() / ld;
    out_shape.back() = pd_words;

    // zeros() so trailing bytes between ceil(ld/8) and row_bytes are 0.
    auto output = at::zeros(out_shape, pack_scalar_type(pw));
    const float* in_f   = inp.data_ptr<float>();
    uint8_t*    out_b   = byte_ptr(output);

    at::parallel_for(0, batch, ROW_GRAIN, [&](int64_t s, int64_t e) {
        for (int64_t r = s; r < e; ++r) {
            hnk::PackFloatToBits(in_f + r * ld, out_b + r * row_bytes, (size_t)ld);
        }
    });
    return output;
}

// 
// pack_bool — bool input → packed bits. Skips the float intermediate that
// _pack_bool used to materialize in Python.
// 
at::Tensor pack_bool(const at::Tensor& input, int64_t pw) {
    TORCH_CHECK(input.dim() >= 1, "pack_bool: input must have >= 1 dim");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "pack_bool: pack_width must be 8, 32, or 64");
    TORCH_CHECK(input.scalar_type() == at::kBool,
                "pack_bool: input must be torch.bool");

    const auto inp = input.contiguous();
    const int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, pack_scalar_type(pw));
    }

    const int64_t pd_words  = (ld + pw - 1) / pw;
    const int64_t row_bytes = pd_words * (pw / 8);
    const int64_t batch     = inp.numel() / ld;
    out_shape.back() = pd_words;

    auto output = at::zeros(out_shape, pack_scalar_type(pw));
    const uint8_t* in_bool = reinterpret_cast<const uint8_t*>(inp.data_ptr<bool>());
    uint8_t*       out_b   = byte_ptr(output);

    at::parallel_for(0, batch, ROW_GRAIN, [&](int64_t s, int64_t e) {
        for (int64_t r = s; r < e; ++r) {
            hnk::PackBoolBytesToBits(in_bool + r * ld, out_b + r * row_bytes, (size_t)ld);
        }
    });
    return output;
}

// 
// unpack_bits — packed → float32 ±1.0
// 
at::Tensor unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw) {
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "unpack_bits: pack_width must be 8, 32, or 64");

    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0) {
        return at::empty(logical_shape.vec(), at::kFloat);
    }

    const int64_t pl_words  = p.size(-1);
    const int64_t row_bytes = pl_words * (pw / 8);
    const int64_t batch     = p.numel() / pl_words;

    auto output = at::empty(logical_shape.vec(), at::kFloat);
    const uint8_t* in_b = byte_ptr(p);
    float*         out_f = output.data_ptr<float>();

    at::parallel_for(0, batch, ROW_GRAIN, [&](int64_t s, int64_t e) {
        for (int64_t r = s; r < e; ++r) {
            hnk::UnpackBitsToPm1(in_b + r * row_bytes, out_f + r * ll, (size_t)ll);
        }
    });
    return output;
}

// 
// unpack_bool — packed → bool (1 byte per logical bit, value 0/1)
// 
at::Tensor unpack_bool(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw) {
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "unpack_bool: pack_width must be 8, 32, or 64");

    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0) {
        return at::empty(logical_shape.vec(), at::kBool);
    }

    const int64_t pl_words  = p.size(-1);
    const int64_t row_bytes = pl_words * (pw / 8);
    const int64_t batch     = p.numel() / pl_words;

    auto output = at::empty(logical_shape.vec(), at::kBool);
    const uint8_t* in_b   = byte_ptr(p);
    uint8_t*       out_bl = reinterpret_cast<uint8_t*>(output.data_ptr<bool>());

    at::parallel_for(0, batch, ROW_GRAIN, [&](int64_t s, int64_t e) {
        for (int64_t r = s; r < e; ++r) {
            hnk::UnpackBitsToBoolBytes(in_b + r * row_bytes, out_bl + r * ll, (size_t)ll);
        }
    });
    return output;
}

// 
// xnor_popcount_matmul — in-register XNOR + PopulationCount + ReduceSum
// fused inner loop. No scratch buffer; pure register pipeline.
// 
at::Tensor xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B,
                                int64_t K, int64_t pw) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "xnor_popcount_matmul: inputs must be 2-D");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "pack_width must be 8, 32, or 64");
    const int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    TORCH_CHECK(Kp == B.size(1), "xnor_popcount_matmul: packed K mismatch");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C  = at::zeros({M, N}, at::kInt);
    int32_t* c = C.data_ptr<int32_t>();

    const int32_t K_eff = (int32_t)(2LL * Kp * pw - K);

    // N-tiling: process B in N_TILE-column chunks so the active B-tile
    // (N_TILE × Kp words) fits in L1/L2 and is reused across all A rows
    // handled by the same parallel worker, rather than streaming the full
    // B matrix once per A row.  N_TILE=64 keeps a 64×Kp tile in L1 for
    // small Kp and comfortably in L2 for larger ones.
    constexpr int64_t N_TILE = 64;

    // Dispatch by pack width to instantiate the right unsigned word type.
    if (pw == 64) {
        const auto* a = reinterpret_cast<const uint64_t*>(Ac.data_ptr<int64_t>());
        const auto* b = reinterpret_cast<const uint64_t*>(Bc.data_ptr<int64_t>());
        at::parallel_for(0, M, ROW_GRAIN, [&](int64_t ms, int64_t me) {
            for (int64_t n_start = 0; n_start < N; n_start += N_TILE) {
                const int64_t n_cnt = std::min(N_TILE, N - n_start);
                for (int64_t m = ms; m < me; ++m) {
                    hnk::XnorPopcountRow<uint64_t>(
                        a + m * Kp, b + n_start * Kp, Kp, n_cnt, K_eff,
                        c + m * N + n_start);
                }
            }
        });
    } else if (pw == 32) {
        const auto* a = reinterpret_cast<const uint32_t*>(Ac.data_ptr<int32_t>());
        const auto* b = reinterpret_cast<const uint32_t*>(Bc.data_ptr<int32_t>());
        at::parallel_for(0, M, ROW_GRAIN, [&](int64_t ms, int64_t me) {
            for (int64_t n_start = 0; n_start < N; n_start += N_TILE) {
                const int64_t n_cnt = std::min(N_TILE, N - n_start);
                for (int64_t m = ms; m < me; ++m) {
                    hnk::XnorPopcountRow<uint32_t>(
                        a + m * Kp, b + n_start * Kp, Kp, n_cnt, K_eff,
                        c + m * N + n_start);
                }
            }
        });
    } else { // pw == 8
        const auto* a = Ac.data_ptr<uint8_t>();
        const auto* b = Bc.data_ptr<uint8_t>();
        at::parallel_for(0, M, ROW_GRAIN, [&](int64_t ms, int64_t me) {
            for (int64_t n_start = 0; n_start < N; n_start += N_TILE) {
                const int64_t n_cnt = std::min(N_TILE, N - n_start);
                for (int64_t m = ms; m < me; ++m) {
                    hnk::XnorPopcountRow<uint8_t>(
                        a + m * Kp, b + n_start * Kp, Kp, n_cnt, K_eff,
                        c + m * N + n_start);
                }
            }
        });
    }
    return C;
}

// 
// popcount — per-element, output is int32 with same shape as input.
// Supports any integer dtype (incl. bool).
// 
at::Tensor popcount(const at::Tensor& x) {
    const auto p = x.contiguous();
    auto out = at::empty(p.sizes(), at::kInt);
    const int64_t n = p.numel();
    if (n == 0) return out;
    int32_t* o = out.data_ptr<int32_t>();

    AT_DISPATCH_INTEGRAL_TYPES_AND(at::kBool, p.scalar_type(), "popcount", [&] {
        const scalar_t* in = p.data_ptr<scalar_t>();
        at::parallel_for(0, n, POPCOUNT_GRAIN, [&](int64_t s, int64_t e) {
            hnk::PopcountPerWord<scalar_t>(in + s, o + s, (size_t)(e - s));
        });
    });
    return out;
}

// 
// packed_popcount — total 1-bit count across the whole buffer as int64 scalar.
// Treats the buffer as a flat byte stream (dtype-agnostic).
// Valid for bit1 because pad bits are 0 by construction.
// 
at::Tensor packed_popcount(const at::Tensor& x) {
    const auto p = x.contiguous();
    const uint64_t total = hnk::TotalPopcountBytes(p.data_ptr(), (size_t)p.nbytes());
    return at::scalar_tensor((int64_t)total, at::kLong);
}

// 
// hamming_distance — per-element popcount(A ^ B). Output int32, broadcasted
// via torch's TensorIterator (handles shape broadcasting & strides for us).
// 
at::Tensor hamming_distance(const at::Tensor& A, const at::Tensor& B) {
    return popcount(at::bitwise_xor(A, B));
}

// 
// bit1_hamming_total — fused total Hamming distance over two identically-
// shaped & typed packed buffers. No XOR temporary; in-register fusion.
// Used by torch.equal / != short-circuits in the Python wrapper.
// 
at::Tensor bit1_hamming_total(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.sizes()       == B.sizes(),       "bit1_hamming_total: shape mismatch");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "bit1_hamming_total: dtype mismatch");
    const auto Ac = A.contiguous();
    const auto Bc = B.contiguous();
    const uint64_t total = hnk::TotalPopcountXor(Ac.data_ptr(), Bc.data_ptr(),
                                                 (size_t)Ac.nbytes());
    return at::scalar_tensor((int64_t)total, at::kLong);
}

// 
// Bitwise pass-throughs. at:: implementations already SIMD-vectorize on the
// packed integer buffer, and they handle broadcasting + strides correctly.
// 
at::Tensor bitwise_and(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_and(A, B); }
at::Tensor bitwise_or (const at::Tensor& A, const at::Tensor& B) { return at::bitwise_or (A, B); }
at::Tensor bitwise_xor(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_xor(A, B); }
at::Tensor bitwise_not(const at::Tensor& A)                       { return at::bitwise_not(A);    }

at::Tensor& randomize_bits(at::Tensor& out) {
    out.random_();
    return out;
}

}} // cbrute::cpu
