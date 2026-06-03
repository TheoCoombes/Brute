// brute CPU backend — driver layer.
//
// All bit1 packed buffers are uint64 (64 bits per int64 word). Drivers do
// Tensor validation, output allocation, and at::parallel_for slicing; all
// heavy lifting lives in the Highway kernels in `kernels/`.
//
// Invariants:
//   * Zero heap allocations inside any parallel_for body.
//   * All kernels see contiguous, type-stable raw pointers.
//   * Output tensors are zero-initialised when padding bits must be 0;
//     otherwise allocated with at::empty.

#include "ops_cpu.h"
#include "kernels/packing.hpp"
#include "kernels/popcount.hpp"
#include "kernels/matmul.hpp"

#include <ATen/Parallel.h>
#include <ATen/Dispatch.h>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace hnk = cbrute::cpu::HWY_NAMESPACE;

namespace cbrute { namespace cpu {

namespace {

constexpr int64_t PACK_WIDTH      = 64;
constexpr int64_t POPCOUNT_GRAIN  = 4096;
constexpr int64_t ROW_GRAIN       = 1;     // per-row workloads (pack/unpack)
constexpr int64_t MATMUL_M_GRAIN  = 4;     // = Mr microkernel block

inline uint8_t*       byte_ptr(at::Tensor& t)       { return static_cast<uint8_t*>(t.data_ptr()); }
inline const uint8_t* byte_ptr(const at::Tensor& t) { return static_cast<const uint8_t*>(t.data_ptr()); }

} // anon

//  pack_bool — bool input → packed bits (int64 storage).
at::Tensor pack_bool(const at::Tensor& input) {
    TORCH_CHECK(input.dim() >= 1, "pack_bool: input must have >= 1 dim");
    TORCH_CHECK(input.scalar_type() == at::kBool,
                "pack_bool: input must be torch.bool");

    const auto inp = input.contiguous();
    const int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, at::kLong);
    }

    const int64_t pd_words  = (ld + PACK_WIDTH - 1) / PACK_WIDTH;
    const int64_t row_bytes = pd_words * (PACK_WIDTH / 8);
    const int64_t batch     = inp.numel() / ld;
    out_shape.back() = pd_words;

    auto output = at::zeros(out_shape, at::kLong);
    const uint8_t* in_bool = reinterpret_cast<const uint8_t*>(inp.data_ptr<bool>());
    uint8_t*       out_b   = byte_ptr(output);

    at::parallel_for(0, batch, ROW_GRAIN, [&](int64_t s, int64_t e) {
        for (int64_t r = s; r < e; ++r) {
            hnk::PackBoolBytesToBits(in_bool + r * ld, out_b + r * row_bytes, (size_t)ld);
        }
    });
    return output;
}

//  pack_sign — numeric input → packed bits, sign(x) >= 0 maps to bit 1.
at::Tensor pack_sign(const at::Tensor& input) {
    TORCH_CHECK(input.dim() >= 1, "pack_sign: input must have >= 1 dim");
    TORCH_CHECK(input.scalar_type() != at::kBool,
                "pack_sign: input must be numeric, not torch.bool");

    const auto inp = input.contiguous();
    const int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, inp.options().dtype(at::kLong));
    }

    const int64_t pd_words = (ld + PACK_WIDTH - 1) / PACK_WIDTH;
    const int64_t batch = inp.numel() / ld;
    out_shape.back() = pd_words;

    auto output = at::zeros(out_shape, inp.options().dtype(at::kLong));
    uint64_t* out = reinterpret_cast<uint64_t*>(output.data_ptr<int64_t>());

    AT_DISPATCH_ALL_TYPES(inp.scalar_type(), "pack_sign", [&] {
        const scalar_t* in = inp.data_ptr<scalar_t>();
        at::parallel_for(0, batch, ROW_GRAIN, [&](int64_t s, int64_t e) {
            for (int64_t r = s; r < e; ++r) {
                const scalar_t* row = in + r * ld;
                uint64_t* out_row = out + r * pd_words;
                for (int64_t w = 0; w < pd_words; ++w) {
                    uint64_t word = 0;
                    const int64_t base = w * PACK_WIDTH;
                    const int64_t live = std::min<int64_t>(PACK_WIDTH, ld - base);
                    for (int64_t bit = 0; bit < live; ++bit) {
                        const auto v = row[base + bit];
                        const bool positive = v >= scalar_t(0);
                        if (positive) word |= (uint64_t(1) << bit);
                    }
                    out_row[w] = word;
                }
            }
        });
    });
    return output;
}

//  unpack_bits — packed → float32 ±1.0
at::Tensor unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape) {
    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0) return at::empty(logical_shape.vec(), at::kFloat);

    const int64_t pl_words  = p.size(-1);
    const int64_t row_bytes = pl_words * (PACK_WIDTH / 8);
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

//  unpack_bool — packed → bool (1 byte per logical bit, value 0/1)
at::Tensor unpack_bool(const at::Tensor& packed, at::IntArrayRef logical_shape) {
    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0) return at::empty(logical_shape.vec(), at::kBool);

    const int64_t pl_words  = p.size(-1);
    const int64_t row_bytes = pl_words * (PACK_WIDTH / 8);
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

//  xnor_popcount_matmul — register-blocked XOR + PopulationCount + ReduceSum.
//  4×8 microkernel (32 accumulators) for N≥8; 4×4 fallback (16 accumulators);
//  single-cell pair for N<4 / M tail. Outer loop is N-tiled (N_TILE=64) so
//  the active B-tile stays in L1/L2 and is reused across all M rows in this
//  worker. Parallel-for slices M in 4-row chunks (one Mr microkernel block).
at::Tensor xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "xnor_popcount_matmul: inputs must be 2-D");
    const int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    TORCH_CHECK(Kp == B.size(1), "xnor_popcount_matmul: packed K mismatch");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C  = at::empty({M, N}, at::kInt);
    int32_t* c = C.data_ptr<int32_t>();

    const int32_t K_logical = (int32_t)K;
    constexpr int64_t N_TILE = 64;

    const uint64_t* a = reinterpret_cast<const uint64_t*>(Ac.data_ptr());
    const uint64_t* b = reinterpret_cast<const uint64_t*>(Bc.data_ptr());

    at::parallel_for(0, M, MATMUL_M_GRAIN, [&](int64_t ms, int64_t me) {
        for (int64_t n_start = 0; n_start < N; n_start += N_TILE) {
            const int64_t n_cnt = std::min(N_TILE, N - n_start);

            int64_t m = ms;
            for (; m + 4 <= me; m += 4) {
                hnk::XorPopcountBlock_4xN(
                    a + (m + 0) * Kp, a + (m + 1) * Kp,
                    a + (m + 2) * Kp, a + (m + 3) * Kp,
                    b + n_start * Kp, Kp, n_cnt, K_logical,
                    c + m * N + n_start, /*ldC=*/N);
            }
            for (; m < me; ++m) {
                hnk::XorPopcountRow(
                    a + m * Kp, b + n_start * Kp, Kp, n_cnt, K_logical,
                    c + m * N + n_start);
            }
        }
    });
    return C;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
bsr_scan(const at::Tensor& q, const at::Tensor& assoc,
         const at::Tensor& decay_shifts, int64_t D) {
    TORCH_CHECK(q.dim() == 3 && assoc.dim() == 3,
                "bsr_scan: q and assoc must be packed tensors with shape (B, n, Kp)");
    TORCH_CHECK(q.sizes() == assoc.sizes(), "bsr_scan: q/assoc shape mismatch");
    TORCH_CHECK(q.scalar_type() == at::kLong && assoc.scalar_type() == at::kLong,
                "bsr_scan: q and assoc must be int64 packed buffers");
    TORCH_CHECK(decay_shifts.dim() == 1 && decay_shifts.scalar_type() == at::kInt,
                "bsr_scan: decay_shifts must be a 1-D int32 tensor");
    TORCH_CHECK(D >= 0, "bsr_scan: D must be non-negative");

    const auto qc = q.contiguous();
    const auto ac = assoc.contiguous();
    const auto sc = decay_shifts.contiguous();
    const int64_t B = qc.size(0), n = qc.size(1), Kp = qc.size(2);
    TORCH_CHECK(Kp == (D + PACK_WIDTH - 1) / PACK_WIDTH,
                "bsr_scan: packed width does not match D");
    const int64_t groups = sc.numel();
    TORCH_CHECK(groups > 0, "bsr_scan: decay_shifts must be non-empty");

    auto read = at::zeros(qc.sizes(), qc.options());
    auto state = at::zeros(qc.sizes(), qc.options());
    auto gate = at::zeros(qc.sizes(), qc.options());
    if (B == 0 || n == 0 || Kp == 0 || D == 0) {
        return {read, state, gate};
    }

    const uint64_t* q_ptr = reinterpret_cast<const uint64_t*>(qc.data_ptr<int64_t>());
    const uint64_t* a_ptr = reinterpret_cast<const uint64_t*>(ac.data_ptr<int64_t>());
    const int32_t* shifts = sc.data_ptr<int32_t>();
    uint64_t* r_ptr = reinterpret_cast<uint64_t*>(read.data_ptr<int64_t>());
    uint64_t* s_ptr = reinterpret_cast<uint64_t*>(state.data_ptr<int64_t>());
    uint64_t* g_ptr = reinterpret_cast<uint64_t*>(gate.data_ptr<int64_t>());

    at::parallel_for(0, B * Kp, ROW_GRAIN, [&](int64_t begin, int64_t end) {
        for (int64_t item = begin; item < end; ++item) {
            const int64_t b = item / Kp;
            const int64_t w = item - b * Kp;
            for (int64_t lane = 0; lane < PACK_WIDTH; ++lane) {
                const int64_t d = w * PACK_WIDTH + lane;
                if (d >= D) break;
                const int64_t group = (d * groups) / std::max<int64_t>(D, 1);
                const int32_t shift = shifts[group];
                const uint64_t mask = uint64_t(1) << lane;
                int32_t A = 0;
                for (int64_t t = 0; t < n; ++t) {
                    const int64_t off = (b * n + t) * Kp + w;
                    const bool q_bit = (q_ptr[off] & mask) != 0;
                    const bool assoc_bit = (a_ptr[off] & mask) != 0;
                    const bool state_bit = A >= 0;
                    if (q_bit == state_bit) r_ptr[off] |= mask;
                    if (state_bit) s_ptr[off] |= mask;
                    const bool gate_bit = state_bit != assoc_bit;
                    if (gate_bit) g_ptr[off] |= mask;
                    int32_t decayed = A;
                    if (shift > 0) decayed = A - (A >> shift);
                    const int32_t update = gate_bit ? (assoc_bit ? 1 : -1) : 0;
                    A = decayed + update;
                }
            }
        }
    });
    return {read, state, gate};
}

//  popcount — per-element, int32 output. Supports any integer dtype + bool.
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

//  packed_popcount — total 1-bit count across the whole buffer (int64 scalar).
//  Pad bits are 0 by construction.
at::Tensor packed_popcount(const at::Tensor& x) {
    const auto p = x.contiguous();
    const uint64_t total = hnk::TotalPopcountBytes(p.data_ptr(), (size_t)p.nbytes());
    return at::scalar_tensor((int64_t)total, at::kLong);
}

//  hamming_distance — per-element popcount(A ^ B). int32 out. TensorIterator
//  in at::bitwise_xor handles shape broadcasting + strides.
at::Tensor hamming_distance(const at::Tensor& A, const at::Tensor& B) {
    return popcount(at::bitwise_xor(A, B));
}

//  bit1_hamming_total — fused total popcount(A ^ B) over equal-shape buffers.
//  No XOR temporary; in-register fusion.
at::Tensor bit1_hamming_total(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.sizes()       == B.sizes(),       "bit1_hamming_total: shape mismatch");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "bit1_hamming_total: dtype mismatch");
    const auto Ac = A.contiguous();
    const auto Bc = B.contiguous();
    const uint64_t total = hnk::TotalPopcountXor(Ac.data_ptr(), Bc.data_ptr(),
                                                 (size_t)Ac.nbytes());
    return at::scalar_tensor((int64_t)total, at::kLong);
}

at::Tensor& randomize_bits(at::Tensor& out) {
    out.random_();
    return out;
}

}} // cbrute::cpu
