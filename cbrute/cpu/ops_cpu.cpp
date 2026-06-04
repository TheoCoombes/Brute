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
#include <cstdint>

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

//  signed_bundle — int8-weight × bit1 → bit1 reduction.
//    out[b,m,d] = sign( Σ_n W[b,m,n] · pm1(V[b,n,d]) ),  pm1(bit) = bit ? +1 : -1.
//  The signed vote accumulator is a *transient* per-(b,m) int32 register (D
//  entries); V is read straight from its packed words (no unpack), the output is
//  packed in place.  sign(0) = +1 (>= 0), matching ``sign_to_bit1``; and because
//  clamping preserves sign, this is bit-identical to the int8-clamped reference.
at::Tensor signed_bundle(const at::Tensor& W, const at::Tensor& V, int64_t D) {
    TORCH_CHECK(W.dim() == 3 && V.dim() == 3,
                "signed_bundle: W must be (B,M,N) and V packed (B,N,Dp)");
    TORCH_CHECK(W.scalar_type() == at::kChar, "signed_bundle: W must be int8");
    const int64_t B = W.size(0), M = W.size(1), N = W.size(2), Dp = V.size(2);
    TORCH_CHECK(V.size(0) == B && V.size(1) == N,
                "signed_bundle: W (B,M,N) / V (B,N,Dp) batch or N mismatch");
    const auto Wc = W.contiguous();
    const auto Vc = V.contiguous();
    auto out = at::zeros({B, M, Dp}, V.options().dtype(at::kLong));
    if (B * M == 0 || Dp == 0) return out;

    const int8_t*   w = Wc.data_ptr<int8_t>();
    const uint64_t* v = reinterpret_cast<const uint64_t*>(Vc.data_ptr());
    uint64_t*       o = reinterpret_cast<uint64_t*>(out.data_ptr());
    const int64_t Dl = D;

    at::parallel_for(0, B * M, ROW_GRAIN, [&](int64_t s, int64_t e) {
        std::vector<int32_t> acc((size_t)(Dp * PACK_WIDTH));   // one alloc per task
        for (int64_t bm = s; bm < e; ++bm) {
            const int64_t b = bm / M, m = bm % M;
            std::fill(acc.begin(), acc.end(), 0);
            const int8_t*   wrow = w + (b * M + m) * N;
            const uint64_t* vbat = v + b * N * Dp;
            for (int64_t nn = 0; nn < N; ++nn) {
                const int32_t wv = (int32_t)wrow[nn];
                if (wv == 0) continue;
                const uint64_t* vrow = vbat + nn * Dp;
                for (int64_t dp = 0; dp < Dp; ++dp) {
                    const uint64_t word = vrow[dp];
                    int32_t* ap = acc.data() + dp * PACK_WIDTH;
                    for (int bit = 0; bit < PACK_WIDTH; ++bit)
                        ap[bit] += ((word >> bit) & 1ULL) ? wv : -wv;
                }
            }
            uint64_t* orow = o + (b * M + m) * Dp;
            for (int64_t dp = 0; dp < Dp; ++dp) {
                const int32_t* ap = acc.data() + dp * PACK_WIDTH;
                const int64_t base = dp * PACK_WIDTH;
                uint64_t outw = 0;
                for (int bit = 0; bit < PACK_WIDTH; ++bit) {
                    if (base + bit >= Dl) break;            // pad bits stay 0
                    if (ap[bit] >= 0) outw |= (uint64_t(1) << bit);
                }
                orow[dp] = outw;
            }
        }
    });
    return out;
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
