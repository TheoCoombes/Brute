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

at::Tensor ternary_bit1_matmul(const at::Tensor& A, const at::Tensor& B, int64_t N) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "ternary_bit1_matmul: inputs must be 2-D");
    TORCH_CHECK(B.scalar_type() == at::kLong,
                "ternary_bit1_matmul: B must be int64 packed storage");
    TORCH_CHECK(A.scalar_type() == at::kChar || A.scalar_type() == at::kShort
                || A.scalar_type() == at::kInt || A.scalar_type() == at::kFloat,
                "ternary_bit1_matmul: A must be int8, int16, int32, or float32");
    const int64_t M = A.size(0), K = A.size(1), Kb = B.size(0), Kp = B.size(1);
    TORCH_CHECK(K == Kb, "ternary_bit1_matmul: reduction dimension mismatch");
    TORCH_CHECK(N >= 0, "ternary_bit1_matmul: N must be non-negative");
    TORCH_CHECK(Kp == (N + PACK_WIDTH - 1) / PACK_WIDTH,
                "ternary_bit1_matmul: packed width does not match N");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C = at::empty({M, N}, at::kInt);
    if (M == 0 || N == 0) return C;

    const uint64_t* b = reinterpret_cast<const uint64_t*>(Bc.data_ptr<int64_t>());
    int32_t* c = C.data_ptr<int32_t>();

    auto run = [&](auto* a) {
        at::parallel_for(0, M, ROW_GRAIN, [&](int64_t begin, int64_t end) {
            for (int64_t m = begin; m < end; ++m) {
                for (int64_t w = 0; w < Kp; ++w) {
                    const int64_t live = std::min<int64_t>(PACK_WIDTH, N - w * PACK_WIDTH);
                    for (int64_t lane = 0; lane < live; ++lane) {
                        const uint64_t mask = uint64_t(1) << lane;
                        int32_t sum = 0;
                        for (int64_t k = 0; k < K; ++k) {
                            const int32_t av = static_cast<int32_t>(a[m * K + k]);
                            if (av == 0) continue;
                            const bool bit = (b[k * Kp + w] & mask) != 0;
                            sum += bit ? av : -av;
                        }
                        c[m * N + w * PACK_WIDTH + lane] = sum;
                    }
                }
            }
        });
    };

    if (A.scalar_type() == at::kChar) {
        run(Ac.data_ptr<int8_t>());
    } else if (A.scalar_type() == at::kShort) {
        run(Ac.data_ptr<int16_t>());
    } else if (A.scalar_type() == at::kInt) {
        run(Ac.data_ptr<int32_t>());
    } else {
        run(Ac.data_ptr<float>());
    }
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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
bsr_delta_scan(const at::Tensor& q, const at::Tensor& assoc,
               const at::Tensor& decay_shift_by_dim,
               const at::Tensor& erase_by_dim,
               const at::Tensor& write_by_dim,
               int64_t state_clip, int64_t D) {
    TORCH_CHECK(q.dim() == 3 && assoc.dim() == 3,
                "bsr_delta_scan: q and assoc must be packed tensors with shape (B, n, Kp)");
    TORCH_CHECK(q.sizes() == assoc.sizes(), "bsr_delta_scan: q/assoc shape mismatch");
    TORCH_CHECK(q.scalar_type() == at::kLong && assoc.scalar_type() == at::kLong,
                "bsr_delta_scan: q and assoc must be int64 packed buffers");
    TORCH_CHECK(decay_shift_by_dim.dim() == 1 && decay_shift_by_dim.scalar_type() == at::kInt,
                "bsr_delta_scan: decay_shift_by_dim must be a 1-D int32 tensor");
    TORCH_CHECK(erase_by_dim.dim() == 1 && write_by_dim.dim() == 1,
                "bsr_delta_scan: erase/write must be 1-D tensors");
    TORCH_CHECK(erase_by_dim.scalar_type() == at::kChar || erase_by_dim.scalar_type() == at::kInt,
                "bsr_delta_scan: erase_by_dim must be int8 or int32");
    TORCH_CHECK(write_by_dim.scalar_type() == at::kChar || write_by_dim.scalar_type() == at::kInt,
                "bsr_delta_scan: write_by_dim must be int8 or int32");
    TORCH_CHECK(D >= 0, "bsr_delta_scan: D must be non-negative");
    TORCH_CHECK(state_clip > 0, "bsr_delta_scan: state_clip must be positive");
    TORCH_CHECK(decay_shift_by_dim.numel() == D && erase_by_dim.numel() == D
                && write_by_dim.numel() == D,
                "bsr_delta_scan: per-dimension tensors must have length D");

    const auto qc = q.contiguous();
    const auto ac = assoc.contiguous();
    const auto sc = decay_shift_by_dim.contiguous();
    const auto ec = erase_by_dim.contiguous();
    const auto wc = write_by_dim.contiguous();
    const int64_t B = qc.size(0), n = qc.size(1), Kp = qc.size(2);
    TORCH_CHECK(Kp == (D + PACK_WIDTH - 1) / PACK_WIDTH,
                "bsr_delta_scan: packed width does not match D");

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

    auto run = [&](auto* erase, auto* write) {
        at::parallel_for(0, B * Kp, ROW_GRAIN, [&](int64_t begin, int64_t end) {
            for (int64_t item = begin; item < end; ++item) {
                const int64_t b = item / Kp;
                const int64_t w = item - b * Kp;
                for (int64_t lane = 0; lane < PACK_WIDTH; ++lane) {
                    const int64_t d = w * PACK_WIDTH + lane;
                    if (d >= D) break;
                    const int32_t shift = shifts[d];
                    const int32_t erase_v = static_cast<int32_t>(erase[d]);
                    const int32_t write_v = static_cast<int32_t>(write[d]);
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
                        if (gate_bit) {
                            const int32_t S = state_bit ? 1 : -1;
                            const int32_t assoc_pm1 = assoc_bit ? 1 : -1;
                            decayed = decayed - erase_v * S + write_v * assoc_pm1;
                        }
                        A = std::max<int32_t>(
                            -static_cast<int32_t>(state_clip),
                            std::min<int32_t>(static_cast<int32_t>(state_clip), decayed));
                    }
                }
            }
        });
    };

    if (erase_by_dim.scalar_type() == at::kChar && write_by_dim.scalar_type() == at::kChar) {
        run(ec.data_ptr<int8_t>(), wc.data_ptr<int8_t>());
    } else if (erase_by_dim.scalar_type() == at::kChar && write_by_dim.scalar_type() == at::kInt) {
        run(ec.data_ptr<int8_t>(), wc.data_ptr<int32_t>());
    } else if (erase_by_dim.scalar_type() == at::kInt && write_by_dim.scalar_type() == at::kChar) {
        run(ec.data_ptr<int32_t>(), wc.data_ptr<int8_t>());
    } else {
        run(ec.data_ptr<int32_t>(), wc.data_ptr<int32_t>());
    }
    return {read, state, gate};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
bold_update_packed(const at::Tensor& packed, const at::Tensor& m,
                   const at::Tensor& q, int64_t beta_num, int64_t beta_den,
                   int64_t eta, int64_t threshold, int64_t m_clip, int64_t D) {
    TORCH_CHECK(packed.dim() >= 1, "bold_update_packed: packed must have >= 1 dim");
    TORCH_CHECK(packed.scalar_type() == at::kLong,
                "bold_update_packed: packed must be int64 packed storage");
    TORCH_CHECK(m.sizes() == q.sizes(), "bold_update_packed: m/q shape mismatch");
    TORCH_CHECK(m.dim() >= 1, "bold_update_packed: m must have >= 1 dim");
    TORCH_CHECK(m.size(-1) == D, "bold_update_packed: m last dim must equal D");
    TORCH_CHECK(q.size(-1) == D, "bold_update_packed: q last dim must equal D");
    TORCH_CHECK(m.scalar_type() == at::kShort || m.scalar_type() == at::kInt,
                "bold_update_packed: m must be int16 or int32");
    TORCH_CHECK(q.scalar_type() == at::kShort || q.scalar_type() == at::kInt,
                "bold_update_packed: q must be int16 or int32");
    TORCH_CHECK(beta_den > 0, "bold_update_packed: beta_den must be positive");
    TORCH_CHECK(eta >= 0, "bold_update_packed: eta must be non-negative");
    TORCH_CHECK(threshold > 0, "bold_update_packed: threshold must be positive");
    TORCH_CHECK(m_clip > 0, "bold_update_packed: m_clip must be positive");
    TORCH_CHECK(D >= 0, "bold_update_packed: D must be non-negative");

    const int64_t Kp = packed.size(-1);
    TORCH_CHECK(Kp == (D + PACK_WIDTH - 1) / PACK_WIDTH,
                "bold_update_packed: packed width does not match D");
    const int64_t rows = (D == 0) ? 0 : m.numel() / D;
    TORCH_CHECK(packed.numel() == rows * Kp,
                "bold_update_packed: packed/logical shape mismatch");

    auto pc = packed.contiguous();
    auto mc = m.contiguous();
    auto qc = q.contiguous();
    auto out_packed = pc.clone();
    auto out_m = at::empty_like(mc);
    auto counts = at::zeros({at::get_num_threads()}, at::kLong);

    const uint64_t* p_in = reinterpret_cast<const uint64_t*>(pc.data_ptr<int64_t>());
    uint64_t* p_out = reinterpret_cast<uint64_t*>(out_packed.data_ptr<int64_t>());

    auto run = [&](auto* m_in, auto* q_in, auto* m_out) {
        using m_t = std::remove_pointer_t<decltype(m_in)>;
        using q_t = std::remove_pointer_t<decltype(q_in)>;
        at::parallel_for(0, rows, ROW_GRAIN, [&](int64_t begin, int64_t end) {
            int thread_idx = at::get_thread_num();
            int64_t local_flips = 0;
            for (int64_t r = begin; r < end; ++r) {
                const int64_t base = r * D;
                const int64_t pbase = r * Kp;
                for (int64_t w = 0; w < Kp; ++w) {
                    const int64_t live = std::min<int64_t>(PACK_WIDTH, D - w * PACK_WIDTH);
                    uint64_t word = p_in[pbase + w];
                    for (int64_t lane = 0; lane < live; ++lane) {
                        const int64_t idx = base + w * PACK_WIDTH + lane;
                        int64_t acc = (static_cast<int64_t>(m_in[idx]) * beta_num) / beta_den;
                        acc += static_cast<int64_t>(q_in[idx]) * eta;
                        acc = std::max<int64_t>(-m_clip, std::min<int64_t>(m_clip, acc));

                        const bool bit = ((word >> lane) & uint64_t(1)) != 0;
                        const int64_t aligned = bit ? acc : -acc;
                        if (aligned >= threshold) {
                            word ^= (uint64_t(1) << lane);
                            acc = 0;
                            ++local_flips;
                        }
                        m_out[idx] = static_cast<m_t>(acc);
                    }
                    p_out[pbase + w] = word;
                }
            }
            counts.data_ptr<int64_t>()[thread_idx] += local_flips;
        });
    };

    if (m.scalar_type() == at::kShort && q.scalar_type() == at::kShort) {
        run(mc.data_ptr<int16_t>(), qc.data_ptr<int16_t>(), out_m.data_ptr<int16_t>());
    } else if (m.scalar_type() == at::kShort && q.scalar_type() == at::kInt) {
        run(mc.data_ptr<int16_t>(), qc.data_ptr<int32_t>(), out_m.data_ptr<int16_t>());
    } else if (m.scalar_type() == at::kInt && q.scalar_type() == at::kShort) {
        run(mc.data_ptr<int32_t>(), qc.data_ptr<int16_t>(), out_m.data_ptr<int32_t>());
    } else {
        run(mc.data_ptr<int32_t>(), qc.data_ptr<int32_t>(), out_m.data_ptr<int32_t>());
    }

    return {out_packed, out_m, counts.sum()};
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

// ── Fused sign ops ──────────────────────────────────────────────────────────

// xnor_popcount_matmul_sign — like xnor_popcount_matmul but the epilogue
// packs sign(K - 2H) directly into bit1 output (M, ceil(N/64)) instead of
// storing int32. Eliminates the intermediate int32 buffer + separate pack_sign
// call in BooleanLinear when boundary_nu is None.
at::Tensor xnor_popcount_matmul_sign(const at::Tensor& A, const at::Tensor& B, int64_t K) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "xnor_popcount_matmul_sign: inputs must be 2-D");
    const int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    TORCH_CHECK(Kp == B.size(1), "xnor_popcount_matmul_sign: packed K mismatch");
    TORCH_CHECK(A.scalar_type() == at::kLong && B.scalar_type() == at::kLong,
                "xnor_popcount_matmul_sign: inputs must be int64 packed buffers");

    const int64_t Np = (N + PACK_WIDTH - 1) / PACK_WIDTH;
    const int32_t K_logical = (int32_t)K;

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C  = at::zeros({M, Np}, at::kLong);

    const uint64_t* a = reinterpret_cast<const uint64_t*>(Ac.data_ptr<int64_t>());
    const uint64_t* b = reinterpret_cast<const uint64_t*>(Bc.data_ptr<int64_t>());
    uint64_t*       c = reinterpret_cast<uint64_t*>(C.data_ptr<int64_t>());

    // Process M rows in MATMUL_M_GRAIN chunks.  For each row we iterate over N
    // output bits in packs of 64 (one output word each), reusing the
    // Highway-vectorised XorPopcountPair for the K inner loop.
    at::parallel_for(0, M, MATMUL_M_GRAIN, [&](int64_t ms, int64_t me) {
        for (int64_t m = ms; m < me; ++m) {
            const uint64_t* a_row = a + m * Kp;
            uint64_t*       c_row = c + m * Np;
            for (int64_t nw = 0; nw < Np; ++nw) {
                uint64_t word = 0;
                const int64_t n_base = nw * PACK_WIDTH;
                const int64_t n_end  = std::min(n_base + PACK_WIDTH, N);
                for (int64_t n = n_base; n < n_end; ++n) {
                    const int32_t sim = hnk::XorPopcountPair(
                        a_row, b + n * Kp, Kp, K_logical);
                    if (sim >= 0) word |= (uint64_t(1) << (n - n_base));
                }
                c_row[nw] = word;
            }
        }
    });
    return C;
}

namespace {
// Bit-parallel ≥ threshold comparator on n_bits partial-sum words.
// partial[j] holds the j-th bit of the per-lane count (j=0 → LSB).
// Returns a 64-bit mask: bit b is 1 iff count[b] >= threshold.
inline uint64_t count_geq_threshold(const uint64_t* partial, int n_bits, int threshold) {
    uint64_t greater = 0;
    uint64_t equal   = ~uint64_t(0);
    for (int bit = n_bits - 1; bit >= 0; --bit) {
        const int t_bit = (threshold >> bit) & 1;
        if (t_bit == 0) {
            greater |= (equal & partial[bit]);
            equal   &= ~partial[bit];
        } else {
            equal &= partial[bit];
        }
    }
    return greater | equal;
}
} // anon

// packed_majority — bit-sliced majority vote over k packed rows.
// rows: (batch, k, Kp) int64.  Output: (batch, Kp) int64 packed bit1.
// Output bit b of word w for batch element b is 1 iff more than k/2
// of the k input rows have that bit set (odd k ⇒ no ties).
at::Tensor packed_majority(const at::Tensor& rows, int64_t k, int64_t D) {
    TORCH_CHECK(rows.dim() == 3,
                "packed_majority: rows must be 3-D (batch, k, Kp)");
    TORCH_CHECK(rows.scalar_type() == at::kLong,
                "packed_majority: rows must be int64");
    TORCH_CHECK(k > 0, "packed_majority: k must be positive");
    TORCH_CHECK(D > 0, "packed_majority: D must be positive");

    const int64_t batch = rows.size(0);
    const int64_t Kp    = rows.size(2);
    TORCH_CHECK(rows.size(1) == k, "packed_majority: dim 1 must equal k");
    TORCH_CHECK(Kp == (D + PACK_WIDTH - 1) / PACK_WIDTH,
                "packed_majority: Kp mismatch with D");

    // Number of bits needed to represent counts 0..k.
    const int n_bits = (k == 1) ? 1
        : static_cast<int>(std::ceil(std::log2(static_cast<double>(k) + 1.0)));
    TORCH_CHECK(n_bits <= 8, "packed_majority: k too large (n_bits > 8)");
    const int threshold = static_cast<int>(k / 2) + 1;  // (k+1)/2 for odd k

    const uint64_t pad_mask = (D % PACK_WIDTH == 0)
        ? ~uint64_t(0) : ((uint64_t(1) << (D % PACK_WIDTH)) - 1);

    auto rc  = rows.contiguous();
    auto out = at::zeros({batch, Kp}, at::kLong);

    const uint64_t* r = reinterpret_cast<const uint64_t*>(rc.data_ptr<int64_t>());
    uint64_t*       o = reinterpret_cast<uint64_t*>(out.data_ptr<int64_t>());

    at::parallel_for(0, batch, ROW_GRAIN, [&](int64_t bs, int64_t be) {
        uint64_t partial[8];
        for (int64_t b = bs; b < be; ++b) {
            for (int64_t w = 0; w < Kp; ++w) {
                std::fill(partial, partial + n_bits, uint64_t(0));
                // Carry-ripple addition of k rows.
                for (int64_t ki = 0; ki < k; ++ki) {
                    uint64_t carry = r[(b * k + ki) * Kp + w];
                    for (int j = 0; j < n_bits && carry; ++j) {
                        const uint64_t s = partial[j] ^ carry;
                        carry = partial[j] & carry;
                        partial[j] = s;
                    }
                }
                uint64_t res = count_geq_threshold(partial, n_bits, threshold);
                if (w == Kp - 1) res &= pad_mask;
                o[b * Kp + w] = res;
            }
        }
    });
    return out;
}

// episodic_causal_search — fused windowed Hamming scan + top-1 + payload gather.
// Intended for the inference/streaming forward path where a query (qc, qp) is
// scored against a ring buffer of keys (kc_buf, pos_buf) of size cnt[b] per
// batch element. Returns the gathered read payload, the argmax index, and the
// best score.
//
// qc, qp: (B, Kp) int64 packed
// kc_buf, pos_buf, payload: (B, N, Kp) int64 packed
// cnt: (B,) int32 valid slot count (<= N)
// Returns: read (B, Kp), idx (B,) int32, score (B,) int32
std::tuple<at::Tensor, at::Tensor, at::Tensor>
episodic_causal_search(const at::Tensor& qc, const at::Tensor& kc_buf,
                        const at::Tensor& qp, const at::Tensor& pos_buf,
                        const at::Tensor& payload, const at::Tensor& cnt,
                        int64_t D) {
    TORCH_CHECK(qc.dim() == 2 && qp.dim() == 2,
                "episodic_causal_search: qc/qp must be 2-D (B, Kp)");
    TORCH_CHECK(kc_buf.dim() == 3 && pos_buf.dim() == 3 && payload.dim() == 3,
                "episodic_causal_search: buffers must be 3-D (B, N, Kp)");
    TORCH_CHECK(cnt.dim() == 1 && cnt.scalar_type() == at::kInt,
                "episodic_causal_search: cnt must be 1-D int32");
    TORCH_CHECK(D > 0, "episodic_causal_search: D must be positive");

    const int64_t B  = qc.size(0);
    const int64_t Kp = qc.size(1);
    TORCH_CHECK(qp.size(0) == B && qp.size(1) == Kp);
    TORCH_CHECK(kc_buf.size(0) == B && kc_buf.size(2) == Kp);
    TORCH_CHECK(pos_buf.sizes() == kc_buf.sizes() && payload.sizes() == kc_buf.sizes());
    TORCH_CHECK(cnt.size(0) == B);

    const int64_t N = kc_buf.size(1);
    const int32_t K_logical = (int32_t)D;

    auto qcc   = qc.contiguous();
    auto qpc   = qp.contiguous();
    auto kcc   = kc_buf.contiguous();
    auto psc   = pos_buf.contiguous();
    auto payc  = payload.contiguous();
    auto cntc  = cnt.contiguous();

    auto read      = at::zeros({B, Kp}, at::kLong);
    auto idx_out   = at::full({B}, int32_t(-1), at::kInt);
    auto score_out = at::full({B}, int32_t(-(int32_t)D * 2 - 2), at::kInt);

    const uint64_t* qc_ptr  = reinterpret_cast<const uint64_t*>(qcc.data_ptr<int64_t>());
    const uint64_t* qp_ptr  = reinterpret_cast<const uint64_t*>(qpc.data_ptr<int64_t>());
    const uint64_t* kc_ptr  = reinterpret_cast<const uint64_t*>(kcc.data_ptr<int64_t>());
    const uint64_t* ps_ptr  = reinterpret_cast<const uint64_t*>(psc.data_ptr<int64_t>());
    const uint64_t* pay_ptr = reinterpret_cast<const uint64_t*>(payc.data_ptr<int64_t>());
    const int32_t*  cnt_ptr = cntc.data_ptr<int32_t>();
    uint64_t*       r_ptr   = reinterpret_cast<uint64_t*>(read.data_ptr<int64_t>());
    int32_t*        i_ptr   = idx_out.data_ptr<int32_t>();
    int32_t*        s_ptr   = score_out.data_ptr<int32_t>();

    at::parallel_for(0, B, ROW_GRAIN, [&](int64_t bs, int64_t be) {
        for (int64_t b = bs; b < be; ++b) {
            const int32_t valid = cnt_ptr[b];
            if (valid <= 0) continue;

            const uint64_t* qc_row  = qc_ptr  + b * Kp;
            const uint64_t* qp_row  = qp_ptr  + b * Kp;
            const uint64_t* kc_rows = kc_ptr  + b * N * Kp;
            const uint64_t* ps_rows = ps_ptr  + b * N * Kp;
            const uint64_t* pa_rows = pay_ptr + b * N * Kp;

            int32_t best = INT32_MIN;
            int32_t best_i = -1;
            for (int32_t i = 0; i < valid; ++i) {
                const int32_t cs = hnk::XorPopcountPair(
                    qc_row, kc_rows + i * Kp, Kp, K_logical);
                const int32_t ps = hnk::XorPopcountPair(
                    qp_row, ps_rows + i * Kp, Kp, K_logical);
                const int32_t total = cs + ps;
                if (total > best) { best = total; best_i = i; }
            }

            i_ptr[b] = best_i;
            s_ptr[b] = best;
            if (best_i >= 0) {
                const uint64_t* src = pa_rows + (int64_t)best_i * Kp;
                uint64_t*       dst = r_ptr   + b * Kp;
                for (int64_t w = 0; w < Kp; ++w) dst[w] = src[w];
            }
        }
    });
    return {read, idx_out, score_out};
}

}} // cbrute::cpu
