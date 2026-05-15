// CUTLASS-backed B1 XOR-popcount GEMM with fused (K − 2H) epilogue.
//
//   A: (M, K_bits)  RowMajor    uint1b_t   — reinterpret of our int64 packed buffer
//   B: (K_bits, N)  ColumnMajor uint1b_t   — same memory as (N, K_bits) RowMajor,
//                                            which is how our B is stored.
//   C: (M, N)       RowMajor    int32      — final output: K_logical − 2·H[m,n].
//
// CUTLASS naturally computes H[m,n] = Σ_k popc(A[m,k] ⊕ B[k,n]). The custom
// epilogue thread op below maps H → K_logical − 2·H *in-register*, eliminating
// the separate post-pass kernel the old design needed. Pad bits in both
// operands are zero by construction, so they XOR to zero and don't perturb H.
//
// Requirements for this path:
//   * Device compute capability ≥ 8.0 (Ampere).
//   * K (in bits) must be a multiple of 256 — the B1 MMA K-tile. Caller pads.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/scale_type.h"

namespace cbrute { namespace cuda { namespace kernels {

// Custom epilogue thread op: D[i] = K_logical − 2·accumulator[i].
//
// `is_source_needed()` and `kScale = ScaleType::Nothing` together tell CUTLASS
// to skip the source-tensor load entirely — we never read the prior C value.
// API surface is a drop-in replacement for `LinearCombination`.
template <typename ElementOutput_, int Count_,
          typename ElementAccumulator_ = ElementOutput_,
          typename ElementCompute_     = ElementAccumulator_>
struct KMinusTwoH {
    using ElementOutput      = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute     = ElementCompute_;
    using FragmentOutput      = cutlass::Array<ElementOutput, Count_>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, Count_>;
    using FragmentCompute     = cutlass::Array<ElementCompute, Count_>;

    static int const kCount = Count_;
    static cutlass::epilogue::thread::ScaleType::Kind const kScale =
        cutlass::epilogue::thread::ScaleType::Nothing;
    static cutlass::FloatRoundStyle const kRound =
        cutlass::FloatRoundStyle::round_to_nearest;

    struct Params {
        ElementCompute K_logical;
        CUTLASS_HOST_DEVICE Params() : K_logical(0) {}
        CUTLASS_HOST_DEVICE Params(ElementCompute K) : K_logical(K) {}
    };

    ElementCompute K_logical;

    CUTLASS_HOST_DEVICE
    KMinusTwoH(Params const& p) : K_logical(p.K_logical) {}

    CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
    CUTLASS_HOST_DEVICE void set_k_partition(int /*k*/, int /*kpc*/) {}

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accum) const {
        FragmentOutput out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Count_; ++i) {
            ElementCompute h = ElementCompute(accum[i]);
            out[i] = ElementOutput(K_logical - ElementCompute(2) * h);
        }
        return out;
    }

    // source-needed variant — unused (kScale = Nothing), but the CUTLASS
    // epilogue interface requires it to compile.
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accum,
                              FragmentOutput const&) const {
        return (*this)(accum);
    }
};

// Threadblock / warp / instruction tile choices — match the standard B1 sm_80
// configuration used in cutlass/test/unit/gemm/device/...sm80.cu.
using CutlassB1XorGemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor,        // A
    cutlass::uint1b_t, cutlass::layout::ColumnMajor,     // B
    int32_t,           cutlass::layout::RowMajor,        // C / D
    int32_t,                                             // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 1024>,            // ThreadBlock
    cutlass::gemm::GemmShape<64, 64, 1024>,              // Warp
    cutlass::gemm::GemmShape<16, 8, 256>,                // Instruction
    KMinusTwoH<int32_t, 128 / cutlass::sizeof_bits<int32_t>::value,
               int32_t, int32_t>,                        // Fused K-2H epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,                                                   // Stages
    128, 128,                                            // AlignmentA, B (bits)
    false,                                               // SplitKSerial
    cutlass::arch::OpXorPopc>;

}}}  // namespace cbrute::cuda::kernels
