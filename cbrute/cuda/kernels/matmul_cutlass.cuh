// CUTLASS-backed B1 XOR-popcount GEMM.
//
//   A: (M, K_bits)  RowMajor    uint1b_t   — reinterpret of our packed buffer
//   B: (K_bits, N)  ColumnMajor uint1b_t   — same memory as (N, K_bits) RowMajor,
//                                            which is how our B is stored.
//   C: (M, N)       RowMajor    int32      — Hamming distance per (m, n) pair.
//
// CUTLASS computes H[m,n] = Σ_k popc(A[m,k] ⊕ B[k,n]) over the bit-K dim.
// Mapping to our existing semantic:
//
//     C_brute[m,n] = K_logical − 2 * H[m,n]
//
// (Pad bits in both operands are zero, so they XOR to zero and contribute
// nothing — popc_xor over the padded buffer equals popc_xor over the logical
// bits.) The (K − 2·) transform is applied by a tiny epilogue kernel
// `k_kminus2_inplace` below.
//
// Requirements for this path:
//   * Device compute capability ≥ 8.0 (Ampere).
//   * K (in bits) must be a multiple of 256 — the B1 MMA instruction's K tile.
//     Caller is responsible for falling back otherwise.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Only enable when CUTLASS is being compiled (host-driver gate selects per call).
#if defined(CUTLASS_ARCH_MMA_B1_XOR_SM80_ENABLED) || 1
// We always include the GEMM template — selection happens at runtime in the
// driver. If the executing device lacks the instruction the kernel will assert
// at launch time, but we never reach that point because the driver checks cc.
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#endif

namespace cbrute { namespace cuda { namespace kernels {

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
    cutlass::epilogue::thread::LinearCombination<
        int32_t,
        128 / cutlass::sizeof_bits<int32_t>::value,      // = 4 ints / vector
        int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,                                                   // Stages
    128, 128,                                            // AlignmentA, B (bits)
    false,                                               // SplitKSerial
    cutlass::arch::OpXorPopc>;

// Tiny epilogue: C[i] = K - 2*C[i]  (turns Hamming distance into our bipolar
// dot-product output). Memory bandwidth bound — single pass over M*N int32.
__global__ inline void k_kminus2_inplace(int32_t* __restrict__ C,
                                         int64_t n_elems, int32_t K) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) C[i] = K - 2 * C[i];
}

}}}  // namespace cbrute::cuda::kernels
