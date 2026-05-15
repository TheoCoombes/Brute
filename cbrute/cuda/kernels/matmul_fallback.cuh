// Hand-tuned XOR-popcount matmul for bit1 (uint64-packed) tensors.
// Used on pre-sm_80 GPUs and when the CUTLASS K-alignment requirements are
// not met (Kp_bits < 256 bits / Kp < 4 uint64 words).
//
//   A: (M, Kp) uint64    B: (N, Kp) uint64    →    C: (M, N) int32
//   C[m,n] = K_logical − 2 · popc_xor(A[m], B[n])
//
// 16×16 block tile. Each thread computes one output element by streaming the
// K dimension in registers. 128-bit vectorised loads via ulonglong2 when
// Kp is even (8-byte alignment is guaranteed by PyTorch's 64-byte allocator,
// so 16-byte alignment follows from Kp%2==0).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda { namespace kernels {

__global__ inline void k_xnor_popcount_matmul(const uint64_t* __restrict__ A,
                                              const uint64_t* __restrict__ B,
                                              int32_t* __restrict__ C,
                                              int64_t M, int64_t N, int64_t Kp,
                                              int32_t K_logical) {
    const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const uint64_t* a_row = A + m * Kp;
    const uint64_t* b_row = B + n * Kp;

    uint64_t acc = 0;
    int64_t k = 0;

    // 128-bit vectorised path (ulonglong2): 2 uint64 words per load.
    if ((Kp & 1) == 0) {
        const ulonglong2* a2 = reinterpret_cast<const ulonglong2*>(a_row);
        const ulonglong2* b2 = reinterpret_cast<const ulonglong2*>(b_row);
        const int64_t Kp2 = Kp / 2;
        for (int64_t w = 0; w < Kp2; ++w) {
            const ulonglong2 va = a2[w], vb = b2[w];
            acc += (uint64_t)__popcll(va.x ^ vb.x)
                 + (uint64_t)__popcll(va.y ^ vb.y);
        }
        k = Kp;
    } else {
        // 4-way scalar unroll for odd-Kp residuals.
        for (; k + 3 < Kp; k += 4) {
            acc += __popcll(a_row[k]   ^ b_row[k])
                 + __popcll(a_row[k+1] ^ b_row[k+1])
                 + __popcll(a_row[k+2] ^ b_row[k+2])
                 + __popcll(a_row[k+3] ^ b_row[k+3]);
        }
    }
    for (; k < Kp; ++k) acc += __popcll(a_row[k] ^ b_row[k]);

    // Pad bits XOR to 0, so popc_xor over padded buffer equals over logical.
    C[m * N + n] = K_logical - 2 * (int32_t)acc;
}

}}}  // namespace cbrute::cuda::kernels
