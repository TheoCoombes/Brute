// Hand-tuned XNOR-popcount matmul, used on pre-sm_80 GPUs or when the K
// alignment requirements of the CUTLASS B1 path are not met.
//
// Output semantics (identical to CPU):
//
//   C[m,n] = 2 * popc_xnor(A_row_m, B_row_n) − K_eff
//          = K_logical − 2 * popc_xor(A_row_m, B_row_n)        (since pad bits are 0)
//
//   where K_eff = 2 * Kp * pw − K_logical.
//
// 16×16 block tile. Each thread computes one output element by streaming the
// K dimension in registers — no shared memory tiling. Vectorized loads via
// uint4 (16 bytes) where the word stride permits. Adequate for the fallback
// case; the CUTLASS path is the hot one on sm_80+.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda { namespace kernels {

template <typename T>
__device__ inline uint64_t bit_popcnt(T x);
template <> __device__ inline uint64_t bit_popcnt<uint8_t> (uint8_t  x) { return __popc((unsigned)x); }
template <> __device__ inline uint64_t bit_popcnt<uint32_t>(uint32_t x) { return __popc(x); }
template <> __device__ inline uint64_t bit_popcnt<uint64_t>(uint64_t x) { return __popcll(x); }

template <typename T>
__global__ void k_xnor_popcount_matmul(const T* __restrict__ A,
                                       const T* __restrict__ B,
                                       int32_t* __restrict__ C,
                                       int64_t M, int64_t N, int64_t Kp,
                                       int32_t K_eff) {
    const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const T* a_row = A + m * Kp;
    const T* b_row = B + n * Kp;

    uint64_t acc = 0;
    int64_t k = 0;
    // 4-way unrolled main loop for ILP — the popc instructions don't depend
    // on each other so the scheduler can overlap them.
    for (; k + 3 < Kp; k += 4) {
        acc += bit_popcnt<T>((T)~(a_row[k]   ^ b_row[k]));
        acc += bit_popcnt<T>((T)~(a_row[k+1] ^ b_row[k+1]));
        acc += bit_popcnt<T>((T)~(a_row[k+2] ^ b_row[k+2]));
        acc += bit_popcnt<T>((T)~(a_row[k+3] ^ b_row[k+3]));
    }
    for (; k < Kp; ++k) {
        acc += bit_popcnt<T>((T)~(a_row[k] ^ b_row[k]));
    }
    C[m * N + n] = 2 * (int32_t)acc - K_eff;
}

}}}  // namespace cbrute::cuda::kernels
