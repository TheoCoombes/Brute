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

    // 128-bit vectorized loads: each load fetches 4×uint32 or 2×uint64 in a
    // single instruction, halving/quartering the number of global load ops.
    // Alignment is guaranteed when Kp × sizeof(T) is a multiple of 16 bytes
    // (i.e., Kp%4==0 for uint32, Kp%2==0 for uint64); PyTorch allocates
    // 64-byte aligned, so row-start alignment follows from that condition.
    // When the condition is not met, fall through to the 4-way scalar path.
    if constexpr (sizeof(T) == 4) {
        if (Kp % 4 == 0) {
            const uint4* a4 = reinterpret_cast<const uint4*>(a_row);
            const uint4* b4 = reinterpret_cast<const uint4*>(b_row);
            const int64_t Kp4 = Kp / 4;
            for (int64_t w = 0; w < Kp4; ++w) {
                const uint4 va = a4[w], vb = b4[w];
                acc += (uint64_t)__popc(~(va.x ^ vb.x))
                     + (uint64_t)__popc(~(va.y ^ vb.y))
                     + (uint64_t)__popc(~(va.z ^ vb.z))
                     + (uint64_t)__popc(~(va.w ^ vb.w));
            }
            k = Kp;
        } else {
            for (; k + 3 < Kp; k += 4) {
                acc += __popc(~(a_row[k]   ^ b_row[k]))
                     + __popc(~(a_row[k+1] ^ b_row[k+1]))
                     + __popc(~(a_row[k+2] ^ b_row[k+2]))
                     + __popc(~(a_row[k+3] ^ b_row[k+3]));
            }
        }
    } else if constexpr (sizeof(T) == 8) {
        if (Kp % 2 == 0) {
            const ulonglong2* a2 = reinterpret_cast<const ulonglong2*>(a_row);
            const ulonglong2* b2 = reinterpret_cast<const ulonglong2*>(b_row);
            const int64_t Kp2 = Kp / 2;
            for (int64_t w = 0; w < Kp2; ++w) {
                const ulonglong2 va = a2[w], vb = b2[w];
                acc += (uint64_t)__popcll(~(va.x ^ vb.x))
                     + (uint64_t)__popcll(~(va.y ^ vb.y));
            }
            k = Kp;
        } else {
            for (; k + 3 < Kp; k += 4) {
                acc += bit_popcnt<T>((T)~(a_row[k]   ^ b_row[k]));
                acc += bit_popcnt<T>((T)~(a_row[k+1] ^ b_row[k+1]));
                acc += bit_popcnt<T>((T)~(a_row[k+2] ^ b_row[k+2]));
                acc += bit_popcnt<T>((T)~(a_row[k+3] ^ b_row[k+3]));
            }
        }
    } else {
        // uint8_t: 4-way scalar unroll
        for (; k + 3 < Kp; k += 4) {
            acc += bit_popcnt<T>((T)~(a_row[k]   ^ b_row[k]));
            acc += bit_popcnt<T>((T)~(a_row[k+1] ^ b_row[k+1]));
            acc += bit_popcnt<T>((T)~(a_row[k+2] ^ b_row[k+2]));
            acc += bit_popcnt<T>((T)~(a_row[k+3] ^ b_row[k+3]));
        }
    }

    for (; k < Kp; ++k) {
        acc += bit_popcnt<T>((T)~(a_row[k] ^ b_row[k]));
    }
    C[m * N + n] = 2 * (int32_t)acc - K_eff;
}

}}}  // namespace cbrute::cuda::kernels
