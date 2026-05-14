// CUDA popcount kernels.
//
//   k_popcnt_*        : per-element popcount, output int32.
//   k_packed_popcount : total popcount over a contiguous byte stream
//                       (block-level reduce → atomic-add into a single int64).
//   k_hamming_total   : fused XOR + popcount of two equal-shape byte streams,
//                       total bit count. No XOR temporary.
//
// All total-popcount kernels reinterpret the byte stream as uint64 + vectorized
// uint4 loads (16 bytes per load) for max effective bandwidth. The tail
// (<16 bytes) is handled by a single thread without branching in the hot loop.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda { namespace kernels {

//  Per-element popcount

__global__ inline void k_popcnt_u8 (const uint8_t* __restrict__ in,
                                    int32_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popc((unsigned)in[i]);
}
__global__ inline void k_popcnt_u16(const uint16_t* __restrict__ in,
                                    int32_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popc((unsigned)in[i]);
}
__global__ inline void k_popcnt_u32(const uint32_t* __restrict__ in,
                                    int32_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popc(in[i]);
}
__global__ inline void k_popcnt_u64(const uint64_t* __restrict__ in,
                                    int32_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popcll(in[i]);
}

//  Block-level reduction helpers (warp shuffle ladder) 

__device__ inline uint64_t warp_reduce_sum_u64(uint64_t v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ inline uint64_t block_reduce_sum_u64(uint64_t v) {
    __shared__ uint64_t s_warp[32];                   // up to 32 warps / block
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warp_reduce_sum_u64(v);
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();

    const int n_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < n_warps) ? s_warp[threadIdx.x] : 0ull;
    if (warp == 0) v = warp_reduce_sum_u64(v);
    return v;     // valid only on thread 0
}

//  Total popcount over a flat byte stream
// Each thread issues 128-bit (ulonglong2) vectorized loads — two uint64 words
// per memory transaction — and falls back to scalar loads on the odd-word
// residual. Blocks reduce locally; one atomicAdd per block.
// Output is a single uint64 (cast to int64 by caller).
//

__global__ inline void k_packed_popcount_total(const uint64_t* __restrict__ in,
                                               int64_t n_words,
                                               unsigned long long* __restrict__ out) {
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    uint64_t local = 0;

    // PyTorch tensor data is 64-byte aligned, so the base pointer always
    // satisfies the 16-byte alignment required for ulonglong2 loads.
    const int64_t n_pairs = n_words >> 1;
    const ulonglong2* in2 = reinterpret_cast<const ulonglong2*>(in);
    for (int64_t i = tid; i < n_pairs; i += stride) {
        ulonglong2 v = in2[i];
        local += (uint64_t)__popcll(v.x) + (uint64_t)__popcll(v.y);
    }
    // Trailing odd word (n_words is odd).
    if ((n_words & 1) && tid == 0) {
        local += (uint64_t)__popcll(in[n_words - 1]);
    }
    uint64_t block_sum = block_reduce_sum_u64(local);
    if (threadIdx.x == 0) {
        atomicAdd(out, (unsigned long long)block_sum);
    }
}

//  Total Hamming distance: fused popc(A XOR B) over the buffer
__global__ inline void k_hamming_total(const uint64_t* __restrict__ a,
                                       const uint64_t* __restrict__ b,
                                       int64_t n_words,
                                       unsigned long long* __restrict__ out) {
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    uint64_t local = 0;

    const int64_t n_pairs = n_words >> 1;
    const ulonglong2* a2 = reinterpret_cast<const ulonglong2*>(a);
    const ulonglong2* b2 = reinterpret_cast<const ulonglong2*>(b);
    for (int64_t i = tid; i < n_pairs; i += stride) {
        ulonglong2 va = a2[i], vb = b2[i];
        local += (uint64_t)__popcll(va.x ^ vb.x)
               + (uint64_t)__popcll(va.y ^ vb.y);
    }
    if ((n_words & 1) && tid == 0) {
        local += (uint64_t)__popcll(a[n_words - 1] ^ b[n_words - 1]);
    }
    uint64_t block_sum = block_reduce_sum_u64(local);
    if (threadIdx.x == 0) {
        atomicAdd(out, (unsigned long long)block_sum);
    }
}

// Tail (< sizeof(uint64) trailing bytes). Single thread, called from host code
// only when `n_bytes % 8 != 0`. Always reads from a small staging buffer set
// up by the caller; written here so all popcount paths share semantics.
__global__ inline void k_popcount_tail_byte(const uint8_t* __restrict__ in,
                                            int n_tail,
                                            unsigned long long* __restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t w = 0;
    for (int b = 0; b < n_tail; ++b) w |= ((uint64_t)in[b]) << (b * 8);
    atomicAdd(out, (unsigned long long)__popcll(w));
}

__global__ inline void k_hamming_tail_byte(const uint8_t* __restrict__ a,
                                           const uint8_t* __restrict__ b,
                                           int n_tail,
                                           unsigned long long* __restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t wa = 0, wb = 0;
    for (int b8 = 0; b8 < n_tail; ++b8) {
        wa |= ((uint64_t)a[b8]) << (b8 * 8);
        wb |= ((uint64_t)b[b8]) << (b8 * 8);
    }
    atomicAdd(out, (unsigned long long)__popcll(wa ^ wb));
}

}}}  // namespace cbrute::cuda::kernels
