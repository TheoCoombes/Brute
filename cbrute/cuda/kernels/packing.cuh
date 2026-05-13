// CUDA bit-packing / unpacking kernels.
//
// Layout matches the CPU side exactly:
//   * Bits LSB-first within each byte.
//   * pw = 8/32/64 controls only the leading-dim alignment (row_bytes is a
//     multiple of pw/8). Within the used portion of a row, the byte stream is
//     identical regardless of pw.
//
// Throughput tricks:
//   * pack: one warp (32 threads) → one 32-bit packed chunk via __ballot_sync.
//   * unpack: lane 0 reads the chunk, __shfl_sync broadcasts, each lane extracts
//             its own bit. No global-memory bank conflicts.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda { namespace kernels {

// 
// pack: one warp per 32-bit chunk per row. Grid: (chunks_per_row, batch).
// out_bytes is zero-initialized by the caller, so trailing pad bytes
// (chunk_idx*4 + 4 > row_bytes) and trailing pad bits within the last chunk
// stay zero.
// 

__global__ inline void k_pack_bool_warp(const uint8_t* __restrict__ in_bool,
                                        uint8_t* __restrict__ out_bytes,
                                        int64_t ld, int64_t row_bytes) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;          // exactly one warp / block
    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;

    const bool b = (bit_in_row < ld) ? (in_bool[row * ld + bit_in_row] != 0)
                                     : false;
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, b);

    if (lane == 0) {
        const int64_t out_off    = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        if (bytes_left >= 4) {
            *reinterpret_cast<unsigned*>(out_bytes + out_off) = mask;
        } else {
            for (int b8 = 0; b8 < bytes_left; ++b8) {
                out_bytes[out_off + b8] = (uint8_t)(mask >> (b8 * 8));
            }
        }
    }
}

__global__ inline void k_pack_float_warp(const float* __restrict__ in_f,
                                         uint8_t* __restrict__ out_bytes,
                                         int64_t ld, int64_t row_bytes) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;
    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;

    const bool b = (bit_in_row < ld) ? (in_f[row * ld + bit_in_row] > 0.0f)
                                     : false;
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, b);

    if (lane == 0) {
        const int64_t out_off    = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        if (bytes_left >= 4) {
            *reinterpret_cast<unsigned*>(out_bytes + out_off) = mask;
        } else {
            for (int b8 = 0; b8 < bytes_left; ++b8) {
                out_bytes[out_off + b8] = (uint8_t)(mask >> (b8 * 8));
            }
        }
    }
}

// 
// unpack: one warp reads one 32-bit chunk, broadcasts to all lanes, each lane
// writes its own bit-expanded element. Grid: (chunks_per_row, batch).
// 

__global__ inline void k_unpack_bool_warp(const uint8_t* __restrict__ in_bytes,
                                          uint8_t* __restrict__ out_bool,
                                          int64_t ll, int64_t row_bytes) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;

    unsigned mask = 0;
    if (lane == 0) {
        const int64_t in_off     = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        if (bytes_left >= 4) {
            mask = *reinterpret_cast<const unsigned*>(in_bytes + in_off);
        } else {
            for (int b8 = 0; b8 < bytes_left; ++b8) {
                mask |= ((unsigned)in_bytes[in_off + b8]) << (b8 * 8);
            }
        }
    }
    mask = __shfl_sync(0xFFFFFFFFu, mask, 0);

    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
    if (bit_in_row < ll) {
        out_bool[row * ll + bit_in_row] = (uint8_t)((mask >> lane) & 1u);
    }
}

__global__ inline void k_unpack_pm1_warp(const uint8_t* __restrict__ in_bytes,
                                         float* __restrict__ out_f,
                                         int64_t ll, int64_t row_bytes) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;

    unsigned mask = 0;
    if (lane == 0) {
        const int64_t in_off     = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        if (bytes_left >= 4) {
            mask = *reinterpret_cast<const unsigned*>(in_bytes + in_off);
        } else {
            for (int b8 = 0; b8 < bytes_left; ++b8) {
                mask |= ((unsigned)in_bytes[in_off + b8]) << (b8 * 8);
            }
        }
    }
    mask = __shfl_sync(0xFFFFFFFFu, mask, 0);

    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
    if (bit_in_row < ll) {
        out_f[row * ll + bit_in_row] = ((mask >> lane) & 1u) ? 1.0f : -1.0f;
    }
}

}}}  // namespace cbrute::cuda::kernels
