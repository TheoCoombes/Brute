// CUDA bit-packing / unpacking kernels for bit1 (uint64-packed) tensors.
//
// Layout: bits LSB-first within each byte; a row of `ld` logical bits consumes
// `pd_words` int64 words (= ceil(ld/64) * 8 bytes). row_bytes is therefore
// always a multiple of 8, so every 4-byte chunk write is naturally aligned —
// no byte-wise fallback needed.
//
// Throughput tricks:
//   * pack: one warp (32 threads) → one 32-bit packed chunk via __ballot_sync.
//     Each thread loads its bool via __ldg (read-only cache); for huge inputs
//     a persistent variant batches multiple chunks per warp to amortise grid
//     launch + setup costs.
//   * unpack: lane 0 reads the chunk, __shfl_sync broadcasts, each lane writes
//     its own bit. No global-memory bank conflicts.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda { namespace kernels {

//  pack: one warp per 32-bit chunk per row. Grid: (chunks_per_row, batch).
//  out_bytes is zero-initialised by the caller, so trailing pad bytes/bits
//  remain zero.
__global__ inline void k_pack_bool_warp(const uint8_t* __restrict__ in_bool,
                                        uint8_t* __restrict__ out_bytes,
                                        int64_t ld, int64_t row_bytes) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;                  // exactly one warp / block
    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;

    // __ldg routes the load through the read-only cache.
    const bool b = (bit_in_row < ld) ? (__ldg(in_bool + row * ld + bit_in_row) != 0)
                                     : false;
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, b);

    if (lane == 0) {
        const int64_t out_off = row * row_bytes + (int64_t)chunk_idx * 4;
        *reinterpret_cast<unsigned*>(out_bytes + out_off) = mask;
    }
}

// Persistent variant: each warp grid-strides over multiple chunks, amortising
// launch overhead for huge batches.
__global__ inline void k_pack_bool_persistent(const uint8_t* __restrict__ in_bool,
                                              uint8_t* __restrict__ out_bytes,
                                              int64_t ld, int64_t row_bytes,
                                              int64_t batch,
                                              int chunks_per_row) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t global_warp_id =
        (int64_t)blockIdx.x * warps_per_block + warp_in_block;
    const int64_t total_warps = (int64_t)gridDim.x * warps_per_block;
    const int64_t total_work  = batch * (int64_t)chunks_per_row;

    for (int64_t w = global_warp_id; w < total_work; w += total_warps) {
        const int64_t row       = w / chunks_per_row;
        const int     chunk_idx = (int)(w % chunks_per_row);
        const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
        const bool b = (bit_in_row < ld) ? (__ldg(in_bool + row * ld + bit_in_row) != 0)
                                         : false;
        const unsigned mask = __ballot_sync(0xFFFFFFFFu, b);
        if (lane == 0) {
            const int64_t out_off = row * row_bytes + (int64_t)chunk_idx * 4;
            *reinterpret_cast<unsigned*>(out_bytes + out_off) = mask;
        }
    }
}

//  unpack: one warp reads one 32-bit chunk, broadcasts to all lanes, each
//  lane writes its own bit-expanded element. Grid: (chunks_per_row, batch).
__global__ inline void k_unpack_bool_warp(const uint8_t* __restrict__ in_bytes,
                                          uint8_t* __restrict__ out_bool,
                                          int64_t ll, int64_t row_bytes) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;

    unsigned mask = 0;
    if (lane == 0) {
        const int64_t in_off = row * row_bytes + (int64_t)chunk_idx * 4;
        mask = *reinterpret_cast<const unsigned*>(in_bytes + in_off);
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
        const int64_t in_off = row * row_bytes + (int64_t)chunk_idx * 4;
        mask = *reinterpret_cast<const unsigned*>(in_bytes + in_off);
    }
    mask = __shfl_sync(0xFFFFFFFFu, mask, 0);

    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
    if (bit_in_row < ll) {
        out_f[row * ll + bit_in_row] = ((mask >> lane) & 1u) ? 1.0f : -1.0f;
    }
}

__global__ inline void k_unpack_bool_persistent(const uint8_t* __restrict__ in_bytes,
                                                uint8_t* __restrict__ out_bool,
                                                int64_t ll, int64_t row_bytes,
                                                int64_t batch,
                                                int chunks_per_row) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t global_warp_id =
        (int64_t)blockIdx.x * warps_per_block + warp_in_block;
    const int64_t total_warps = (int64_t)gridDim.x * warps_per_block;
    const int64_t total_work  = batch * (int64_t)chunks_per_row;

    for (int64_t w = global_warp_id; w < total_work; w += total_warps) {
        const int64_t row       = w / chunks_per_row;
        const int     chunk_idx = (int)(w % chunks_per_row);
        unsigned mask = 0;
        if (lane == 0) {
            const int64_t in_off = row * row_bytes + (int64_t)chunk_idx * 4;
            mask = *reinterpret_cast<const unsigned*>(in_bytes + in_off);
        }
        mask = __shfl_sync(0xFFFFFFFFu, mask, 0);
        const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
        if (bit_in_row < ll) {
            out_bool[row * ll + bit_in_row] = (uint8_t)((mask >> lane) & 1u);
        }
    }
}

}}}  // namespace cbrute::cuda::kernels
