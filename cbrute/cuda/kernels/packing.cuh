// CUDA bit-packing / unpacking kernels.
//
// Layout matches the CPU side exactly:
//   * Bits LSB-first within each byte.
//   * pw = 8/32/64 controls only the leading-dim alignment (row_bytes is a
//     multiple of pw/8). Within the used portion of a row, the byte stream is
//     identical regardless of pw.
//
// Alignment safety:
//   * For pw=32 / pw=64 the packed-storage dtype is int32 / int64 (4-byte and
//     8-byte aligned base pointer) AND row_bytes is a multiple of 4/8, so any
//     4-byte chunk write within a row is naturally 4-byte aligned.
//   * For pw=8 the storage dtype is uint8 (1-byte aligned) and row_bytes is
//     1 byte per word — neither the base nor any chunk offset is guaranteed
//     to be 4-byte aligned. Earlier versions of this kernel did
//     `*reinterpret_cast<unsigned*>(out_bytes + out_off) = mask`, which
//     triggered cudaErrorMisalignedAddress and corrupted the CUDA context
//     for the rest of the process. The `aligned4` flag (set true by the
//     caller iff 4-byte alignment is provable) selects a safe byte-wise
//     fallback in those cases.
//
// Throughput tricks:
//   * pack: one warp (32 threads) → one 32-bit packed chunk via __ballot_sync.
//     Each thread loads its bool via `__ldg` (read-only cache); for huge inputs
//     a persistent variant batches multiple chunks per warp to amortise grid
//     launch + setup costs.
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

__device__ inline void store_chunk(uint8_t* dst, unsigned mask,
                                   int64_t bytes_left, bool aligned4) {
    // Fast path: 4 bytes available AND caller promises 4-byte alignment.
    if (bytes_left >= 4 && aligned4) {
        *reinterpret_cast<unsigned*>(dst) = mask;
        return;
    }
    const int n = (bytes_left >= 4) ? 4 : (int)bytes_left;
    for (int b8 = 0; b8 < n; ++b8) {
        dst[b8] = (uint8_t)(mask >> (b8 * 8));
    }
}

__device__ inline unsigned load_chunk(const uint8_t* src,
                                      int64_t bytes_left, bool aligned4) {
    if (bytes_left >= 4 && aligned4) {
        return *reinterpret_cast<const unsigned*>(src);
    }
    unsigned mask = 0;
    const int n = (bytes_left >= 4) ? 4 : (int)bytes_left;
    for (int b8 = 0; b8 < n; ++b8) {
        mask |= ((unsigned)src[b8]) << (b8 * 8);
    }
    return mask;
}


__global__ inline void k_pack_bool_warp(const uint8_t* __restrict__ in_bool,
                                        uint8_t* __restrict__ out_bytes,
                                        int64_t ld, int64_t row_bytes,
                                        int aligned4) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;          // exactly one warp / block
    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;

    // __ldg: route load through the read-only cache. pack_bool is a one-shot
    // read (no aliasing risk) so the texture cache amortises L2 traffic
    // across warps that touch nearby bytes.
    const bool b = (bit_in_row < ld) ? (__ldg(in_bool + row * ld + bit_in_row) != 0)
                                     : false;
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, b);

    if (lane == 0) {
        const int64_t out_off    = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        store_chunk(out_bytes + out_off, mask, bytes_left, aligned4 != 0);
    }
}

// Persistent kernel variant: each warp grid-strides over multiple chunks,
// amortising launch overhead for huge batches. The driver picks this when
// total work (n_chunks * batch) is large enough that scheduler overhead would
// otherwise dominate.
__global__ inline void k_pack_bool_persistent(const uint8_t* __restrict__ in_bool,
                                              uint8_t* __restrict__ out_bytes,
                                              int64_t ld, int64_t row_bytes,
                                              int64_t batch,
                                              int chunks_per_row,
                                              int aligned4) {
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
            const int64_t out_off    = row * row_bytes + (int64_t)chunk_idx * 4;
            const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
            store_chunk(out_bytes + out_off, mask, bytes_left, aligned4 != 0);
        }
    }
}

__global__ inline void k_pack_float_warp(const float* __restrict__ in_f,
                                         uint8_t* __restrict__ out_bytes,
                                         int64_t ld, int64_t row_bytes,
                                         int aligned4) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;
    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;

    const bool b = (bit_in_row < ld) ? (__ldg(in_f + row * ld + bit_in_row) > 0.0f)
                                     : false;
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, b);

    if (lane == 0) {
        const int64_t out_off    = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        store_chunk(out_bytes + out_off, mask, bytes_left, aligned4 != 0);
    }
}

//
// unpack: one warp reads one 32-bit chunk, broadcasts to all lanes, each lane
// writes its own bit-expanded element. Grid: (chunks_per_row, batch).
//

__global__ inline void k_unpack_bool_warp(const uint8_t* __restrict__ in_bytes,
                                          uint8_t* __restrict__ out_bool,
                                          int64_t ll, int64_t row_bytes,
                                          int aligned4) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;

    unsigned mask = 0;
    if (lane == 0) {
        const int64_t in_off     = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        mask = load_chunk(in_bytes + in_off, bytes_left, aligned4 != 0);
    }
    mask = __shfl_sync(0xFFFFFFFFu, mask, 0);

    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
    if (bit_in_row < ll) {
        out_bool[row * ll + bit_in_row] = (uint8_t)((mask >> lane) & 1u);
    }
}

__global__ inline void k_unpack_pm1_warp(const uint8_t* __restrict__ in_bytes,
                                         float* __restrict__ out_f,
                                         int64_t ll, int64_t row_bytes,
                                         int aligned4) {
    const int chunk_idx = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int lane      = threadIdx.x;

    unsigned mask = 0;
    if (lane == 0) {
        const int64_t in_off     = row * row_bytes + (int64_t)chunk_idx * 4;
        const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
        mask = load_chunk(in_bytes + in_off, bytes_left, aligned4 != 0);
    }
    mask = __shfl_sync(0xFFFFFFFFu, mask, 0);

    const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
    if (bit_in_row < ll) {
        out_f[row * ll + bit_in_row] = ((mask >> lane) & 1u) ? 1.0f : -1.0f;
    }
}

// Persistent unpack-bool: amortise launch overhead for huge batches.
__global__ inline void k_unpack_bool_persistent(const uint8_t* __restrict__ in_bytes,
                                                uint8_t* __restrict__ out_bool,
                                                int64_t ll, int64_t row_bytes,
                                                int64_t batch,
                                                int chunks_per_row,
                                                int aligned4) {
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
            const int64_t in_off     = row * row_bytes + (int64_t)chunk_idx * 4;
            const int64_t bytes_left = row_bytes - (int64_t)chunk_idx * 4;
            mask = load_chunk(in_bytes + in_off, bytes_left, aligned4 != 0);
        }
        mask = __shfl_sync(0xFFFFFFFFu, mask, 0);
        const int64_t bit_in_row = (int64_t)chunk_idx * 32 + lane;
        if (bit_in_row < ll) {
            out_bool[row * ll + bit_in_row] = (uint8_t)((mask >> lane) & 1u);
        }
    }
}

}}}  // namespace cbrute::cuda::kernels
