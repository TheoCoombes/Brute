// Pack / unpack kernels — simdgroup-cooperative ballot/shuffle.
//
// Layout (matches CPU + CUDA):
//   bits LSB-first within each byte; row stride = pd_words * (pw/8) bytes.
//   Within the used portion of a row the byte stream is identical regardless
//   of pack-width — pw only controls the trailing zero-pad alignment.
//
// Dispatch convention (set in the .mm driver):
//   threads per threadgroup = (256, 1, 1)            // 8 simdgroups
//   grid                    = (n_chunks * 256, rows) // where n_chunks =
//                                                       ceil(ld / 32)
//   thread_index_in_simdgroup ∈ [0,32) is the lane (= one input bit)
//   simdgroup_index_in_threadgroup ∈ [0,8) is which 32-bit chunk this SG
//                                          handles within the threadgroup
//   threadgroup_position_in_grid.x ∈ [0, n_chunks/8) is the chunk batch
//
// Each simdgroup packs exactly 32 input bits ⇒ writes 4 output bytes. Final
// bytes beyond ceil(ld/8) inside a row stay zero (the buffer is zero-init
// by the host).

#include <metal_stdlib>
using namespace metal;

// Extract the active-thread bitmask from a simd_ballot result. Apple GPU
// simdgroup width is 32, so only the low 32 bits are meaningful.
static inline uint ballot_to_u32(simd_vote v) {
    return (uint)((ulong)v);
}

//  Common chunk-write helper: write 4 bytes (or fewer at row tail) 
static inline void write_chunk_bytes(device uchar* out,
                                     int out_off,
                                     int bytes_left_in_row,
                                     uint mask) {
    if (bytes_left_in_row >= 4) {
        *((device uint*)(out + out_off)) = mask;
    } else {
        for (int b = 0; b < bytes_left_in_row; ++b) {
            out[out_off + b] = (uchar)(mask >> (b * 8));
        }
    }
}

// 
// pack_bits: float → packed bits. Bit = 1 iff input > 0.f.
// Kernel name kept identical to the legacy version (driver pw_suffix dispatch).
// pw_suffix only affects the driver's output-tensor dtype + row_bytes math;
// the kernel itself is dtype-agnostic — it writes bytes.
// 
kernel void pack_bits_chunked(
    device const float* input     [[buffer(0)]],
    device uchar*       output    [[buffer(1)]],
    constant int&       ld        [[buffer(2)]],
    constant int&       row_bytes [[buffer(3)]],
    uint  sg_lane [[thread_index_in_simdgroup]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]])
{
    const int chunk = (int)(tgid.x * 8 + sg_id);
    const int row   = (int)tgid.y;
    const int bit   = chunk * 32 + (int)sg_lane;
    const bool v    = (bit < ld) ? (input[row * ld + bit] > 0.0f) : false;
    const uint mask = ballot_to_u32(simd_ballot(v));
    if (sg_lane == 0) {
        const int out_off    = row * row_bytes + chunk * 4;
        const int bytes_left = row_bytes - chunk * 4;
        if (bytes_left > 0) write_chunk_bytes(output, out_off, bytes_left, mask);
    }
}

// 
// pack_bool: bool bytes → packed bits. Skips the float intermediate that the
// Python composite fallback used to materialize.
// 
kernel void pack_bool_chunked(
    device const uchar* input     [[buffer(0)]],
    device uchar*       output    [[buffer(1)]],
    constant int&       ld        [[buffer(2)]],
    constant int&       row_bytes [[buffer(3)]],
    uint  sg_lane [[thread_index_in_simdgroup]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]])
{
    const int chunk = (int)(tgid.x * 8 + sg_id);
    const int row   = (int)tgid.y;
    const int bit   = chunk * 32 + (int)sg_lane;
    const bool v    = (bit < ld) ? (input[row * ld + bit] != 0) : false;
    const uint mask = ballot_to_u32(simd_ballot(v));
    if (sg_lane == 0) {
        const int out_off    = row * row_bytes + chunk * 4;
        const int bytes_left = row_bytes - chunk * 4;
        if (bytes_left > 0) write_chunk_bytes(output, out_off, bytes_left, mask);
    }
}

// 
// unpack to ±1.0f. Lane 0 reads one 32-bit chunk; shuffle-broadcast to all
// lanes; each lane writes its own bit.
// 
kernel void unpack_pm1_chunked(
    device const uchar* input     [[buffer(0)]],
    device float*       output    [[buffer(1)]],
    constant int&       ll        [[buffer(2)]],   // logical row length in bits
    constant int&       row_bytes [[buffer(3)]],
    uint  sg_lane [[thread_index_in_simdgroup]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]])
{
    const int chunk = (int)(tgid.x * 8 + sg_id);
    const int row   = (int)tgid.y;

    uint mask = 0u;
    if (sg_lane == 0) {
        const int in_off     = row * row_bytes + chunk * 4;
        const int bytes_left = row_bytes - chunk * 4;
        if (bytes_left >= 4) {
            mask = *((device const uint*)(input + in_off));
        } else if (bytes_left > 0) {
            for (int b = 0; b < bytes_left; ++b)
                mask |= ((uint)input[in_off + b]) << (b * 8);
        }
    }
    mask = simd_broadcast_first(mask);

    const int bit = chunk * 32 + (int)sg_lane;
    if (bit < ll) {
        output[row * ll + bit] = ((mask >> sg_lane) & 1u) ? 1.0f : -1.0f;
    }
}

// 
// unpack to bool (1 byte per logical bit). Same shape as unpack_pm1.
// 
kernel void unpack_bool_chunked(
    device const uchar* input     [[buffer(0)]],
    device uchar*       output    [[buffer(1)]],
    constant int&       ll        [[buffer(2)]],
    constant int&       row_bytes [[buffer(3)]],
    uint  sg_lane [[thread_index_in_simdgroup]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]])
{
    const int chunk = (int)(tgid.x * 8 + sg_id);
    const int row   = (int)tgid.y;

    uint mask = 0u;
    if (sg_lane == 0) {
        const int in_off     = row * row_bytes + chunk * 4;
        const int bytes_left = row_bytes - chunk * 4;
        if (bytes_left >= 4) {
            mask = *((device const uint*)(input + in_off));
        } else if (bytes_left > 0) {
            for (int b = 0; b < bytes_left; ++b)
                mask |= ((uint)input[in_off + b]) << (b * 8);
        }
    }
    mask = simd_broadcast_first(mask);

    const int bit = chunk * 32 + (int)sg_lane;
    if (bit < ll) {
        output[row * ll + bit] = (uchar)((mask >> sg_lane) & 1u);
    }
}
