// Pack / unpack kernels — simdgroup ballot/shuffle.
//
// Layout: bits LSB-first within each byte. Row stride = pd_words * 8 bytes
// (pack_width is fixed at 64). chunk*4 byte offsets are therefore always
// 4-byte aligned and always within row_bytes for every dispatched chunk.
//
// Dispatch convention (set by the .mm driver):
//   threads per threadgroup = (256, 1, 1)              // 8 simdgroups
//   grid                    = (n_chunks * 256, rows)   // n_chunks = ceil(ld/32)
//
// Within a threadgroup:
//   sg_id  ∈ [0, 8)   which 32-bit chunk this simdgroup handles
//   lane   ∈ [0, 32)  one input bit per lane
//
// Each simdgroup packs exactly 32 input bits → writes 4 output bytes.

#include <metal_stdlib>
using namespace metal;

// simd_ballot's vote → 32-bit mask. Apple GPU simdgroup width is 32, so only
// the low 32 bits are meaningful.
static inline uint ballot_to_u32(simd_vote v) { return (uint)((ulong)v); }

//  pack_bool: bool bytes → packed bits.
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
        const int out_off = row * row_bytes + chunk * 4;
        if (chunk * 4 < row_bytes) {
            *((device uint*)(output + out_off)) = mask;
        }
    }
}

kernel void pack_sign_i32_chunked(
    device const int*   input     [[buffer(0)]],
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
    const bool v    = (bit < ld) ? (input[row * ld + bit] >= 0) : false;
    const uint mask = ballot_to_u32(simd_ballot(v));
    if (sg_lane == 0) {
        const int out_off = row * row_bytes + chunk * 4;
        if (chunk * 4 < row_bytes) {
            *((device uint*)(output + out_off)) = mask;
        }
    }
}

kernel void pack_sign_f32_chunked(
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
    const bool v    = (bit < ld) ? (input[row * ld + bit] >= 0.0f) : false;
    const uint mask = ballot_to_u32(simd_ballot(v));
    if (sg_lane == 0) {
        const int out_off = row * row_bytes + chunk * 4;
        if (chunk * 4 < row_bytes) {
            *((device uint*)(output + out_off)) = mask;
        }
    }
}

kernel void bsr_scan_chunked(
    device const ulong* q         [[buffer(0)]],
    device const ulong* assoc     [[buffer(1)]],
    device const int*   shifts    [[buffer(2)]],
    device uchar*       read      [[buffer(3)]],
    device uchar*       state     [[buffer(4)]],
    device uchar*       gate      [[buffer(5)]],
    constant int&       n         [[buffer(6)]],
    constant int&       Kp        [[buffer(7)]],
    constant int&       D         [[buffer(8)]],
    constant int&       groups    [[buffer(9)]],
    uint  sg_lane [[thread_index_in_simdgroup]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]])
{
    const int chunk = (int)(tgid.x * 8 + sg_id);
    const int row   = (int)tgid.y;
    const int d     = chunk * 32 + (int)sg_lane;
    const bool active = d < D;
    const int word_idx = d >> 6;
    const int bit_idx = d & 63;
    const ulong bit_mask = 1ul << bit_idx;
    const int shift = active ? shifts[(d * groups) / D] : 0;
    const int row_bytes = Kp * 8;
    int A = 0;

    for (int t = 0; t < n; ++t) {
        bool q_bit = false;
        bool assoc_bit = false;
        bool state_bit = false;
        bool gate_bit = false;
        if (active) {
            const int off = (row * n + t) * Kp + word_idx;
            q_bit = (q[off] & bit_mask) != 0ul;
            assoc_bit = (assoc[off] & bit_mask) != 0ul;
            state_bit = A >= 0;
            gate_bit = state_bit != assoc_bit;
        }

        const uint read_mask = ballot_to_u32(simd_ballot(active && (q_bit == state_bit)));
        const uint state_mask = ballot_to_u32(simd_ballot(active && state_bit));
        const uint gate_mask = ballot_to_u32(simd_ballot(active && gate_bit));
        if (sg_lane == 0) {
            const int out_off = (row * n + t) * row_bytes + chunk * 4;
            if (chunk * 4 < row_bytes) {
                *((device uint*)(read + out_off)) = read_mask;
                *((device uint*)(state + out_off)) = state_mask;
                *((device uint*)(gate + out_off)) = gate_mask;
            }
        }

        if (active) {
            int decayed = A;
            if (shift > 0) {
                decayed = A - (A >> shift);
            }
            const int update = gate_bit ? (assoc_bit ? 1 : -1) : 0;
            A = decayed + update;
        }
    }
}

//  unpack to ±1.0f. Lane 0 reads the chunk; shuffle-broadcast to all lanes;
//  each lane writes its own bit.
kernel void unpack_pm1_chunked(
    device const uchar* input     [[buffer(0)]],
    device float*       output    [[buffer(1)]],
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
        const int in_off = row * row_bytes + chunk * 4;
        if (chunk * 4 < row_bytes) {
            mask = *((device const uint*)(input + in_off));
        }
    }
    mask = simd_broadcast_first(mask);

    const int bit = chunk * 32 + (int)sg_lane;
    if (bit < ll) {
        output[row * ll + bit] = ((mask >> sg_lane) & 1u) ? 1.0f : -1.0f;
    }
}

//  unpack to bool (1 byte per logical bit).
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
        const int in_off = row * row_bytes + chunk * 4;
        if (chunk * 4 < row_bytes) {
            mask = *((device const uint*)(input + in_off));
        }
    }
    mask = simd_broadcast_first(mask);

    const int bit = chunk * 32 + (int)sg_lane;
    if (bit < ll) {
        output[row * ll + bit] = (uchar)((mask >> sg_lane) & 1u);
    }
}
