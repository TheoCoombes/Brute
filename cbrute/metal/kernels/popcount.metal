// Popcount kernels — per-element + total-over-buffer.
//
//   popcnt_{u8,u32,u64}    : per-element, one thread per input word.
//   packed_popcount_partial: total popcount over a flat byte stream. Each
//       threadgroup computes its slice's total and writes ONE int32 to a
//       partials buffer; the host follows up with at::sum(partials,
//       dtype=kLong). This avoids 64-bit atomics, which aren't universally
//       available, and is bandwidth-bound either way.
//
// The byte-stream interpretation means we don't care about the source dtype
// (uint8/int32/int64); we treat the contiguous buffer as a stream of ulong
// words and popcount each. Tail bytes (< 8) are handled host-side via the
// `tail_popcount_u8` kernel: a single threadgroup reads up to 7 bytes,
// popcounts the resulting partial ulong, and atomically adds to the same
// partials buffer's slot 0.

#include <metal_stdlib>
using namespace metal;

//  per-element popcount 
kernel void popcnt_u8(device const uchar* in [[buffer(0)]],
                      device int*         out[[buffer(1)]],
                      uint g [[thread_position_in_grid]]) {
    out[g] = (int)popcount((uint)in[g]);
}
kernel void popcnt_u32(device const uint* in [[buffer(0)]],
                       device int*        out[[buffer(1)]],
                       uint g [[thread_position_in_grid]]) {
    out[g] = (int)popcount(in[g]);
}
kernel void popcnt_u64(device const ulong* in [[buffer(0)]],
                       device int*         out[[buffer(1)]],
                       uint g [[thread_position_in_grid]]) {
    out[g] = (int)popcount(in[g]);
}

//  Total popcount of a byte stream — partial reduction 
//
// Grid-stride loop over the input as ulong words; simdgroup-reduce; one
// uint per threadgroup written to `partials`. Host runs at::sum(partials,
// dtype=kLong) to finish.
//
// Constants:
//   n_words   = floor(n_bytes / 8)
//   (tail bytes < 8 are handled by `tail_popcount_u8` in a separate dispatch)
kernel void packed_popcount_partial(
    device const ulong* input    [[buffer(0)]],
    device int*         partials [[buffer(1)]],
    constant int&       n_words  [[buffer(2)]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  lane    [[thread_index_in_simdgroup]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint3 tg_grid [[threadgroups_per_grid]])
{
    constexpr int THREADS_PER_TG = 256;
    constexpr int SIMDS_PER_TG   = 8;
    threadgroup uint sg_totals[SIMDS_PER_TG];

    const int gid    = (int)(tgid.x * THREADS_PER_TG + tid);
    const int stride = (int)(tg_grid.x * THREADS_PER_TG);
    uint local = 0u;
    for (int i = gid; i < n_words; i += stride) {
        local += (uint)popcount(input[i]);
    }
    const uint simd_total = simd_sum(local);
    if (lane == 0) sg_totals[sg_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0) {
        const uint v = (lane < (uint)SIMDS_PER_TG) ? sg_totals[lane] : 0u;
        const uint tg_total = simd_sum(v);
        if (lane == 0) partials[tgid.x] = (int)tg_total;
    }
}

//  Tail (< 8 bytes) — single-threadgroup, written to partials[0] 
//
// `tail_bytes` is the byte count to read (must be < 8). The host writes the
// tail bytes into a small staging buffer for us; this kernel popcounts them
// and atomically adds the result to `partials[0]`.
kernel void tail_popcount_u8(
    device const uchar* input    [[buffer(0)]],
    device atomic_int*  partials [[buffer(1)]],   // partials[0] only
    constant int&       tail_bytes [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tid != 0) return;
    ulong w = 0;
    for (int b = 0; b < tail_bytes; ++b) {
        w |= ((ulong)input[b]) << (b * 8);
    }
    atomic_fetch_add_explicit(partials, (int)popcount(w), memory_order_relaxed);
}
