// Fused bit1_hamming_total: total popcount(A ^ B) over equal-shape byte
// streams in a single pass — no XOR intermediate.
//
// Per-element `hamming_distance(A, B)` is composed in the driver as
// `popcount(at::bitwise_xor(A, B))`, which is fine on MPS because both ops
// have native MPSGraph impls. We only keep the fused total kernel because
// the (xor → popcount → sum) composite materialises a 4× intermediate.
//
// Tail (< 8 bytes) is handled by `tail_hamming_u8`: a single threadgroup
// reads up to 7 bytes, popcounts the partial XOR, atomically accumulates.

#include <metal_stdlib>
using namespace metal;

kernel void bit1_hamming_total_partial(
    device const ulong* a        [[buffer(0)]],
    device const ulong* b        [[buffer(1)]],
    device int*         partials [[buffer(2)]],
    constant int&       n_words  [[buffer(3)]],
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
        local += (uint)popcount(a[i] ^ b[i]);
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

kernel void tail_hamming_u8(
    device const uchar* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device atomic_int*  partials   [[buffer(2)]],
    constant int&       tail_bytes [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tid != 0) return;
    ulong wa = 0, wb = 0;
    for (int k = 0; k < tail_bytes; ++k) {
        wa |= ((ulong)a[k]) << (k * 8);
        wb |= ((ulong)b[k]) << (k * 8);
    }
    atomic_fetch_add_explicit(partials, (int)popcount(wa ^ wb), memory_order_relaxed);
}
