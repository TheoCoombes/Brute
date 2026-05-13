// Reductions — per-element hamming + fused bit1_hamming_total.
//
//   hamming_{u8,u32,u64}        : per-element popcount(A ^ B) → int32.
//   bit1_hamming_total_partial  : fused XOR + popcount + reduce → int32 partials
//                                 (host follows up with at::sum(.., kLong)).
//
// The fused total kernel exists because the CompositeExplicitAutograd fallback
// (xor → popcount → sum) materializes a per-element int32 intermediate that's
// 4× the input size — wasteful and a known crash trigger on the legacy MPS
// path. The fused version reads each (a, b) word once, popcounts the XOR in
// register, never writes the intermediate.

#include <metal_stdlib>
using namespace metal;

//  Per-element Hamming distance 
kernel void hamming_u8 (device const uchar* a[[buffer(0)]],
                        device const uchar* b[[buffer(1)]],
                        device int*         c[[buffer(2)]],
                        uint g [[thread_position_in_grid]]) {
    c[g] = (int)popcount((uint)(uchar)(a[g] ^ b[g]));
}
kernel void hamming_u32(device const uint* a[[buffer(0)]],
                        device const uint* b[[buffer(1)]],
                        device int*        c[[buffer(2)]],
                        uint g [[thread_position_in_grid]]) {
    c[g] = (int)popcount(a[g] ^ b[g]);
}
kernel void hamming_u64(device const ulong* a[[buffer(0)]],
                        device const ulong* b[[buffer(1)]],
                        device int*         c[[buffer(2)]],
                        uint g [[thread_position_in_grid]]) {
    c[g] = (int)popcount(a[g] ^ b[g]);
}

//  Fused total Hamming distance — partial reduction 
// Treats both inputs as byte streams of identical length; tail < 8 bytes
// handled host-side via `tail_hamming_u8`.
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
    device atomic_int*  partials [[buffer(2)]],   // partials[0] only
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
