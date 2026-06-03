// XNOR-popcount matmul — simdgroup-per-output-cell.
//
// A: (M, Kp) ulong packed   B: (N, Kp) ulong packed   →   C: (M, N) int32
//   C[m, n] = K_logical − 2 · popc_xor(A[m,:], B[n,:])
//
// Each simdgroup (32 threads) owns one output cell. Lanes do a grid-stride
// loop over Kp, accumulating popcounts in a per-lane register. `simd_sum`
// reduces across the simdgroup in one instruction. Lane 0 writes the result.
//
// Two kernels:
//   * xnor_u64       — baseline, one ulong per lane per K-iter. Used when
//                      Kp is odd (vectorised load wouldn't be aligned).
//   * xnor_u64_wide  — primary fast path: each lane loads ulong2 per K-iter,
//                      doubling per-lane arithmetic per memory op. Used when
//                      Kp % 2 == 0 (almost always).
//
// Dispatch (set in .mm driver):
//   grid        = (N * 32, M, 1)
//   threadgroup = (32, 1, 1)            // each TG = one simdgroup
//   threadgroup_position_in_grid = (n, m)

#include <metal_stdlib>
using namespace metal;

kernel void xnor_u64(
    device const ulong* A         [[buffer(0)]],
    device const ulong* B         [[buffer(1)]],
    device int*         C         [[buffer(2)]],
    constant int&       N         [[buffer(3)]],
    constant int&       Kp        [[buffer(4)]],
    constant int&       K_logical [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const int n = (int)tgid.x;
    const int m = (int)tgid.y;
    uint acc = 0u;
    for (int k = (int)lane; k < Kp; k += 32) {
        acc += (uint)popcount(A[m * Kp + k] ^ B[n * Kp + k]);
    }
    const uint total = simd_sum(acc);
    if (lane == 0) {
        C[m * N + n] = K_logical - 2 * (int)total;
    }
}

// Each simdgroup lane processes 2 ulong words per K-iter via ulong2 loads.
// Caller passes Kp2 = Kp/2; requires Kp % 2 == 0.
kernel void xnor_u64_wide(
    device const ulong2* A         [[buffer(0)]],
    device const ulong2* B         [[buffer(1)]],
    device int*          C         [[buffer(2)]],
    constant int&        N         [[buffer(3)]],
    constant int&        Kp2       [[buffer(4)]],
    constant int&        K_logical [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const int n = (int)tgid.x;
    const int m = (int)tgid.y;
    uint acc = 0u;
    for (int k = (int)lane; k < Kp2; k += 32) {
        const ulong2 va = A[m * Kp2 + k];
        const ulong2 vb = B[n * Kp2 + k];
        const ulong2 xr = va ^ vb;
        acc += (uint)popcount(xr.x) + (uint)popcount(xr.y);
    }
    const uint total = simd_sum(acc);
    if (lane == 0) {
        C[m * N + n] = K_logical - 2 * (int)total;
    }
}
