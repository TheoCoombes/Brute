// XNOR-popcount matmul — simdgroup-per-output-cell.
//
// A: (M, Kp) packed   B: (N, Kp) packed   →   C: (M, N) int32
// Output semantic identical to CPU / CUDA:
//   C[m, n] = 2 * popcount_xnor(A[m, :], B[n, :]) − K_eff
//          = 2 * Σ_k popcount(~(A[m,k] ^ B[n,k]))  − (2*Kp*pw − K_logical)
//
// Each simdgroup (32 threads) owns one output cell. Lanes do a grid-stride
// loop over Kp, accumulating popcounts in a per-lane register. `simd_sum`
// reduces across the simdgroup in one instruction. Lane 0 writes the result.
//
// Dispatch (from .mm):
//   threads = (N * 32, M, 1)
//   threadgroup = (32, 1, 1)        ⇒ each TG is exactly one simdgroup
//   threadgroup_position_in_grid = (n, m)
//
// A tiled variant (threadgroup memory caching of A-row / B-row tiles) would
// help when both M and N are large; the simd-per-cell version is bandwidth-
// optimal for small to medium output matrices and is the only kernel needed
// to match CPU/CUDA *correctness*. See the TODO in ops_metal.mm.

#include <metal_stdlib>
using namespace metal;

kernel void xnor_u8(
    device const uchar* A     [[buffer(0)]],
    device const uchar* B     [[buffer(1)]],
    device int*         C     [[buffer(2)]],
    constant int&       N     [[buffer(3)]],
    constant int&       Kp    [[buffer(4)]],
    constant int&       K_eff [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const int n = (int)tgid.x;
    const int m = (int)tgid.y;
    uint acc = 0u;
    for (int k = (int)lane; k < Kp; k += 32) {
        // For u8 we mask down to 8 bits before popcounting — otherwise the
        // implicit promotion to uint would set the high 24 bits via ~, adding
        // 24 spurious matches per word.
        acc += (uint)popcount((uint)(uchar)(~(A[m * Kp + k] ^ B[n * Kp + k])));
    }
    const uint total = simd_sum(acc);
    if (lane == 0) {
        C[m * N + n] = 2 * (int)total - K_eff;
    }
}

kernel void xnor_u32(
    device const uint* A     [[buffer(0)]],
    device const uint* B     [[buffer(1)]],
    device int*        C     [[buffer(2)]],
    constant int&      N     [[buffer(3)]],
    constant int&      Kp    [[buffer(4)]],
    constant int&      K_eff [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const int n = (int)tgid.x;
    const int m = (int)tgid.y;
    uint acc = 0u;
    for (int k = (int)lane; k < Kp; k += 32) {
        acc += (uint)popcount(~(A[m * Kp + k] ^ B[n * Kp + k]));
    }
    const uint total = simd_sum(acc);
    if (lane == 0) {
        C[m * N + n] = 2 * (int)total - K_eff;
    }
}

kernel void xnor_u64(
    device const ulong* A     [[buffer(0)]],
    device const ulong* B     [[buffer(1)]],
    device int*         C     [[buffer(2)]],
    constant int&       N     [[buffer(3)]],
    constant int&       Kp    [[buffer(4)]],
    constant int&       K_eff [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const int n = (int)tgid.x;
    const int m = (int)tgid.y;
    uint acc = 0u;
    for (int k = (int)lane; k < Kp; k += 32) {
        acc += (uint)popcount(~(A[m * Kp + k] ^ B[n * Kp + k]));
    }
    const uint total = simd_sum(acc);
    if (lane == 0) {
        C[m * N + n] = 2 * (int)total - K_eff;
    }
}
