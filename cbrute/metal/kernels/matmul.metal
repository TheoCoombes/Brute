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

// ── Wide-vector variants ──────────────────────────────────────────────────────
// Each simdgroup lane processes 4 (uint32) or 2 (ulong) words per K-iteration,
// reducing loop-overhead and maximising arithmetic per load instruction.
// Caller passes Kp4 = Kp/4 (u32) or Kp2 = Kp/2 (u64); Kp must be divisible.

kernel void xnor_u32_wide(
    device const uint4* A     [[buffer(0)]],
    device const uint4* B     [[buffer(1)]],
    device int*         C     [[buffer(2)]],
    constant int&       N     [[buffer(3)]],
    constant int&       Kp4   [[buffer(4)]],   // Kp / 4
    constant int&       K_eff [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const int n = (int)tgid.x;
    const int m = (int)tgid.y;
    uint acc = 0u;
    for (int k = (int)lane; k < Kp4; k += 32) {
        uint4 va = A[m * Kp4 + k];
        uint4 vb = B[n * Kp4 + k];
        uint4 xr = ~(va ^ vb);
        acc += popcount(xr.x) + popcount(xr.y) + popcount(xr.z) + popcount(xr.w);
    }
    const uint total = simd_sum(acc);
    if (lane == 0) {
        C[m * N + n] = 2 * (int)total - K_eff;
    }
}

kernel void xnor_u64_wide(
    device const ulong2* A    [[buffer(0)]],
    device const ulong2* B    [[buffer(1)]],
    device int*          C    [[buffer(2)]],
    constant int&        N    [[buffer(3)]],
    constant int&        Kp2  [[buffer(4)]],   // Kp / 2
    constant int&        K_eff [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    const int n = (int)tgid.x;
    const int m = (int)tgid.y;
    uint acc = 0u;
    for (int k = (int)lane; k < Kp2; k += 32) {
        ulong2 va = A[m * Kp2 + k];
        ulong2 vb = B[n * Kp2 + k];
        ulong2 xr = ~(va ^ vb);
        acc += (uint)popcount(xr.x) + (uint)popcount(xr.y);
    }
    const uint total = simd_sum(acc);
    if (lane == 0) {
        C[m * N + n] = 2 * (int)total - K_eff;
    }
}

// ── Threadgroup-tiled matmul ──────────────────────────────────────────────────
// Each threadgroup computes a (TM × TN) tile of output cells. A-tile and
// B-tile are loaded collaboratively into threadgroup memory, amortising
// global-memory bandwidth across TN and TM simdgroups respectively.
//
// Layout:  TG = (TN*32, TM, 1)  →  TM*TN simdgroups.
//   simdgroup_index_in_threadgroup = sg_y * TN + sg_x
//   Each simdgroup owns output cell  (m_base + sg_y, n_base + sg_x).
//
// K dimension is processed in KT-word tiles.  threadgroup memory:
//   A_tile[TM * KT],  B_tile[TN * KT]   (uint32; ~6 KB for TM=2,TN=4,KT=256)
//
// Grid  : (ceil(N/TN)*TN*32, ceil(M/TM)*TM, 1)
// TG    : (TN*32, TM, 1) = (128, 2, 1) = 256 threads

kernel void xnor_u32_tiled(
    device const uint* A     [[buffer(0)]],
    device const uint* B     [[buffer(1)]],
    device int*        C     [[buffer(2)]],
    constant int&      N     [[buffer(3)]],
    constant int&      Kp    [[buffer(4)]],
    constant int&      K_eff [[buffer(5)]],
    constant int&      M     [[buffer(6)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sg_id [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  tid   [[thread_index_in_threadgroup]])
{
    constexpr int TM = 2, TN = 4, KT = 256;
    threadgroup uint A_tile[TM * KT];
    threadgroup uint B_tile[TN * KT];

    const int n_base = (int)tgid.x;   // tgid.x = tile col index (0..ceil(N/TN)-1)
    const int m_base = (int)tgid.y;   // tgid.y = tile row index (0..ceil(M/TM)-1)

    // Decompose simdgroup index into (sg_y, sg_x) within the tile.
    const int sg_y = (int)sg_id / TN;  // 0 or 1 — which A-row in tile
    const int sg_x = (int)sg_id % TN;  // 0..3   — which B-row in tile

    const int m = m_base * TM + sg_y;
    const int n = n_base * TN + sg_x;

    uint total = 0u;

    for (int k_start = 0; k_start < Kp; k_start += KT) {
        const int k_cnt = min(KT, Kp - k_start);

        // ── Collaborative load ─────────────────────────────────────────────
        // All TM*TN*32 = 256 threads load A_tile (TM*KT ≤ 512 words) and
        // B_tile (TN*KT ≤ 1024 words) in strided loops.
        const int total_threads = TM * TN * 32;

        for (int t = (int)tid; t < TM * k_cnt; t += total_threads) {
            const int row = t / k_cnt, col = t % k_cnt;
            const int src_m = m_base * TM + row;
            A_tile[row * KT + col] =
                (src_m < M) ? A[src_m * Kp + k_start + col] : 0u;
        }
        for (int t = (int)tid; t < TN * k_cnt; t += total_threads) {
            const int row = t / k_cnt, col = t % k_cnt;
            const int src_n = n_base * TN + row;
            B_tile[row * KT + col] =
                (src_n < N) ? B[src_n * Kp + k_start + col] : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Compute from threadgroup memory ───────────────────────────────
        // acc is always accumulated so that simd_sum is called uniformly by
        // all 32 threads in the simdgroup — out-of-bounds threads just add 0.
        uint acc = 0u;
        if (m < M && n < N) {
            for (int k = (int)lane; k < k_cnt; k += 32) {
                acc += popcount(~(A_tile[sg_y * KT + k] ^ B_tile[sg_x * KT + k]));
            }
        }
        total += simd_sum(acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0 && m < M && n < N) {
        C[m * N + n] = 2 * (int)total - K_eff;
    }
}
