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

// ── xnor_u64_sign ────────────────────────────────────────────────────────────
//
// Like xnor_u64 but packs sign(K − 2H) directly into bit1 output.
//
// Output C: (M, Np) ulong packed, logical (M, N) bit1.
//   C[m, n] = 1  iff  K_logical − 2·popc_xor(A[m,:], B[n,:]) >= 0
//
// Dispatch:
//   grid        = (Np * 64, M, 1)   (Np = ceil(N/64))
//   threadgroup = (64, 1, 1)        // 2 simdgroups: lower/upper 32 bits
//   tgid.x = nw (output word), tgid.y = m
//   lanes 0..31  → simdgroup 0 → lower 32 bits of word nw
//   lanes 32..63 → simdgroup 1 → upper 32 bits of word nw
//
// No atomics: lane 0 of each simdgroup writes its half to non-overlapping
// 4-byte-aligned regions of the int64 output word.

kernel void xnor_u64_sign(
    device const ulong* A         [[buffer(0)]],
    device const ulong* B         [[buffer(1)]],
    device ulong*       C         [[buffer(2)]],  // (M, Np) int64 packed, zero-initialised
    constant int&       N         [[buffer(3)]],
    constant int&       Kp        [[buffer(4)]],
    constant int&       K_logical [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])    // 0..63
{
    const int nw = (int)tgid.x;   // output word index 0..Np-1
    const int m  = (int)tgid.y;   // row index
    const int Np = (N + 63) / 64;
    if (nw >= Np) return;

    // Each lane handles one n-value within this word.
    // lanes 0..31: n = nw*64 + 0..31
    // lanes 32..63: n = nw*64 + 32..63
    const int n = nw * 64 + (int)lane;
    bool result = false;
    if (n < N) {
        uint acc = 0u;
        for (int k = 0; k < Kp; k++) {
            acc += (uint)popcount(A[(ulong)m * Kp + k] ^ B[(ulong)n * Kp + k]);
        }
        result = ((int)K_logical - 2 * (int)acc) >= 0;
    }

    // Pack 32 results via simd_ballot; each simdgroup covers a 32-bit half of nw.
    const uint sg_mask = (uint)((ulong)simd_ballot(result));

    // Write each half directly from the first lane of its simdgroup.
    // Treat C as uint (32-bit) array: index = (m*Np + nw)*2 + half_id (0 or 1).
    device uint* C_halves = (device uint*)C;
    if (lane == 0) {
        C_halves[(ulong)(m * Np + nw) * 2 + 0] = sg_mask;  // lower 32 bits
    }
    if (lane == 32) {
        C_halves[(ulong)(m * Np + nw) * 2 + 1] = sg_mask;  // upper 32 bits
    }
}

// ── packed_majority ───────────────────────────────────────────────────────────
//
// Bit-sliced majority vote over k packed rows per (batch, word) pair.
// rows: (batch, k, Kp) ulong packed.  Output: (batch, Kp) ulong packed.
//
// Dispatch: one thread per (batch_element, word_index).
//   grid = (batch * Kp, 1, 1), threadgroup = (256, 1, 1)

kernel void packed_majority_k(
    device const ulong* rows      [[buffer(0)]],   // (batch, k, Kp)
    device       ulong* out       [[buffer(1)]],   // (batch, Kp)
    constant int&       k_val     [[buffer(2)]],
    constant int&       Kp_val    [[buffer(3)]],
    constant int&       threshold [[buffer(4)]],
    constant int&       n_bits    [[buffer(5)]],
    constant int&       D_val     [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const int Kp_  = Kp_val;
    const int b    = (int)tid / Kp_;
    const int w    = (int)tid % Kp_;

    ulong partial[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Carry-ripple addition of k rows.
    for (int ki = 0; ki < k_val; ki++) {
        ulong carry = rows[((ulong)b * k_val + ki) * Kp_ + w];
        for (int j = 0; j < n_bits && carry != 0; j++) {
            ulong s  = partial[j] ^ carry;
            carry    = partial[j] & carry;
            partial[j] = s;
        }
    }

    // Bit-parallel >= threshold comparison.
    ulong greater = 0;
    ulong equal   = ~ulong(0);
    for (int bit = n_bits - 1; bit >= 0; bit--) {
        const int t_bit = (threshold >> bit) & 1;
        if (t_bit == 0) {
            greater |= equal & partial[bit];
            equal   &= ~partial[bit];
        } else {
            equal &= partial[bit];
        }
    }
    ulong result = greater | equal;

    // Zero pad bits in the last word.
    if (w == Kp_ - 1 && D_val % 64 != 0) {
        result &= (ulong(1) << (D_val % 64)) - 1;
    }
    out[(ulong)b * Kp_ + w] = result;
}

// ── episodic_causal_search ────────────────────────────────────────────────────
//
// Windowed content+position Hamming scan + top-1 + payload gather.
// One thread per batch element; iterates over `cnt[b]` valid ring-buffer slots.
//
// qc, qp: (B, Kp) packed   kc_buf, pos_buf, payload: (B, N_max, Kp) packed
// cnt: (B,) int32           read_out: (B, Kp), idx_out/score_out: (B,) int32
//
// Dispatch: grid = (B, 1, 1), threadgroup = (256, 1, 1)

kernel void episodic_search_k(
    device const ulong*  qc        [[buffer(0)]],
    device const ulong*  kc_buf    [[buffer(1)]],
    device const ulong*  qp        [[buffer(2)]],
    device const ulong*  pos_buf   [[buffer(3)]],
    device const ulong*  payload   [[buffer(4)]],
    device const int*    cnt       [[buffer(5)]],
    device       ulong*  read_out  [[buffer(6)]],
    device       int*    idx_out   [[buffer(7)]],
    device       int*    score_out [[buffer(8)]],
    constant int&        N_max     [[buffer(9)]],
    constant int&        Kp_       [[buffer(10)]],
    constant int&        D_        [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    const int b     = (int)tid;
    const int valid = cnt[b];
    if (valid <= 0) {
        idx_out[b]   = -1;
        score_out[b] = -(D_ * 2 + 2);
        return;
    }

    int best_score = INT_MIN;
    int best_i     = -1;

    for (int i = 0; i < valid; i++) {
        uint c_acc = 0, p_acc = 0;
        for (int k = 0; k < Kp_; k++) {
            c_acc += (uint)popcount(qc[(ulong)b * Kp_ + k]
                                    ^ kc_buf[((ulong)b * N_max + i) * Kp_ + k]);
            p_acc += (uint)popcount(qp[(ulong)b * Kp_ + k]
                                    ^ pos_buf[((ulong)b * N_max + i) * Kp_ + k]);
        }
        const int score = (D_ - 2*(int)c_acc) + (D_ - 2*(int)p_acc);
        if (score > best_score) { best_score = score; best_i = i; }
    }

    idx_out[b]   = best_i;
    score_out[b] = best_score;
    if (best_i >= 0) {
        for (int k = 0; k < Kp_; k++)
            read_out[(ulong)b * Kp_ + k] =
                payload[((ulong)b * N_max + best_i) * Kp_ + k];
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
