// Fused packed ops: matmul_sign, packed_majority, episodic_causal_search.
//
// All operate on uint64-packed (int64 storage) bit1 buffers.

#pragma once

#include <cstdint>
#include <climits>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda { namespace kernels {

// ── xnor_popcount_matmul_sign ─────────────────────────────────────────────
// A: (M, Kp), B: (N, Kp) → C: (M, Np) packed, C[m,n] = 1 iff K-2H >= 0.
// One thread per (m, n); uses 64-bit atomicOr to write the packed bit.
__global__ inline void k_xnor_popcount_matmul_sign(
    const uint64_t* __restrict__ A,
    const uint64_t* __restrict__ B,
    uint64_t* __restrict__ C,
    int M, int N, int Kp, int K_logical, int Np)
{
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    int h = 0;
    for (int k = 0; k < Kp; k++) {
        h += __popcll(A[(int64_t)m * Kp + k] ^ B[(int64_t)n * Kp + k]);
    }
    if (K_logical - 2 * h >= 0) {
        const int word = n / 64;
        const int bit  = n % 64;
        atomicOr(reinterpret_cast<unsigned long long*>(&C[(int64_t)m * Np + word]),
                 (unsigned long long)(1ULL << bit));
    }
}

// ── packed_majority ────────────────────────────────────────────────────────
// rows: (batch, k, Kp) → out: (batch, Kp).  One thread per (batch, word).
__global__ inline void k_packed_majority(
    const uint64_t* __restrict__ rows,
    uint64_t* __restrict__ out,
    int batch, int k, int Kp, int threshold, int n_bits, int D)
{
    const int total = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = total / Kp;
    const int w = total % Kp;
    if (b >= batch || w >= Kp) return;

    uint64_t partial[8] = {};
    for (int ki = 0; ki < k; ki++) {
        uint64_t carry = rows[((int64_t)b * k + ki) * Kp + w];
        for (int j = 0; j < n_bits && carry; j++) {
            const uint64_t s = partial[j] ^ carry;
            carry = partial[j] & carry;
            partial[j] = s;
        }
    }

    uint64_t greater = 0, equal = ~uint64_t(0);
    for (int bit = n_bits - 1; bit >= 0; bit--) {
        const int t_bit = (threshold >> bit) & 1;
        if (t_bit == 0) {
            greater |= equal & partial[bit];
            equal   &= ~partial[bit];
        } else {
            equal &= partial[bit];
        }
    }
    uint64_t result = greater | equal;
    if (w == Kp - 1 && D % 64 != 0) result &= (uint64_t(1) << (D % 64)) - 1;
    out[(int64_t)b * Kp + w] = result;
}

// ── episodic_causal_search ─────────────────────────────────────────────────
// One thread per batch element; sequential scan over valid slots.
__global__ inline void k_episodic_causal_search(
    const uint64_t* __restrict__ qc,
    const uint64_t* __restrict__ kc_buf,
    const uint64_t* __restrict__ qp,
    const uint64_t* __restrict__ pos_buf,
    const uint64_t* __restrict__ payload,
    const int32_t*  __restrict__ cnt,
    uint64_t* __restrict__ read_out,
    int32_t*  __restrict__ idx_out,
    int32_t*  __restrict__ score_out,
    int N_max, int Kp, int D)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    const int valid = cnt[b];
    if (valid <= 0) {
        idx_out[b]   = -1;
        score_out[b] = -(D * 2 + 2);
        return;
    }

    int best = INT_MIN, best_i = -1;
    for (int i = 0; i < valid; i++) {
        int c_h = 0, p_h = 0;
        for (int k = 0; k < Kp; k++) {
            c_h += __popcll(qc[(int64_t)b * Kp + k]
                             ^ kc_buf[((int64_t)b * N_max + i) * Kp + k]);
            p_h += __popcll(qp[(int64_t)b * Kp + k]
                             ^ pos_buf[((int64_t)b * N_max + i) * Kp + k]);
        }
        const int score = (D - 2*c_h) + (D - 2*p_h);
        if (score > best) { best = score; best_i = i; }
    }

    idx_out[b]   = best_i;
    score_out[b] = best;
    if (best_i >= 0) {
        for (int k = 0; k < Kp; k++)
            read_out[(int64_t)b * Kp + k] =
                payload[((int64_t)b * N_max + best_i) * Kp + k];
    }
}

}}} // cbrute::cuda::kernels
