// Signed vote bundle — int8-weight × bit1 → bit1 (CUDA).
//
//   W: (B, M, N) int8     V: (B, N, Dp) uint64 packed   →   O: (B, M, Dp) uint64
//   O[b,m,d] = sign( Σ_n W[b,m,n] · pm1(V[b,n,d]) ),   pm1(bit) = bit ? +1 : -1
//
// One thread owns one output WORD (b, m, dp): it loops over the N source rows,
// accumulating a 64-entry signed vote register, then packs the signs.  V is read
// straight from its packed words; pad bits (d >= D) stay 0; sign(0) = +1.  This
// mirrors the CPU/Metal kernels exactly and is bit-identical to the int8-clamped
// reference.  NOTE: this Mac has no NVIDIA GPU — the CUDA path is written to
// mirror the existing CUDA kernels and is compile-shaped only here, not run.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda { namespace kernels {

__global__ void k_signed_bundle(
    const int8_t*   __restrict__ W,    // (B,M,N)
    const uint64_t* __restrict__ V,    // (B,N,Dp)
    uint64_t*       __restrict__ O,    // (B,M,Dp)
    int B, int M, int N, int Dp, int D)
{
    const long total = (long)B * M * Dp;
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    const int dp = (int)(gid % Dp);
    const long bm = gid / Dp;
    const int m  = (int)(bm % M);
    const int b  = (int)(bm / M);

    int acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0;

    const int8_t* wrow = W + ((long)b * M + m) * N;
    const long vbat = (long)b * N * Dp;
    for (int n = 0; n < N; ++n) {
        const int wv = (int)wrow[n];
        if (wv == 0) continue;
        const uint64_t word = V[vbat + (long)n * Dp + dp];
        for (int bit = 0; bit < 64; ++bit)
            acc[bit] += ((word >> bit) & 1ULL) ? wv : -wv;
    }

    uint64_t outw = 0;
    const int base = dp * 64;
    for (int bit = 0; bit < 64; ++bit) {
        if (base + bit >= D) break;                  // pad bits stay 0
        if (acc[bit] >= 0) outw |= (uint64_t(1) << bit);   // sign(0) = +1
    }
    O[((long)b * M + m) * Dp + dp] = outw;
}

}}} // cbrute::cuda::kernels
