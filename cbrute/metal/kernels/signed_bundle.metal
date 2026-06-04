// Signed vote bundle — int8-weight × bit1 → bit1.
//
//   W: (B, M, N) int8     V: (B, N, Dp) ulong packed   →   O: (B, M, Dp) ulong
//   O[b,m,d] = sign( Σ_n W[b,m,n] · pm1(V[b,n,d]) ),   pm1(bit) = bit ? +1 : -1
//
// One thread owns one output WORD (b, m, dp): it loops over the N source rows,
// accumulating a 64-entry signed vote register (one per bit of the word), then
// packs the signs into the output word.  V is read straight from its packed
// words — never materialised — and pad bits (d >= D) are left 0.  sign(0)=+1.
//
// Dispatch (set in the .mm driver):
//   grid = (B*M*Dp, 1, 1)   threadgroup = (256, 1, 1)

#include <metal_stdlib>
using namespace metal;

kernel void signed_bundle_kernel(
    device const char*  W  [[buffer(0)]],     // (B,M,N) int8
    device const ulong* V  [[buffer(1)]],     // (B,N,Dp) packed bit1
    device ulong*       O  [[buffer(2)]],     // (B,M,Dp) packed bit1
    constant int&       B  [[buffer(3)]],
    constant int&       M  [[buffer(4)]],
    constant int&       N  [[buffer(5)]],
    constant int&       Dp [[buffer(6)]],
    constant int&       D  [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const int total = B * M * Dp;
    if ((int)gid >= total) return;

    const int dp = (int)gid % Dp;
    const int bm = (int)gid / Dp;
    const int m  = bm % M;
    const int b  = bm / M;

    int acc[64];
    for (int i = 0; i < 64; ++i) acc[i] = 0;

    device const char* wrow = W + ((long)b * M + m) * N;
    const long vbat = (long)b * N * Dp;
    for (int n = 0; n < N; ++n) {
        const int wv = (int)wrow[n];
        if (wv == 0) continue;
        const ulong word = V[vbat + (long)n * Dp + dp];
        for (int bit = 0; bit < 64; ++bit)
            acc[bit] += ((word >> bit) & 1UL) ? wv : -wv;
    }

    ulong outw = 0;
    const int base = dp * 64;
    for (int bit = 0; bit < 64; ++bit) {
        if (base + bit >= D) break;               // pad bits stay 0
        if (acc[bit] >= 0) outw |= (1UL << bit);   // sign(0) = +1
    }
    O[((long)b * M + m) * Dp + dp] = outw;
}
