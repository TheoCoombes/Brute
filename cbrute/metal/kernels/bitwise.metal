// Elementwise bitwise kernels. PyTorch's TensorIterator on MPS already
// vectorizes these on integer dtypes; we keep them here only so the brute::
// schema's MPS impl table is uniform with CPU/CUDA. One thread per word —
// trivially memory-bound.

#include <metal_stdlib>
using namespace metal;

kernel void bw_and_u8 (device const uchar* a[[buffer(0)]], device const uchar* b[[buffer(1)]], device uchar* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] & b[g]; }
kernel void bw_and_u32(device const uint*  a[[buffer(0)]], device const uint*  b[[buffer(1)]], device uint*  c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] & b[g]; }
kernel void bw_and_u64(device const ulong* a[[buffer(0)]], device const ulong* b[[buffer(1)]], device ulong* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] & b[g]; }

kernel void bw_or_u8  (device const uchar* a[[buffer(0)]], device const uchar* b[[buffer(1)]], device uchar* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] | b[g]; }
kernel void bw_or_u32 (device const uint*  a[[buffer(0)]], device const uint*  b[[buffer(1)]], device uint*  c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] | b[g]; }
kernel void bw_or_u64 (device const ulong* a[[buffer(0)]], device const ulong* b[[buffer(1)]], device ulong* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] | b[g]; }

kernel void bw_xor_u8 (device const uchar* a[[buffer(0)]], device const uchar* b[[buffer(1)]], device uchar* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] ^ b[g]; }
kernel void bw_xor_u32(device const uint*  a[[buffer(0)]], device const uint*  b[[buffer(1)]], device uint*  c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] ^ b[g]; }
kernel void bw_xor_u64(device const ulong* a[[buffer(0)]], device const ulong* b[[buffer(1)]], device ulong* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] ^ b[g]; }

kernel void bw_not_u8 (device const uchar* a[[buffer(0)]], device uchar* b[[buffer(1)]], uint g[[thread_position_in_grid]]) { b[g] = (uchar)~a[g]; }
kernel void bw_not_u32(device const uint*  a[[buffer(0)]], device uint*  b[[buffer(1)]], uint g[[thread_position_in_grid]]) { b[g] = ~a[g]; }
kernel void bw_not_u64(device const ulong* a[[buffer(0)]], device ulong* b[[buffer(1)]], uint g[[thread_position_in_grid]]) { b[g] = ~a[g]; }
