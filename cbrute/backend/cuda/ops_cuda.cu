#ifdef HAVE_CUDA
#include "ops_cuda.h"
#include <torch/torch.h>

namespace cbrute { namespace cuda {

// ── pack_bits ────────────────────────────────────────────
__global__ void k_pack_u8(const float* in, uint8_t* out,
                           int64_t ld, int64_t pd, int64_t batch) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = blockIdx.y;
    if (j >= pd || i >= batch) return;
    uint8_t w = 0;
    for (int b = 0; b < 8; b++) {
        int64_t k = j * 8 + b;
        if (k < ld && in[i * ld + k] >= 0.f) w |= (uint8_t)(1u << b);
    }
    out[i * pd + j] = w;
}

__global__ void k_pack_u32(const float* in, int32_t* out,
                            int64_t ld, int64_t pd, int64_t batch) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = blockIdx.y;
    if (j >= pd || i >= batch) return;
    uint32_t w = 0;
    for (int b = 0; b < 32; b++) {
        int64_t k = j * 32 + b;
        if (k < ld && in[i * ld + k] >= 0.f) w |= (1u << b);
    }
    out[i * pd + j] = (int32_t)w;
}

__global__ void k_pack_u64(const float* in, int64_t* out,
                            int64_t ld, int64_t pd, int64_t batch) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = blockIdx.y;
    if (j >= pd || i >= batch) return;
    uint64_t w = 0;
    for (int b = 0; b < 64; b++) {
        int64_t k = j * 64 + b;
        if (k < ld && in[i * ld + k] >= 0.f) w |= (1ull << b);
    }
    out[i * pd + j] = (int64_t)w;
}

at::Tensor pack_bits(const at::Tensor& input, int64_t pw) {
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64);
    auto inp      = input.contiguous().to(at::kFloat);
    int64_t ld    = inp.size(-1);
    int64_t pd    = (ld + pw - 1) / pw;
    int64_t batch = inp.numel() / ld;
    auto out_shape = inp.sizes().vec(); out_shape.back() = pd;
    at::ScalarType st = (pw == 8) ? at::kByte : (pw == 32) ? at::kInt : at::kLong;
    auto output = at::zeros(out_shape, inp.options().dtype(st));
    dim3 threads(256), grid((pd + 255) / 256, (unsigned)batch);
    if (pw == 8)
        k_pack_u8<<<grid, threads>>>(inp.data_ptr<float>(), output.data_ptr<uint8_t>(), ld, pd, batch);
    else if (pw == 32)
        k_pack_u32<<<grid, threads>>>(inp.data_ptr<float>(), output.data_ptr<int32_t>(), ld, pd, batch);
    else
        k_pack_u64<<<grid, threads>>>(inp.data_ptr<float>(), output.data_ptr<int64_t>(), ld, pd, batch);
    return output;
}

// ── unpack_bits ──────────────────────────────────────────
__global__ void k_unpack_u8(const uint8_t* in, float* out,
                             int64_t ll, int64_t pl, int64_t batch) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = blockIdx.y;
    if (j >= pl || i >= batch) return;
    uint8_t w = in[i * pl + j];
    for (int b = 0; b < 8; b++) {
        int64_t k = j * 8 + b;
        if (k < ll) out[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
    }
}

__global__ void k_unpack_u32(const int32_t* in, float* out,
                              int64_t ll, int64_t pl, int64_t batch) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = blockIdx.y;
    if (j >= pl || i >= batch) return;
    uint32_t w = (uint32_t)in[i * pl + j];
    for (int b = 0; b < 32; b++) {
        int64_t k = j * 32 + b;
        if (k < ll) out[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
    }
}

__global__ void k_unpack_u64(const int64_t* in, float* out,
                              int64_t ll, int64_t pl, int64_t batch) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = blockIdx.y;
    if (j >= pl || i >= batch) return;
    uint64_t w = (uint64_t)in[i * pl + j];
    for (int b = 0; b < 64; b++) {
        int64_t k = j * 64 + b;
        if (k < ll) out[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
    }
}

at::Tensor unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw) {
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64);
    auto p        = packed.contiguous();
    int64_t ll    = logical_shape.back();
    int64_t pl    = p.size(-1);
    int64_t batch = p.numel() / pl;
    auto output   = at::empty(logical_shape.vec(), p.options().dtype(at::kFloat));
    dim3 threads(256), grid((pl + 255) / 256, (unsigned)batch);
    if (pw == 8)
        k_unpack_u8<<<grid, threads>>>(p.data_ptr<uint8_t>(), output.data_ptr<float>(), ll, pl, batch);
    else if (pw == 32)
        k_unpack_u32<<<grid, threads>>>(p.data_ptr<int32_t>(), output.data_ptr<float>(), ll, pl, batch);
    else
        k_unpack_u64<<<grid, threads>>>(p.data_ptr<int64_t>(), output.data_ptr<float>(), ll, pl, batch);
    return output;
}

// ── xnor_popcount_matmul ─────────────────────────────────
__global__ void k_xnor_u64(const int64_t* A, const int64_t* B, int32_t* C,
                            int64_t M, int64_t N, int64_t Kp, int64_t K) {
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    int32_t acc = 0;
    for (int64_t k = 0; k < Kp; k++)
        acc += __popcll(~((uint64_t)A[m*Kp+k] ^ (uint64_t)B[n*Kp+k]));
    C[m*N+n] = 2*acc - (int32_t)K;
}

__global__ void k_xnor_u32(const int32_t* A, const int32_t* B, int32_t* C,
                            int64_t M, int64_t N, int64_t Kp, int64_t K) {
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    int32_t acc = 0;
    for (int64_t k = 0; k < Kp; k++)
        acc += __popc(~((uint32_t)A[m*Kp+k] ^ (uint32_t)B[n*Kp+k]));
    C[m*N+n] = 2*acc - (int32_t)K;
}

__global__ void k_xnor_u8(const uint8_t* A, const uint8_t* B, int32_t* C,
                           int64_t M, int64_t N, int64_t Kp, int64_t K) {
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    int32_t acc = 0;
    for (int64_t k = 0; k < Kp; k++)
        acc += __popc((uint32_t)(uint8_t)(~(A[m*Kp+k] ^ B[n*Kp+k])));
    C[m*N+n] = 2*acc - (int32_t)K;
}

at::Tensor xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K, int64_t pw) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    auto Ac = A.contiguous(), Bc = B.contiguous();
    auto C  = at::zeros({M, N}, A.options().dtype(at::kInt));
    dim3 threads(16, 16), grid((N+15)/16, (M+15)/16);
    if (pw == 64)
        k_xnor_u64<<<grid,threads>>>(Ac.data_ptr<int64_t>(), Bc.data_ptr<int64_t>(),
                                      C.data_ptr<int32_t>(), M, N, Kp, K);
    else if (pw == 32)
        k_xnor_u32<<<grid,threads>>>(Ac.data_ptr<int32_t>(), Bc.data_ptr<int32_t>(),
                                      C.data_ptr<int32_t>(), M, N, Kp, K);
    else
        k_xnor_u8<<<grid,threads>>>(Ac.data_ptr<uint8_t>(), Bc.data_ptr<uint8_t>(),
                                     C.data_ptr<int32_t>(), M, N, Kp, K);
    return C;
}

// ── popcount ─────────────────────────────────────────────
__global__ void k_popcnt_u8(const uint8_t* in, int32_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popc((uint32_t)in[i]);
}
__global__ void k_popcnt_u32(const int32_t* in, int32_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popc((uint32_t)in[i]);
}
__global__ void k_popcnt_u64(const int64_t* in, int32_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popcll((uint64_t)in[i]);
}

at::Tensor popcount(const at::Tensor& packed) {
    auto p   = packed.contiguous();
    auto out = at::empty(p.sizes(), p.options().dtype(at::kInt));
    int64_t n = p.numel();
    dim3 threads(256), grid((n + 255) / 256);
    if (p.scalar_type() == at::kByte)
        k_popcnt_u8<<<grid,threads>>>(p.data_ptr<uint8_t>(), out.data_ptr<int32_t>(), n);
    else if (p.scalar_type() == at::kInt)
        k_popcnt_u32<<<grid,threads>>>(p.data_ptr<int32_t>(), out.data_ptr<int32_t>(), n);
    else
        k_popcnt_u64<<<grid,threads>>>(p.data_ptr<int64_t>(), out.data_ptr<int32_t>(), n);
    return out;
}

at::Tensor hamming_distance(const at::Tensor& A, const at::Tensor& B) {
    return popcount(at::bitwise_xor(A, B));
}

at::Tensor bitwise_and(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_and(A, B); }
at::Tensor bitwise_or (const at::Tensor& A, const at::Tensor& B) { return at::bitwise_or(A, B);  }
at::Tensor bitwise_xor(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_xor(A, B); }
at::Tensor bitwise_not(const at::Tensor& A)                       { return at::bitwise_not(A);    }

at::Tensor& randomize_bits(at::Tensor& out) { out.random_(); return out; }

}} // cbrute::cuda
#endif // HAVE_CUDA
