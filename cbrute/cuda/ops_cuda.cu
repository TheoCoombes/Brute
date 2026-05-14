// brute CUDA backend — driver layer.
//
// Public ops registered as TORCH_LIBRARY_IMPL handlers (see ext.cpp).
// All heavy lifting is delegated to the kernels in `kernels/`. This file
// only:
//   * validates the at::Tensor inputs,
//   * allocates outputs ONCE (before launching kernels),
//   * picks the right kernel variant (pack-width / dtype / CUTLASS vs fallback),
//   * launches with sensible grid/block dims on the current CUDA stream.
//
// Invariants:
//   * Zero device-side allocations inside any kernel.
//   * All kernels see contiguous, type-stable raw pointers.
//   * Output is zero-initialized when pad-bytes must remain zero (pack ops,
//     packed_popcount accumulator) — otherwise allocated with at::empty.
//   * Bit-exact parity with the CPU implementation for every op.

#ifdef HAVE_CUDA
#include "ops_cuda.h"
#include "kernels/packing.cuh"
#include "kernels/popcount.cuh"
#include "kernels/matmul_fallback.cuh"
#include "kernels/matmul_cutlass.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda {

namespace {

at::ScalarType pack_scalar_type(int64_t pw) {
    switch (pw) {
        case 8:  return at::kByte;
        case 32: return at::kInt;
        case 64: return at::kLong;
        default: TORCH_CHECK(false, "pack_width must be 8, 32, or 64; got ", pw);
    }
}

inline cudaStream_t cur_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

// Compute capability of the current device, as cc = major*10 + minor.
inline int compute_capability() {
    int dev = -1;
    cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor;
}

// Reinterpret a byte-aligned device pointer as a uint64*. Caller guarantees
// at least 8-byte alignment (true for any at::Tensor data_ptr).
inline const uint64_t* as_u64(const at::Tensor& t) { return reinterpret_cast<const uint64_t*>(t.data_ptr()); }
inline uint8_t*        as_u8 (at::Tensor& t)       { return reinterpret_cast<uint8_t*>(t.data_ptr()); }
inline const uint8_t*  as_u8 (const at::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); }

}  // anon

// 
// pack_bits (float input → packed bits)
// 
at::Tensor pack_bits(const at::Tensor& input, int64_t pw) {
    TORCH_CHECK(input.dim() >= 1, "pack_bits: input must have >= 1 dim");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "pack_bits: pack_width must be 8, 32, or 64");

    const auto inp = input.contiguous().to(at::kFloat);
    const int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, inp.options().dtype(pack_scalar_type(pw)));
    }

    const int64_t pd_words  = (ld + pw - 1) / pw;
    const int64_t row_bytes = pd_words * (pw / 8);
    const int64_t batch     = inp.numel() / ld;
    out_shape.back() = pd_words;

     auto output = at::zeros(out_shape, inp.options().dtype(pack_scalar_type(pw)));
     const int chunks = (int)((ld + 31) / 32);
     const bool aligned4 = (pw == 32 || pw == 64);
     dim3 grid((unsigned)chunks, (unsigned)batch);
     dim3 block(32);   // exactly one warp per block — matches __ballot_sync usage
     kernels::k_pack_float_warp<<<grid, block, 0, cur_stream()>>>(
         inp.data_ptr<float>(),
         as_u8(output),
         ld, row_bytes, aligned4);
     return output;
 }

// 
// pack_bool (bool input → packed bits; fast path used by Python _pack_bool)
// 
at::Tensor pack_bool(const at::Tensor& input, int64_t pw) {
    TORCH_CHECK(input.dim() >= 1, "pack_bool: input must have >= 1 dim");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "pack_bool: pack_width must be 8, 32, or 64");
    TORCH_CHECK(input.scalar_type() == at::kBool,
                "pack_bool: input must be torch.bool");

    const auto inp = input.contiguous();
    const int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, inp.options().dtype(pack_scalar_type(pw)));
    }

    const int64_t pd_words  = (ld + pw - 1) / pw;
    const int64_t row_bytes = pd_words * (pw / 8);
    const int64_t batch     = inp.numel() / ld;
    out_shape.back() = pd_words;

     auto output = at::zeros(out_shape, inp.options().dtype(pack_scalar_type(pw)));
     const int chunks = (int)((ld + 31) / 32);
     const bool aligned4 = (pw == 32 || pw == 64);
     dim3 grid((unsigned)chunks, (unsigned)batch);
     dim3 block(32);
     kernels::k_pack_bool_warp<<<grid, block, 0, cur_stream()>>>(
         reinterpret_cast<const uint8_t*>(inp.data_ptr<bool>()),
         as_u8(output),
         ld, row_bytes, aligned4);
     return output;
 }

// 
// unpack_bits (packed → ±1 float32)
// 
at::Tensor unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw) {
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "unpack_bits: pack_width must be 8, 32, or 64");

    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0) {
        return at::empty(logical_shape.vec(), p.options().dtype(at::kFloat));
    }

     const int64_t pl_words  = p.size(-1);
     const int64_t row_bytes = pl_words * (pw / 8);
     const int64_t batch     = p.numel() / pl_words;
     const bool aligned4 = (pw == 32 || pw == 64);

     auto output = at::empty(logical_shape.vec(), p.options().dtype(at::kFloat));
     const int chunks = (int)((ll + 31) / 32);
     dim3 grid((unsigned)chunks, (unsigned)batch);
     dim3 block(32);
     kernels::k_unpack_pm1_warp<<<grid, block, 0, cur_stream()>>>(
         as_u8(p),
         output.data_ptr<float>(),
         ll, row_bytes, aligned4);
     return output;
 }

// 
// unpack_bool (packed → bool)
// 
at::Tensor unpack_bool(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw) {
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "unpack_bool: pack_width must be 8, 32, or 64");

    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0) {
        return at::empty(logical_shape.vec(), p.options().dtype(at::kBool));
    }

     const int64_t pl_words  = p.size(-1);
     const int64_t row_bytes = pl_words * (pw / 8);
     const int64_t batch     = p.numel() / pl_words;
     const bool aligned4 = (pw == 32 || pw == 64);

     auto output = at::empty(logical_shape.vec(), p.options().dtype(at::kBool));
     const int chunks = (int)((ll + 31) / 32);
     dim3 grid((unsigned)chunks, (unsigned)batch);
     dim3 block(32);
     kernels::k_unpack_bool_warp<<<grid, block, 0, cur_stream()>>>(
         as_u8(p),
         reinterpret_cast<uint8_t*>(output.data_ptr<bool>()),
         ll, row_bytes, aligned4);
     return output;
 }

// 
// xnor_popcount_matmul — CUTLASS B1 GEMM where supported, hand kernel else.
// Output semantic identical to CPU: C = 2*popc_xnor − K_eff = K − 2*popc_xor.
// 
at::Tensor xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B,
                                int64_t K, int64_t pw) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "xnor_popcount_matmul: inputs must be 2-D");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64,
                "pack_width must be 8, 32, or 64");
    const int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    TORCH_CHECK(Kp == B.size(1), "xnor_popcount_matmul: packed K mismatch");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C  = at::zeros({M, N}, A.options().dtype(at::kInt));

    const int64_t Kp_bits = Kp * pw;
    const int     cc      = compute_capability();
    const bool    cutlass_ok =
        (cc >= 80) && (Kp_bits % 256 == 0) && (Kp_bits >= 256)
        && (M % 8 == 0) && (N % 8 == 0);

    if (cutlass_ok) {
        // CUTLASS B1 XOR-popc GEMM. The accumulator is the Hamming distance H;
        // we then post-process C = K − 2·H to match the CPU semantic.
        using Gemm = kernels::CutlassB1XorGemm;
        Gemm gemm_op;

        // Bit-level leading dimensions: A is (M, Kp_bits), B is laid out in
        // memory as (N, Kp_bits) row-major, which is equivalent to a
        // (Kp_bits, N) column-major view with leading dim = Kp_bits.
        typename Gemm::Arguments args{
            {(int)M, (int)N, (int)Kp_bits},
            {reinterpret_cast<cutlass::uint1b_t const*>(Ac.data_ptr()), (int)Kp_bits},
            {reinterpret_cast<cutlass::uint1b_t const*>(Bc.data_ptr()), (int)Kp_bits},
            {C.data_ptr<int32_t>(), (int)N},
            {C.data_ptr<int32_t>(), (int)N},
            {1, 0}    // alpha=1, beta=0 — accum is H directly.
        };

        cutlass::Status status = gemm_op(args, /*workspace=*/nullptr, cur_stream());
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "CUTLASS B1 GEMM failed: ", int(status));

        const int64_t n_elems = M * N;
        dim3 block(256);
        dim3 grid((unsigned)((n_elems + 255) / 256));
        kernels::k_kminus2_inplace<<<grid, block, 0, cur_stream()>>>(
            C.data_ptr<int32_t>(), n_elems, (int32_t)K);
        return C;
    }

    // Fallback path: hand-tuned XNOR-popc kernel.
    const int32_t K_eff = (int32_t)(2LL * Kp * pw - K);
    dim3 block(16, 16);
    dim3 grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16));

    if (pw == 64) {
        kernels::k_xnor_popcount_matmul<uint64_t><<<grid, block, 0, cur_stream()>>>(
            reinterpret_cast<const uint64_t*>(Ac.data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(Bc.data_ptr<int64_t>()),
            C.data_ptr<int32_t>(), M, N, Kp, K_eff);
    } else if (pw == 32) {
        kernels::k_xnor_popcount_matmul<uint32_t><<<grid, block, 0, cur_stream()>>>(
            reinterpret_cast<const uint32_t*>(Ac.data_ptr<int32_t>()),
            reinterpret_cast<const uint32_t*>(Bc.data_ptr<int32_t>()),
            C.data_ptr<int32_t>(), M, N, Kp, K_eff);
    } else { // pw == 8
        kernels::k_xnor_popcount_matmul<uint8_t><<<grid, block, 0, cur_stream()>>>(
            Ac.data_ptr<uint8_t>(),
            Bc.data_ptr<uint8_t>(),
            C.data_ptr<int32_t>(), M, N, Kp, K_eff);
    }
    return C;
}

// 
// popcount — per-element, output int32. Supports all integer dtypes (+ bool).
// 
at::Tensor popcount(const at::Tensor& x) {
    const auto p = x.contiguous();
    auto out = at::empty(p.sizes(), p.options().dtype(at::kInt));
    const int64_t n = p.numel();
    if (n == 0) return out;

    dim3 block(256);
    dim3 grid((unsigned)((n + 255) / 256));

    switch (p.scalar_type()) {
        case at::kByte:
        case at::kBool:
        case at::kChar:
            kernels::k_popcnt_u8<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint8_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        case at::kShort:
            kernels::k_popcnt_u16<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint16_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        case at::kInt:
            kernels::k_popcnt_u32<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint32_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        case at::kLong:
            kernels::k_popcnt_u64<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint64_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        default:
            TORCH_CHECK(false, "popcount: unsupported dtype ", p.scalar_type());
    }
    return out;
}

// 
// packed_popcount — total bit count across the buffer (int64 scalar).
// 
at::Tensor packed_popcount(const at::Tensor& x) {
    const auto p = x.contiguous();
    auto out = at::zeros({}, p.options().dtype(at::kLong));

    const int64_t n_bytes = (int64_t)p.nbytes();
    const int64_t n_words = n_bytes / (int64_t)sizeof(uint64_t);
    const int64_t tail    = n_bytes - n_words * (int64_t)sizeof(uint64_t);

    if (n_words > 0) {
        dim3 block(256);
        dim3 grid((unsigned)std::min<int64_t>((n_words + 255) / 256, 65535));
        kernels::k_packed_popcount_total<<<grid, block, 0, cur_stream()>>>(
            as_u64(p), n_words,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    if (tail > 0) {
        kernels::k_popcount_tail_byte<<<1, 1, 0, cur_stream()>>>(
            as_u8(p) + n_words * sizeof(uint64_t), (int)tail,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    return out;
}

// 
// hamming_distance — per-element popcount(A^B), output int32.
// Broadcasting handled by torch's TensorIterator on the XOR.
// 
at::Tensor hamming_distance(const at::Tensor& A, const at::Tensor& B) {
    return popcount(at::bitwise_xor(A, B));
}

// 
// bit1_hamming_total — fused XOR + popcount over equal-shape packed buffers
// (int64 scalar). No XOR temporary.
// 
at::Tensor bit1_hamming_total(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.sizes()       == B.sizes(),       "bit1_hamming_total: shape mismatch");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "bit1_hamming_total: dtype mismatch");
    const auto Ac = A.contiguous();
    const auto Bc = B.contiguous();
    auto out = at::zeros({}, A.options().dtype(at::kLong));

    const int64_t n_bytes = (int64_t)Ac.nbytes();
    const int64_t n_words = n_bytes / (int64_t)sizeof(uint64_t);
    const int64_t tail    = n_bytes - n_words * (int64_t)sizeof(uint64_t);

    if (n_words > 0) {
        dim3 block(256);
        dim3 grid((unsigned)std::min<int64_t>((n_words + 255) / 256, 65535));
        kernels::k_hamming_total<<<grid, block, 0, cur_stream()>>>(
            as_u64(Ac), as_u64(Bc), n_words,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    if (tail > 0) {
        kernels::k_hamming_tail_byte<<<1, 1, 0, cur_stream()>>>(
            as_u8(Ac) + n_words * sizeof(uint64_t),
            as_u8(Bc) + n_words * sizeof(uint64_t),
            (int)tail,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    return out;
}

// 
// Bitwise pass-throughs.
// 
at::Tensor bitwise_and(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_and(A, B); }
at::Tensor bitwise_or (const at::Tensor& A, const at::Tensor& B) { return at::bitwise_or (A, B); }
at::Tensor bitwise_xor(const at::Tensor& A, const at::Tensor& B) { return at::bitwise_xor(A, B); }
at::Tensor bitwise_not(const at::Tensor& A)                       { return at::bitwise_not(A);    }

at::Tensor& randomize_bits(at::Tensor& out) { out.random_(); return out; }

}} // cbrute::cuda
#endif // HAVE_CUDA
