#ifdef HAVE_MPS
#include <libpopcnt.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <torch/torch.h>

// PyTorch MPS stream integration (not in TORCH_STABLE_ONLY builds).
#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
  #include <ATen/mps/MPSStream.h>
  #define BRUTE_MPS_STREAM 1
#endif

#include "ops_metal.h"
#include <map>
#include <mutex>
#include <string>

namespace cbrute { namespace mps {

// ── MSL kernel source ────────────────────────────────────
static const char* kMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// pack_bits: float -> packed integer. Bit=1 when input >= 0, packed LSB-first.
kernel void pack_u8(
    device const float* input [[buffer(0)]],
    device uint8_t*     output[[buffer(1)]],
    constant int&       ld    [[buffer(2)]],
    constant int&       pd    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int j = (int)gid.x, i = (int)gid.y;
    uint8_t w = 0;
    for (int b = 0; b < 8; b++) {
        int k = j * 8 + b;
        if (k < ld && input[i * ld + k] > 0.f) w |= (uint8_t)(1u << b);
    }
    output[i * pd + j] = w;
}
kernel void pack_u32(
    device const float* input [[buffer(0)]],
    device uint*        output[[buffer(1)]],
    constant int&       ld    [[buffer(2)]],
    constant int&       pd    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int j = (int)gid.x, i = (int)gid.y;
    uint w = 0;
    for (int b = 0; b < 32; b++) {
        int k = j * 32 + b;
        if (k < ld && input[i * ld + k] > 0.f) w |= (1u << b);
    }
    output[i * pd + j] = w;
}
kernel void pack_u64(
    device const float* input [[buffer(0)]],
    device ulong*       output[[buffer(1)]],
    constant int&       ld    [[buffer(2)]],
    constant int&       pd    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int j = (int)gid.x, i = (int)gid.y;
    ulong w = 0;
    for (int b = 0; b < 64; b++) {
        int k = j * 64 + b;
        if (k < ld && input[i * ld + k] > 0.f) w |= (1ul << b);
    }
    output[i * pd + j] = w;
}

// unpack_bits: packed integer -> float +1/-1
kernel void unpack_u8(
    device const uint8_t* input [[buffer(0)]],
    device float*         output[[buffer(1)]],
    constant int&         ll    [[buffer(2)]],
    constant int&         pl    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int j = (int)gid.x, i = (int)gid.y;
    uint8_t w = input[i * pl + j];
    for (int b = 0; b < 8; b++) {
        int k = j * 8 + b;
        if (k < ll) output[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
    }
}
kernel void unpack_u32(
    device const uint* input [[buffer(0)]],
    device float*      output[[buffer(1)]],
    constant int&      ll    [[buffer(2)]],
    constant int&      pl    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int j = (int)gid.x, i = (int)gid.y;
    uint w = input[i * pl + j];
    for (int b = 0; b < 32; b++) {
        int k = j * 32 + b;
        if (k < ll) output[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
    }
}
kernel void unpack_u64(
    device const ulong* input [[buffer(0)]],
    device float*       output[[buffer(1)]],
    constant int&       ll    [[buffer(2)]],
    constant int&       pl    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int j = (int)gid.x, i = (int)gid.y;
    ulong w = input[i * pl + j];
    for (int b = 0; b < 64; b++) {
        int k = j * 64 + b;
        if (k < ll) output[i * ll + k] = ((w >> b) & 1) ? 1.f : -1.f;
    }
}

// xnor_popcount_matmul
// A(M,Kp) x B(N,Kp) -> C(M,N) int32, output = 2*popcount(~(A^B)) - K
kernel void xnor_u8(
    device const uint8_t* A  [[buffer(0)]],
    device const uint8_t* B  [[buffer(1)]],
    device int*           C  [[buffer(2)]],
    constant int&         N  [[buffer(3)]],
    constant int&         Kp [[buffer(4)]],
    constant int&         K  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int n = (int)gid.x, m = (int)gid.y;
    int acc = 0;
    for (int k = 0; k < Kp; k++)
        acc += popcount((uint)(uint8_t)(~(A[m*Kp+k] ^ B[n*Kp+k])));
    C[m*N+n] = 2*acc - K;
}
kernel void xnor_u32(
    device const uint* A  [[buffer(0)]],
    device const uint* B  [[buffer(1)]],
    device int*        C  [[buffer(2)]],
    constant int&      N  [[buffer(3)]],
    constant int&      Kp [[buffer(4)]],
    constant int&      K  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int n = (int)gid.x, m = (int)gid.y;
    int acc = 0;
    for (int k = 0; k < Kp; k++)
        acc += popcount(~(A[m*Kp+k] ^ B[n*Kp+k]));
    C[m*N+n] = 2*acc - K;
}
kernel void xnor_u64(
    device const ulong* A  [[buffer(0)]],
    device const ulong* B  [[buffer(1)]],
    device int*         C  [[buffer(2)]],
    constant int&       N  [[buffer(3)]],
    constant int&       Kp [[buffer(4)]],
    constant int&       K  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int n = (int)gid.x, m = (int)gid.y;
    int acc = 0;
    for (int k = 0; k < Kp; k++)
        acc += popcount(~(A[m*Kp+k] ^ B[n*Kp+k]));
    C[m*N+n] = 2*acc - K;
}

// per-element popcount
kernel void popcnt_u8 (device const uint8_t* i[[buffer(0)]], device int* o[[buffer(1)]], uint g[[thread_position_in_grid]]) { o[g] = popcount((uint)i[g]); }
kernel void popcnt_u32(device const uint*    i[[buffer(0)]], device int* o[[buffer(1)]], uint g[[thread_position_in_grid]]) { o[g] = popcount(i[g]); }
kernel void popcnt_u64(device const ulong*   i[[buffer(0)]], device int* o[[buffer(1)]], uint g[[thread_position_in_grid]]) { o[g] = popcount(i[g]); }

// hamming distance
kernel void hamming_u8 (device const uint8_t* a[[buffer(0)]], device const uint8_t* b[[buffer(1)]], device int* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = popcount((uint)(uint8_t)(a[g]^b[g])); }
kernel void hamming_u32(device const uint*    a[[buffer(0)]], device const uint*    b[[buffer(1)]], device int* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = popcount(a[g]^b[g]); }
kernel void hamming_u64(device const ulong*   a[[buffer(0)]], device const ulong*   b[[buffer(1)]], device int* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = popcount(a[g]^b[g]); }

// bitwise ops
kernel void bw_and_u8 (device const uint8_t* a[[buffer(0)]], device const uint8_t* b[[buffer(1)]], device uint8_t* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] & b[g]; }
kernel void bw_and_u32(device const uint*    a[[buffer(0)]], device const uint*    b[[buffer(1)]], device uint*    c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] & b[g]; }
kernel void bw_and_u64(device const ulong*   a[[buffer(0)]], device const ulong*   b[[buffer(1)]], device ulong*   c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] & b[g]; }

kernel void bw_or_u8  (device const uint8_t* a[[buffer(0)]], device const uint8_t* b[[buffer(1)]], device uint8_t* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] | b[g]; }
kernel void bw_or_u32 (device const uint*    a[[buffer(0)]], device const uint*    b[[buffer(1)]], device uint*    c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] | b[g]; }
kernel void bw_or_u64 (device const ulong*   a[[buffer(0)]], device const ulong*   b[[buffer(1)]], device ulong*   c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] | b[g]; }

kernel void bw_xor_u8 (device const uint8_t* a[[buffer(0)]], device const uint8_t* b[[buffer(1)]], device uint8_t* c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] ^ b[g]; }
kernel void bw_xor_u32(device const uint*    a[[buffer(0)]], device const uint*    b[[buffer(1)]], device uint*    c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] ^ b[g]; }
kernel void bw_xor_u64(device const ulong*   a[[buffer(0)]], device const ulong*   b[[buffer(1)]], device ulong*   c[[buffer(2)]], uint g[[thread_position_in_grid]]) { c[g] = a[g] ^ b[g]; }

kernel void bw_not_u8 (device const uint8_t* a[[buffer(0)]], device uint8_t* b[[buffer(1)]], uint g[[thread_position_in_grid]]) { b[g] = ~a[g]; }
kernel void bw_not_u32(device const uint*    a[[buffer(0)]], device uint*    b[[buffer(1)]], uint g[[thread_position_in_grid]]) { b[g] = ~a[g]; }
kernel void bw_not_u64(device const ulong*   a[[buffer(0)]], device ulong*   b[[buffer(1)]], uint g[[thread_position_in_grid]]) { b[g] = ~a[g]; }
)MSL";

// ── Pipeline cache ────────────────────────────────────────
struct PipelineCache {
    std::mutex mtx;
    id<MTLDevice>      dev  = nil;
    id<MTLLibrary>     lib  = nil;
    id<MTLCommandQueue> own_queue = nil;  // used when MPS stream is unavailable
    std::map<std::string, id<MTLComputePipelineState>> states;

    void ensure_lib() {
        if (lib) return;
        dev = MTLCreateSystemDefaultDevice();
        own_queue = [dev newCommandQueue];
        NSError* err = nil;
        lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kMSL]
                                options:nil error:&err];
        TORCH_CHECK(lib, "Metal shader compile error: ",
                    [[err localizedDescription] UTF8String]);
    }

    id<MTLComputePipelineState> get(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = states.find(name);
        if (it != states.end()) return it->second;
        ensure_lib();
        NSError* err = nil;
        auto fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        TORCH_CHECK(fn, "Metal kernel not found: ", name);
        auto ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        TORCH_CHECK(ps, "Metal pipeline error: ", [[err localizedDescription] UTF8String]);
        states[name] = ps;
        return ps;
    }

    // Return a command buffer integrated with PyTorch's MPS stream when
    // available; otherwise fall back to our own queue.
    id<MTLCommandBuffer> cmd_buf() {
#if BRUTE_MPS_STREAM
        // Encode into PyTorch's current MPS command buffer; it handles commit.
        return (id<MTLCommandBuffer>)at::mps::getCurrentMPSStream()->commandBuffer();
#else
        std::lock_guard<std::mutex> lock(mtx);
        ensure_lib();
        return [own_queue commandBuffer];
#endif
    }

    void commit_if_standalone(id<MTLCommandBuffer> buf) {
#if !BRUTE_MPS_STREAM
        [buf commit];
        [buf waitUntilCompleted];
#endif
    }
};

static PipelineCache g_cache;

// ── Helpers ───────────────────────────────────────────────

// Get the MTLBuffer backing an MPS tensor.
static id<MTLBuffer> mtl_buf(const at::Tensor& t) {
    return (__bridge id<MTLBuffer>)(t.storage().data());
}

static const char* pw_suffix(int64_t pw) {
    if (pw == 8)  return "_u8";
    if (pw == 32) return "_u32";
    return "_u64";
}

static const char* type_suffix(const at::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kByte: return "_u8";
        case at::kInt:  return "_u32";
        default:        return "_u64";
    }
}

// Dispatch a kernel with a 2-D grid (width x height) and up to 6 buffer args
// followed by int32 constant args passed via setBytes.
static void dispatch2d(const std::string& name,
                       std::vector<std::pair<id<MTLBuffer>, NSUInteger>> bufs,
                       std::vector<int32_t> consts,
                       int64_t width, int64_t height) {
    auto ps     = g_cache.get(name);
    auto cmd    = g_cache.cmd_buf();
    auto enc    = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [enc setComputePipelineState:ps];
    for (NSUInteger i = 0; i < (NSUInteger)bufs.size(); i++)
        [enc setBuffer:bufs[i].first offset:bufs[i].second atIndex:i];
    NSUInteger base = bufs.size();
    for (NSUInteger i = 0; i < (NSUInteger)consts.size(); i++) {
        int32_t v = consts[i];
        [enc setBytes:&v length:sizeof(v) atIndex:base + i];
    }
    NSUInteger tw = ps.threadExecutionWidth;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [enc endEncoding];
    g_cache.commit_if_standalone(cmd);
}

static void dispatch1d(const std::string& name,
                       std::vector<std::pair<id<MTLBuffer>, NSUInteger>> bufs,
                       int64_t n) {
    auto ps  = g_cache.get(name);
    auto cmd = g_cache.cmd_buf();
    auto enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [enc setComputePipelineState:ps];
    for (NSUInteger i = 0; i < (NSUInteger)bufs.size(); i++)
        [enc setBuffer:bufs[i].first offset:bufs[i].second atIndex:i];
    NSUInteger tw = ps.threadExecutionWidth;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [enc endEncoding];
    g_cache.commit_if_standalone(cmd);
}

// ── Op implementations ────────────────────────────────────

at::Tensor pack_bits(const at::Tensor& input, int64_t pw) {
    TORCH_CHECK(input.is_mps(), "brute::mps::pack_bits expects MPS tensor");
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64);
    auto inp      = input.contiguous().to(at::kFloat);
    int64_t ld    = inp.size(-1);
    int64_t pd    = (ld + pw - 1) / pw;
    int64_t batch = inp.numel() / ld;
    at::ScalarType st = (pw == 8) ? at::kByte : (pw == 32) ? at::kInt : at::kLong;
    auto out_shape = inp.sizes().vec(); out_shape.back() = pd;
    auto output = at::zeros(out_shape, inp.options().dtype(st));
    NSUInteger off_in  = (NSUInteger)(inp.storage_offset() * inp.element_size());
    NSUInteger off_out = (NSUInteger)(output.storage_offset() * output.element_size());
    dispatch2d(std::string("pack") + pw_suffix(pw),
               {{mtl_buf(inp), off_in}, {mtl_buf(output), off_out}},
               {(int32_t)ld, (int32_t)pd},
               pd, batch);
    return output;
}

at::Tensor unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw) {
    TORCH_CHECK(packed.is_mps());
    TORCH_CHECK(pw == 8 || pw == 32 || pw == 64);
    auto p        = packed.contiguous();
    int64_t ll    = logical_shape.back();
    int64_t pl    = p.size(-1);
    int64_t batch = p.numel() / pl;
    auto output   = at::empty(logical_shape.vec(), p.options().dtype(at::kFloat));
    NSUInteger off_p   = (NSUInteger)(p.storage_offset() * p.element_size());
    NSUInteger off_out = 0;
    dispatch2d(std::string("unpack") + pw_suffix(pw),
               {{mtl_buf(p), off_p}, {mtl_buf(output), off_out}},
               {(int32_t)ll, (int32_t)pl},
               pl, batch);
    return output;
}

at::Tensor xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K, int64_t pw) {
    TORCH_CHECK(A.is_mps() && B.is_mps());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    TORCH_CHECK(Kp == B.size(1));
    auto Ac = A.contiguous(), Bc = B.contiguous();
    auto C  = at::zeros({M, N}, A.options().dtype(at::kInt));
    int32_t K_eff = (int32_t)(2LL * Kp * pw - K);
    dispatch2d(std::string("xnor") + pw_suffix(pw),
               {{mtl_buf(Ac), 0}, {mtl_buf(Bc), 0}, {mtl_buf(C), 0}},
               {(int32_t)N, (int32_t)Kp, K_eff},
               N, M);
    return C;
}

at::Tensor popcount(const at::Tensor& packed) {
    auto p   = packed.contiguous();
    auto out = at::empty(p.sizes(), p.options().dtype(at::kInt));
    dispatch1d(std::string("popcnt") + type_suffix(p),
               {{mtl_buf(p), 0}, {mtl_buf(out), 0}},
               p.numel());
    return out;
}

static at::Tensor binary_op(const std::string& prefix, const at::Tensor& A, const at::Tensor& B) {
    auto Ac = A.contiguous(), Bc = B.contiguous();
    auto C  = at::empty_like(Ac);
    dispatch1d(prefix + type_suffix(Ac),
               {{mtl_buf(Ac), 0}, {mtl_buf(Bc), 0}, {mtl_buf(C), 0}},
               Ac.numel());
    return C;
}

at::Tensor hamming_distance(const at::Tensor& A, const at::Tensor& B) { return binary_op("hamming", A, B); }
at::Tensor bitwise_and(const at::Tensor& A, const at::Tensor& B)      { return binary_op("bw_and", A, B); }
at::Tensor bitwise_or (const at::Tensor& A, const at::Tensor& B)      { return binary_op("bw_or",  A, B); }
at::Tensor bitwise_xor(const at::Tensor& A, const at::Tensor& B)      { return binary_op("bw_xor", A, B); }

at::Tensor bitwise_not(const at::Tensor& A) {
    auto Ac = A.contiguous();
    auto B  = at::empty_like(Ac);
    dispatch1d(std::string("bw_not") + type_suffix(Ac),
               {{mtl_buf(Ac), 0}, {mtl_buf(B), 0}},
               Ac.numel());
    return B;
}

at::Tensor& randomize_bits(at::Tensor& out) {
    out.random_();
    return out;
}

// packed_popcount — total 1-bit count using libpopcnt via CPU round-trip.
// Not on the matmul hot path; the CPU→MPS transfer is acceptable here.
at::Tensor packed_popcount(const at::Tensor& packed) {
    auto p_cpu = packed.cpu().contiguous();
    uint64_t total = popcnt(p_cpu.data_ptr(), static_cast<uint64_t>(p_cpu.nbytes()));
    return at::scalar_tensor((int64_t)total, at::kLong);
}

}} // cbrute::mps
#endif // HAVE_MPS
