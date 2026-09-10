// Test-only SM90 arithmetic probe. Does not replace the production extension.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "pyutils/torchutils.cuh"

#if !defined(KITTENS_SM90)
#error "warprole_promotion_probe requires SM90"
#endif

namespace {
using namespace kittens;
using a_st = st_fp8e4m3<64, 128>;
using b_st = st_fp8e4m3<128, 128>;
using acc_rt = rt_fl<16, 128>;
using a_gl = gl<fp8e4m3, 1, 1, -1, -1>;
using b_gl = gl<fp8e4m3, 1, -1, -1, -1>;
using p_gl = gl<float, 1, -1, -1, -1>;
struct globals { a_gl a; b_gl b; p_gl partials; };

// Exactly the production M64 N128 K128 WGMMA shape, materialized before
// block scaling. Synchronous generic loads isolate arithmetic from the ring.
__global__ __launch_bounds__(128) void partial_kernel(
        const __grid_constant__ globals g) {
    extern __shared__ int raw[];
    shared_allocator allocator(raw);
    auto &a = allocator.allocate<a_st>();
    auto &b = allocator.allocate<b_st>();
    for (int kb = 0; kb < 32; ++kb) {
        warpgroup::load(a, g.a, {int(blockIdx.x), kb});
        warpgroup::load(b, g.b, {0, int(blockIdx.y), kb});
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        acc_rt partial;
        warpgroup::mm_ABt(partial, a, b);
        warpgroup::mma_async_wait<0>();
        warpgroup::store(g.partials, partial,
                         {0, kb, int(blockIdx.x), int(blockIdx.y)});
        __syncthreads();
    }
}

// Explicit intrinsics prevent compiler contraction in the separate arm.
// Both arms start with the same rounded first product as the production core.
__global__ void promote_kernel(const float *p, const float *sx,
                               const float *sw, float *out) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 128 * 256) return;
    const int row = index / 256, nblock = (index % 256) / 128;
    float scale = __fmul_rn(sx[row * 32], sw[nblock * 32]);
    float separate = __fmul_rn(p[index], scale), fused = separate;
    for (int kb = 1; kb < 32; ++kb) {
        scale = __fmul_rn(sx[row * 32 + kb], sw[nblock * 32 + kb]);
        const float partial = p[kb * 128 * 256 + index];
        separate = __fadd_rn(separate, __fmul_rn(partial, scale));
        fused = __fmaf_rn(partial, scale, fused);
    }
    out[index] = separate;
    out[128 * 256 + index] = fused;
}

std::vector<at::Tensor> run(const at::Tensor &x, const at::Tensor &w,
                           const at::Tensor &sx, const at::Tensor &sw) {
    auto check = [](const at::Tensor &t, at::ScalarType type,
                    at::IntArrayRef shape) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                    t.scalar_type() == type && t.sizes() == shape,
                    "incorrect probe tensor device, dtype, contiguity or shape");
    };
    check(x, at::kFloat8_e4m3fn, {128, 4096});
    check(w, at::kFloat8_e4m3fn, {1, 256, 4096});
    check(sx, at::kFloat, {128, 32});
    check(sw, at::kFloat, {1, 2, 32});
    kittens::py::device_check(x, w, sx, sw);
    c10::cuda::CUDAGuard guard(x.device());
    cudaDeviceProp prop{};
    CUDACHECK(cudaGetDeviceProperties(&prop, x.get_device()));
    TORCH_CHECK(prop.major == 9 && prop.minor == 0, "SM90 required");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    cudaStreamCaptureStatus capture;
    CUDACHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                "diagnostic probe must run outside graph capture");
    auto p = at::full({32, 128, 256}, NAN, x.options().dtype(at::kFloat));
    auto out = at::full({2, 128, 256}, NAN, x.options().dtype(at::kFloat));
    globals g{kittens::py::tensor_to_gl<a_gl>(x),
              kittens::py::tensor_to_gl<b_gl>(w),
              kittens::py::tensor_to_gl<p_gl>(p)};
    constexpr int smem = sizeof(a_st) + sizeof(b_st) + 1024;
    partial_kernel<<<dim3(2, 2), 128, smem, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    promote_kernel<<<128, 256, 0, stream>>>(p.data_ptr<float>(),
        sx.data_ptr<float>(), sw.data_ptr<float>(), out.data_ptr<float>());
    CUDACHECK(cudaGetLastError());
    return {p, out};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
