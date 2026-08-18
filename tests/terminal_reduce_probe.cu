#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ void reduce_element(
    const __nv_bfloat16 *combine, const float *weights,
    __nv_bfloat16 *output, int token, int column, int topk, int hidden) {
    const size_t route_base = static_cast<size_t>(token) * topk;
    float accumulator = __fmul_rn(
        __bfloat162float(combine[route_base * hidden + column]),
        weights[route_base]);
    for (int route = 1; route < topk; ++route) {
        // ThunderKittens' production rv_fl mul+add sequence is compiled as
        // FFMA under MoK's --use_fast_math build.  Spell that instruction
        // explicitly so exact output does not depend on optimizer fusion.
        accumulator = __fmaf_rn(
            __bfloat162float(
                combine[(route_base + route) * hidden + column]),
            weights[route_base + route], accumulator);
    }
    output[static_cast<size_t>(token) * hidden + column] =
        __float2bfloat16_rn(accumulator);
}

__global__ __launch_bounds__(2 * kThreads, 1)
void reference_kernel(const __nv_bfloat16 *combine, const float *weights,
                      __nv_bfloat16 *output, int tokens, int topk,
                      int hidden) {
    const int token = blockIdx.x;
    if (token >= tokens) return;
    for (int column = threadIdx.x; column < hidden;
         column += 2 * kThreads)
        reduce_element(
            combine, weights, output, token, column, topk, hidden);
}

// Numeric core of the ready-token terminal epilogue.  One cluster owns one
// token; its two CTAs form a 256-thread column group and loop over hidden.
// Route slots are accumulated in the same fixed order as routed_epilogue.
__cluster_dims__(2, 1, 1) __launch_bounds__(kThreads, 1)
__global__ void reduce_kernel(const __nv_bfloat16 *combine,
                              const float *weights,
                              __nv_bfloat16 *output, int tokens,
                              int topk, int hidden) {
    const int token = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    const int worker = cta_rank * kThreads + threadIdx.x;
    if (token >= tokens) return;
    for (int column = worker; column < hidden; column += 2 * kThreads)
        reduce_element(
            combine, weights, output, token, column, topk, hidden);
}

void run(const at::Tensor &combine, const at::Tensor &weights,
         const at::Tensor &output, bool clustered) {
    TORCH_CHECK(combine.is_cuda()
                    && combine.scalar_type() == at::kBFloat16
                    && combine.is_contiguous() && combine.dim() == 2,
                "combine must be contiguous CUDA BF16 [T*topk,H]");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat
                    && weights.is_contiguous() && weights.dim() == 2
                    && weights.size(0) > 0 && weights.size(1) > 0,
                "weights must be contiguous CUDA FP32 [T,topk]");
    const int64_t tokens = weights.size(0);
    const int64_t topk = weights.size(1);
    const int64_t hidden = combine.size(1);
    TORCH_CHECK(combine.size(0) == tokens * topk && hidden > 0,
                "combine shape must match T*topk and have positive H");
    TORCH_CHECK(output.is_cuda()
                    && output.scalar_type() == at::kBFloat16
                    && output.is_contiguous() && output.dim() == 2
                    && output.size(0) == tokens && output.size(1) == hidden,
                "output must be contiguous CUDA BF16 [T,H]");
    TORCH_CHECK(combine.device() == weights.device()
                    && combine.device() == output.device(),
                "all tensors must share one CUDA device");
    TORCH_CHECK(tokens <= INT32_MAX && topk <= INT32_MAX
                    && hidden <= INT32_MAX,
                "probe dimensions exceed int32");

    c10::cuda::CUDAGuard guard(output.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(output.get_device());
    const auto *combine_ptr =
        reinterpret_cast<const __nv_bfloat16 *>(combine.data_ptr());
    auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
    if (clustered) {
        reduce_kernel<<<static_cast<int>(tokens) * 2, kThreads, 0, stream>>>(
            combine_ptr, weights.data_ptr<float>(), output_ptr,
            static_cast<int>(tokens), static_cast<int>(topk),
            static_cast<int>(hidden));
    } else {
        reference_kernel<<<static_cast<int>(tokens), 2 * kThreads, 0, stream>>>(
            combine_ptr, weights.data_ptr<float>(), output_ptr,
            static_cast<int>(tokens), static_cast<int>(topk),
            static_cast<int>(hidden));
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal reduce probe launch failed");
}

std::vector<int64_t> attributes() {
    cudaFuncAttributes reference{};
    cudaFuncAttributes clustered{};
    TORCH_CHECK(cudaFuncGetAttributes(&reference, reference_kernel)
                    == cudaSuccess,
                "reference attribute query failed");
    TORCH_CHECK(cudaFuncGetAttributes(&clustered, reduce_kernel)
                    == cudaSuccess,
                "reduce attribute query failed");
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int occupancy = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(
                    &occupancy, reduce_kernel, &config) == cudaSuccess,
                "reduce occupancy query failed");
    return {
        static_cast<int64_t>(reference.numRegs),
        static_cast<int64_t>(reference.sharedSizeBytes),
        static_cast<int64_t>(reference.localSizeBytes),
        static_cast<int64_t>(clustered.numRegs),
        static_cast<int64_t>(clustered.sharedSizeBytes),
        static_cast<int64_t>(clustered.localSizeBytes),
        static_cast<int64_t>(occupancy),
    };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("attributes", &attributes);
}
