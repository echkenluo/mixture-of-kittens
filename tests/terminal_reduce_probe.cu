#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 128;

// Numeric core of the ready-token terminal epilogue.  One cluster owns one
// token; its two CTAs form a 256-thread column group and loop over H7168.
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
    const size_t route_base = static_cast<size_t>(token) * topk;
    for (int column = worker; column < hidden; column += 2 * kThreads) {
        float accumulator = __fmul_rn(
            __bfloat162float(combine[route_base * hidden + column]),
            weights[route_base]);
        for (int route = 1; route < topk; ++route) {
            const float term = __fmul_rn(
                __bfloat162float(
                    combine[(route_base + route) * hidden + column]),
                weights[route_base + route]);
            accumulator = __fadd_rn(accumulator, term);
        }
        output[static_cast<size_t>(token) * hidden + column] =
            __float2bfloat16_rn(accumulator);
    }
}

void run(const at::Tensor &combine, const at::Tensor &weights,
         const at::Tensor &output) {
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
    reduce_kernel<<<static_cast<int>(tokens) * 2, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(combine.data_ptr()),
        weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
        static_cast<int>(tokens), static_cast<int>(topk),
        static_cast<int>(hidden));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal reduce probe launch failed");
}

std::vector<int64_t> attributes() {
    cudaFuncAttributes attributes{};
    TORCH_CHECK(cudaFuncGetAttributes(&attributes, reduce_kernel)
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
        static_cast<int64_t>(attributes.numRegs),
        static_cast<int64_t>(attributes.sharedSizeBytes),
        static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(occupancy),
    };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("attributes", &attributes);
}
