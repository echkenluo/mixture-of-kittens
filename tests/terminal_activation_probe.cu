#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "../csrc/sm90_fp8_block_pipeline_primitives.cuh"

namespace {

namespace pipeline = mok_sm90::fp8_block_pipeline;

constexpr int kIntermediate = pipeline::V4_INTERMEDIATE;
constexpr int kGateUp = pipeline::V4_GATE_UP;
constexpr int kGroup = pipeline::V4_FP8_GROUP;
constexpr int kReferenceThreads = pipeline::V4_ACTIVATION_WORKERS;
constexpr int kClusterThreads = kReferenceThreads / 2;

__global__ __launch_bounds__(kReferenceThreads, 1)
void reference_kernel(const __nv_bfloat16 *input, uint8_t *output,
                      float *output_scale, int rows, float limit) {
    const int row = blockIdx.x;
    if (row < rows)
        pipeline::activate_quant_worker(
            input, output, output_scale, row, threadIdx.x, limit);
}

// Terminal mapping: one cluster owns one M64.  The two 128-thread CTAs form
// the exact 256-thread contiguous activation row and loop the 64 rows without
// a per-row global task or cluster barrier.  Their column ranges and scale
// groups are disjoint, so only the terminal stage boundary needs cluster sync.
__cluster_dims__(2, 1, 1) __launch_bounds__(kClusterThreads, 1)
__global__ void cluster_kernel(const __nv_bfloat16 *input, uint8_t *output,
                               float *output_scale, int rows, float limit) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    const int worker = cta_rank * kClusterThreads + threadIdx.x;
    const int first_row = cluster * 64;
#pragma unroll 1
    for (int row = first_row; row < first_row + 64 && row < rows; ++row)
        pipeline::activate_quant_worker(
            input, output, output_scale, row, worker, limit);
}

void check_tensors(const at::Tensor &input, const at::Tensor &output,
                   const at::Tensor &output_scale) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16
                    && input.is_contiguous() && input.dim() == 2
                    && input.size(0) > 0 && input.size(0) % 64 == 0
                    && input.size(1) == kGateUp,
                "input must be contiguous CUDA BF16 [M64,4096]");
    TORCH_CHECK(output.is_cuda()
                    && output.scalar_type() == at::kFloat8_e4m3fn
                    && output.is_contiguous()
                    && output.size(0) == input.size(0)
                    && output.size(1) == kIntermediate,
                "output must be contiguous CUDA E4M3 [M,2048]");
    TORCH_CHECK(output_scale.is_cuda()
                    && output_scale.scalar_type() == at::kFloat
                    && output_scale.is_contiguous()
                    && output_scale.size(0) == input.size(0)
                    && output_scale.size(1) == kIntermediate / kGroup,
                "output_scale must be contiguous CUDA FP32 [M,16]");
    TORCH_CHECK(input.device() == output.device()
                    && input.device() == output_scale.device(),
                "all tensors must share one CUDA device");
}

void run(const at::Tensor &input, const at::Tensor &output,
         const at::Tensor &output_scale, bool clustered, double limit) {
    check_tensors(input, output, output_scale);
    c10::cuda::CUDAGuard guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
    const int rows = static_cast<int>(input.size(0));
    const auto *in = reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr());
    auto *out = reinterpret_cast<uint8_t *>(output.data_ptr());
    auto *scale = output_scale.data_ptr<float>();
    if (clustered) {
        cluster_kernel<<<(rows / 64) * 2, kClusterThreads, 0, stream>>>(
            in, out, scale, rows, static_cast<float>(limit));
    } else {
        reference_kernel<<<rows, kReferenceThreads, 0, stream>>>(
            in, out, scale, rows, static_cast<float>(limit));
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal activation probe launch failed");
}

std::vector<int64_t> attributes() {
    cudaFuncAttributes reference{};
    cudaFuncAttributes clustered{};
    TORCH_CHECK(cudaFuncGetAttributes(&reference, reference_kernel)
                    == cudaSuccess,
                "reference attribute query failed");
    TORCH_CHECK(cudaFuncGetAttributes(&clustered, cluster_kernel)
                    == cudaSuccess,
                "cluster attribute query failed");
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(kClusterThreads, 1, 1);
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int occupancy = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(
                    &occupancy, cluster_kernel, &config) == cudaSuccess,
                "cluster occupancy query failed");
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
