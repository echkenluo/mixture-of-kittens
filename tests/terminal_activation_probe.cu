#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kIntermediate = 2048;
constexpr int kGateUp = 2 * kIntermediate;
constexpr int kGroup = 128;
constexpr int kValuesPerThread = 8;
constexpr int kReferenceThreads = kIntermediate / kValuesPerThread;
constexpr int kClusterThreads = kReferenceThreads / 2;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ uint16_t pack_fp8x2(float x, float y) {
    x = fmaxf(fminf(x, kFp8Max), -kFp8Max);
    y = fmaxf(fminf(y, kFp8Max), -kFp8Max);
    uint16_t result;
    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
                 : "=h"(result) : "f"(x), "f"(y));
    return result;
}

__device__ __forceinline__ float subgroup_max_16(float value) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}

__device__ __forceinline__ void activate_row(
    const __nv_bfloat16 *input, uint8_t *output, float *output_scale,
    int row, int worker, float limit) {
    const auto *row_pairs = reinterpret_cast<const __nv_bfloat162 *>(
        input + static_cast<size_t>(row) * kGateUp);
    auto *out_pairs = reinterpret_cast<uint16_t *>(
        output + static_cast<size_t>(row) * kIntermediate);
    const int element = worker * kValuesPerThread;
    const int pair = element / 2;
    const __nv_bfloat162 limit2 = __floats2bfloat162_rn(limit, limit);
    const __nv_bfloat162 neg_limit2 = __floats2bfloat162_rn(-limit, -limit);
    float values[kValuesPerThread];
    float local_max = 0.0f;

#pragma unroll
    for (int index = 0; index < kValuesPerThread / 2; ++index) {
        __nv_bfloat162 gate = __hmin2(row_pairs[pair + index], limit2);
        __nv_bfloat162 up = __hmax2(
            row_pairs[kIntermediate / 2 + pair + index], neg_limit2);
        up = __hmin2(up, limit2);
        const float2 gate_f = __bfloat1622float2(gate);
        const float2 up_f = __bfloat1622float2(up);
        const float x = gate_f.x / (1.0f + __expf(-gate_f.x)) * up_f.x;
        const float y = gate_f.y / (1.0f + __expf(-gate_f.y)) * up_f.y;
        values[2 * index] = x;
        values[2 * index + 1] = y;
        local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
    }

    const float absmax = fmaxf(subgroup_max_16(local_max), 1e-10f);
    const float scale = absmax / kFp8Max;
    const float inv_scale = 1.0f / scale;
#pragma unroll
    for (int index = 0; index < kValuesPerThread / 2; ++index)
        out_pairs[pair + index] = pack_fp8x2(
            values[2 * index] * inv_scale,
            values[2 * index + 1] * inv_scale);
    if ((threadIdx.x & 15) == 0)
        output_scale[static_cast<size_t>(row) * (kIntermediate / kGroup)
                     + worker / 16] = scale;
}

__global__ __launch_bounds__(kReferenceThreads, 1)
void reference_kernel(const __nv_bfloat16 *input, uint8_t *output,
                      float *output_scale, int rows, float limit) {
    const int row = blockIdx.x;
    if (row < rows)
        activate_row(input, output, output_scale, row, threadIdx.x, limit);
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
        activate_row(input, output, output_scale, row, worker, limit);
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
