#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

namespace cg = cooperative_groups;

constexpr int THREADS = 128;

__device__ __forceinline__ uint32_t do_work(uint32_t accumulator,
                                            uint32_t task,
                                            int work_iterations) {
    accumulator += task + 1u;
    for (int iteration = 0; iteration < work_iterations; ++iteration)
        accumulator = accumulator * 1664525u + 1013904223u + task;
    return accumulator;
}

// Phase-1 safe protocol: one cluster sync publishes the descriptor and a
// second cluster sync closes the task before the next descriptor is written.
__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void double_barrier_kernel(unsigned int *descriptor,
                                      unsigned int *output, int tasks,
                                      int work_iterations) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    unsigned int *cluster_descriptor = descriptor + cluster * 2;
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t accumulator = static_cast<uint32_t>(output_index + 1);
    cg::cluster_group cluster_group = cg::this_cluster();

    for (int task = 0; task < tasks; ++task) {
        if (cta_rank == 0 && threadIdx.x == 0)
            cluster_descriptor[task & 1] = static_cast<unsigned int>(task);
        cluster_group.sync();
        const uint32_t current = cluster_descriptor[task & 1];
        accumulator = do_work(accumulator, current, work_iterations);
        cluster_group.sync();
    }
    output[output_index] = accumulator;
}

// Terminal candidate: the next iteration's publication sync is also the
// previous task's drain boundary.  The first descriptor needs one opening
// sync; every later task adds only one boundary sync.
__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void single_boundary_kernel(unsigned int *descriptor,
                                       unsigned int *output, int tasks,
                                       int work_iterations) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    unsigned int *cluster_descriptor = descriptor + cluster * 2;
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t accumulator = static_cast<uint32_t>(output_index + 1);
    cg::cluster_group cluster_group = cg::this_cluster();

    if (cta_rank == 0 && threadIdx.x == 0) cluster_descriptor[0] = 0u;
    cluster_group.sync();
    for (int task = 0; task < tasks; ++task) {
        // Ping-pong slots are required: rank 0 may finish the current work
        // before the peer CTA has loaded its descriptor.  Publishing into
        // the other phase slot avoids a read-after-write race without a
        // second cluster barrier.
        const uint32_t current = cluster_descriptor[task & 1];
        accumulator = do_work(accumulator, current, work_iterations);
        if (cta_rank == 0 && threadIdx.x == 0)
            cluster_descriptor[(task + 1) & 1] =
                static_cast<unsigned int>(task + 1);
        // This closes the current task and publishes the next descriptor.
        cluster_group.sync();
    }
    output[output_index] = accumulator;
}

void run_cluster_boundary_probe(const at::Tensor &descriptor,
                                const at::Tensor &output, int64_t tasks,
                                int64_t work_iterations,
                                bool single_boundary) {
    TORCH_CHECK(descriptor.is_cuda() && descriptor.scalar_type() == at::kInt
                    && descriptor.is_contiguous(),
                "descriptor must be contiguous CUDA int32");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kInt
                    && output.is_contiguous(),
                "output must be contiguous CUDA int32");
    TORCH_CHECK(tasks > 0 && tasks <= (1 << 20),
                "tasks must be in [1, 2^20]");
    TORCH_CHECK(work_iterations >= 0 && work_iterations <= (1 << 20),
                "work_iterations must be in [0, 2^20]");
    TORCH_CHECK(descriptor.numel() % 2 == 0,
                "descriptor must contain two phase slots per cluster");
    const int clusters = static_cast<int>(descriptor.numel() / 2);
    TORCH_CHECK(clusters > 0 && output.numel() == clusters * 2 * THREADS,
                "output must contain (descriptor.numel()/2) * 2 * 128 "
                "int32 values");
    TORCH_CHECK(descriptor.device() == output.device(),
                "descriptor and output must share one device");

    c10::cuda::CUDAGuard guard(output.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(output.get_device());
    const dim3 grid(clusters * 2, 1, 1);
    const dim3 block(THREADS, 1, 1);
    if (single_boundary)
        single_boundary_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<unsigned int *>(descriptor.data_ptr<int>()),
            reinterpret_cast<unsigned int *>(output.data_ptr<int>()),
            static_cast<int>(tasks), static_cast<int>(work_iterations));
    else
        double_barrier_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<unsigned int *>(descriptor.data_ptr<int>()),
            reinterpret_cast<unsigned int *>(output.data_ptr<int>()),
            static_cast<int>(tasks), static_cast<int>(work_iterations));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "cluster boundary probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run_cluster_boundary_probe);
}
