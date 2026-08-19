#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

namespace cg = cooperative_groups;

constexpr int THREADS = 128;
constexpr unsigned int TASK_NONE = ~1u;
constexpr unsigned int TASK_STOP = ~0u;

__device__ __forceinline__ unsigned int load_acquire(
    const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void store_release(unsigned int *address,
                                               unsigned int value) {
    asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned int cas_acq_rel(
    unsigned int *address, unsigned int expected, unsigned int desired) {
    unsigned int prior;
    asm volatile("{atom.cas.acq_rel.gpu.global.b32 %0, [%1], %2, %3;}"
                 : "=r"(prior)
                 : "l"(address), "r"(expected), "r"(desired)
                 : "memory");
    return prior;
}

__device__ __forceinline__ unsigned int claim_tile(
    unsigned int *head, unsigned int tiles) {
    while (true) {
        const unsigned int current = load_acquire(head);
        if (current >= tiles) return TASK_STOP;
        if (cas_acq_rel(head, current, current + 1u) == current)
            return current;
    }
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void tile_pipeline_scheduler_kernel(
    unsigned int *head, unsigned int *terminal_count,
    unsigned int *worker_descriptor, unsigned int *copy_visits,
    unsigned int *w13_visits, unsigned int *act_visits,
    unsigned int *w2_visits, unsigned int *push_visits,
    unsigned int *mismatch, int m_tiles,
    unsigned long long delay_cycles) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    unsigned int phase = 0;
    unsigned int previous_task = TASK_NONE;
    cg::cluster_group cluster_group = cg::this_cluster();

    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            const unsigned int next = claim_tile(
                head, static_cast<unsigned int>(m_tiles));
            store_release(worker_descriptor + cluster * 2 + phase, next);
        }

        // The one boundary closes the whole previous M64 pipeline and
        // publishes the next tile descriptor to both CTAs.
        cluster_group.sync();
        if (previous_task != TASK_NONE && threadIdx.x == 0) {
            atomicAdd(copy_visits + previous_task, 1u);
            atomicAdd(w13_visits + previous_task, 1u);
            atomicAdd(act_visits + previous_task, 1u);
            atomicAdd(w2_visits + previous_task, 1u);
            atomicAdd(push_visits + previous_task, 1u);
            if (cta_rank == 0)
                atomicAdd(terminal_count, 1u);
        }

        const unsigned int task = load_acquire(
            worker_descriptor + cluster * 2 + phase);
        if (task == TASK_STOP) break;
        if (task >= static_cast<unsigned int>(m_tiles)) {
            if (threadIdx.x == 0) atomicAdd(mismatch, 1u);
            break;
        }

        // Delay one CTA for selected tiles.  The following cluster boundary
        // must absorb the skew without duplicating or losing the tile.
        if (delay_cycles != 0ull && cta_rank == 0 && threadIdx.x == 0
            && task % 17u == 0u) {
            const unsigned long long start = clock64();
            while (static_cast<unsigned long long>(clock64()) - start
                   < delay_cycles) { }
        }
        previous_task = task;
        phase ^= 1u;
    }
}

int64_t max_active_clusters(int64_t device_index) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(THREADS, 1, 1);
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int clusters = 0;
    const cudaError_t status = cudaOccupancyMaxActiveClusters(
        &clusters, tile_pipeline_scheduler_kernel, &config);
    TORCH_CHECK(status == cudaSuccess,
                "tile scheduler occupancy query failed: ",
                cudaGetErrorString(status));
    TORCH_CHECK(clusters >= 1, "tile scheduler has zero active clusters");
    return static_cast<int64_t>(clusters);
}

std::vector<int64_t> kernel_attributes(int64_t device_index) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    cudaFuncAttributes attributes = {};
    const cudaError_t status = cudaFuncGetAttributes(
        &attributes, tile_pipeline_scheduler_kernel);
    TORCH_CHECK(status == cudaSuccess,
                "tile scheduler attribute query failed: ",
                cudaGetErrorString(status));
    return {
        static_cast<int64_t>(attributes.numRegs),
        static_cast<int64_t>(attributes.sharedSizeBytes),
        static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.maxThreadsPerBlock),
        static_cast<int64_t>(attributes.binaryVersion),
        static_cast<int64_t>(attributes.ptxVersion),
    };
}

void run(const at::Tensor &head, const at::Tensor &terminal_count,
         const at::Tensor &worker_descriptor,
         const at::Tensor &copy_visits, const at::Tensor &w13_visits,
         const at::Tensor &act_visits, const at::Tensor &w2_visits,
         const at::Tensor &push_visits, const at::Tensor &mismatch,
         int64_t delay_cycles) {
    const at::Tensor *tensors[] = {
        &head, &terminal_count, &worker_descriptor, &copy_visits,
        &w13_visits, &act_visits, &w2_visits, &push_visits, &mismatch};
    for (const at::Tensor *tensor : tensors) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "all tile scheduler tensors must be contiguous CUDA int32");
        TORCH_CHECK(tensor->device() == head.device(),
                    "all tile scheduler tensors must share one device");
    }
    TORCH_CHECK(head.numel() == 1 && terminal_count.numel() == 1
                    && mismatch.numel() == 1,
                "scalar state must be int32 [1]");
    TORCH_CHECK(worker_descriptor.numel() > 0
                    && worker_descriptor.numel() % 2 == 0,
                "worker_descriptor needs two slots per cluster");
    const int64_t m_tiles = copy_visits.numel();
    TORCH_CHECK(w13_visits.numel() == m_tiles
                    && act_visits.numel() == m_tiles
                    && w2_visits.numel() == m_tiles
                    && push_visits.numel() == m_tiles,
                "all stage visit arrays must share the M-tile count");
    TORCH_CHECK(m_tiles <= static_cast<int64_t>(~1u),
                "M-tile count exceeds descriptor encoding");
    TORCH_CHECK(delay_cycles >= 0, "delay_cycles must be nonnegative");

    const int clusters = static_cast<int>(worker_descriptor.numel() / 2);
    c10::cuda::CUDAGuard guard(head.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(head.get_device());
    tile_pipeline_scheduler_kernel<<<clusters * 2, THREADS, 0, stream>>>(
        reinterpret_cast<unsigned int *>(head.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(terminal_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(worker_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(copy_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w13_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(act_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w2_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(push_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(mismatch.data_ptr<int>()),
        static_cast<int>(m_tiles),
        static_cast<unsigned long long>(delay_cycles));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "tile pipeline scheduler probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("max_active_clusters", &max_active_clusters);
    module.def("kernel_attributes", &kernel_attributes);
}
