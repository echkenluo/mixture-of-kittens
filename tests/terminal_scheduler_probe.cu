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
constexpr unsigned int TASK_NONE = 0u;
constexpr unsigned int TASK_STOP = ~0u;
constexpr unsigned int TYPE_COPY = 1u;
constexpr unsigned int TYPE_W13 = 2u;
constexpr unsigned int TYPE_ACT = 3u;
constexpr unsigned int TYPE_W2 = 4u;
constexpr unsigned int TYPE_REDUCE = 5u;
constexpr unsigned int TYPE_SHIFT = 28u;
constexpr unsigned int INDEX_MASK = (1u << TYPE_SHIFT) - 1u;

struct queue_view {
    unsigned int *state;       // reserve_tail, visible_tail, head
    unsigned int *descriptor;
    unsigned int *commit;
    int capacity;
};

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

__device__ __forceinline__ unsigned int encode_task(unsigned int type,
                                                     unsigned int index) {
    return (type << TYPE_SHIFT) | index;
}

__device__ __forceinline__ unsigned int task_type(unsigned int task) {
    return task >> TYPE_SHIFT;
}

__device__ __forceinline__ unsigned int task_index(unsigned int task) {
    return task & INDEX_MASK;
}

__device__ __forceinline__ bool reserve_bounded(unsigned int *counter,
                                                 unsigned int capacity,
                                                 unsigned int &position) {
    while (true) {
        const unsigned int current = load_acquire(counter);
        if (current >= capacity) return false;
        if (cas_acq_rel(counter, current, current + 1u) == current) {
            position = current;
            return true;
        }
    }
}

__device__ __forceinline__ void help_visible(const queue_view &queue) {
    while (true) {
        const unsigned int current = load_acquire(queue.state + 1);
        if (current >= static_cast<unsigned int>(queue.capacity)
            || load_acquire(queue.commit + current) == 0u)
            return;
        cas_acq_rel(queue.state + 1, current, current + 1u);
    }
}

__device__ __forceinline__ void enqueue(const queue_view &queue,
                                        unsigned int task,
                                        unsigned int *mismatch) {
    unsigned int position;
    if (!reserve_bounded(queue.state, queue.capacity, position)) {
        atomicAdd(mismatch, 1u);
        return;
    }
    queue.descriptor[position] = task;
    store_release(queue.commit + position, 1u);
    help_visible(queue);
}

__device__ __forceinline__ unsigned int pop(const queue_view &queue) {
    help_visible(queue);
    while (true) {
        const unsigned int visible = load_acquire(queue.state + 1);
        const unsigned int head = load_acquire(queue.state + 2);
        if (head >= visible) return TASK_NONE;
        if (cas_acq_rel(queue.state + 2, head, head + 1u) == head)
            return queue.descriptor[head];
    }
}

__device__ __forceinline__ void enqueue_range(
    const queue_view &queue, unsigned int type, int base, int count,
    unsigned int *mismatch) {
    for (int offset = 0; offset < count; ++offset)
        enqueue(queue, encode_task(type, base + offset), mismatch);
}

__device__ __forceinline__ void complete_task(
    unsigned int task, const queue_view &w13_queue,
    const queue_view &act_queue, const queue_view &w2_queue,
    unsigned int *copy_visits, unsigned int *w13_visits,
    unsigned int *act_visits, unsigned int *w2_visits,
    unsigned int *reduce_visits, unsigned int *w13_done,
    unsigned int *act_done, unsigned int *w2_done,
    unsigned int *reduce_ready, unsigned int *terminal_count,
    unsigned int *mismatch, int m_tiles, int w13_per_m, int act_per_m,
    int w2_per_m) {
    if (task == TASK_NONE) return;
    const unsigned int type = task_type(task);
    const unsigned int index = task_index(task);
    if (type == TYPE_COPY) {
        if (index >= static_cast<unsigned int>(m_tiles)) {
            atomicAdd(mismatch, 1u);
            return;
        }
        atomicAdd(copy_visits + index, 1u);
        enqueue_range(w13_queue, TYPE_W13, index * w13_per_m,
                      w13_per_m, mismatch);
    } else if (type == TYPE_W13) {
        if (index >= static_cast<unsigned int>(m_tiles * w13_per_m)) {
            atomicAdd(mismatch, 1u);
            return;
        }
        atomicAdd(w13_visits + index, 1u);
        const int m = static_cast<int>(index) / w13_per_m;
        if (atomicAdd(w13_done + m, 1u) + 1u
            == static_cast<unsigned int>(w13_per_m))
            enqueue_range(act_queue, TYPE_ACT, m * act_per_m,
                          act_per_m, mismatch);
    } else if (type == TYPE_ACT) {
        if (index >= static_cast<unsigned int>(m_tiles * act_per_m)) {
            atomicAdd(mismatch, 1u);
            return;
        }
        atomicAdd(act_visits + index, 1u);
        const int m = static_cast<int>(index) / act_per_m;
        if (atomicAdd(act_done + m, 1u) + 1u
            == static_cast<unsigned int>(act_per_m))
            enqueue_range(w2_queue, TYPE_W2, m * w2_per_m,
                          w2_per_m, mismatch);
    } else if (type == TYPE_W2) {
        if (index >= static_cast<unsigned int>(m_tiles * w2_per_m)) {
            atomicAdd(mismatch, 1u);
            return;
        }
        atomicAdd(w2_visits + index, 1u);
        const int m = static_cast<int>(index) / w2_per_m;
        if (atomicAdd(w2_done + m, 1u) + 1u
            == static_cast<unsigned int>(w2_per_m))
            store_release(reduce_ready + m, 1u);
    } else if (type == TYPE_REDUCE) {
        if (index >= static_cast<unsigned int>(m_tiles)) {
            atomicAdd(mismatch, 1u);
            return;
        }
        atomicAdd(reduce_visits + index, 1u);
        atomicAdd(terminal_count, 1u);
    } else {
        atomicAdd(mismatch, 1u);
    }
}

__device__ __forceinline__ unsigned int claim_reduce(
    unsigned int *reduce_ready, int m_tiles, unsigned int &scan) {
    for (int attempt = 0; attempt < m_tiles; ++attempt) {
        const unsigned int m = scan++ % static_cast<unsigned int>(m_tiles);
        if (load_acquire(reduce_ready + m) == 1u
            && cas_acq_rel(reduce_ready + m, 1u, 2u) == 1u)
            return encode_task(TYPE_REDUCE, m);
    }
    return TASK_NONE;
}

__device__ __forceinline__ unsigned int select_task(
    unsigned int *copy_head, const queue_view &w13_queue,
    const queue_view &act_queue, const queue_view &w2_queue,
    unsigned int *reduce_ready, unsigned int *terminal_count,
    int m_tiles, unsigned int &reduce_scan) {
    unsigned int copy;
    if (reserve_bounded(copy_head, static_cast<unsigned int>(m_tiles), copy))
        return encode_task(TYPE_COPY, copy);
    unsigned int task = claim_reduce(reduce_ready, m_tiles, reduce_scan);
    if (task != TASK_NONE) return task;
    task = pop(w2_queue);
    if (task != TASK_NONE) return task;
    task = pop(act_queue);
    if (task != TASK_NONE) return task;
    task = pop(w13_queue);
    if (task != TASK_NONE) return task;
    if (load_acquire(terminal_count) >= static_cast<unsigned int>(m_tiles))
        return TASK_STOP;
    return TASK_NONE;
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void terminal_scheduler_kernel(
    unsigned int *copy_head, unsigned int *terminal_count,
    queue_view w13_queue, queue_view act_queue, queue_view w2_queue,
    unsigned int *reduce_ready, unsigned int *copy_visits,
    unsigned int *w13_visits, unsigned int *act_visits,
    unsigned int *w2_visits, unsigned int *reduce_visits,
    unsigned int *w13_done, unsigned int *act_done,
    unsigned int *w2_done, unsigned int *worker_descriptor,
    unsigned int *mismatch, int m_tiles, int w13_per_m, int act_per_m,
    int w2_per_m, unsigned long long delay_cycles) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    unsigned int phase = 0;
    unsigned int previous_task = TASK_NONE;
    unsigned int reduce_scan = static_cast<unsigned int>(cluster);
    cg::cluster_group cluster_group = cg::this_cluster();

    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            const unsigned int next = select_task(
                copy_head, w13_queue, act_queue, w2_queue, reduce_ready,
                terminal_count, m_tiles, reduce_scan);
            store_release(worker_descriptor + cluster * 2 + phase, next);
        }
        // One boundary proves both CTAs drained previous_task and publishes
        // the preselected descriptor.  Its successor is intentionally not
        // selectable until the following iteration.
        cluster_group.sync();
        if (cta_rank == 0 && threadIdx.x == 0)
            complete_task(
                previous_task, w13_queue, act_queue, w2_queue, copy_visits,
                w13_visits, act_visits, w2_visits, reduce_visits, w13_done,
                act_done, w2_done, reduce_ready, terminal_count, mismatch,
                m_tiles, w13_per_m, act_per_m, w2_per_m);

        const unsigned int task =
            load_acquire(worker_descriptor + cluster * 2 + phase);
        if (task == TASK_STOP) break;
        if (task != TASK_NONE && delay_cycles != 0ull
            && threadIdx.x == 0
            && (task_index(task) + task_type(task)) % 17u == 0u) {
            const unsigned long long start = clock64();
            while (static_cast<unsigned long long>(clock64()) - start
                   < delay_cycles) { }
        }
        previous_task = task;
        phase ^= 1u;
    }
}

int64_t terminal_scheduler_max_active_clusters(int64_t device_index) {
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
    int max_clusters = 0;
    const cudaError_t status = cudaOccupancyMaxActiveClusters(
        &max_clusters, terminal_scheduler_kernel, &config);
    TORCH_CHECK(status == cudaSuccess,
                "terminal scheduler occupancy query failed: ",
                cudaGetErrorString(status));
    TORCH_CHECK(max_clusters >= 1,
                "terminal scheduler has zero active cluster occupancy");
    return static_cast<int64_t>(max_clusters);
}

std::vector<int64_t> terminal_scheduler_kernel_attributes(
    int64_t device_index) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    cudaFuncAttributes attributes = {};
    const cudaError_t status = cudaFuncGetAttributes(
        &attributes, terminal_scheduler_kernel);
    TORCH_CHECK(status == cudaSuccess,
                "terminal scheduler attribute query failed: ",
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

void run_terminal_scheduler_probe(
    const at::Tensor &copy_head, const at::Tensor &terminal_count,
    const at::Tensor &w13_state, const at::Tensor &w13_descriptor,
    const at::Tensor &w13_commit, const at::Tensor &act_state,
    const at::Tensor &act_descriptor, const at::Tensor &act_commit,
    const at::Tensor &w2_state, const at::Tensor &w2_descriptor,
    const at::Tensor &w2_commit, const at::Tensor &reduce_ready,
    const at::Tensor &copy_visits, const at::Tensor &w13_visits,
    const at::Tensor &act_visits, const at::Tensor &w2_visits,
    const at::Tensor &reduce_visits, const at::Tensor &w13_done,
    const at::Tensor &act_done, const at::Tensor &w2_done,
    const at::Tensor &worker_descriptor, const at::Tensor &mismatch,
    int64_t w13_per_m, int64_t act_per_m, int64_t w2_per_m,
    int64_t delay_cycles) {
    const at::Tensor *tensors[] = {
        &copy_head, &terminal_count, &w13_state, &w13_descriptor,
        &w13_commit, &act_state, &act_descriptor, &act_commit, &w2_state,
        &w2_descriptor, &w2_commit, &reduce_ready, &copy_visits,
        &w13_visits, &act_visits, &w2_visits, &reduce_visits, &w13_done,
        &act_done, &w2_done, &worker_descriptor, &mismatch};
    for (const at::Tensor *tensor : tensors) {
        TORCH_CHECK(tensor->is_cuda() && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "all scheduler tensors must be contiguous CUDA int32");
        TORCH_CHECK(tensor->device() == copy_head.device(),
                    "all scheduler tensors must share one device");
    }
    TORCH_CHECK(copy_head.numel() == 1 && terminal_count.numel() == 1
                    && mismatch.numel() == 1,
                "scalar scheduler state must be int32 [1]");
    for (const at::Tensor *state : {&w13_state, &act_state, &w2_state})
        TORCH_CHECK(state->numel() == 3, "queue state must be int32 [3]");
    TORCH_CHECK(w13_descriptor.numel() == w13_commit.numel()
                    && act_descriptor.numel() == act_commit.numel()
                    && w2_descriptor.numel() == w2_commit.numel(),
                "each queue descriptor and commit array must match");
    const int m_tiles = static_cast<int>(copy_visits.numel());
    TORCH_CHECK(m_tiles > 0 && reduce_ready.numel() == m_tiles
                    && reduce_visits.numel() == m_tiles
                    && w13_done.numel() == m_tiles
                    && act_done.numel() == m_tiles
                    && w2_done.numel() == m_tiles,
                "per-M state must share the positive copy_visits size");
    TORCH_CHECK(w13_per_m > 0 && act_per_m > 0 && w2_per_m > 0,
                "per-M task counts must be positive");
    TORCH_CHECK(w13_descriptor.numel() == m_tiles * w13_per_m
                    && w13_visits.numel() == w13_descriptor.numel()
                    && act_descriptor.numel() == m_tiles * act_per_m
                    && act_visits.numel() == act_descriptor.numel()
                    && w2_descriptor.numel() == m_tiles * w2_per_m
                    && w2_visits.numel() == w2_descriptor.numel(),
                "queue capacities and visit arrays must match task counts");
    TORCH_CHECK(worker_descriptor.numel() > 0
                    && worker_descriptor.numel() % 2 == 0,
                "worker_descriptor needs two phase slots per cluster");
    TORCH_CHECK(delay_cycles >= 0, "delay_cycles must be nonnegative");

    const int clusters = static_cast<int>(worker_descriptor.numel() / 2);
    c10::cuda::CUDAGuard guard(copy_head.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(copy_head.get_device());
    queue_view w13{
        reinterpret_cast<unsigned int *>(w13_state.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w13_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w13_commit.data_ptr<int>()),
        static_cast<int>(w13_descriptor.numel())};
    queue_view act{
        reinterpret_cast<unsigned int *>(act_state.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(act_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(act_commit.data_ptr<int>()),
        static_cast<int>(act_descriptor.numel())};
    queue_view w2{
        reinterpret_cast<unsigned int *>(w2_state.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w2_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w2_commit.data_ptr<int>()),
        static_cast<int>(w2_descriptor.numel())};
    terminal_scheduler_kernel<<<clusters * 2, THREADS, 0, stream>>>(
        reinterpret_cast<unsigned int *>(copy_head.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(terminal_count.data_ptr<int>()),
        w13, act, w2,
        reinterpret_cast<unsigned int *>(reduce_ready.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(copy_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w13_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(act_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w2_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(reduce_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w13_done.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(act_done.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w2_done.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(worker_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(mismatch.data_ptr<int>()), m_tiles,
        static_cast<int>(w13_per_m), static_cast<int>(act_per_m),
        static_cast<int>(w2_per_m),
        static_cast<unsigned long long>(delay_cycles));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal scheduler probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run_terminal_scheduler_probe);
    module.def("max_active_clusters", &terminal_scheduler_max_active_clusters);
    module.def("kernel_attributes", &terminal_scheduler_kernel_attributes);
}
