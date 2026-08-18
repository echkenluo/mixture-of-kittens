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
constexpr int STAGES = 6;
constexpr int QUEUED_STAGES = STAGES - 1;
constexpr unsigned int TASK_NONE = 0u;
constexpr unsigned int TASK_STOP = ~0u;
constexpr unsigned int QUEUE_EMPTY = ~0u;
constexpr unsigned int TYPE_SHIFT = 28u;
constexpr unsigned int INDEX_MASK = (1u << TYPE_SHIFT) - 1u;

enum stage_id : unsigned int {
    STAGE_DISPATCH = 0u,
    STAGE_W13 = 1u,
    STAGE_ACT = 2u,
    STAGE_W2 = 3u,
    STAGE_PUSH = 4u,
    STAGE_REDUCE = 5u,
};

struct stage_cycles {
    unsigned long long values[STAGES];
};

struct queue_view {
    unsigned int *state;       // reserve_tail, visible_tail, head
    unsigned int *descriptor;
    unsigned int *commit;
    int capacity;
};

struct probe_state {
    unsigned int *source_head;
    unsigned int *inflight;
    unsigned int *terminal_count;
    unsigned int *queue_state;
    unsigned int *queue_descriptor;
    unsigned int *queue_commit;
    unsigned int *tile_state;
    unsigned int *leader_visits;
    unsigned int *paired_visits;
    unsigned int *owner_cluster;
    unsigned long long *ready_ns;
    unsigned long long *start_ns;
    unsigned long long *end_ns;
    unsigned int *active_stage;
    unsigned int *max_active_stage;
    unsigned long long *overlap_mask;
    unsigned int *worker_descriptor;
    unsigned int *mismatch;
    unsigned int *cycle_sink;
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

__device__ __forceinline__ unsigned long long global_time_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ unsigned int encode_task(unsigned int stage,
                                                     unsigned int tile) {
    return ((stage + 1u) << TYPE_SHIFT) | tile;
}

__device__ __forceinline__ unsigned int task_stage(unsigned int task) {
    return (task >> TYPE_SHIFT) - 1u;
}

__device__ __forceinline__ unsigned int task_tile(unsigned int task) {
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

__device__ __forceinline__ queue_view get_queue(
    probe_state state, int queue, int m_tiles) {
    return queue_view{
        state.queue_state + queue * 3,
        state.queue_descriptor + queue * m_tiles,
        state.queue_commit + queue * m_tiles,
        m_tiles,
    };
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

__device__ __forceinline__ void enqueue_stage(
    probe_state state, unsigned int stage, unsigned int tile, int m_tiles) {
    if (stage == STAGE_DISPATCH || stage >= STAGES) {
        atomicAdd(state.mismatch, 1u);
        return;
    }
    queue_view queue = get_queue(
        state, static_cast<int>(stage - 1u), m_tiles);
    unsigned int position = 0u;
    if (!reserve_bounded(queue.state,
                         static_cast<unsigned int>(queue.capacity),
                         position)) {
        atomicAdd(state.mismatch, 1u);
        return;
    }
    queue.descriptor[position] = tile;
    state.ready_ns[stage * m_tiles + tile] = global_time_ns();
    store_release(queue.commit + position, 1u);
    help_visible(queue);
}

__device__ __forceinline__ unsigned int pop_queue(
    const queue_view &queue) {
    help_visible(queue);
    while (true) {
        const unsigned int visible = load_acquire(queue.state + 1);
        const unsigned int head = load_acquire(queue.state + 2);
        if (head >= visible) return QUEUE_EMPTY;
        if (cas_acq_rel(queue.state + 2, head, head + 1u) == head)
            return queue.descriptor[head];
    }
}

__device__ __forceinline__ unsigned int claim_source(
    probe_state state, int m_tiles, unsigned int window) {
    while (true) {
        const unsigned int active = load_acquire(state.inflight);
        if (active >= window) return TASK_NONE;
        if (cas_acq_rel(state.inflight, active, active + 1u) == active)
            break;
    }

    unsigned int tile = 0u;
    if (!reserve_bounded(state.source_head,
                         static_cast<unsigned int>(m_tiles), tile)) {
        atomicSub(state.inflight, 1u);
        return TASK_NONE;
    }
    state.ready_ns[tile] = global_time_ns();
    return encode_task(STAGE_DISPATCH, tile);
}

__device__ __forceinline__ unsigned int select_queued_task(
    probe_state state, int m_tiles, unsigned int window,
    int cluster, unsigned int &iteration, unsigned int &queue_cursor) {
    unsigned int task = TASK_NONE;

    // Fill a bounded live-tile window, but only on alternating scheduler
    // turns once the initial wave exists.  This prevents both an all-dispatch
    // flood and a single-tile depth-first chain.
    if (((iteration + static_cast<unsigned int>(cluster)) & 1u) == 0u)
        task = claim_source(state, m_tiles, window);

    if (task == TASK_NONE) {
        for (int offset = 0; offset < QUEUED_STAGES; ++offset) {
            const unsigned int queue =
                (queue_cursor + static_cast<unsigned int>(offset))
                % QUEUED_STAGES;
            const unsigned int tile = pop_queue(
                get_queue(state, static_cast<int>(queue), m_tiles));
            if (tile != QUEUE_EMPTY) {
                queue_cursor = (queue + 1u) % QUEUED_STAGES;
                task = encode_task(queue + 1u, tile);
                break;
            }
        }
    }

    if (task == TASK_NONE)
        task = claim_source(state, m_tiles, window);

    ++iteration;
    if (task != TASK_NONE) return task;
    if (load_acquire(state.terminal_count)
            >= static_cast<unsigned int>(m_tiles)
        && load_acquire(state.inflight) == 0u
        && load_acquire(state.source_head)
               >= static_cast<unsigned int>(m_tiles))
        return TASK_STOP;
    return TASK_NONE;
}

__device__ __forceinline__ void begin_task(
    probe_state state, unsigned int task, int m_tiles, int cluster,
    int cta_rank) {
    if (task == TASK_NONE) return;
    const unsigned int stage = task_stage(task);
    const unsigned int tile = task_tile(task);
    if (stage >= STAGES || tile >= static_cast<unsigned int>(m_tiles)) {
        if (cta_rank == 0 && threadIdx.x == 0)
            atomicAdd(state.mismatch, 1u);
        return;
    }
    const unsigned int index = stage * m_tiles + tile;
    if (threadIdx.x == 0)
        atomicAdd(state.paired_visits + index, 1u);
    if (cta_rank != 0 || threadIdx.x != 0) return;

    if (atomicAdd(state.leader_visits + index, 1u) != 0u)
        atomicAdd(state.mismatch, 1u);
    if (atomicCAS(state.owner_cluster + index, 0u,
                  static_cast<unsigned int>(cluster + 1)) != 0u)
        atomicAdd(state.mismatch, 1u);
    state.start_ns[index] = global_time_ns();
    const unsigned int active = atomicAdd(state.active_stage + stage, 1u) + 1u;
    atomicMax(state.max_active_stage + stage, active);
    __threadfence();
    for (unsigned int other = 0; other < STAGES; ++other) {
        if (other == stage
            || load_acquire(state.active_stage + other) == 0u)
            continue;
        const unsigned int bit_a = stage * STAGES + other;
        const unsigned int bit_b = other * STAGES + stage;
        atomicOr(state.overlap_mask,
                 (1ull << bit_a) | (1ull << bit_b));
    }
}

__device__ __forceinline__ void finish_task_common(
    probe_state state, unsigned int task, int m_tiles) {
    if (task == TASK_NONE) return;
    const unsigned int stage = task_stage(task);
    const unsigned int tile = task_tile(task);
    if (stage >= STAGES || tile >= static_cast<unsigned int>(m_tiles)) {
        atomicAdd(state.mismatch, 1u);
        return;
    }
    const unsigned int index = stage * m_tiles + tile;
    state.end_ns[index] = global_time_ns();
    atomicSub(state.active_stage + stage, 1u);
    if (cas_acq_rel(state.tile_state + tile, stage, stage + 1u) != stage)
        atomicAdd(state.mismatch, 1u);
}

__device__ __forceinline__ void finish_queued_task(
    probe_state state, unsigned int task, int m_tiles) {
    if (task == TASK_NONE) return;
    const unsigned int stage = task_stage(task);
    const unsigned int tile = task_tile(task);
    finish_task_common(state, task, m_tiles);
    if (stage + 1u < STAGES) {
        enqueue_stage(state, stage + 1u, tile, m_tiles);
    } else {
        atomicAdd(state.terminal_count, 1u);
        atomicSub(state.inflight, 1u);
    }
}

__device__ __forceinline__ unsigned long long task_cycles(
    const stage_cycles &cycles, unsigned int stage, unsigned int tile,
    unsigned long long skew_cycles) {
    unsigned long long value = cycles.values[stage];
    if (skew_cycles != 0ull) {
        if (stage == STAGE_W13 && tile % 17u == 0u)
            value += skew_cycles;
        if (stage == STAGE_W2 && tile % 19u == 0u)
            value += skew_cycles + skew_cycles / 2ull;
    }
    return value;
}

__device__ __forceinline__ void synthetic_work(
    unsigned long long cycles, unsigned int stage, unsigned int tile,
    unsigned int *cycle_sink) {
    if (cycles == 0ull) return;
    unsigned int value =
        (tile + 1u) * 0x9e3779b9u
        ^ (stage + 1u) * 0x85ebca6bu
        ^ static_cast<unsigned int>(threadIdx.x + 1);
    const unsigned long long start = clock64();
    do {
        value = value * 1664525u + 1013904223u;
        asm volatile("" : "+r"(value));
    } while (static_cast<unsigned long long>(clock64()) - start < cycles);
    if (threadIdx.x == 0) atomicXor(cycle_sink, value);
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void held_stage_pipeline_kernel(
    probe_state state, int m_tiles, stage_cycles cycles,
    unsigned long long skew_cycles) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    cg::cluster_group cluster_group = cg::this_cluster();
    unsigned int phase = 0u;
    unsigned int previous_task = TASK_NONE;

    while (true) {
        cluster_group.sync();
        if (cta_rank == 0 && threadIdx.x == 0) {
            unsigned int next = TASK_NONE;
            if (previous_task != TASK_NONE) {
                const unsigned int stage = task_stage(previous_task);
                const unsigned int tile = task_tile(previous_task);
                finish_task_common(state, previous_task, m_tiles);
                if (stage + 1u < STAGES) {
                    state.ready_ns[(stage + 1u) * m_tiles + tile]
                        = global_time_ns();
                    next = encode_task(stage + 1u, tile);
                } else {
                    atomicAdd(state.terminal_count, 1u);
                }
            }
            if (next == TASK_NONE) {
                unsigned int tile = 0u;
                if (reserve_bounded(state.source_head,
                                    static_cast<unsigned int>(m_tiles), tile)) {
                    state.ready_ns[tile] = global_time_ns();
                    next = encode_task(STAGE_DISPATCH, tile);
                } else if (load_acquire(state.terminal_count)
                           >= static_cast<unsigned int>(m_tiles)) {
                    next = TASK_STOP;
                }
            }
            store_release(state.worker_descriptor + cluster * 2 + phase,
                          next);
        }
        cluster_group.sync();
        const unsigned int task = load_acquire(
            state.worker_descriptor + cluster * 2 + phase);
        phase ^= 1u;
        if (task == TASK_STOP) break;
        if (task == TASK_NONE) {
            if (threadIdx.x == 0) __nanosleep(128);
            previous_task = TASK_NONE;
            continue;
        }
        begin_task(state, task, m_tiles, cluster, cta_rank);
        synthetic_work(
            task_cycles(cycles, task_stage(task), task_tile(task),
                        skew_cycles),
            task_stage(task), task_tile(task), state.cycle_sink);
        previous_task = task;
    }
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void queued_stage_pipeline_kernel(
    probe_state state, int m_tiles, unsigned int window,
    stage_cycles cycles, unsigned long long skew_cycles) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    cg::cluster_group cluster_group = cg::this_cluster();
    unsigned int phase = 0u;
    unsigned int previous_task = TASK_NONE;
    unsigned int iteration = 0u;
    unsigned int queue_cursor = static_cast<unsigned int>(cluster)
                                % QUEUED_STAGES;

    while (true) {
        cluster_group.sync();
        if (cta_rank == 0 && threadIdx.x == 0) {
            finish_queued_task(state, previous_task, m_tiles);
            const unsigned int next = select_queued_task(
                state, m_tiles, window, cluster, iteration, queue_cursor);
            store_release(state.worker_descriptor + cluster * 2 + phase,
                          next);
        }
        cluster_group.sync();
        const unsigned int task = load_acquire(
            state.worker_descriptor + cluster * 2 + phase);
        phase ^= 1u;
        if (task == TASK_STOP) break;
        if (task == TASK_NONE) {
            if (threadIdx.x == 0) __nanosleep(128);
            previous_task = TASK_NONE;
            continue;
        }
        begin_task(state, task, m_tiles, cluster, cta_rank);
        synthetic_work(
            task_cycles(cycles, task_stage(task), task_tile(task),
                        skew_cycles),
            task_stage(task), task_tile(task), state.cycle_sink);
        previous_task = task;
    }
}

int64_t max_active_clusters(int64_t device_index, int64_t mode) {
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
    cudaError_t status;
    if (mode == 0) {
        status = cudaOccupancyMaxActiveClusters(
            &clusters, held_stage_pipeline_kernel, &config);
    } else {
        TORCH_CHECK(mode == 1, "mode must be 0 (held) or 1 (queued)");
        status = cudaOccupancyMaxActiveClusters(
            &clusters, queued_stage_pipeline_kernel, &config);
    }
    TORCH_CHECK(status == cudaSuccess,
                "coarse stage occupancy query failed: ",
                cudaGetErrorString(status));
    TORCH_CHECK(clusters >= 1,
                "coarse stage probe has zero active clusters");
    return static_cast<int64_t>(clusters);
}

std::vector<int64_t> kernel_attributes(int64_t device_index, int64_t mode) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    cudaFuncAttributes attributes = {};
    cudaError_t status;
    if (mode == 0) {
        status = cudaFuncGetAttributes(
            &attributes, held_stage_pipeline_kernel);
    } else {
        TORCH_CHECK(mode == 1, "mode must be 0 (held) or 1 (queued)");
        status = cudaFuncGetAttributes(
            &attributes, queued_stage_pipeline_kernel);
    }
    TORCH_CHECK(status == cudaSuccess,
                "coarse stage attribute query failed: ",
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

void run(
    const at::Tensor &source_head, const at::Tensor &inflight,
    const at::Tensor &terminal_count, const at::Tensor &queue_state,
    const at::Tensor &queue_descriptor, const at::Tensor &queue_commit,
    const at::Tensor &tile_state, const at::Tensor &leader_visits,
    const at::Tensor &paired_visits, const at::Tensor &owner_cluster,
    const at::Tensor &ready_ns, const at::Tensor &start_ns,
    const at::Tensor &end_ns, const at::Tensor &active_stage,
    const at::Tensor &max_active_stage, const at::Tensor &overlap_mask,
    const at::Tensor &worker_descriptor, const at::Tensor &mismatch,
    const at::Tensor &cycle_sink, const std::vector<int64_t> &cycle_values,
    int64_t skew_cycles, int64_t window, int64_t mode) {
    const at::Tensor *int_tensors[] = {
        &source_head, &inflight, &terminal_count, &queue_state,
        &queue_descriptor, &queue_commit, &tile_state, &leader_visits,
        &paired_visits, &owner_cluster, &active_stage, &max_active_stage,
        &worker_descriptor, &mismatch, &cycle_sink,
    };
    for (const at::Tensor *tensor : int_tensors) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "probe int state must be contiguous CUDA int32");
        TORCH_CHECK(tensor->device() == source_head.device(),
                    "all probe tensors must share one CUDA device");
    }
    for (const at::Tensor *tensor :
         {&ready_ns, &start_ns, &end_ns, &overlap_mask}) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kLong
                        && tensor->is_contiguous(),
                    "probe timestamps/mask must be contiguous CUDA int64");
        TORCH_CHECK(tensor->device() == source_head.device(),
                    "all probe tensors must share one CUDA device");
    }
    TORCH_CHECK(source_head.numel() == 1 && inflight.numel() == 1
                    && terminal_count.numel() == 1
                    && overlap_mask.numel() == 1
                    && mismatch.numel() == 1 && cycle_sink.numel() == 1,
                "scalar probe state must contain one element");
    TORCH_CHECK(queue_state.numel() == QUEUED_STAGES * 3,
                "queue_state must be [5,3]");
    TORCH_CHECK(active_stage.numel() == STAGES
                    && max_active_stage.numel() == STAGES,
                "active stage arrays must have six entries");
    TORCH_CHECK(worker_descriptor.numel() > 0
                    && worker_descriptor.numel() % 2 == 0,
                "worker_descriptor needs two slots per cluster");
    const int64_t m_tiles = tile_state.numel();
    TORCH_CHECK(m_tiles <= INDEX_MASK,
                "M64 tile count exceeds task encoding");
    TORCH_CHECK(queue_descriptor.numel() == QUEUED_STAGES * m_tiles
                    && queue_commit.numel() == QUEUED_STAGES * m_tiles,
                "queue arrays must be [5,M]");
    const int64_t stage_items = STAGES * m_tiles;
    for (const at::Tensor *tensor :
         {&leader_visits, &paired_visits, &owner_cluster})
        TORCH_CHECK(tensor->numel() == stage_items,
                    "visit/owner arrays must be [6,M]");
    for (const at::Tensor *tensor : {&ready_ns, &start_ns, &end_ns})
        TORCH_CHECK(tensor->numel() == stage_items,
                    "timestamp arrays must be [6,M]");
    TORCH_CHECK(cycle_values.size() == STAGES,
                "cycle_values must contain six stage costs");
    TORCH_CHECK(skew_cycles >= 0 && window > 0,
                "skew cycles/window must be positive-domain values");
    TORCH_CHECK(mode == 0 || mode == 1,
                "mode must be 0 (held) or 1 (queued)");

    stage_cycles cycles = {};
    for (int stage = 0; stage < STAGES; ++stage) {
        TORCH_CHECK(cycle_values[stage] >= 0,
                    "stage cycles must be nonnegative");
        cycles.values[stage] =
            static_cast<unsigned long long>(cycle_values[stage]);
    }
    probe_state state{
        reinterpret_cast<unsigned int *>(source_head.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(inflight.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(terminal_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(queue_state.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(queue_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(queue_commit.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(tile_state.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(leader_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(paired_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(owner_cluster.data_ptr<int>()),
        reinterpret_cast<unsigned long long *>(ready_ns.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long *>(start_ns.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long *>(end_ns.data_ptr<int64_t>()),
        reinterpret_cast<unsigned int *>(active_stage.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(max_active_stage.data_ptr<int>()),
        reinterpret_cast<unsigned long long *>(
            overlap_mask.data_ptr<int64_t>()),
        reinterpret_cast<unsigned int *>(worker_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(mismatch.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(cycle_sink.data_ptr<int>()),
    };

    const int clusters = static_cast<int>(worker_descriptor.numel() / 2);
    c10::cuda::CUDAGuard guard(source_head.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(source_head.get_device());
    if (mode == 0) {
        held_stage_pipeline_kernel<<<clusters * 2, THREADS, 0, stream>>>(
            state, static_cast<int>(m_tiles), cycles,
            static_cast<unsigned long long>(skew_cycles));
    } else {
        queued_stage_pipeline_kernel<<<clusters * 2, THREADS, 0, stream>>>(
            state, static_cast<int>(m_tiles),
            static_cast<unsigned int>(window), cycles,
            static_cast<unsigned long long>(skew_cycles));
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "coarse stage pipeline probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("max_active_clusters", &max_active_clusters);
    module.def("kernel_attributes", &kernel_attributes);
}
