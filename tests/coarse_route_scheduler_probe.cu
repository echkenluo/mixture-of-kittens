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
constexpr unsigned int TASK_REDUCE = 0x80000000u;
constexpr unsigned int TASK_POLL = 0xfffffffeu;
constexpr unsigned int TASK_STOP = 0xffffffffu;

__device__ __forceinline__ unsigned int load_acquire_gpu(
    const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ unsigned int load_acquire_sys(
    const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.sys.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void store_release_gpu(
    unsigned int *address, unsigned int value) {
    asm volatile("{st.release.gpu.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void store_release_sys(
    unsigned int *address, unsigned int value) {
    asm volatile("{st.release.sys.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned int cas_acq_rel_gpu(
    unsigned int *address, unsigned int expected, unsigned int desired) {
    unsigned int prior;
    asm volatile("{atom.cas.acq_rel.gpu.global.b32 %0, [%1], %2, %3;}"
                 : "=r"(prior)
                 : "l"(address), "r"(expected), "r"(desired)
                 : "memory");
    return prior;
}

__device__ __forceinline__ unsigned int claim_bounded(
    unsigned int *head, unsigned int count) {
    while (true) {
        const unsigned int current = load_acquire_gpu(head);
        if (current >= count) return TASK_STOP;
        if (cas_acq_rel_gpu(head, current, current + 1u) == current)
            return current;
    }
}

__device__ __forceinline__ unsigned int payload_pattern(
    int source_rank, unsigned int global_route) {
    unsigned int value = static_cast<unsigned int>(source_rank + 1)
                         * 0x9e3779b9u;
    value ^= (global_route + 1u) * 0x85ebca6bu;
    value ^= value >> 16;
    return value | 1u;
}

__device__ __forceinline__ int source_for_route(
    unsigned int global_route, const unsigned int *tiles_by_rank,
    int world) {
    unsigned int end = 0;
    for (int source = 0; source < world; ++source) {
        end += tiles_by_rank[source] * 64u;
        if (global_route < end) return source;
    }
    return -1;
}

__device__ __forceinline__ unsigned int try_claim_ready_token(
    const unsigned int *local_flags, unsigned int *claims,
    unsigned int *scan_cursor, int tokens, int routes) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        const unsigned int token =
            atomicAdd(scan_cursor, 1u) % static_cast<unsigned int>(tokens);
        bool ready = true;
        for (int route = 0; route < routes; ++route) {
            ready &= load_acquire_sys(
                         local_flags + token * routes + route) != 0u;
        }
        if (ready && cas_acq_rel_gpu(claims + token, 0u, 1u) == 0u)
            return TASK_REDUCE | token;
    }
    return TASK_POLL;
}

__cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
__global__ void coarse_route_scheduler_kernel(
    unsigned int *local_payload, unsigned int *local_flags,
    const unsigned long long *peer_payload_ptrs,
    const unsigned long long *peer_flag_ptrs,
    const unsigned int *tiles_by_rank, unsigned int *tile_head,
    unsigned int *scan_cursor, unsigned int *claims,
    unsigned int *reduced_count, unsigned int *timeout_count,
    unsigned int *mismatch_count, unsigned int *worker_descriptor,
    unsigned int *poll_counts, unsigned int *copy_visits,
    unsigned int *w13_visits, unsigned int *act_visits,
    unsigned int *w2_visits, unsigned int *push_visits,
    unsigned int *reduce_visits, int rank, int world, int tokens,
    int routes, int delay_rank, unsigned long long delay_cycles,
    long long omit_global_route, unsigned int poll_limit) {
    const int cluster = blockIdx.x / 2;
    const int cta_rank = blockIdx.x & 1;
    const unsigned int local_tiles = tiles_by_rank[rank];
    unsigned int local_prefix_tiles = 0;
    for (int source = 0; source < rank; ++source)
        local_prefix_tiles += tiles_by_rank[source];

    unsigned int phase = 0;
    bool prefer_reduce = false;
    cg::cluster_group cluster_group = cg::this_cluster();

    while (true) {
        if (cta_rank == 0 && threadIdx.x == 0) {
            unsigned int task = TASK_POLL;
            if (load_acquire_gpu(timeout_count) != 0u) {
                task = TASK_STOP;
            } else {
                const bool tiles_exhausted =
                    load_acquire_gpu(tile_head) >= local_tiles;
                if (prefer_reduce || tiles_exhausted) {
                    task = try_claim_ready_token(
                        local_flags, claims, scan_cursor, tokens, routes);
                    prefer_reduce = false;
                }
                if (task == TASK_POLL && !tiles_exhausted) {
                    task = claim_bounded(tile_head, local_tiles);
                }
                if (task == TASK_STOP) task = TASK_POLL;
                if (task == TASK_POLL && tiles_exhausted
                    && load_acquire_gpu(reduced_count)
                           >= static_cast<unsigned int>(tokens))
                    task = TASK_STOP;
            }
            store_release_gpu(worker_descriptor + cluster * 2 + phase,
                              task);
        }

        cluster_group.sync();
        const unsigned int task = load_acquire_gpu(
            worker_descriptor + cluster * 2 + phase);
        phase ^= 1u;

        if (task == TASK_STOP) break;
        if (task == TASK_POLL) {
            if (cta_rank == 0 && threadIdx.x == 0) {
                const unsigned int polls = atomicAdd(
                    poll_counts + cluster, 1u) + 1u;
                if (polls >= poll_limit)
                    atomicCAS(timeout_count, 0u, 1u);
                __nanosleep(128);
            }
            continue;
        }

        if ((task & TASK_REDUCE) != 0u) {
            const unsigned int token = task & ~TASK_REDUCE;
            for (int route = cta_rank * blockDim.x + threadIdx.x;
                 route < routes; route += blockDim.x * 2) {
                const unsigned int global_route =
                    (token * routes + route) * world + rank;
                const int source = source_for_route(
                    global_route, tiles_by_rank, world);
                const unsigned int ready = load_acquire_sys(
                    local_flags + token * routes + route);
                const unsigned int observed =
                    local_payload[token * routes + route];
                if (ready == 0u || source < 0
                    || observed != payload_pattern(source, global_route))
                    atomicAdd(mismatch_count, 1u);
            }
            cluster_group.sync();
            if (cta_rank == 0 && threadIdx.x == 0) {
                atomicAdd(reduce_visits + token, 1u);
                atomicAdd(reduced_count, 1u);
            }
            continue;
        }

        const unsigned int tile = task;
        if (tile >= local_tiles) {
            if (cta_rank == 0 && threadIdx.x == 0)
                atomicAdd(mismatch_count, 1u);
            continue;
        }
        if (threadIdx.x == 0) {
            atomicAdd(copy_visits + tile, 1u);
            atomicAdd(w13_visits + tile, 1u);
            atomicAdd(act_visits + tile, 1u);
            atomicAdd(w2_visits + tile, 1u);
            atomicAdd(push_visits + tile, 1u);
        }
        if (rank == delay_rank && delay_cycles != 0ull
            && cta_rank == 0 && threadIdx.x == 0 && tile % 17u == 0u) {
            const unsigned long long start = clock64();
            while (static_cast<unsigned long long>(clock64()) - start
                   < delay_cycles) { }
        }

        const int row = cta_rank * 32 + threadIdx.x;
        unsigned int global_route = 0;
        unsigned int destination = 0;
        unsigned int local_index = 0;
        if (threadIdx.x < 32) {
            global_route = (local_prefix_tiles + tile) * 64u + row;
            destination = global_route % static_cast<unsigned int>(world);
            const unsigned int quotient =
                global_route / static_cast<unsigned int>(world);
            const unsigned int token =
                quotient / static_cast<unsigned int>(routes);
            const unsigned int route =
                quotient % static_cast<unsigned int>(routes);
            local_index = token * routes + route;
            auto *peer_payload = reinterpret_cast<unsigned int *>(
                peer_payload_ptrs[destination]);
            peer_payload[local_index] = payload_pattern(rank, global_route);
            __threadfence_system();
        }
        cluster_group.sync();
        if (threadIdx.x < 32
            && static_cast<long long>(global_route) != omit_global_route) {
            auto *peer_flags = reinterpret_cast<unsigned int *>(
                peer_flag_ptrs[destination]);
            store_release_sys(peer_flags + local_index, 1u);
        }
        cluster_group.sync();
        if (cta_rank == 0 && threadIdx.x == 0) prefer_reduce = true;
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
        &clusters, coarse_route_scheduler_kernel, &config);
    TORCH_CHECK(status == cudaSuccess,
                "coarse route occupancy query failed: ",
                cudaGetErrorString(status));
    return static_cast<int64_t>(clusters);
}

std::vector<int64_t> kernel_attributes(int64_t device_index) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_index));
    cudaFuncAttributes attributes = {};
    const cudaError_t status = cudaFuncGetAttributes(
        &attributes, coarse_route_scheduler_kernel);
    TORCH_CHECK(status == cudaSuccess,
                "coarse route attribute query failed: ",
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
    const at::Tensor &local_payload, const at::Tensor &local_flags,
    const at::Tensor &peer_payload_ptrs, const at::Tensor &peer_flag_ptrs,
    const at::Tensor &tiles_by_rank, const at::Tensor &tile_head,
    const at::Tensor &scan_cursor, const at::Tensor &claims,
    const at::Tensor &reduced_count, const at::Tensor &timeout_count,
    const at::Tensor &mismatch_count, const at::Tensor &worker_descriptor,
    const at::Tensor &poll_counts, const at::Tensor &copy_visits,
    const at::Tensor &w13_visits, const at::Tensor &act_visits,
    const at::Tensor &w2_visits, const at::Tensor &push_visits,
    const at::Tensor &reduce_visits, int64_t rank, int64_t delay_rank,
    int64_t delay_cycles, int64_t omit_global_route, int64_t poll_limit) {
    const at::Tensor *int_tensors[] = {
        &local_payload, &local_flags, &tiles_by_rank, &tile_head,
        &scan_cursor, &claims, &reduced_count, &timeout_count,
        &mismatch_count, &worker_descriptor, &poll_counts, &copy_visits,
        &w13_visits, &act_visits, &w2_visits, &push_visits,
        &reduce_visits};
    for (const at::Tensor *tensor : int_tensors) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "probe state tensors must be contiguous CUDA int32");
        TORCH_CHECK(tensor->device() == local_payload.device(),
                    "probe state tensors must share one device");
    }
    for (const at::Tensor *tensor : {&peer_payload_ptrs, &peer_flag_ptrs}) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kLong
                        && tensor->is_contiguous()
                        && tensor->device() == local_payload.device(),
                    "peer pointer tables must be contiguous CUDA int64");
    }
    const int64_t world = tiles_by_rank.numel();
    TORCH_CHECK(world == 4 && rank >= 0 && rank < world,
                "probe requires exactly four valid ranks");
    TORCH_CHECK(peer_payload_ptrs.numel() == world
                    && peer_flag_ptrs.numel() == world,
                "peer pointer tables must have one entry per rank");
    const int64_t routes = 4;
    TORCH_CHECK(local_payload.numel() == local_flags.numel()
                    && local_payload.numel() % routes == 0,
                "local payload/flags must be [tokens,4]");
    const int64_t tokens = local_payload.numel() / routes;
    TORCH_CHECK(claims.numel() == tokens
                    && reduce_visits.numel() == tokens,
                "claim/reduce arrays must have one entry per token");
    TORCH_CHECK(tile_head.numel() == 1 && scan_cursor.numel() == 1
                    && reduced_count.numel() == 1
                    && timeout_count.numel() == 1
                    && mismatch_count.numel() == 1,
                "scalar state tensors must contain one int32");
    const int64_t clusters = worker_descriptor.numel() / 2;
    TORCH_CHECK(clusters > 0 && worker_descriptor.numel() == clusters * 2
                    && poll_counts.numel() == clusters,
                "descriptor/poll state must match cluster count");
    const int64_t local_tiles = copy_visits.numel();
    TORCH_CHECK(w13_visits.numel() == local_tiles
                    && act_visits.numel() == local_tiles
                    && w2_visits.numel() == local_tiles
                    && push_visits.numel() == local_tiles,
                "stage visit arrays must share local tile count");
    TORCH_CHECK(delay_cycles >= 0 && poll_limit > 0,
                "delay and poll limit must be valid");

    c10::cuda::CUDAGuard guard(local_payload.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(local_payload.get_device());
    coarse_route_scheduler_kernel<<<clusters * 2, THREADS, 0, stream>>>(
        reinterpret_cast<unsigned int *>(local_payload.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(local_flags.data_ptr<int>()),
        reinterpret_cast<const unsigned long long *>(
            peer_payload_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const unsigned long long *>(
            peer_flag_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const unsigned int *>(tiles_by_rank.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(tile_head.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(scan_cursor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(claims.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(reduced_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(timeout_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(mismatch_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(worker_descriptor.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(poll_counts.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(copy_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w13_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(act_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(w2_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(push_visits.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(reduce_visits.data_ptr<int>()),
        static_cast<int>(rank), static_cast<int>(world),
        static_cast<int>(tokens), static_cast<int>(routes),
        static_cast<int>(delay_rank),
        static_cast<unsigned long long>(delay_cycles),
        static_cast<long long>(omit_global_route),
        static_cast<unsigned int>(poll_limit));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "coarse route scheduler probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("max_active_clusters", &max_active_clusters);
    module.def("kernel_attributes", &kernel_attributes);
}
