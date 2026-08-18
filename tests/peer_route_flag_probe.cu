#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <climits>
#include <cstdint>
#include <vector>

#include "../csrc/sm90_fp8_block_terminal_route_flags.cuh"

namespace {

namespace terminal = mok_sm90::fp8_block_terminal_route_flags;

constexpr int kTerminalWorld = 4;
constexpr int kTerminalRoutes = terminal::TOPK;
constexpr int kTerminalThreads = 128;

struct terminal_peer_globals {
    uint8_t *combine_peer[kTerminalWorld];
    unsigned int *route_ready_peer[kTerminalWorld];
    const uint8_t *routed_y;
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *num_tokens;
    int ep_size;
    int num_local_tokens;
    int hidden_size;
    int topk;
    int schedule_capacity;
};

__device__ __forceinline__ uint16_t pattern(int source_rank, int iteration,
                                             int route, int column) {
    uint32_t value = static_cast<uint32_t>(source_rank + 1) * 0x9e3779b9u;
    value ^= static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu;
    value ^= static_cast<uint32_t>(route + 1) * 0xc2b2ae35u;
    value ^= static_cast<uint32_t>(column + 1) * 0x27d4eb2du;
    value ^= value >> 16;
    return static_cast<uint16_t>(value | 1u);
}

__global__ __launch_bounds__(kTerminalThreads, 2)
void terminal_peer_route_kernel(
    terminal_peer_globals g, const int *topk_ids,
    unsigned int *epilogue_claim, unsigned int *mismatch_count,
    unsigned int *claim_count, unsigned int *timeout_count,
    int source_rank, int expected_source_rank, int iteration,
    int delay_source_rank, unsigned long long delay_cycles,
    unsigned long long spin_limit) {
    if (blockIdx.x == 0) {
        if (threadIdx.x >= 32)
            return;
        const int lane = threadIdx.x;
        const int rows = g.num_tokens[0] < g.schedule_capacity
            ? g.num_tokens[0]
            : g.schedule_capacity;
        auto *source = reinterpret_cast<uint16_t *>(
            const_cast<uint8_t *>(g.routed_y));
        for (int row = 0; row < rows; ++row) {
            const int route = g.schedule_peer_token_idx[row];
            if (route >= 0 && route < kTerminalRoutes) {
                for (int column = lane; column < g.hidden_size;
                     column += 32) {
                    source[static_cast<size_t>(row) * g.hidden_size + column]
                        = pattern(source_rank, iteration, route, column);
                }
            }
            __syncwarp(0xffffffffu);
            if (source_rank == delay_source_rank && row == 0
                && lane == 0 && delay_cycles != 0ull) {
                const unsigned long long start = clock64();
                while (static_cast<unsigned long long>(clock64()) - start
                       < delay_cycles) { }
            }
            __syncwarp(0xffffffffu);
            terminal::push_routed_row_and_publish(g, row, lane);
            __syncwarp(0xffffffffu);
        }
        return;
    }

    __shared__ int owned;
    if (threadIdx.x == 0) {
        owned = 0;
        for (unsigned long long spin = 0; spin < spin_limit; ++spin) {
            const terminal::claim_result result =
                terminal::try_claim_ready_token(
                    g.route_ready_peer[source_rank], epilogue_claim, 0, 1);
            if (result == terminal::claim_result::claimed) {
                owned = 1;
                atomicAdd(claim_count, 1u);
                break;
            }
            if (result == terminal::claim_result::already_claimed)
                break;
            __nanosleep(128);
        }
        if (!owned && atomicAdd(epilogue_claim, 0u) == terminal::CLAIM_FREE)
            atomicAdd(timeout_count, 1u);
    }
    __syncthreads();
    if (!owned)
        return;

    const auto *combine = reinterpret_cast<const uint16_t *>(
        g.combine_peer[source_rank]);
    for (int route = 0; route < kTerminalRoutes; ++route) {
        if (topk_ids[route] < 0)
            continue;
        for (int column = threadIdx.x; column < g.hidden_size;
             column += blockDim.x) {
            const uint16_t observed =
                combine[static_cast<size_t>(route) * g.hidden_size + column];
            const uint16_t expected = pattern(
                expected_source_rank, iteration, route, column);
            if (observed != expected)
                atomicAdd(mismatch_count, 1u);
        }
    }
}

__device__ __forceinline__ void store_release(unsigned int *address,
                                               unsigned int value) {
    asm volatile("{st.release.sys.global.u32 [%0], %1;}" ::
                 "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned int load_acquire(
    const unsigned int *address) {
    unsigned int value;
    asm volatile("{ld.acquire.sys.global.u32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ bool wait_phase(const unsigned int *address,
                                            unsigned int phase,
                                            unsigned long long spin_limit) {
    for (unsigned long long iteration = 0; iteration < spin_limit;
         ++iteration) {
        if (load_acquire(address) >= phase) return true;
        __nanosleep(128);
    }
    return false;
}

__global__ void peer_route_flag_kernel(
    uint16_t *local_payload, unsigned int *local_flags,
    unsigned int *local_ack, uint16_t *peer_payload,
    unsigned int *peer_flags, unsigned int *peer_ack,
    unsigned int *mismatch_count, unsigned int *timeout_count,
    int source_rank, int expected_source_rank, int routes, int columns,
    int iterations, int omit_source_rank, int omit_route,
    unsigned long long spin_limit) {
    __shared__ unsigned int iteration_failed;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const unsigned int phase = static_cast<unsigned int>(iteration + 1);
        const int elements = routes * columns;
        for (int index = threadIdx.x; index < elements;
             index += blockDim.x) {
            const int route = index / columns;
            const int column = index - route * columns;
            peer_payload[index] =
                pattern(source_rank, iteration, route, column);
        }

        // Every writer fences its own stores.  The block barrier then lets
        // thread 0 publish per-route flags only after all row stores are
        // system-visible.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int route = 0; route < routes; ++route) {
                if (source_rank == omit_source_rank && route == omit_route)
                    continue;
                store_release(peer_flags + route, phase);
            }
            iteration_failed = 0;
        }
        __syncthreads();

        // Each thread acquires the flag for every route whose payload
        // elements it validates.  This is deliberately stronger than a
        // single-thread acquire followed by a block barrier.
        for (int route = threadIdx.x; route < routes;
             route += blockDim.x) {
            if (!wait_phase(local_flags + route, phase, spin_limit)) {
                atomicAdd(timeout_count, 1u);
                atomicExch(&iteration_failed, 1u);
            }
        }
        __syncthreads();
        if (iteration_failed) return;

        for (int index = threadIdx.x; index < elements;
             index += blockDim.x) {
            const int route = index / columns;
            const int column = index - route * columns;
            if (local_payload[index]
                != pattern(expected_source_rank, iteration, route, column))
                atomicAdd(mismatch_count, 1u);
        }

        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int route = 0; route < routes; ++route)
                store_release(peer_ack + route, phase);
            iteration_failed = 0;
        }
        __syncthreads();

        for (int route = threadIdx.x; route < routes;
             route += blockDim.x) {
            if (!wait_phase(local_ack + route, phase, spin_limit)) {
                atomicAdd(timeout_count, 1u);
                atomicExch(&iteration_failed, 1u);
            }
        }
        __syncthreads();
        if (iteration_failed) return;
    }
}

void run_peer_route_flag_probe(
    const at::Tensor &local_payload, const at::Tensor &local_flags,
    const at::Tensor &local_ack, int64_t peer_payload_ptr,
    int64_t peer_flags_ptr, int64_t peer_ack_ptr,
    const at::Tensor &mismatch_count, const at::Tensor &timeout_count,
    int64_t source_rank, int64_t expected_source_rank, int64_t routes,
    int64_t columns, int64_t iterations, int64_t omit_source_rank,
    int64_t omit_route, int64_t spin_limit) {
    TORCH_CHECK(local_payload.is_cuda()
                    && local_payload.scalar_type() == at::kShort
                    && local_payload.is_contiguous()
                    && local_payload.numel() == routes * columns,
                "local_payload must be contiguous CUDA int16 [routes,columns]");
    for (const at::Tensor *tensor : {&local_flags, &local_ack,
                                      &mismatch_count, &timeout_count}) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "flag and result tensors must be contiguous CUDA int32");
    }
    TORCH_CHECK(local_flags.numel() == routes && local_ack.numel() == routes,
                "local flag arrays must have one entry per route");
    TORCH_CHECK(mismatch_count.numel() == 1 && timeout_count.numel() == 1,
                "result counters must be scalar tensors");
    TORCH_CHECK(routes > 0 && columns > 0 && iterations > 0,
                "routes, columns, and iterations must be positive");
    TORCH_CHECK(peer_payload_ptr > 0 && peer_flags_ptr > 0
                    && peer_ack_ptr > 0,
                "peer pointers must be positive");

    c10::cuda::CUDAGuard guard(local_payload.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(local_payload.get_device());
    peer_route_flag_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<uint16_t *>(local_payload.data_ptr<int16_t>()),
        reinterpret_cast<unsigned int *>(local_flags.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(local_ack.data_ptr<int>()),
        reinterpret_cast<uint16_t *>(peer_payload_ptr),
        reinterpret_cast<unsigned int *>(peer_flags_ptr),
        reinterpret_cast<unsigned int *>(peer_ack_ptr),
        reinterpret_cast<unsigned int *>(mismatch_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(timeout_count.data_ptr<int>()),
        static_cast<int>(source_rank), static_cast<int>(expected_source_rank),
        static_cast<int>(routes), static_cast<int>(columns),
        static_cast<int>(iterations), static_cast<int>(omit_source_rank),
        static_cast<int>(omit_route),
        static_cast<unsigned long long>(spin_limit));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "peer route flag probe launch failed");
}

void run_terminal_peer_route_probe(
    const at::Tensor &routed_y,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &combine,
    const at::Tensor &route_ready,
    const at::Tensor &topk_ids,
    const at::Tensor &epilogue_claim,
    const at::Tensor &mismatch_count,
    const at::Tensor &claim_count,
    const at::Tensor &timeout_count,
    const std::vector<int64_t> &combine_ptrs,
    const std::vector<int64_t> &route_ready_ptrs,
    int64_t source_rank, int64_t expected_source_rank,
    int64_t iteration, int64_t delay_source_rank,
    int64_t delay_cycles, int64_t spin_limit) {
    TORCH_CHECK(routed_y.is_cuda()
                    && routed_y.scalar_type() == at::kBFloat16
                    && routed_y.is_contiguous() && routed_y.dim() == 2
                    && routed_y.size(0) == kTerminalRoutes
                    && routed_y.size(1) > 0
                    && routed_y.size(1) <= INT_MAX
                    && routed_y.size(1) % 8 == 0,
                "routed_y must be contiguous CUDA BF16 [6,H]");
    const int64_t hidden = routed_y.size(1);
    TORCH_CHECK(combine.is_cuda()
                    && combine.scalar_type() == at::kBFloat16
                    && combine.is_contiguous()
                    && combine.sizes()
                        == at::IntArrayRef({kTerminalRoutes, hidden}),
                "combine must be contiguous CUDA BF16 [6,H]");
    for (const at::Tensor *tensor : {
             &schedule_peer_rank, &schedule_peer_token_idx,
             &num_tokens, &route_ready, &topk_ids, &epilogue_claim,
             &mismatch_count, &claim_count, &timeout_count}) {
        TORCH_CHECK(tensor->is_cuda()
                        && tensor->scalar_type() == at::kInt
                        && tensor->is_contiguous(),
                    "terminal control tensors must be contiguous CUDA int32");
        TORCH_CHECK(tensor->get_device() == routed_y.get_device(),
                    "terminal tensors must share one CUDA device");
    }
    TORCH_CHECK(schedule_peer_rank.numel() == kTerminalRoutes
                    && schedule_peer_token_idx.numel() == kTerminalRoutes
                    && route_ready.numel() == kTerminalRoutes
                    && topk_ids.numel() == kTerminalRoutes,
                "terminal route tensors must contain six entries");
    TORCH_CHECK(num_tokens.numel() == 1 && epilogue_claim.numel() == 1
                    && mismatch_count.numel() == 1
                    && claim_count.numel() == 1
                    && timeout_count.numel() == 1,
                "terminal scalar controls must contain one entry");
    TORCH_CHECK(combine.get_device() == routed_y.get_device(),
                "terminal tensors must share one CUDA device");
    TORCH_CHECK(combine_ptrs.size() == kTerminalWorld
                    && route_ready_ptrs.size() == kTerminalWorld,
                "terminal peer-pointer lists must contain four ranks");
    TORCH_CHECK(source_rank >= 0 && source_rank < kTerminalWorld
                    && expected_source_rank >= -1
                    && expected_source_rank < kTerminalWorld,
                "invalid source rank");
    TORCH_CHECK(iteration >= 0 && iteration <= INT_MAX
                    && delay_cycles >= 0 && spin_limit > 0,
                "invalid iteration, delay, or spin limit");

    c10::cuda::CUDAGuard guard(routed_y.device());
    terminal_peer_globals g{};
    for (int rank = 0; rank < kTerminalWorld; ++rank) {
        TORCH_CHECK(combine_ptrs[rank] > 0 && route_ready_ptrs[rank] > 0,
                    "terminal peer pointers must be positive");
        g.combine_peer[rank] =
            reinterpret_cast<uint8_t *>(combine_ptrs[rank]);
        g.route_ready_peer[rank] =
            reinterpret_cast<unsigned int *>(route_ready_ptrs[rank]);
    }
    g.routed_y = reinterpret_cast<const uint8_t *>(routed_y.data_ptr());
    g.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.ep_size = kTerminalWorld;
    g.num_local_tokens = 1;
    g.hidden_size = static_cast<int>(hidden);
    g.topk = kTerminalRoutes;
    g.schedule_capacity = kTerminalRoutes;

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(routed_y.get_device());
    terminal_peer_route_kernel<<<2, kTerminalThreads, 0, stream>>>(
        g, topk_ids.data_ptr<int>(),
        reinterpret_cast<unsigned int *>(epilogue_claim.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(mismatch_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(claim_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(timeout_count.data_ptr<int>()),
        static_cast<int>(source_rank), static_cast<int>(expected_source_rank),
        static_cast<int>(iteration), static_cast<int>(delay_source_rank),
        static_cast<unsigned long long>(delay_cycles),
        static_cast<unsigned long long>(spin_limit));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal peer route probe launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run_peer_route_flag_probe);
    module.def("run_terminal", &run_terminal_peer_route_probe);
}
