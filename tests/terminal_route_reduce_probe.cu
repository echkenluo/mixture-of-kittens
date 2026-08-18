#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "../csrc/sm90_fp8_block_terminal_route_flags.cuh"

namespace {

namespace terminal = mok_sm90::fp8_block_terminal_route_flags;

constexpr int kTopk = terminal::TOPK;
constexpr int kTokens = 4;
constexpr int kScheduleRows = 17;
constexpr int kThreads = 128;
constexpr int kReducerBlocks = 2;
constexpr int kProbeBlocks = 1 + kReducerBlocks;

void check_int_tensor(
    const at::Tensor &tensor, int64_t elements, const char *name);

__global__ void terminal_claim_race_kernel(
    const unsigned int *route_ready, unsigned int *round_claim,
    unsigned int *round_arrivals, unsigned int *claimed_count,
    unsigned int *already_claimed_count, unsigned int *unexpected_count,
    unsigned int *timeout_count, int rounds,
    unsigned long long spin_limit) {
    if (blockIdx.x >= 2 || threadIdx.x != 0)
        return;
    for (int round = 0; round < rounds; ++round) {
        // Both contenders must complete all six acquire loads before either
        // is allowed to perform the ownership CAS for this round.
        if (!terminal::all_routes_ready_once(route_ready, 0)) {
            atomicAdd(unexpected_count + round, 1u);
            return;
        }
        atomicAdd(round_arrivals + round, 1u);
        bool gate_open = false;
        for (unsigned long long spin = 0; spin < spin_limit; ++spin) {
            if (atomicAdd(round_arrivals + round, 0u) == 2u) {
                gate_open = true;
                break;
            }
            __nanosleep(64);
        }
        if (!gate_open) {
            atomicAdd(timeout_count, 1u);
            return;
        }
        const terminal::claim_result result =
            terminal::claim_token_after_ready(round_claim + round, 0);
        if (result == terminal::claim_result::claimed)
            atomicAdd(claimed_count + round, 1u);
        else if (result == terminal::claim_result::already_claimed)
            atomicAdd(already_claimed_count + round, 1u);
        else
            atomicAdd(unexpected_count + round, 1u);
    }
}

struct probe_globals {
    uint8_t *combine_peer[1];
    unsigned int *route_ready_peer[1];
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

__device__ __forceinline__ bool bounded_wait_at_least(
    const unsigned int *address, unsigned int expected,
    unsigned long long spin_limit) {
    for (unsigned long long spin = 0; spin < spin_limit; ++spin) {
        if (terminal::load_acquire_system(address) >= expected)
            return true;
        __nanosleep(64);
    }
    return false;
}

__device__ __forceinline__ void publish_row(
    const probe_globals &g, int row, int lane) {
    terminal::push_routed_row_and_publish(g, row, lane);
    // The production primitive only needs convergence before lane zero's
    // release store.  The second convergence is probe instrumentation: it
    // prevents the producer from advancing its arrival epoch first.
    __syncwarp(0xffffffffu);
}

__device__ void producer_role(
    const probe_globals &g, unsigned int *epilogue_claim,
    unsigned int *arrival_epoch, unsigned int *observed_epoch,
    unsigned int *reduced_signal, unsigned int *early_claim,
    unsigned int *overlap_count, unsigned int *timeout_count,
    unsigned long long spin_limit) {
    const int lane = threadIdx.x & 31;
    constexpr int token0_order[kTopk] = {5, 1, 3, 0, 4, 2};

    // Token zero arrives in a deliberately scrambled route order.  After each
    // of the first five publications, reducer block one must complete a
    // non-blocking probe while the token remains unclaimed.
#pragma unroll
    for (int step = 0; step < kTopk; ++step) {
        publish_row(g, token0_order[step], lane);
        const unsigned int epoch = static_cast<unsigned int>(step + 1);
        if (lane == 0)
            terminal::store_release_system(arrival_epoch, epoch);
        int wait_ok = 1;
        if (lane == 0) {
            wait_ok = bounded_wait_at_least(
                observed_epoch, epoch, spin_limit) ? 1 : 0;
            if (!wait_ok)
                atomicAdd(timeout_count, 1u);
            if (step + 1 < kTopk
                && atomicAdd(epilogue_claim, 0u) != terminal::CLAIM_FREE)
                atomicAdd(early_claim, 1u);
        }
        wait_ok = __shfl_sync(0xffffffffu, wait_ok, 0);
        if (!wait_ok)
            return;
    }

    // Token one has one invalid/pre-ready slot (3).  Leave its last valid
    // route outstanding until token zero has been reduced.  The remaining
    // push work after that observation is the probe's reducer/push overlap
    // witness; there is no rank-wide completion barrier.
    publish_row(g, 6, lane);   // token 1, slot 0
    publish_row(g, 7, lane);   // token 1, slot 1
    publish_row(g, 8, lane);   // token 1, slot 2
    publish_row(g, 9, lane);   // token 1, slot 4
    int reduce_observed = 1;
    if (lane == 0) {
        reduce_observed = bounded_wait_at_least(
            reduced_signal, 1u, spin_limit) ? 1 : 0;
        if (reduce_observed)
            atomicAdd(overlap_count, 1u);
        else
            atomicAdd(timeout_count, 1u);
    }
    reduce_observed = __shfl_sync(
        0xffffffffu, reduce_observed, 0);
    if (!reduce_observed)
        return;
    publish_row(g, 10, lane);  // token 1, slot 5

    // Token two is all-invalid and therefore has no source rows.  Token three
    // is an all-valid, all-zero payload row set.
#pragma unroll
    for (int row = 11; row < kScheduleRows; ++row)
        publish_row(g, row, lane);
}

__device__ void reducer_role(
    const probe_globals &g, const float *weights, const int *topk_ids,
    __nv_bfloat16 *output, unsigned int *epilogue_claim,
    unsigned int *reduce_count, unsigned int *not_ready_count,
    unsigned int *ready_snapshot, unsigned int *total_done,
    unsigned int *arrival_epoch, unsigned int *observed_epoch,
    unsigned int *reduced_signal, unsigned int *timeout_count,
    unsigned long long spin_limit) {
    __shared__ int selected_token;
    __shared__ int selected_result;
    __shared__ int all_done;
    const int reducer = static_cast<int>(blockIdx.x) - 1;
    const auto *combine =
        reinterpret_cast<const __nv_bfloat16 *>(g.combine_peer[0]);

    for (unsigned long long iteration = 0; iteration < spin_limit;
         ++iteration) {
        if (threadIdx.x == 0) {
            const int token = static_cast<int>(
                (iteration + static_cast<unsigned long long>(reducer))
                % kTokens);
            const unsigned int epoch = token == 0
                ? terminal::load_acquire_system(arrival_epoch)
                : 0u;
            const terminal::claim_result result =
                terminal::try_claim_ready_token(
                    g.route_ready_peer[0], epilogue_claim,
                    token, kTokens);
            selected_token = token;
            selected_result = static_cast<int>(result);
            if (result == terminal::claim_result::not_ready)
                atomicAdd(not_ready_count + token, 1u);
            // One designated helper acknowledges an epoch only after it has
            // executed the full six-flag acquire probe for token zero.
            if (token == 0 && reducer == 0 && epoch != 0)
                terminal::store_release_system(observed_epoch, epoch);
        }
        __syncthreads();

        if (selected_result
            == static_cast<int>(terminal::claim_result::claimed)) {
            terminal::reduce_claimed_token(
                combine, weights, topk_ids, output,
                selected_token, g.hidden_size,
                threadIdx.x, blockDim.x);
            // Probe-only completion publication follows the same writer
            // discipline as route rows: every output writer fences before
            // thread zero releases reduced_signal.
            terminal::release_fence_system();
        }
        __syncthreads();

        if (threadIdx.x == 0
            && selected_result
                == static_cast<int>(terminal::claim_result::claimed)) {
            unsigned int snapshot = 0u;
#pragma unroll
            for (int route = 0; route < kTopk; ++route) {
                const unsigned int observed =
                    terminal::load_acquire_system(
                        g.route_ready_peer[0]
                        + selected_token * kTopk + route);
                if (observed == terminal::ROUTE_READY)
                    snapshot |= 1u << route;
            }
            ready_snapshot[selected_token] = snapshot;
            atomicAdd(reduce_count + selected_token, 1u);
            atomicAdd(total_done, 1u);
            terminal::store_release_system(
                reduced_signal + selected_token, 1u);
        }
        __syncthreads();

        if (threadIdx.x == 0)
            all_done = atomicAdd(total_done, 0u) == kTokens ? 1 : 0;
        __syncthreads();
        if (all_done)
            return;
        if (selected_result
            != static_cast<int>(terminal::claim_result::claimed))
            __nanosleep(64);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(timeout_count, 1u);
}

__global__ __launch_bounds__(kThreads, 3)
void terminal_route_probe_kernel(
    probe_globals g, const float *weights, const int *topk_ids,
    __nv_bfloat16 *output, unsigned int *epilogue_claim,
    unsigned int *reduce_count, unsigned int *not_ready_count,
    unsigned int *early_claim, unsigned int *ready_snapshot,
    unsigned int *total_done, unsigned int *arrival_epoch,
    unsigned int *observed_epoch, unsigned int *reduced_signal,
    unsigned int *overlap_count, unsigned int *timeout_count,
    unsigned long long spin_limit) {
    if (blockIdx.x == 0) {
        if (threadIdx.x < 32) {
            producer_role(
                g, epilogue_claim, arrival_epoch, observed_epoch,
                reduced_signal, early_claim, overlap_count,
                timeout_count, spin_limit);
        }
        return;
    }
    reducer_role(
        g, weights, topk_ids, output, epilogue_claim,
        reduce_count, not_ready_count, ready_snapshot, total_done,
        arrival_epoch, observed_epoch, reduced_signal, timeout_count,
        spin_limit);
}

// Independent arithmetic oracle.  It explicitly skips invalid routes, keeps
// valid slots in their original order, and uses the same required rn mul/FMA
// instruction sequence without calling the terminal helper.
__global__ void reference_kernel(
    const __nv_bfloat16 *combine, const float *weights,
    const int *topk_ids, __nv_bfloat16 *output, int hidden) {
    const int token = static_cast<int>(blockIdx.x);
    for (int column = threadIdx.x; column < hidden;
         column += blockDim.x) {
        const size_t route_base = static_cast<size_t>(token) * kTopk;
        float accumulator = 0.0f;
        bool initialized = false;
#pragma unroll
        for (int route = 0; route < kTopk; ++route) {
            const size_t route_index = route_base + route;
            if (topk_ids[route_index] < 0)
                continue;
            const float value = __bfloat162float(
                combine[route_index * hidden + column]);
            if (!initialized) {
                accumulator = __fmul_rn(value, weights[route_index]);
                initialized = true;
            } else {
                accumulator = __fmaf_rn(
                    value, weights[route_index], accumulator);
            }
        }
        const __nv_bfloat16 reduced = initialized
            ? __float2bfloat16_rn(accumulator)
            : __float2bfloat16_rn(0.0f);
        output[static_cast<size_t>(token) * hidden + column] = reduced;
    }
}

// Resource-only candidate kernel.  The full route probe deliberately carries
// producer/reducer orchestration and bounded-wait instrumentation, so its
// stack frame is not a valid proxy for the inlined terminal reducer.  Keep a
// minimal candidate entry point to make local-memory regressions in the
// production arithmetic helper directly observable through cudaFuncAttributes.
__global__ void terminal_reduce_resource_kernel(
    const __nv_bfloat16 *combine, const float *weights,
    const int *topk_ids, __nv_bfloat16 *output, int hidden) {
    const int token = static_cast<int>(blockIdx.x);
    terminal::reduce_claimed_token(
        combine, weights, topk_ids, output, token, hidden,
        threadIdx.x, blockDim.x);
}

void run_claim_race(
    const at::Tensor &route_ready, const at::Tensor &round_claim,
    const at::Tensor &round_arrivals, const at::Tensor &claimed_count,
    const at::Tensor &already_claimed_count,
    const at::Tensor &unexpected_count, const at::Tensor &timeout_count,
    int64_t rounds, int64_t spin_limit) {
    TORCH_CHECK(rounds >= 1000 && rounds <= INT_MAX,
                "claim race requires at least 1000 rounds");
    check_int_tensor(route_ready, kTopk, "race route_ready");
    for (const at::Tensor *tensor : {
             &round_claim, &round_arrivals, &claimed_count,
             &already_claimed_count, &unexpected_count}) {
        check_int_tensor(*tensor, rounds, "race round tensor");
        TORCH_CHECK(tensor->get_device() == route_ready.get_device(),
                    "race tensors must share one CUDA device");
    }
    check_int_tensor(timeout_count, 1, "race timeout_count");
    TORCH_CHECK(timeout_count.get_device() == route_ready.get_device(),
                "race tensors must share one CUDA device");
    TORCH_CHECK(spin_limit > 0, "spin_limit must be positive");

    c10::cuda::CUDAGuard guard(route_ready.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(route_ready.get_device());
    terminal_claim_race_kernel<<<2, 32, 0, stream>>>(
        reinterpret_cast<const unsigned int *>(route_ready.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(round_claim.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(round_arrivals.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(claimed_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(
            already_claimed_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(unexpected_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(timeout_count.data_ptr<int>()),
        static_cast<int>(rounds),
        static_cast<unsigned long long>(spin_limit));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal claim race launch failed");
}

std::vector<int64_t> kernel_attributes() {
    cudaFuncAttributes route{};
    cudaFuncAttributes reduce{};
    cudaFuncAttributes race{};
    TORCH_CHECK(cudaFuncGetAttributes(&route, terminal_route_probe_kernel)
                    == cudaSuccess,
                "failed to read route kernel attributes");
    TORCH_CHECK(cudaFuncGetAttributes(
                    &reduce, terminal_reduce_resource_kernel) == cudaSuccess,
                "failed to read reducer resource kernel attributes");
    TORCH_CHECK(cudaFuncGetAttributes(&race, terminal_claim_race_kernel)
                    == cudaSuccess,
                "failed to read race kernel attributes");
    return {
        route.numRegs, static_cast<int64_t>(route.localSizeBytes),
        reduce.numRegs, static_cast<int64_t>(reduce.localSizeBytes),
        race.numRegs, static_cast<int64_t>(race.localSizeBytes),
    };
}

void check_int_tensor(
    const at::Tensor &tensor, int64_t elements, const char *name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kInt
                    && tensor.is_contiguous() && tensor.numel() == elements,
                name, " must be contiguous CUDA int32 with ", elements,
                " elements");
}

void run_probe(
    const at::Tensor &routed_y,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &combine,
    const at::Tensor &route_ready,
    const at::Tensor &topk_ids,
    const at::Tensor &weights,
    const at::Tensor &output,
    const at::Tensor &reference,
    const at::Tensor &epilogue_claim,
    const at::Tensor &reduce_count,
    const at::Tensor &not_ready_count,
    const at::Tensor &early_claim,
    const at::Tensor &ready_snapshot,
    const at::Tensor &total_done,
    const at::Tensor &arrival_epoch,
    const at::Tensor &observed_epoch,
    const at::Tensor &reduced_signal,
    const at::Tensor &overlap_count,
    const at::Tensor &timeout_count,
    int64_t spin_limit) {
    TORCH_CHECK(routed_y.is_cuda()
                    && routed_y.scalar_type() == at::kBFloat16
                    && routed_y.is_contiguous() && routed_y.dim() == 2
                    && routed_y.size(0) == kScheduleRows
                    && routed_y.size(1) > 0
                    && routed_y.size(1) <= INT_MAX,
                "routed_y must be contiguous CUDA BF16 [17,H]");
    const int64_t hidden = routed_y.size(1);
    TORCH_CHECK(combine.is_cuda()
                    && combine.scalar_type() == at::kBFloat16
                    && combine.is_contiguous()
                    && combine.sizes()
                        == at::IntArrayRef({kTokens * kTopk, hidden}),
                "combine must be contiguous CUDA BF16 [24,H]");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat
                    && weights.is_contiguous()
                    && weights.sizes()
                        == at::IntArrayRef({kTokens, kTopk}),
                "weights must be contiguous CUDA FP32 [4,6]");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16
                    && output.is_contiguous()
                    && output.sizes()
                        == at::IntArrayRef({kTokens, hidden})
                    && reference.sizes() == output.sizes()
                    && reference.scalar_type() == at::kBFloat16
                    && reference.is_contiguous(),
                "output/reference must be contiguous CUDA BF16 [4,H]");

    check_int_tensor(schedule_peer_rank, kScheduleRows,
                     "schedule_peer_rank");
    check_int_tensor(schedule_peer_token_idx, kScheduleRows,
                     "schedule_peer_token_idx");
    check_int_tensor(num_tokens, 1, "num_tokens");
    check_int_tensor(route_ready, kTokens * kTopk, "route_ready");
    check_int_tensor(topk_ids, kTokens * kTopk, "topk_ids");
    check_int_tensor(epilogue_claim, kTokens, "epilogue_claim");
    check_int_tensor(reduce_count, kTokens, "reduce_count");
    check_int_tensor(not_ready_count, kTokens, "not_ready_count");
    check_int_tensor(early_claim, 1, "early_claim");
    check_int_tensor(ready_snapshot, kTokens, "ready_snapshot");
    check_int_tensor(total_done, 1, "total_done");
    check_int_tensor(arrival_epoch, 1, "arrival_epoch");
    check_int_tensor(observed_epoch, 1, "observed_epoch");
    check_int_tensor(reduced_signal, kTokens, "reduced_signal");
    check_int_tensor(overlap_count, 1, "overlap_count");
    check_int_tensor(timeout_count, 1, "timeout_count");
    TORCH_CHECK(spin_limit > 0, "spin_limit must be positive");

    const int device = routed_y.get_device();
    for (const at::Tensor *tensor : {
             &schedule_peer_rank, &schedule_peer_token_idx, &num_tokens,
             &combine, &route_ready, &topk_ids, &weights, &output,
             &reference, &epilogue_claim, &reduce_count, &not_ready_count,
             &early_claim, &ready_snapshot, &total_done, &arrival_epoch,
             &observed_epoch, &reduced_signal, &overlap_count,
             &timeout_count}) {
        TORCH_CHECK(tensor->get_device() == device,
                    "all tensors must share one CUDA device");
    }

    c10::cuda::CUDAGuard guard(routed_y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
    probe_globals g{};
    g.combine_peer[0] = reinterpret_cast<uint8_t *>(combine.data_ptr());
    g.route_ready_peer[0] =
        reinterpret_cast<unsigned int *>(route_ready.data_ptr<int>());
    g.routed_y = reinterpret_cast<const uint8_t *>(routed_y.data_ptr());
    g.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.ep_size = 1;
    g.num_local_tokens = kTokens;
    g.hidden_size = static_cast<int>(hidden);
    g.topk = kTopk;
    g.schedule_capacity = kScheduleRows;

    terminal_route_probe_kernel<<<kProbeBlocks, kThreads, 0, stream>>>(
        g, weights.data_ptr<float>(), topk_ids.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
        reinterpret_cast<unsigned int *>(epilogue_claim.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(reduce_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(not_ready_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(early_claim.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(ready_snapshot.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(total_done.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(arrival_epoch.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(observed_epoch.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(reduced_signal.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(overlap_count.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(timeout_count.data_ptr<int>()),
        static_cast<unsigned long long>(spin_limit));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal route probe launch failed");

    reference_kernel<<<kTokens, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(combine.data_ptr()),
        weights.data_ptr<float>(), topk_ids.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16 *>(reference.data_ptr()),
        static_cast<int>(hidden));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "terminal route reference launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_probe", &run_probe);
    module.def("run_claim_race", &run_claim_race);
    module.def("kernel_attributes", &kernel_attributes);
}
