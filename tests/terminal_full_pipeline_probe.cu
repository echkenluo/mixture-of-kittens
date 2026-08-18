#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "pyutils/torchutils.cuh"
#include "../csrc/sm90_fp8_block_terminal_full.cuh"
#include "../csrc/sm90_fp8_block_worker_test.cuh"

#if !defined(KITTENS_SM90)
#error "terminal_full_pipeline_probe requires KITTENS_SM90"
#endif

namespace {

namespace terminal = mok_sm90::fp8_block_terminal;
namespace full = mok_sm90::fp8_block_terminal_full;
namespace pipeline = mok_sm90::fp8_block_pipeline;
namespace split = mok_sm90::fp8_block_test::contiguous;

using a_gl = split::a_gl;
using b_gl = split::b_gl;
using d_gl = split::d_gl;

constexpr int kThreads = terminal::THREADS_PER_CTA;
constexpr int kActivationThreads = pipeline::V4_ACTIVATION_WORKERS;

struct gemm_problem {
    a_gl A;
    b_gl B;
    d_gl D;
    const float *A_scale;
    const float *B_scale;
    int n;
    int k_blocks;
    int n_tiles;
};

struct reference_globals {
    const uint8_t *x_peer[terminal::EP_SIZE];
    const float *x_scale_peer[terminal::EP_SIZE];
    uint8_t *routed_x;
    float *routed_x_scale;
    int *m_indices;
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *tokens_per_expert;
    const int *push_order;
    const __nv_bfloat16 *routed_y;
    __nv_bfloat16 *combine;
    const float *weights;
    const int *topk_ids;
    __nv_bfloat16 *output;
    int rows;
    int local_tokens;
    int hidden;
    int scale_columns;
    int experts;
};

__global__ void reference_dispatch_kernel(reference_globals g) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= g.rows)
        return;
    const int peer = g.schedule_peer_rank[row];
    const int slot = g.schedule_peer_token_idx[row];
    const bool valid = peer >= 0 && peer < terminal::EP_SIZE
        && slot >= 0 && slot < g.local_tokens * terminal::TOP_K;
    const int source = valid ? slot / terminal::TOP_K : -1;
    uint8_t *dst = g.routed_x + static_cast<size_t>(row) * g.hidden;
    float *dst_scale = g.routed_x_scale
        + static_cast<size_t>(row) * g.scale_columns;
    for (int column = threadIdx.x; column < g.hidden;
         column += blockDim.x) {
        dst[column] = valid
            ? g.x_peer[peer][static_cast<size_t>(source) * g.hidden + column]
            : 0u;
    }
    for (int column = threadIdx.x; column < g.scale_columns;
         column += blockDim.x) {
        dst_scale[column] = valid
            ? g.x_scale_peer[peer][
                static_cast<size_t>(source) * g.scale_columns + column]
            : 0.0f;
    }
    if (threadIdx.x == 0) {
        int offset = 0;
        int expert = 0;
        while (expert < g.experts - 1) {
            offset += g.tokens_per_expert[expert];
            if (row < offset)
                break;
            ++expert;
        }
        g.m_indices[row] = expert;
    }
}

__global__ __launch_bounds__(kActivationThreads, 1)
void reference_activation_kernel(
        const __nv_bfloat16 *gate_up, uint8_t *hidden,
        float *hidden_scale, int rows, float limit) {
    const int row = static_cast<int>(blockIdx.x);
    if (row < rows) {
        pipeline::activate_quant_worker(
            gate_up, hidden, hidden_scale, row, threadIdx.x, limit);
    }
}

__global__ void reference_push_kernel(reference_globals g) {
    const int position = static_cast<int>(blockIdx.x);
    if (position >= g.rows)
        return;
    const int row = g.push_order[position];
    const int peer = g.schedule_peer_rank[row];
    const int slot = g.schedule_peer_token_idx[row];
    if (peer < 0 || peer >= terminal::EP_SIZE
            || slot < 0 || slot >= g.local_tokens * terminal::TOP_K)
        return;
    const auto *src = g.routed_y + static_cast<size_t>(row) * g.hidden;
    auto *dst = g.combine
        + (static_cast<size_t>(peer) * g.local_tokens * terminal::TOP_K
           + slot) * g.hidden;
    for (int column = threadIdx.x; column < g.hidden;
         column += blockDim.x)
        dst[column] = src[column];
}

__global__ void reference_reduce_kernel(reference_globals g) {
    const int global_token = static_cast<int>(blockIdx.x);
    const int peer = global_token / g.local_tokens;
    const int token = global_token % g.local_tokens;
    const size_t route_base =
        (static_cast<size_t>(peer) * g.local_tokens + token)
        * terminal::TOP_K;
    for (int column = threadIdx.x; column < g.hidden;
         column += blockDim.x) {
        float accumulator = 0.0f;
        bool initialized = false;
#pragma unroll
        for (int route = 0; route < terminal::TOP_K; ++route) {
            const size_t route_index = route_base + route;
            if (g.topk_ids[route_index] < 0)
                continue;
            const float value = __bfloat162float(
                g.combine[route_index * g.hidden + column]);
            if (!initialized) {
                accumulator = __fmul_rn(value, g.weights[route_index]);
                initialized = true;
            } else {
                accumulator = __fmaf_rn(
                    value, g.weights[route_index], accumulator);
            }
        }
        __nv_bfloat16 result = initialized
            ? __float2bfloat16_rn(accumulator)
            : __float2bfloat16_rn(0.0f);
        g.output[(static_cast<size_t>(peer) * g.local_tokens + token)
                     * g.hidden + column] = result;
    }
}

constexpr int kDynamicSmem =
    full::compute::PIPE_DEPTH
        * (sizeof(full::compute::a_st) + sizeof(full::compute::b_st))
    + sizeof(full::compute::d_st) + 1024;
static_assert(kDynamicSmem == 41984,
              "sequential N256 must retain legacy shared memory");
static_assert(
    full::compute::SEQUENTIAL_N128_LIVE_ACCUMULATOR_WORDS == 64,
    "sequential N256 must retain legacy accumulator budget");

int resident_clusters() {
    CUDACHECK(cudaFuncSetAttribute(
        full::kernel<gemm_problem>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmem));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(
        (full::COMM_CLUSTERS + 1) * terminal::CLUSTER_CTAS, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = kDynamicSmem;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = terminal::CLUSTER_CTAS;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    int clusters = 0;
    CUDACHECK(cudaOccupancyMaxActiveClusters(
        &clusters, full::kernel<gemm_problem>, &config));
    TORCH_CHECK(clusters >= full::COMM_CLUSTERS + 1,
                "full terminal kernel cannot co-reside comm+compute");
    return clusters;
}

reference_globals make_reference_globals(
        const at::Tensor &peer_x, const at::Tensor &peer_scale,
        const at::Tensor &schedule_peer, const at::Tensor &schedule_slot,
        const at::Tensor &tokens_per_expert, const at::Tensor &push_order,
        const at::Tensor &routed_x, const at::Tensor &routed_x_scale,
        const at::Tensor &m_indices, const at::Tensor &routed_y,
        const at::Tensor &combine, const at::Tensor &weights,
        const at::Tensor &topk_ids, const at::Tensor &output) {
    reference_globals result{};
    const int local_tokens = static_cast<int>(peer_x.size(1));
    const int hidden = static_cast<int>(peer_x.size(2));
    const int scales = hidden / 128;
    const auto *x_base = peer_x.data_ptr<uint8_t>();
    const auto *scale_base = peer_scale.data_ptr<float>();
    for (int peer = 0; peer < terminal::EP_SIZE; ++peer) {
        result.x_peer[peer] = x_base
            + static_cast<size_t>(peer) * local_tokens * hidden;
        result.x_scale_peer[peer] = scale_base
            + static_cast<size_t>(peer) * local_tokens * scales;
    }
    result.routed_x = reinterpret_cast<uint8_t *>(routed_x.data_ptr());
    result.routed_x_scale = routed_x_scale.data_ptr<float>();
    result.m_indices = m_indices.data_ptr<int>();
    result.schedule_peer_rank = schedule_peer.data_ptr<int>();
    result.schedule_peer_token_idx = schedule_slot.data_ptr<int>();
    result.tokens_per_expert = tokens_per_expert.data_ptr<int>();
    result.push_order = push_order.data_ptr<int>();
    result.routed_y = reinterpret_cast<const __nv_bfloat16 *>(
        routed_y.data_ptr());
    result.combine = reinterpret_cast<__nv_bfloat16 *>(combine.data_ptr());
    result.weights = weights.data_ptr<float>();
    result.topk_ids = topk_ids.data_ptr<int>();
    result.output = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
    result.rows = static_cast<int>(routed_x.size(0));
    result.local_tokens = local_tokens;
    result.hidden = hidden;
    result.scale_columns = scales;
    result.experts = static_cast<int>(tokens_per_expert.numel());
    return result;
}

void run_split(
        const at::Tensor &peer_x, const at::Tensor &peer_scale,
        const at::Tensor &w13, const at::Tensor &w13_scale,
        const at::Tensor &w2, const at::Tensor &w2_scale,
        const at::Tensor &schedule_peer, const at::Tensor &schedule_slot,
        const at::Tensor &tokens_per_expert, const at::Tensor &push_order,
        const at::Tensor &weights, const at::Tensor &topk_ids,
        const at::Tensor &routed_x, const at::Tensor &routed_x_scale,
        const at::Tensor &m_indices, const at::Tensor &gate_up,
        const at::Tensor &hidden, const at::Tensor &hidden_scale,
        const at::Tensor &routed_y, const at::Tensor &combine,
        const at::Tensor &output, double limit) {
    c10::cuda::CUDAGuard guard(peer_x.device());
    const reference_globals g = make_reference_globals(
        peer_x, peer_scale, schedule_peer, schedule_slot,
        tokens_per_expert, push_order, routed_x, routed_x_scale,
        m_indices, routed_y, combine, weights, topk_ids, output);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(peer_x.get_device());
    reference_dispatch_kernel<<<g.rows, kThreads, 0, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    split::entry_pipelined_out(
        const_cast<at::Tensor &>(routed_x),
        const_cast<at::Tensor &>(w13),
        const_cast<at::Tensor &>(routed_x_scale),
        const_cast<at::Tensor &>(w13_scale),
        const_cast<at::Tensor &>(m_indices),
        const_cast<at::Tensor &>(gate_up));
    reference_activation_kernel<<<g.rows, kActivationThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
        reinterpret_cast<uint8_t *>(hidden.data_ptr()),
        hidden_scale.data_ptr<float>(),
        g.rows, static_cast<float>(limit));
    CUDACHECK(cudaGetLastError());
    split::entry_pipelined_out(
        const_cast<at::Tensor &>(hidden),
        const_cast<at::Tensor &>(w2),
        const_cast<at::Tensor &>(hidden_scale),
        const_cast<at::Tensor &>(w2_scale),
        const_cast<at::Tensor &>(m_indices),
        const_cast<at::Tensor &>(routed_y));
    reference_push_kernel<<<g.rows, kThreads, 0, stream>>>(g);
    CUDACHECK(cudaGetLastError());
    reference_reduce_kernel<<<terminal::EP_SIZE * g.local_tokens,
                              kThreads, 0, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

void run_full(
        const at::Tensor &peer_x, const at::Tensor &peer_scale,
        const at::Tensor &w13, const at::Tensor &w13_scale,
        const at::Tensor &w2, const at::Tensor &w2_scale,
        const at::Tensor &schedule_peer, const at::Tensor &schedule_slot,
        const at::Tensor &num_tokens, const at::Tensor &tokens_per_expert,
        const at::Tensor &push_order, const at::Tensor &weights,
        const at::Tensor &topk_ids, const at::Tensor &routed_x,
        const at::Tensor &routed_x_scale, const at::Tensor &m_indices,
        const at::Tensor &gate_up, const at::Tensor &hidden,
        const at::Tensor &hidden_scale, const at::Tensor &routed_y,
        const at::Tensor &combine, const at::Tensor &route_ready,
        const at::Tensor &output, const at::Tensor &gate_up_ready,
        const at::Tensor &hidden_ready, const at::Tensor &y_ready,
        const at::Tensor &x_ready, const at::Tensor &cursor,
        const at::Tensor &worker_ticket,
        const at::Tensor &worker_failed,
        const at::Tensor &next_reduce_probe,
        const at::Tensor &reduce_done, const at::Tensor &comm_closed,
        const at::Tensor &comm_failed,
        const at::Tensor &task_visits,
        const at::Tensor &dispatch_visits,
        const at::Tensor &push_visits,
        const at::Tensor &epilogue_claim,
        const at::Tensor &reduce_visits, const at::Tensor &errors,
        const at::Tensor &progress_timeouts,
        const at::Tensor &dispatch_tiles_done,
        const at::Tensor &compute_started,
        const at::Tensor &overlap_witness,
        const at::Tensor &comm_owner,
        const at::Tensor &comm_worker_ticket,
        const at::Tensor &producer_done,
        const at::Tensor &push_done,
        const at::Tensor &terminate,
        int64_t ep_rank, int64_t compute_clusters, int64_t minibatch_rows,
        int64_t macrobatch_rows,
        int64_t overlap_delay_cycles, int64_t spin_limit, double limit) {
    c10::cuda::CUDAGuard guard(peer_x.device());
    const int rows = static_cast<int>(routed_x.size(0));
    const int local_tokens = static_cast<int>(peer_x.size(1));
    const int hidden_size = static_cast<int>(peer_x.size(2));
    const int scales = hidden_size / 128;
    TORCH_CHECK(rows == 64 || rows == 128,
                "full probe supports 64/128 active rows");
    TORCH_CHECK(ep_rank >= 0 && ep_rank < terminal::EP_SIZE,
                "ep_rank must be in [0, EP_SIZE)");
    TORCH_CHECK(num_tokens.is_cuda() && num_tokens.is_contiguous()
                    && num_tokens.scalar_type() == at::kInt
                    && num_tokens.numel() == 1,
                "num_tokens must be CUDA int32 [1]");
    TORCH_CHECK(spin_limit > 0, "spin_limit must be positive");
    TORCH_CHECK(overlap_delay_cycles >= 0
                    && overlap_delay_cycles <= UINT32_MAX,
                "overlap delay must fit uint32");
    const int max_resident_clusters = resident_clusters();
    TORCH_CHECK(compute_clusters >= 0
                    && compute_clusters
                        <= max_resident_clusters - full::COMM_CLUSTERS,
                "compute_clusters must be zero for the owner-only probe or "
                "fit measured resident capacity");
    const int64_t worker_slots = compute_clusters == 0 ? 1 : compute_clusters;
    TORCH_CHECK(worker_ticket.is_cuda() && worker_ticket.is_contiguous()
                    && worker_ticket.scalar_type() == at::kInt
                    && worker_ticket.numel() == worker_slots,
                "worker_ticket must be int32 [max(1, compute_clusters)]");
    TORCH_CHECK(worker_failed.is_cuda() && worker_failed.is_contiguous()
                    && worker_failed.scalar_type() == at::kInt
                    && worker_failed.numel() == worker_slots,
                "worker_failed must be int32 [max(1, compute_clusters)]");
    TORCH_CHECK(progress_timeouts.is_cuda()
                    && progress_timeouts.is_contiguous()
                    && progress_timeouts.scalar_type() == at::kInt
                    && progress_timeouts.numel() == 1,
                "progress_timeouts must be CUDA int32 [1]");
    for (const auto &[tensor, name] :
         std::array<std::pair<const at::Tensor *, const char *>, 5>{{
             {&comm_owner, "comm_owner"},
             {&comm_worker_ticket, "comm_worker_ticket"},
             {&producer_done, "producer_done"},
             {&push_done, "push_done"},
             {&terminate, "terminate"},
         }}) {
        TORCH_CHECK(tensor->is_cuda() && tensor->is_contiguous()
                        && tensor->scalar_type() == at::kInt
                        && tensor->numel() == 1,
                    name, " must be CUDA int32 [1]");
    }
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous()
                    && weights.scalar_type() == at::kFloat
                    && weights.numel()
                        == local_tokens * terminal::TOP_K,
                "weights must be rank-local float32 [local_tokens, TOP_K]");
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous()
                    && topk_ids.scalar_type() == at::kInt
                    && topk_ids.numel()
                        == local_tokens * terminal::TOP_K,
                "topk_ids must be rank-local int32 [local_tokens, TOP_K]");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous()
                    && output.scalar_type() == at::kBFloat16
                    && output.numel() == local_tokens * hidden_size,
                "output must be rank-local BF16 [local_tokens, hidden]");
    TORCH_CHECK(epilogue_claim.is_cuda() && epilogue_claim.is_contiguous()
                    && epilogue_claim.scalar_type() == at::kInt
                    && epilogue_claim.numel() == local_tokens,
                "epilogue_claim must be rank-local int32 [local_tokens]");
    TORCH_CHECK(reduce_visits.is_cuda() && reduce_visits.is_contiguous()
                    && reduce_visits.scalar_type() == at::kInt
                    && reduce_visits.numel() == local_tokens,
                "reduce_visits must be rank-local int32 [local_tokens]");

    gemm_problem w13_problem{
        kittens::py::tensor_to_gl<a_gl>(
            const_cast<at::Tensor &>(routed_x)),
        kittens::py::tensor_to_gl<b_gl>(const_cast<at::Tensor &>(w13)),
        kittens::py::tensor_to_gl<d_gl>(const_cast<at::Tensor &>(gate_up)),
        routed_x_scale.data_ptr<float>(), w13_scale.data_ptr<float>(),
        2 * terminal::INTERMEDIATE_SIZE,
        terminal::HIDDEN_SIZE / 128,
        (2 * terminal::INTERMEDIATE_SIZE) / 64,
    };
    gemm_problem w2_problem{
        kittens::py::tensor_to_gl<a_gl>(const_cast<at::Tensor &>(hidden)),
        kittens::py::tensor_to_gl<b_gl>(const_cast<at::Tensor &>(w2)),
        kittens::py::tensor_to_gl<d_gl>(const_cast<at::Tensor &>(routed_y)),
        hidden_scale.data_ptr<float>(), w2_scale.data_ptr<float>(),
        terminal::HIDDEN_SIZE,
        terminal::INTERMEDIATE_SIZE / 128,
        terminal::HIDDEN_SIZE / 64,
    };

    full::globals<gemm_problem> g(w13_problem, w2_problem);
    g.activation = {
        reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
        reinterpret_cast<uint8_t *>(hidden.data_ptr()),
        hidden_scale.data_ptr<float>(),
        static_cast<float>(limit),
    };
    g.ready = {
        reinterpret_cast<unsigned int *>(
            gate_up_ready.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(
            hidden_ready.data_ptr<int>()),
        reinterpret_cast<unsigned int *>(
            y_ready.data_ptr<int>()),
    };
    const auto *x_base = peer_x.data_ptr<uint8_t>();
    const auto *scale_base = peer_scale.data_ptr<float>();
    auto *combine_base = reinterpret_cast<uint8_t *>(
        combine.data_ptr());
    auto *ready_base = reinterpret_cast<unsigned int *>(
        route_ready.data_ptr<int>());
    for (int peer = 0; peer < terminal::EP_SIZE; ++peer) {
        g.x_peer[peer] = x_base
            + static_cast<size_t>(peer) * local_tokens * hidden_size;
        g.x_scale_peer[peer] = scale_base
            + static_cast<size_t>(peer) * local_tokens * scales;
        g.combine_peer[peer] = combine_base
            + static_cast<size_t>(peer) * local_tokens * terminal::TOP_K
                * hidden_size * sizeof(__nv_bfloat16);
        g.route_ready_peer[peer] = ready_base
            + static_cast<size_t>(peer) * local_tokens * terminal::TOP_K;
    }
    g.combine_local = g.combine_peer[ep_rank];
    g.route_ready_local = g.route_ready_peer[ep_rank];
    g.routed_x = reinterpret_cast<uint8_t *>(routed_x.data_ptr());
    g.routed_x_scale = routed_x_scale.data_ptr<float>();
    g.m_indices = m_indices.data_ptr<int>();
    g.schedule_peer_rank = schedule_peer.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_slot.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.tokens_per_expert = tokens_per_expert.data_ptr<int>();
    g.ep_size = terminal::EP_SIZE;
    g.ep_rank = static_cast<int>(ep_rank);
    g.num_local_tokens = local_tokens;
    g.hidden_size = hidden_size;
    g.scale_columns = scales;
    g.topk = terminal::TOP_K;
    g.num_local_experts = static_cast<int>(tokens_per_expert.numel());
    g.schedule_capacity = rows;
    g.routed_y = reinterpret_cast<const uint8_t *>(routed_y.data_ptr());
    g.push_order = push_order.data_ptr<int>();
    g.weights = weights.data_ptr<float>();
    g.topk_ids = topk_ids.data_ptr<int>();
    g.output = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
#define MOK_STATE_PTR(field, tensor)                                      \
    g.field = reinterpret_cast<unsigned int *>(                           \
        (tensor).data_ptr<int>())
    MOK_STATE_PTR(epilogue_claim, epilogue_claim);
    MOK_STATE_PTR(x_ready, x_ready);
    MOK_STATE_PTR(cursor, cursor);
    MOK_STATE_PTR(worker_ticket, worker_ticket);
    MOK_STATE_PTR(comm_worker_ticket, comm_worker_ticket);
    MOK_STATE_PTR(worker_failed, worker_failed);
    MOK_STATE_PTR(next_reduce_probe, next_reduce_probe);
    MOK_STATE_PTR(reduce_done, reduce_done);
    MOK_STATE_PTR(comm_closed, comm_closed);
    MOK_STATE_PTR(comm_failed, comm_failed);
    MOK_STATE_PTR(task_visits, task_visits);
    MOK_STATE_PTR(dispatch_visits, dispatch_visits);
    MOK_STATE_PTR(push_visits, push_visits);
    MOK_STATE_PTR(reduce_visits, reduce_visits);
    MOK_STATE_PTR(errors, errors);
    MOK_STATE_PTR(progress_timeouts, progress_timeouts);
    MOK_STATE_PTR(dispatch_tiles_done, dispatch_tiles_done);
    MOK_STATE_PTR(compute_started, compute_started);
    MOK_STATE_PTR(overlap_witness, overlap_witness);
    if (compute_clusters == 0) {
        // Test-only forced owner case: launch exactly one physical cluster and
        // connect the production closure plane.  With no non-owner in the
        // grid, task_visits proves that every producer ran on the owner.
        MOK_STATE_PTR(comm_owner, comm_owner);
        MOK_STATE_PTR(producer_done, producer_done);
        MOK_STATE_PTR(push_done, push_done);
        MOK_STATE_PTR(terminate, terminate);
    }
#undef MOK_STATE_PTR
    g.compute_clusters = static_cast<int>(compute_clusters);
    g.minibatch_rows = static_cast<int>(minibatch_rows);
    g.macrobatch_rows = static_cast<int>(macrobatch_rows);
    g.overlap_delay_after_first_dispatch_cycles =
        static_cast<unsigned int>(overlap_delay_cycles);
    g.spin_limit = static_cast<unsigned long long>(spin_limit);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(peer_x.get_device());
    full::kernel<gemm_problem>
        <<<(full::COMM_CLUSTERS + g.compute_clusters)
               * terminal::CLUSTER_CTAS,
           kThreads, kDynamicSmem, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

std::vector<int64_t> attributes() {
    const int clusters = resident_clusters();
    cudaFuncAttributes attributes{};
    CUDACHECK(cudaFuncGetAttributes(
        &attributes, full::kernel<gemm_problem>));
    return {
        static_cast<int64_t>(attributes.numRegs),
        static_cast<int64_t>(attributes.sharedSizeBytes),
        static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.maxDynamicSharedSizeBytes),
        static_cast<int64_t>(kDynamicSmem),
        static_cast<int64_t>(clusters),
        static_cast<int64_t>(clusters - full::COMM_CLUSTERS),
    };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_split", &run_split);
    module.def("run_full", &run_full);
    module.def("attributes", &attributes);
}
