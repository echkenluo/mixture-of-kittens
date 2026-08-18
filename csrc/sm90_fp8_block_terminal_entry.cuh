#pragma once

#if defined(KITTENS_SM90)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "pyutils/torchutils.cuh"
#include "sm90_fp8_block_routed.cuh"
#include "sm90_fp8_block_terminal_full.cuh"
#include "utils.cuh"

namespace mok_sm90::fp8_block_terminal_entry {

namespace terminal = fp8_block_terminal;
namespace full = fp8_block_terminal_full;
namespace compute = fp8_block_terminal_compute;

using namespace kittens;

using a_gl = gl<fp8e4m3, 1, 1, -1, -1, compute::a_st>;
using b_gl = gl<fp8e4m3, 1, 1, -1, -1, compute::b_st>;
using d_gl = gl<bf16, 1, 1, -1, -1, compute::d_st>;

// This type is deliberately fixed.  The terminal entry is not a generic
// grouped GEMM API: it is the EP4/H4096/I2048/top-6 DeepSeek-V4 forward.
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

constexpr int DYNAMIC_SMEM =
    compute::PIPE_DEPTH * (sizeof(compute::a_st) + sizeof(compute::b_st))
    + sizeof(compute::d_st) + 1024;
constexpr int PREPARE_THREADS = 256;

struct prepare_globals {
    const int *topk_ids;
    unsigned int *route_ready;
    int64_t route_count;
    int64_t valid_route_count;

    unsigned int *x_routed_ready;
    int64_t x_routed_ready_count;
    unsigned int *gate_up_tile_ready;
    int64_t gate_up_tile_ready_count;
    unsigned int *hidden_row_block_ready;
    int64_t hidden_row_block_ready_count;
    unsigned int *y_routed_ready;
    int64_t y_routed_ready_count;
    unsigned int *y_routed_done;
    int64_t y_routed_done_count;
    unsigned int *epilogue_claim;
    int64_t epilogue_claim_count;
    unsigned int *next_logical_cluster;
    unsigned int *next_reduce_probe;
    unsigned int *worker_ticket;
    int64_t worker_ticket_count;
    unsigned int *producer_done;
    unsigned int *comm_closed;
    unsigned int *push_done;
    unsigned int *reduce_done;
    unsigned int *terminate;
    unsigned int *epilogue_done;
    unsigned int *input_expected_scratch;
    int64_t max_count;
};

__device__ __forceinline__ void clear_if_present(
        unsigned int *address, int64_t count, int64_t index) {
    if (index < count)
        address[index] = 0u;
}

__global__ void prepare_kernel(prepare_globals g) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
             + threadIdx.x;
         index < g.max_count; index += stride) {
        if (index < g.route_count) {
            // Padding tokens and invalid top-k slots are pre-closed.  A valid
            // producer slot starts at zero and is release-published to one by
            // the terminal communication role.
            g.route_ready[index] =
                index < g.valid_route_count && g.topk_ids[index] >= 0
                    ? 0u : 1u;
        }
        clear_if_present(
            g.x_routed_ready, g.x_routed_ready_count, index);
        clear_if_present(
            g.gate_up_tile_ready, g.gate_up_tile_ready_count, index);
        clear_if_present(
            g.hidden_row_block_ready,
            g.hidden_row_block_ready_count, index);
        clear_if_present(
            g.y_routed_ready, g.y_routed_ready_count, index);
        clear_if_present(
            g.y_routed_done, g.y_routed_done_count, index);
        clear_if_present(
            g.epilogue_claim, g.epilogue_claim_count, index);
        clear_if_present(g.worker_ticket, g.worker_ticket_count, index);
        if (index == 0) {
            *g.next_logical_cluster = 0u;
            *g.next_reduce_probe = 0u;
            *g.producer_done = 0u;
            *g.comm_closed = 0u;
            *g.push_done = 0u;
            *g.reduce_done = 0u;
            *g.terminate = 0u;
            *g.epilogue_done = 0u;
            *g.input_expected_scratch = 0u;
        }
    }
}

inline void check_cuda_contiguous(
        const at::Tensor &tensor, at::ScalarType dtype,
        const at::Device &device, const char *name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device,
                name, " must be on the terminal CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype,
                name, " has the wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_i32_state(
        const at::Tensor &tensor, const at::Device &device,
        int64_t count, const char *name) {
    check_cuda_contiguous(tensor, at::kInt, device, name);
    TORCH_CHECK(tensor.numel() == count,
                name, " has the wrong number of elements: expected ", count,
                ", got ", tensor.numel());
}

inline unsigned int *u32_ptr(const at::Tensor &tensor) {
    return reinterpret_cast<unsigned int *>(tensor.data_ptr<int>());
}

inline void entry_prepare_out(
    const at::Tensor &topk_ids, const at::Tensor &route_ready,
    const at::Tensor &x_routed_ready,
    const at::Tensor &gate_up_tile_ready,
    const at::Tensor &hidden_row_block_ready,
    const at::Tensor &y_routed_ready, const at::Tensor &y_routed_done,
    const at::Tensor &epilogue_claim,
    const at::Tensor &next_logical_cluster,
    const at::Tensor &next_reduce_probe,
    const at::Tensor &worker_ticket,
    const at::Tensor &producer_done, const at::Tensor &comm_closed,
    const at::Tensor &push_done, const at::Tensor &reduce_done,
    const at::Tensor &terminate, const at::Tensor &epilogue_done,
    const at::Tensor &input_expected_scratch) {
    TORCH_CHECK(topk_ids.dim() == 2
                    && topk_ids.size(1) == terminal::TOP_K
                    && topk_ids.size(0) > 0,
                "topk_ids must be int32 [local_tokens,6]");
    const at::Device device = topk_ids.device();
    check_cuda_contiguous(topk_ids, at::kInt, device, "topk_ids");
    TORCH_CHECK(route_ready.dim() == 2
                    && route_ready.size(1) == terminal::TOP_K
                    && route_ready.size(0) >= topk_ids.size(0)
                    && route_ready.size(0) % terminal::M_TILE == 0,
                "route_ready must be int32 [padded_local_tokens,6] with "
                "M64 token padding");
    check_cuda_contiguous(route_ready, at::kInt, device, "route_ready");

    const int64_t m_tiles = x_routed_ready.numel();
    TORCH_CHECK(m_tiles > 0, "terminal capacity must contain an M64 tile");
    check_i32_state(x_routed_ready, device, m_tiles, "x_routed_ready");
    check_i32_state(
        gate_up_tile_ready, device,
        m_tiles * terminal::W13_N_TILES, "gate_up_tile_ready");
    check_i32_state(hidden_row_block_ready, device, m_tiles,
                    "hidden_row_block_ready");
    check_i32_state(y_routed_ready, device, m_tiles, "y_routed_ready");
    check_i32_state(y_routed_done, device, m_tiles, "y_routed_done");
    check_i32_state(epilogue_claim, device, route_ready.size(0),
                    "epilogue_claim");
    check_i32_state(next_logical_cluster, device, 1,
                    "next_logical_cluster");
    check_i32_state(next_reduce_probe, device, 1, "next_reduce_probe");
    check_cuda_contiguous(worker_ticket, at::kInt, device, "worker_ticket");
    TORCH_CHECK(worker_ticket.dim() == 1 && worker_ticket.numel() > 0,
                "worker_ticket must be nonempty int32 [compute_clusters]");
    for (const auto &[tensor, name] :
         std::array<std::pair<const at::Tensor *, const char *>, 8>{{
             {&producer_done, "producer_done"},
             {&comm_closed, "comm_closed"},
             {&push_done, "push_done"},
             {&reduce_done, "reduce_done"},
             {&terminate, "terminate"},
             {&epilogue_done, "epilogue_done"},
             {&input_expected_scratch, "input_expected_scratch"},
             {&next_reduce_probe, "next_reduce_probe"},
         }})
        check_i32_state(*tensor, device, 1, name);

    c10::cuda::CUDAGuard guard(device);
    prepare_globals g{
        topk_ids.data_ptr<int>(), u32_ptr(route_ready), route_ready.numel(),
        topk_ids.numel(), u32_ptr(x_routed_ready), x_routed_ready.numel(),
        u32_ptr(gate_up_tile_ready), gate_up_tile_ready.numel(),
        u32_ptr(hidden_row_block_ready), hidden_row_block_ready.numel(),
        u32_ptr(y_routed_ready), y_routed_ready.numel(),
        u32_ptr(y_routed_done), y_routed_done.numel(),
        u32_ptr(epilogue_claim), epilogue_claim.numel(),
        u32_ptr(next_logical_cluster), u32_ptr(next_reduce_probe),
        u32_ptr(worker_ticket), worker_ticket.numel(),
        u32_ptr(producer_done), u32_ptr(comm_closed), u32_ptr(push_done),
        u32_ptr(reduce_done), u32_ptr(terminate), u32_ptr(epilogue_done),
        u32_ptr(input_expected_scratch), 0,
    };
    g.max_count = std::max({
        g.route_count, g.x_routed_ready_count,
        g.gate_up_tile_ready_count, g.hidden_row_block_ready_count,
        g.y_routed_ready_count, g.y_routed_done_count,
        g.epilogue_claim_count, g.worker_ticket_count, int64_t{1}});
    const int blocks = static_cast<int>(std::min<int64_t>(
        4096, (g.max_count + PREPARE_THREADS - 1) / PREPARE_THREADS));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    prepare_kernel<<<blocks, PREPARE_THREADS, 0, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

// Per-device cache: workspace creation calls prewarm before capture; a cold
// entry inside capture fails closed instead of issuing an illegal query.
inline int &occupancy_slot(int device_index) {
    static std::array<int, 64> cache = [] {
        std::array<int, 64> result{};
        result.fill(-1);
        return result;
    }();
    return cache.at(static_cast<size_t>(device_index));
}

inline int occupancy_cache(int device_index) {
    return occupancy_slot(device_index);
}

inline int prewarm(int device_index) {
    int device_count = 0;
    CUDACHECK(cudaGetDeviceCount(&device_count));
    TORCH_CHECK(device_index >= 0 && device_index < device_count,
                "terminal prewarm device index is out of range");
    TORCH_CHECK(device_index < 64,
                "terminal occupancy cache supports at most 64 devices");
    c10::cuda::CUDAGuard guard(device_index);
    CUDACHECK(cudaFuncSetAttribute(
        full::kernel<gemm_problem>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(
        (full::COMM_CLUSTERS + 1) * terminal::CLUSTER_CTAS, 1, 1);
    config.blockDim = dim3(terminal::THREADS_PER_CTA, 1, 1);
    config.dynamicSmemBytes = DYNAMIC_SMEM;
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
                "terminal kernel cannot co-reside one comm and one compute "
                "cluster");
    occupancy_slot(device_index) = clusters;
    return clusters;
}

inline int64_t entry_prewarm(int64_t device_index) {
    TORCH_CHECK(device_index >= std::numeric_limits<int>::min()
                    && device_index <= std::numeric_limits<int>::max(),
                "device_index does not fit int");
    const int resident = prewarm(static_cast<int>(device_index));
    return static_cast<int64_t>(resident - full::COMM_CLUSTERS);
}

inline void entry_out(
    const at::Tensor &x_buffer, const std::vector<int64_t> &x_ptrs,
    const at::Tensor &x_scale_buffer,
    const std::vector<int64_t> &x_scale_ptrs,
    const at::Tensor &routed_x, const at::Tensor &routed_x_scale,
    const at::Tensor &m_indices, const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx, const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    const at::Tensor &w13, const at::Tensor &w13_scale,
    const at::Tensor &gate_up, const at::Tensor &down_input,
    const at::Tensor &down_input_scale,
    const at::Tensor &w2, const at::Tensor &w2_scale,
    const at::Tensor &routed_y,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,
    const at::Tensor &route_ready,
    const std::vector<int64_t> &route_ready_ptrs,
    const at::Tensor &topk_weights, const at::Tensor &topk_ids,
    const at::Tensor &output,
    const at::Tensor &x_routed_ready,
    const at::Tensor &gate_up_tile_ready,
    const at::Tensor &hidden_row_block_ready,
    const at::Tensor &y_routed_ready, const at::Tensor &y_routed_done,
    const at::Tensor &epilogue_claim,
    const at::Tensor &next_logical_cluster,
    const at::Tensor &next_reduce_probe,
    const at::Tensor &worker_ticket,
    const at::Tensor &producer_done, const at::Tensor &comm_closed,
    const at::Tensor &push_done, const at::Tensor &reduce_done,
    const at::Tensor &terminate, const at::Tensor &epilogue_done,
    const at::Tensor &in_use, const at::Tensor &barrier_buffer,
    const at::Tensor &barrier_target,
    const at::Tensor &input_expected_scratch,
    int64_t barrier_buffer_multicast_ptr, int64_t trap_record_ptr,
    int64_t ep_rank, int64_t compute_clusters,
    int64_t minibatch_rows, int64_t macrobatch_rows,
    double swiglu_limit, int64_t spin_limit) {
    const at::Device device = routed_x.device();
    check_cuda_contiguous(routed_x, at::kFloat8_e4m3fn, device, "routed_x");
    TORCH_CHECK(routed_x.dim() == 2
                    && routed_x.size(0) > 0
                    && routed_x.size(0) % terminal::M_TILE == 0
                    && routed_x.size(1) == terminal::HIDDEN_SIZE,
                "routed_x must be FP8 [capacity(M64),4096]");
    const int64_t capacity = routed_x.size(0);
    const int64_t m_tiles = capacity / terminal::M_TILE;
    TORCH_CHECK(capacity <= std::numeric_limits<int>::max(),
                "terminal capacity does not fit int");

    check_cuda_contiguous(x_buffer, at::kFloat8_e4m3fn, device, "x_buffer");
    TORCH_CHECK(x_buffer.dim() == 2 && x_buffer.size(0) > 0
                    && x_buffer.size(1) == terminal::HIDDEN_SIZE,
                "x_buffer must be FP8 [local_tokens,4096]");
    const int64_t local_tokens = x_buffer.size(0);
    TORCH_CHECK(local_tokens <= std::numeric_limits<int>::max(),
                "local token count does not fit int");
    check_cuda_contiguous(
        x_scale_buffer, at::kFloat, device, "x_scale_buffer");
    TORCH_CHECK(x_scale_buffer.sizes()
                    == at::IntArrayRef({local_tokens,
                                       terminal::HIDDEN_SIZE / 128}),
                "x_scale_buffer must be float32 [local_tokens,32]");
    check_cuda_contiguous(
        routed_x_scale, at::kFloat, device, "routed_x_scale");
    TORCH_CHECK(routed_x_scale.sizes()
                    == at::IntArrayRef({capacity,
                                       terminal::HIDDEN_SIZE / 128}),
                "routed_x_scale must be float32 [capacity,32]");
    check_i32_state(m_indices, device, capacity, "m_indices");

    fp8_block_routed::check_pointer_list(x_ptrs, "x_ptrs");
    fp8_block_routed::check_pointer_list(x_scale_ptrs, "x_scale_ptrs");
    fp8_block_routed::check_pointer_list(
        combine_buffer_ptrs, "combine_buffer_ptrs");
    fp8_block_routed::check_pointer_list(
        route_ready_ptrs, "route_ready_ptrs");
    TORCH_CHECK(x_ptrs.size() == terminal::EP_SIZE
                    && x_scale_ptrs.size() == terminal::EP_SIZE
                    && combine_buffer_ptrs.size() == terminal::EP_SIZE
                    && route_ready_ptrs.size() == terminal::EP_SIZE,
                "terminal peer pointer lists must all have EP4 entries");
    TORCH_CHECK(ep_rank >= 0 && ep_rank < terminal::EP_SIZE,
                "ep_rank must be in [0,4)");
    const size_t rank = static_cast<size_t>(ep_rank);
    TORCH_CHECK(x_ptrs[rank] ==
                    reinterpret_cast<int64_t>(x_buffer.data_ptr()),
                "x_ptrs[ep_rank] must alias x_buffer");
    TORCH_CHECK(x_scale_ptrs[rank] ==
                    reinterpret_cast<int64_t>(x_scale_buffer.data_ptr()),
                "x_scale_ptrs[ep_rank] must alias x_scale_buffer");

    fp8_block_routed::check_schedule(
        schedule_peer_rank, schedule_peer_token_idx, num_tokens,
        tokens_per_expert, capacity);
    for (const auto &[tensor, name] :
         std::array<std::pair<const at::Tensor *, const char *>, 4>{{
             {&schedule_peer_rank, "schedule_peer_rank"},
             {&schedule_peer_token_idx, "schedule_peer_token_idx"},
             {&num_tokens, "num_tokens"},
             {&tokens_per_expert, "tokens_per_expert"},
         }})
        TORCH_CHECK(tensor->device() == device,
                    name, " must be on the terminal device");
    const int64_t experts = tokens_per_expert.numel();
    TORCH_CHECK(experts > 0 && experts <= full::MAX_EXPERTS,
                "terminal local expert count must be in [1,256]");

    check_cuda_contiguous(w13, at::kFloat8_e4m3fn, device, "w13");
    TORCH_CHECK(w13.sizes() == at::IntArrayRef(
                    {experts, 2 * terminal::INTERMEDIATE_SIZE,
                     terminal::HIDDEN_SIZE}),
                "w13 must be packed FP8 [E,4096,4096]");
    check_cuda_contiguous(w13_scale, at::kFloat, device, "w13_scale");
    TORCH_CHECK(w13_scale.sizes() == at::IntArrayRef({experts, 32, 32}),
                "w13_scale must be float32 [E,32,32]");
    check_cuda_contiguous(gate_up, at::kBFloat16, device, "gate_up");
    TORCH_CHECK(gate_up.sizes() == at::IntArrayRef({capacity, 4096}),
                "gate_up must be BF16 [capacity,4096]");
    check_cuda_contiguous(
        down_input, at::kFloat8_e4m3fn, device, "down_input");
    TORCH_CHECK(down_input.sizes() == at::IntArrayRef({capacity, 2048}),
                "down_input must be FP8 [capacity,2048]");
    check_cuda_contiguous(
        down_input_scale, at::kFloat, device, "down_input_scale");
    TORCH_CHECK(down_input_scale.sizes()
                    == at::IntArrayRef({capacity, 16}),
                "down_input_scale must be float32 [capacity,16]");
    check_cuda_contiguous(w2, at::kFloat8_e4m3fn, device, "w2");
    TORCH_CHECK(w2.sizes() == at::IntArrayRef({experts, 4096, 2048}),
                "w2 must be FP8 [E,4096,2048]");
    check_cuda_contiguous(w2_scale, at::kFloat, device, "w2_scale");
    TORCH_CHECK(w2_scale.sizes() == at::IntArrayRef({experts, 32, 16}),
                "w2_scale must be float32 [E,32,16]");
    check_cuda_contiguous(routed_y, at::kBFloat16, device, "routed_y");
    TORCH_CHECK(routed_y.sizes() == at::IntArrayRef({capacity, 4096}),
                "routed_y must be BF16 [capacity,4096]");

    check_cuda_contiguous(
        combine_buffer, at::kBFloat16, device, "combine_buffer");
    TORCH_CHECK(combine_buffer.dim() == 2
                    && combine_buffer.size(1) == terminal::HIDDEN_SIZE
                    && combine_buffer.size(0) % terminal::TOP_K == 0,
                "combine_buffer must be BF16 [padded_tokens*6,4096]");
    const int64_t padded_tokens =
        combine_buffer.size(0) / terminal::TOP_K;
    TORCH_CHECK(padded_tokens >= local_tokens
                    && padded_tokens % terminal::M_TILE == 0
                    && padded_tokens <= std::numeric_limits<int>::max(),
                "combine_buffer token domain must be M64 padded");
    check_cuda_contiguous(route_ready, at::kInt, device, "route_ready");
    TORCH_CHECK(route_ready.sizes()
                    == at::IntArrayRef({padded_tokens, terminal::TOP_K}),
                "route_ready must be int32 [padded_tokens,6]");
    TORCH_CHECK(combine_buffer_ptrs[rank] ==
                    reinterpret_cast<int64_t>(combine_buffer.data_ptr()),
                "combine_buffer_ptrs[ep_rank] must alias combine_buffer");
    TORCH_CHECK(route_ready_ptrs[rank] ==
                    reinterpret_cast<int64_t>(route_ready.data_ptr()),
                "route_ready_ptrs[ep_rank] must alias route_ready");
    check_cuda_contiguous(
        topk_weights, at::kFloat, device, "topk_weights");
    check_cuda_contiguous(topk_ids, at::kInt, device, "topk_ids");
    TORCH_CHECK(topk_weights.sizes()
                    == at::IntArrayRef({local_tokens, terminal::TOP_K})
                    && topk_ids.sizes() == topk_weights.sizes(),
                "topk weights/ids must be [local_tokens,6]");
    check_cuda_contiguous(output, at::kBFloat16, device, "output");
    TORCH_CHECK(output.sizes()
                    == at::IntArrayRef({local_tokens,
                                       terminal::HIDDEN_SIZE}),
                "output must be BF16 [local_tokens,4096]");

    check_i32_state(x_routed_ready, device, m_tiles, "x_routed_ready");
    check_i32_state(gate_up_tile_ready, device,
                    m_tiles * terminal::W13_N_TILES,
                    "gate_up_tile_ready");
    check_i32_state(hidden_row_block_ready, device, m_tiles,
                    "hidden_row_block_ready");
    check_i32_state(y_routed_ready, device, m_tiles, "y_routed_ready");
    check_i32_state(y_routed_done, device, m_tiles, "y_routed_done");
    check_i32_state(epilogue_claim, device, padded_tokens,
                    "epilogue_claim");
    check_i32_state(next_logical_cluster, device, 1,
                    "next_logical_cluster");
    check_i32_state(next_reduce_probe, device, 1, "next_reduce_probe");
    check_i32_state(producer_done, device, 1, "producer_done");
    check_i32_state(comm_closed, device, 1, "comm_closed");
    check_i32_state(push_done, device, 1, "push_done");
    check_i32_state(reduce_done, device, 1, "reduce_done");
    check_i32_state(terminate, device, 1, "terminate");
    check_i32_state(epilogue_done, device, 1, "epilogue_done");
    check_i32_state(in_use, device, 1, "in_use");
    check_i32_state(barrier_buffer, device, 1, "barrier_buffer");
    check_i32_state(barrier_target, device, 1, "barrier_target");
    check_i32_state(input_expected_scratch, device, 1,
                    "input_expected_scratch");

    TORCH_CHECK(compute_clusters > 0
                    && compute_clusters <= std::numeric_limits<int>::max(),
                "compute_clusters must be a positive int");
    check_i32_state(worker_ticket, device, compute_clusters, "worker_ticket");
    TORCH_CHECK(minibatch_rows > 0 && minibatch_rows % terminal::M_TILE == 0
                    && minibatch_rows <= std::numeric_limits<int>::max(),
                "minibatch_rows must be a positive M64-aligned int");
    TORCH_CHECK(macrobatch_rows >= minibatch_rows
                    && macrobatch_rows % minibatch_rows == 0
                    && macrobatch_rows <= std::numeric_limits<int>::max(),
                "macrobatch_rows must be a positive multiple of minibatch_rows");
    TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit > 0.0
                    && swiglu_limit <= std::numeric_limits<float>::max(),
                "swiglu_limit must be positive, finite, and fit float32");
    TORCH_CHECK(spin_limit > 0,
                "spin_limit must be positive");
    TORCH_CHECK(barrier_buffer_multicast_ptr > 0,
                "barrier multicast pointer must be positive");
    TORCH_CHECK(trap_record_ptr > 0,
                "trap_record_ptr must be a mapped pinned host address");

    c10::cuda::CUDAGuard guard(device);
    const int device_index = device.index();
    TORCH_CHECK(device_index >= 0 && device_index < 64,
                "terminal occupancy cache device index is out of range");
    int resident = occupancy_cache(device_index);
    if (resident < 0) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDACHECK(cudaStreamIsCapturing(
            at::cuda::getCurrentCUDAStream(device_index), &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                    "terminal occupancy cache is cold during graph capture; "
                    "call fp8_block_megakernel_prewarm at workspace creation");
        resident = prewarm(device_index);
    }
    TORCH_CHECK(full::COMM_CLUSTERS + compute_clusters <= resident,
                "requested 1+N terminal clusters exceed resident capacity");

    gemm_problem w13_problem{
        kittens::py::tensor_to_gl<a_gl>(const_cast<at::Tensor &>(routed_x)),
        kittens::py::tensor_to_gl<b_gl>(const_cast<at::Tensor &>(w13)),
        kittens::py::tensor_to_gl<d_gl>(const_cast<at::Tensor &>(gate_up)),
        routed_x_scale.data_ptr<float>(), w13_scale.data_ptr<float>(),
        2 * terminal::INTERMEDIATE_SIZE,
        terminal::HIDDEN_SIZE / 128,
        (2 * terminal::INTERMEDIATE_SIZE) / 64,
    };
    gemm_problem w2_problem{
        kittens::py::tensor_to_gl<a_gl>(const_cast<at::Tensor &>(down_input)),
        kittens::py::tensor_to_gl<b_gl>(const_cast<at::Tensor &>(w2)),
        kittens::py::tensor_to_gl<d_gl>(const_cast<at::Tensor &>(routed_y)),
        down_input_scale.data_ptr<float>(), w2_scale.data_ptr<float>(),
        terminal::HIDDEN_SIZE,
        terminal::INTERMEDIATE_SIZE / 128,
        terminal::HIDDEN_SIZE / 64,
    };
    full::globals<gemm_problem> g(w13_problem, w2_problem);
    g.activation = {
        reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr()),
        reinterpret_cast<uint8_t *>(down_input.data_ptr()),
        down_input_scale.data_ptr<float>(),
        static_cast<float>(swiglu_limit),
    };
    g.ready = {
        u32_ptr(gate_up_tile_ready), u32_ptr(hidden_row_block_ready),
        u32_ptr(y_routed_ready),
    };
    for (int peer = 0; peer < terminal::EP_SIZE; ++peer) {
        g.x_peer[peer] = reinterpret_cast<const uint8_t *>(x_ptrs[peer]);
        g.x_scale_peer[peer] =
            reinterpret_cast<const float *>(x_scale_ptrs[peer]);
        g.combine_peer[peer] =
            reinterpret_cast<uint8_t *>(combine_buffer_ptrs[peer]);
        g.route_ready_peer[peer] =
            reinterpret_cast<unsigned int *>(route_ready_ptrs[peer]);
    }
    g.routed_x = reinterpret_cast<uint8_t *>(routed_x.data_ptr());
    g.routed_x_scale = routed_x_scale.data_ptr<float>();
    g.m_indices = m_indices.data_ptr<int>();
    g.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    g.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    g.num_tokens = num_tokens.data_ptr<int>();
    g.tokens_per_expert = tokens_per_expert.data_ptr<int>();
    g.ep_size = terminal::EP_SIZE;
    g.ep_rank = static_cast<int>(ep_rank);
    g.num_local_tokens = static_cast<int>(local_tokens);
    g.hidden_size = terminal::HIDDEN_SIZE;
    g.scale_columns = terminal::HIDDEN_SIZE / 128;
    g.topk = terminal::TOP_K;
    g.num_local_experts = static_cast<int>(experts);
    g.schedule_capacity = static_cast<int>(capacity);
    g.routed_y = reinterpret_cast<const uint8_t *>(routed_y.data_ptr());
    g.combine_local = reinterpret_cast<const uint8_t *>(
        combine_buffer.data_ptr());
    g.route_ready_local = u32_ptr(route_ready);
    g.push_order = nullptr;
    g.weights = topk_weights.data_ptr<float>();
    g.topk_ids = topk_ids.data_ptr<int>();
    g.output = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
    g.epilogue_claim = u32_ptr(epilogue_claim);
    g.x_ready = u32_ptr(x_routed_ready);
    g.cursor = u32_ptr(next_logical_cluster);
    g.worker_ticket = u32_ptr(worker_ticket);
    g.worker_failed = nullptr;
    g.next_reduce_probe = u32_ptr(next_reduce_probe);
    g.reduce_done = u32_ptr(reduce_done);
    g.comm_closed = u32_ptr(comm_closed);
    g.comm_failed = nullptr;
    g.task_visits = nullptr;
    g.dispatch_visits = nullptr;
    g.push_visits = nullptr;
    g.reduce_visits = nullptr;
    g.errors = nullptr;
    g.progress_timeouts = nullptr;
    g.dispatch_tiles_done = nullptr;
    g.compute_started = nullptr;
    g.overlap_witness = nullptr;
    g.barrier_flag = u32_ptr(barrier_buffer);
    g.barrier_target = u32_ptr(barrier_target);
    g.barrier_multicast_ptr = reinterpret_cast<unsigned int *>(
        barrier_buffer_multicast_ptr);
    g.input_expected_scratch = u32_ptr(input_expected_scratch);
    g.in_use = u32_ptr(in_use);
    g.epilogue_done = u32_ptr(epilogue_done);
    g.producer_done = u32_ptr(producer_done);
    g.push_done = u32_ptr(push_done);
    g.terminate = u32_ptr(terminate);
    g.trap_record = utils::mok_resolve_trap_record(trap_record_ptr);
    g.compute_clusters = static_cast<int>(compute_clusters);
    g.minibatch_rows = static_cast<int>(minibatch_rows);
    g.macrobatch_rows = static_cast<int>(macrobatch_rows);
    g.overlap_delay_after_first_dispatch_cycles = 0;
    g.spin_limit = static_cast<unsigned long long>(spin_limit);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index);
    full::kernel<gemm_problem>
        <<<(full::COMM_CLUSTERS + g.compute_clusters)
                   * terminal::CLUSTER_CTAS,
           terminal::THREADS_PER_CTA, DYNAMIC_SMEM, stream>>>(g);
    CUDACHECK(cudaGetLastError());
}

}  // namespace mok_sm90::fp8_block_terminal_entry

#endif  // defined(KITTENS_SM90)
