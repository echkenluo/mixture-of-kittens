#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <initializer_list>

#include "../csrc/sm90_fp8_block_terminal_comm_primitives.cuh"

namespace {

namespace comm = mok_sm90::fp8_block_terminal_comm;

constexpr int kPeers = 4;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxExperts = 16;

struct dispatch_globals {
    const uint8_t *x_peer[kPeers];
    const float *x_scale_peer[kPeers];
    uint8_t *routed_x;
    float *routed_x_scale;
    int *m_indices;
    const int *schedule_peer_rank;
    const int *schedule_peer_token_idx;
    const int *num_tokens;
    const int *tokens_per_expert;
    int ep_size;
    int num_local_tokens;
    int hidden_size;
    int scale_columns;
    int topk;
    int num_local_experts;
    int schedule_capacity;
};

struct push_globals {
    uint8_t *combine_peer[kPeers];
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

__global__ __launch_bounds__(kThreads, 1)
void shared_dispatch_kernel(dispatch_globals g) {
    __shared__ int expert_row_end[kMaxExperts];
    if (threadIdx.x == 0)
        comm::build_expert_row_ends(g, expert_row_end);
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row < comm::bounded_valid_rows(g)) {
        comm::dispatch_copy_row(g, expert_row_end, row, lane);
        __syncwarp();
    }
}

// Independent transcription of the pre-extraction row semantics.  This is
// intentionally not implemented through the shared helper: the probe compares
// the helper path with this legacy-shaped reference byte for byte.
__global__ __launch_bounds__(kThreads, 1)
void legacy_dispatch_reference_kernel(dispatch_globals g) {
    const int row = static_cast<int>(blockIdx.x);
    const int device_rows = g.num_tokens[0];
    const int valid_rows = device_rows < g.schedule_capacity
        ? device_rows
        : g.schedule_capacity;
    if (row >= valid_rows)
        return;
    const int peer_rank = g.schedule_peer_rank[row];
    const int peer_token_idx = g.schedule_peer_token_idx[row];
    const bool valid = peer_rank >= 0 && peer_rank < g.ep_size
        && peer_token_idx >= 0
        && peer_token_idx < g.num_local_tokens * g.topk;
    const int fp8_vectors = g.hidden_size / static_cast<int>(sizeof(uint4));
    auto *dst = reinterpret_cast<uint4 *>(g.routed_x)
        + static_cast<size_t>(row) * fp8_vectors;
    float *dst_scale = g.routed_x_scale
        + static_cast<size_t>(row) * g.scale_columns;
    if (valid) {
        const int source_row = peer_token_idx / g.topk;
        const auto *src = reinterpret_cast<const uint4 *>(g.x_peer[peer_rank])
            + static_cast<size_t>(source_row) * fp8_vectors;
        const float *src_scale = g.x_scale_peer[peer_rank]
            + static_cast<size_t>(source_row) * g.scale_columns;
        for (int index = threadIdx.x; index < fp8_vectors;
             index += blockDim.x)
            dst[index] = src[index];
        for (int index = threadIdx.x; index < g.scale_columns;
             index += blockDim.x)
            dst_scale[index] = src_scale[index];
    } else {
        const uint4 zero{0, 0, 0, 0};
        for (int index = threadIdx.x; index < fp8_vectors;
             index += blockDim.x)
            dst[index] = zero;
        for (int index = threadIdx.x; index < g.scale_columns;
             index += blockDim.x)
            dst_scale[index] = 0.0f;
    }
    if (threadIdx.x == 0) {
        int expert = 0;
        int offset = g.tokens_per_expert[0];
        while (expert < g.num_local_experts - 1 && row >= offset) {
            ++expert;
            offset += g.tokens_per_expert[expert];
        }
        g.m_indices[row] = expert;
    }
}

__global__ __launch_bounds__(kThreads, 1)
void shared_push_kernel(push_globals g, int *route_receipt) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row < comm::bounded_valid_rows(g)) {
        const comm::route_mapping route =
            comm::push_routed_row(g, row, lane);
        __syncwarp();
        if (lane == 0) {
            int *receipt = route_receipt + static_cast<size_t>(row) * 4;
            receipt[0] = route.peer_rank;
            receipt[1] = route.peer_token_idx;
            receipt[2] = route.source_row;
            receipt[3] = route.valid ? 1 : 0;
        }
    }
}

__global__ __launch_bounds__(kThreads, 1)
void legacy_push_reference_kernel(push_globals g) {
    const int row = static_cast<int>(blockIdx.x);
    const int device_rows = g.num_tokens[0];
    const int valid_rows = device_rows < g.schedule_capacity
        ? device_rows
        : g.schedule_capacity;
    if (row >= valid_rows)
        return;
    const int peer_rank = g.schedule_peer_rank[row];
    const int peer_token_idx = g.schedule_peer_token_idx[row];
    if (peer_rank < 0 || peer_rank >= g.ep_size || peer_token_idx < 0
        || peer_token_idx >= g.num_local_tokens * g.topk)
        return;
    const int row_vectors =
        g.hidden_size * 2 / static_cast<int>(sizeof(uint4));
    const auto *src = reinterpret_cast<const uint4 *>(g.routed_y)
        + static_cast<size_t>(row) * row_vectors;
    auto *dst = reinterpret_cast<uint4 *>(g.combine_peer[peer_rank])
        + static_cast<size_t>(peer_token_idx) * row_vectors;
    for (int index = threadIdx.x; index < row_vectors;
         index += blockDim.x)
        dst[index] = src[index];
}

void check_common(const at::Tensor &schedule_peer_rank,
                  const at::Tensor &schedule_peer_token_idx,
                  const at::Tensor &num_tokens, int64_t capacity) {
    TORCH_CHECK(schedule_peer_rank.is_cuda()
                    && schedule_peer_token_idx.is_cuda()
                    && num_tokens.is_cuda(),
                "schedule tensors must be CUDA tensors");
    TORCH_CHECK(schedule_peer_rank.scalar_type() == at::kInt
                    && schedule_peer_token_idx.scalar_type() == at::kInt
                    && num_tokens.scalar_type() == at::kInt,
                "schedule tensors must be int32");
    TORCH_CHECK(schedule_peer_rank.is_contiguous()
                    && schedule_peer_token_idx.is_contiguous()
                    && num_tokens.is_contiguous(),
                "schedule tensors must be contiguous");
    TORCH_CHECK(schedule_peer_rank.numel() == capacity
                    && schedule_peer_token_idx.numel() == capacity
                    && num_tokens.numel() == 1,
                "schedule shapes do not match capacity");
}

void run_dispatch(
    const at::Tensor &peer_x, const at::Tensor &peer_scale,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens, const at::Tensor &tokens_per_expert,
    const at::Tensor &shared_x, const at::Tensor &shared_scale,
    const at::Tensor &shared_indices, const at::Tensor &reference_x,
    const at::Tensor &reference_scale,
    const at::Tensor &reference_indices, int64_t topk) {
    TORCH_CHECK(peer_x.is_cuda() && peer_x.scalar_type() == at::kByte
                    && peer_x.is_contiguous() && peer_x.dim() == 3
                    && peer_x.size(0) == kPeers,
                "peer_x must be contiguous CUDA bytes [4,L,H]");
    const int64_t local_tokens = peer_x.size(1);
    const int64_t hidden = peer_x.size(2);
    TORCH_CHECK(hidden > 0 && hidden % 128 == 0,
                "hidden must be a positive K128 multiple");
    const int64_t scales = hidden / 128;
    TORCH_CHECK(peer_scale.is_cuda()
                    && peer_scale.scalar_type() == at::kFloat
                    && peer_scale.is_contiguous()
                    && peer_scale.sizes()
                        == at::IntArrayRef({kPeers, local_tokens, scales}),
                "peer_scale must be contiguous CUDA FP32 [4,L,H/128]");
    const int64_t capacity = shared_x.size(0);
    TORCH_CHECK(shared_x.is_cuda() && shared_x.scalar_type() == at::kByte
                    && shared_x.is_contiguous() && shared_x.dim() == 2
                    && shared_x.size(1) == hidden
                    && reference_x.sizes() == shared_x.sizes()
                    && reference_x.scalar_type() == at::kByte
                    && reference_x.is_contiguous(),
                "dispatch outputs must be contiguous CUDA bytes [capacity,H]");
    TORCH_CHECK(shared_scale.is_cuda()
                    && shared_scale.scalar_type() == at::kFloat
                    && shared_scale.is_contiguous()
                    && shared_scale.sizes()
                        == at::IntArrayRef({capacity, scales})
                    && reference_scale.sizes() == shared_scale.sizes()
                    && reference_scale.scalar_type() == at::kFloat
                    && reference_scale.is_contiguous(),
                "scale outputs must be contiguous CUDA FP32 [capacity,H/128]");
    TORCH_CHECK(shared_indices.is_cuda()
                    && shared_indices.scalar_type() == at::kInt
                    && shared_indices.is_contiguous()
                    && shared_indices.numel() == capacity
                    && reference_indices.sizes() == shared_indices.sizes()
                    && reference_indices.scalar_type() == at::kInt
                    && reference_indices.is_contiguous(),
                "expert outputs must be contiguous CUDA int32 [capacity]");
    TORCH_CHECK(tokens_per_expert.is_cuda()
                    && tokens_per_expert.scalar_type() == at::kInt
                    && tokens_per_expert.is_contiguous()
                    && tokens_per_expert.numel() > 0
                    && tokens_per_expert.numel() <= kMaxExperts,
                "tokens_per_expert must be CUDA int32 [1..16]");
    TORCH_CHECK(topk > 0 && topk <= INT32_MAX,
                "topk must fit positive int32");
    check_common(schedule_peer_rank, schedule_peer_token_idx,
                 num_tokens, capacity);
    TORCH_CHECK(peer_x.device() == peer_scale.device()
                    && peer_x.device() == shared_x.device()
                    && peer_x.device() == shared_scale.device()
                    && peer_x.device() == shared_indices.device()
                    && peer_x.device() == reference_x.device()
                    && peer_x.device() == reference_scale.device()
                    && peer_x.device() == reference_indices.device()
                    && peer_x.device() == schedule_peer_rank.device()
                    && peer_x.device() == schedule_peer_token_idx.device()
                    && peer_x.device() == num_tokens.device()
                    && peer_x.device() == tokens_per_expert.device(),
                "all dispatch tensors must share one device");

    c10::cuda::CUDAGuard guard(peer_x.device());
    dispatch_globals shared{};
    dispatch_globals reference{};
    const auto *x_base = peer_x.data_ptr<uint8_t>();
    const auto *scale_base = peer_scale.data_ptr<float>();
    for (int peer = 0; peer < kPeers; ++peer) {
        shared.x_peer[peer] = reference.x_peer[peer] =
            x_base + static_cast<size_t>(peer) * local_tokens * hidden;
        shared.x_scale_peer[peer] = reference.x_scale_peer[peer] =
            scale_base + static_cast<size_t>(peer) * local_tokens * scales;
    }
    shared.routed_x = shared_x.data_ptr<uint8_t>();
    reference.routed_x = reference_x.data_ptr<uint8_t>();
    shared.routed_x_scale = shared_scale.data_ptr<float>();
    reference.routed_x_scale = reference_scale.data_ptr<float>();
    shared.m_indices = shared_indices.data_ptr<int>();
    reference.m_indices = reference_indices.data_ptr<int>();
    for (dispatch_globals *g : {&shared, &reference}) {
        g->schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
        g->schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
        g->num_tokens = num_tokens.data_ptr<int>();
        g->tokens_per_expert = tokens_per_expert.data_ptr<int>();
        g->ep_size = kPeers;
        g->num_local_tokens = static_cast<int>(local_tokens);
        g->hidden_size = static_cast<int>(hidden);
        g->scale_columns = static_cast<int>(scales);
        g->topk = static_cast<int>(topk);
        g->num_local_experts = static_cast<int>(tokens_per_expert.numel());
        g->schedule_capacity = static_cast<int>(capacity);
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(peer_x.get_device());
    shared_dispatch_kernel<<<(capacity + kWarps - 1) / kWarps,
                             kThreads, 0, stream>>>(shared);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "shared dispatch helper probe launch failed");
    legacy_dispatch_reference_kernel<<<capacity, kThreads, 0, stream>>>(
        reference);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "legacy dispatch reference launch failed");
}

void run_push(
    const at::Tensor &routed_y, const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens, const at::Tensor &shared_combine,
    const at::Tensor &reference_combine, const at::Tensor &route_receipt,
    int64_t topk) {
    TORCH_CHECK(routed_y.is_cuda()
                    && routed_y.scalar_type() == at::kBFloat16
                    && routed_y.is_contiguous() && routed_y.dim() == 2,
                "routed_y must be contiguous CUDA BF16 [capacity,H]");
    const int64_t capacity = routed_y.size(0);
    const int64_t hidden = routed_y.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 128 == 0,
                "hidden must be a positive K128 multiple");
    TORCH_CHECK(shared_combine.is_cuda()
                    && shared_combine.scalar_type() == at::kBFloat16
                    && shared_combine.is_contiguous()
                    && shared_combine.dim() == 3
                    && shared_combine.size(0) == kPeers
                    && shared_combine.size(2) == hidden
                    && reference_combine.sizes() == shared_combine.sizes()
                    && reference_combine.scalar_type() == at::kBFloat16
                    && reference_combine.is_contiguous(),
                "combine outputs must be CUDA BF16 [4,L*topk,H]");
    TORCH_CHECK(topk > 0 && shared_combine.size(1) % topk == 0,
                "combine route slots must be divisible by topk");
    TORCH_CHECK(route_receipt.is_cuda()
                    && route_receipt.scalar_type() == at::kInt
                    && route_receipt.is_contiguous()
                    && route_receipt.sizes()
                        == at::IntArrayRef({capacity, 4}),
                "route_receipt must be CUDA int32 [capacity,4]");
    check_common(schedule_peer_rank, schedule_peer_token_idx,
                 num_tokens, capacity);
    TORCH_CHECK(routed_y.device() == shared_combine.device()
                    && routed_y.device() == reference_combine.device()
                    && routed_y.device() == route_receipt.device()
                    && routed_y.device() == schedule_peer_rank.device()
                    && routed_y.device() == schedule_peer_token_idx.device()
                    && routed_y.device() == num_tokens.device(),
                "all push tensors must share one device");

    c10::cuda::CUDAGuard guard(routed_y.device());
    push_globals shared{};
    push_globals reference{};
    auto *shared_base = reinterpret_cast<uint8_t *>(
        shared_combine.data_ptr<at::BFloat16>());
    auto *reference_base = reinterpret_cast<uint8_t *>(
        reference_combine.data_ptr<at::BFloat16>());
    const size_t peer_bytes = static_cast<size_t>(shared_combine.size(1))
        * hidden * sizeof(__nv_bfloat16);
    for (int peer = 0; peer < kPeers; ++peer) {
        shared.combine_peer[peer] = shared_base + peer * peer_bytes;
        reference.combine_peer[peer] = reference_base + peer * peer_bytes;
    }
    shared.routed_y = reference.routed_y =
        reinterpret_cast<const uint8_t *>(
            routed_y.data_ptr<at::BFloat16>());
    for (push_globals *g : {&shared, &reference}) {
        g->schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
        g->schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
        g->num_tokens = num_tokens.data_ptr<int>();
        g->ep_size = kPeers;
        g->num_local_tokens =
            static_cast<int>(shared_combine.size(1) / topk);
        g->hidden_size = static_cast<int>(hidden);
        g->topk = static_cast<int>(topk);
        g->schedule_capacity = static_cast<int>(capacity);
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(routed_y.get_device());
    shared_push_kernel<<<(capacity + kWarps - 1) / kWarps,
                         kThreads, 0, stream>>>(
        shared, route_receipt.data_ptr<int>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "shared push helper probe launch failed");
    legacy_push_reference_kernel<<<capacity, kThreads, 0, stream>>>(reference);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "legacy push reference launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_dispatch", &run_dispatch);
    module.def("run_push", &run_push);
}
