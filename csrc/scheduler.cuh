#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

namespace scheduler {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1024;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
};

struct globals {
    using topk_gl = gl<int, 1, -1, -1, -1>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    topk_gl topk;                        // (world_size, num_local_tokens, topk)
    index_gl schedule_peer_rank;         // (schedule_capacity,) must be initialized to -1
    index_gl schedule_peer_token_idx;    // (schedule_capacity,) original_token_idx * topk + k
    index_gl num_tokens;                 // (1,) total padded token count, must be zero-initialized
    index_gl tokens_per_expert;          // (num_local_experts,) padded per-expert token counts
    index_gl tokens_per_expert_and_peer; // (num_local_experts * world_size,) per-(local_expert, peer_rank) token counts, must be zero-initialized

    int rank;                            // this (destination) rank
    int expert_padding;                  // row alignment required by the consumer
};

// Stage 1: Count the number of tokens routed from each peer rank to each local expert
static __device__ __forceinline__ void count_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_global_tokens = world_size * rank_stride;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;
    const int last_expert = first_expert + num_local_experts;

    extern __shared__ int tokens_per_expert_and_peer[]; // (num_local_experts, world_size)
    for (int i = threadIdx.x; i < G.tokens_per_expert_and_peer.cols(); i += blockDim.x)
        tokens_per_expert_and_peer[i] = 0;
    __syncthreads();

    const int grid_stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_global_tokens; idx += grid_stride) {
        const int peer_rank = idx / rank_stride;
        const int peer_token_idx = idx - peer_rank * rank_stride;
        const int expert_idx = G.topk[{peer_rank, peer_token_idx / topk, peer_token_idx % topk}];
        if (expert_idx >= first_expert && expert_idx < last_expert)
            atomicAdd(&tokens_per_expert_and_peer[(expert_idx - first_expert) * world_size + peer_rank], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < G.tokens_per_expert_and_peer.cols(); i += blockDim.x)
        if (tokens_per_expert_and_peer[i] != 0)
            atomicAdd(&G.tokens_per_expert_and_peer[{i}], tokens_per_expert_and_peer[i]);
}

// Stage 2: Pad each expert's total token count for the selected consumer.
static __device__ __forceinline__ void pad_kernel(const globals &G) {
    const int local_expert = blockIdx.x;
    const int world_size = G.topk.depth();
    int num_tokens = 0;
    for (int peer_rank = 0; peer_rank < world_size; ++peer_rank)
        num_tokens += G.tokens_per_expert_and_peer[{local_expert * world_size + peer_rank}];
    const int padded_num_tokens =
        (num_tokens + G.expert_padding - 1) / G.expert_padding
        * G.expert_padding;
    G.tokens_per_expert[{local_expert}] = padded_num_tokens;
    atomicAdd(&G.num_tokens[{0}], padded_num_tokens);
}

// Stage 3: Schedule each token into its expert's padded segment.
static __device__ __forceinline__ void schedule_kernel(const globals &G) {
    const int world_size = G.topk.depth();
    const int num_local_tokens = G.topk.rows();
    const int topk = G.topk.cols();
    const int rank_stride = num_local_tokens * topk;
    const int num_local_experts = G.tokens_per_expert.cols();
    const int first_expert = G.rank * num_local_experts;

    if (G.num_tokens[{0}] > G.schedule_peer_rank.cols()) asm volatile("{trap;}");

    extern __shared__ int tokens_per_peer_rank[]; // (world_size,) this expert's per-peer-rank counts
    __shared__ int cumulative_tokens_from_peer_rank[config::NUM_WARPS];

    for (int idx = blockIdx.x; idx < num_local_experts * world_size; idx += gridDim.x) {
        const int local_expert = idx / world_size;
        const int peer_rank = idx % world_size;

        // Base row of this expert's padded segment
        int expert_base = 0;
        for (int expert_idx = 0; expert_idx < local_expert; ++expert_idx) 
            expert_base += G.tokens_per_expert[{expert_idx}];

        for (int rank = threadIdx.x; rank < world_size; rank += blockDim.x)
            tokens_per_peer_rank[rank] = G.tokens_per_expert_and_peer[{local_expert * world_size + rank}];
        __syncthreads();

        // Step 1. Count the number of tokens routed from this peer rank to this expert
        int _tokens_from_peer_rank = 0;
        for (int peer_token_idx = threadIdx.x; peer_token_idx < rank_stride; peer_token_idx += blockDim.x) {
            const int expert_idx = G.topk[{peer_rank, peer_token_idx / topk, peer_token_idx % topk}];
            _tokens_from_peer_rank += (expert_idx - first_expert == local_expert) ? 1 : 0;
        }
        // Step 2. Cumulative sum within a warp: thread i's `inclusive` will have the sum from thread 0 to thread i
        int inclusive = _tokens_from_peer_rank;
        for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
            const int n = __shfl_up_sync(0xffffffff, inclusive, offset);
            if (warp::laneid() >= offset) inclusive += n;
        }
        if (warp::laneid() == WARP_THREADS - 1) cumulative_tokens_from_peer_rank[warpid()] = inclusive;
        __syncthreads();
        // Step 3: Cumulative sum across warps
        if (warpid() == 0) {
            int warp_total = (warp::laneid() < config::NUM_WARPS) ? cumulative_tokens_from_peer_rank[warp::laneid()] : 0;
            for (int offset = 1; offset < WARP_THREADS; offset *= 2) {
                const int n = __shfl_up_sync(0xffffffff, warp_total, offset);
                if (warp::laneid() >= offset) warp_total += n;
            }
            if (warp::laneid() < config::NUM_WARPS) cumulative_tokens_from_peer_rank[warp::laneid()] = warp_total;
        }
        __syncthreads();
        int j = (warpid() == 0 ? 0 : cumulative_tokens_from_peer_rank[warpid() - 1]) + inclusive - _tokens_from_peer_rank;

        for (int peer_token_idx = threadIdx.x; peer_token_idx < rank_stride; peer_token_idx += blockDim.x) {
            const int orig_token_idx = peer_token_idx / topk;
            const int expert_idx = G.topk[{peer_rank, orig_token_idx, peer_token_idx % topk}];
            if (expert_idx - first_expert == local_expert) {
                int dst_token_idx = expert_base;
                for (int rank = 0; rank < world_size; ++rank) {
                    const int num_tokens = tokens_per_peer_rank[rank];
                    dst_token_idx += min(num_tokens, j);
                    dst_token_idx += (rank < peer_rank && num_tokens > j) ? 1 : 0;
                }
                G.schedule_peer_rank[{dst_token_idx}] = peer_rank;
                G.schedule_peer_token_idx[{dst_token_idx}] = peer_token_idx; // original_token_idx * topk + k
                ++j;
            }
        }
        __syncthreads(); // before the next iteration reuses cumulative_tokens_from_peer_rank
    }
}

static __host__ void schedule_out(
    const at::Tensor &topk_all,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    const at::Tensor &tokens_per_expert_and_peer,
    const int rank,
    const int expert_padding
) {
    TORCH_CHECK(
        expert_padding == 64 || expert_padding == 128
            || expert_padding == 256,
        "expert_padding must be one of 64, 128, 256");
    TORCH_CHECK(
        topk_all.dim() == 3 && topk_all.is_cuda()
            && topk_all.scalar_type() == at::kInt
            && topk_all.is_contiguous(),
        "topk_all must be contiguous CUDA int32 [ep_size,T,topk]");
    const int world_size = static_cast<int>(topk_all.size(0));
    const int num_local_tokens = static_cast<int>(topk_all.size(1));
    const int topk = static_cast<int>(topk_all.size(2));
    const int num_local_experts = static_cast<int>(tokens_per_expert.numel());
    const int schedule_capacity = static_cast<int>(schedule_peer_rank.numel());
    TORCH_CHECK(
        world_size == 4 || world_size == 8 || world_size == 16
            || world_size == 32 || world_size == 64,
        "topk_all ep_size must be one of 4, 8, 16, 32, 64");
    TORCH_CHECK(num_local_tokens >= 256 && num_local_tokens % 256 == 0,
                "topk_all T must be at least 256 and divisible by 256");
    TORCH_CHECK(topk > 0 && topk <= 255, "topk must be in [1,255]");
    TORCH_CHECK(num_local_experts > 0, "num_local_experts must be positive");
    TORCH_CHECK(rank >= 0 && rank < world_size, "rank must be in [0,ep_size)");
    TORCH_CHECK(
        schedule_capacity > 0 && schedule_capacity % 256 == 0
            && schedule_capacity >= num_local_tokens * topk,
        "schedule capacity must be M256 aligned and hold local routes");
    const at::Tensor outputs[] = {
        schedule_peer_rank, schedule_peer_token_idx, num_tokens,
        tokens_per_expert, tokens_per_expert_and_peer,
    };
    for (const auto &tensor : outputs) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.device() == topk_all.device()
                && tensor.scalar_type() == at::kInt
                && tensor.is_contiguous(),
            "schedule outputs must be contiguous CUDA int32 tensors on the topk device");
    }
    TORCH_CHECK(schedule_peer_rank.dim() == 1,
                "schedule_peer_rank must be one-dimensional");
    TORCH_CHECK(
        schedule_peer_token_idx.dim() == 1
            && schedule_peer_token_idx.numel() == schedule_capacity,
        "schedule_peer_token_idx must match schedule capacity");
    TORCH_CHECK(num_tokens.dim() == 1 && num_tokens.numel() == 1,
                "num_tokens must have shape [1]");
    TORCH_CHECK(tokens_per_expert.dim() == 1,
                "tokens_per_expert must be one-dimensional");
    TORCH_CHECK(
        tokens_per_expert_and_peer.dim() == 1
            && tokens_per_expert_and_peer.numel()
                == num_local_experts * world_size,
        "tokens_per_expert_and_peer must have shape [E_local*ep_size]");

    c10::cuda::CUDAGuard device_guard(topk_all.device());
    auto stream = at::cuda::getCurrentCUDAStream(topk_all.get_device());
    CUDACHECK(cudaMemsetAsync(
        schedule_peer_rank.data_ptr<int>(), 0xff,
        schedule_capacity * sizeof(int), stream));
    CUDACHECK(cudaMemsetAsync(
        num_tokens.data_ptr<int>(), 0, sizeof(int), stream));
    CUDACHECK(cudaMemsetAsync(
        tokens_per_expert_and_peer.data_ptr<int>(), 0,
        num_local_experts * world_size * sizeof(int), stream));

    globals G {
        .topk = kittens::py::tensor_to_gl<globals::topk_gl>(topk_all),
        .schedule_peer_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<globals::index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert),
        .tokens_per_expert_and_peer =kittens::py::tensor_to_gl<globals::index_gl>(tokens_per_expert_and_peer),
        .rank = rank,
        .expert_padding = expert_padding,
    };

    kittens::py::global_kernel<config, globals, scheduler::count_kernel>
        <<<(G.topk.numel() + config::NUM_THREADS - 1) / config::NUM_THREADS, config::NUM_THREADS, num_local_experts * world_size * sizeof(int), stream>>>(G);
    kittens::py::global_kernel<config, globals, scheduler::pad_kernel>
        <<<num_local_experts, 1, 0, stream>>>(G);
    kittens::py::global_kernel<config, globals, scheduler::schedule_kernel>
        <<<num_local_experts * world_size, config::NUM_THREADS, world_size * sizeof(int), stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schedule(
    const at::Tensor &topk_all,
    const int num_local_experts,
    const int schedule_capacity,
    const int rank,
    const int expert_padding
) {
    at::Tensor schedule_peer_rank = at::empty(
        {schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor schedule_peer_token_idx = at::empty(
        {schedule_capacity}, topk_all.options().dtype(at::kInt));
    at::Tensor num_tokens = at::empty({1}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert = at::empty(
        {num_local_experts}, topk_all.options().dtype(at::kInt));
    at::Tensor tokens_per_expert_and_peer = at::empty(
        {num_local_experts * topk_all.size(0)},
        topk_all.options().dtype(at::kInt));
    schedule_out(
        topk_all, schedule_peer_rank, schedule_peer_token_idx, num_tokens,
        tokens_per_expert, tokens_per_expert_and_peer, rank, expert_padding);

    return {schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert};
}

} // namespace scheduler
