#pragma once

// Correctness-first communication primitives for the production Hopper FP8
// contract.  The source activations and their K128 scales live in symmetric
// memory on every EP rank.  Dispatch follows MoK's padded expert schedule and
// produces one contiguous, expert-major local buffer.  Combine sends BF16
// expert results back to the source rank/route slot.
//
// These kernels deliberately preserve the device-resident MoK schedule: they
// never read num_tokens on the host.  Dispatch keeps its one-CTA-per-row launch
// shape, but exits at the device-resident active-row count before touching the
// unused capacity tail.  This preserves CUDA Graph replay and the copy kernel's
// row parallelism without paying for unnecessary tail writes.
#if defined(KITTENS_SM90)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <vector>

namespace mok_sm90::fp8_block_routed {

constexpr int MAX_EP_SIZE = 64;
constexpr int THREADS = 256;

struct dispatch_globals {
    const uint8_t *x_peer[MAX_EP_SIZE];
    const float *x_scale_peer[MAX_EP_SIZE];
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

__global__ __launch_bounds__(THREADS, 1)
void dispatch_kernel(const __grid_constant__ dispatch_globals g) {
    const int row = blockIdx.x;
    if (row >= g.schedule_capacity)
        return;
    const int device_rows = g.num_tokens[0];
    const int valid_rows =
        device_rows < g.schedule_capacity ? device_rows : g.schedule_capacity;
    if (row >= valid_rows)
        return;
    const int peer_rank = g.schedule_peer_rank[row];
    const int peer_token_idx = g.schedule_peer_token_idx[row];
    const bool valid = peer_rank >= 0 && peer_rank < g.ep_size
                       && peer_token_idx >= 0
                       && peer_token_idx < g.num_local_tokens * g.topk;

    // hidden_size is K128 aligned, so every FP8 row is uint4 aligned.
    const int fp8_vectors = g.hidden_size / static_cast<int>(sizeof(uint4));
    auto *dst_vectors = reinterpret_cast<uint4 *>(g.routed_x)
                        + static_cast<size_t>(row) * fp8_vectors;
    const uint4 zero{0, 0, 0, 0};
    if (valid) {
        const int source_row = peer_token_idx / g.topk;
        const auto *src_vectors =
            reinterpret_cast<const uint4 *>(g.x_peer[peer_rank])
            + static_cast<size_t>(source_row) * fp8_vectors;
        for (int index = threadIdx.x; index < fp8_vectors;
             index += blockDim.x)
            dst_vectors[index] = src_vectors[index];
    } else {
        for (int index = threadIdx.x; index < fp8_vectors;
             index += blockDim.x)
            dst_vectors[index] = zero;
    }

    float *dst_scale = g.routed_x_scale
                       + static_cast<size_t>(row) * g.scale_columns;
    if (valid) {
        const int source_row = peer_token_idx / g.topk;
        const float *src_scale = g.x_scale_peer[peer_rank]
                                 + static_cast<size_t>(source_row)
                                       * g.scale_columns;
        for (int index = threadIdx.x; index < g.scale_columns;
             index += blockDim.x)
            dst_scale[index] = src_scale[index];
    } else {
        for (int index = threadIdx.x; index < g.scale_columns;
             index += blockDim.x)
            dst_scale[index] = 0.0f;
    }

    if (threadIdx.x == 0) {
        int expert = 0;
        int offset = 0;
        for (int candidate = 0; candidate < g.num_local_experts;
             ++candidate) {
            const int next = offset + g.tokens_per_expert[candidate];
            if (row < next) {
                expert = candidate;
                break;
            }
            offset = next;
        }
        // Expert segments and active-row count are M64 aligned. Rows past
        // valid_rows are never consumed by the dynamic grouped GEMM.
        g.m_indices[row] = expert;
    }
}

struct combine_globals {
    uint8_t *combine_peer[MAX_EP_SIZE];
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

__global__ __launch_bounds__(THREADS, 1)
void combine_kernel(const __grid_constant__ combine_globals g) {
    const int row = blockIdx.x;
    if (row >= g.schedule_capacity || row >= g.num_tokens[0])
        return;
    const int peer_rank = g.schedule_peer_rank[row];
    const int peer_token_idx = g.schedule_peer_token_idx[row];
    if (peer_rank < 0 || peer_rank >= g.ep_size || peer_token_idx < 0
        || peer_token_idx >= g.num_local_tokens * g.topk)
        return;

    // BF16 rows are also uint4 aligned because hidden_size is K128 aligned.
    const int row_bytes = g.hidden_size * 2;
    const int vectors = row_bytes / static_cast<int>(sizeof(uint4));
    const auto *src = reinterpret_cast<const uint4 *>(g.routed_y)
                      + static_cast<size_t>(row) * vectors;
    auto *dst = reinterpret_cast<uint4 *>(g.combine_peer[peer_rank])
                + static_cast<size_t>(peer_token_idx) * vectors;
    for (int index = threadIdx.x; index < vectors; index += blockDim.x)
        dst[index] = src[index];
}

inline void check_pointer_list(const std::vector<int64_t> &pointers,
                               const char *name) {
    TORCH_CHECK(pointers.size() == 4 || pointers.size() == 8
                    || pointers.size() == 16 || pointers.size() == 32
                    || pointers.size() == 64,
                name, " length must be one of 4, 8, 16, 32, 64");
    for (const int64_t pointer : pointers)
        TORCH_CHECK(pointer > 0, name, " must contain positive pointers");
}

inline void check_schedule(const at::Tensor &schedule_peer_rank,
                           const at::Tensor &schedule_peer_token_idx,
                           const at::Tensor &num_tokens,
                           const at::Tensor &tokens_per_expert,
                           int64_t schedule_capacity) {
    TORCH_CHECK(schedule_peer_rank.is_cuda()
                    && schedule_peer_token_idx.is_cuda()
                    && num_tokens.is_cuda() && tokens_per_expert.is_cuda(),
                "schedule tensors must be CUDA tensors");
    TORCH_CHECK(schedule_peer_rank.scalar_type() == at::kInt
                    && schedule_peer_token_idx.scalar_type() == at::kInt
                    && num_tokens.scalar_type() == at::kInt
                    && tokens_per_expert.scalar_type() == at::kInt,
                "schedule tensors must be int32");
    TORCH_CHECK(schedule_peer_rank.is_contiguous()
                    && schedule_peer_token_idx.is_contiguous()
                    && num_tokens.is_contiguous()
                    && tokens_per_expert.is_contiguous(),
                "schedule tensors must be contiguous");
    TORCH_CHECK(schedule_peer_rank.dim() == 1
                    && schedule_peer_rank.numel() == schedule_capacity
                    && schedule_peer_token_idx.sizes()
                           == schedule_peer_rank.sizes(),
                "schedule tensors must match schedule capacity");
    TORCH_CHECK(num_tokens.dim() == 1 && num_tokens.numel() == 1,
                "num_tokens must have shape [1]");
    TORCH_CHECK(tokens_per_expert.dim() == 1
                    && tokens_per_expert.numel() > 0,
                "tokens_per_expert must be a nonempty vector");
}

inline void check_combine_schedule(const at::Tensor &schedule_peer_rank,
                                   const at::Tensor &schedule_peer_token_idx,
                                   const at::Tensor &num_tokens,
                                   int64_t schedule_capacity) {
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
    TORCH_CHECK(schedule_peer_rank.dim() == 1
                    && schedule_peer_rank.numel() == schedule_capacity
                    && schedule_peer_token_idx.sizes()
                           == schedule_peer_rank.sizes(),
                "schedule tensors must match schedule capacity");
    TORCH_CHECK(num_tokens.dim() == 1 && num_tokens.numel() == 1,
                "num_tokens must have shape [1]");
}

inline void dispatch_out(
    const at::Tensor &x, const std::vector<int64_t> &x_ptrs,
    const at::Tensor &x_scale, const std::vector<int64_t> &x_scale_ptrs,
    const at::Tensor &routed_x, const at::Tensor &routed_x_scale,
    const at::Tensor &m_indices, const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx, const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert, int64_t topk) {
    TORCH_CHECK(x.dim() == 2 && x.is_cuda()
                    && x.scalar_type() == at::kFloat8_e4m3fn
                    && x.is_contiguous(),
                "x must be contiguous CUDA float8_e4m3fn [T,H]");
    const int64_t num_local_tokens = x.size(0);
    const int64_t hidden_size = x.size(1);
    TORCH_CHECK(num_local_tokens > 0 && hidden_size >= 128
                    && hidden_size % 128 == 0,
                "x dimensions must be positive and H must be K128 aligned");
    TORCH_CHECK(x_scale.dim() == 2 && x_scale.is_cuda()
                    && x_scale.scalar_type() == at::kFloat
                    && x_scale.is_contiguous()
                    && x_scale.size(0) == num_local_tokens
                    && x_scale.size(1) == hidden_size / 128,
                "x_scale must be contiguous CUDA float32 [T,H/128]");
    TORCH_CHECK(routed_x.dim() == 2 && routed_x.is_cuda()
                    && routed_x.scalar_type() == at::kFloat8_e4m3fn
                    && routed_x.is_contiguous()
                    && routed_x.size(1) == hidden_size,
                "routed_x must be contiguous CUDA float8_e4m3fn [capacity,H]");
    const int64_t schedule_capacity = routed_x.size(0);
    TORCH_CHECK(schedule_capacity > 0 && schedule_capacity % 64 == 0,
                "schedule capacity must be positive and M64 aligned");
    TORCH_CHECK(routed_x_scale.dim() == 2 && routed_x_scale.is_cuda()
                    && routed_x_scale.scalar_type() == at::kFloat
                    && routed_x_scale.is_contiguous()
                    && routed_x_scale.size(0) == schedule_capacity
                    && routed_x_scale.size(1) == hidden_size / 128,
                "routed_x_scale must be contiguous CUDA float32 [capacity,H/128]");
    TORCH_CHECK(m_indices.dim() == 1 && m_indices.is_cuda()
                    && m_indices.scalar_type() == at::kInt
                    && m_indices.is_contiguous()
                    && m_indices.numel() == schedule_capacity,
                "m_indices must be contiguous CUDA int32 [capacity]");
    TORCH_CHECK(topk > 0 && topk <= 255, "topk must be in [1,255]");
    check_pointer_list(x_ptrs, "x_ptrs");
    check_pointer_list(x_scale_ptrs, "x_scale_ptrs");
    TORCH_CHECK(x_ptrs.size() == x_scale_ptrs.size(),
                "x and scale pointer lists must have equal length");
    check_schedule(schedule_peer_rank, schedule_peer_token_idx, num_tokens,
                   tokens_per_expert, schedule_capacity);
    TORCH_CHECK(x.device() == x_scale.device()
                    && x.device() == routed_x.device()
                    && x.device() == routed_x_scale.device()
                    && x.device() == m_indices.device()
                    && x.device() == schedule_peer_rank.device()
                    && x.device() == schedule_peer_token_idx.device()
                    && x.device() == num_tokens.device()
                    && x.device() == tokens_per_expert.device(),
                "all local tensors must be on the same CUDA device");

    c10::cuda::CUDAGuard device_guard(x.device());

    dispatch_globals globals{};
    for (size_t rank = 0; rank < x_ptrs.size(); ++rank) {
        globals.x_peer[rank] =
            reinterpret_cast<const uint8_t *>(x_ptrs[rank]);
        globals.x_scale_peer[rank] =
            reinterpret_cast<const float *>(x_scale_ptrs[rank]);
    }
    globals.routed_x = reinterpret_cast<uint8_t *>(routed_x.data_ptr());
    globals.routed_x_scale = routed_x_scale.data_ptr<float>();
    globals.m_indices = m_indices.data_ptr<int>();
    globals.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    globals.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    globals.num_tokens = num_tokens.data_ptr<int>();
    globals.tokens_per_expert = tokens_per_expert.data_ptr<int>();
    globals.ep_size = static_cast<int>(x_ptrs.size());
    globals.num_local_tokens = static_cast<int>(num_local_tokens);
    globals.hidden_size = static_cast<int>(hidden_size);
    globals.scale_columns = static_cast<int>(hidden_size / 128);
    globals.topk = static_cast<int>(topk);
    globals.num_local_experts = static_cast<int>(tokens_per_expert.numel());
    globals.schedule_capacity = static_cast<int>(schedule_capacity);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    dispatch_kernel<<<schedule_capacity, THREADS, 0, stream>>>(globals);
    CUDACHECK(cudaGetLastError());
}

inline void combine_out(
    const at::Tensor &routed_y, const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx, const at::Tensor &num_tokens,
    int64_t topk) {
    TORCH_CHECK(routed_y.dim() == 2 && routed_y.is_cuda()
                    && routed_y.scalar_type() == at::kBFloat16
                    && routed_y.is_contiguous(),
                "routed_y must be contiguous CUDA bfloat16 [capacity,H]");
    const int64_t schedule_capacity = routed_y.size(0);
    const int64_t hidden_size = routed_y.size(1);
    TORCH_CHECK(schedule_capacity > 0 && schedule_capacity % 64 == 0
                    && hidden_size >= 128 && hidden_size % 128 == 0,
                "routed_y must have M64 capacity and K128 hidden size");
    TORCH_CHECK(combine_buffer.dim() == 2 && combine_buffer.is_cuda()
                    && combine_buffer.scalar_type() == at::kBFloat16
                    && combine_buffer.is_contiguous()
                    && combine_buffer.size(1) == hidden_size,
                "combine_buffer must be contiguous CUDA bfloat16 [T*topk,H]");
    TORCH_CHECK(topk > 0 && topk <= 255
                    && combine_buffer.size(0) % topk == 0,
                "topk must divide combine_buffer rows and be in [1,255]");
    check_pointer_list(combine_buffer_ptrs, "combine_buffer_ptrs");
    check_combine_schedule(schedule_peer_rank, schedule_peer_token_idx,
                           num_tokens, schedule_capacity);
    TORCH_CHECK(routed_y.device() == combine_buffer.device()
                    && routed_y.device() == schedule_peer_rank.device()
                    && routed_y.device() == schedule_peer_token_idx.device()
                    && routed_y.device() == num_tokens.device(),
                "all local tensors must be on the same CUDA device");

    combine_globals globals{};
    for (size_t rank = 0; rank < combine_buffer_ptrs.size(); ++rank)
        globals.combine_peer[rank] =
            reinterpret_cast<uint8_t *>(combine_buffer_ptrs[rank]);
    globals.routed_y = reinterpret_cast<const uint8_t *>(routed_y.data_ptr());
    globals.schedule_peer_rank = schedule_peer_rank.data_ptr<int>();
    globals.schedule_peer_token_idx = schedule_peer_token_idx.data_ptr<int>();
    globals.num_tokens = num_tokens.data_ptr<int>();
    globals.ep_size = static_cast<int>(combine_buffer_ptrs.size());
    globals.num_local_tokens =
        static_cast<int>(combine_buffer.size(0) / topk);
    globals.hidden_size = static_cast<int>(hidden_size);
    globals.topk = static_cast<int>(topk);
    globals.schedule_capacity = static_cast<int>(schedule_capacity);

    c10::cuda::CUDAGuard device_guard(routed_y.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(routed_y.get_device());
    combine_kernel<<<schedule_capacity, THREADS, 0, stream>>>(globals);
    CUDACHECK(cudaGetLastError());
}

}  // namespace mok_sm90::fp8_block_routed

#endif
