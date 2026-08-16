#pragma once

#if defined(KITTENS_SM90)

#include "sm90_fp8_block_routed.cuh"
#include "utils.cuh"

namespace mok_sm90::fp8_block_route_fused {

inline void check_barrier(
    const at::Tensor &barrier_buffer,
    const std::vector<int64_t> &barrier_buffer_ptrs,
    int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target, const at::Device &device) {
    TORCH_CHECK(
        barrier_buffer.is_cuda() && barrier_target.is_cuda()
            && barrier_buffer.device() == device
            && barrier_target.device() == device
            && barrier_buffer.scalar_type() == at::kInt
            && barrier_target.scalar_type() == at::kInt
            && barrier_buffer.is_contiguous()
            && barrier_target.is_contiguous()
            && barrier_buffer.dim() == 1 && barrier_buffer.numel() == 1
            && barrier_target.dim() == 1 && barrier_target.numel() == 1,
        "barrier tensors must be contiguous CUDA int32 [1] tensors on the route device");
    mok_sm90::fp8_block_routed::check_pointer_list(
        barrier_buffer_ptrs, "barrier_buffer_ptrs");
    TORCH_CHECK(barrier_buffer_multicast_ptr > 0,
                "barrier multicast pointer must be positive");
}

inline void dispatch_copy_out(
    const at::Tensor &x, const at::Tensor &x_buffer,
    const std::vector<int64_t> &x_buffer_ptrs,
    const at::Tensor &x_scale, const at::Tensor &x_scale_buffer,
    const std::vector<int64_t> &x_scale_buffer_ptrs,
    const at::Tensor &barrier_buffer,
    const std::vector<int64_t> &barrier_buffer_ptrs,
    int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target,
    const at::Tensor &routed_x, const at::Tensor &routed_x_scale,
    const at::Tensor &m_indices,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert, int64_t topk) {
    TORCH_CHECK(
        x.is_cuda() && x_buffer.is_cuda() && x.device() == x_buffer.device()
            && x.scalar_type() == at::kFloat8_e4m3fn
            && x_buffer.scalar_type() == at::kFloat8_e4m3fn
            && x.is_contiguous() && x_buffer.is_contiguous()
            && x.sizes() == x_buffer.sizes(),
        "x and x_buffer must be matching contiguous CUDA FP8 tensors");
    TORCH_CHECK(
        x_scale.is_cuda() && x_scale_buffer.is_cuda()
            && x_scale.device() == x.device()
            && x_scale_buffer.device() == x.device()
            && x_scale.scalar_type() == at::kFloat
            && x_scale_buffer.scalar_type() == at::kFloat
            && x_scale.is_contiguous() && x_scale_buffer.is_contiguous()
            && x_scale.sizes() == x_scale_buffer.sizes(),
        "x_scale and x_scale_buffer must be matching contiguous CUDA float32 tensors");
    check_barrier(
        barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, barrier_target, x.device());

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    CUDACHECK(cudaMemcpyAsync(
        x_buffer.data_ptr(), x.data_ptr(),
        static_cast<size_t>(x.numel() * x.element_size()),
        cudaMemcpyDeviceToDevice, stream));
    CUDACHECK(cudaMemcpyAsync(
        x_scale_buffer.data_ptr(), x_scale.data_ptr(),
        static_cast<size_t>(x_scale.numel() * x_scale.element_size()),
        cudaMemcpyDeviceToDevice, stream));
    utils::barrier_all::entrypoint(
        barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, barrier_target);

    if (routed_x.size(0) == 0) {
        TORCH_CHECK(
            routed_x.dim() == 2 && routed_x_scale.dim() == 2
                && m_indices.dim() == 1 && routed_x_scale.size(0) == 0
                && m_indices.numel() == 0,
            "empty dispatch outputs must consistently have zero rows");
        return;
    }
    mok_sm90::fp8_block_routed::dispatch_out(
        x_buffer, x_buffer_ptrs, x_scale_buffer, x_scale_buffer_ptrs,
        routed_x, routed_x_scale, m_indices, schedule_peer_rank,
        schedule_peer_token_idx, num_tokens, tokens_per_expert, topk);
}

inline void combine_reduce_out(
    const at::Tensor &routed_y, const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens, const at::Tensor &topk_weights,
    const at::Tensor &output,
    const at::Tensor &barrier_buffer,
    const std::vector<int64_t> &barrier_buffer_ptrs,
    int64_t barrier_buffer_multicast_ptr,
    const at::Tensor &barrier_target, int64_t topk) {
    TORCH_CHECK(topk > 0 && topk <= 255, "topk must be in [1,255]");
    TORCH_CHECK(
        output.dim() == 2 && output.is_cuda()
            && output.scalar_type() == at::kBFloat16
            && output.is_contiguous() && output.size(0) > 0
            && output.size(1) >= 128 && output.size(1) % 128 == 0,
        "output must be contiguous CUDA bfloat16 [T,H]");
    TORCH_CHECK(
        topk_weights.dim() == 2 && topk_weights.is_cuda()
            && topk_weights.device() == output.device()
            && topk_weights.scalar_type() == at::kFloat
            && topk_weights.is_contiguous()
            && topk_weights.size(0) == output.size(0)
            && topk_weights.size(1) == topk,
        "topk_weights must be contiguous CUDA float32 [T,topk]");
    TORCH_CHECK(
        combine_buffer.dim() == 2 && combine_buffer.is_cuda()
            && combine_buffer.device() == output.device()
            && combine_buffer.scalar_type() == at::kBFloat16
            && combine_buffer.is_contiguous()
            && combine_buffer.size(0) == output.size(0) * topk
            && combine_buffer.size(1) == output.size(1),
        "combine_buffer must be contiguous CUDA bfloat16 [T*topk,H]");
    TORCH_CHECK(
        routed_y.dim() == 2 && routed_y.is_cuda()
            && routed_y.device() == output.device()
            && routed_y.scalar_type() == at::kBFloat16
            && routed_y.is_contiguous()
            && routed_y.size(1) == output.size(1),
        "routed_y must be contiguous CUDA bfloat16 [M,H]");
    check_barrier(
        barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, barrier_target, output.device());

    c10::cuda::CUDAGuard device_guard(output.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(output.get_device());
    CUDACHECK(cudaMemsetAsync(
        combine_buffer.data_ptr(), 0,
        static_cast<size_t>(combine_buffer.numel()
                            * combine_buffer.element_size()),
        stream));
    utils::barrier_all::entrypoint(
        barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, barrier_target);
    if (routed_y.size(0) != 0) {
        mok_sm90::fp8_block_routed::combine_out(
            routed_y, combine_buffer, combine_buffer_ptrs,
            schedule_peer_rank, schedule_peer_token_idx, num_tokens, topk);
    }
    utils::barrier_all::entrypoint(
        barrier_buffer, barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr, barrier_target);
    utils::routed_epilogue_out(combine_buffer, topk_weights, output);
}

}  // namespace mok_sm90::fp8_block_route_fused

#endif
