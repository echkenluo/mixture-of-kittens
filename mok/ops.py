import math

import torch

from . import _C


def _sm90_reject(op_name: str) -> None:
    """Fail fast for SM90 operations that have not been ported."""
    import torch
    if torch.cuda.get_device_capability() == (9, 0):
        raise NotImplementedError(
            f"MoK SM90 port: {op_name} is not supported")



@torch.library.custom_op("mok::all_gather_top_experts", mutates_args=("all_gather_top_experts_buffer",))

def all_gather_top_experts(
    top_experts: torch.Tensor,
    all_gather_top_experts_buffer: torch.Tensor,
    all_gather_top_experts_buffer_multicast_ptr: int,
    rank: int,
    chunk_bytes: int,
) -> None:
    """All-gathers top-expert assignments across expert-parallel ranks.

    Inputs:
        top_experts:                                 int32 [num_local_tokens, topk]
        all_gather_top_experts_buffer:               int32 [ep_size, num_local_tokens, topk]
        all_gather_top_experts_buffer_multicast_ptr: int
        rank:                                        int
        chunk_bytes:                                 int

    Outputs:
        None
    """
    if (not top_experts.is_cuda or top_experts.dtype != torch.int32 or not top_experts.is_contiguous() or top_experts.ndim != 2):
        raise ValueError("top_experts must be contiguous CUDA int32 [num_local_tokens, topk]")
    if not all_gather_top_experts_buffer.is_cuda:
        raise ValueError("all_gather_top_experts_buffer must be a CUDA tensor")
    if all_gather_top_experts_buffer.dtype != torch.int32:
        raise TypeError("all_gather_top_experts_buffer must have dtype torch.int32")
    if not all_gather_top_experts_buffer.is_contiguous():
        raise ValueError("all_gather_top_experts_buffer must be contiguous")
    if all_gather_top_experts_buffer.ndim != 3:
        raise ValueError("all_gather_top_experts_buffer must have shape (ep_size, num_local_tokens, topk)")
    if any(size <= 0 for size in top_experts.shape):
        raise ValueError("top_experts dimensions must be positive")
    if any(size <= 0 for size in all_gather_top_experts_buffer.shape):
        raise ValueError("all_gather_top_experts_buffer dimensions must be positive")
    ep_size = all_gather_top_experts_buffer.shape[0]
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("all_gather_top_experts_buffer ep_size must be one of 4, 8, 16, 32, 64")
    if (all_gather_top_experts_buffer.device != top_experts.device
            or tuple(all_gather_top_experts_buffer.shape[1:]) != tuple(top_experts.shape)):
        raise ValueError("all_gather_top_experts_buffer must match top_experts shape and device")
    if type(all_gather_top_experts_buffer_multicast_ptr) is not int or all_gather_top_experts_buffer_multicast_ptr <= 0:
        raise TypeError("all_gather_top_experts_buffer_multicast_ptr must be a positive integer")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if chunk_bytes % 16 != 0:
        raise ValueError("chunk_bytes must be divisible by 16")
    rank_buffer_bytes = top_experts.numel() * top_experts.element_size()
    if rank_buffer_bytes % chunk_bytes != 0:
        raise ValueError("chunk_bytes must divide one rank's route-buffer bytes")

    _C.all_gather_top_experts(top_experts, all_gather_top_experts_buffer, all_gather_top_experts_buffer_multicast_ptr, rank, chunk_bytes)


@torch.library.custom_op("mok::barrier_all", mutates_args=("barrier_buffer", "target"))
def barrier_all(
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    target: torch.Tensor,
) -> None:
    """Synchronizes all expert-parallel ranks at device-side.

    Inputs:
        barrier_buffer:               int32 [1]
        barrier_buffer_ptrs:          list[int] [ep_size]
        barrier_buffer_multicast_ptr: int
        target:                       int32 [1]

    Outputs:
        None
    """
    if not barrier_buffer.is_cuda:
        raise ValueError("barrier_buffer must be a CUDA tensor")
    if barrier_buffer.dtype != torch.int32:
        raise TypeError("barrier_buffer must have dtype torch.int32")
    if not barrier_buffer.is_contiguous():
        raise ValueError("barrier_buffer must be contiguous")
    if tuple(barrier_buffer.shape) != (1,):
        raise ValueError("barrier_buffer must have shape (1,)")
    if not isinstance(barrier_buffer_ptrs, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in barrier_buffer_ptrs):
        raise TypeError("barrier_buffer_ptrs must be a list of positive integers")
    ep_size = len(barrier_buffer_ptrs)
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("barrier_buffer_ptrs length must be one of 4, 8, 16, 32, 64")
    if type(barrier_buffer_multicast_ptr) is not int or barrier_buffer_multicast_ptr <= 0:
        raise TypeError("barrier_buffer_multicast_ptr must be a positive integer")
    if (not target.is_cuda or target.device != barrier_buffer.device
            or target.dtype != torch.int32 or not target.is_contiguous()
            or tuple(target.shape) != (1,)):
        raise ValueError("target must be contiguous int32 [1] on the barrier CUDA device")

    _C.barrier_all(barrier_buffer, barrier_buffer_ptrs, barrier_buffer_multicast_ptr, target)


@torch.library.custom_op("mok::schedule", mutates_args=())
def schedule(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
    expert_padding: int = 256,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Builds the routed-token schedule for the current expert-parallel rank.

    Inputs:
        topk_all:          int32 [ep_size, num_local_tokens, topk]
        num_local_experts: int
        schedule_capacity: int
        rank:              int

    Outputs:
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
    """
    if topk_all.ndim != 3:
        raise ValueError("topk_all must have shape (ep_size, num_local_tokens, topk)")
    ep_size, num_local_tokens, topk = topk_all.shape
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("topk_all ep_size must be one of 4, 8, 16, 32, 64")
    if num_local_tokens < 256 or num_local_tokens % 256 != 0:
        raise ValueError(
            "topk_all num_local_tokens must be at least 256 and divisible by 256"
        )
    if not 0 < topk <= 255:
        raise ValueError("topk_all topk must be in [1, 255]")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if (type(schedule_capacity) is not int or schedule_capacity <= 0
            or schedule_capacity % 256 != 0):
        raise ValueError("schedule_capacity must be positive and divisible by 256")
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError("schedule_capacity must hold at least one rank's routed tokens")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")
    if type(expert_padding) is not int or expert_padding not in (64, 128, 256):
        raise ValueError("expert_padding must be one of 64, 128, 256")

    return _C.schedule(
        topk_all,
        num_local_experts,
        schedule_capacity,
        rank,
        expert_padding,
    )


@torch.library.custom_op(
    "mok::fp8_block_build_schedule_out",
    mutates_args=(
        "all_gather_buffer",
        "barrier_buffer",
        "barrier_target",
        "schedule_peer_rank",
        "schedule_peer_token_idx",
        "num_tokens",
        "tokens_per_expert",
        "tokens_per_expert_and_peer",
    ),
)
def fp8_block_build_schedule_out(
    top_experts: torch.Tensor,
    all_gather_buffer: torch.Tensor,
    all_gather_multicast_ptr: int,
    rank: int,
    chunk_bytes: int,
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    tokens_per_expert_and_peer: torch.Tensor,
    expert_padding: int,
) -> None:
    """All-gather and build an FP8 route schedule into caller storage."""
    if (
        not top_experts.is_cuda
        or top_experts.dtype != torch.int32
        or not top_experts.is_contiguous()
        or top_experts.ndim != 2
        or any(size <= 0 for size in top_experts.shape)
    ):
        raise ValueError("top_experts must be contiguous CUDA int32 [T,topk]")
    if (
        not all_gather_buffer.is_cuda
        or all_gather_buffer.device != top_experts.device
        or all_gather_buffer.dtype != torch.int32
        or not all_gather_buffer.is_contiguous()
        or all_gather_buffer.ndim != 3
        or tuple(all_gather_buffer.shape[1:]) != tuple(top_experts.shape)
    ):
        raise ValueError(
            "all_gather_buffer must be contiguous CUDA int32 [ep_size,T,topk]"
        )
    ep_size = all_gather_buffer.shape[0]
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("all_gather_buffer ep_size must be one of 4, 8, 16, 32, 64")
    if type(all_gather_multicast_ptr) is not int or all_gather_multicast_ptr <= 0:
        raise TypeError("all_gather_multicast_ptr must be a positive integer")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0,ep_size)")
    if (
        type(chunk_bytes) is not int
        or chunk_bytes <= 0
        or chunk_bytes % 16 != 0
        or top_experts.numel() * top_experts.element_size() % chunk_bytes != 0
    ):
        raise ValueError("chunk_bytes must be M16 and divide one rank's route bytes")
    _validate_pointer_list(barrier_buffer_ptrs, "barrier_buffer_ptrs")
    if len(barrier_buffer_ptrs) != ep_size:
        raise ValueError("barrier and all-gather EP sizes must match")
    if (
        type(barrier_buffer_multicast_ptr) is not int
        or barrier_buffer_multicast_ptr <= 0
    ):
        raise TypeError("barrier_buffer_multicast_ptr must be a positive integer")
    for name, tensor, shape in (
        ("barrier_buffer", barrier_buffer, (1,)),
        ("barrier_target", barrier_target, (1,)),
        ("num_tokens", num_tokens, (1,)),
    ):
        if (
            not tensor.is_cuda
            or tensor.device != top_experts.device
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA int32 with shape {shape}"
            )
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity <= 0 or schedule_capacity % 256 != 0:
        raise ValueError("schedule capacity must be positive and M256 aligned")
    num_local_experts = tokens_per_expert.numel()
    for name, tensor, shape in (
        ("schedule_peer_rank", schedule_peer_rank, (schedule_capacity,)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
        (
            "tokens_per_expert_and_peer",
            tokens_per_expert_and_peer,
            (num_local_experts * ep_size,),
        ),
    ):
        if (
            not tensor.is_cuda
            or tensor.device != top_experts.device
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA int32 with shape {shape}"
            )
    if num_local_experts <= 0:
        raise ValueError("tokens_per_expert must be nonempty")
    if type(expert_padding) is not int or expert_padding not in (64, 128, 256):
        raise ValueError("expert_padding must be one of 64, 128, 256")
    if not hasattr(_C, "fp8_block_build_schedule_out"):
        raise RuntimeError("the loaded MoK extension lacks fused FP8 scheduling")
    _C.fp8_block_build_schedule_out(
        top_experts,
        all_gather_buffer,
        all_gather_multicast_ptr,
        rank,
        chunk_bytes,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
        tokens_per_expert_and_peer,
        expert_padding,
    )


def _validate_pointer_list(pointers: list[int], name: str) -> None:
    if not isinstance(pointers, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in pointers
    ):
        raise TypeError(f"{name} must be a list of positive integers")
    if len(pointers) not in (4, 8, 16, 32, 64):
        raise ValueError(f"{name} length must be one of 4, 8, 16, 32, 64")


def _validate_fp8_route_schedule(
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    schedule_capacity: int,
) -> None:
    for name, tensor in (
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
    ):
        if not tensor.is_cuda or tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be a CUDA int32 tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if tuple(schedule_peer_rank.shape) != (schedule_capacity,):
        raise ValueError(
            "schedule_peer_rank must have shape (schedule_capacity,)"
        )
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError(
            "schedule_peer_token_idx must have shape (schedule_capacity,)"
        )
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")


@torch.library.custom_op(
    "mok::fp8_block_routed_dispatch_out",
    mutates_args=("routed_x", "routed_x_scale", "m_indices"),
)
def fp8_block_routed_dispatch_out(
    x: torch.Tensor,
    x_ptrs: list[int],
    x_scale: torch.Tensor,
    x_scale_ptrs: list[int],
    routed_x: torch.Tensor,
    routed_x_scale: torch.Tensor,
    m_indices: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
) -> None:
    """Dispatch production FP8/K128 rows into an expert-major buffer."""
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be contiguous CUDA [num_local_tokens, hidden_size]")
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("x must use torch.float8_e4m3fn")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens <= 0 or hidden_size < 128 or hidden_size % 128 != 0:
        raise ValueError("x dimensions must be positive and hidden_size K128 aligned")
    expected_scale_shape = (num_local_tokens, hidden_size // 128)
    if (
        not x_scale.is_cuda
        or x_scale.dtype != torch.float32
        or not x_scale.is_contiguous()
        or tuple(x_scale.shape) != expected_scale_shape
    ):
        raise ValueError(
            f"x_scale must be contiguous CUDA float32 {expected_scale_shape}"
        )
    if routed_x.ndim != 2 or routed_x.shape[1] != hidden_size:
        raise ValueError("routed_x must have shape (schedule_capacity, hidden_size)")
    schedule_capacity = routed_x.shape[0]
    if schedule_capacity <= 0 or schedule_capacity % 64 != 0:
        raise ValueError("schedule_capacity must be positive and divisible by 64")
    if (
        not routed_x.is_cuda
        or routed_x.dtype != torch.float8_e4m3fn
        or not routed_x.is_contiguous()
    ):
        raise ValueError("routed_x must be contiguous CUDA float8_e4m3fn")
    expected_routed_scale_shape = (schedule_capacity, hidden_size // 128)
    if (
        not routed_x_scale.is_cuda
        or routed_x_scale.dtype != torch.float32
        or not routed_x_scale.is_contiguous()
        or tuple(routed_x_scale.shape) != expected_routed_scale_shape
    ):
        raise ValueError(
            "routed_x_scale must be contiguous CUDA float32 "
            f"{expected_routed_scale_shape}"
        )
    if (
        not m_indices.is_cuda
        or m_indices.dtype != torch.int32
        or not m_indices.is_contiguous()
        or tuple(m_indices.shape) != (schedule_capacity,)
    ):
        raise ValueError(
            "m_indices must be contiguous CUDA int32 [schedule_capacity]"
        )
    if (
        not tokens_per_expert.is_cuda
        or tokens_per_expert.dtype != torch.int32
        or not tokens_per_expert.is_contiguous()
        or tokens_per_expert.ndim != 1
        or tokens_per_expert.numel() == 0
    ):
        raise ValueError("tokens_per_expert must be a nonempty CUDA int32 vector")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    _validate_pointer_list(x_ptrs, "x_ptrs")
    _validate_pointer_list(x_scale_ptrs, "x_scale_ptrs")
    if len(x_ptrs) != len(x_scale_ptrs):
        raise ValueError("x_ptrs and x_scale_ptrs must have equal length")
    _validate_fp8_route_schedule(
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        schedule_capacity,
    )
    tensors = (
        x_scale,
        routed_x,
        routed_x_scale,
        m_indices,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
    )
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all local FP8 dispatch tensors must share one device")
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        raise NotImplementedError("FP8 routed dispatch currently requires SM90")
    if not hasattr(_C, "fp8_block_routed_dispatch_out"):
        raise RuntimeError("the loaded MoK extension lacks FP8 routed dispatch")

    _C.fp8_block_routed_dispatch_out(
        x,
        x_ptrs,
        x_scale,
        x_scale_ptrs,
        routed_x,
        routed_x_scale,
        m_indices,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
        topk,
    )


@torch.library.custom_op(
    "mok::fp8_block_routed_combine_out",
    mutates_args=("combine_buffer",),
)
def fp8_block_routed_combine_out(
    routed_y: torch.Tensor,
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    topk: int,
) -> None:
    """Return routed BF16 rows to their source rank and route slot."""
    if routed_y.ndim != 2 or not routed_y.is_cuda or not routed_y.is_contiguous():
        raise ValueError("routed_y must be contiguous CUDA [capacity, hidden_size]")
    if routed_y.dtype != torch.bfloat16:
        raise TypeError("routed_y must use torch.bfloat16")
    schedule_capacity, hidden_size = routed_y.shape
    if (
        schedule_capacity <= 0
        or schedule_capacity % 64 != 0
        or hidden_size < 128
        or hidden_size % 128 != 0
    ):
        raise ValueError("routed_y must have M64 capacity and K128 hidden size")
    if (
        combine_buffer.ndim != 2
        or not combine_buffer.is_cuda
        or combine_buffer.dtype != torch.bfloat16
        or not combine_buffer.is_contiguous()
        or combine_buffer.shape[1] != hidden_size
    ):
        raise ValueError(
            "combine_buffer must be contiguous CUDA bfloat16 [T*topk,H]"
        )
    if (
        type(topk) is not int
        or not 0 < topk <= 255
        or combine_buffer.shape[0] % topk != 0
    ):
        raise ValueError("topk must divide combine_buffer rows and be in [1,255]")
    _validate_pointer_list(combine_buffer_ptrs, "combine_buffer_ptrs")
    _validate_fp8_route_schedule(
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        schedule_capacity,
    )
    tensors = (
        combine_buffer,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
    )
    if any(tensor.device != routed_y.device for tensor in tensors):
        raise ValueError("all local FP8 combine tensors must share one device")
    if torch.cuda.get_device_capability(routed_y.device) != (9, 0):
        raise NotImplementedError("FP8 routed combine currently requires SM90")
    if not hasattr(_C, "fp8_block_routed_combine_out"):
        raise RuntimeError("the loaded MoK extension lacks FP8 routed combine")

    _C.fp8_block_routed_combine_out(
        routed_y,
        combine_buffer,
        combine_buffer_ptrs,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        topk,
    )


@torch.library.custom_op(
    "mok::fp8_block_routed_dispatch_copy_out",
    mutates_args=(
        "x_buffer",
        "x_scale_buffer",
        "barrier_buffer",
        "barrier_target",
        "routed_x",
        "routed_x_scale",
        "m_indices",
    ),
)
def fp8_block_routed_dispatch_copy_out(
    x: torch.Tensor,
    x_buffer: torch.Tensor,
    x_buffer_ptrs: list[int],
    x_scale: torch.Tensor,
    x_scale_buffer: torch.Tensor,
    x_scale_buffer_ptrs: list[int],
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    routed_x: torch.Tensor,
    routed_x_scale: torch.Tensor,
    m_indices: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
) -> None:
    """Copy symmetric inputs, synchronize peers, and dispatch in one call."""
    if not hasattr(_C, "fp8_block_routed_dispatch_copy_out"):
        raise RuntimeError("the loaded MoK extension lacks fused FP8 dispatch")
    _C.fp8_block_routed_dispatch_copy_out(
        x,
        x_buffer,
        x_buffer_ptrs,
        x_scale,
        x_scale_buffer,
        x_scale_buffer_ptrs,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        routed_x,
        routed_x_scale,
        m_indices,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
        topk,
    )


@torch.library.custom_op(
    "mok::fp8_block_routed_combine_reduce_out",
    mutates_args=("combine_buffer", "output", "barrier_buffer", "barrier_target"),
)
def fp8_block_routed_combine_reduce_out(
    routed_y: torch.Tensor,
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    topk: int,
    combine_precleared: bool = False,
) -> None:
    """Combine, synchronize, and reduce, optionally reusing an earlier clear."""
    if type(combine_precleared) is not bool:
        raise TypeError("combine_precleared must be a bool")
    if not hasattr(_C, "fp8_block_routed_combine_reduce_out"):
        raise RuntimeError("the loaded MoK extension lacks fused FP8 combine")
    _C.fp8_block_routed_combine_reduce_out(
        routed_y,
        combine_buffer,
        combine_buffer_ptrs,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        topk_weights,
        output,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        topk,
        combine_precleared,
    )


@torch.library.custom_op(
    "mok::fp8_block_routed_combine_reduce_fused_out",
    mutates_args=(
        "combine_buffer",
        "output",
        "barrier_buffer",
        "barrier_target",
        "combine_completion",
        "barrier_expected_scratch",
    ),
)
def fp8_block_routed_combine_reduce_fused_out(
    routed_y: torch.Tensor,
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    topk: int,
    combine_completion: torch.Tensor,
    barrier_expected_scratch: torch.Tensor,
) -> None:
    """Precleared combine with the post-combine barrier fused in kernel.

    A separate op from the legacy 14-argument form because torch.library
    handles optional mutated tensors poorly (positional-index bookkeeping in
    ADInplaceOrView breaks when they arrive as keywords); the fused path
    always has both state tensors, so they are simply required here.
    """
    if not hasattr(_C, "fp8_block_routed_combine_reduce_out"):
        raise RuntimeError("the loaded MoK extension lacks fused FP8 combine")
    _C.fp8_block_routed_combine_reduce_out(
        routed_y,
        combine_buffer,
        combine_buffer_ptrs,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        topk_weights,
        output,
        barrier_buffer,
        barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr,
        barrier_target,
        topk,
        True,
        combine_completion,
        barrier_expected_scratch,
    )


@torch.library.custom_op(
    "mok::fp8_block_dispatch_gemm_fused_out",
    mutates_args=(
        "routed_x",
        "routed_x_scale",
        "m_indices",
        "barrier_buffer",
        "barrier_target",
        "input_expected_scratch",
        "tile_ready",
        "gate_up",
        "ticket_counter",
        "worker_ticket",
        "ticket_visit",
    ),
)
def fp8_block_dispatch_gemm_fused_out(
    x_buffer: torch.Tensor,
    x_ptrs: list[int],
    x_scale_buffer: torch.Tensor,
    x_scale_ptrs: list[int],
    routed_x: torch.Tensor,
    routed_x_scale: torch.Tensor,
    m_indices: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    barrier_buffer: torch.Tensor,
    barrier_buffer_multicast_ptr: int,
    barrier_target: torch.Tensor,
    input_expected_scratch: torch.Tensor,
    tile_ready: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    gate_up: torch.Tensor,
    ep_rank: int,
    ticket_counter: torch.Tensor,
    worker_ticket: torch.Tensor,
    trap_record_ptr: int,
    ticket_visit: torch.Tensor,
    copy_clusters: int = 8,
    forced_worker_clusters: int = 0,
    delay_ticket0_cycles: int = 0,
    spin_trap_iters: int = 0,
    record_visits: int = 0,
) -> None:
    """Input barrier, pull dispatch, and gate/up GEMM on a ticket-queue
    resident-worker grid (scheduling-order independent)."""
    if not hasattr(_C, "fp8_block_dispatch_gemm_fused_out"):
        raise RuntimeError(
            "the loaded MoK extension lacks the fused dispatch+GEMM kernel"
        )
    _C.fp8_block_dispatch_gemm_fused_out(
        x_buffer,
        x_ptrs,
        x_scale_buffer,
        x_scale_ptrs,
        routed_x,
        routed_x_scale,
        m_indices,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
        topk,
        barrier_buffer,
        barrier_buffer_multicast_ptr,
        barrier_target,
        input_expected_scratch,
        tile_ready,
        weight,
        weight_scale,
        gate_up,
        copy_clusters,
        ep_rank,
        ticket_counter,
        worker_ticket,
        trap_record_ptr,
        forced_worker_clusters,
        delay_ticket0_cycles,
        spin_trap_iters,
        ticket_visit,
        record_visits,
    )


@torch.library.custom_op(
    "mok::fp8_block_gemm_combine_fused_out",
    mutates_args=(
        "routed_y",
        "combine_buffer",
        "down_ready",
        "combine_completion",
        "barrier_target",
        "barrier_expected_scratch",
    ),
)
def fp8_block_gemm_combine_fused_out(
    down_input: torch.Tensor,
    down_input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    num_tokens: torch.Tensor,
    routed_y: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    topk: int,
    down_ready: torch.Tensor,
    combine_completion: torch.Tensor,
    barrier_target: torch.Tensor,
    barrier_expected_scratch: torch.Tensor,
    barrier_buffer_multicast_ptr: int,
) -> None:
    """Down GEMM with last-arriver combine push in one persistent kernel."""
    if not hasattr(_C, "fp8_block_gemm_combine_fused_out"):
        raise RuntimeError(
            "the loaded MoK extension lacks the fused GEMM+combine kernel"
        )
    _C.fp8_block_gemm_combine_fused_out(
        down_input,
        down_input_scale,
        weight,
        weight_scale,
        m_indices,
        num_tokens,
        routed_y,
        schedule_peer_rank,
        schedule_peer_token_idx,
        combine_buffer,
        combine_buffer_ptrs,
        topk,
        down_ready,
        combine_completion,
        barrier_target,
        barrier_expected_scratch,
        barrier_buffer_multicast_ptr,
    )


@torch.library.custom_op(
    "mok::routed_epilogue_fused_out",
    mutates_args=("output", "in_use", "epilogue_done"),
)
def routed_epilogue_fused_out(
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_buffer: torch.Tensor,
    barrier_expected_scratch: torch.Tensor,
    in_use: torch.Tensor,
    epilogue_done: torch.Tensor,
    trap_record_ptr: int,
    ep_rank: int,
    do_release: int = 0,
) -> None:
    """Routed epilogue: fused-barrier spin head (timeout-trapped); when
    do_release is set the last CTA performs the lease release chain."""
    if not hasattr(_C, "routed_epilogue_fused_out"):
        raise RuntimeError(
            "the loaded MoK extension lacks the fused-wait epilogue"
        )
    _C.routed_epilogue_fused_out(
        combine_buffer,
        topk_weights,
        output,
        barrier_buffer,
        barrier_expected_scratch,
        in_use,
        epilogue_done,
        trap_record_ptr,
        ep_rank,
        do_release,
    )


def fp8_block_dispatch_gemm_prewarm(device_index: int) -> int:
    """Warm the K1 occupancy cache for the given device (host-only; call at
    workspace creation, never inside a CUDA graph capture).  Returns the
    cudaOccupancyMaxActiveClusters value for the acceptance record."""
    if hasattr(_C, "fp8_block_dispatch_gemm_prewarm"):
        return int(_C.fp8_block_dispatch_gemm_prewarm(device_index))
    return -1


def require_fp8_block_megakernel() -> None:
    """Fail closed unless the complete terminal production API is loaded."""
    required = (
        "fp8_block_build_schedule_out",
        "fp8_block_megakernel_prewarm",
        "fp8_block_megakernel_prepare_out",
        "fp8_block_megakernel_out",
        "mok_workspace_lease_acquire",
    )
    missing = [name for name in required if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            "the loaded MoK extension lacks terminal FP8 megakernel APIs: "
            + ", ".join(missing)
        )


def fp8_block_megakernel_prewarm(
    device_index: int, comm_clusters: int = 1
) -> int:
    """Warm the terminal megakernel occupancy cache outside graph capture.

    Unlike the legacy K1 helper, the terminal path has no unsupported-op
    fallback: all three production entry points must be present before a
    graph-stable workspace can be created.  Returns the maximum number of
    compute clusters after reserving ``comm_clusters`` communication roles.
    """
    if type(device_index) is not int or device_index < 0:
        raise ValueError("device_index must be a nonnegative integer")
    if type(comm_clusters) is not int or comm_clusters <= 0:
        raise ValueError("comm_clusters must be a positive integer")
    require_fp8_block_megakernel()
    maximum = int(
        _C.fp8_block_megakernel_prewarm(device_index, comm_clusters)
    )
    if maximum <= 0:
        raise RuntimeError(
            "terminal FP8 megakernel occupancy prewarm returned no compute "
            "clusters"
        )
    return maximum


def _terminal_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_cuda
        or tensor.device != device
        or tensor.dtype != dtype
        or not tensor.is_contiguous()
        or tuple(tensor.shape) != shape
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA {dtype} with shape {shape} "
            f"on {device}"
        )


def _terminal_scalar(
    name: str, tensor: torch.Tensor, *, device: torch.device
) -> None:
    _terminal_tensor(
        name, tensor, device=device, dtype=torch.int32, shape=(1,)
    )


def _terminal_peer_ptrs(name: str, pointers: list[int]) -> None:
    if (
        not isinstance(pointers, list)
        or len(pointers) != 4
        or any(type(pointer) is not int or pointer <= 0 for pointer in pointers)
    ):
        raise ValueError(f"{name} must contain exactly four positive pointers")


@torch.library.custom_op(
    "mok::fp8_block_megakernel_prepare_out",
    mutates_args=(
        "route_ready",
        "x_routed_ready",
        "gate_up_tile_ready",
        "hidden_row_block_ready",
        "y_routed_ready",
        "y_routed_done",
        "epilogue_claim",
        "next_logical_cluster",
        "next_reduce_probe",
        "role_cursor",
        "cluster_role",
        "dispatch_tile_cursor",
        "dispatch_tiles_done",
        "push_tile_cursor",
        "worker_ticket",
        "comm_owner",
        "comm_worker_ticket",
        "producer_done",
        "comm_closed",
        "push_done",
        "reduce_done",
        "terminate",
        "epilogue_done",
        "input_expected_scratch",
    ),
)
def fp8_block_megakernel_prepare_out(
    topk_ids: torch.Tensor,
    route_ready: torch.Tensor,
    x_routed_ready: torch.Tensor,
    gate_up_tile_ready: torch.Tensor,
    hidden_row_block_ready: torch.Tensor,
    y_routed_ready: torch.Tensor,
    y_routed_done: torch.Tensor,
    epilogue_claim: torch.Tensor,
    next_logical_cluster: torch.Tensor,
    next_reduce_probe: torch.Tensor,
    role_cursor: torch.Tensor,
    cluster_role: torch.Tensor,
    dispatch_tile_cursor: torch.Tensor,
    dispatch_tiles_done: torch.Tensor,
    push_tile_cursor: torch.Tensor,
    worker_ticket: torch.Tensor,
    comm_owner: torch.Tensor,
    comm_worker_ticket: torch.Tensor,
    producer_done: torch.Tensor,
    comm_closed: torch.Tensor,
    push_done: torch.Tensor,
    reduce_done: torch.Tensor,
    terminate: torch.Tensor,
    epilogue_done: torch.Tensor,
    input_expected_scratch: torch.Tensor,
) -> None:
    """Prepare one terminal forward after its workspace lease is acquired."""
    if not hasattr(_C, "fp8_block_megakernel_prepare_out"):
        raise RuntimeError(
            "the loaded MoK extension lacks terminal megakernel prepare"
        )
    if (
        not isinstance(topk_ids, torch.Tensor)
        or not topk_ids.is_cuda
        or topk_ids.dtype != torch.int32
        or not topk_ids.is_contiguous()
        or topk_ids.ndim != 2
        or topk_ids.shape[0] <= 0
        or topk_ids.shape[1] != 6
    ):
        raise ValueError("topk_ids must be contiguous CUDA int32 [T,6]")
    device = topk_ids.device
    local_tokens = topk_ids.shape[0]
    if (
        not route_ready.is_cuda
        or route_ready.device != device
        or route_ready.dtype != torch.int32
        or not route_ready.is_contiguous()
        or route_ready.ndim != 2
        or route_ready.shape[0] < local_tokens
        or route_ready.shape[0] % 64 != 0
        or route_ready.shape[1] != 6
    ):
        raise ValueError(
            "route_ready must be contiguous CUDA int32 [padded_T,6], "
            "where padded_T >= T and is divisible by 64"
        )
    padded_tokens = route_ready.shape[0]
    if (
        not x_routed_ready.is_cuda
        or x_routed_ready.device != device
        or x_routed_ready.dtype != torch.int32
        or not x_routed_ready.is_contiguous()
        or x_routed_ready.ndim != 1
        or x_routed_ready.numel() <= 0
    ):
        raise ValueError("x_routed_ready must be contiguous CUDA int32 [M_tiles]")
    m_tiles = x_routed_ready.numel()
    _terminal_tensor(
        "gate_up_tile_ready",
        gate_up_tile_ready,
        device=device,
        dtype=torch.int32,
        shape=(m_tiles, 16),
    )
    for name, tensor in (
        ("hidden_row_block_ready", hidden_row_block_ready),
        ("y_routed_ready", y_routed_ready),
        ("y_routed_done", y_routed_done),
    ):
        _terminal_tensor(
            name, tensor, device=device, dtype=torch.int32, shape=(m_tiles,)
        )
    _terminal_tensor(
        "epilogue_claim",
        epilogue_claim,
        device=device,
        dtype=torch.int32,
        shape=(padded_tokens,),
    )
    if (
        not worker_ticket.is_cuda
        or worker_ticket.device != device
        or worker_ticket.dtype != torch.int32
        or not worker_ticket.is_contiguous()
        or worker_ticket.ndim != 1
        or worker_ticket.numel() <= 0
    ):
        raise ValueError(
            "worker_ticket must be contiguous CUDA int32 [compute_clusters]"
        )
    if (
        not cluster_role.is_cuda
        or cluster_role.device != device
        or cluster_role.dtype != torch.int32
        or not cluster_role.is_contiguous()
        or cluster_role.ndim != 1
        or cluster_role.numel() <= worker_ticket.numel()
    ):
        raise ValueError("cluster_role must be contiguous CUDA int32 [C+N]")
    if (
        not comm_worker_ticket.is_cuda
        or comm_worker_ticket.device != device
        or comm_worker_ticket.dtype != torch.int32
        or not comm_worker_ticket.is_contiguous()
        or comm_worker_ticket.ndim != 1
        or comm_worker_ticket.numel() <= 0
        or cluster_role.numel()
        != worker_ticket.numel() + comm_worker_ticket.numel()
    ):
        raise ValueError(
            "comm_worker_ticket and worker_ticket must partition cluster_role"
        )
    for name, tensor in (
        ("next_logical_cluster", next_logical_cluster),
        ("next_reduce_probe", next_reduce_probe),
        ("role_cursor", role_cursor),
        ("dispatch_tile_cursor", dispatch_tile_cursor),
        ("dispatch_tiles_done", dispatch_tiles_done),
        ("push_tile_cursor", push_tile_cursor),
        ("comm_owner", comm_owner),
        ("producer_done", producer_done),
        ("comm_closed", comm_closed),
        ("push_done", push_done),
        ("reduce_done", reduce_done),
        ("terminate", terminate),
        ("epilogue_done", epilogue_done),
        ("input_expected_scratch", input_expected_scratch),
    ):
        _terminal_scalar(name, tensor, device=device)
    _C.fp8_block_megakernel_prepare_out(
        topk_ids,
        route_ready,
        x_routed_ready,
        gate_up_tile_ready,
        hidden_row_block_ready,
        y_routed_ready,
        y_routed_done,
        epilogue_claim,
        next_logical_cluster,
        next_reduce_probe,
        role_cursor,
        cluster_role,
        dispatch_tile_cursor,
        dispatch_tiles_done,
        push_tile_cursor,
        worker_ticket,
        comm_owner,
        comm_worker_ticket,
        producer_done,
        comm_closed,
        push_done,
        reduce_done,
        terminate,
        epilogue_done,
        input_expected_scratch,
    )


@torch.library.custom_op(
    "mok::fp8_block_megakernel_out",
    mutates_args=(
        "routed_x",
        "routed_x_scale",
        "m_indices",
        "gate_up",
        "down_input",
        "down_input_scale",
        "routed_y",
        "combine_buffer",
        "route_ready",
        "output",
        "x_routed_ready",
        "gate_up_tile_ready",
        "hidden_row_block_ready",
        "y_routed_ready",
        "y_routed_done",
        "epilogue_claim",
        "next_logical_cluster",
        "next_reduce_probe",
        "role_cursor",
        "cluster_role",
        "dispatch_tile_cursor",
        "dispatch_tiles_done",
        "push_tile_cursor",
        "worker_ticket",
        "comm_owner",
        "comm_worker_ticket",
        "producer_done",
        "comm_closed",
        "push_done",
        "reduce_done",
        "terminate",
        "epilogue_done",
        "in_use",
        "barrier_buffer",
        "barrier_target",
        "input_expected_scratch",
    ),
)
def fp8_block_megakernel_out(
    x_buffer: torch.Tensor,
    x_ptrs: list[int],
    x_scale_buffer: torch.Tensor,
    x_scale_ptrs: list[int],
    routed_x: torch.Tensor,
    routed_x_scale: torch.Tensor,
    m_indices: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    gate_up: torch.Tensor,
    down_input: torch.Tensor,
    down_input_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    routed_y: torch.Tensor,
    combine_buffer: torch.Tensor,
    combine_ptrs: list[int],
    route_ready: torch.Tensor,
    route_ready_ptrs: list[int],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    x_routed_ready: torch.Tensor,
    gate_up_tile_ready: torch.Tensor,
    hidden_row_block_ready: torch.Tensor,
    y_routed_ready: torch.Tensor,
    y_routed_done: torch.Tensor,
    epilogue_claim: torch.Tensor,
    next_logical_cluster: torch.Tensor,
    next_reduce_probe: torch.Tensor,
    role_cursor: torch.Tensor,
    cluster_role: torch.Tensor,
    dispatch_tile_cursor: torch.Tensor,
    dispatch_tiles_done: torch.Tensor,
    push_tile_cursor: torch.Tensor,
    worker_ticket: torch.Tensor,
    comm_owner: torch.Tensor,
    comm_worker_ticket: torch.Tensor,
    producer_done: torch.Tensor,
    comm_closed: torch.Tensor,
    push_done: torch.Tensor,
    reduce_done: torch.Tensor,
    terminate: torch.Tensor,
    epilogue_done: torch.Tensor,
    in_use: torch.Tensor,
    barrier_buffer: torch.Tensor,
    barrier_target: torch.Tensor,
    input_expected_scratch: torch.Tensor,
    barrier_multicast_ptr: int,
    trap_record_ptr: int,
    ep_rank: int,
    comm_clusters: int,
    compute_clusters: int,
    minibatch_rows: int,
    macrobatch_rows: int,
    swiglu_limit: float,
    spin_limit: int,
) -> None:
    """Execute the fixed EP4/H4096/I2048/top-6 terminal megakernel."""
    if not hasattr(_C, "fp8_block_megakernel_out"):
        raise RuntimeError("the loaded MoK extension lacks terminal megakernel")
    if (
        not isinstance(x_buffer, torch.Tensor)
        or not x_buffer.is_cuda
        or x_buffer.dtype != torch.float8_e4m3fn
        or not x_buffer.is_contiguous()
        or x_buffer.ndim != 2
        or x_buffer.shape[0] <= 0
        or x_buffer.shape[1] != 4096
    ):
        raise ValueError("x_buffer must be contiguous CUDA float8_e4m3fn [T,4096]")
    device = x_buffer.device
    local_tokens = x_buffer.shape[0]
    _terminal_peer_ptrs("x_ptrs", x_ptrs)
    _terminal_peer_ptrs("x_scale_ptrs", x_scale_ptrs)
    _terminal_peer_ptrs("combine_ptrs", combine_ptrs)
    _terminal_peer_ptrs("route_ready_ptrs", route_ready_ptrs)
    _terminal_tensor(
        "x_scale_buffer",
        x_scale_buffer,
        device=device,
        dtype=torch.float32,
        shape=(local_tokens, 32),
    )
    if (
        not routed_x.is_cuda
        or routed_x.device != device
        or routed_x.dtype != torch.float8_e4m3fn
        or not routed_x.is_contiguous()
        or routed_x.ndim != 2
        or routed_x.shape[0] <= 0
        or routed_x.shape[0] % 64 != 0
        or routed_x.shape[1] != 4096
    ):
        raise ValueError(
            "routed_x must be contiguous CUDA float8_e4m3fn "
            "[schedule_capacity,4096] with M divisible by 64"
        )
    schedule_capacity = routed_x.shape[0]
    m_tiles = schedule_capacity // 64
    _terminal_tensor(
        "routed_x_scale",
        routed_x_scale,
        device=device,
        dtype=torch.float32,
        shape=(schedule_capacity, 32),
    )
    for name, tensor in (
        ("m_indices", m_indices),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
    ):
        _terminal_tensor(
            name,
            tensor,
            device=device,
            dtype=torch.int32,
            shape=(schedule_capacity,),
        )
    _terminal_scalar("num_tokens", num_tokens, device=device)
    if (
        not tokens_per_expert.is_cuda
        or tokens_per_expert.device != device
        or tokens_per_expert.dtype != torch.int32
        or not tokens_per_expert.is_contiguous()
        or tokens_per_expert.ndim != 1
        or not 1 <= tokens_per_expert.numel() <= 256
    ):
        raise ValueError(
            "tokens_per_expert must be contiguous CUDA int32 [E], 1 <= E <= 256"
        )
    experts = tokens_per_expert.numel()
    _terminal_tensor(
        "w13",
        w13,
        device=device,
        dtype=torch.float8_e4m3fn,
        shape=(experts, 4096, 4096),
    )
    _terminal_tensor(
        "w13_scale",
        w13_scale,
        device=device,
        dtype=torch.float32,
        shape=(experts, 32, 32),
    )
    _terminal_tensor(
        "gate_up",
        gate_up,
        device=device,
        dtype=torch.bfloat16,
        shape=(schedule_capacity, 4096),
    )
    _terminal_tensor(
        "down_input",
        down_input,
        device=device,
        dtype=torch.float8_e4m3fn,
        shape=(schedule_capacity, 2048),
    )
    _terminal_tensor(
        "down_input_scale",
        down_input_scale,
        device=device,
        dtype=torch.float32,
        shape=(schedule_capacity, 16),
    )
    _terminal_tensor(
        "w2",
        w2,
        device=device,
        dtype=torch.float8_e4m3fn,
        shape=(experts, 4096, 2048),
    )
    _terminal_tensor(
        "w2_scale",
        w2_scale,
        device=device,
        dtype=torch.float32,
        shape=(experts, 32, 16),
    )
    _terminal_tensor(
        "routed_y",
        routed_y,
        device=device,
        dtype=torch.bfloat16,
        shape=(schedule_capacity, 4096),
    )
    if (
        not route_ready.is_cuda
        or route_ready.device != device
        or route_ready.dtype != torch.int32
        or not route_ready.is_contiguous()
        or route_ready.ndim != 2
        or route_ready.shape[0] < local_tokens
        or route_ready.shape[0] % 64 != 0
        or route_ready.shape[1] != 6
    ):
        raise ValueError(
            "route_ready must be contiguous CUDA int32 [padded_T,6]"
        )
    padded_tokens = route_ready.shape[0]
    _terminal_tensor(
        "combine_buffer",
        combine_buffer,
        device=device,
        dtype=torch.bfloat16,
        shape=(padded_tokens * 6, 4096),
    )
    _terminal_tensor(
        "topk_weights",
        topk_weights,
        device=device,
        dtype=torch.float32,
        shape=(local_tokens, 6),
    )
    _terminal_tensor(
        "topk_ids",
        topk_ids,
        device=device,
        dtype=torch.int32,
        shape=(local_tokens, 6),
    )
    _terminal_tensor(
        "output",
        output,
        device=device,
        dtype=torch.bfloat16,
        shape=(local_tokens, 4096),
    )
    _terminal_tensor(
        "x_routed_ready",
        x_routed_ready,
        device=device,
        dtype=torch.int32,
        shape=(m_tiles,),
    )
    _terminal_tensor(
        "gate_up_tile_ready",
        gate_up_tile_ready,
        device=device,
        dtype=torch.int32,
        shape=(m_tiles, 16),
    )
    for name, tensor in (
        ("hidden_row_block_ready", hidden_row_block_ready),
        ("y_routed_ready", y_routed_ready),
        ("y_routed_done", y_routed_done),
    ):
        _terminal_tensor(
            name, tensor, device=device, dtype=torch.int32, shape=(m_tiles,)
        )
    _terminal_tensor(
        "epilogue_claim",
        epilogue_claim,
        device=device,
        dtype=torch.int32,
        shape=(padded_tokens,),
    )
    if (
        type(comm_clusters) is not int
        or comm_clusters <= 0
        or comm_clusters > (1 << 31) - 1
    ):
        raise ValueError("comm_clusters must be a positive int32")
    if (
        type(compute_clusters) is not int
        or compute_clusters <= 0
        or compute_clusters > (1 << 31) - 1
    ):
        raise ValueError("compute_clusters must be a positive int32")
    _terminal_tensor(
        "cluster_role",
        cluster_role,
        device=device,
        dtype=torch.int32,
        shape=(comm_clusters + compute_clusters,),
    )
    _terminal_tensor(
        "worker_ticket",
        worker_ticket,
        device=device,
        dtype=torch.int32,
        shape=(compute_clusters,),
    )
    _terminal_tensor(
        "comm_worker_ticket",
        comm_worker_ticket,
        device=device,
        dtype=torch.int32,
        shape=(comm_clusters,),
    )
    for name, tensor in (
        ("next_logical_cluster", next_logical_cluster),
        ("next_reduce_probe", next_reduce_probe),
        ("role_cursor", role_cursor),
        ("dispatch_tile_cursor", dispatch_tile_cursor),
        ("dispatch_tiles_done", dispatch_tiles_done),
        ("push_tile_cursor", push_tile_cursor),
        ("comm_owner", comm_owner),
        ("producer_done", producer_done),
        ("comm_closed", comm_closed),
        ("push_done", push_done),
        ("reduce_done", reduce_done),
        ("terminate", terminate),
        ("epilogue_done", epilogue_done),
        ("in_use", in_use),
        ("barrier_buffer", barrier_buffer),
        ("barrier_target", barrier_target),
        ("input_expected_scratch", input_expected_scratch),
    ):
        _terminal_scalar(name, tensor, device=device)
    if type(barrier_multicast_ptr) is not int or barrier_multicast_ptr <= 0:
        raise ValueError("barrier_multicast_ptr must be a positive integer")
    if type(trap_record_ptr) is not int or trap_record_ptr <= 0:
        raise ValueError("trap_record_ptr must be a positive integer")
    if type(ep_rank) is not int or not 0 <= ep_rank < 4:
        raise ValueError("ep_rank must be an integer in [0,4)")
    if (
        type(minibatch_rows) is not int
        or minibatch_rows <= 0
        or minibatch_rows > (1 << 31) - 1
        or minibatch_rows % 64 != 0
    ):
        raise ValueError("minibatch_rows must be positive and divisible by 64")
    if (
        type(macrobatch_rows) is not int
        or macrobatch_rows <= 0
        or macrobatch_rows > (1 << 31) - 1
        or macrobatch_rows % minibatch_rows != 0
    ):
        raise ValueError(
            "macrobatch_rows must be positive and divisible by minibatch_rows"
        )
    if (
        type(swiglu_limit) not in (int, float)
        or not math.isfinite(float(swiglu_limit))
        or swiglu_limit <= 0
        or swiglu_limit > 3.4028234663852886e38
    ):
        raise ValueError("swiglu_limit must be a finite positive number")
    if (
        type(spin_limit) is not int
        or spin_limit <= 0
        or spin_limit > (1 << 63) - 1
    ):
        raise ValueError("spin_limit must be a positive int64")
    _C.fp8_block_megakernel_out(
        x_buffer,
        x_ptrs,
        x_scale_buffer,
        x_scale_ptrs,
        routed_x,
        routed_x_scale,
        m_indices,
        schedule_peer_rank,
        schedule_peer_token_idx,
        num_tokens,
        tokens_per_expert,
        w13,
        w13_scale,
        gate_up,
        down_input,
        down_input_scale,
        w2,
        w2_scale,
        routed_y,
        combine_buffer,
        combine_ptrs,
        route_ready,
        route_ready_ptrs,
        topk_weights,
        topk_ids,
        output,
        x_routed_ready,
        gate_up_tile_ready,
        hidden_row_block_ready,
        y_routed_ready,
        y_routed_done,
        epilogue_claim,
        next_logical_cluster,
        next_reduce_probe,
        role_cursor,
        cluster_role,
        dispatch_tile_cursor,
        dispatch_tiles_done,
        push_tile_cursor,
        worker_ticket,
        comm_owner,
        comm_worker_ticket,
        producer_done,
        comm_closed,
        push_done,
        reduce_done,
        terminate,
        epilogue_done,
        in_use,
        barrier_buffer,
        barrier_target,
        input_expected_scratch,
        barrier_multicast_ptr,
        trap_record_ptr,
        ep_rank,
        comm_clusters,
        compute_clusters,
        minibatch_rows,
        macrobatch_rows,
        float(swiglu_limit),
        spin_limit,
    )


@torch.library.custom_op(
    "mok::workspace_lease_acquire", mutates_args=("in_use",)
)
def workspace_lease_acquire(
    in_use: torch.Tensor, trap_record_ptr: int, ep_rank: int
) -> None:
    """First device operation of an orchestration entry: atom.exch.acquire
    on in_use; a concurrent holder fail-closes via the REENTRANT trap."""
    if not hasattr(_C, "mok_workspace_lease_acquire"):
        raise RuntimeError("the loaded MoK extension lacks the lease kernels")
    _C.mok_workspace_lease_acquire(in_use, trap_record_ptr, ep_rank)


@torch.library.custom_op(
    "mok::workspace_lease_release", mutates_args=("in_use",)
)
def workspace_lease_release(in_use: torch.Tensor) -> None:
    """Trailing lease release for entries that do not hand the lease to the
    epilogue release chain."""
    if not hasattr(_C, "mok_workspace_lease_release"):
        raise RuntimeError("the loaded MoK extension lacks the lease kernels")
    _C.mok_workspace_lease_release(in_use)


def _validate_fp8_block_grouped_contiguous(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output: torch.Tensor,
    num_tokens: torch.Tensor | None = None,
) -> None:
    if input.ndim != 2 or not input.is_cuda or not input.is_contiguous():
        raise ValueError("input must be contiguous CUDA [M,K]")
    if input.dtype != torch.float8_e4m3fn:
        raise TypeError("input must use torch.float8_e4m3fn")
    total_m, reduction = input.shape
    if total_m < 64 or total_m % 64 != 0:
        raise ValueError("input M must be at least 64 and divisible by 64")
    if reduction < 128 or reduction % 128 != 0:
        raise ValueError("input K must be at least 128 and divisible by 128")
    if (
        weight.ndim != 3
        or not weight.is_cuda
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous CUDA float8_e4m3fn [E,N,K]")
    experts, output_size, weight_reduction = weight.shape
    if experts <= 0 or output_size < 128 or output_size % 128 != 0:
        raise ValueError("weight E must be positive and N must be N128 aligned")
    if weight_reduction != reduction:
        raise ValueError("input and weight reduction dimensions must match")
    expected_input_scale_shape = (total_m, reduction // 128)
    expected_weight_scale_shape = (
        experts,
        output_size // 128,
        reduction // 128,
    )
    if (
        not input_scale.is_cuda
        or input_scale.dtype != torch.float32
        or not input_scale.is_contiguous()
        or tuple(input_scale.shape) != expected_input_scale_shape
    ):
        raise ValueError(
            "input_scale must be contiguous CUDA float32 "
            f"{expected_input_scale_shape}"
        )
    if (
        not weight_scale.is_cuda
        or weight_scale.dtype != torch.float32
        or not weight_scale.is_contiguous()
        or tuple(weight_scale.shape) != expected_weight_scale_shape
    ):
        raise ValueError(
            "weight_scale must be contiguous CUDA float32 "
            f"{expected_weight_scale_shape}"
        )
    if (
        not m_indices.is_cuda
        or m_indices.dtype != torch.int32
        or not m_indices.is_contiguous()
        or tuple(m_indices.shape) != (total_m,)
    ):
        raise ValueError("m_indices must be contiguous CUDA int32 [M]")
    if (
        output.ndim != 2
        or not output.is_cuda
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
        or tuple(output.shape) != (total_m, output_size)
    ):
        raise ValueError("output must be contiguous CUDA bfloat16 [M,N]")
    tensors = (weight, input_scale, weight_scale, m_indices, output)
    if any(tensor.device != input.device for tensor in tensors):
        raise ValueError("all grouped GEMM tensors must share one device")
    if num_tokens is not None and (
        not num_tokens.is_cuda
        or num_tokens.device != input.device
        or num_tokens.dtype != torch.int32
        or not num_tokens.is_contiguous()
        or tuple(num_tokens.shape) != (1,)
    ):
        raise ValueError("num_tokens must be contiguous CUDA int32 [1]")
    if torch.cuda.get_device_capability(input.device) != (9, 0):
        raise NotImplementedError("FP8 grouped contiguous GEMM currently requires SM90")


@torch.library.custom_op(
    "mok::fp8_block_grouped_contiguous_out",
    mutates_args=("output",),
)
def fp8_block_grouped_contiguous_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run an SM90 FP8/K128 expert-major grouped GEMM into caller storage."""
    _validate_fp8_block_grouped_contiguous(
        input, weight, input_scale, weight_scale, m_indices, output
    )
    if not hasattr(_C, "fp8_block_grouped_contiguous_out"):
        raise RuntimeError("the loaded MoK extension lacks FP8 grouped GEMM")

    _C.fp8_block_grouped_contiguous_out(
        input, weight, input_scale, weight_scale, m_indices, output
    )


@torch.library.custom_op(
    "mok::fp8_block_grouped_contiguous_dynamic_out",
    mutates_args=("output",),
)
def fp8_block_grouped_contiguous_dynamic_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run grouped GEMM over device-selected valid rows in caller storage."""
    _validate_fp8_block_grouped_contiguous(
        input,
        weight,
        input_scale,
        weight_scale,
        m_indices,
        output,
        num_tokens,
    )
    if not hasattr(_C, "fp8_block_grouped_contiguous_dynamic_out"):
        raise RuntimeError(
            "the loaded MoK extension lacks dynamic FP8 grouped GEMM"
        )
    _C.fp8_block_grouped_contiguous_dynamic_out(
        input,
        weight,
        input_scale,
        weight_scale,
        m_indices,
        num_tokens,
        output,
    )


@torch.library.custom_op(
    "mok::routed_epilogue_out",
    mutates_args=("output",),
)
def routed_epilogue_out(
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Reduce route slots with router weights into caller-owned BF16 output."""
    if (
        output.ndim != 2
        or not output.is_cuda
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous CUDA bfloat16 [T,H]")
    num_tokens, hidden_size = output.shape
    if num_tokens < 256 or num_tokens % 256 != 0:
        raise ValueError("output T must be at least 256 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("output H must be positive and divisible by 256")
    if (
        topk_weights.ndim != 2
        or not topk_weights.is_cuda
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
        or topk_weights.shape[0] != num_tokens
    ):
        raise ValueError("topk_weights must be contiguous CUDA float32 [T,topk]")
    topk = topk_weights.shape[1]
    if not 0 < topk <= 255:
        raise ValueError("topk must be in [1,255]")
    device_properties = torch.cuda.get_device_properties(output.device)
    dynamic_smem_bytes = 2 * (topk * 2048 + topk * 4) + 1024
    if dynamic_smem_bytes > device_properties.shared_memory_per_block_optin:
        raise ValueError("topk requires more dynamic shared memory than the device supports")
    if (
        combine_buffer.ndim != 2
        or not combine_buffer.is_cuda
        or combine_buffer.dtype != torch.bfloat16
        or not combine_buffer.is_contiguous()
        or tuple(combine_buffer.shape) != (num_tokens * topk, hidden_size)
    ):
        raise ValueError("combine_buffer must be contiguous CUDA bfloat16 [T*topk,H]")
    if combine_buffer.device != output.device or topk_weights.device != output.device:
        raise ValueError("all routed epilogue tensors must share one device")
    if torch.cuda.get_device_capability(output.device) != (9, 0):
        raise NotImplementedError("routed epilogue currently requires SM90")
    if not hasattr(_C, "routed_epilogue_out"):
        raise RuntimeError("the loaded MoK extension lacks routed epilogue")

    _C.routed_epilogue_out(combine_buffer, topk_weights, output)


@torch.library.custom_op(
    "mok::mxfp8_quantize", mutates_args=(),
    schema="(Tensor x_bf16, bool return_normal, bool return_transposed) -> (Tensor?, Tensor?, Tensor?, Tensor?)",
)
def mxfp8_quantize(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    _sm90_reject("mxfp8_quantize")
    """Quantizes BF16 matrices to MXFP8 in normal and/or transposed layouts.

    Inputs:
        x_bf16:           bfloat16 [M, N] or [E, M, N]
        return_normal:     bool
        return_transposed: bool

    Outputs:
        x_fp8:   float8_e4m3fn [M, N] or [E, M, N] | None
        x_sc:    uint8 [E * M // 128, N // 128, 32, 16] | None
        x_fp8_t: float8_e4m3fn [N, M] or [E, N, M] | None
        x_sc_t:  uint8 [E * N // 128, M // 128, 32, 16] | None
    """
    if x_bf16.ndim not in (2, 3):
        raise ValueError("x_bf16 must have shape (M, N) or (E, M, N)")
    if any(size <= 0 for size in x_bf16.shape):
        raise ValueError("x_bf16 dimensions must be positive")
    if x_bf16.shape[-2] % 128 != 0 or x_bf16.shape[-1] % 128 != 0:
        raise ValueError("x_bf16 M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("at least one quantized layout must be requested")

    return _C.mxfp8_quantize(x_bf16, return_normal, return_transposed)


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_mxfp8", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_mxfp8(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    w_routed_down_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _sm90_reject("dispatch_mlp_swiglu_combine_fwd_mxfp8")
    """Runs the fused MXFP8 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [num_local_tokens * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:        uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:          uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_down_sc:        uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_fp8_t_routed:      float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:       uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:         bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:           bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:       float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:        uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:       bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed: float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:  uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        y_shared:            bfloat16 [num_local_tokens, hidden_size]
        y_routed:            bfloat16 [macrobatch_size, hidden_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0
            or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0
            or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (("x_ptrs", x_ptrs),
                                   ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if w_shared_gate.ndim != 2:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    expected_shapes = (
        ("combine_buffer", combine_buffer, (num_local_tokens * topk, hidden_size)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_down_sc", w_routed_down_sc,
         (num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("w_routed_down_sc", w_routed_down_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.dispatch_mlp_swiglu_combine_fwd_mxfp8(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_routed_gate_sc,
        w_shared_up, w_routed_up, w_routed_up_sc,
        w_shared_down, w_routed_down, w_routed_down_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_bf16", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_bf16(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused BF16 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [num_local_tokens * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           bfloat16 [num_local_experts, hidden_size, intermediate_size]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_routed:      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        y_shared:      bfloat16 [num_local_tokens, hidden_size]
        y_routed:      bfloat16 [macrobatch_size, hidden_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (("x_ptrs", x_ptrs), ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("combine_buffer", combine_buffer, (num_local_tokens * topk, hidden_size)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")

    return _C.dispatch_mlp_swiglu_combine_fwd_bf16(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_mxfp8",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer",
                  "x_fp8_t_routed", "x_sc_t_routed",
                  "gate_fp8_routed", "gate_sc_routed", "up_fp8_routed", "up_sc_routed",
                  "hidden_fp8_t_routed", "hidden_sc_t_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_mxfp8(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate_T: torch.Tensor,
    w_routed_gate_T_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up_T: torch.Tensor,
    w_routed_up_T_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down_T: torch.Tensor,
    w_routed_down_T_sc: torch.Tensor,
    x_fp8_t_routed: torch.Tensor,
    x_sc_t_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_fp8_routed: torch.Tensor,
    gate_sc_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_fp8_routed: torch.Tensor,
    up_sc_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_fp8_t_routed: torch.Tensor,
    hidden_sc_t_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _sm90_reject("dispatch_mlp_swiglu_combine_bwd_mxfp8")
    """Runs the fused MXFP8 MoE backward pass.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [num_local_tokens * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [num_local_tokens, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [num_local_tokens, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate_T:               float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_gate_T_sc:            uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up_T:                 float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_up_T_sc:              uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down_T:               float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_down_T_sc:            uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        x_fp8_t_routed:                float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:                 uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:               float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:                uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:                 float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:                  uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed:           float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:            uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        w_routed_gate:                 float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:              uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_routed_up:                   float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:                uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int

    Outputs:
        d_x_shared:          bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:          bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:       bfloat16 [num_local_tokens, intermediate_size]
        d_gate_fp8_routed:   float8_e4m3fn [macrobatch_size, intermediate_size]
        d_gate_sc_routed:    uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_up_shared:         bfloat16 [num_local_tokens, intermediate_size]
        d_up_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        d_up_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_hidden_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_y_fp8_routed:      float8_e4m3fn [macrobatch_size, hidden_size]
        d_y_sc_routed:       uint8 [macrobatch_size // 128, hidden_size // 128, 32, 16]
        d_w_shared_gate:     bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate:     bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:       bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:       bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down:     bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down:     bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0
            or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0
            or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    ):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 4, 8, 16, 32, 64")
    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
    ):
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    mb_i_sc = (macrobatch_size // 128, intermediate_size // 128, 32, 16)
    i_mb_sc = (intermediate_size // 128, macrobatch_size // 128, 32, 16)
    h_mb_sc = (hidden_size // 128, macrobatch_size // 128, 32, 16)
    e_i_h_sc = (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)
    e_h_i_sc = (num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16)
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (num_local_tokens * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (num_local_tokens, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (num_local_tokens, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate_T", w_routed_gate_T,
         (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc, e_h_i_sc),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up_T", w_routed_up_T, (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_up_T_sc", w_routed_up_T_sc, e_h_i_sc),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down_T", w_routed_down_T,
         (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_down_T_sc", w_routed_down_T_sc, e_i_h_sc),
        ("x_fp8_t_routed", x_fp8_t_routed, (hidden_size, macrobatch_size)),
        ("x_sc_t_routed", x_sc_t_routed, h_mb_sc),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_fp8_routed", gate_fp8_routed, (macrobatch_size, intermediate_size)),
        ("gate_sc_routed", gate_sc_routed, mb_i_sc),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_fp8_routed", up_fp8_routed, (macrobatch_size, intermediate_size)),
        ("up_sc_routed", up_sc_routed, mb_i_sc),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed, (intermediate_size, macrobatch_size)),
        ("hidden_sc_t_routed", hidden_sc_t_routed, i_mb_sc),
        ("w_routed_gate", w_routed_gate,
         (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc, e_i_h_sc),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc, e_i_h_sc),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("d_y_buffer", d_y_buffer),
        ("d_x_routed_buffer", d_x_routed_buffer),
        ("router_weight_buffer", router_weight_buffer),
        ("d_router_weight_buffer", d_router_weight_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate_T", w_routed_gate_T),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up_T", w_routed_up_T),
        ("w_routed_up_T_sc", w_routed_up_T_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down_T", w_routed_down_T),
        ("w_routed_down_T_sc", w_routed_down_T_sc),
        ("x_fp8_t_routed", x_fp8_t_routed),
        ("x_sc_t_routed", x_sc_t_routed),
        ("gate_shared", gate_shared),
        ("gate_fp8_routed", gate_fp8_routed),
        ("gate_sc_routed", gate_sc_routed),
        ("up_shared", up_shared),
        ("up_fp8_routed", up_fp8_routed),
        ("up_sc_routed", up_sc_routed),
        ("hidden_shared", hidden_shared),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed),
        ("hidden_sc_t_routed", hidden_sc_t_routed),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.dispatch_mlp_swiglu_combine_bwd_mxfp8(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
        w_shared_up, w_routed_up_T, w_routed_up_T_sc,
        w_shared_down, w_routed_down_T, w_routed_down_T_sc,
        x_fp8_t_routed, x_sc_t_routed,
        gate_shared, gate_fp8_routed, gate_sc_routed,
        up_shared, up_fp8_routed, up_sc_routed,
        hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
        x, x_ptrs, w_routed_gate, w_routed_gate_sc, w_routed_up, w_routed_up_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_bf16",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer", "x_routed", "gate_routed", "up_routed", "hidden_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_bf16(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    x_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _sm90_reject("dispatch_mlp_swiglu_combine_bwd_bf16")
    """Runs the fused BF16 MoE backward pass.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [num_local_tokens * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [num_local_tokens, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [num_local_tokens, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:                 bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up:                   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down:                 bfloat16 [num_local_experts, hidden_size, intermediate_size]
        x_routed:                      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:                   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:                     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed:                 bfloat16 [macrobatch_size, intermediate_size]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int

    Outputs:
        d_x_shared:      bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        d_gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        d_up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        d_y_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_w_shared_gate: bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:   bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down: bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down: bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    pointer_lists = (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    )
    for pointer_name, pointers in pointer_lists:
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 4, 8, 16, 32, 64")
    for pointer_name, pointers in pointer_lists[:-1]:
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (num_local_tokens * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (num_local_tokens, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (num_local_tokens, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("x_routed", x_routed, (macrobatch_size, hidden_size)),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_routed", gate_routed, (macrobatch_size, intermediate_size)),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_routed", up_routed, (macrobatch_size, intermediate_size)),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_routed", hidden_routed, (macrobatch_size, intermediate_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor, _ in expected_shapes:
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.device != x.device:
        raise ValueError(f"schedule_peer_rank must be on {x.device}")

    return _C.dispatch_mlp_swiglu_combine_bwd_bf16(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        x_routed, gate_shared, gate_routed, up_shared, up_routed,
        hidden_shared, hidden_routed,
        x, x_ptrs,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::fwd_epilogue", mutates_args=())
def fwd_epilogue(
    y_shared: torch.Tensor,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Combines shared and router-weighted routed expert outputs.

    Inputs:
        y_shared:       bfloat16 [num_local_tokens, hidden_size]
        combine_buffer: bfloat16 [num_local_tokens * topk, hidden_size]
        topk_weights:   float32 [num_local_tokens, topk]

    Outputs:
        output: bfloat16 [num_local_tokens, hidden_size]
    """
    if y_shared.ndim != 2:
        raise ValueError("y_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = y_shared.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if (topk_weights.device != y_shared.device
            or combine_buffer.device != y_shared.device):
        raise ValueError("all tensors must be on the same CUDA device")
    if topk_weights.ndim != 2 or topk_weights.shape[0] != num_local_tokens:
        raise ValueError("topk_weights must have shape (num_local_tokens, topk)")
    topk = topk_weights.shape[1]
    if not 0 < topk <= 255:
        raise ValueError("topk must be in [1, 255]")
    if tuple(combine_buffer.shape) != (num_local_tokens * topk, hidden_size):
        raise ValueError("combine_buffer must have shape (num_local_tokens * topk, hidden_size)")

    return _C.fwd_epilogue(y_shared, combine_buffer, topk_weights)


@torch.library.custom_op("mok::bwd_epilogue", mutates_args=())
def bwd_epilogue(
    d_x_shared: torch.Tensor,
    d_x_routed_buffer: torch.Tensor,
) -> torch.Tensor:
    """Combines shared and routed input gradients.

    Inputs:
        d_x_shared:        bfloat16 [num_local_tokens, hidden_size]
        d_x_routed_buffer: bfloat16 [num_local_tokens * topk, hidden_size]

    Outputs:
        d_x: bfloat16 [num_local_tokens, hidden_size]
    """
    if d_x_shared.ndim != 2:
        raise ValueError("d_x_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = d_x_shared.shape
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be at least 512 and divisible by 256")
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if d_x_routed_buffer.device != d_x_shared.device:
        raise ValueError("d_x_routed_buffer must be on the same CUDA device")
    if d_x_routed_buffer.ndim != 2:
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")
    if (d_x_routed_buffer.shape[1] != hidden_size
            or d_x_routed_buffer.shape[0] % num_local_tokens != 0
            or d_x_routed_buffer.shape[0] == 0
            or d_x_routed_buffer.shape[0] > num_local_tokens * 255):
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")

    return _C.bwd_epilogue(d_x_shared, d_x_routed_buffer)


def interleave_w13(
    w13: torch.Tensor, w13_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder a [E, 2I, K] gate/up weight so that gate block j and up block j
    are adjacent N128 tiles: rows become [gate 0:128, up 0:128, gate 128:256,
    up 128:256, ...].  The warprole W13 task for intermediate tile j then reads
    N tiles 2j (gate) and 2j+1 (up) from one weight tensor.  Scales
    [E, 2I/128, K/128] are permuted the same way.  Values are untouched, so the
    GEMM stays bitwise identical to the split path."""
    if w13.dim() != 3 or w13_scale.dim() != 3:
        raise ValueError("w13 must be [E, 2I, K] and w13_scale [E, 2I/128, K/128]")
    experts, n2, k = w13.shape
    if n2 % 256 != 0 or k % 128 != 0:
        raise ValueError("2I must be a multiple of 256 and K a multiple of 128")
    if tuple(w13_scale.shape) != (experts, n2 // 128, k // 128):
        raise ValueError("w13_scale shape must be [E, 2I/128, K/128]")
    inter = n2 // 2
    blocks = inter // 128
    gate = w13[:, :inter].reshape(experts, blocks, 128, k)
    up = w13[:, inter:].reshape(experts, blocks, 128, k)
    weight = torch.stack([gate, up], dim=2).reshape(experts, n2, k).contiguous()
    gate_scale = w13_scale[:, :blocks]
    up_scale = w13_scale[:, blocks:]
    scale = torch.stack([gate_scale, up_scale], dim=2).reshape(experts, n2 // 128, k // 128).contiguous()
    return weight, scale
