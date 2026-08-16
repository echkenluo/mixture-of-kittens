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
) -> None:
    """Clear, combine, synchronize, and reduce routed rows in one call."""
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
    )


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
    if torch.cuda.get_device_capability(input.device) != (9, 0):
        raise NotImplementedError("FP8 grouped contiguous GEMM currently requires SM90")
    if not hasattr(_C, "fp8_block_grouped_contiguous_out"):
        raise RuntimeError("the loaded MoK extension lacks FP8 grouped GEMM")

    _C.fp8_block_grouped_contiguous_out(
        input, weight, input_scale, weight_scale, m_indices, output
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
