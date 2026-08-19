import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ._terminal_tma_contract import validate_terminal_tma_dispatch_layout
from .ops import (
    all_gather_top_experts,
    barrier_all,
    bwd_epilogue,
    dispatch_mlp_swiglu_combine_bwd_mxfp8,
    dispatch_mlp_swiglu_combine_bwd_bf16,
    dispatch_mlp_swiglu_combine_fwd_mxfp8,
    dispatch_mlp_swiglu_combine_fwd_bf16,
    fp8_block_build_schedule_out,
    fp8_block_dispatch_gemm_fused_out,
    fp8_block_gemm_combine_fused_out,
    fp8_block_grouped_contiguous_out,
    fp8_block_grouped_contiguous_dynamic_out,
    fp8_block_routed_combine_reduce_fused_out,
    fp8_block_routed_combine_reduce_out,
    fp8_block_routed_combine_out,
    fp8_block_routed_dispatch_copy_out,
    fp8_block_routed_dispatch_out,
    fwd_epilogue,
    fp8_block_dispatch_gemm_prewarm,
    fp8_block_megakernel_out,
    fp8_block_megakernel_prepare_out,
    fp8_block_megakernel_prewarm,
    routed_epilogue_fused_out,
    routed_epilogue_out,
    require_fp8_block_megakernel,
    schedule,
    workspace_lease_acquire,
    workspace_lease_release,
)


@dataclass(frozen=True, slots=True)
class MoKConfig:
    fwd_num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 131072
    schedule_capacity_multiplier: float = 0.5
    all_gather_top_experts_chunk_bytes: int = 2048


@dataclass(frozen=True, slots=True)
class MoKSchedule:
    peer_rank: torch.Tensor          # (schedule_capacity,) int32
    peer_token_idx: torch.Tensor     # (schedule_capacity,) int32
    num_tokens: torch.Tensor         # (1,) int32
    tokens_per_expert: torch.Tensor  # (num_local_experts,) int32
    expert_padding: int = 256        # () int


@dataclass(frozen=True, slots=True)
class MoKForwardContext:
    x_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    up_shared: torch.Tensor
    up_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class MoKWorkspace:
    group_name: str                                   # () str
    ep_rank: int                                      # () int
    ep_size: int                                      # () int
    device: torch.device                              # () torch.device
    num_local_tokens: int                             # () int
    hidden_size: int                                  # () int
    topk: int                                         # () int
    schedule_capacity: int                            # () int
    x_buffer: torch.Tensor                            # (num_local_tokens, hidden_size) bfloat16
    x_buffer_handle: Any                              # () SymmetricMemory handle
    x_buffer_ptrs: list[int]                          # (ep_size,) uintptr64
    combine_buffer: torch.Tensor                      # (num_local_tokens * topk, hidden_size) bfloat16
    combine_buffer_handle: Any                        # () SymmetricMemory handle
    combine_buffer_ptrs: list[int]                    # (ep_size,) uintptr64
    d_y_buffer: torch.Tensor                          # (num_local_tokens, hidden_size) bfloat16
    d_y_buffer_handle: Any                            # () SymmetricMemory handle
    d_y_buffer_ptrs: list[int]                        # (ep_size,) uintptr64
    d_x_routed_buffer: torch.Tensor                   # (num_local_tokens * topk, hidden_size) bfloat16
    d_x_routed_buffer_handle: Any                     # () SymmetricMemory handle
    d_x_routed_buffer_ptrs: list[int]                 # (ep_size,) uintptr64
    router_weight_buffer: torch.Tensor                # (num_local_tokens, topk) float32
    router_weight_buffer_handle: Any                  # () SymmetricMemory handle
    router_weight_buffer_ptrs: list[int]              # (ep_size,) uintptr64
    d_router_weight_buffer: torch.Tensor              # (num_local_tokens, topk) float32
    d_router_weight_buffer_handle: Any                # () SymmetricMemory handle
    d_router_weight_buffer_ptrs: list[int]            # (ep_size,) uintptr64
    all_gather_top_experts_buffer: torch.Tensor       # (ep_size, num_local_tokens, topk) int32
    all_gather_top_experts_buffer_handle: Any         # () SymmetricMemory handle
    all_gather_top_experts_buffer_multicast_ptr: int  # () uintptr64
    barrier_buffer: torch.Tensor                      # (1,) int32
    barrier_buffer_handle: Any                        # () SymmetricMemory handle
    barrier_buffer_ptrs: list[int]                    # (ep_size,) uintptr64
    barrier_buffer_multicast_ptr: int                 # () uintptr64
    barrier_target: torch.Tensor                      # (1,) int32


@dataclass(slots=True)
class MoKFP8RouteWorkspace:
    """Caller-owned production FP8 dispatch/combine storage for SM90."""

    group_name: str
    ep_rank: int
    ep_size: int
    device: torch.device
    num_local_tokens: int
    hidden_size: int
    topk: int
    num_local_experts: int
    schedule_capacity: int
    x_buffer: torch.Tensor
    x_buffer_handle: Any
    x_buffer_ptrs: list[int]
    x_scale_buffer: torch.Tensor
    x_scale_buffer_handle: Any
    x_scale_buffer_ptrs: list[int]
    combine_buffer: torch.Tensor
    combine_buffer_handle: Any
    combine_buffer_ptrs: list[int]
    output: torch.Tensor
    routed_x: torch.Tensor
    routed_x_scale: torch.Tensor
    m_indices: torch.Tensor
    schedule_peer_rank: torch.Tensor
    schedule_peer_token_idx: torch.Tensor
    schedule_num_tokens: torch.Tensor
    schedule_tokens_per_expert: torch.Tensor
    schedule_tokens_per_expert_and_peer: torch.Tensor
    all_gather_top_experts_buffer: torch.Tensor
    all_gather_top_experts_buffer_handle: Any
    all_gather_top_experts_buffer_multicast_ptr: int
    barrier_buffer: torch.Tensor
    barrier_buffer_handle: Any
    barrier_buffer_ptrs: list[int]
    barrier_buffer_multicast_ptr: int
    barrier_target: torch.Tensor
    combine_completion: torch.Tensor       # (1,) int32, fused-arrive counter
    barrier_expected_scratch: torch.Tensor  # (1,) int32, fused-wait expected
    input_expected_scratch: torch.Tensor    # (1,) int32, fused dispatch wait
    tile_ready: torch.Tensor  # (capacity/64,) int32 producer->consumer count
    down_ready: torch.Tensor  # (capacity/64,) int32 down-GEMM tile count
    ticket_counter: torch.Tensor  # (1,) int32, K1 worker-queue head
    worker_ticket: torch.Tensor   # (1024,) int32, per-cluster publish slots
    trap_record: torch.Tensor     # (8,) int64 host-mapped pinned; alloc-zeroed
    trap_record_ptr: int          # host address; device ptr resolved in C++
    in_use: torch.Tensor          # (1,) int32 production lease guard
    epilogue_done: torch.Tensor   # (1,) int32 release completion counter


@dataclass(slots=True)
class MoKFP8TerminalWorkspace:
    """Caller-owned storage for the terminal SM90 FP8 megakernel.

    This remains separate from :class:`MoKFP8RouteWorkspace`: terminal work
    must not inherit the K1/K2 ticket and readiness protocol.
    """

    group_name: str
    ep_rank: int
    ep_size: int
    device: torch.device
    num_local_tokens: int
    padded_num_local_tokens: int
    hidden_size: int
    intermediate_size: int
    topk: int
    num_local_experts: int
    schedule_capacity: int
    comm_clusters: int
    compute_clusters: int
    max_compute_clusters: int
    x_buffer: torch.Tensor
    x_buffer_handle: Any
    x_buffer_ptrs: list[int]
    x_buffer_bytes_per_rank: list[int]
    x_scale_buffer: torch.Tensor
    x_scale_buffer_handle: Any
    x_scale_buffer_ptrs: list[int]
    x_scale_buffer_bytes_per_rank: list[int]
    combine_buffer: torch.Tensor
    combine_buffer_handle: Any
    combine_buffer_ptrs: list[int]
    route_ready: torch.Tensor
    route_ready_handle: Any
    route_ready_ptrs: list[int]
    barrier_buffer: torch.Tensor
    barrier_buffer_handle: Any
    barrier_buffer_ptrs: list[int]
    barrier_buffer_multicast_ptr: int
    barrier_target: torch.Tensor
    input_expected_scratch: torch.Tensor
    routed_x: torch.Tensor
    routed_x_scale: torch.Tensor
    m_indices: torch.Tensor
    schedule_peer_rank: torch.Tensor
    schedule_peer_token_idx: torch.Tensor
    schedule_num_tokens: torch.Tensor
    schedule_tokens_per_expert: torch.Tensor
    schedule_tokens_per_expert_and_peer: torch.Tensor
    all_gather_top_experts_buffer: torch.Tensor
    all_gather_top_experts_buffer_handle: Any
    all_gather_top_experts_buffer_multicast_ptr: int
    gate_up: torch.Tensor
    down_input: torch.Tensor
    down_input_scale: torch.Tensor
    routed_y: torch.Tensor
    x_routed_ready: torch.Tensor
    gate_up_tile_ready: torch.Tensor
    hidden_row_block_ready: torch.Tensor
    y_routed_ready: torch.Tensor
    y_routed_done: torch.Tensor
    epilogue_claim: torch.Tensor
    next_logical_cluster: torch.Tensor
    next_reduce_probe: torch.Tensor
    role_cursor: torch.Tensor
    cluster_role: torch.Tensor
    # Legacy name: dense native communication D/C ticket cursor.
    dispatch_tile_cursor: torch.Tensor
    dispatch_tiles_done: torch.Tensor
    # Completed combine-ticket receipt; it is not an ownership cursor.
    push_tile_cursor: torch.Tensor
    worker_ticket: torch.Tensor  # (compute_clusters,) compute-role publish slot
    comm_owner: torch.Tensor  # (1,) physical cluster holding role 0 or -1
    comm_worker_ticket: torch.Tensor  # (comm_clusters,) per-role publish slots
    producer_done: torch.Tensor
    comm_closed: torch.Tensor
    push_done: torch.Tensor
    reduce_done: torch.Tensor
    terminate: torch.Tensor
    epilogue_done: torch.Tensor
    in_use: torch.Tensor
    trap_record: torch.Tensor
    trap_record_ptr: int


_WORKSPACE_CACHE: dict[tuple[str, int, int, int, int, int], MoKWorkspace] = {}
_FP8_ROUTE_WORKSPACE_CACHE: dict[
    tuple[str, int, int, int, int, int, int], MoKFP8RouteWorkspace
] = {}
_FP8_TERMINAL_WORKSPACE_CACHE: dict[
    tuple[str, int, int, int, int, int, int], MoKFP8TerminalWorkspace
] = {}


def validate_workspace_args(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    min_num_local_tokens: int = 512,
    num_local_tokens_alignment: int = 256,
) -> None:
    """Validates the arguments used to create or retrieve a workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        None
    """
    if (
        type(config.schedule_capacity_multiplier) not in (int, float)
        or not math.isfinite(config.schedule_capacity_multiplier)
        or config.schedule_capacity_multiplier <= 0
    ):
        raise ValueError("schedule_capacity_multiplier must be a positive finite number")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    if device_index != torch.cuda.current_device():
        raise ValueError("MoK workspace device must be the current CUDA device")
    cap = torch.cuda.get_device_capability(device)
    if cap not in ((9, 0), (10, 0), (10, 3)):
        raise NotImplementedError("MoK currently requires an SM90, SM100 or SM103 GPU")
    if cap == (9, 0):
        import os
        if os.environ.get("MOK_SM90_EXPERIMENTAL") != "1":
            raise NotImplementedError(
                "MoK SM90 port is experimental: only the BF16 forward path is "
                "being brought up (MXFP8 and backward are not implemented). "
                "Set MOK_SM90_EXPERIMENTAL=1 to proceed.")
    device_properties = torch.cuda.get_device_properties(device)
    if (
        type(num_local_tokens_alignment) is not int
        or num_local_tokens_alignment <= 0
    ):
        raise ValueError("num_local_tokens_alignment must be a positive integer")
    if type(num_local_tokens) is not int or num_local_tokens < min_num_local_tokens:
        raise ValueError(
            "num_local_tokens must be an integer at least "
            f"{min_num_local_tokens}"
        )
    if num_local_tokens % num_local_tokens_alignment != 0:
        raise ValueError(
            "num_local_tokens must be divisible by "
            f"{num_local_tokens_alignment}"
        )
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if hidden_size % 256 != 0:
        raise ValueError("hidden_size must be divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    fwd_epilogue_smem_bytes = 2 * ((topk + 1) * 2048 + topk * 4) + 1024
    if fwd_epilogue_smem_bytes > device_properties.shared_memory_per_block_optin:
        raise ValueError("topk requires more dynamic shared memory than the device supports")

    group_name = group.group_name
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError("process group must have a nonempty group_name")
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    if ep_size not in (4, 8, 16, 32, 64):
        raise ValueError("MoK EP size must be one of 4, 8, 16, 32, 64")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError("current process is not a member of the EP process group")


def create_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> MoKWorkspace:
    """Creates a new caller-owned workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        workspace: MoKWorkspace
    """
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    local_shape = torch.tensor([num_local_tokens, hidden_size, topk], dtype=torch.int64, device=device)
    gathered_shapes = torch.empty(ep_size * local_shape.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered_shapes, local_shape, group=group)
    gathered_shapes = gathered_shapes.view(ep_size, local_shape.numel())
    if not torch.all(gathered_shapes == local_shape).item():  # .item() here is fine since this is one-time setup
        raise ValueError("MoK requires identical token, hidden, and top-k shapes on every EP rank")
    symm_mem.enable_symm_mem_for_group(group_name)

    schedule_capacity = num_local_tokens * topk * schedule_capacity_factor

    x_buffer = symm_mem.empty(num_local_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    x_buffer_handle = symm_mem.rendezvous(x_buffer, group_name)
    x_buffer_ptrs = [int(x_buffer_handle.buffer_ptrs[peer_rank]) for peer_rank in range(ep_size)]

    combine_buffer = symm_mem.empty(num_local_tokens * topk, hidden_size,
                                    dtype=torch.bfloat16, device=device)
    combine_buffer_handle = symm_mem.rendezvous(combine_buffer, group_name)
    combine_buffer_ptrs = [int(combine_buffer_handle.buffer_ptrs[peer_rank])
                           for peer_rank in range(ep_size)]

    d_y_buffer = symm_mem.empty(num_local_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    d_y_buffer_handle = symm_mem.rendezvous(d_y_buffer, group_name)
    d_y_buffer_ptrs = [int(d_y_buffer_handle.buffer_ptrs[peer_rank])
                       for peer_rank in range(ep_size)]

    d_x_routed_buffer = symm_mem.empty(num_local_tokens * topk, hidden_size,
                                      dtype=torch.bfloat16, device=device)
    d_x_routed_buffer_handle = symm_mem.rendezvous(d_x_routed_buffer, group_name)
    d_x_routed_buffer_ptrs = [int(d_x_routed_buffer_handle.buffer_ptrs[peer_rank])
                              for peer_rank in range(ep_size)]

    router_weight_buffer = symm_mem.empty(num_local_tokens, topk, dtype=torch.float32, device=device)
    router_weight_buffer_handle = symm_mem.rendezvous(router_weight_buffer, group_name)
    router_weight_buffer_ptrs = [int(router_weight_buffer_handle.buffer_ptrs[peer_rank])
                                 for peer_rank in range(ep_size)]

    d_router_weight_buffer = symm_mem.empty(num_local_tokens, topk,
                                            dtype=torch.float32, device=device)
    d_router_weight_buffer_handle = symm_mem.rendezvous(d_router_weight_buffer, group_name)
    d_router_weight_buffer_ptrs = [int(d_router_weight_buffer_handle.buffer_ptrs[peer_rank])
                                   for peer_rank in range(ep_size)]

    all_gather_top_experts_buffer = symm_mem.empty(
        ep_size, num_local_tokens, topk, dtype=torch.int32, device=device)
    all_gather_top_experts_buffer_handle = symm_mem.rendezvous(all_gather_top_experts_buffer,
                                                               group_name)
    all_gather_top_experts_buffer_multicast_ptr = int(all_gather_top_experts_buffer_handle.multicast_ptr)

    barrier_buffer = symm_mem.empty(1, dtype=torch.int32, device=device)
    barrier_buffer.zero_()
    barrier_buffer_handle = symm_mem.rendezvous(barrier_buffer, group_name)
    barrier_buffer_ptrs = [int(barrier_buffer_handle.buffer_ptrs[peer_rank])
                           for peer_rank in range(ep_size)]
    barrier_buffer_multicast_ptr = int(barrier_buffer_handle.multicast_ptr)
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)

    dist.barrier(group=group, async_op=True, device_ids=[device_index]).block_current_stream()

    workspace = MoKWorkspace(
        group_name=group_name, ep_rank=ep_rank, ep_size=ep_size, device=device,
        num_local_tokens=num_local_tokens, hidden_size=hidden_size, topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=x_buffer, x_buffer_handle=x_buffer_handle, x_buffer_ptrs=x_buffer_ptrs,
        combine_buffer=combine_buffer, combine_buffer_handle=combine_buffer_handle,
        combine_buffer_ptrs=combine_buffer_ptrs,
        d_y_buffer=d_y_buffer, d_y_buffer_handle=d_y_buffer_handle,
        d_y_buffer_ptrs=d_y_buffer_ptrs,
        d_x_routed_buffer=d_x_routed_buffer, d_x_routed_buffer_handle=d_x_routed_buffer_handle,
        d_x_routed_buffer_ptrs=d_x_routed_buffer_ptrs,
        router_weight_buffer=router_weight_buffer,
        router_weight_buffer_handle=router_weight_buffer_handle,
        router_weight_buffer_ptrs=router_weight_buffer_ptrs,
        d_router_weight_buffer=d_router_weight_buffer,
        d_router_weight_buffer_handle=d_router_weight_buffer_handle,
        d_router_weight_buffer_ptrs=d_router_weight_buffer_ptrs,
        all_gather_top_experts_buffer=all_gather_top_experts_buffer,
        all_gather_top_experts_buffer_handle=all_gather_top_experts_buffer_handle,
        all_gather_top_experts_buffer_multicast_ptr=all_gather_top_experts_buffer_multicast_ptr,
        barrier_buffer=barrier_buffer, barrier_buffer_handle=barrier_buffer_handle,
        barrier_buffer_ptrs=barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr=barrier_buffer_multicast_ptr,
        barrier_target=barrier_target,
    )
    return workspace


def create_fp8_route_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    num_local_experts: int,
) -> MoKFP8RouteWorkspace:
    """Create SM90 storage for production FP8 dispatch and BF16 combine."""
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        min_num_local_tokens=2,
        num_local_tokens_alignment=2,
    )

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise NotImplementedError("the production FP8 route workspace requires SM90")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    group_name = group.group_name
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(
        2, math.ceil(ep_size * config.schedule_capacity_multiplier)
    )
    schedule_capacity = num_local_tokens * topk * schedule_capacity_factor

    local_shape = torch.tensor(
        [num_local_tokens, hidden_size, topk, num_local_experts],
        dtype=torch.int64,
        device=device,
    )
    gathered_shapes = torch.empty(
        ep_size * local_shape.numel(), dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered_shapes, local_shape, group=group)
    gathered_shapes = gathered_shapes.view(ep_size, local_shape.numel())
    if not torch.all(gathered_shapes == local_shape).item():
        raise ValueError(
            "MoK requires identical token, hidden, top-k, and local-expert "
            "shapes on every EP rank"
        )

    x_buffer = symm_mem.empty(
        num_local_tokens,
        hidden_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    x_buffer_handle = symm_mem.rendezvous(x_buffer, group_name)
    x_buffer_ptrs = [
        int(x_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]

    x_scale_buffer = symm_mem.empty(
        num_local_tokens,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    x_scale_buffer_handle = symm_mem.rendezvous(x_scale_buffer, group_name)
    x_scale_buffer_ptrs = [
        int(x_scale_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]

    combine_buffer = symm_mem.empty(
        num_local_tokens * topk,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    combine_buffer_handle = symm_mem.rendezvous(combine_buffer, group_name)
    combine_buffer_ptrs = [
        int(combine_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]

    output = torch.empty(
        num_local_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    routed_x = torch.empty(
        schedule_capacity,
        hidden_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    routed_x_scale = torch.empty(
        schedule_capacity,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    m_indices = torch.empty(
        schedule_capacity, dtype=torch.int32, device=device
    )
    schedule_peer_rank = torch.empty(
        schedule_capacity, dtype=torch.int32, device=device
    )
    schedule_peer_token_idx = torch.empty_like(schedule_peer_rank)
    schedule_num_tokens = torch.empty(1, dtype=torch.int32, device=device)
    schedule_tokens_per_expert = torch.empty(
        num_local_experts, dtype=torch.int32, device=device
    )
    schedule_tokens_per_expert_and_peer = torch.empty(
        num_local_experts * ep_size, dtype=torch.int32, device=device
    )

    all_gather_top_experts_buffer = symm_mem.empty(
        ep_size,
        num_local_tokens,
        topk,
        dtype=torch.int32,
        device=device,
    )
    all_gather_top_experts_buffer_handle = symm_mem.rendezvous(
        all_gather_top_experts_buffer, group_name
    )
    all_gather_top_experts_buffer_multicast_ptr = int(
        all_gather_top_experts_buffer_handle.multicast_ptr
    )

    barrier_buffer = symm_mem.empty(1, dtype=torch.int32, device=device)
    barrier_buffer.zero_()
    barrier_buffer_handle = symm_mem.rendezvous(barrier_buffer, group_name)
    barrier_buffer_ptrs = [
        int(barrier_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]
    barrier_buffer_multicast_ptr = int(barrier_buffer_handle.multicast_ptr)
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)
    combine_completion = torch.zeros(1, dtype=torch.int32, device=device)
    barrier_expected_scratch = torch.zeros(1, dtype=torch.int32, device=device)
    input_expected_scratch = torch.zeros(1, dtype=torch.int32, device=device)
    tile_ready = torch.zeros(
        schedule_capacity // 64, dtype=torch.int32, device=device
    )
    down_ready = torch.zeros(
        schedule_capacity // 64, dtype=torch.int32, device=device
    )
    ticket_counter = torch.zeros(1, dtype=torch.int32, device=device)
    worker_ticket = torch.zeros(1024, dtype=torch.int32, device=device)
    # Host-mapped pinned trap record: readable by the CPU after a device
    # trap poisons the context (no CUDA call needed).  Zeroed here only --
    # never in the per-iteration prepare phase, so a competing call's
    # first-writer record survives for post-mortem.
    trap_record = torch.zeros(8, dtype=torch.int64).pin_memory()
    trap_record_ptr = trap_record.data_ptr()
    in_use = torch.zeros(1, dtype=torch.int32, device=device)
    epilogue_done = torch.zeros(1, dtype=torch.int32, device=device)
    # Warm the K1 occupancy cache for THIS workspace's device while we are
    # guaranteed to be outside any CUDA graph capture; keep the measured
    # cluster occupancy for acceptance records.
    k1_max_clusters = fp8_block_dispatch_gemm_prewarm(
        epilogue_done.device.index
    )
    print(
        f"MOK_K1_OCCUPANCY|device={epilogue_done.device.index}"
        f"|max_clusters={k1_max_clusters}",
        flush=True,
    )

    dist.barrier(
        group=group, async_op=True, device_ids=[device_index]
    ).block_current_stream()

    return MoKFP8RouteWorkspace(
        group_name=group_name,
        ep_rank=ep_rank,
        ep_size=ep_size,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_local_experts=num_local_experts,
        schedule_capacity=schedule_capacity,
        x_buffer=x_buffer,
        x_buffer_handle=x_buffer_handle,
        x_buffer_ptrs=x_buffer_ptrs,
        x_scale_buffer=x_scale_buffer,
        x_scale_buffer_handle=x_scale_buffer_handle,
        x_scale_buffer_ptrs=x_scale_buffer_ptrs,
        combine_buffer=combine_buffer,
        combine_buffer_handle=combine_buffer_handle,
        combine_buffer_ptrs=combine_buffer_ptrs,
        output=output,
        routed_x=routed_x,
        routed_x_scale=routed_x_scale,
        m_indices=m_indices,
        schedule_peer_rank=schedule_peer_rank,
        schedule_peer_token_idx=schedule_peer_token_idx,
        schedule_num_tokens=schedule_num_tokens,
        schedule_tokens_per_expert=schedule_tokens_per_expert,
        schedule_tokens_per_expert_and_peer=(
            schedule_tokens_per_expert_and_peer
        ),
        all_gather_top_experts_buffer=all_gather_top_experts_buffer,
        all_gather_top_experts_buffer_handle=all_gather_top_experts_buffer_handle,
        all_gather_top_experts_buffer_multicast_ptr=(
            all_gather_top_experts_buffer_multicast_ptr
        ),
        barrier_buffer=barrier_buffer,
        barrier_buffer_handle=barrier_buffer_handle,
        barrier_buffer_ptrs=barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr=barrier_buffer_multicast_ptr,
        barrier_target=barrier_target,
        combine_completion=combine_completion,
        barrier_expected_scratch=barrier_expected_scratch,
        input_expected_scratch=input_expected_scratch,
        tile_ready=tile_ready,
        down_ready=down_ready,
        ticket_counter=ticket_counter,
        worker_ticket=worker_ticket,
        trap_record=trap_record,
        trap_record_ptr=trap_record_ptr,
        in_use=in_use,
        epilogue_done=epilogue_done,
    )


def create_fp8_terminal_workspace(
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    schedule_capacity: int,
    num_local_experts: int,
    comm_clusters: int = 1,
    compute_clusters: int | None = None,
) -> MoKFP8TerminalWorkspace:
    """Allocate the graph-stable storage for the terminal H20 forward.

    The first terminal specialization is intentionally fixed to EP4,
    H4096/I2048/top-6.  ``schedule_capacity`` is physical routed-row storage,
    not the active device-side token count, and must therefore already include
    padding to an M64 boundary.  The production ``from_topk`` orchestrator
    additionally requires M256 because the fused schedule builder does; the
    M64 allocation contract remains available to the independent-schedule EP4
    probe.  Occupancy is prewarmed here, outside graph capture, to derive or
    validate ``compute_clusters``.  The factory does not launch a forward or
    prepare route flags for a particular iteration.
    """
    import os

    hidden_size = 4096
    intermediate_size = 2048
    topk = 6
    ep_size_required = 4
    m_tile = 64
    w13_n128_tiles = intermediate_size // 128

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    device = torch.device("cuda", device_index)
    if device_index != torch.cuda.current_device():
        raise ValueError("terminal workspace device must be the current CUDA device")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise NotImplementedError("the terminal FP8 workspace requires SM90")
    if os.environ.get("MOK_SM90_EXPERIMENTAL") != "1":
        raise NotImplementedError(
            "MoK terminal SM90 support is experimental; set "
            "MOK_SM90_EXPERIMENTAL=1 to proceed"
        )
    if type(num_local_tokens) is not int or num_local_tokens <= 0:
        raise ValueError("num_local_tokens must be a positive integer")
    if (
        type(schedule_capacity) is not int
        or schedule_capacity <= 0
        or schedule_capacity % m_tile != 0
    ):
        raise ValueError(
            "schedule_capacity must be a positive multiple of 64"
        )
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError(
            "schedule_capacity must hold at least one rank's routed tokens"
        )
    if (
        type(num_local_experts) is not int
        or not 1 <= num_local_experts <= 256
    ):
        raise ValueError("num_local_experts must be an integer in [1, 256]")
    if type(comm_clusters) is not int or comm_clusters <= 0:
        raise ValueError("comm_clusters must be a positive integer")
    max_compute_clusters = fp8_block_megakernel_prewarm(
        device_index, comm_clusters
    )
    if compute_clusters is None:
        compute_clusters = max_compute_clusters
    elif (
        type(compute_clusters) is not int
        or compute_clusters <= 0
        or compute_clusters > max_compute_clusters
    ):
        raise ValueError(
            "compute_clusters must be in [1, "
            f"{max_compute_clusters}] for this terminal kernel/device"
        )

    group_name = group.group_name
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError("process group must have a nonempty group_name")
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    if ep_size != ep_size_required:
        raise ValueError("the terminal FP8 workspace currently requires EP4")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError("current process is not a member of the EP group")

    padded_num_local_tokens = (
        (num_local_tokens + m_tile - 1) // m_tile * m_tile
    )
    m_tiles = schedule_capacity // m_tile

    # Symmetric allocations must have the same shape and rendezvous order on
    # every rank.  Check the complete graph bucket before entering rendezvous.
    local_shape = torch.tensor(
        [
            num_local_tokens,
            padded_num_local_tokens,
            schedule_capacity,
            num_local_experts,
            comm_clusters,
            compute_clusters,
        ],
        dtype=torch.int64,
        device=device,
    )
    gathered_shapes = torch.empty(
        ep_size * local_shape.numel(), dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered_shapes, local_shape, group=group)
    gathered_shapes = gathered_shapes.view(ep_size, local_shape.numel())
    if not torch.all(gathered_shapes == local_shape).item():
        raise ValueError(
            "terminal workspaces require identical graph buckets on all EP ranks"
        )

    x_buffer = symm_mem.empty(
        num_local_tokens,
        hidden_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    x_buffer_handle = symm_mem.rendezvous(x_buffer, group_name)
    x_buffer_ptrs = [
        int(x_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]
    x_buffer_bytes_per_rank = [
        int(x_buffer.untyped_storage().nbytes())
        for _ in range(ep_size)
    ]

    x_scale_buffer = symm_mem.empty(
        num_local_tokens,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    x_scale_buffer_handle = symm_mem.rendezvous(x_scale_buffer, group_name)
    x_scale_buffer_ptrs = [
        int(x_scale_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]
    x_scale_buffer_bytes_per_rank = [
        int(x_scale_buffer.untyped_storage().nbytes())
        for _ in range(ep_size)
    ]

    combine_buffer = symm_mem.empty(
        padded_num_local_tokens * topk,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    combine_buffer_handle = symm_mem.rendezvous(combine_buffer, group_name)
    combine_buffer_ptrs = [
        int(combine_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]

    # Invalid routes are initialized to one and valid routes to zero by the
    # future leased prepare step.  Do not blanket-zero this storage here.
    route_ready = symm_mem.empty(
        padded_num_local_tokens,
        topk,
        dtype=torch.int32,
        device=device,
    )
    route_ready_handle = symm_mem.rendezvous(route_ready, group_name)
    route_ready_ptrs = [
        int(route_ready_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]

    barrier_buffer = symm_mem.empty(1, dtype=torch.int32, device=device)
    barrier_buffer.zero_()
    barrier_buffer_handle = symm_mem.rendezvous(barrier_buffer, group_name)
    barrier_buffer_ptrs = [
        int(barrier_buffer_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(ep_size)
    ]
    barrier_buffer_multicast_ptr = int(barrier_buffer_handle.multicast_ptr)
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)
    input_expected_scratch = torch.zeros(
        1, dtype=torch.int32, device=device
    )

    routed_x = torch.empty(
        schedule_capacity,
        hidden_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    routed_x_scale = torch.empty(
        schedule_capacity,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    m_indices = torch.empty(
        schedule_capacity, dtype=torch.int32, device=device
    )
    schedule_peer_rank = torch.empty(
        schedule_capacity, dtype=torch.int32, device=device
    )
    schedule_peer_token_idx = torch.empty_like(schedule_peer_rank)
    schedule_num_tokens = torch.empty(1, dtype=torch.int32, device=device)
    schedule_tokens_per_expert = torch.empty(
        num_local_experts, dtype=torch.int32, device=device
    )
    schedule_tokens_per_expert_and_peer = torch.empty(
        num_local_experts * ep_size, dtype=torch.int32, device=device
    )

    # The terminal orchestrator owns routing too: keep its all-gather and
    # schedule outputs in this workspace instead of borrowing a second, large
    # MoKFP8RouteWorkspace.  The buffers are graph-stable and never allocated
    # from a forward call.
    all_gather_top_experts_buffer = symm_mem.empty(
        ep_size,
        num_local_tokens,
        topk,
        dtype=torch.int32,
        device=device,
    )
    all_gather_top_experts_buffer_handle = symm_mem.rendezvous(
        all_gather_top_experts_buffer, group_name
    )
    all_gather_top_experts_buffer_multicast_ptr = int(
        all_gather_top_experts_buffer_handle.multicast_ptr
    )
    gate_up = torch.empty(
        schedule_capacity,
        2 * intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    down_input = torch.empty(
        schedule_capacity,
        intermediate_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    down_input_scale = torch.empty(
        schedule_capacity,
        intermediate_size // 128,
        dtype=torch.float32,
        device=device,
    )
    routed_y = torch.empty(
        schedule_capacity,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    # Capacity-sized readiness arrays and graph-stable cursor/closure state.
    x_routed_ready = torch.zeros(m_tiles, dtype=torch.int32, device=device)
    gate_up_tile_ready = torch.zeros(
        m_tiles, w13_n128_tiles, dtype=torch.int32, device=device
    )
    hidden_row_block_ready = torch.zeros(
        m_tiles, dtype=torch.int32, device=device
    )
    y_routed_ready = torch.zeros(m_tiles, dtype=torch.int32, device=device)
    y_routed_done = torch.zeros(m_tiles, dtype=torch.int32, device=device)
    epilogue_claim = torch.zeros(
        padded_num_local_tokens, dtype=torch.int32, device=device
    )
    next_logical_cluster = torch.zeros(1, dtype=torch.int32, device=device)
    next_reduce_probe = torch.zeros(1, dtype=torch.int32, device=device)
    role_cursor = torch.zeros(1, dtype=torch.int32, device=device)
    cluster_role = torch.full(
        (comm_clusters + compute_clusters,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    dispatch_tile_cursor = torch.zeros(1, dtype=torch.int32, device=device)
    dispatch_tiles_done = torch.zeros(1, dtype=torch.int32, device=device)
    push_tile_cursor = torch.zeros(1, dtype=torch.int32, device=device)
    worker_ticket = torch.zeros(
        compute_clusters, dtype=torch.int32, device=device
    )
    comm_owner = torch.full((1,), -1, dtype=torch.int32, device=device)
    comm_worker_ticket = torch.zeros(
        comm_clusters, dtype=torch.int32, device=device
    )
    producer_done = torch.zeros(1, dtype=torch.int32, device=device)
    comm_closed = torch.zeros(1, dtype=torch.int32, device=device)
    push_done = torch.zeros(1, dtype=torch.int32, device=device)
    reduce_done = torch.zeros(1, dtype=torch.int32, device=device)
    terminate = torch.zeros(1, dtype=torch.int32, device=device)
    epilogue_done = torch.zeros(1, dtype=torch.int32, device=device)
    in_use = torch.zeros(1, dtype=torch.int32, device=device)
    # Keep the fatal record host-readable after a device trap poisons the CUDA
    # context.  The future terminal entry resolves this host address to the
    # device mapping; no per-forward allocation or CUDA call is needed here.
    trap_record = torch.zeros(8, dtype=torch.int64).pin_memory()
    trap_record_ptr = trap_record.data_ptr()

    dist.barrier(
        group=group, async_op=True, device_ids=[device_index]
    ).block_current_stream()

    print(
        f"MOK_TERMINAL_OCCUPANCY|device={device_index}"
        f"|max_compute_clusters={max_compute_clusters}"
        f"|comm_clusters={comm_clusters}"
        f"|compute_clusters={compute_clusters}"
        f"|physical_clusters={comm_clusters + compute_clusters}",
        flush=True,
    )

    return MoKFP8TerminalWorkspace(
        group_name=group_name,
        ep_rank=ep_rank,
        ep_size=ep_size,
        device=device,
        num_local_tokens=num_local_tokens,
        padded_num_local_tokens=padded_num_local_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
        num_local_experts=num_local_experts,
        schedule_capacity=schedule_capacity,
        comm_clusters=comm_clusters,
        compute_clusters=compute_clusters,
        max_compute_clusters=max_compute_clusters,
        x_buffer=x_buffer,
        x_buffer_handle=x_buffer_handle,
        x_buffer_ptrs=x_buffer_ptrs,
        x_buffer_bytes_per_rank=x_buffer_bytes_per_rank,
        x_scale_buffer=x_scale_buffer,
        x_scale_buffer_handle=x_scale_buffer_handle,
        x_scale_buffer_ptrs=x_scale_buffer_ptrs,
        x_scale_buffer_bytes_per_rank=x_scale_buffer_bytes_per_rank,
        combine_buffer=combine_buffer,
        combine_buffer_handle=combine_buffer_handle,
        combine_buffer_ptrs=combine_buffer_ptrs,
        route_ready=route_ready,
        route_ready_handle=route_ready_handle,
        route_ready_ptrs=route_ready_ptrs,
        barrier_buffer=barrier_buffer,
        barrier_buffer_handle=barrier_buffer_handle,
        barrier_buffer_ptrs=barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr=barrier_buffer_multicast_ptr,
        barrier_target=barrier_target,
        input_expected_scratch=input_expected_scratch,
        routed_x=routed_x,
        routed_x_scale=routed_x_scale,
        m_indices=m_indices,
        schedule_peer_rank=schedule_peer_rank,
        schedule_peer_token_idx=schedule_peer_token_idx,
        schedule_num_tokens=schedule_num_tokens,
        schedule_tokens_per_expert=schedule_tokens_per_expert,
        schedule_tokens_per_expert_and_peer=(
            schedule_tokens_per_expert_and_peer
        ),
        all_gather_top_experts_buffer=all_gather_top_experts_buffer,
        all_gather_top_experts_buffer_handle=(
            all_gather_top_experts_buffer_handle
        ),
        all_gather_top_experts_buffer_multicast_ptr=(
            all_gather_top_experts_buffer_multicast_ptr
        ),
        gate_up=gate_up,
        down_input=down_input,
        down_input_scale=down_input_scale,
        routed_y=routed_y,
        x_routed_ready=x_routed_ready,
        gate_up_tile_ready=gate_up_tile_ready,
        hidden_row_block_ready=hidden_row_block_ready,
        y_routed_ready=y_routed_ready,
        y_routed_done=y_routed_done,
        epilogue_claim=epilogue_claim,
        next_logical_cluster=next_logical_cluster,
        next_reduce_probe=next_reduce_probe,
        role_cursor=role_cursor,
        cluster_role=cluster_role,
        dispatch_tile_cursor=dispatch_tile_cursor,
        dispatch_tiles_done=dispatch_tiles_done,
        push_tile_cursor=push_tile_cursor,
        worker_ticket=worker_ticket,
        comm_owner=comm_owner,
        comm_worker_ticket=comm_worker_ticket,
        producer_done=producer_done,
        comm_closed=comm_closed,
        push_done=push_done,
        reduce_done=reduce_done,
        terminate=terminate,
        epilogue_done=epilogue_done,
        in_use=in_use,
        trap_record=trap_record,
        trap_record_ptr=trap_record_ptr,
    )


def get_fp8_terminal_workspace(
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    schedule_capacity: int,
    num_local_experts: int,
    comm_clusters: int = 1,
    compute_clusters: int | None = None,
) -> MoKFP8TerminalWorkspace:
    """Return a cached graph-stable terminal workspace.

    ``None`` and an explicit compute-cluster count are distinct cache keys:
    callers that request the occupancy-derived default must keep that choice
    stable for the lifetime of a graph bucket.
    """
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError("device must be a CUDA torch.device")
    if compute_clusters is not None and (
        type(compute_clusters) is not int or compute_clusters <= 0
    ):
        raise ValueError("compute_clusters must be None or a positive integer")
    if type(comm_clusters) is not int or comm_clusters <= 0:
        raise ValueError("comm_clusters must be a positive integer")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    cluster_key = -1 if compute_clusters is None else compute_clusters
    cache_key = (
        group.group_name,
        device_index,
        num_local_tokens,
        schedule_capacity,
        num_local_experts,
        comm_clusters,
        cluster_key,
    )
    cached_workspace = _FP8_TERMINAL_WORKSPACE_CACHE.get(cache_key)
    if cached_workspace is not None:
        return cached_workspace

    workspace = create_fp8_terminal_workspace(
        group,
        device=torch.device("cuda", device_index),
        num_local_tokens=num_local_tokens,
        schedule_capacity=schedule_capacity,
        num_local_experts=num_local_experts,
        comm_clusters=comm_clusters,
        compute_clusters=compute_clusters,
    )
    _FP8_TERMINAL_WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def get_fp8_route_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
    num_local_experts: int,
) -> MoKFP8RouteWorkspace:
    """Return a cached production FP8 route workspace."""
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        min_num_local_tokens=2,
        num_local_tokens_alignment=2,
    )
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    ep_size = dist.get_world_size(group=group)
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    schedule_capacity_factor = max(
        2, math.ceil(ep_size * config.schedule_capacity_multiplier)
    )
    cache_key = (
        group.group_name,
        device_index,
        num_local_tokens,
        hidden_size,
        topk,
        num_local_experts,
        schedule_capacity_factor,
    )
    cached_workspace = _FP8_ROUTE_WORKSPACE_CACHE.get(cache_key)
    if cached_workspace is not None:
        return cached_workspace

    workspace = create_fp8_route_workspace(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_local_experts=num_local_experts,
    )
    _FP8_ROUTE_WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def get_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> MoKWorkspace:
    """Returns a cached workspace, creating and caching one if absent.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        workspace: MoKWorkspace
    """
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    cache_key = (group_name, device_index, num_local_tokens, hidden_size, topk, schedule_capacity_factor)
    cached_workspace = _WORKSPACE_CACHE.get(cache_key)
    if cached_workspace is not None:
        return cached_workspace

    workspace = create_workspace(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    _WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def acquire_workspace_lease(
    workspace: MoKFP8RouteWorkspace | MoKFP8TerminalWorkspace,
) -> None:
    """Explicit lease acquire for orchestrators that span multiple entries."""
    workspace_lease_acquire(
        workspace.in_use, workspace.trap_record_ptr, workspace.ep_rank
    )


def release_workspace_lease(
    workspace: MoKFP8RouteWorkspace | MoKFP8TerminalWorkspace,
) -> None:
    """Explicit trailing lease release (counterpart of acquire above)."""
    workspace_lease_release(workspace.in_use)


def format_trap_record(
    workspace: MoKFP8RouteWorkspace | MoKFP8TerminalWorkspace,
) -> str | None:
    """Post-mortem reader: call AFTER a CUDA API returned an error, and do
    not issue further CUDA calls first -- the record lives in host-mapped
    pinned memory precisely so this read needs no working context.  Returns
    None when no trap fired OR while a trap is claimed but its payload is
    not yet committed (slot[0] holds the ~0 sentinel, which reads as -1 in
    the int64 view; two-phase publication guarantees the payload is complete
    once the final code lands)."""
    head = int(workspace.trap_record[0].item())
    if head == 0 or head == -1:
        return None
    rec = workspace.trap_record.tolist()
    return (
        "MOK_TRAP|code=%d|site=%d|slot=%d|expected=%d|observed=%d"
        "|rank=%d|ticket=%d|iters=%d" % tuple(rec)
    )


def format_terminal_transaction_failure(
    workspace: MoKFP8TerminalWorkspace,
    phase: str,
    error: BaseException,
) -> str:
    """Format a CPU-only fatal receipt after a terminal lease was attempted.

    Once terminal acquire is enqueued, callers cannot know whether a later
    exception left a usable CUDA context or an acquired workspace.  They must
    not issue a release kernel or continue serving.  This helper only reads the
    host-mapped trap record and formats a receipt for an immediate process
    exit; it deliberately performs no CUDA operation.
    """
    try:
        trap = format_trap_record(workspace)
    except BaseException:
        trap = None
    if trap is not None:
        return trap
    return (
        "MOK_TERMINAL_TRANSACTION_FATAL"
        f"|phase={phase}|error_type={type(error).__name__}"
    )


def clear_workspace_cache() -> None:
    """Clears cached workspaces after all participating ranks synchronize.

    Inputs:
        None

    Outputs:
        None
    """
    workspaces = (
        list(_WORKSPACE_CACHE.values())
        + list(_FP8_ROUTE_WORKSPACE_CACHE.values())
        + list(_FP8_TERMINAL_WORKSPACE_CACHE.values())
    )
    for workspace in workspaces:
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                    workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        torch.cuda.synchronize(workspace.device)
    _WORKSPACE_CACHE.clear()
    _FP8_ROUTE_WORKSPACE_CACHE.clear()
    _FP8_TERMINAL_WORKSPACE_CACHE.clear()


def _validate_build_schedule_inputs(
    workspace: MoKWorkspace | MoKFP8RouteWorkspace | MoKFP8TerminalWorkspace,
    config: MoKConfig,
    top_experts: torch.Tensor,
    *,
    num_local_experts: int,
    expert_padding: int,
) -> torch.Tensor:
    """Validate every fallible host-side schedule condition before launch."""
    if not isinstance(
        workspace,
        (MoKWorkspace, MoKFP8RouteWorkspace, MoKFP8TerminalWorkspace),
    ):
        raise TypeError("workspace must be a MoK workspace")
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    device_properties = torch.cuda.get_device_properties(workspace.device)
    if type(config.fwd_num_comm_sms) is not int or config.fwd_num_comm_sms <= 0:
        raise ValueError("fwd_num_comm_sms must be a positive integer")
    if config.fwd_num_comm_sms % 2 != 0:
        raise ValueError("fwd_num_comm_sms must be even")
    if type(config.bwd_num_comm_sms) is not int or config.bwd_num_comm_sms <= 0:
        raise ValueError("bwd_num_comm_sms must be a positive integer")
    if config.bwd_num_comm_sms % 2 != 0:
        raise ValueError("bwd_num_comm_sms must be even")
    if (config.fwd_num_comm_sms >= device_properties.multi_processor_count
            or config.bwd_num_comm_sms >= device_properties.multi_processor_count):
        raise ValueError("communication-SM counts must leave at least one compute SM")
    if (type(config.minibatch_size) is not int or config.minibatch_size <= 0
            or config.minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(config.macrobatch_size) is not int or config.macrobatch_size <= 0
            or config.macrobatch_size % config.minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if (type(config.all_gather_top_experts_chunk_bytes) is not int
            or config.all_gather_top_experts_chunk_bytes <= 0
            or config.all_gather_top_experts_chunk_bytes % 16 != 0):
        raise ValueError("all_gather_top_experts_chunk_bytes must be a positive multiple of 16")
    if (config.all_gather_top_experts_chunk_bytes + 1024
            > device_properties.shared_memory_per_block_optin):
        raise ValueError("all_gather_top_experts_chunk_bytes exceeds device dynamic shared-memory capacity")
    route_buffer_bytes = workspace.num_local_tokens * workspace.topk * 4
    if route_buffer_bytes % config.all_gather_top_experts_chunk_bytes != 0:
        raise ValueError("all_gather_top_experts_chunk_bytes must divide one rank's route-buffer bytes")
    if not top_experts.is_cuda or top_experts.device != workspace.device:
        raise ValueError("top_experts must be on the workspace CUDA device")
    if top_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError("top_experts must have dtype torch.int32 or torch.int64")
    if not top_experts.is_contiguous():
        raise ValueError("top_experts must be contiguous")
    if tuple(top_experts.shape) != (workspace.num_local_tokens, workspace.topk):
        raise ValueError("top_experts must have shape (num_local_tokens, topk)")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if (
        isinstance(workspace, (MoKFP8RouteWorkspace, MoKFP8TerminalWorkspace))
        and num_local_experts != workspace.num_local_experts
    ):
        raise ValueError(
            "num_local_experts must match the FP8 workspace"
        )
    if type(expert_padding) is not int or expert_padding not in (64, 128, 256):
        raise ValueError("expert_padding must be one of 64, 128, 256")
    if (
        isinstance(workspace, (MoKFP8RouteWorkspace, MoKFP8TerminalWorkspace))
        and (
            workspace.schedule_capacity <= 0
            or workspace.schedule_capacity % 256 != 0
            or workspace.schedule_capacity
            < workspace.num_local_tokens * workspace.topk
        )
    ):
        raise ValueError(
            "FP8 fused schedule capacity must be positive, M256 aligned, "
            "and hold all local routes"
        )

    return (
        top_experts
        if top_experts.dtype == torch.int32
        else top_experts.to(torch.int32)
    )


def _build_schedule_validated(
    workspace: MoKWorkspace | MoKFP8RouteWorkspace | MoKFP8TerminalWorkspace,
    config: MoKConfig,
    top_experts_int32: torch.Tensor,
    *,
    num_local_experts: int,
    expert_padding: int,
) -> MoKSchedule:
    """Launch schedule construction after host validation has succeeded."""
    if isinstance(workspace, (MoKFP8RouteWorkspace, MoKFP8TerminalWorkspace)):
        fp8_block_build_schedule_out(
            top_experts_int32,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            workspace.ep_rank,
            config.all_gather_top_experts_chunk_bytes,
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
            workspace.schedule_peer_rank,
            workspace.schedule_peer_token_idx,
            workspace.schedule_num_tokens,
            workspace.schedule_tokens_per_expert,
            workspace.schedule_tokens_per_expert_and_peer,
            expert_padding,
        )
        return MoKSchedule(
            peer_rank=workspace.schedule_peer_rank,
            peer_token_idx=workspace.schedule_peer_token_idx,
            num_tokens=workspace.schedule_num_tokens,
            tokens_per_expert=workspace.schedule_tokens_per_expert,
            expert_padding=expert_padding,
        )

    all_gather_top_experts(
        top_experts_int32, workspace.all_gather_top_experts_buffer,
        workspace.all_gather_top_experts_buffer_multicast_ptr, workspace.ep_rank,
        config.all_gather_top_experts_chunk_bytes,
    )
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    (schedule_peer_rank, schedule_peer_token_idx,
     num_tokens, tokens_per_expert) = schedule(
        workspace.all_gather_top_experts_buffer, num_local_experts,
        workspace.schedule_capacity, workspace.ep_rank, expert_padding)
    return MoKSchedule(
        peer_rank=schedule_peer_rank, peer_token_idx=schedule_peer_token_idx,
        num_tokens=num_tokens, tokens_per_expert=tokens_per_expert,
        expert_padding=expert_padding,
    )


def build_schedule(
    workspace: MoKWorkspace | MoKFP8RouteWorkspace | MoKFP8TerminalWorkspace,
    config: MoKConfig,
    top_experts: torch.Tensor,
    *,
    num_local_experts: int,
    expert_padding: int = 256,
) -> MoKSchedule:
    """All-gather routes and build a graph-stable padded expert schedule.

    FP8 route and terminal workspaces both own the all-gather and schedule
    outputs they mutate.  The terminal top-level orchestrator calls the same
    validated implementation only after acquiring its workspace lease.  A
    direct terminal call is therefore a leased sub-operation; callers must
    already own the terminal lease and must follow it with
    ``megakernel_fp8_block_leased``.  Production callers should use
    ``megakernel_fp8_block_from_topk`` instead.
    """
    top_experts_int32 = _validate_build_schedule_inputs(
        workspace,
        config,
        top_experts,
        num_local_experts=num_local_experts,
        expert_padding=expert_padding,
    )
    return _build_schedule_validated(
        workspace,
        config,
        top_experts_int32,
        num_local_experts=num_local_experts,
        expert_padding=expert_padding,
    )


def dispatch_fp8_block(
    workspace: MoKFP8RouteWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    *,
    trim_to_active_rows: bool = False,
    prepare_combine: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy and dispatch production FP8/K128 activations on the current stream.

    ``trim_to_active_rows`` pays one device-to-host synchronization to read
    ``schedule.num_tokens``. Storage remains capacity-sized, while downstream
    compute receives views containing only the padded rows with real routes.
    """
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    expected_x_shape = (workspace.num_local_tokens, workspace.hidden_size)
    if (
        not x.is_cuda
        or x.device != workspace.device
        or x.dtype != torch.float8_e4m3fn
        or not x.is_contiguous()
        or tuple(x.shape) != expected_x_shape
    ):
        raise ValueError(
            "x must be contiguous CUDA float8_e4m3fn with shape "
            f"{expected_x_shape}"
        )
    expected_scale_shape = (
        workspace.num_local_tokens,
        workspace.hidden_size // 128,
    )
    if (
        not x_scale.is_cuda
        or x_scale.device != workspace.device
        or x_scale.dtype != torch.float32
        or not x_scale.is_contiguous()
        or tuple(x_scale.shape) != expected_scale_shape
    ):
        raise ValueError(
            "x_scale must be contiguous CUDA float32 with shape "
            f"{expected_scale_shape}"
        )
    if type(trim_to_active_rows) is not bool:
        raise TypeError("trim_to_active_rows must be a bool")
    if type(prepare_combine) is not bool:
        raise TypeError("prepare_combine must be a bool")
    active_rows = (
        int(schedule.num_tokens.item())
        if trim_to_active_rows
        else workspace.schedule_capacity
    )
    if (
        active_rows < 0
        or active_rows > workspace.schedule_capacity
        or (active_rows != 0 and active_rows % schedule.expert_padding != 0)
    ):
        raise RuntimeError(
            "schedule num_tokens must be zero or expert-padding aligned "
            "within capacity"
        )

    routed_x = workspace.routed_x[:active_rows]
    routed_x_scale = workspace.routed_x_scale[:active_rows]
    m_indices = workspace.m_indices[:active_rows]
    if prepare_combine:
        # The fused dispatch barrier runs after this clear and after publishing
        # the symmetric input buffers.  It therefore also proves that every
        # destination is clear before any later peer combine stores begin.
        workspace.combine_buffer.zero_()
        # Reset the fused-barrier expected slot for this iteration; the
        # epilogue spins on it becoming nonzero, so zeroing must happen
        # strictly before combine publishes it (stream order does that).
        workspace.barrier_expected_scratch.zero_()
    fp8_block_routed_dispatch_copy_out(
        x,
        workspace.x_buffer,
        workspace.x_buffer_ptrs,
        x_scale,
        workspace.x_scale_buffer,
        workspace.x_scale_buffer_ptrs,
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
        routed_x,
        routed_x_scale,
        m_indices,
        schedule.peer_rank[:active_rows],
        schedule.peer_token_idx[:active_rows],
        schedule.num_tokens,
        schedule.tokens_per_expert,
        workspace.topk,
    )
    return routed_x, routed_x_scale, m_indices


def _validate_terminal_forward(
    workspace: MoKFP8TerminalWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    minibatch_rows: int,
    macrobatch_rows: int,
    swiglu_limit: float,
    spin_limit: int,
    *,
    inputs_preloaded: bool = False,
) -> None:
    """Host-only validation run before the terminal lease is acquired."""
    if not isinstance(workspace, MoKFP8TerminalWorkspace):
        raise TypeError("workspace must be a MoKFP8TerminalWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    if (
        workspace.ep_size != 4
        or not 0 <= workspace.ep_rank < 4
        or workspace.hidden_size != 4096
        or workspace.intermediate_size != 2048
        or workspace.topk != 6
        or not 1 <= workspace.num_local_experts <= 256
        or workspace.schedule_capacity <= 0
        or workspace.schedule_capacity % 64 != 0
        or workspace.padded_num_local_tokens < workspace.num_local_tokens
        or workspace.padded_num_local_tokens % 64 != 0
        or workspace.comm_clusters <= 0
        or workspace.compute_clusters <= 0
        or workspace.compute_clusters > workspace.max_compute_clusters
    ):
        raise ValueError(
            "terminal workspace must satisfy the fixed "
            "EP4/H4096/I2048/top-6 contract"
        )
    if workspace.device.type != "cuda":
        raise ValueError("terminal workspace must be on CUDA")
    device = workspace.device

    def tensor(
        name: str,
        value: torch.Tensor,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_cuda
            or value.device != device
            or value.dtype != dtype
            or not value.is_contiguous()
            or tuple(value.shape) != shape
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA {dtype} with shape {shape} "
                f"on {device}"
            )

    def peer_ptrs(name: str, values: list[int]) -> None:
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(type(value) is not int or value <= 0 for value in values)
        ):
            raise ValueError(
                f"{name} must contain exactly four positive pointers"
            )

    local_tokens = workspace.num_local_tokens
    padded_tokens = workspace.padded_num_local_tokens
    capacity = workspace.schedule_capacity
    experts = workspace.num_local_experts
    m_tiles = capacity // 64
    tensor(
        "workspace.x_buffer",
        workspace.x_buffer,
        torch.float8_e4m3fn,
        (local_tokens, 4096),
    )
    tensor(
        "workspace.x_scale_buffer",
        workspace.x_scale_buffer,
        torch.float32,
        (local_tokens, 32),
    )
    peer_ptrs("workspace.x_buffer_ptrs", workspace.x_buffer_ptrs)
    peer_ptrs("workspace.x_scale_buffer_ptrs", workspace.x_scale_buffer_ptrs)
    x_storage_bytes = int(workspace.x_buffer.untyped_storage().nbytes())
    x_scale_storage_bytes = int(
        workspace.x_scale_buffer.untyped_storage().nbytes()
    )
    if (
        not isinstance(workspace.x_buffer_bytes_per_rank, list)
        or workspace.x_buffer_bytes_per_rank != [x_storage_bytes] * 4
        or not isinstance(workspace.x_scale_buffer_bytes_per_rank, list)
        or workspace.x_scale_buffer_bytes_per_rank
        != [x_scale_storage_bytes] * 4
    ):
        raise ValueError(
            "terminal symmetric source byte capacities must match on all ranks"
        )
    if (
        workspace.x_buffer_ptrs[workspace.ep_rank] != workspace.x_buffer.data_ptr()
        or workspace.x_scale_buffer_ptrs[workspace.ep_rank]
        != workspace.x_scale_buffer.data_ptr()
    ):
        raise ValueError(
            "rank-local symmetric x pointers must alias workspace storage"
        )
    tensor("x", x, torch.float8_e4m3fn, (local_tokens, 4096))
    tensor("x_scale", x_scale, torch.float32, (local_tokens, 32))
    tensor(
        "workspace.routed_x",
        workspace.routed_x,
        torch.float8_e4m3fn,
        (capacity, 4096),
    )
    tensor(
        "workspace.routed_x_scale",
        workspace.routed_x_scale,
        torch.float32,
        (capacity, 32),
    )
    raw_bulk_tensors = (
        ("workspace.x_buffer", workspace.x_buffer),
        ("workspace.x_scale_buffer", workspace.x_scale_buffer),
        ("workspace.routed_x", workspace.routed_x),
        ("workspace.routed_x_scale", workspace.routed_x_scale),
    )
    if any(
        tensor.data_ptr() != tensor.untyped_storage().data_ptr()
        for _, tensor in raw_bulk_tensors
    ):
        raise ValueError(
            "terminal raw bulk-TMA tensors must begin at their storage base"
        )
    validate_terminal_tma_dispatch_layout(
        workspace.x_buffer_ptrs,
        workspace.x_buffer_bytes_per_rank,
        workspace.x_scale_buffer_ptrs,
        workspace.x_scale_buffer_bytes_per_rank,
        required_x_bytes=(
            workspace.x_buffer.numel() * workspace.x_buffer.element_size()
        ),
        required_x_scale_bytes=(
            workspace.x_scale_buffer.numel()
            * workspace.x_scale_buffer.element_size()
        ),
        routed_x_pointer=workspace.routed_x.data_ptr(),
        routed_x_bytes=int(workspace.routed_x.untyped_storage().nbytes()),
        routed_x_scale_pointer=workspace.routed_x_scale.data_ptr(),
        routed_x_scale_bytes=int(
            workspace.routed_x_scale.untyped_storage().nbytes()
        ),
    )
    tensor("workspace.m_indices", workspace.m_indices, torch.int32, (capacity,))
    tensor(
        "workspace.schedule_peer_rank",
        workspace.schedule_peer_rank,
        torch.int32,
        (capacity,),
    )
    tensor(
        "workspace.schedule_peer_token_idx",
        workspace.schedule_peer_token_idx,
        torch.int32,
        (capacity,),
    )
    tensor(
        "workspace.schedule_num_tokens",
        workspace.schedule_num_tokens,
        torch.int32,
        (1,),
    )
    tensor(
        "workspace.schedule_tokens_per_expert",
        workspace.schedule_tokens_per_expert,
        torch.int32,
        (experts,),
    )
    tensor(
        "workspace.schedule_tokens_per_expert_and_peer",
        workspace.schedule_tokens_per_expert_and_peer,
        torch.int32,
        (experts * 4,),
    )
    tensor(
        "workspace.all_gather_top_experts_buffer",
        workspace.all_gather_top_experts_buffer,
        torch.int32,
        (4, local_tokens, 6),
    )
    if (
        type(workspace.all_gather_top_experts_buffer_multicast_ptr) is not int
        or workspace.all_gather_top_experts_buffer_multicast_ptr <= 0
    ):
        raise ValueError(
            "terminal all-gather multicast pointer must be positive"
        )
    tensor(
        "workspace.gate_up",
        workspace.gate_up,
        torch.bfloat16,
        (capacity, 4096),
    )
    tensor(
        "workspace.down_input",
        workspace.down_input,
        torch.float8_e4m3fn,
        (capacity, 2048),
    )
    tensor(
        "workspace.down_input_scale",
        workspace.down_input_scale,
        torch.float32,
        (capacity, 16),
    )
    tensor(
        "workspace.routed_y",
        workspace.routed_y,
        torch.bfloat16,
        (capacity, 4096),
    )
    tensor(
        "workspace.combine_buffer",
        workspace.combine_buffer,
        torch.bfloat16,
        (padded_tokens * 6, 4096),
    )
    tensor(
        "workspace.route_ready",
        workspace.route_ready,
        torch.int32,
        (padded_tokens, 6),
    )
    peer_ptrs("workspace.combine_buffer_ptrs", workspace.combine_buffer_ptrs)
    peer_ptrs("workspace.route_ready_ptrs", workspace.route_ready_ptrs)
    peer_ptrs("workspace.barrier_buffer_ptrs", workspace.barrier_buffer_ptrs)
    if (
        workspace.combine_buffer_ptrs[workspace.ep_rank]
        != workspace.combine_buffer.data_ptr()
        or workspace.route_ready_ptrs[workspace.ep_rank]
        != workspace.route_ready.data_ptr()
    ):
        raise ValueError(
            "rank-local symmetric combine pointers must alias workspace storage"
        )
    tensor(
        "workspace.x_routed_ready",
        workspace.x_routed_ready,
        torch.int32,
        (m_tiles,),
    )
    tensor(
        "workspace.gate_up_tile_ready",
        workspace.gate_up_tile_ready,
        torch.int32,
        (m_tiles, 16),
    )
    for name, value in (
        ("hidden_row_block_ready", workspace.hidden_row_block_ready),
        ("y_routed_ready", workspace.y_routed_ready),
        ("y_routed_done", workspace.y_routed_done),
    ):
        tensor(f"workspace.{name}", value, torch.int32, (m_tiles,))
    tensor(
        "workspace.epilogue_claim",
        workspace.epilogue_claim,
        torch.int32,
        (padded_tokens,),
    )
    tensor(
        "workspace.cluster_role",
        workspace.cluster_role,
        torch.int32,
        (workspace.comm_clusters + workspace.compute_clusters,),
    )
    tensor(
        "workspace.worker_ticket",
        workspace.worker_ticket,
        torch.int32,
        (workspace.compute_clusters,),
    )
    tensor(
        "workspace.comm_worker_ticket",
        workspace.comm_worker_ticket,
        torch.int32,
        (workspace.comm_clusters,),
    )
    for name, value in (
        ("next_logical_cluster", workspace.next_logical_cluster),
        ("next_reduce_probe", workspace.next_reduce_probe),
        ("role_cursor", workspace.role_cursor),
        ("dispatch_tile_cursor", workspace.dispatch_tile_cursor),
        ("dispatch_tiles_done", workspace.dispatch_tiles_done),
        ("push_tile_cursor", workspace.push_tile_cursor),
        ("comm_owner", workspace.comm_owner),
        ("producer_done", workspace.producer_done),
        ("comm_closed", workspace.comm_closed),
        ("push_done", workspace.push_done),
        ("reduce_done", workspace.reduce_done),
        ("terminate", workspace.terminate),
        ("epilogue_done", workspace.epilogue_done),
        ("in_use", workspace.in_use),
        ("barrier_buffer", workspace.barrier_buffer),
        ("barrier_target", workspace.barrier_target),
        ("input_expected_scratch", workspace.input_expected_scratch),
    ):
        tensor(f"workspace.{name}", value, torch.int32, (1,))
    if (
        type(workspace.barrier_buffer_multicast_ptr) is not int
        or workspace.barrier_buffer_multicast_ptr <= 0
        or type(workspace.trap_record_ptr) is not int
        or workspace.trap_record_ptr <= 0
    ):
        raise ValueError("terminal workspace device pointers must be positive")
    if (
        workspace.trap_record.is_cuda
        or workspace.trap_record.dtype != torch.int64
        or not workspace.trap_record.is_contiguous()
        or tuple(workspace.trap_record.shape) != (8,)
        or not workspace.trap_record.is_pinned()
        or workspace.trap_record_ptr != workspace.trap_record.data_ptr()
    ):
        raise ValueError(
            "trap_record must be contiguous pinned CPU int64 [8] and its "
            "pointer must match trap_record_ptr"
        )

    tensor(
        "schedule.peer_rank",
        schedule.peer_rank,
        torch.int32,
        (capacity,),
    )
    tensor(
        "schedule.peer_token_idx",
        schedule.peer_token_idx,
        torch.int32,
        (capacity,),
    )
    tensor("schedule.num_tokens", schedule.num_tokens, torch.int32, (1,))
    tensor(
        "schedule.tokens_per_expert",
        schedule.tokens_per_expert,
        torch.int32,
        (experts,),
    )
    if schedule.expert_padding != 64:
        raise ValueError("terminal schedules must use expert_padding=64")
    tensor(
        "w13", w13, torch.float8_e4m3fn, (experts, 4096, 4096)
    )
    tensor("w13_scale", w13_scale, torch.float32, (experts, 32, 32))
    tensor("w2", w2, torch.float8_e4m3fn, (experts, 4096, 2048))
    tensor("w2_scale", w2_scale, torch.float32, (experts, 32, 16))
    tensor(
        "topk_weights", topk_weights, torch.float32, (local_tokens, 6)
    )
    tensor("topk_ids", topk_ids, torch.int32, (local_tokens, 6))
    tensor("output", output, torch.bfloat16, (local_tokens, 4096))
    if (
        type(minibatch_rows) is not int
        or minibatch_rows <= 0
        or minibatch_rows > (1 << 31) - 1
        or minibatch_rows % 64 != 0
        or type(macrobatch_rows) is not int
        or macrobatch_rows <= 0
        or macrobatch_rows > (1 << 31) - 1
        or macrobatch_rows % minibatch_rows != 0
    ):
        raise ValueError(
            "minibatch_rows must be a positive M64 multiple and "
            "macrobatch_rows must be a positive multiple of minibatch_rows"
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

    # Release happens inside the terminal kernel.  Its output must outlive the
    # released workspace and therefore cannot share storage with any workspace
    # tensor or another input that the kernel consumes.
    output_storage = output.untyped_storage().data_ptr()
    workspace_storages = {
        value.untyped_storage().data_ptr()
        for field in workspace.__dataclass_fields__
        if isinstance((value := getattr(workspace, field)), torch.Tensor)
    }
    if inputs_preloaded:
        if x is not workspace.x_buffer or x_scale is not workspace.x_scale_buffer:
            raise ValueError(
                "preloaded terminal inputs must be the workspace x buffers"
            )
    elif (
        x.untyped_storage().data_ptr() in workspace_storages
        or x_scale.untyped_storage().data_ptr() in workspace_storages
    ):
        raise ValueError(
            "x and x_scale must be caller-owned and must not already alias "
            "terminal workspace storage"
        )
    forbidden = set(workspace_storages)
    forbidden.update(
        value.untyped_storage().data_ptr()
        for value in (
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            x,
            x_scale,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
        )
    )
    if output_storage in forbidden:
        raise ValueError(
            "output must be caller-owned contiguous storage and must not "
            "alias the terminal workspace, schedule, weights, or routes"
        )


def megakernel_fp8_block_leased(
    workspace: MoKFP8TerminalWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    minibatch_rows: int = 4096,
    macrobatch_rows: int = 131072,
    swiglu_limit: float = 10.0,
    spin_limit: int = 1 << 27,
    inputs_preloaded: bool = False,
) -> torch.Tensor:
    """Launch a prevalidated terminal forward while its lease is held.

    This is the only leased sub-entry.  It deliberately performs no host-side
    validation and no lease transition: the owning wrapper must validate every
    argument before acquire.  ``inputs_preloaded=True`` is reserved for the
    from-topk preloaded entry and requires the exact symmetric workspace
    tensors; it skips both input copies.  The terminal kernel itself releases
    the lease after materializing ``output``; there is no Python release tail.
    """
    if inputs_preloaded:
        if x is not workspace.x_buffer or x_scale is not workspace.x_scale_buffer:
            raise RuntimeError(
                "preloaded terminal inputs must be the workspace x buffers"
            )
    else:
        workspace.x_buffer.copy_(x)
        workspace.x_scale_buffer.copy_(x_scale)
    fp8_block_megakernel_prepare_out(
        topk_ids,
        workspace.route_ready,
        workspace.x_routed_ready,
        workspace.gate_up_tile_ready,
        workspace.hidden_row_block_ready,
        workspace.y_routed_ready,
        workspace.y_routed_done,
        workspace.epilogue_claim,
        workspace.next_logical_cluster,
        workspace.next_reduce_probe,
        workspace.role_cursor,
        workspace.cluster_role,
        workspace.dispatch_tile_cursor,
        workspace.dispatch_tiles_done,
        workspace.push_tile_cursor,
        workspace.worker_ticket,
        workspace.comm_owner,
        workspace.comm_worker_ticket,
        workspace.producer_done,
        workspace.comm_closed,
        workspace.push_done,
        workspace.reduce_done,
        workspace.terminate,
        workspace.epilogue_done,
        workspace.input_expected_scratch,
    )
    fp8_block_megakernel_out(
        workspace.x_buffer,
        workspace.x_buffer_ptrs,
        workspace.x_buffer_bytes_per_rank,
        workspace.x_scale_buffer,
        workspace.x_scale_buffer_ptrs,
        workspace.x_scale_buffer_bytes_per_rank,
        workspace.routed_x,
        workspace.routed_x_scale,
        workspace.m_indices,
        schedule.peer_rank,
        schedule.peer_token_idx,
        schedule.num_tokens,
        schedule.tokens_per_expert,
        w13,
        w13_scale,
        workspace.gate_up,
        workspace.down_input,
        workspace.down_input_scale,
        w2,
        w2_scale,
        workspace.routed_y,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        workspace.route_ready,
        workspace.route_ready_ptrs,
        topk_weights,
        topk_ids,
        output,
        workspace.x_routed_ready,
        workspace.gate_up_tile_ready,
        workspace.hidden_row_block_ready,
        workspace.y_routed_ready,
        workspace.y_routed_done,
        workspace.epilogue_claim,
        workspace.next_logical_cluster,
        workspace.next_reduce_probe,
        workspace.role_cursor,
        workspace.cluster_role,
        workspace.dispatch_tile_cursor,
        workspace.dispatch_tiles_done,
        workspace.push_tile_cursor,
        workspace.worker_ticket,
        workspace.comm_owner,
        workspace.comm_worker_ticket,
        workspace.producer_done,
        workspace.comm_closed,
        workspace.push_done,
        workspace.reduce_done,
        workspace.terminate,
        workspace.epilogue_done,
        workspace.in_use,
        workspace.barrier_buffer,
        workspace.barrier_target,
        workspace.input_expected_scratch,
        workspace.barrier_buffer_multicast_ptr,
        workspace.trap_record_ptr,
        workspace.ep_rank,
        workspace.comm_clusters,
        workspace.compute_clusters,
        minibatch_rows,
        macrobatch_rows,
        swiglu_limit,
        spin_limit,
    )
    return output


def megakernel_fp8_block(
    workspace: MoKFP8TerminalWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    minibatch_rows: int = 4096,
    macrobatch_rows: int = 131072,
    swiglu_limit: float = 10.0,
    spin_limit: int = 1 << 27,
) -> torch.Tensor:
    """Owned manual-schedule terminal entry retained for EP4 probes.

    The supplied schedule must already exist independently of this terminal
    workspace.  This wrapper validates first, acquires exactly once, and then
    delegates to :func:`megakernel_fp8_block_leased`.  Production integration
    should use :func:`megakernel_fp8_block_from_topk`, which builds the
    workspace-owned schedule only after acquiring the same lease.
    """
    require_fp8_block_megakernel()
    if not isinstance(workspace, MoKFP8TerminalWorkspace):
        raise TypeError("workspace must be a MoKFP8TerminalWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    terminal_schedule_storages = {
        workspace.schedule_peer_rank.untyped_storage().data_ptr(),
        workspace.schedule_peer_token_idx.untyped_storage().data_ptr(),
        workspace.schedule_num_tokens.untyped_storage().data_ptr(),
        workspace.schedule_tokens_per_expert.untyped_storage().data_ptr(),
    }
    if any(
        value.untyped_storage().data_ptr() in terminal_schedule_storages
        for value in (
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
        )
    ):
        raise ValueError(
            "workspace-owned schedules require megakernel_fp8_block_from_topk; "
            "the manual owned entry accepts only an independent schedule"
        )
    _validate_terminal_forward(
        workspace,
        schedule,
        x,
        x_scale,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        minibatch_rows,
        macrobatch_rows,
        swiglu_limit,
        spin_limit,
    )
    workspace_lease_acquire(
        workspace.in_use, workspace.trap_record_ptr, workspace.ep_rank
    )
    return megakernel_fp8_block_leased(
        workspace,
        schedule,
        x,
        x_scale,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        minibatch_rows=minibatch_rows,
        macrobatch_rows=macrobatch_rows,
        swiglu_limit=swiglu_limit,
        spin_limit=spin_limit,
    )


def _terminal_workspace_schedule(
    workspace: MoKFP8TerminalWorkspace,
) -> MoKSchedule:
    return MoKSchedule(
        peer_rank=workspace.schedule_peer_rank,
        peer_token_idx=workspace.schedule_peer_token_idx,
        num_tokens=workspace.schedule_num_tokens,
        tokens_per_expert=workspace.schedule_tokens_per_expert,
        expert_padding=64,
    )


def _validate_and_acquire_terminal_from_topk(
    workspace: MoKFP8TerminalWorkspace,
    config: MoKConfig,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    swiglu_limit: float = 10.0,
    spin_limit: int = 1 << 27,
    inputs_preloaded: bool = False,
) -> None:
    if not isinstance(workspace, MoKFP8TerminalWorkspace):
        raise TypeError("workspace must be a MoKFP8TerminalWorkspace")
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    schedule = _terminal_workspace_schedule(workspace)
    require_fp8_block_megakernel()
    _validate_terminal_forward(
        workspace,
        schedule,
        x,
        x_scale,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        config.minibatch_size,
        config.macrobatch_size,
        swiglu_limit,
        spin_limit,
        inputs_preloaded=inputs_preloaded,
    )
    topk_ids_int32 = _validate_build_schedule_inputs(
        workspace,
        config,
        topk_ids,
        num_local_experts=workspace.num_local_experts,
        expert_padding=64,
    )
    # The terminal forward contract already requires int32, so validation
    # above must not materialize a conversion before the lease boundary.
    if topk_ids_int32 is not topk_ids:
        raise RuntimeError("terminal schedule validation unexpectedly copied routes")

    workspace_lease_acquire(
        workspace.in_use, workspace.trap_record_ptr, workspace.ep_rank
    )


def acquire_megakernel_fp8_block_from_topk_lease(
    workspace: MoKFP8TerminalWorkspace,
    config: MoKConfig,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    swiglu_limit: float = 10.0,
    spin_limit: int = 1 << 27,
) -> None:
    """Validate and acquire a terminal lease before direct input population.

    This is the first half of the preloaded from-topk transaction.  It
    validates the complete terminal call against ``workspace.x_buffer`` and
    ``workspace.x_scale_buffer`` without writing either tensor, then launches
    the normal graph-safe lease acquire.  After acquire is attempted, every
    exception is process-fatal: callers must not issue a release kernel or
    continue with a possibly acquired/poisoned workspace.  On success, the
    caller must write both buffers on the same stream and invoke
    :func:`megakernel_fp8_block_from_topk_preloaded_leased`.
    """
    _validate_and_acquire_terminal_from_topk(
        workspace,
        config,
        workspace.x_buffer,
        workspace.x_scale_buffer,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        swiglu_limit=swiglu_limit,
        spin_limit=spin_limit,
        inputs_preloaded=True,
    )


def megakernel_fp8_block_from_topk_preloaded_leased(
    workspace: MoKFP8TerminalWorkspace,
    config: MoKConfig,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    swiglu_limit: float = 10.0,
    spin_limit: int = 1 << 27,
) -> torch.Tensor:
    """Build the route schedule and run terminal on leased preloaded inputs.

    All host validation and the lease transition belong to
    :func:`acquire_megakernel_fp8_block_from_topk_lease`.  This half performs
    only stream-ordered device work and never copies the symmetric input
    buffers.  The terminal kernel releases the lease after writing ``output``.
    """
    schedule = _build_schedule_validated(
        workspace,
        config,
        topk_ids,
        num_local_experts=workspace.num_local_experts,
        expert_padding=64,
    )
    return megakernel_fp8_block_leased(
        workspace,
        schedule,
        workspace.x_buffer,
        workspace.x_scale_buffer,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        minibatch_rows=config.minibatch_size,
        macrobatch_rows=config.macrobatch_size,
        swiglu_limit=swiglu_limit,
        spin_limit=spin_limit,
        inputs_preloaded=True,
    )


def megakernel_fp8_block_from_topk(
    workspace: MoKFP8TerminalWorkspace,
    config: MoKConfig,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    swiglu_limit: float = 10.0,
    spin_limit: int = 1 << 27,
) -> torch.Tensor:
    """Own one strict route-to-output terminal transaction.

    All host validation completes before lease acquisition.  Device work is
    then stream ordered as acquire -> build schedule -> copy inputs -> prepare
    -> one terminal compute kernel.  The leased sub-entry cannot reacquire,
    and the terminal kernel releases after writing caller-owned ``output``.
    No fallback, dynamic allocation, host read, or Python release occurs in
    this forward path.
    """
    _validate_and_acquire_terminal_from_topk(
        workspace,
        config,
        x,
        x_scale,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        swiglu_limit=swiglu_limit,
        spin_limit=spin_limit,
    )
    schedule = _build_schedule_validated(
        workspace,
        config,
        topk_ids,
        num_local_experts=workspace.num_local_experts,
        expert_padding=64,
    )
    return megakernel_fp8_block_leased(
        workspace,
        schedule,
        x,
        x_scale,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        minibatch_rows=config.minibatch_size,
        macrobatch_rows=config.macrobatch_size,
        swiglu_limit=swiglu_limit,
        spin_limit=spin_limit,
    )


def dispatch_gemm_fused_fp8_block(
    workspace: MoKFP8RouteWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    gate_up: torch.Tensor,
    copy_clusters: int = 8,
    forced_worker_clusters: int = 0,
    delay_ticket0_cycles: int = 0,
    spin_trap_iters: int = 0,
    ticket_visit: torch.Tensor | None = None,
    record_visits: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input barrier, pull dispatch, and gate/up GEMM as one persistent kernel.

    Leased-mode sub-entry: the caller (orchestrator) owns the workspace
    lease across the whole pipeline (acquire_workspace_lease before the
    first workspace write, e.g. build_schedule) -- this function never
    acquires or releases it.

    Strict-contract producer/consumer fusion: communication CTAs pull routed
    rows and publish per-M64-tile ready counters, GEMM CTAs start each tile as
    soon as its own rows have landed.  Replaces the dispatch_fp8_block(...,
    prepare_combine=True) + gate/up grouped-GEMM pair; downstream stages are
    unchanged.  Always operates on the full capacity view (no host reads).
    """
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    expected_x_shape = (workspace.num_local_tokens, workspace.hidden_size)
    if (
        not x.is_cuda
        or x.device != workspace.device
        or x.dtype != torch.float8_e4m3fn
        or not x.is_contiguous()
        or tuple(x.shape) != expected_x_shape
    ):
        raise ValueError(
            "x must be contiguous CUDA float8_e4m3fn with shape "
            f"{expected_x_shape}"
        )
    expected_scale_shape = (
        workspace.num_local_tokens,
        workspace.hidden_size // 128,
    )
    if (
        not x_scale.is_cuda
        or x_scale.device != workspace.device
        or x_scale.dtype != torch.float32
        or not x_scale.is_contiguous()
        or tuple(x_scale.shape) != expected_scale_shape
    ):
        raise ValueError(
            "x_scale must be contiguous CUDA float32 with shape "
            f"{expected_scale_shape}"
        )

    # Same iteration-preparation contract as dispatch_fp8_block with
    # prepare_combine=True: the fused kernel's in-kernel input barrier also
    # proves every rank finished these clears before peer stores begin.
    workspace.combine_buffer.zero_()
    workspace.barrier_expected_scratch.zero_()
    # Producer/consumer handoff state consumed by this very kernel.  Stream
    # order separates these clears from the previous iteration's readers,
    # which is what keeps CUDA graph replay valid with no host counters.
    workspace.input_expected_scratch.zero_()
    workspace.tile_ready.zero_()
    # Iteration state only: trap_record and (future lease state) are never
    # cleared here -- see the K1 redesign's zeroing contract.
    workspace.ticket_counter.zero_()
    # Publish the symmetric input copies; they precede the fused kernel on
    # the stream, so its first-CTA arrive covers them.
    workspace.x_buffer.copy_(x)
    workspace.x_scale_buffer.copy_(x_scale)
    fp8_block_dispatch_gemm_fused_out(
        workspace.x_buffer,
        workspace.x_buffer_ptrs,
        workspace.x_scale_buffer,
        workspace.x_scale_buffer_ptrs,
        workspace.routed_x,
        workspace.routed_x_scale,
        workspace.m_indices,
        schedule.peer_rank,
        schedule.peer_token_idx,
        schedule.num_tokens,
        schedule.tokens_per_expert,
        workspace.topk,
        workspace.barrier_buffer,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
        workspace.input_expected_scratch,
        workspace.tile_ready,
        weight,
        weight_scale,
        gate_up,
        ep_rank=workspace.ep_rank,
        ticket_counter=workspace.ticket_counter,
        worker_ticket=workspace.worker_ticket,
        trap_record_ptr=workspace.trap_record_ptr,
        ticket_visit=(
            ticket_visit if ticket_visit is not None
            else workspace.worker_ticket
        ),
        copy_clusters=copy_clusters,
        forced_worker_clusters=forced_worker_clusters,
        delay_ticket0_cycles=delay_ticket0_cycles,
        spin_trap_iters=spin_trap_iters,
        record_visits=record_visits,
    )
    return workspace.routed_x, workspace.routed_x_scale, workspace.m_indices


def gemm_combine_fused_fp8_block(
    workspace: MoKFP8RouteWorkspace,
    schedule: MoKSchedule,
    down_input: torch.Tensor,
    down_input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    routed_y: torch.Tensor,
    topk_weights: torch.Tensor,
    release_lease: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Down GEMM, last-arriver combine push, fused arrive, waiting epilogue.

    Leased-mode sub-entry.  release_lease=True is set only by the
    orchestrator that owns the lease and ends its pipeline here: the
    epilogue's last-finishing CTA then performs the owner-qualified release.
    When releasing in-pipeline, pass a caller-owned ``output`` tensor so the
    result does not live in the workspace after the lease is gone (a later
    acquirer may overwrite workspace state before the caller reads it).

    Replaces the dynamic down GEMM + precleared combine_reduce pair.  The
    GEMM CTA that completes each M64 block last pushes the block's rows to
    the peers, so no resident communication CTAs are needed.  Requires the
    same-iteration dispatch to have run with prepare_combine semantics
    (combine buffers cleared and proven by the input barrier,
    barrier_expected_scratch zeroed).  down_ready is cleared here,
    stream-ordered after the previous iteration's readers.
    """
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    out = output if output is not None else workspace.output
    if release_lease:
        ref = workspace.output
        workspace_storages = {
            t.untyped_storage().data_ptr()
            for t in (
                workspace.output,
                workspace.combine_buffer,
                workspace.routed_x,
                workspace.routed_x_scale,
                workspace.m_indices,
                workspace.x_buffer,
                workspace.x_scale_buffer,
                workspace.schedule_peer_rank,
                workspace.schedule_peer_token_idx,
            )
        }
        if (
            output is None
            or output.shape != ref.shape
            or output.dtype != ref.dtype
            or output.device != ref.device
            or not output.is_contiguous()
            or output.untyped_storage().data_ptr() in workspace_storages
        ):
            raise ValueError(
                "release_lease=True requires a caller-owned output tensor "
                "(matching shape/dtype/device, contiguous, NOT aliasing any "
                "workspace-owned storage): the workspace may be overwritten "
                "by a new acquirer after release"
            )
    workspace.down_ready.zero_()
    workspace.epilogue_done.zero_()
    fp8_block_gemm_combine_fused_out(
        down_input,
        down_input_scale,
        weight,
        weight_scale,
        workspace.m_indices,
        schedule.num_tokens,
        routed_y,
        schedule.peer_rank,
        schedule.peer_token_idx,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        workspace.topk,
        workspace.down_ready,
        workspace.combine_completion,
        workspace.barrier_target,
        workspace.barrier_expected_scratch,
        workspace.barrier_buffer_multicast_ptr,
    )
    routed_epilogue_fused_out(
        workspace.combine_buffer,
        topk_weights,
        out,
        workspace.barrier_buffer,
        workspace.barrier_expected_scratch,
        workspace.in_use,
        workspace.epilogue_done,
        workspace.trap_record_ptr,
        workspace.ep_rank,
        do_release=1 if release_lease else 0,
    )
    return out


def combine_fp8_block(
    workspace: MoKFP8RouteWorkspace,
    schedule: MoKSchedule,
    routed_y: torch.Tensor,
) -> torch.Tensor:
    """Combine BF16 routed rows and wait until all peer writes are visible."""
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    if (
        not routed_y.is_cuda
        or routed_y.device != workspace.device
        or routed_y.dtype != torch.bfloat16
        or not routed_y.is_contiguous()
        or routed_y.ndim != 2
        or routed_y.shape[1] != workspace.hidden_size
        or routed_y.shape[0] > workspace.schedule_capacity
        or (
            routed_y.shape[0] != 0
            and routed_y.shape[0] % schedule.expert_padding != 0
        )
    ):
        raise ValueError(
            "routed_y must be contiguous CUDA bfloat16 [M,H] with M zero or "
            "expert-padding aligned and no larger than schedule capacity"
        )

    # Every rank must finish clearing its local target before any peer starts
    # remote stores.  This makes invalid/padded route slots deterministic and
    # prevents a late clear on one rank from erasing an early peer write.
    workspace.combine_buffer.zero_()
    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    active_rows = routed_y.shape[0]
    if active_rows:
        fp8_block_routed_combine_out(
            routed_y,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            schedule.peer_rank[:active_rows],
            schedule.peer_token_idx[:active_rows],
            schedule.num_tokens,
            workspace.topk,
        )
    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    return workspace.combine_buffer


def grouped_gemm_fp8_block_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the public caller-owned SM90 contiguous expert GEMM."""
    fp8_block_grouped_contiguous_out(
        input,
        weight,
        input_scale,
        weight_scale,
        m_indices,
        output,
    )
    return output


def grouped_gemm_fp8_block_dynamic_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run grouped GEMM while reading the valid-row count on the device."""
    fp8_block_grouped_contiguous_dynamic_out(
        input,
        weight,
        input_scale,
        weight_scale,
        m_indices,
        num_tokens,
        output,
    )
    return output


def reduce_fp8_block_routes(
    workspace: MoKFP8RouteWorkspace,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply router weights to returned route slots without a shared addend."""
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    expected_shape = (workspace.num_local_tokens, workspace.topk)
    if (
        not topk_weights.is_cuda
        or topk_weights.device != workspace.device
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
        or tuple(topk_weights.shape) != expected_shape
    ):
        raise ValueError(
            "topk_weights must be contiguous CUDA float32 with shape "
            f"{expected_shape}"
        )
    routed_epilogue_out(
        workspace.combine_buffer,
        topk_weights,
        workspace.output,
    )
    return workspace.output


def combine_reduce_fp8_block_routes(
    workspace: MoKFP8RouteWorkspace,
    schedule: MoKSchedule,
    routed_y: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    combine_precleared: bool = False,
) -> torch.Tensor:
    """Combine remote BF16 rows and reduce route slots in one host call.

    ``combine_precleared`` is valid only after the matching dispatch used
    ``prepare_combine=True`` on every rank.  That lets dispatch's existing
    peer barrier cover the early clear and removes the later pre-combine
    barrier from the critical path.
    """
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    if type(combine_precleared) is not bool:
        raise TypeError("combine_precleared must be a bool")
    expected_weights_shape = (workspace.num_local_tokens, workspace.topk)
    if (
        not routed_y.is_cuda
        or routed_y.device != workspace.device
        or routed_y.dtype != torch.bfloat16
        or not routed_y.is_contiguous()
        or routed_y.ndim != 2
        or routed_y.shape[1] != workspace.hidden_size
        or routed_y.shape[0] > workspace.schedule_capacity
        or (
            routed_y.shape[0] != 0
            and routed_y.shape[0] % schedule.expert_padding != 0
        )
    ):
        raise ValueError(
            "routed_y must be contiguous CUDA bfloat16 [M,H] with M zero or "
            "expert-padding aligned and no larger than schedule capacity"
        )
    if (
        not topk_weights.is_cuda
        or topk_weights.device != workspace.device
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
        or tuple(topk_weights.shape) != expected_weights_shape
    ):
        raise ValueError(
            "topk_weights must be contiguous CUDA float32 with shape "
            f"{expected_weights_shape}"
        )
    active_rows = routed_y.shape[0]
    # The fused in-kernel barrier requires the scratch slot zeroed earlier in
    # this iteration, which dispatch's prepare_combine path guarantees; the
    # fused op is therefore tied to the precleared path.
    if combine_precleared:
        fp8_block_routed_combine_reduce_fused_out(
            routed_y,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            schedule.peer_rank[:active_rows],
            schedule.peer_token_idx[:active_rows],
            schedule.num_tokens,
            topk_weights,
            workspace.output,
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
            workspace.topk,
            workspace.combine_completion,
            workspace.barrier_expected_scratch,
        )
        return workspace.output
    fp8_block_routed_combine_reduce_out(
        routed_y,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        schedule.peer_rank[:active_rows],
        schedule.peer_token_idx[:active_rows],
        schedule.num_tokens,
        topk_weights,
        workspace.output,
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
        workspace.topk,
        combine_precleared,
    )
    return workspace.output


def validate_inputs(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    grad_output: torch.Tensor | None = None,
) -> None:
    """Validates runtime inputs against the workspace and schedule.

    Inputs:
        config:         MoKConfig
        workspace:      MoKWorkspace
        schedule:       MoKSchedule
        x:              bfloat16 [num_local_tokens, hidden_size]
        router_weights: float32 [num_local_tokens, topk]
        grad_output:    bfloat16 [num_local_tokens, hidden_size] | None

    Outputs:
        None
    """
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    if not isinstance(workspace, MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    expected_activation_shape = (workspace.num_local_tokens, workspace.hidden_size)
    tensors = [("x", x, torch.bfloat16, expected_activation_shape)]
    if grad_output is not None:
        tensors.append(("grad_output", grad_output, torch.bfloat16, expected_activation_shape))
    tensors.append(("router_weights", router_weights, torch.float32, (workspace.num_local_tokens, workspace.topk)))
    for tensor_name, tensor, expected_dtype, expected_shape in tensors:
        if not tensor.is_cuda or tensor.device != workspace.device or tensor.dtype != expected_dtype or not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous {expected_dtype} on the workspace CUDA device")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} shape does not match the workspace")
    if schedule.peer_rank.numel() != workspace.schedule_capacity:
        raise ValueError("schedule capacity does not match the workspace")


def forward(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
) -> tuple[
    torch.Tensor,
    MoKForwardContext,
]:
    """Runs the MoE forward pass.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 data/scale tuple
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 data/scale tuple
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size] or MXFP8 data/scale tuple
        swiglu_limit:        float | None

    Outputs:
        output:          bfloat16 [num_local_tokens, hidden_size]
        forward_context: MoKForwardContext
    """
    validate_inputs(config, workspace, schedule, x, router_weights)

    workspace.x_buffer.copy_(x)  # TODO: we can remove this
    workspace.router_weight_buffer.copy_(router_weights)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

    if isinstance(routed_gate_weights, tuple):
        routed_gate_weights_fp8, routed_gate_weights_sc = routed_gate_weights
        routed_up_weights_fp8, routed_up_weights_sc = routed_up_weights
        routed_down_weights_fp8, routed_down_weights_sc = routed_down_weights
        (x_fp8_t_routed, x_sc_t_routed,
         gate_shared, gate_fp8_routed, gate_sc_routed,
         up_shared, up_fp8_routed, up_sc_routed,
         hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
         y_shared, y_routed) = dispatch_mlp_swiglu_combine_fwd_mxfp8(
            workspace.x_buffer, workspace.x_buffer_ptrs,
            workspace.combine_buffer, workspace.combine_buffer_ptrs,
            shared_gate_weights, routed_gate_weights_fp8, routed_gate_weights_sc,
            shared_up_weights, routed_up_weights_fp8, routed_up_weights_sc,
            shared_down_weights, routed_down_weights_fp8, routed_down_weights_sc,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.fwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )
        forward_context = MoKForwardContext(
            x_routed=(x_fp8_t_routed, x_sc_t_routed),
            gate_shared=gate_shared,
            gate_routed=(gate_fp8_routed, gate_sc_routed),
            up_shared=up_shared,
            up_routed=(up_fp8_routed, up_sc_routed),
            hidden_shared=hidden_shared,
            hidden_routed=(hidden_fp8_t_routed, hidden_sc_t_routed),
        )
    else:
        (x_routed, gate_shared, gate_routed, up_shared, up_routed,
         hidden_shared, hidden_routed, y_shared, y_routed) = dispatch_mlp_swiglu_combine_fwd_bf16(
            workspace.x_buffer, workspace.x_buffer_ptrs,
            workspace.combine_buffer, workspace.combine_buffer_ptrs,
            shared_gate_weights, routed_gate_weights,
            shared_up_weights, routed_up_weights,
            shared_down_weights, routed_down_weights,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.fwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )
        forward_context = MoKForwardContext(
            x_routed=x_routed,
            gate_shared=gate_shared,
            gate_routed=gate_routed,
            up_shared=up_shared,
            up_routed=up_routed,
            hidden_shared=hidden_shared,
            hidden_routed=hidden_routed,
        )

    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    output = fwd_epilogue(y_shared, workspace.combine_buffer, workspace.router_weight_buffer)
    return output, forward_context


def backward(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    forward_context: MoKForwardContext,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the MoE backward pass.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        forward_context:     MoKForwardContext
        grad_output:         bfloat16 [num_local_tokens, hidden_size]
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 tensor tuple
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 tensor tuple
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size] or MXFP8 tensor tuple
        swiglu_limit:        float | None

    Outputs:
        d_x:                   bfloat16 [num_local_tokens, hidden_size]
        d_router_weights:      float32 [num_local_tokens, topk]
        d_routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size]
        d_shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        d_shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        d_shared_down_weights: bfloat16 [hidden_size, intermediate_size]
    """
    validate_inputs(config, workspace, schedule, x, router_weights, grad_output)
    if not isinstance(forward_context, MoKForwardContext):
        raise TypeError("forward_context must be a MoKForwardContext")

    workspace.d_y_buffer.copy_(grad_output)                # TODO: we can remove this
    workspace.x_buffer.copy_(x)                            # TODO: we can remove this
    workspace.router_weight_buffer.copy_(router_weights)   # TODO: we can remove this
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    if isinstance(routed_gate_weights, tuple):
        (routed_gate_weights_fp8, routed_gate_weights_sc,
         routed_gate_weights_t_fp8, routed_gate_weights_t_sc) = routed_gate_weights
        (routed_up_weights_fp8, routed_up_weights_sc,
         routed_up_weights_t_fp8, routed_up_weights_t_sc) = routed_up_weights
        routed_down_weights_t_fp8, routed_down_weights_t_sc = routed_down_weights
        x_fp8_t_routed, x_sc_t_routed = forward_context.x_routed
        gate_fp8_routed, gate_sc_routed = forward_context.gate_routed
        up_fp8_routed, up_sc_routed = forward_context.up_routed
        hidden_fp8_t_routed, hidden_sc_t_routed = forward_context.hidden_routed
        (d_x_shared, d_x_routed,
         d_gate_shared, d_gate_fp8_routed, d_gate_sc_routed,
         d_up_shared, d_up_fp8_routed, d_up_sc_routed,
         d_hidden_shared, d_hidden_routed, d_y_fp8_routed, d_y_sc_routed,
         d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up,
         d_w_shared_down, d_w_routed_down) = dispatch_mlp_swiglu_combine_bwd_mxfp8(
            workspace.d_y_buffer, workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer, workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer, workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer, workspace.d_router_weight_buffer_ptrs,
            shared_gate_weights, routed_gate_weights_t_fp8, routed_gate_weights_t_sc,
            shared_up_weights, routed_up_weights_t_fp8, routed_up_weights_t_sc,
            shared_down_weights, routed_down_weights_t_fp8, routed_down_weights_t_sc,
            x_fp8_t_routed, x_sc_t_routed,
            forward_context.gate_shared, gate_fp8_routed, gate_sc_routed,
            forward_context.up_shared, up_fp8_routed, up_sc_routed,
            forward_context.hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
            workspace.x_buffer, workspace.x_buffer_ptrs,
            routed_gate_weights_fp8, routed_gate_weights_sc,
            routed_up_weights_fp8, routed_up_weights_sc,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.bwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )
    else:
        x_routed = forward_context.x_routed
        gate_routed = forward_context.gate_routed
        up_routed = forward_context.up_routed
        hidden_routed = forward_context.hidden_routed
        (d_x_shared, d_x_routed, d_gate_shared, d_gate_routed,
         d_up_shared, d_up_routed, d_hidden_shared, d_hidden_routed, d_y_routed,
         d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up,
         d_w_shared_down, d_w_routed_down) = dispatch_mlp_swiglu_combine_bwd_bf16(
            workspace.d_y_buffer, workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer, workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer, workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer, workspace.d_router_weight_buffer_ptrs,
            shared_gate_weights, routed_gate_weights,
            shared_up_weights, routed_up_weights,
            shared_down_weights, routed_down_weights,
            x_routed, forward_context.gate_shared, gate_routed,
            forward_context.up_shared, up_routed,
            forward_context.hidden_shared, hidden_routed,
            workspace.x_buffer, workspace.x_buffer_ptrs,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.bwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )

    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    d_x = bwd_epilogue(d_x_shared, workspace.d_x_routed_buffer)
    d_router_weights = workspace.d_router_weight_buffer.clone()  # TODO: we can remove this
    return (d_x, d_router_weights, d_w_routed_gate, d_w_routed_up, d_w_routed_down,
            d_w_shared_gate, d_w_shared_up, d_w_shared_down)
