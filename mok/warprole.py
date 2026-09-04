"""Python driver for the warp-role megakernel (plan Task 6, steps 3 to 5).

The megakernel replaces the whole split routed-expert sequence

    dispatch_fp8_block
    grouped_gemm_fp8_block_dynamic_out   (W13)
    silu_and_mul_contig_post_quant       (SGLang activation + FP8 quant)
    grouped_gemm_fp8_block_dynamic_out   (W2)
    combine_reduce_fp8_block_routes

with one launch that keeps a producer, one or two consumers and a comm
warpgroup resident in every CTA.  Everything the launch needs beyond the route
workspace lives in :class:`WarpRoleState`: the intermediate FP8 activations and
their K128 scales, the BF16 routed output rows, the four dependency counters,
the symmetric per-rank arrival counter ``push_done`` and the interleaved gate/up
weights.

The route workspace already owns the symmetric input and combine buffers, the
routed FP8 rows, the schedule arrays, the cross-rank barrier and the trap
record, so the state borrows those instead of allocating a second copy.

This module deliberately imports nothing from SGLang: the activation and its
FP8 quantization are fused into the kernel's W13 epilogue, so the only SGLang
dependency left is in the split reference used by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from . import _C
from .functional import (
    MoKFP8RouteWorkspace,
    MoKSchedule,
    acquire_workspace_lease,
    format_trap_record,
    release_workspace_lease,
)
from .ops import interleave_w13

# Frozen model shape, mirroring csrc/sm90_fp8_block_warprole_config.cuh.
HIDDEN = 4096
INTERMEDIATE = 2048
TOPK = 6
M_TILE = 64
MINIBATCH_ROWS = 1024
K_GROUP = 128

# Same defaults the terminal path uses (mok/functional.py megakernel entries).
DEFAULT_SWIGLU_LIMIT = 10.0
DEFAULT_SPIN_LIMIT = 1 << 27

VARIANT_ENTRIES = {
    "c1s6": "fp8_block_warprole_c1s6_out",
    "c2s4": "fp8_block_warprole_c2s4_out",
}


@dataclass(slots=True)
class WarpRoleState:
    """Per-rank launch state for the warp-role megakernel.

    ``routed_x``/``routed_x_scale``/``m_indices`` are the route workspace's own
    buffers; the comm warpgroup writes them exactly as the split dispatch
    kernel does.  Everything else is allocated here because the route workspace
    has no field for it.
    """

    device: torch.device
    ep_rank: int
    ep_size: int
    num_local_tokens: int
    capacity: int
    # Borrowed from the route workspace.
    routed_x: torch.Tensor        # (capacity, 4096) float8_e4m3fn
    routed_x_scale: torch.Tensor  # (capacity, 32) float32
    m_indices: torch.Tensor       # (capacity,) int32
    # Owned by the state.
    hidden: torch.Tensor          # (capacity, 2048) float8_e4m3fn
    hidden_scale: torch.Tensor    # (capacity, 16) float32
    routed_y: torch.Tensor        # (capacity, 4096) bfloat16
    output: torch.Tensor          # (num_local_tokens, 4096) bfloat16
    x_ready: torch.Tensor         # (ceil(capacity/1024),) int32
    hidden_ready: torch.Tensor    # (capacity/64,) int32
    y_ready: torch.Tensor         # (capacity/64,) int32
    push_done_local: torch.Tensor      # (1,) int32
    input_expected_scratch: torch.Tensor  # (1,) int32
    push_done: torch.Tensor       # (1,) int32, symmetric
    push_done_handle: Any
    push_done_ptrs: list[int]
    # id(w13) -> (w13, w13_interleaved, w13_scale_interleaved).  The original
    # tensor is kept so the id cannot be recycled by a freed weight.
    interleaved_w13: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        field(default_factory=dict)
    )


# (id(workspace), capacity) -> (workspace, state).  The workspace is kept alive
# by the cache so its id cannot be recycled under a live state.
_WARPROLE_STATE_CACHE: dict[
    tuple[int, int], tuple[MoKFP8RouteWorkspace, WarpRoleState]
] = {}


def clear_warprole_state_cache() -> None:
    """Drop cached states; call only after every rank has synchronized."""
    _WARPROLE_STATE_CACHE.clear()


def create_warprole_state(
    workspace: MoKFP8RouteWorkspace,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    capacity: int,
) -> WarpRoleState:
    """Allocate the launch state for one route workspace.

    ``capacity`` must equal ``workspace.schedule_capacity``: the C++ entry
    derives the routed row capacity from ``routed_x.size(0)`` and then requires
    the schedule arrays, ``m_indices``, ``hidden`` and ``routed_y`` to have
    exactly that many rows, and the schedule the workspace builds always spans
    its full capacity.
    """
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if type(capacity) is not int or capacity <= 0 or capacity % M_TILE != 0:
        raise ValueError("capacity must be a positive multiple of 64")
    if capacity != workspace.schedule_capacity:
        raise ValueError(
            "warprole capacity must equal the route workspace schedule "
            f"capacity ({workspace.schedule_capacity}); the schedule arrays "
            "and the routed buffers are sized by it"
        )
    if workspace.hidden_size != HIDDEN or workspace.topk != TOPK:
        raise ValueError("warprole is specialized for hidden 4096 and top-6")
    if group.group_name != workspace.group_name:
        raise ValueError("group must be the workspace's process group")
    if dist.get_world_size(group=group) != workspace.ep_size:
        raise ValueError("group world size must match the workspace ep_size")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    device = torch.device("cuda", device_index)
    if device != workspace.device:
        raise ValueError("device must be the workspace device")

    hidden = torch.empty(
        capacity, INTERMEDIATE, dtype=torch.float8_e4m3fn, device=device
    )
    hidden_scale = torch.empty(
        capacity, INTERMEDIATE // K_GROUP, dtype=torch.float32, device=device
    )
    routed_y = torch.empty(capacity, HIDDEN, dtype=torch.bfloat16, device=device)
    output = torch.empty(
        workspace.num_local_tokens, HIDDEN, dtype=torch.bfloat16, device=device
    )
    x_ready = torch.zeros(
        (capacity + MINIBATCH_ROWS - 1) // MINIBATCH_ROWS,
        dtype=torch.int32,
        device=device,
    )
    hidden_ready = torch.zeros(capacity // M_TILE, dtype=torch.int32, device=device)
    y_ready = torch.zeros(capacity // M_TILE, dtype=torch.int32, device=device)
    push_done_local = torch.zeros(1, dtype=torch.int32, device=device)
    input_expected_scratch = torch.zeros(1, dtype=torch.int32, device=device)

    # The symmetric arrival counter every rank's combine_finish adds to.  It
    # must come from symmetric memory: the kernel dereferences the peer
    # pointers directly, and a plain cudaMalloc pointer from another process is
    # not mapped here.  Allocated the way the workspace allocates its barrier.
    push_done = symm_mem.empty(1, dtype=torch.int32, device=device)
    push_done.zero_()
    push_done_handle = symm_mem.rendezvous(push_done, workspace.group_name)
    push_done_ptrs = [
        int(push_done_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(workspace.ep_size)
    ]

    dist.barrier(
        group=group, async_op=True, device_ids=[device_index]
    ).block_current_stream()

    return WarpRoleState(
        device=device,
        ep_rank=workspace.ep_rank,
        ep_size=workspace.ep_size,
        num_local_tokens=workspace.num_local_tokens,
        capacity=capacity,
        routed_x=workspace.routed_x,
        routed_x_scale=workspace.routed_x_scale,
        m_indices=workspace.m_indices,
        hidden=hidden,
        hidden_scale=hidden_scale,
        routed_y=routed_y,
        output=output,
        x_ready=x_ready,
        hidden_ready=hidden_ready,
        y_ready=y_ready,
        push_done_local=push_done_local,
        input_expected_scratch=input_expected_scratch,
        push_done=push_done,
        push_done_handle=push_done_handle,
        push_done_ptrs=push_done_ptrs,
    )


def get_warprole_state(
    workspace: MoKFP8RouteWorkspace,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    capacity: int,
) -> WarpRoleState:
    """Return a cached launch state, creating one on first use.

    Creation rendezvouses symmetric memory and barriers, so every rank must
    reach this call with the same workspace and capacity.
    """
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    cache_key = (id(workspace), capacity)
    cached = _WARPROLE_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached[1]
    state = create_warprole_state(
        workspace, group, device=device, capacity=capacity
    )
    _WARPROLE_STATE_CACHE[cache_key] = (workspace, state)
    return state


def _interleaved_weights(
    state: WarpRoleState, w13: torch.Tensor, w13_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave gate/up N128 blocks once per weight tensor and cache it."""
    cached = state.interleaved_w13.get(id(w13))
    if cached is not None and cached[0] is w13:
        return cached[1], cached[2]
    w13i, w13i_scale = interleave_w13(w13, w13_scale)
    state.interleaved_w13[id(w13)] = (w13, w13i, w13i_scale)
    return w13i, w13i_scale


def _check_tensor(
    name: str,
    value: torch.Tensor,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
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
            f"{name} must be contiguous CUDA {dtype} with shape {shape} on "
            f"{device}"
        )


def warprole_forward(
    workspace: MoKFP8RouteWorkspace,
    state: WarpRoleState,
    schedule: MoKSchedule,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    router_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    variant: str = "c2s4",
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    spin_limit: int = DEFAULT_SPIN_LIMIT,
) -> torch.Tensor:
    """Run the routed expert segment in one launch and return the BF16 output.

    ``x_fp8``/``x_scale`` are this rank's quantized activations.  They are
    published into the workspace's symmetric input buffers -- the same buffers
    the split dispatch publishes into -- because the comm warpgroup pulls its
    routed rows straight out of the peers' copies of them.  Passing the
    workspace buffers themselves skips the copy.

    The returned tensor is ``state.output`` and is overwritten by the next
    call.

    The trap check afterwards is a non-blocking read of the host-mapped record:
    it reports a trap from an earlier launch immediately and one from this
    launch only after the caller synchronizes.  Callers that need a verdict for
    this launch must synchronize first.
    """
    if not isinstance(workspace, MoKFP8RouteWorkspace):
        raise TypeError("workspace must be a MoKFP8RouteWorkspace")
    if not isinstance(state, WarpRoleState):
        raise TypeError("state must be a WarpRoleState")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    if variant not in VARIANT_ENTRIES:
        raise ValueError(f"variant must be one of {sorted(VARIANT_ENTRIES)}")
    entry_name = VARIANT_ENTRIES[variant]
    if not hasattr(_C, entry_name):
        raise RuntimeError(f"the loaded MoK extension lacks {entry_name}")
    if state.capacity != workspace.schedule_capacity or (
        state.routed_x is not workspace.routed_x
    ):
        raise ValueError("state does not belong to this workspace")
    if (
        type(swiglu_limit) not in (int, float)
        or not float(swiglu_limit) > 0.0
    ):
        raise ValueError("swiglu_limit must be a positive number")
    if type(spin_limit) is not int or spin_limit <= 0:
        raise ValueError("spin_limit must be a positive int")

    device = workspace.device
    tokens = workspace.num_local_tokens
    experts = workspace.num_local_experts
    routes = tokens * TOPK
    _check_tensor(
        "x_fp8", x_fp8, torch.float8_e4m3fn, (tokens, HIDDEN), device
    )
    _check_tensor(
        "x_scale", x_scale, torch.float32, (tokens, HIDDEN // K_GROUP), device
    )
    _check_tensor(
        "router_weights", router_weights, torch.float32, (tokens, TOPK), device
    )
    _check_tensor("topk_ids", topk_ids, torch.int32, (tokens, TOPK), device)
    if tuple(workspace.combine_buffer.shape) != (routes, HIDDEN):
        raise ValueError("combine buffer must hold one BF16 slot per route")
    _check_tensor(
        "w13", w13, torch.float8_e4m3fn, (experts, 2 * INTERMEDIATE, HIDDEN), device
    )
    _check_tensor(
        "w13_scale",
        w13_scale,
        torch.float32,
        (experts, 2 * INTERMEDIATE // K_GROUP, HIDDEN // K_GROUP),
        device,
    )
    _check_tensor(
        "w2", w2, torch.float8_e4m3fn, (experts, HIDDEN, INTERMEDIATE), device
    )
    _check_tensor(
        "w2_scale",
        w2_scale,
        torch.float32,
        (experts, HIDDEN // K_GROUP, INTERMEDIATE // K_GROUP),
        device,
    )

    w13i, w13i_scale = _interleaved_weights(state, w13, w13_scale)

    # Publish this rank's activations where the peers' comm warpgroups read
    # them.  The kernel's phase-0 rank barrier is what makes the peers wait for
    # these stores, and stream order makes the copies precede that barrier.
    if x_fp8 is not workspace.x_buffer:
        workspace.x_buffer.copy_(x_fp8)
    if x_scale is not workspace.x_scale_buffer:
        workspace.x_scale_buffer.copy_(x_scale)

    acquire_workspace_lease(workspace)
    _C.fp8_block_warprole_prepare_out(
        x_ready=state.x_ready,
        hidden_ready=state.hidden_ready,
        y_ready=state.y_ready,
        push_done_local=state.push_done_local,
        push_done=state.push_done,
        input_expected_scratch=state.input_expected_scratch,
    )
    getattr(_C, entry_name)(
        x=workspace.x_buffer,
        x_ptrs=workspace.x_buffer_ptrs,
        x_scale=workspace.x_scale_buffer,
        x_scale_ptrs=workspace.x_scale_buffer_ptrs,
        routed_x=state.routed_x,
        routed_x_scale=state.routed_x_scale,
        m_indices=state.m_indices,
        schedule_peer_rank=schedule.peer_rank,
        schedule_peer_token_idx=schedule.peer_token_idx,
        num_tokens=schedule.num_tokens,
        tokens_per_expert=schedule.tokens_per_expert,
        topk=TOPK,
        w13i=w13i,
        w13i_scale=w13i_scale,
        w2=w2,
        w2_scale=w2_scale,
        hidden=state.hidden,
        hidden_scale=state.hidden_scale,
        routed_y=state.routed_y,
        combine_ptrs=workspace.combine_buffer_ptrs,
        combine_local=workspace.combine_buffer,
        weights=router_weights.view(-1),
        topk_ids=topk_ids.view(-1),
        output=state.output,
        push_done_ptrs=state.push_done_ptrs,
        ep_rank=workspace.ep_rank,
        x_ready=state.x_ready,
        hidden_ready=state.hidden_ready,
        y_ready=state.y_ready,
        push_done_local=state.push_done_local,
        barrier_buffer=workspace.barrier_buffer,
        barrier_target=workspace.barrier_target,
        barrier_multicast_ptr=workspace.barrier_buffer_multicast_ptr,
        input_expected_scratch=state.input_expected_scratch,
        trap_record_ptr=workspace.trap_record_ptr,
        swiglu_limit=float(swiglu_limit),
        spin_limit=spin_limit,
    )
    release_workspace_lease(workspace)

    if int(workspace.trap_record[0].item()) != 0:
        raise RuntimeError(
            format_trap_record(workspace)
            or "MOK_TRAP|claimed but payload not committed"
        )
    return state.output
