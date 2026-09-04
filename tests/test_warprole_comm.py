"""Four-rank tests for the warp-role comm warpgroup (plan Task 5).

The comm warpgroup has to move exactly the bytes the split dispatch and combine
kernels move: the same routed FP8 rows, the same K128 scales, the same expert
ids, and the same destination slots on the peers.  These tests pin that down bit
for bit against the split entries and check the three counters the composed
kernel will later depend on -- ``x_ready`` per minibatch, the local CTA arrival
counter ``push_done_local``, and the symmetric per-rank arrival counter
``push_done``.

The routing is uniform on purpose: 2048 tokens per rank, top-6, four ranks, so
every rank receives 12288 routed rows spread evenly over its eight local
experts.  That is the shape the megakernel is designed around (12 minibatches of
1024 rows, 192 M64 tiles) and it keeps both the reference and the candidate on
the same active-row count.

Run on four H20s:

    torchrun --standalone --nproc-per-node=4 -m pytest -s tests/test_warprole_comm.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from mok import _C, functional, ops

NUM_LOCAL_TOKENS = 2048
HIDDEN = 4096
TOPK = 6
NUM_LOCAL_EXPERTS = 8
EXPERT_PADDING = 256
# Mirrors csrc/sm90_fp8_block_warprole_config.cuh: rows per minibatch, rows per
# M64 tile, and geometry<1>::W2_TASKS_PER_TILE = HIDDEN / N_TILE, which is the
# value the combine mode waits for on every active tile.
MINIBATCH_ROWS = 1024
M_TILE = 64
Y_READY_TARGET = HIDDEN // 128
DISPATCH_MODE = 0
COMBINE_MODE = 1
PUSH_DONE_TIMEOUT_S = 30.0


def require_warprole_comm(device: torch.device) -> None:
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("the warp-role comm warpgroup is SM90 only")
    for name in (
        "fp8_block_warprole_comm_bench_out",
        "fp8_block_routed_dispatch_out",
        "fp8_block_routed_combine_out",
    ):
        assert hasattr(_C, name), f"SM90 build did not register {name}"


@dataclass(slots=True)
class Harness:
    """Everything the three tests share: one workspace, one schedule, one input."""

    rank: int
    world_size: int
    device: torch.device
    workspace: functional.MoKFP8RouteWorkspace
    schedule: functional.MoKSchedule
    capacity: int
    active_rows: int
    minibatch_rows: list[int]
    routed_y: torch.Tensor
    x_ready: torch.Tensor
    y_ready: torch.Tensor
    push_done_local: torch.Tensor
    push_done: torch.Tensor
    push_done_ptrs: list[int]
    reference_routed_x: torch.Tensor
    reference_routed_x_scale: torch.Tensor
    reference_m_indices: torch.Tensor
    reference_combine: torch.Tensor


def uniform_top_experts(
    device: torch.device, world_size: int
) -> torch.Tensor:
    """Route token t's j-th slot to rank (t+j)%R, local expert (t//R + j)%E.

    Both indices advance with the route slot, so no token picks one expert
    twice, and over the batch every (rank, local expert) pair receives the same
    number of rows: 2048 * 6 * 4 / (4 * 8) = 1536.  With expert padding 256 that
    lands on exactly 12288 active rows and no padded tail.
    """
    token_index = torch.arange(NUM_LOCAL_TOKENS, device=device).unsqueeze(1)
    route_index = torch.arange(TOPK, device=device).unsqueeze(0)
    destination_rank = (token_index + route_index) % world_size
    local_expert = (
        torch.div(token_index, world_size, rounding_mode="floor") + route_index
    ) % NUM_LOCAL_EXPERTS
    return destination_rank * NUM_LOCAL_EXPERTS + local_expert


def bench_call(harness: Harness, mode: int) -> None:
    """Launch the step-2 microbenchmark kernel in dispatch or combine mode."""
    workspace = harness.workspace
    schedule = harness.schedule
    _C.fp8_block_warprole_comm_bench_out(
        mode,
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
        TOPK,
        harness.routed_y,
        workspace.combine_buffer_ptrs,
        harness.push_done_ptrs,
        workspace.ep_rank,
        harness.x_ready,
        harness.y_ready,
        harness.push_done_local,
    )


def wait_for_push_done(harness: Harness) -> int:
    """Spin on the host until every rank has reported its pushes complete.

    Nothing on the device waits for this counter in the microbenchmark, so the
    host is the only place the cross-rank completion can be observed.  A rank
    that never arrives is a real failure of the release path, not a slow run,
    hence the hard deadline.
    """
    deadline = time.monotonic() + PUSH_DONE_TIMEOUT_S
    observed = int(harness.push_done.item())
    while observed < harness.world_size:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"rank {harness.rank}: push_done stalled at {observed} of "
                f"{harness.world_size} after {PUSH_DONE_TIMEOUT_S:.0f}s"
            )
        time.sleep(0.001)
        observed = int(harness.push_done.item())
    return observed


@pytest.fixture(scope="module")
def harness(context: tuple[int, int, torch.device]) -> Harness:
    rank, world_size, device = context
    require_warprole_comm(device)
    assert world_size == 4, "the step-2 comm contract is EP4"

    config = functional.MoKConfig(schedule_capacity_multiplier=0.5)
    workspace = functional.get_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=NUM_LOCAL_TOKENS,
        hidden_size=HIDDEN,
        topk=TOPK,
        num_local_experts=NUM_LOCAL_EXPERTS,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        uniform_top_experts(device, world_size),
        num_local_experts=NUM_LOCAL_EXPERTS,
        expert_padding=EXPERT_PADDING,
    )
    capacity = workspace.schedule_capacity
    active_rows = int(schedule.num_tokens.item())
    assert active_rows == NUM_LOCAL_TOKENS * TOPK, (
        "uniform routing must fill exactly one local batch worth of rows; "
        f"got {active_rows}"
    )
    assert schedule.tokens_per_expert.tolist() == (
        [active_rows // NUM_LOCAL_EXPERTS] * NUM_LOCAL_EXPERTS
    ), "uniform routing must give every local expert the same row count"
    assert active_rows % MINIBATCH_ROWS == 0 and capacity >= active_rows

    # Publish this rank's activations and their K128 scales into the symmetric
    # buffers the dispatch reads from, then let every rank finish publishing
    # before any peer pulls.  Same generator as the split dispatch/combine test.
    token_index = torch.arange(NUM_LOCAL_TOKENS, device=device).unsqueeze(1)
    columns = torch.arange(HIDDEN, device=device).unsqueeze(0)
    scale_columns = torch.arange(HIDDEN // 128, device=device).unsqueeze(0)
    workspace.x_buffer.copy_(
        ((token_index * HIDDEN + columns) % 31 - 15)
        .add(rank * 0.25)
        .to(torch.float8_e4m3fn)
    )
    workspace.x_scale_buffer.copy_(
        rank * 10000 + token_index * 100 + scale_columns
    )

    # BF16 rows for combine: a per-row term that differs across ranks plus a
    # per-column term, both small integers so bf16 holds them exactly and a
    # shifted row or column shows up as a mismatch rather than as rounding.
    routed_y = torch.empty(
        capacity, HIDDEN, dtype=torch.bfloat16, device=device
    )
    row_term = (
        ((torch.arange(capacity, device=device) + rank * 37) % 251) - 125
    ).to(torch.bfloat16)
    column_term = ((torch.arange(HIDDEN, device=device) % 13) - 6).to(
        torch.bfloat16
    )
    torch.add(row_term.unsqueeze(1), column_term.unsqueeze(0), out=routed_y)

    x_ready = torch.zeros(
        (capacity + MINIBATCH_ROWS - 1) // MINIBATCH_ROWS,
        dtype=torch.int32,
        device=device,
    )
    y_ready = torch.zeros(capacity // M_TILE, dtype=torch.int32, device=device)
    push_done_local = torch.zeros(1, dtype=torch.int32, device=device)
    # The symmetric arrival counter.  The workspace has no push_done field yet
    # (plan Task 6 adds one), so it is allocated here the same way the workspace
    # allocates barrier_buffer: symmetric memory plus a rendezvous, which is what
    # makes the peer pointers below valid in this process -- a plain cudaMalloc
    # pointer from another process would not be mapped here.
    push_done = symm_mem.empty(1, dtype=torch.int32, device=device)
    push_done.zero_()
    push_done_handle = symm_mem.rendezvous(push_done, workspace.group_name)
    push_done_ptrs = [
        int(push_done_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(world_size)
    ]

    torch.cuda.synchronize()
    dist.barrier()

    # Split dispatch into private buffers: the reference for test 1.  The tail
    # past the active rows is deliberately left untouched by both paths, so both
    # sides start from zero and stay comparable over the whole capacity.
    reference_routed_x = torch.empty_like(workspace.routed_x).fill_(0)
    reference_routed_x_scale = torch.zeros_like(workspace.routed_x_scale)
    reference_m_indices = torch.zeros_like(workspace.m_indices)
    ops.fp8_block_routed_dispatch_out(
        workspace.x_buffer,
        workspace.x_buffer_ptrs,
        workspace.x_scale_buffer,
        workspace.x_scale_buffer_ptrs,
        reference_routed_x,
        reference_routed_x_scale,
        reference_m_indices,
        schedule.peer_rank,
        schedule.peer_token_idx,
        schedule.num_tokens,
        schedule.tokens_per_expert,
        TOPK,
    )
    torch.cuda.synchronize()
    dist.barrier()

    # Split combine into the workspace's own symmetric buffer: the reference for
    # test 2.  combine_fp8_block clears the destination, barriers, pushes, and
    # barriers again, so the snapshot below is the settled result.
    reference_combine = functional.combine_fp8_block(
        workspace, schedule, routed_y
    ).clone()
    torch.cuda.synchronize()
    dist.barrier()

    minibatch_rows = [
        min(MINIBATCH_ROWS, active_rows - start)
        for start in range(0, active_rows, MINIBATCH_ROWS)
    ]
    return Harness(
        rank=rank,
        world_size=world_size,
        device=device,
        workspace=workspace,
        schedule=schedule,
        capacity=capacity,
        active_rows=active_rows,
        minibatch_rows=minibatch_rows,
        routed_y=routed_y,
        x_ready=x_ready,
        y_ready=y_ready,
        push_done_local=push_done_local,
        push_done=push_done,
        push_done_ptrs=push_done_ptrs,
        reference_routed_x=reference_routed_x,
        reference_routed_x_scale=reference_routed_x_scale,
        reference_m_indices=reference_m_indices,
        reference_combine=reference_combine,
    )


def check_dispatch(harness: Harness) -> None:
    """Run dispatch mode once and compare it with the split dispatch."""
    workspace = harness.workspace
    workspace.routed_x.fill_(0)
    workspace.routed_x_scale.zero_()
    workspace.m_indices.zero_()
    harness.x_ready.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    bench_call(harness, DISPATCH_MODE)
    torch.cuda.synchronize()
    dist.barrier()

    assert torch.equal(
        workspace.routed_x.view(torch.uint8),
        harness.reference_routed_x.view(torch.uint8),
    ), "dispatched FP8 rows differ from the split dispatch"
    assert not torch.isnan(workspace.routed_x_scale).any(), (
        "dispatched K128 scales contain NaN"
    )
    assert torch.equal(
        workspace.routed_x_scale, harness.reference_routed_x_scale
    ), "dispatched K128 scales differ from the split dispatch"
    assert torch.equal(workspace.m_indices, harness.reference_m_indices), (
        "dispatched expert ids differ from the split dispatch"
    )

    expected_x_ready = [0] * harness.x_ready.numel()
    for minibatch, rows in enumerate(harness.minibatch_rows):
        expected_x_ready[minibatch] = rows
    assert harness.x_ready.tolist() == expected_x_ready, (
        "x_ready must hold each minibatch's row count and nothing else"
    )


def check_combine(harness: Harness) -> None:
    """Run combine mode once and compare it with the split combine."""
    workspace = harness.workspace
    workspace.combine_buffer.zero_()
    harness.push_done.zero_()
    harness.push_done_local.zero_()
    harness.y_ready.zero_()
    # The consumers are idle in this mode, so the host stands in for them and
    # publishes the finished-tile count for every active M64 tile.
    harness.y_ready[: harness.active_rows // M_TILE].fill_(Y_READY_TARGET)
    torch.cuda.synchronize()
    dist.barrier()

    bench_call(harness, COMBINE_MODE)
    torch.cuda.synchronize()

    assert wait_for_push_done(harness) == harness.world_size, (
        "every rank must report its pushes exactly once"
    )
    assert int(harness.push_done_local.item()) == 0, (
        "push_done_local must be reset by the last CTA for the next launch"
    )
    assert not torch.isnan(workspace.combine_buffer).any(), (
        "combined rows contain NaN"
    )
    assert torch.equal(workspace.combine_buffer, harness.reference_combine), (
        "combined rows differ from the split combine"
    )
    dist.barrier()


def test_warprole_comm_dispatch(harness: Harness) -> None:
    check_dispatch(harness)


def test_warprole_comm_combine(harness: Harness) -> None:
    check_combine(harness)


def test_warprole_comm_repeats(harness: Harness) -> None:
    """Both modes must survive a second launch on the same workspace.

    The counters are the reason: x_ready is cleared by the host before each
    launch, push_done_local is cleared by the last CTA of the previous launch,
    and push_done is cleared by the host between combine rounds.  A launch that
    left any of them dirty shows up on the second round as a wrong row count, a
    nonzero push_done_local, or a combine that never reaches four arrivals.
    """
    for _ in range(2):
        check_dispatch(harness)
        check_combine(harness)
