"""Stage-C stress coverage for the K1 resident-worker ticket queue.

In-suite half of the Phase 1 hard gates (the destructive trap negatives run
in the separate subprocess gate script, since a device trap poisons the
whole process group):

- worker-count sweep {1, C-1, C, C+1, occupancy-max}: R < C progression and
  the consecutive multi-task-per-worker path (a single cluster executes
  every copy AND GEMM ticket back to back);
- >= 1000 fused iterations against a fixed split reference;
- resident-but-slow producer: ticket-0 delayed by a bounded busy-wait after
  its claim, before the arrive/copy;
- CUDA-graph capture with >= 1000 replays;
- single active M64 tile with a large inactive capacity tail.
"""

import pytest
import torch
import torch.distributed as dist

from mok import _C
from mok import functional

from .test_sm90_worker import require_sm90


def _build_case(context, route, num_local_tokens=512):
    rank, world_size, device = context
    require_sm90(device)
    assert world_size in (4, 8, 16, 32, 64)
    if not hasattr(_C, "fp8_block_dispatch_gemm_fused_out"):
        pytest.skip("extension lacks the fused dispatch+GEMM kernel")

    hidden_size = 256
    topk = 1
    num_local_experts = 2
    n = 256
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=4096,
        schedule_capacity_multiplier=1.0,
    )
    workspace = functional.get_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_local_experts=num_local_experts,
    )
    token_indices = torch.arange(num_local_tokens, device=device)
    if route == "balanced":
        destination_ranks = token_indices % world_size
        local_experts = ((token_indices // world_size) % 4 == 0).to(torch.int64)
    elif route == "skewed":
        hot = (token_indices % 10) < 7
        destination_ranks = torch.where(
            hot, torch.zeros_like(token_indices), token_indices % world_size
        )
        local_experts = torch.where(
            hot,
            torch.zeros_like(token_indices),
            (token_indices * 7 + 3) % num_local_experts,
        )
    elif route == "one_empty":
        # Nobody routes to the LAST rank: its fused K1 sees num_tokens == 0
        # (copy tickets do zero rows, every GEMM ticket early-exits, the
        # input arrive must still fire).
        destination_ranks = token_indices % (world_size - 1)
        local_experts = (token_indices % 2).to(torch.int64)
    elif route == "multi_empty":
        # Everything piles on rank 0: all other ranks are empty (the
        # strongest multi-rank-empty case reachable while every token must
        # route somewhere; an all-rank-empty layer cannot arise upstream).
        destination_ranks = torch.zeros_like(token_indices)
        local_experts = (token_indices % 2).to(torch.int64)
    else:  # single_tile: at most one M64 tile of work per destination rank
        keep = token_indices < (64 // world_size) * world_size
        destination_ranks = torch.where(
            keep, token_indices % world_size, torch.zeros_like(token_indices)
        )
        local_experts = torch.zeros_like(token_indices)
        # Tokens beyond the first 64 all pile on rank 0 expert 0; the other
        # ranks see a single active tile and a large inactive tail.
    top_experts = (
        destination_ranks * num_local_experts + local_experts
    ).view(-1, 1)
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=num_local_experts,
        expert_padding=64,
    )
    x = torch.empty(
        num_local_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=device
    )
    columns = torch.arange(hidden_size, device=device)
    x.copy_(
        ((token_indices[:, None] * hidden_size + columns[None, :]) % 31 - 15)
        .add(rank * 0.25)
        .to(torch.float8_e4m3fn)
    )
    scale_columns = torch.arange(hidden_size // 128, device=device)
    x_scale = (
        rank * 10000 + token_indices[:, None] * 100 + scale_columns[None, :]
    ).to(torch.float32).contiguous()
    generator = torch.Generator(device=device).manual_seed(20260817 + rank)
    weight = torch.randn(
        (num_local_experts, n, hidden_size), generator=generator, device=device
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    weight_scale = (
        torch.rand(
            (num_local_experts, n // 128, hidden_size // 128),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    capacity = workspace.schedule_capacity
    valid_rows = int(schedule.num_tokens.item())
    assert 0 <= valid_rows <= capacity and valid_rows % 64 == 0

    # Split reference, computed once.
    workspace.routed_x.fill_(7)
    workspace.routed_x_scale.fill_(-999)
    workspace.m_indices.fill_(-777)
    routed_x, routed_x_scale, m_indices = functional.dispatch_fp8_block(
        workspace,
        schedule,
        x,
        x_scale,
        trim_to_active_rows=False,
        prepare_combine=True,
    )
    gate_up_ref = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    functional.grouped_gemm_fp8_block_dynamic_out(
        routed_x,
        weight,
        routed_x_scale,
        weight_scale,
        m_indices,
        schedule.num_tokens,
        gate_up_ref,
    )
    refs = (
        workspace.routed_x.clone(),
        workspace.routed_x_scale.clone(),
        workspace.m_indices.clone(),
        gate_up_ref,
    )
    return workspace, schedule, x, x_scale, weight, weight_scale, refs, (
        capacity,
        n,
        valid_rows,
    )


def _fused_mismatch(workspace, gate_up_fused, refs, dims):
    routed_x_ref, routed_x_scale_ref, m_indices_ref, gate_up_ref = refs
    capacity, n, valid_rows = dims
    return torch.stack(
        [
            (
                workspace.routed_x[:valid_rows].view(torch.uint8)
                != routed_x_ref[:valid_rows].view(torch.uint8)
            ).sum(),
            (
                workspace.routed_x_scale[:valid_rows]
                != routed_x_scale_ref[:valid_rows]
            ).sum(),
            (workspace.m_indices != m_indices_ref).sum(),
            (
                gate_up_fused[:valid_rows].view(torch.uint16)
                != gate_up_ref[:valid_rows].view(torch.uint16)
            ).sum(),
            (gate_up_fused[valid_rows:] != 101.0).sum(),
        ]
    )


def _poison(workspace):
    workspace.routed_x.fill_(7)
    workspace.routed_x_scale.fill_(-999)
    workspace.m_indices.fill_(-777)


@pytest.mark.parametrize(
    "route", ["balanced", "skewed", "one_empty", "multi_empty"]
)
@pytest.mark.parametrize("forced", [1, 7, 8, 9, 0], ids=lambda f: f"w{f}")
def test_k1_worker_sweep(
    context: tuple[int, int, torch.device], route: str, forced: int
) -> None:
    """Progress and exactness for any resident worker count: forced=1 makes
    a single cluster execute every copy and GEMM ticket consecutively
    (R < C progression plus the multi-task drain path); 0 = occupancy max."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, route
    )
    capacity, n, _ = dims
    _poison(ws)
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    functional.dispatch_gemm_fused_fp8_block(
        ws, schedule, x, x_scale, w, w_scale, gate_up,
        copy_clusters=8, forced_worker_clusters=forced,
    )
    mism = _fused_mismatch(ws, gate_up, refs, dims)
    dist.all_reduce(mism, op=dist.ReduceOp.MAX)
    values = mism.cpu().tolist()
    print(
        f"K1_SWEEP|rank={rank}|route={route}|forced={forced}|values={values}",
        flush=True,
    )
    assert not any(values)


def test_k1_iteration_stress(
    context: tuple[int, int, torch.device],
) -> None:
    """>= 1000 consecutive fused iterations reproduce the split reference
    exactly; the mismatch max accumulates on-device so the loop stays free
    of host synchronization."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, "balanced"
    )
    capacity, n, _ = dims
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    worst = torch.zeros(5, dtype=torch.int64, device=device)
    iters = 1000
    for _ in range(iters):
        _poison(ws)
        gate_up.fill_(101.0)
        functional.dispatch_gemm_fused_fp8_block(
            ws, schedule, x, x_scale, w, w_scale, gate_up, copy_clusters=8
        )
        worst = torch.maximum(worst, _fused_mismatch(ws, gate_up, refs, dims))
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    values = worst.cpu().tolist()
    print(f"K1_STRESS|rank={rank}|iters={iters}|values={values}", flush=True)
    assert not any(values)


def test_k1_single_worker_thousand(
    context: tuple[int, int, torch.device],
) -> None:
    """1000 iterations with a single worker cluster: the R < C progression
    and consecutive multi-task drain path at scenario-frozen volume."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, "balanced"
    )
    capacity, n, _ = dims
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    worst = torch.zeros(5, dtype=torch.int64, device=device)
    for _ in range(1000):
        _poison(ws)
        gate_up.fill_(101.0)
        functional.dispatch_gemm_fused_fp8_block(
            ws, schedule, x, x_scale, w, w_scale, gate_up,
            copy_clusters=8, forced_worker_clusters=1,
        )
        worst = torch.maximum(worst, _fused_mismatch(ws, gate_up, refs, dims))
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    values = worst.cpu().tolist()
    print(f"K1_W1_1000|rank={rank}|values={values}", flush=True)
    assert not any(values)


def test_k1_delay_ticket0(
    context: tuple[int, int, torch.device],
) -> None:
    """Resident-but-slow producer: ticket 0 busy-waits ~0.1s after claiming
    its ticket and before the input arrive; every other worker must wait it
    out and the result stays exact (progress, not just correctness)."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, "balanced"
    )
    capacity, n, _ = dims
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    worst = torch.zeros(5, dtype=torch.int64, device=device)
    # One long delay (~0.1s) plus 999 short ones (~2.5ms each): scenario
    # volume at the frozen 1000 while keeping suite runtime bounded.
    delays = [200_000_000] + [5_000_000] * 999
    for delay in delays:
        _poison(ws)
        gate_up.fill_(101.0)
        functional.dispatch_gemm_fused_fp8_block(
            ws, schedule, x, x_scale, w, w_scale, gate_up,
            copy_clusters=8, delay_ticket0_cycles=delay,
        )
        worst = torch.maximum(worst, _fused_mismatch(ws, gate_up, refs, dims))
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    values = worst.cpu().tolist()
    print(f"K1_DELAY_1000|rank={rank}|values={values}", flush=True)
    assert not any(values)


def test_k1_arrival_skew(
    context: tuple[int, int, torch.device],
) -> None:
    """Cross-rank arrival skew: rank 0 sleeps ~0.1s BEFORE entering K1, so
    every other rank spins in the input-publish barrier until the straggler
    arrives; repeated 200x with short skews for volume."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, "balanced"
    )
    capacity, n, _ = dims
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    worst = torch.zeros(5, dtype=torch.int64, device=device)
    skews = [200_000_000] + [5_000_000] * 199
    for skew in skews:
        _poison(ws)
        gate_up.fill_(101.0)
        if rank == 0:
            torch.cuda._sleep(skew)
        functional.dispatch_gemm_fused_fp8_block(
            ws, schedule, x, x_scale, w, w_scale, gate_up, copy_clusters=8
        )
        worst = torch.maximum(worst, _fused_mismatch(ws, gate_up, refs, dims))
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    values = worst.cpu().tolist()
    print(f"K1_ARRIVAL_SKEW|rank={rank}|values={values}", flush=True)
    assert not any(values)


def test_k1_exactly_once_tickets(
    context: tuple[int, int, torch.device],
) -> None:
    """Exactly-once debug gate: every ticket below total_tickets is visited
    exactly once, none above, across worker counts."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, "balanced"
    )
    capacity, n, _ = dims
    m_tiles = capacity // 64
    n_pairs = (n // 64) // 2
    total = 8 + m_tiles * n_pairs
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    for forced in (1, 8, 0):
        visits = torch.zeros(total + 64, dtype=torch.int32, device=device)
        _poison(ws)
        gate_up.fill_(101.0)
        functional.dispatch_gemm_fused_fp8_block(
            ws, schedule, x, x_scale, w, w_scale, gate_up,
            copy_clusters=8, forced_worker_clusters=forced,
            ticket_visit=visits, record_visits=1,
        )
        bad = torch.tensor(
            [
                int((visits[:total] != 1).sum()),
                int((visits[total:] != 0).sum()),
            ],
            device=device,
        )
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        values = bad.cpu().tolist()
        print(
            f"K1_EXACTLY_ONCE|rank={rank}|forced={forced}|total={total}"
            f"|values={values}",
            flush=True,
        )
        assert not any(values)


def test_k1_graph_replay_stress(
    context: tuple[int, int, torch.device],
) -> None:
    """Capture the fused pipeline once and replay >= 1000 times; the final
    state must still match the split reference exactly.  All ranks capture
    the same sequence in the same order (capture records without executing,
    so the barrier arrive history stays symmetric)."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, "balanced"
    )
    capacity, n, _ = dims
    _poison(ws)
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    # Warm the exact captured shape for real once, rank-synchronous.
    functional.dispatch_gemm_fused_fp8_block(
        ws, schedule, x, x_scale, w, w_scale, gate_up, copy_clusters=8
    )
    torch.cuda.synchronize(device)
    dist.barrier()
    # The graph itself poisons, runs the pipeline, computes the mismatch
    # vector and folds it into a running max -- EVERY replay is checked,
    # not just the final state (a mid-run corruption later overwritten
    # cannot hide).
    worst = torch.zeros(5, dtype=torch.int64, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _poison(ws)
        gate_up.fill_(101.0)
        functional.dispatch_gemm_fused_fp8_block(
            ws, schedule, x, x_scale, w, w_scale, gate_up, copy_clusters=8
        )
        step = _fused_mismatch(ws, gate_up, refs, dims)
        torch.maximum(worst, step, out=worst)
    replays = 1000
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    values = worst.cpu().tolist()
    print(
        f"K1_GRAPH_REPLAY|rank={rank}|replays={replays}|values={values}",
        flush=True,
    )
    assert not any(values)


def test_k1_single_active_tile_tail(
    context: tuple[int, int, torch.device],
) -> None:
    """A route with (at most) one active M64 tile on most ranks plus a large
    inactive capacity tail: early-exiting GEMM tickets and untouched tail
    slots must leave the poison pattern intact while the active tile stays
    exact."""
    rank, _, device = context
    ws, schedule, x, x_scale, w, w_scale, refs, dims = _build_case(
        context, "single_tile"
    )
    capacity, n, _ = dims
    _poison(ws)
    gate_up = torch.full(
        (capacity, n), 101.0, dtype=torch.bfloat16, device=device
    )
    functional.dispatch_gemm_fused_fp8_block(
        ws, schedule, x, x_scale, w, w_scale, gate_up, copy_clusters=8
    )
    mism = _fused_mismatch(ws, gate_up, refs, dims)
    dist.all_reduce(mism, op=dist.ReduceOp.MAX)
    values = mism.cpu().tolist()
    print(f"K1_SINGLE_TILE|rank={rank}|values={values}", flush=True)
    assert not any(values)


def test_k1_concurrent_workspaces(
    context: tuple[int, int, torch.device],
) -> None:
    """Two independent workspaces on two streams, fused pipelines running
    concurrently for 1000 iterations: the legal concurrency path (each
    workspace has its own lease/queue state) stays exact on both."""
    rank, _, device = context
    ws_a, sch_a, x_a, xs_a, w_a, wsc_a, refs_a, dims_a = _build_case(
        context, "balanced"
    )
    ws_b, sch_b, x_b, xs_b, w_b, wsc_b, refs_b, dims_b = _build_case(
        context, "skewed", num_local_tokens=1024
    )
    assert ws_a is not ws_b
    cap_a, n_a, _ = dims_a
    cap_b, n_b, _ = dims_b
    gate_a = torch.full(
        (cap_a, n_a), 101.0, dtype=torch.bfloat16, device=device
    )
    gate_b = torch.full(
        (cap_b, n_b), 101.0, dtype=torch.bfloat16, device=device
    )
    s1 = torch.cuda.Stream(device=device)
    s2 = torch.cuda.Stream(device=device)
    worst = torch.zeros(10, dtype=torch.int64, device=device)
    for _ in range(1000):
        _poison(ws_a)
        _poison(ws_b)
        gate_a.fill_(101.0)
        gate_b.fill_(101.0)
        torch.cuda.synchronize(device)
        with torch.cuda.stream(s1):
            functional.dispatch_gemm_fused_fp8_block(
                ws_a, sch_a, x_a, xs_a, w_a, wsc_a, gate_a, copy_clusters=8
            )
        with torch.cuda.stream(s2):
            functional.dispatch_gemm_fused_fp8_block(
                ws_b, sch_b, x_b, xs_b, w_b, wsc_b, gate_b, copy_clusters=8
            )
        torch.cuda.synchronize(device)
        worst[:5] = torch.maximum(
            worst[:5], _fused_mismatch(ws_a, gate_a, refs_a, dims_a)
        )
        worst[5:] = torch.maximum(
            worst[5:], _fused_mismatch(ws_b, gate_b, refs_b, dims_b)
        )
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    values = worst.cpu().tolist()
    print(f"K1_CONCURRENT_WS|rank={rank}|values={values}", flush=True)
    assert not any(values)
