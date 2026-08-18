#!/usr/bin/env python3
"""EP4 production benchmark for terminal MoK versus split and K1/K2.

Launch with ``torchrun --nproc-per-node=4``.  Every reported latency sample is
the maximum CUDA-event duration across the four EP ranks.  Each comparison is
run as terminal A1 -> baseline B -> terminal A2 after interleaved warmup; the
A1/A2 drift gate prevents a thermally or externally unstable run from being
reported as a performance result.

The 766-token cell uses a 768-token graph bucket but keeps exactly 766*6 valid
routes.  Both counts are emitted so bucket work is never mislabeled as useful
token throughput.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
from dataclasses import dataclass
from typing import Callable

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")

import torch
import torch.distributed as dist

from mok import _C
from mok.functional import (
    MoKConfig,
    MoKSchedule,
    acquire_workspace_lease,
    combine_reduce_fp8_block_routes,
    create_fp8_route_workspace,
    create_fp8_terminal_workspace,
    dispatch_fp8_block,
    dispatch_gemm_fused_fp8_block,
    gemm_combine_fused_fp8_block,
    grouped_gemm_fp8_block_dynamic_out,
    megakernel_fp8_block,
    release_workspace_lease,
)
from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant


EP_SIZE = 4
HIDDEN = 4096
INTERMEDIATE = 2048
TOPK = 6
LOCAL_EXPERTS = 64
M_TILE = 64
DEFAULT_TOKENS = (128, 766, 2048)
DEFAULT_BASELINES = ("split", "k1k2")


@dataclass(frozen=True, slots=True)
class Cell:
    effective_tokens: int
    graph_tokens: int
    valid_routes: int
    active_rows: int
    schedule_capacity: int


@dataclass(slots=True)
class PipelineRunners:
    terminal: Callable[[], torch.Tensor]
    split: Callable[[], torch.Tensor]
    k1k2: Callable[[], torch.Tensor]
    calls: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="128,766,2048")
    parser.add_argument("--baselines", default="split,k1k2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--aba-drift-limit", type=float, default=0.05)
    parser.add_argument(
        "--compute-clusters",
        type=int,
        default=0,
        help="0 uses the prewarmed resident maximum",
    )
    parser.add_argument("--copy-clusters", type=int, default=8)
    parser.add_argument("--minibatch-rows", type=int, default=4096)
    parser.add_argument("--macrobatch-rows", type=int, default=131072)
    parser.add_argument("--spin-limit", type=int, default=1 << 29)
    parser.add_argument(
        "--output",
        default="/tmp/mok-terminal-production-benchmark.json",
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-so-sha256", required=True)
    parser.add_argument("--expected-harness-sha256", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="allow a reduced token list, warmup, or iteration count",
    )
    args = parser.parse_args()
    args.tokens = tuple(
        int(value) for value in args.tokens.split(",") if value.strip()
    )
    args.baselines = tuple(
        value.strip() for value in args.baselines.split(",") if value.strip()
    )
    if not args.tokens or any(value <= 0 for value in args.tokens):
        raise ValueError("--tokens must contain positive integers")
    if set(args.baselines) != set(DEFAULT_BASELINES):
        raise ValueError("--baselines must contain split and k1k2 exactly")
    if not args.smoke:
        if args.tokens != DEFAULT_TOKENS:
            raise ValueError(
                "formal run requires --tokens 128,766,2048; use --smoke "
                "for a reduced matrix"
            )
        if args.warmup < 5 or args.iters < 20:
            raise ValueError("formal run requires warmup>=5 and iters>=20")
    elif args.warmup < 1 or args.iters < 1:
        raise ValueError("smoke run still requires positive warmup/iters")
    if (
        not math.isfinite(args.aba_drift_limit)
        or not 0 <= args.aba_drift_limit <= 0.25
    ):
        raise ValueError("--aba-drift-limit must be finite and in [0,0.25]")
    if args.compute_clusters < 0 or args.copy_clusters <= 0:
        raise ValueError("cluster counts are invalid")
    if (
        args.minibatch_rows <= 0
        or args.minibatch_rows % M_TILE
        or args.macrobatch_rows < args.minibatch_rows
        or args.macrobatch_rows % args.minibatch_rows
        or args.spin_limit <= 0
    ):
        raise ValueError("terminal scheduling parameters are invalid")
    return args


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_state() -> tuple[pathlib.Path, str, list[str]]:
    repo = pathlib.Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode or status.returncode:
        raise RuntimeError("cannot resolve MoK source provenance")
    return repo, head.stdout.strip(), status.stdout.splitlines()


def validate_provenance(args: argparse.Namespace) -> dict:
    repo, head, source_state = repository_state()
    extension = pathlib.Path(_C.__file__).resolve()
    harness = pathlib.Path(__file__).resolve()
    activation = pathlib.Path(
        silu_and_mul_contig_post_quant.__code__.co_filename
    ).resolve()
    record = {
        "git_head": head,
        "source_state": source_state,
        "so_path": str(extension),
        "so_sha256": sha256(extension),
        "harness_path": str(harness),
        "harness_sha256": sha256(harness),
        "activation_path": str(activation),
        "activation_sha256": sha256(activation),
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES", "unset"
        ),
    }
    mismatches = []
    for label, actual, expected in (
        ("git_head", record["git_head"], args.expected_head),
        ("so_sha256", record["so_sha256"], args.expected_so_sha256),
        (
            "harness_sha256",
            record["harness_sha256"],
            args.expected_harness_sha256,
        ),
    ):
        if actual != expected:
            mismatches.append(f"{label}: actual={actual} expected={expected}")
    if source_state:
        mismatches.append(f"source_state is dirty: {source_state}")
    if mismatches:
        raise RuntimeError("frozen provenance gate failed: " + "; ".join(mismatches))
    required_apis = (
        "fp8_block_megakernel_prewarm",
        "fp8_block_megakernel_prepare_out",
        "fp8_block_megakernel_out",
        "fp8_block_dispatch_gemm_fused_out",
        "fp8_block_gemm_combine_fused_out",
        "fp8_block_routed_dispatch_copy_out",
        "fp8_block_grouped_contiguous_dynamic_out",
        "fp8_block_routed_combine_reduce_out",
    )
    missing = [name for name in required_apis if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(f"loaded extension lacks benchmark paths: {missing}")
    source = harness.read_text(encoding="utf-8")
    forbidden = (
        "terminal_" + "production_ep4_probe",
        "terminal_" + "full_pipeline_probe",
        "build_" + "extension",
        ".run_" + "split(",
    )
    leaked = [needle for needle in forbidden if needle in source]
    if leaked:
        raise RuntimeError(f"probe-only path leaked into benchmark: {leaked}")
    tree = ast.parse(source)
    make_runners_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_runners"
    )
    nested = {
        node.name: node
        for node in make_runners_node.body
        if isinstance(node, ast.FunctionDef)
    }

    def calls(function_name: str) -> list[str]:
        result = []
        for node in ast.walk(nested[function_name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                result.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                result.append(node.func.attr)
        return result

    terminal_calls = calls("run_terminal")
    split_calls = calls("run_split")
    k1k2_calls = calls("run_k1k2")
    if terminal_calls.count("megakernel_fp8_block") != 1:
        raise RuntimeError("terminal timing path is not exactly one megakernel call")
    if (
        split_calls.count("dispatch_fp8_block") != 1
        or split_calls.count("grouped_gemm_fp8_block_dynamic_out") != 2
        or split_calls.count("combine_reduce_fp8_block_routes") != 1
        or split_calls.count("release_workspace_lease") != 1
    ):
        raise RuntimeError(f"split timing path changed: {split_calls}")
    if (
        k1k2_calls.count("dispatch_gemm_fused_fp8_block") != 1
        or k1k2_calls.count("gemm_combine_fused_fp8_block") != 1
        or "release_workspace_lease" in k1k2_calls
    ):
        raise RuntimeError(f"K1/K2 timing path changed: {k1k2_calls}")
    cross_path_forbidden = {
        "run_terminal": {
            "dispatch_fp8_block",
            "dispatch_gemm_fused_fp8_block",
            "gemm_combine_fused_fp8_block",
        },
        "run_split": {
            "megakernel_fp8_block",
            "dispatch_gemm_fused_fp8_block",
            "gemm_combine_fused_fp8_block",
        },
        "run_k1k2": {
            "megakernel_fp8_block",
            "dispatch_fp8_block",
            "combine_reduce_fp8_block_routes",
        },
    }
    for name, disallowed in cross_path_forbidden.items():
        overlap = sorted(disallowed.intersection(calls(name)))
        if overlap:
            raise RuntimeError(f"{name} contains cross-path calls: {overlap}")
    record["static_path_receipt"] = {
        "terminal": terminal_calls,
        "split": split_calls,
        "k1k2": k1k2_calls,
    }
    return record


def make_cell(effective_tokens: int) -> Cell:
    graph_tokens = math.ceil(effective_tokens / M_TILE) * M_TILE
    valid_routes = effective_tokens * TOPK
    active_rows = math.ceil(valid_routes / M_TILE) * M_TILE
    # Match create_fp8_route_workspace's minimum capacity factor for EP4.
    schedule_capacity = graph_tokens * TOPK * 2
    if schedule_capacity % M_TILE or active_rows > schedule_capacity:
        raise RuntimeError("derived terminal schedule bucket is invalid")
    return Cell(
        effective_tokens=effective_tokens,
        graph_tokens=graph_tokens,
        valid_routes=valid_routes,
        active_rows=active_rows,
        schedule_capacity=schedule_capacity,
    )


def make_fp8_full(
    shape: tuple[int, ...], value: float, device: torch.device
) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    result.fill_(value)
    return result


def allocate_weights(
    rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w13 = make_fp8_full(
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        0.0078125 * (rank + 1),
        device,
    )
    w13_scale = torch.full(
        (LOCAL_EXPERTS, 32, 32),
        0.5 + 0.0625 * rank,
        dtype=torch.float32,
        device=device,
    )
    w2 = make_fp8_full(
        (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
        0.00390625 * (rank + 1),
        device,
    )
    w2_scale = torch.full(
        (LOCAL_EXPERTS, 32, 16),
        0.5 + 0.0625 * rank,
        dtype=torch.float32,
        device=device,
    )
    return w13, w13_scale, w2, w2_scale


def expert_rows(active_rows: int, device: torch.device) -> torch.Tensor:
    tiles = active_rows // M_TILE
    quotient, remainder = divmod(tiles, LOCAL_EXPERTS)
    rows = torch.full(
        (LOCAL_EXPERTS,), quotient * M_TILE,
        dtype=torch.int32, device=device,
    )
    if remainder:
        rows[:remainder].add_(M_TILE)
    if int(rows.sum().item()) != active_rows:
        raise RuntimeError("expert row partition does not cover active rows")
    return rows


def make_inputs_and_schedule(
    cell: Cell, rank: int, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MoKSchedule,
]:
    token_values = (
        torch.arange(cell.graph_tokens, dtype=torch.float32, device=device)
        .remainder_(251)
        .mul_(0.0009765625)
        .add_(0.03125 * (rank + 1))
    )
    x = token_values[:, None].expand(-1, HIDDEN).to(
        torch.float8_e4m3fn
    ).contiguous()
    x_scale = (
        torch.arange(HIDDEN // 128, dtype=torch.float32, device=device)
        .mul_(0.0009765625)
        .add_(0.5 + 0.0625 * rank)
        .expand(cell.graph_tokens, -1)
        .contiguous()
    )
    base_weights = torch.arange(1, TOPK + 1, dtype=torch.float32, device=device)
    topk_weights = torch.zeros(
        (cell.graph_tokens, TOPK), dtype=torch.float32, device=device
    )
    topk_weights[: cell.effective_tokens] = torch.roll(
        base_weights, rank
    ).div(base_weights.sum())

    tokens_per_expert = expert_rows(cell.active_rows, device)
    expert_for_row = torch.repeat_interleave(
        torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device),
        tokens_per_expert.to(torch.int64),
    )
    if expert_for_row.numel() != cell.active_rows:
        raise RuntimeError("expert row map has the wrong size")
    topk_ids = torch.full(
        (cell.graph_tokens, TOPK), -1, dtype=torch.int32, device=device
    )
    previous_rank = (rank - 1) % EP_SIZE
    topk_ids.view(-1)[: cell.valid_routes] = (
        previous_rank * LOCAL_EXPERTS
        + expert_for_row[: cell.valid_routes]
    )

    peer_rank = torch.full(
        (cell.schedule_capacity,), -1, dtype=torch.int32, device=device
    )
    peer_token_idx = torch.full_like(peer_rank, -1)
    peer_rank[: cell.valid_routes] = (rank + 1) % EP_SIZE
    peer_token_idx[: cell.valid_routes] = torch.arange(
        cell.valid_routes, dtype=torch.int32, device=device
    )
    schedule = MoKSchedule(
        peer_rank=peer_rank,
        peer_token_idx=peer_token_idx,
        num_tokens=torch.tensor(
            [cell.active_rows], dtype=torch.int32, device=device
        ),
        tokens_per_expert=tokens_per_expert,
        expert_padding=M_TILE,
    )
    return x, x_scale, topk_weights, topk_ids, schedule


def activate(
    gate_up: torch.Tensor,
    down_input: torch.Tensor,
    down_input_scale: torch.Tensor,
    active_rows: int,
) -> None:
    silu_and_mul_contig_post_quant(
        input=gate_up[:active_rows],
        output=down_input[:active_rows],
        output_scale=down_input_scale[:active_rows],
        quant_group_size=128,
        scale_ue8m0=False,
        transposed=False,
        swiglu_limit=10.0,
        swizzle=False,
    )


def make_runners(
    args: argparse.Namespace,
    cell: Cell,
    rank: int,
    device: torch.device,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[
    PipelineRunners,
    object,
    object,
    dict[str, torch.Tensor],
]:
    w13, w13_scale, w2, w2_scale = weights
    x, x_scale, topk_weights, topk_ids, schedule = make_inputs_and_schedule(
        cell, rank, device
    )
    config = MoKConfig(schedule_capacity_multiplier=0.5)
    route_workspace = create_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=cell.graph_tokens,
        hidden_size=HIDDEN,
        topk=TOPK,
        num_local_experts=LOCAL_EXPERTS,
    )
    if route_workspace.schedule_capacity != cell.schedule_capacity:
        raise RuntimeError("route and terminal capacity contracts diverged")
    requested_compute = args.compute_clusters or None
    terminal_workspace = create_fp8_terminal_workspace(
        dist.group.WORLD,
        device=device,
        num_local_tokens=cell.graph_tokens,
        schedule_capacity=cell.schedule_capacity,
        num_local_experts=LOCAL_EXPERTS,
        compute_clusters=requested_compute,
    )

    gate_up = torch.empty(
        (cell.schedule_capacity, 2 * INTERMEDIATE),
        dtype=torch.bfloat16,
        device=device,
    )
    down_input = torch.empty(
        (cell.schedule_capacity, INTERMEDIATE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    down_input_scale = torch.empty(
        (cell.schedule_capacity, INTERMEDIATE // 128),
        dtype=torch.float32,
        device=device,
    )
    routed_y = torch.empty(
        (cell.schedule_capacity, HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    outputs = {
        name: torch.empty(
            (cell.graph_tokens, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        for name in ("terminal", "split", "k1k2")
    }
    calls = {name: 0 for name in outputs}

    def run_terminal() -> torch.Tensor:
        calls["terminal"] += 1
        return megakernel_fp8_block(
            terminal_workspace,
            schedule,
            x,
            x_scale,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            outputs["terminal"],
            minibatch_rows=args.minibatch_rows,
            macrobatch_rows=args.macrobatch_rows,
            spin_limit=args.spin_limit,
        )

    def run_split() -> torch.Tensor:
        calls["split"] += 1
        acquire_workspace_lease(route_workspace)
        dispatch_fp8_block(
            route_workspace,
            schedule,
            x,
            x_scale,
            trim_to_active_rows=False,
            prepare_combine=True,
        )
        grouped_gemm_fp8_block_dynamic_out(
            route_workspace.routed_x,
            w13,
            route_workspace.routed_x_scale,
            w13_scale,
            route_workspace.m_indices,
            schedule.num_tokens,
            gate_up,
        )
        activate(gate_up, down_input, down_input_scale, cell.active_rows)
        grouped_gemm_fp8_block_dynamic_out(
            down_input,
            w2,
            down_input_scale,
            w2_scale,
            route_workspace.m_indices,
            schedule.num_tokens,
            routed_y,
        )
        reduced = combine_reduce_fp8_block_routes(
            route_workspace,
            schedule,
            routed_y[: cell.active_rows],
            topk_weights,
            combine_precleared=True,
        )
        outputs["split"].copy_(reduced)
        release_workspace_lease(route_workspace)
        return outputs["split"]

    def run_k1k2() -> torch.Tensor:
        calls["k1k2"] += 1
        acquire_workspace_lease(route_workspace)
        dispatch_gemm_fused_fp8_block(
            route_workspace,
            schedule,
            x,
            x_scale,
            w13,
            w13_scale,
            gate_up,
            copy_clusters=args.copy_clusters,
        )
        activate(gate_up, down_input, down_input_scale, cell.active_rows)
        return gemm_combine_fused_fp8_block(
            route_workspace,
            schedule,
            down_input,
            down_input_scale,
            w2,
            w2_scale,
            routed_y,
            topk_weights,
            release_lease=True,
            output=outputs["k1k2"],
        )

    tensors = {
        "x": x,
        "x_scale": x_scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "schedule_peer_rank": schedule.peer_rank,
        "schedule_peer_token_idx": schedule.peer_token_idx,
        "schedule_num_tokens": schedule.num_tokens,
        "schedule_tokens_per_expert": schedule.tokens_per_expert,
        **outputs,
    }
    return (
        PipelineRunners(run_terminal, run_split, run_k1k2, calls),
        terminal_workspace,
        route_workspace,
        tensors,
    )


def require_exact(
    name: str, expected: torch.Tensor, actual: torch.Tensor
) -> None:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise RuntimeError(
            f"{name} metadata mismatch: {expected.shape}/{expected.dtype} "
            f"vs {actual.shape}/{actual.dtype}"
        )
    if expected.dtype == torch.bfloat16:
        lhs, rhs = expected.view(torch.int16), actual.view(torch.int16)
    elif expected.dtype == torch.float32:
        lhs, rhs = expected.view(torch.int32), actual.view(torch.int32)
    else:
        raise TypeError(f"unsupported exact dtype: {expected.dtype}")
    mismatch = int((lhs != rhs).sum().item())
    if mismatch:
        raise RuntimeError(f"{name} bitwise mismatch: {mismatch} elements")


def assert_trap_clear(workspace: object, label: str) -> None:
    if int(workspace.trap_record[0].item()) != 0:
        raise RuntimeError(
            f"{label} trap record is nonzero: {workspace.trap_record.tolist()}"
        )


def assert_route_closed(workspace: object, label: str) -> None:
    if int(workspace.in_use.item()) != 0:
        raise RuntimeError(f"{label} workspace lease did not close")
    assert_trap_clear(workspace, label)


def assert_terminal_closed(workspace: object, cell: Cell) -> None:
    total_tasks = cell.active_rows // M_TILE * 65
    expected = {
        "in_use": 0,
        "next_logical_cluster": total_tasks,
        "producer_done": total_tasks,
        "comm_closed": 2,
        "push_done": cell.active_rows,
        "reduce_done": cell.graph_tokens,
        "terminate": 1,
        "epilogue_done": 1 + workspace.compute_clusters,
    }
    observed = {
        name: int(getattr(workspace, name).item()) for name in expected
    }
    if observed != expected:
        raise RuntimeError(
            f"terminal closure mismatch: expected={expected} observed={observed}"
        )
    if not bool(torch.all(workspace.route_ready == 1).item()):
        raise RuntimeError("terminal route_ready is not closed")
    if not bool(torch.all(workspace.epilogue_claim == 1).item()):
        raise RuntimeError("terminal epilogue tokens were not claimed once")
    assert_trap_clear(workspace, "terminal")


def rank_max_event_samples(
    function: Callable[[], torch.Tensor], iterations: int, device: torch.device
) -> list[float]:
    dist.barrier()
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(iterations)
    ]
    keep = None
    for start, end in events:
        start.record()
        keep = function()
        end.record()
    torch.cuda.synchronize(device)
    if keep is None:
        raise RuntimeError("timed path did not execute")
    local = torch.tensor(
        [start.elapsed_time(end) for start, end in events],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return local.cpu().tolist()


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(samples: list[float], cell: Cell) -> dict:
    p50 = statistics.median(samples)
    p95 = percentile(samples, 0.95)
    seconds = p50 / 1000.0
    flops_per_route = 2 * (
        HIDDEN * (2 * INTERMEDIATE) + INTERMEDIATE * HIDDEN
    )
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "effective_tokens_per_s": EP_SIZE * cell.effective_tokens / seconds,
        "graph_tokens_per_s": EP_SIZE * cell.graph_tokens / seconds,
        "valid_routes_per_s": EP_SIZE * cell.valid_routes / seconds,
        "scheduled_rows_per_s": EP_SIZE * cell.active_rows / seconds,
        "effective_tflops": (
            EP_SIZE * cell.valid_routes * flops_per_route / seconds / 1e12
        ),
        "scheduled_tflops": (
            EP_SIZE * cell.active_rows * flops_per_route / seconds / 1e12
        ),
        "samples_ms": samples,
    }


def warm_aba(
    terminal: Callable[[], torch.Tensor],
    baseline: Callable[[], torch.Tensor],
    iterations: int,
    device: torch.device,
) -> None:
    for _ in range(iterations):
        terminal()
        baseline()
        terminal()
    torch.cuda.synchronize(device)


def measure_aba(
    args: argparse.Namespace,
    cell: Cell,
    runners: PipelineRunners,
    baseline_name: str,
    device: torch.device,
) -> dict:
    baseline = getattr(runners, baseline_name)
    before = dict(runners.calls)
    warm_aba(runners.terminal, baseline, args.warmup, device)
    a1 = rank_max_event_samples(runners.terminal, args.iters, device)
    b = rank_max_event_samples(baseline, args.iters, device)
    a2 = rank_max_event_samples(runners.terminal, args.iters, device)
    expected_delta = {
        "terminal": 2 * (args.warmup + args.iters),
        baseline_name: args.warmup + args.iters,
    }
    observed_delta = {
        name: runners.calls[name] - before[name] for name in expected_delta
    }
    if observed_delta != expected_delta:
        raise RuntimeError(
            f"path receipt mismatch for {baseline_name}: "
            f"expected={expected_delta} observed={observed_delta}"
        )
    a1_summary = summarize(a1, cell)
    baseline_summary = summarize(b, cell)
    a2_summary = summarize(a2, cell)
    midpoint = 0.5 * (a1_summary["p50_ms"] + a2_summary["p50_ms"])
    drift = abs(a2_summary["p50_ms"] - a1_summary["p50_ms"]) / midpoint
    if drift > args.aba_drift_limit:
        raise RuntimeError(
            f"terminal A/A drift {drift:.3%} exceeds "
            f"{args.aba_drift_limit:.3%} for {baseline_name}"
        )
    terminal_p50 = midpoint
    return {
        "order": ["terminal_a1", baseline_name, "terminal_a2"],
        "path_receipt": observed_delta,
        "terminal_a1": a1_summary,
        baseline_name: baseline_summary,
        "terminal_a2": a2_summary,
        "terminal_aa_drift": drift,
        "terminal_midpoint_p50_ms": terminal_p50,
        "baseline_over_terminal_speedup": (
            baseline_summary["p50_ms"] / terminal_p50
        ),
    }


def benchmark_cell(
    args: argparse.Namespace,
    cell: Cell,
    rank: int,
    device: torch.device,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict:
    runners, terminal_workspace, route_workspace, tensors = make_runners(
        args, cell, rank, device, weights
    )
    dist.barrier()
    split_output = runners.split()
    torch.cuda.synchronize(device)
    assert_route_closed(route_workspace, "split")
    terminal_output = runners.terminal()
    torch.cuda.synchronize(device)
    assert_terminal_closed(terminal_workspace, cell)
    k1k2_output = runners.k1k2()
    torch.cuda.synchronize(device)
    assert_route_closed(route_workspace, "k1k2")
    # Graph-bucket padding has no caller-visible output: its route IDs are -1
    # and its weights are zero, so backing combine rows may remain unspecified.
    # Exactness therefore applies to the effective token prefix only.
    effective = slice(0, cell.effective_tokens)
    require_exact(
        "terminal_vs_split", split_output[effective], terminal_output[effective]
    )
    require_exact(
        "k1k2_vs_split", split_output[effective], k1k2_output[effective]
    )

    comparisons = {}
    for baseline in args.baselines:
        comparisons[baseline] = measure_aba(
            args, cell, runners, baseline, device
        )
        assert_terminal_closed(terminal_workspace, cell)
        assert_route_closed(route_workspace, baseline)
    result = {
        "effective_tokens": cell.effective_tokens,
        "graph_tokens": cell.graph_tokens,
        "valid_routes": cell.valid_routes,
        "active_rows": cell.active_rows,
        "schedule_capacity": cell.schedule_capacity,
        "compute_clusters": terminal_workspace.compute_clusters,
        "max_compute_clusters": terminal_workspace.max_compute_clusters,
        "copy_clusters": args.copy_clusters,
        "correctness": {
            "terminal_vs_split": "bitwise_exact",
            "k1k2_vs_split": "bitwise_exact",
            "scope": "effective_token_prefix",
        },
        "closure": "PASS",
        "comparisons": comparisons,
    }
    del split_output, terminal_output, k1k2_output
    del runners, terminal_workspace, route_workspace, tensors
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != EP_SIZE:
        raise RuntimeError(f"benchmark requires EP4, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("terminal production benchmark requires SM90")
        provenance = validate_provenance(args)
        weights = allocate_weights(rank, device)
        cells = []
        for token_count in args.tokens:
            cell = make_cell(token_count)
            result = benchmark_cell(args, cell, rank, device, weights)
            cells.append(result)
            if rank == 0:
                for baseline in args.baselines:
                    comparison = result["comparisons"][baseline]
                    print(
                        "TERMINAL_PRODUCTION_BENCH"
                        f"|tokens={cell.effective_tokens}"
                        f"|bucket={cell.graph_tokens}"
                        f"|baseline={baseline}"
                        f"|terminal_ms={comparison['terminal_midpoint_p50_ms']:.6f}"
                        f"|baseline_ms={comparison[baseline]['p50_ms']:.6f}"
                        f"|speedup={comparison['baseline_over_terminal_speedup']:.6f}"
                        f"|aa_drift={comparison['terminal_aa_drift']:.6f}"
                        "|correctness=bitwise_exact|closure=PASS",
                        flush=True,
                    )
        record = {
            "schema": "mok-terminal-production-benchmark-v1",
            "formal": not args.smoke,
            "hardware": "H20-SM90",
            "shape": {
                "ep_size": EP_SIZE,
                "hidden": HIDDEN,
                "intermediate": INTERMEDIATE,
                "topk": TOPK,
                "local_experts": LOCAL_EXPERTS,
            },
            "timing": {
                "clock": "cuda_event",
                "rank_aggregation": "per_iteration_max_ep4",
                "warmup": args.warmup,
                "iterations": args.iters,
                "aba_drift_limit": args.aba_drift_limit,
            },
            "provenance": provenance,
            "cells": cells,
        }
        dist.barrier()
        if rank == 0:
            output = pathlib.Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"TERMINAL_PRODUCTION_BENCH_RESULT|path={output}|result=PASS")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
