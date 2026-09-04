#!/usr/bin/env python3
"""Step 2 of the warp-role megakernel plan: comm bandwidth and co-residency.

Three questions, one kernel shape (one CTA per SM, 78 CTAs, a producer, a
consumer and a comm warpgroup inside each):

  dispatch  the comm warpgroup alone pulls this rank's routed rows and their
            K128 scales out of the peers' symmetric buffers;
  combine   the comm warpgroup alone pushes finished BF16 rows into the peers'
            combine buffers and publishes the per-rank arrival counter;
  co-resident  the comm warpgroup runs dispatch while the producer and consumer
            of the same CTA run an independent grouped GEMM, timed against the
            same GEMM running alone.

The routing is the uniform EP4 case the megakernel is designed around: 2048
tokens per rank, top-6, eight local experts, so every rank receives 12288 routed
rows.  The co-resident GEMM is the W2 shape over those same 12288 rows.

Every reported latency is the maximum across the four ranks of that rank's p50,
and every timed call is preceded by a barrier so the ranks start together --
dispatch reads the peers and combine writes them, so a rank running alone would
measure an empty machine rather than the contended one.

Launch on four H20s of one node:

    torchrun --standalone --nproc-per-node=4 -m benchmarks.bench_warprole_comm
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from mok import _C, functional

EP_SIZE = 4
NUM_LOCAL_TOKENS = 2048
HIDDEN = 4096
TOPK = 6
NUM_LOCAL_EXPERTS = 8
EXPERT_PADDING = 256
# csrc/sm90_fp8_block_warprole_config.cuh: rows per minibatch, rows per M64
# tile, and geometry<1>::W2_TASKS_PER_TILE, the value combine waits for.
MINIBATCH_ROWS = 1024
M_TILE = 64
Y_READY_TARGET = HIDDEN // 128
DISPATCH_MODE = 0
COMBINE_MODE = 1

# Co-resident GEMM: the W2 shape (N=4096, K=2048) over the 12288 routed rows of
# one rank, 64 experts in contiguous row order.  Same generator as
# tests/test_warprole_gemm.py with the row pattern halved, which keeps the same
# spread of expert sizes while landing on 12288 rows instead of 24576.
GEMM_ENTRY = "fp8_block_warprole_gemm_c1s6_out"
GEMM_EXPERTS = 64
GEMM_ROW_PATTERN = [160, 168, 176, 184, 200, 208, 216, 224]   # 12288 rows
GEMM_N = 4096
GEMM_K = 2048
GEMM_SEED = 20260905

DISPATCH_MIN_GBPS = 60.0
COMBINE_MIN_GBPS = 60.0
GEMM_SLOWDOWN_LIMIT = 0.03


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch and combine bandwidth of the warp-role comm warpgroup, "
            "and the slowdown it inflicts on a co-resident grouped GEMM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=f"/mok/runtime-logs/warprole-step2-{utc_stamp()}.json",
        help="JSON result path on rank 0; the parent directory is created if needed",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="untimed calls before each timed block"
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="timed calls per stage"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("--warmup must be >= 0 and --iters must be >= 1")
    return args


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(device: torch.device, rank: int, world_size: int) -> dict:
    repo = pathlib.Path(__file__).resolve().parents[1]
    shared_objects = sorted((repo / "mok").glob("_C*.so"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        clocks = smi.stdout.strip().splitlines() if smi.returncode == 0 else None
    except OSError:
        clocks = None
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else "unavailable",
        "source_state": status.stdout.splitlines(),
        "harness_sha256": sha256(pathlib.Path(__file__).resolve()),
        "so_path": str(shared_objects[0]) if len(shared_objects) == 1 else None,
        "so_sha256": sha256(shared_objects[0]) if len(shared_objects) == 1 else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "multi_processor_count": torch.cuda.get_device_properties(
            device
        ).multi_processor_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_clocks_sm_mem": clocks,
        "rank": rank,
        "world_size": world_size,
        "timestamp_utc": utc_stamp(),
    }


def resolve_entry(name: str):
    entry = getattr(_C, name, None)
    if entry is None:
        raise RuntimeError(
            f"the binding {name!r} is missing from "
            f"{getattr(_C, '__file__', 'the mok extension')}. The warp-role comm "
            "entries exist only after csrc/sm90_fp8_block_warprole_comm.cuh and "
            "the matching m.def lines in csrc/bindings.cu are compiled; rebuild "
            "with benchmarks/warprole/build_sm90.sh and reinstall the extension "
            "before running this harness."
        )
    return entry


def uniform_top_experts(device: torch.device, world_size: int) -> torch.Tensor:
    """Route token t's j-th slot to rank (t+j)%R, local expert (t//R + j)%E.

    No token picks one expert twice and every (rank, local expert) pair receives
    the same number of rows, so each rank ends up with exactly 12288 active rows
    and no padded tail at expert padding 256.
    """
    token_index = torch.arange(NUM_LOCAL_TOKENS, device=device).unsqueeze(1)
    route_index = torch.arange(TOPK, device=device).unsqueeze(0)
    destination_rank = (token_index + route_index) % world_size
    local_expert = (
        torch.div(token_index, world_size, rounding_mode="floor") + route_index
    ) % NUM_LOCAL_EXPERTS
    return destination_rank * NUM_LOCAL_EXPERTS + local_expert


def make_gemm_inputs(device: torch.device):
    """Same generator as tests/test_warprole_gemm.py, halved row pattern."""
    generator = torch.Generator(device=device).manual_seed(GEMM_SEED)
    rows = GEMM_ROW_PATTERN * (GEMM_EXPERTS // len(GEMM_ROW_PATTERN))
    total_m = sum(rows)
    assert total_m % M_TILE == 0
    k_blocks = GEMM_K // 128
    a = torch.randn(
        (total_m, GEMM_K), generator=generator, device=device, dtype=torch.bfloat16
    )
    a = a.clamp(-3, 3).to(torch.float8_e4m3fn)
    b = torch.randn(
        (GEMM_EXPERTS, GEMM_N, GEMM_K),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    b = b.clamp(-3, 3).to(torch.float8_e4m3fn)
    a_scale = (
        torch.rand((total_m, k_blocks), generator=generator, device=device) * 0.09
        + 0.01
    )
    b_scale = (
        torch.rand(
            (GEMM_EXPERTS, GEMM_N // 128, k_blocks),
            generator=generator,
            device=device,
        )
        * 0.09
        + 0.01
    )
    m_indices = torch.repeat_interleave(
        torch.arange(GEMM_EXPERTS, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int64, device=device),
    ).contiguous()
    num_tokens = torch.tensor([total_m], dtype=torch.int32, device=device)
    output = torch.empty(
        (total_m, GEMM_N), dtype=torch.bfloat16, device=device
    )
    return a, b, a_scale, b_scale, m_indices, num_tokens, output, total_m


def build_state(rank: int, world_size: int, device: torch.device) -> dict:
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
    if active_rows != NUM_LOCAL_TOKENS * TOPK:
        raise RuntimeError(
            "uniform routing must produce one local batch worth of rows, got "
            f"{active_rows}"
        )

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
    # Combine waits on this counter for every active tile; the consumers are
    # idle in this microbenchmark, so the host publishes the finished-tile count
    # once, before any timed call.
    y_ready[: active_rows // M_TILE].fill_(Y_READY_TARGET)
    push_done_local = torch.zeros(1, dtype=torch.int32, device=device)
    # Symmetric arrival counter, allocated the way the workspace allocates
    # barrier_buffer.  The rendezvous is what makes the peer pointers below
    # usable from this process: a plain device pointer from another process is
    # not mapped here.
    push_done = symm_mem.empty(1, dtype=torch.int32, device=device)
    push_done.zero_()
    push_done_handle = symm_mem.rendezvous(push_done, workspace.group_name)
    push_done_ptrs = [
        int(push_done_handle.buffer_ptrs[peer_rank])
        for peer_rank in range(world_size)
    ]

    (
        gemm_a,
        gemm_b,
        gemm_a_scale,
        gemm_b_scale,
        gemm_m_indices,
        gemm_num_tokens,
        gemm_output,
        gemm_rows,
    ) = make_gemm_inputs(device)

    torch.cuda.synchronize()
    dist.barrier()
    return {
        "workspace": workspace,
        "schedule": schedule,
        "capacity": capacity,
        "active_rows": active_rows,
        "routed_y": routed_y,
        "x_ready": x_ready,
        "y_ready": y_ready,
        "push_done_local": push_done_local,
        "push_done": push_done,
        "push_done_ptrs": push_done_ptrs,
        "gemm": (
            gemm_a,
            gemm_b,
            gemm_a_scale,
            gemm_b_scale,
            gemm_m_indices,
            gemm_num_tokens,
            gemm_output,
        ),
        "gemm_rows": gemm_rows,
    }


def comm_args(state: dict) -> tuple:
    workspace = state["workspace"]
    schedule = state["schedule"]
    return (
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
        state["routed_y"],
        workspace.combine_buffer_ptrs,
        state["push_done_ptrs"],
        workspace.ep_rank,
        state["x_ready"],
        state["y_ready"],
        state["push_done_local"],
    )


def time_calls(function, warmup: int, iters: int) -> list[float]:
    """One CUDA-event pair per call, all four ranks released by a barrier.

    The counters run free during timing: x_ready and push_done accumulate over
    the repeated launches because nothing in the microbenchmark waits on them.
    Their values are checked in tests/test_warprole_comm.py, where each launch
    starts from a cleared counter; here they are only read as proof that the
    kernels did work at all.
    """
    for _ in range(warmup):
        dist.barrier()
        function()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile) - 1))
    return ordered[index]


def rank_max_stage(
    name: str, function, args, world_size: int
) -> dict:
    """Time one stage on every rank, then keep the slowest rank's p50."""
    samples = time_calls(function, args.warmup, args.iters)
    local = {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    per_rank_p50 = [entry["p50_ms"] for entry in gathered]
    return {
        "stage": name,
        "p50_ms": max(per_rank_p50),
        "p95_ms": max(entry["p95_ms"] for entry in gathered),
        "slowest_rank": per_rank_p50.index(max(per_rank_p50)),
        "per_rank_p50_ms": per_rank_p50,
        "per_rank": gathered,
    }


def gigabytes_per_second(byte_count: int, p50_ms: float) -> float:
    return byte_count / (p50_ms / 1000.0) / 1e9


def print_table(record: dict) -> None:
    stages = record["stages"]
    verdict = record["verdict"]
    header = (
        f"{'stage':<20} {'p50 ms':>9} {'p95 ms':>9} {'GB/s':>9} "
        f"{'slowest rank':>13}"
    )
    print(header)
    print("-" * len(header))
    for name in ("dispatch", "combine", "gemm_only", "dispatch_with_gemm"):
        stage = stages[name]
        rate = stage.get("gigabytes_per_second")
        print(
            f"{name:<20} {stage['p50_ms']:>9.4f} {stage['p95_ms']:>9.4f} "
            f"{(f'{rate:.1f}' if rate is not None else '-'):>9} "
            f"{stage['slowest_rank']:>13}"
        )
    print(
        f"dispatch {verdict['dispatch_gigabytes_per_second']:.1f} GB/s "
        f"(line {DISPATCH_MIN_GBPS:.0f}), combine "
        f"{verdict['combine_gigabytes_per_second']:.1f} GB/s "
        f"(line {COMBINE_MIN_GBPS:.0f}), co-resident GEMM slowdown "
        f"{verdict['gemm_slowdown'] * 100:.2f}% "
        f"(line {GEMM_SLOWDOWN_LIMIT * 100:.0f}%)"
    )
    print(
        "VERDICT"
        f"|dispatch_pass={verdict['dispatch_pass']}"
        f"|combine_pass={verdict['combine_pass']}"
        f"|gemm_pass={verdict['gemm_pass']}"
        f"|all_pass={verdict['all_pass']}"
        f"|out={record['output']}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required; this benchmark times kernels")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != EP_SIZE:
        raise RuntimeError(
            f"the step-2 comm benchmark is an EP{EP_SIZE} contract, got "
            f"{world_size} ranks"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(
            f"{device} reports capability {torch.cuda.get_device_capability(device)}; "
            "the warp-role entries are SM90 only"
        )
    dist.init_process_group(backend="nccl", device_id=device)

    try:
        comm_bench = resolve_entry("fp8_block_warprole_comm_bench_out")
        comm_gemm_bench = resolve_entry("fp8_block_warprole_comm_gemm_bench_out")
        gemm_only = resolve_entry(GEMM_ENTRY)

        state = build_state(rank, world_size, device)
        shared = comm_args(state)
        gemm = state["gemm"]

        stages = {
            "dispatch": rank_max_stage(
                "dispatch",
                lambda: comm_bench(DISPATCH_MODE, *shared),
                args,
                world_size,
            ),
            "combine": rank_max_stage(
                "combine",
                lambda: comm_bench(COMBINE_MODE, *shared),
                args,
                world_size,
            ),
            "gemm_only": rank_max_stage(
                "gemm_only", lambda: gemm_only(*gemm), args, world_size
            ),
            "dispatch_with_gemm": rank_max_stage(
                "dispatch_with_gemm",
                lambda: comm_gemm_bench(*shared, *gemm),
                args,
                world_size,
            ),
        }

        # Loud guard against reporting a bandwidth for a kernel that exited
        # early: dispatch publishes rows into x_ready, combine publishes one
        # arrival per rank into push_done, and every rank has finished all of
        # its launches by the time the stage gather returns.
        if int(state["x_ready"].sum().item()) == 0:
            raise RuntimeError("dispatch published no rows; x_ready stayed zero")
        if int(state["push_done"].item()) < world_size:
            raise RuntimeError(
                "combine published fewer arrivals than there are ranks: "
                f"push_done={int(state['push_done'].item())}"
            )

        active_rows = state["active_rows"]
        dispatch_bytes = active_rows * (HIDDEN + (HIDDEN // 128) * 4)
        combine_bytes = active_rows * HIDDEN * 2
        stages["dispatch"]["bytes"] = dispatch_bytes
        stages["dispatch"]["gigabytes_per_second"] = gigabytes_per_second(
            dispatch_bytes, stages["dispatch"]["p50_ms"]
        )
        stages["combine"]["bytes"] = combine_bytes
        stages["combine"]["gigabytes_per_second"] = gigabytes_per_second(
            combine_bytes, stages["combine"]["p50_ms"]
        )
        slowdown = (
            stages["dispatch_with_gemm"]["p50_ms"] / stages["gemm_only"]["p50_ms"]
            - 1.0
        )
        verdict = {
            "dispatch_gigabytes_per_second": stages["dispatch"][
                "gigabytes_per_second"
            ],
            "combine_gigabytes_per_second": stages["combine"][
                "gigabytes_per_second"
            ],
            "gemm_slowdown": slowdown,
            "dispatch_pass": stages["dispatch"]["gigabytes_per_second"]
            >= DISPATCH_MIN_GBPS,
            "combine_pass": stages["combine"]["gigabytes_per_second"]
            >= COMBINE_MIN_GBPS,
            "gemm_pass": slowdown <= GEMM_SLOWDOWN_LIMIT,
            "lines": {
                "dispatch_min_gigabytes_per_second": DISPATCH_MIN_GBPS,
                "combine_min_gigabytes_per_second": COMBINE_MIN_GBPS,
                "gemm_slowdown_limit": GEMM_SLOWDOWN_LIMIT,
            },
        }
        verdict["all_pass"] = (
            verdict["dispatch_pass"]
            and verdict["combine_pass"]
            and verdict["gemm_pass"]
        )

        record = {
            "schema": "bench-warprole-comm.v1",
            "output": args.output,
            "config": {
                "warmup": args.warmup,
                "iters": args.iters,
                "ep_size": world_size,
                "num_local_tokens": NUM_LOCAL_TOKENS,
                "hidden": HIDDEN,
                "topk": TOPK,
                "num_local_experts": NUM_LOCAL_EXPERTS,
                "expert_padding": EXPERT_PADDING,
                "active_rows": active_rows,
                "schedule_capacity": state["capacity"],
                "gemm_entry": GEMM_ENTRY,
                "gemm_rows": state["gemm_rows"],
                "gemm_n": GEMM_N,
                "gemm_k": GEMM_K,
                "gemm_experts": GEMM_EXPERTS,
                "gemm_row_pattern": GEMM_ROW_PATTERN,
                "gemm_seed": GEMM_SEED,
                "timing_boundary": "binding + kernel, one barrier before each call",
                "reported_statistic": "rank-max of the per-rank p50",
            },
            "provenance": provenance(device, rank, world_size),
            "stages": stages,
            "verdict": verdict,
        }

        if rank == 0:
            output = pathlib.Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            with tmp.open("w") as sink:
                json.dump(record, sink, indent=1)
            os.replace(tmp, output)
            print_table(record)
        dist.barrier()
    finally:
        functional.clear_workspace_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
