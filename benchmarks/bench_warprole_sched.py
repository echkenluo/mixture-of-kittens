#!/usr/bin/env python3
"""Step 3 of the warp-role megakernel plan: what the scheduling itself costs.

The megakernel does two grouped GEMMs per routed row -- W13 with a fused
SwiGLU/FP8 epilogue, then W2 -- and wraps them in a task stream, three warp
roles and four dependency counters.  This harness prices the wrapper.  It runs
the fused kernel with every benchmark-only knob set

    MOK_WARPROLE_NO_DEPS=1 MOK_WARPROLE_COMM_OFF=1 MOK_WARPROLE_REDUCE_OFF=1

so the comm warpgroup idles, the phase-5 reduce is skipped and no role waits on
a counter.  What is left inside the launch is exactly the GEMM work plus the
task decode and the warp-role handover.  The reference is the same GEMM work
done by the two standalone entries on the same buffers:

    fp8_block_warprole_w13_<variant>_out   routed_x -> hidden, hidden_scale
    fp8_block_warprole_gemm_<variant>_out  hidden   -> routed_y

    overhead = (fused_p50 - (w13_p50 + w2_p50)) / (w13_p50 + w2_p50)

and the pass line is 3%.

The knobs make the fused output garbage, so nothing here checks values; the
inputs are still the real thing.  Before any timing the split dispatch runs
once, which fills routed_x, routed_x_scale and m_indices with this rank's
actual routed rows -- the fused kernel with the comm warpgroup off reads
whatever is already in those buffers, which is what makes the two arms read
identical bytes.

The timed region is one binding call plus its kernel, on both arms.  The
megakernel's Python driver (``mok.warprole.warprole_forward``) also zeroes the
counters, takes the workspace lease and reads the trap record; those are real
costs of the production path but they are not scheduling, and charging them to
the fused arm while the standalone arm pays nothing comparable would inflate
the number this harness exists to measure.  They run outside the timed blocks.

Launch on four H20s of one node:

    torchrun --standalone --nproc-per-node=4 -m benchmarks.bench_warprole_sched
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import time
from dataclasses import dataclass

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")
# Set before the first entry call, and never unset: the C++ entry reads all
# three with getenv on every call (csrc/sm90_fp8_block_warprole.cuh env_flag),
# and it refuses COMM_OFF or REDUCE_OFF unless NO_DEPS is on as well.
WARPROLE_KNOBS = {
    "MOK_WARPROLE_NO_DEPS": "1",
    "MOK_WARPROLE_COMM_OFF": "1",
    "MOK_WARPROLE_REDUCE_OFF": "1",
}
os.environ.update(WARPROLE_KNOBS)

import torch
import torch.distributed as dist

from mok import _C, functional, warprole

EP_SIZE = 4
TOTAL_EXPERTS = 64
LOCAL_EXPERTS = TOTAL_EXPERTS // EP_SIZE
HIDDEN = 4096
INTERMEDIATE = 2048
TOPK = 6
M_TILE = 64
K_GROUP = 128
FP8_MAX = 448.0
SWIGLU_LIMIT = 10.0
# Expert segments are padded to one M64 tile, the granularity the warprole task
# decode works in; the same value tests/test_warprole_ep4.py uses.
EXPERT_PADDING = 64
SEED = 20260904
VARIANTS = ("c1s6", "c2s4")
W13_ENTRIES = {
    "c1s6": "fp8_block_warprole_w13_c1s6_out",
    "c2s4": "fp8_block_warprole_w13_c2s4_out",
}
W2_ENTRIES = {
    "c1s6": "fp8_block_warprole_gemm_c1s6_out",
    "c2s4": "fp8_block_warprole_gemm_c2s4_out",
}
AA_DRIFT_LIMIT = 0.0025    # |p50(A1) - p50(A2)| / midpoint
OVERHEAD_LIMIT = 0.03      # the plan's step-3 pass line


@dataclass(frozen=True, slots=True)
class Case:
    """One measured shape, mirroring tests/test_warprole_ep4.py::Case.

    ``capacity_multiplier`` feeds ``MoKConfig.schedule_capacity_multiplier``,
    which the workspace turns into ``max(2, ceil(ep_size * multiplier))`` times
    ``tokens * topk`` rows.  3888 tokens round up to a 3904-token graph bucket,
    which leaves the schedule a padded tail -- the second shape exists for
    exactly that reason.
    """

    name: str
    effective_tokens: int
    capacity_multiplier: float

    @property
    def graph_tokens(self) -> int:
        return math.ceil(self.effective_tokens / M_TILE) * M_TILE


CASES = {
    case.name: case
    for case in (
        Case("uniform_2048", 2048, 0.5),
        Case("tail_3888", 3888, 0.5),
        Case("service_uniform_1024", 1024, 1.25),
        Case("service_uniform_2048", 2048, 1.25),
    )
}


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _subset(raw: str, allowed, flag: str) -> tuple[str, ...]:
    names = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = [name for name in names if name not in allowed]
    if not names or unknown:
        raise ValueError(f"{flag} must be a subset of {sorted(allowed)}, got {raw!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{flag} must not repeat a name, got {raw!r}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scheduling overhead of the warp-role megakernel: the knobbed "
            "fused launch against the two standalone GEMM entries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default=f"/mok/runtime-logs/warprole-step3-{utc_stamp()}.json",
        help="JSON result path on rank 0; the parent directory is created if needed",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="untimed calls before each timed block"
    )
    parser.add_argument(
        "--iters", type=int, default=30, help="timed samples per arm, shape and variant"
    )
    parser.add_argument(
        "--calls-per-sample",
        type=int,
        default=5,
        help="back-to-back calls inside one CUDA-event pair; a sample is the mean per call",
    )
    parser.add_argument(
        "--burn-in-seconds",
        type=float,
        default=2.0,
        help="run the fused arm untimed for this long before the first timed block of a case",
    )
    parser.add_argument(
        "--knobs",
        default="all",
        choices=("all", "nodeps", "comm", "reduce", "real"),
        help=(
            "all: NO_DEPS + COMM_OFF + REDUCE_OFF (scheduling overhead only); "
            "nodeps: NO_DEPS alone, so the comm warpgroup and the final reduce run "
            "(the difference to 'all' is their contribution); "
            "comm: NO_DEPS + REDUCE_OFF (comm warpgroup on, reduce off); "
            "reduce: NO_DEPS + COMM_OFF (reduce on, comm warpgroup off); "
            "real: no knob at all, the production kernel with its dependency waits "
            "(the difference to 'nodeps' is what the waits cost); the counters are "
            "reset by the prepare entry before every fused call, as in production"
        ),
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "real mode only: after the timed blocks run one more fused call with "
            "MOK_WARPROLE_PROBE=1 and report the per-CTA globaltimer stamps "
            "(rank barrier, dispatch, first W13, last task, combine, push_done, reduce) "
            "relative to the earliest CTA entry, in microseconds"
        ),
    )
    parser.add_argument(
        "--total-experts", type=int, choices=(64, 256), default=64,
        help="global expert count; use 256 for the current DSV4 service geometry",
    )
    parser.add_argument(
        "--cases",
        default="uniform_2048,tail_3888",
        help="comma-separated subset of " + ",".join(CASES),
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma-separated subset of " + ",".join(VARIANTS),
    )
    args = parser.parse_args()
    args.cases = _subset(args.cases, CASES, "--cases")
    args.variants = _subset(args.variants, VARIANTS, "--variants")
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
    """Read-only record of what produced these numbers."""
    # A frozen harness may be mounted separately from the immutable runtime.
    repo = pathlib.Path(functional.__file__).resolve().parents[1]
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
        "warprole_knobs": {
            name: os.environ.get(name, "unset") for name in WARPROLE_KNOBS
        },
        "rank": rank,
        "world_size": world_size,
        "timestamp_utc": utc_stamp(),
    }


def resolve_entry(name: str):
    entry = getattr(_C, name, None)
    if entry is None:
        raise RuntimeError(
            f"the binding {name!r} is missing from "
            f"{getattr(_C, '__file__', 'the mok extension')}. The warp-role "
            "entries exist only after the csrc/sm90_fp8_block_warprole*.cuh "
            "files and the matching m.def lines in csrc/bindings.cu are "
            "compiled; rebuild with benchmarks/warprole/build_sm90.sh and "
            "reinstall the extension before running this harness."
        )
    return entry


def chunk_bytes_for(graph_tokens: int) -> int:
    """Largest legal all-gather chunk that divides one rank's route buffer.

    Same rule as tests/test_warprole_ep4.py: ``build_schedule`` refuses a chunk
    that does not divide ``tokens * topk * 4`` bytes, and the 3904-token bucket
    is not divisible by the 2048-byte default.  Candidates stay multiples of
    128 so the TMA bulk copy inside the all-gather keeps its alignment.
    """
    route_bytes = graph_tokens * TOPK * 4
    for candidate in range(2048, 0, -128):
        if route_bytes % candidate == 0:
            return candidate
    raise RuntimeError(f"no legal chunk size for {route_bytes} route bytes")


def quantize_k128(activations: torch.Tensor):
    """FP8 activations with one FP32 scale per 128 columns.

    Same generator as tests/test_warprole_ep4.py::quantize_k128.  Nothing here
    is compared on values, so this only has to be a valid FP8/K128 pair.
    """
    rows, columns = activations.shape
    grouped = activations.float().view(rows, columns // K_GROUP, K_GROUP)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    quantized = (grouped / scale).clamp(-FP8_MAX, FP8_MAX)
    return (
        quantized.view(rows, columns).to(torch.float8_e4m3fn).contiguous(),
        scale.view(rows, columns // K_GROUP).contiguous(),
    )


def make_weights(device: torch.device):
    """Random FP8 expert weights in the plain [E, 2I, K] / [E, K, I] layout.

    Same generator as tests/test_warprole_ep4.py::make_weights: values clamped
    into FP8 range before the cast, block scales in [0.01, 0.10).  The seed does
    not depend on the rank; each rank holds TOTAL_EXPERTS / EP_SIZE experts. The
    fused kernel and the standalone W13 entry read the gate/up weight as the
    model stores it, so no reordered copy exists.
    """
    generator = torch.Generator(device=device).manual_seed(SEED)

    def fp8(*shape: int) -> torch.Tensor:
        values = torch.randn(
            shape, generator=generator, device=device, dtype=torch.bfloat16
        )
        return values.clamp(-3, 3).to(torch.float8_e4m3fn).contiguous()

    def block_scale(*shape: int) -> torch.Tensor:
        values = torch.rand(shape, generator=generator, device=device)
        return (values * 0.09 + 0.01).to(torch.float32).contiguous()

    w13 = fp8(LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w13_scale = block_scale(
        LOCAL_EXPERTS, 2 * INTERMEDIATE // K_GROUP, HIDDEN // K_GROUP
    )
    w2 = fp8(LOCAL_EXPERTS, HIDDEN, INTERMEDIATE)
    w2_scale = block_scale(
        LOCAL_EXPERTS, HIDDEN // K_GROUP, INTERMEDIATE // K_GROUP
    )
    return w13, w13_scale, w2, w2_scale


def make_routing(case: Case, rank: int, device: torch.device):
    """Uniform top-6 routing over the configured global experts.

    Tokens past ``effective_tokens`` are padding: expert -1 keeps them out of
    the schedule.  The fused kernel never reduces them here anyway, because
    MOK_WARPROLE_REDUCE_OFF skips phase 5.
    """
    generator = torch.Generator(device=device).manual_seed(SEED + rank)
    probabilities = torch.ones(TOTAL_EXPERTS, device=device)
    chosen = torch.multinomial(
        probabilities.expand(case.effective_tokens, TOTAL_EXPERTS).contiguous(),
        TOPK,
        replacement=False,
        generator=generator,
    ).to(torch.int32)

    top_experts = torch.full(
        (case.graph_tokens, TOPK), -1, dtype=torch.int32, device=device
    )
    top_experts[: case.effective_tokens] = chosen
    router_weights = torch.zeros(
        (case.graph_tokens, TOPK), dtype=torch.float32, device=device
    )
    logits = torch.rand(
        (case.effective_tokens, TOPK),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    router_weights[: case.effective_tokens] = logits / logits.sum(
        dim=-1, keepdim=True
    )
    return top_experts.contiguous(), router_weights.contiguous()


@dataclass(slots=True)
class Harness:
    """Everything one shape needs, with its routed rows already dispatched."""

    case: Case
    workspace: functional.MoKFP8RouteWorkspace
    state: warprole.WarpRoleState
    schedule: functional.MoKSchedule
    active_rows: int
    router_weights_flat: torch.Tensor
    topk_ids_flat: torch.Tensor
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


def build_harness(
    case: Case, rank: int, device: torch.device, weights
) -> Harness:
    """Create the workspace, state and schedule, then dispatch once.

    Every step is collective in lockstep across the four ranks: the workspace
    rendezvouses symmetric memory, the state rendezvouses its ``push_done``
    counter, ``build_schedule`` all-gathers the routes and the dispatch runs a
    cross-rank barrier.  All of it happens before any timed block.
    """
    w13, w13_scale, w2, w2_scale = weights
    config = functional.MoKConfig(
        schedule_capacity_multiplier=case.capacity_multiplier,
        all_gather_top_experts_chunk_bytes=chunk_bytes_for(case.graph_tokens),
    )
    workspace = functional.get_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=case.graph_tokens,
        hidden_size=HIDDEN,
        topk=TOPK,
        num_local_experts=LOCAL_EXPERTS,
    )
    capacity = workspace.schedule_capacity
    state = warprole.get_warprole_state(
        workspace, dist.group.WORLD, device=device, capacity=capacity
    )

    top_experts, router_weights = make_routing(case, rank, device)
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=LOCAL_EXPERTS,
        expert_padding=EXPERT_PADDING,
    )
    active_rows = int(schedule.num_tokens.item())
    if active_rows % M_TILE or not 0 < active_rows <= capacity:
        raise RuntimeError(
            f"{case.name}: {active_rows} routed rows are not an M64-aligned "
            f"count inside capacity {capacity}"
        )

    activations = torch.randn(
        (case.graph_tokens, HIDDEN),
        generator=torch.Generator(device=device).manual_seed(SEED + 97 * rank),
        device=device,
        dtype=torch.bfloat16,
    )
    x_fp8, x_scale = quantize_k128(activations)

    # The routed rows both arms read.  The fused kernel runs with the comm
    # warpgroup off, so it never refills these buffers; this single dispatch is
    # what puts real activations, real K128 scales and a real expert id per row
    # behind both arms.
    functional.acquire_workspace_lease(workspace)
    functional.dispatch_fp8_block(
        workspace,
        schedule,
        x_fp8,
        x_scale,
        trim_to_active_rows=False,
        prepare_combine=True,
    )
    functional.release_workspace_lease(workspace)

    # The W2 half of the fused launch reads `hidden` without waiting for the
    # W13 half to write it (MOK_WARPROLE_NO_DEPS), so the buffer must start out
    # defined rather than whatever the allocator handed back.
    state.hidden.zero_()
    state.hidden_scale.zero_()
    state.routed_y.zero_()

    torch.cuda.synchronize(device)
    dist.barrier()
    return Harness(
        case=case,
        workspace=workspace,
        state=state,
        schedule=schedule,
        active_rows=active_rows,
        router_weights_flat=router_weights.view(-1),
        topk_ids_flat=top_experts.view(-1),
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
    )


def fused_kwargs(harness: Harness) -> dict:
    """The fused entry's arguments, byte for byte what warprole_forward passes.

    Kept as a prebuilt dict so the timed call is one binding invocation and no
    tensor views are constructed inside the timed region.
    """
    workspace = harness.workspace
    state = harness.state
    schedule = harness.schedule
    return {
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "x_scale": workspace.x_scale_buffer,
        "x_scale_ptrs": workspace.x_scale_buffer_ptrs,
        "routed_x": state.routed_x,
        "routed_x_scale": state.routed_x_scale,
        "m_indices": state.m_indices,
        "schedule_peer_rank": schedule.peer_rank,
        "schedule_peer_token_idx": schedule.peer_token_idx,
        "num_tokens": schedule.num_tokens,
        "tokens_per_expert": schedule.tokens_per_expert,
        "topk": TOPK,
        "w13": harness.w13,
        "w13_scale": harness.w13_scale,
        "w2": harness.w2,
        "w2_scale": harness.w2_scale,
        "hidden": state.hidden,
        "hidden_scale": state.hidden_scale,
        "routed_y": state.routed_y,
        "combine_ptrs": workspace.combine_buffer_ptrs,
        "combine_local": workspace.combine_buffer,
        "weights": harness.router_weights_flat,
        "topk_ids": harness.topk_ids_flat,
        "output": state.output,
        "push_done_ptrs": state.push_done_ptrs,
        "ep_rank": workspace.ep_rank,
        "x_ready": state.x_ready,
        "hidden_ready": state.hidden_ready,
        "y_ready": state.y_ready,
        "push_done_local": state.push_done_local,
        "barrier_buffer": workspace.barrier_buffer,
        "barrier_target": workspace.barrier_target,
        "barrier_multicast_ptr": workspace.barrier_buffer_multicast_ptr,
        "input_expected_scratch": state.input_expected_scratch,
        "trap_record_ptr": workspace.trap_record_ptr,
        "swiglu_limit": SWIGLU_LIMIT,
        "spin_limit": warprole.DEFAULT_SPIN_LIMIT,
    }


def w13_kwargs(harness: Harness) -> dict:
    """The W13 half of the fused launch, done by the standalone entry."""
    state = harness.state
    return {
        "input": state.routed_x,
        "input_scale": state.routed_x_scale,
        "w13": harness.w13,
        "w13_scale": harness.w13_scale,
        "m_indices": state.m_indices,
        "num_tokens": harness.schedule.num_tokens,
        "hidden": state.hidden,
        "hidden_scale": state.hidden_scale,
        "swiglu_limit": SWIGLU_LIMIT,
    }


def w2_kwargs(harness: Harness) -> dict:
    """The W2 half of the fused launch, done by the standalone entry."""
    state = harness.state
    return {
        "input": state.hidden,
        "weight": harness.w2,
        "input_scale": state.hidden_scale,
        "weight_scale": harness.w2_scale,
        "m_indices": state.m_indices,
        "num_tokens": harness.schedule.num_tokens,
        "output": state.routed_y,
    }


def time_calls(function, warmup: int, iters: int, calls_per_sample: int) -> list[float]:
    """One CUDA-event pair around `calls_per_sample` back-to-back calls, a device
    synchronize after each pair; the sample is the mean time per call.  Same
    reasoning as bench_warprole_gemm: with one call per pair the host-side launch
    gap sits inside every sample, and on GPU9 (load average ~35) that alone moved
    the two fused blocks of a case apart by 0.3-0.8%."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / calls_per_sample)
    return samples


def burn_in(function, seconds: float) -> None:
    """Run `function` untimed for `seconds` of wall clock before the first timed block."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for _ in range(10):
            function()
        torch.cuda.synchronize()


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(samples: list[float]) -> dict:
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def measure_local(harness: Harness, variant: str, args) -> dict:
    """A/B/A on one rank: fused, the standalone pair, fused again.

    The four ranks are released together before each block, but not before each
    call: with the comm warpgroup off the launch touches no peer, so a per-call
    barrier would only add jitter to a measurement of local work.
    """
    fused_entry = resolve_entry(warprole.VARIANT_ENTRIES[variant])
    w13_entry = resolve_entry(W13_ENTRIES[variant])
    w2_entry = resolve_entry(W2_ENTRIES[variant])
    fused = fused_kwargs(harness)
    w13 = w13_kwargs(harness)
    w2 = w2_kwargs(harness)

    prepare = dict(
        x_ready=harness.state.x_ready,
        hidden_ready=harness.state.hidden_ready,
        y_ready=harness.state.y_ready,
        push_done_local=harness.state.push_done_local,
        push_done=harness.state.push_done,
        input_expected_scratch=harness.state.input_expected_scratch,
    )

    def run_fused() -> None:
        if args.knobs == "real":
            # With the waits live every call must start from zeroed counters; the
            # production launcher (mok/warprole.py) issues the same prepare call.
            _C.fp8_block_warprole_prepare_out(**prepare)
        fused_entry(**fused)

    def run_w13() -> None:
        w13_entry(**w13)

    def run_w2() -> None:
        w2_entry(**w2)

    # Leaves the counters at zero and gives `hidden`/`routed_y` real values, so
    # neither arm's first timed call is the first launch to touch them.
    _C.fp8_block_warprole_prepare_out(**prepare)
    run_w13()
    run_w2()
    torch.cuda.synchronize()
    dist.barrier()

    if args.burn_in_seconds > 0:
        burn_in(run_fused, args.burn_in_seconds)
    dist.barrier()
    a1 = summarize(time_calls(run_fused, args.warmup, args.iters, args.calls_per_sample))
    dist.barrier()
    w13_row = summarize(time_calls(run_w13, args.warmup, args.iters, args.calls_per_sample))
    dist.barrier()
    w2_row = summarize(time_calls(run_w2, args.warmup, args.iters, args.calls_per_sample))
    dist.barrier()
    a2 = summarize(time_calls(run_fused, args.warmup, args.iters, args.calls_per_sample))
    dist.barrier()

    if int(harness.workspace.trap_record[0].item()) != 0:
        raise RuntimeError(
            functional.format_trap_record(harness.workspace)
            or "MOK_TRAP|claimed but payload not committed"
        )

    midpoint = 0.5 * (a1["p50_ms"] + a2["p50_ms"])
    drift = abs(a2["p50_ms"] - a1["p50_ms"]) / midpoint
    standalone_p50 = w13_row["p50_ms"] + w2_row["p50_ms"]
    probe = None
    probe_stamps = None
    if args.probe and args.knobs == "real":
        dist.barrier()
        os.environ["MOK_WARPROLE_PROBE"] = "1"
        try:
            run_fused()
            torch.cuda.synchronize()
        finally:
            os.environ["MOK_WARPROLE_PROBE"] = "0"
        stamps = _C.fp8_block_warprole_probe_read().cpu()
        probe_stamps = stamps.tolist()
        probe = summarize_probe(stamps)
    return {
        "rank": dist.get_rank(),
        "order": ["fused_a1", "w13", "w2", "fused_a2"],
        "fused_a1": a1,
        "w13": w13_row,
        "w2": w2_row,
        "fused_a2": a2,
        "fused_aa_drift": drift,
        "fused_midpoint_p50_ms": midpoint,
        "standalone_pair_p50_ms": standalone_p50,
        "overhead": (midpoint - standalone_p50) / standalone_p50,
        "probe_us": probe,
        "probe_raw_globaltimer_ns": probe_stamps,
    }


PROBE_SLOT_NAMES = (
    "entry", "rank_barrier", "dispatch_done", "first_w13", "last_task",
    "combine_finish", "push_done", "reduce_done",
)


def summarize_probe(stamps: torch.Tensor) -> dict:
    """Per slot, the min / median / max over CTAs in microseconds after the earliest entry.

    Slots a CTA never reaches stay zero (e.g. reduce_done on nothing) and are dropped.
    """
    stamps = stamps.to(torch.float64)
    t0 = stamps[:, 0][stamps[:, 0] > 0].min()
    out = {}
    for slot, name in enumerate(PROBE_SLOT_NAMES):
        col = stamps[:, slot]
        col = col[col > 0]
        if col.numel() == 0:
            continue
        rel = (col - t0) / 1000.0
        out[name] = {
            "min": float(rel.min()),
            "median": float(rel.median()),
            "max": float(rel.max()),
            "ctas": int(col.numel()),
        }
    return out


def combine_ranks(local: dict, world_size: int) -> dict:
    """Gather the per-rank results and keep the worst rank on every line."""
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    per_rank_overhead = [entry["overhead"] for entry in gathered]
    per_rank_drift = [entry["fused_aa_drift"] for entry in gathered]
    worst = per_rank_overhead.index(max(per_rank_overhead))
    return {
        "per_rank": gathered,
        "per_rank_overhead": per_rank_overhead,
        "per_rank_fused_midpoint_p50_ms": [
            entry["fused_midpoint_p50_ms"] for entry in gathered
        ],
        "per_rank_standalone_pair_p50_ms": [
            entry["standalone_pair_p50_ms"] for entry in gathered
        ],
        "per_rank_fused_aa_drift": per_rank_drift,
        "rank_max_overhead": max(per_rank_overhead),
        "worst_rank": worst,
        "rank_max_fused_aa_drift": max(per_rank_drift),
        "aa_drift_limit": AA_DRIFT_LIMIT,
        "valid": max(per_rank_drift) <= AA_DRIFT_LIMIT,
        "overhead_limit": OVERHEAD_LIMIT,
        "pass": max(per_rank_overhead) <= OVERHEAD_LIMIT,
    }


def print_probes(record: dict) -> None:
    for key in sorted(record["measurements"]):
        for entry in record["measurements"][key]["per_rank"]:
            probe = entry.get("probe_us")
            if not probe:
                continue
            fields = "|".join(
                f"{name}={v['min']:.0f}/{v['median']:.0f}/{v['max']:.0f}" for name, v in probe.items()
            )
            print(f"WARPROLE_PROBE|{key}|rank={entry['rank']}|{fields}")


def print_verdicts(record: dict) -> None:
    for key in sorted(record["measurements"]):
        result = record["measurements"][key]
        # The two latencies printed are the worst rank's own, so the pair the
        # line shows is exactly the pair the overhead on it was computed from.
        worst = result["worst_rank"]
        print(
            f"WARPROLE_STEP3|{key}"
            f"|effective_tokens={result['effective_tokens']}"
            f"|graph_tokens={result['graph_tokens']}"
            f"|active_rows={result['active_rows']}"
            f"|fused_ms={result['per_rank_fused_midpoint_p50_ms'][worst]:.4f}"
            f"|standalone_ms={result['per_rank_standalone_pair_p50_ms'][worst]:.4f}"
            f"|rank_max_overhead={result['rank_max_overhead'] * 100:.2f}%"
            f"|worst_rank={worst}"
            f"|aa_drift={result['rank_max_fused_aa_drift'] * 100:.3f}%"
            f"|valid={result['valid']}"
            f"|limit={OVERHEAD_LIMIT * 100:.0f}%"
            f"|pass={result['pass']}",
            flush=True,
        )
    print(
        f"WARPROLE_STEP3_VERDICT|all_pass={record['verdict']['all_pass']}"
        f"|all_valid={record['verdict']['all_valid']}"
        f"|step3_gate_applicable={record['verdict']['step3_overhead_gate_applicable']}"
        f"|out={record['output']}",
        flush=True,
    )


def require_warprole(device: torch.device, variants: tuple[str, ...]) -> None:
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("the warp-role entries are SM90 only")
    for variant in variants:
        for name in (
            warprole.VARIANT_ENTRIES[variant],
            W13_ENTRIES[variant],
            W2_ENTRIES[variant],
        ):
            resolve_entry(name)
    resolve_entry("fp8_block_warprole_prepare_out")


def main() -> None:
    global TOTAL_EXPERTS, LOCAL_EXPERTS
    args = parse_args()
    TOTAL_EXPERTS = args.total_experts
    LOCAL_EXPERTS = TOTAL_EXPERTS // EP_SIZE
    # The entry reads the knobs with getenv on every call, so flipping them
    # here (before any entry call) is enough; NO_DEPS stays on in every mode
    # except "real", which runs the production kernel.
    if args.knobs in ("nodeps", "comm", "real"):
        os.environ["MOK_WARPROLE_COMM_OFF"] = "0"
    if args.knobs in ("nodeps", "reduce", "real"):
        os.environ["MOK_WARPROLE_REDUCE_OFF"] = "0"
    if args.knobs == "real":
        os.environ["MOK_WARPROLE_NO_DEPS"] = "0"
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required; this benchmark times kernels")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != EP_SIZE:
        raise RuntimeError(
            f"the step-3 harness is an EP{EP_SIZE} contract, got {world_size} ranks"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    try:
        require_warprole(device, args.variants)
        weights = make_weights(device)
        measurements = {}
        for case_name in args.cases:
            case = CASES[case_name]
            harness = build_harness(case, rank, device, weights)
            for variant in args.variants:
                local = measure_local(harness, variant, args)
                result = combine_ranks(local, world_size)
                result.update(
                    {
                        "case": case_name,
                        "variant": variant,
                        "effective_tokens": case.effective_tokens,
                        "graph_tokens": case.graph_tokens,
                        "active_rows": harness.active_rows,
                        "schedule_capacity": harness.workspace.schedule_capacity,
                    }
                )
                measurements[f"{case_name}/{variant}"] = result

        verdict = {
            "formal_quality_or_performance_go": False,
            "step3_overhead_gate_applicable": args.knobs == "all",
            "gate": (
                f"rank-max overhead <= {OVERHEAD_LIMIT * 100:.0f}% in every "
                f"shape and variant; fused A/A drift <= "
                f"{AA_DRIFT_LIMIT * 100:.2f}%"
            ),
            "all_pass": all(
                result["pass"] for result in measurements.values()
            ),
            "all_valid": all(
                result["valid"] for result in measurements.values()
            ),
            "failing": sorted(
                key for key, result in measurements.items() if not result["pass"]
            ),
        }
        record = {
            "schema": "bench-warprole-sched.v2",
            "output": args.out,
            "config": {
                "warmup": args.warmup,
                "iters": args.iters,
                "calls_per_sample": args.calls_per_sample,
                "burn_in_seconds": args.burn_in_seconds,
                "knobs": args.knobs,
                "ep_size": world_size,
                "cases": list(args.cases),
                "variants": list(args.variants),
                "hidden": HIDDEN,
                "intermediate": INTERMEDIATE,
                "topk": TOPK,
                "num_local_experts": LOCAL_EXPERTS,
                "total_experts": TOTAL_EXPERTS,
                "expert_padding": EXPERT_PADDING,
                "swiglu_limit": SWIGLU_LIMIT,
                "seed": SEED,
                "fused_entries": {
                    variant: warprole.VARIANT_ENTRIES[variant]
                    for variant in args.variants
                },
                "standalone_entries": {
                    variant: [W13_ENTRIES[variant], W2_ENTRIES[variant]]
                    for variant in args.variants
                },
                "timing_boundary": (
                    "CUDA-event elapsed time of binding + kernel calls; real mode "
                    "includes prepare counter reset in every fused call. Workspace "
                    "lease, setup and trap read are outside. The standalone W13/W2 "
                    "pair excludes dispatch, activation/quant, combine and final reduce; "
                    "real-mode overhead is not a matched full-pipeline speedup."
                ),
                "reported_statistic": (
                    "per-rank p50, and the rank-max of the per-rank overhead"
                ),
                "overhead_formula": (
                    "(fused_midpoint_p50 - (w13_p50 + w2_p50)) / "
                    "(w13_p50 + w2_p50)"
                ),
            },
            "provenance": provenance(device, rank, world_size),
            "measurements": measurements,
            "verdict": verdict,
        }

        if rank == 0:
            output = pathlib.Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            with tmp.open("w") as sink:
                json.dump(record, sink, indent=1)
            os.replace(tmp, output)
            print_verdicts(record)
            print_probes(record)
        dist.barrier()
    finally:
        warprole.clear_warprole_state_cache()
        functional.clear_workspace_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
