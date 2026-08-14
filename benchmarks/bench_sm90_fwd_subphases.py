"""Diagnostic phase attribution inside the SM90 BF16 forward call.

This canary-only harness reproduces ``functional.forward`` with the same
low-level operations and places CUDA events around each operation boundary.
It does not replace ``bench_sm90_fwd``: Python validation/context construction
is intentionally outside the measured path, and phase rank-max medians are not
assumed to be algebraically additive.
"""

import glob
import hashlib
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist

from benchmarks.bench_sm90_fwd import (
    COMM_SMS,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    MACROBATCH_SIZE,
    MINIBATCH_SIZE,
    NUM_EXPERTS,
    NUM_LOCAL_TOKENS,
    TOPK,
    WARMUP,
    _loadavg,
    gpu_snapshot,
    rank_max_samples,
)
from benchmarks.utils import TIMED_ITERS, get_num_local_experts, init_distributed
from mok import functional
from tests.utils import (
    BF16_TOLERANCE,
    generate_inputs,
    get_error_stats,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
)

OUTPUT = os.environ.get(
    "BENCH_SUBPHASE_OUTPUT", "/mok/bench-sm90-fwd-subphases.json"
)
PROFILE_ONCE = os.environ.get("BENCH_PROFILE_ONCE", "0") == "1"
PHASES = (
    "input_copies",
    "pre_megakernel_barrier",
    "fused_op_including_internal_allocs",
    "post_megakernel_barrier",
    "fwd_epilogue",
    "total",
)


def percentiles(samples):
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[max(0, int(len(ordered) * 0.95) - 1)]


def provenance():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.abspath(__file__), "rb") as stream:
        harness_sha = hashlib.sha256(stream.read()).hexdigest()
    shared_objects = sorted(glob.glob(os.path.join(repo, "mok", "_C*.so")))
    if len(shared_objects) != 1:
        raise RuntimeError(
            f"need exactly one MoK shared object, found {len(shared_objects)}"
        )
    with open(shared_objects[0], "rb") as stream:
        so_sha = hashlib.sha256(stream.read()).hexdigest()
    return {
        "source_commit_env": os.environ.get("MOK_SOURCE_COMMIT", "unset"),
        "harness_sha256": harness_sha,
        "so_path": shared_objects[0],
        "so_sha256": so_sha,
        "diagnostic_only": True,
    }


def safe_gpu_snapshot():
    try:
        return gpu_snapshot()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def main() -> None:
    wall0 = time.time()
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(
        rank,
        device,
        NUM_EXPERTS,
        num_local_experts,
        TOPK,
        NUM_LOCAL_TOKENS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
    )
    (
        x,
        topk_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        _d_output,
    ) = inputs

    config = functional.MoKConfig(
        fwd_num_comm_sms=COMM_SMS,
        bwd_num_comm_sms=COMM_SMS,
        minibatch_size=MINIBATCH_SIZE,
        macrobatch_size=MACROBATCH_SIZE,
    )
    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=x.device,
        num_local_tokens=x.shape[0],
        hidden_size=x.shape[1],
        topk=topk_experts.shape[1],
    )
    schedule = functional.build_schedule(
        workspace, config, topk_experts, num_local_experts=num_local_experts
    )
    routed_token_counts = [torch.empty_like(schedule.num_tokens) for _ in range(world_size)]
    dist.all_gather(routed_token_counts, schedule.num_tokens)
    routed_tokens_per_rank = [int(value.item()) for value in routed_token_counts]

    def mark(pair, index):
        if pair is not None:
            pair[index].record()

    def run_decomposed(event_pairs=None):
        pair = None if event_pairs is None else event_pairs["total"]
        mark(pair, 0)

        pair = None if event_pairs is None else event_pairs["input_copies"]
        mark(pair, 0)
        workspace.x_buffer.copy_(x)
        workspace.router_weight_buffer.copy_(router_weights)
        mark(pair, 1)

        pair = None if event_pairs is None else event_pairs["pre_megakernel_barrier"]
        mark(pair, 0)
        functional.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )
        mark(pair, 1)

        pair = (
            None
            if event_pairs is None
            else event_pairs["fused_op_including_internal_allocs"]
        )
        mark(pair, 0)
        (
            _x_routed,
            _gate_shared,
            _gate_routed,
            _up_shared,
            _up_routed,
            _hidden_shared,
            _hidden_routed,
            y_shared,
            _y_routed,
        ) = functional.dispatch_mlp_swiglu_combine_fwd_bf16(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            w_shared_gate,
            w_routed_gate,
            w_shared_up,
            w_routed_up,
            w_shared_down,
            w_routed_down,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.topk,
            None,
            config.fwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
        )
        mark(pair, 1)

        pair = None if event_pairs is None else event_pairs["post_megakernel_barrier"]
        mark(pair, 0)
        functional.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )
        mark(pair, 1)

        pair = None if event_pairs is None else event_pairs["fwd_epilogue"]
        mark(pair, 0)
        output = functional.fwd_epilogue(
            y_shared, workspace.combine_buffer, workspace.router_weight_buffer
        )
        mark(pair, 1)

        pair = None if event_pairs is None else event_pairs["total"]
        mark(pair, 1)
        return output

    torch.cuda.synchronize()
    setup_end = time.time()

    (ref_combine_buffer, _rg, _ru, _rh, ref_y_shared) = run_forward_reference_bf16(
        x,
        topk_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    ref_out = run_fwd_epilogue_reference(
        ref_y_shared, ref_combine_buffer, router_weights
    )
    del ref_combine_buffer, _rg, _ru, _rh, ref_y_shared
    out = run_decomposed()
    torch.cuda.synchronize()
    abs_mean, abs_max, relative = get_error_stats(ref_out, out)
    absolute_tolerance, relative_tolerance = BF16_TOLERANCE
    gate_pass = (
        all(math.isfinite(value) for value in (abs_mean, abs_max, relative))
        and abs_max <= absolute_tolerance
        and relative <= relative_tolerance
    )
    if not gate_pass:
        raise RuntimeError(
            f"correctness gate FAILED on rank {rank}: abs_mean={abs_mean:.6f} "
            f"abs_max={abs_max:.6f} relative={relative:.6f} vs tolerance "
            f"(abs={absolute_tolerance}, rel={relative_tolerance})"
        )
    del ref_out, out
    correctness_end = time.time()

    for _ in range(WARMUP):
        run_decomposed()
    torch.cuda.synchronize()
    warmup_end = time.time()

    if PROFILE_ONCE:
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        profile_output = run_decomposed()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        del profile_output
        dist.barrier()
    profile_end = time.time()

    events = {
        phase: [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(TIMED_ITERS)
        ]
        for phase in PHASES
    }
    barrier = dist.barrier(async_op=True)
    barrier.block_current_stream()
    for index in range(TIMED_ITERS):
        run_decomposed({phase: events[phase][index] for phase in PHASES})
    torch.cuda.synchronize()
    dist.barrier()
    measured_end = time.time()

    samples = {
        phase: rank_max_samples(
            [start.elapsed_time(end) for start, end in events[phase]], device
        )
        for phase in PHASES
    }

    if rank == 0:
        summaries = {}
        for phase in PHASES:
            p50, p95 = percentiles(samples[phase])
            summaries[phase] = {
                "samples_ms": [round(value, 4) for value in samples[phase]],
                "p50_ms": round(p50, 4),
                "p95_ms": round(p95, 4),
            }
        phase_sum = sum(summaries[phase]["p50_ms"] for phase in PHASES[:-1])
        total = summaries["total"]["p50_ms"]
        record = {
            "schema": "bench-sm90-fwd-subphases.v1",
            "meta": {
                "shape": {
                    "tokens_per_rank": NUM_LOCAL_TOKENS,
                    "hidden": HIDDEN_DIM,
                    "intermediate": INTERMEDIATE_DIM,
                    "experts": NUM_EXPERTS,
                    "topk": TOPK,
                    "world_size": world_size,
                },
                "comm_sms": COMM_SMS,
                "minibatch": MINIBATCH_SIZE,
                "macrobatch": MACROBATCH_SIZE,
                "schedule_capacity_per_rank": workspace.schedule_capacity,
                "routed_tokens_per_rank": routed_tokens_per_rank,
                "warmup_iters": WARMUP,
                "timed_iters": TIMED_ITERS,
                "profile_once": PROFILE_ONCE,
                "semantics": (
                    "diagnostic only; CUDA-event boundaries reproduce the low-level "
                    "BF16 forward operations; the fused-op phase includes its internal "
                    "tensor allocation/zeroing plus the megakernel; Python validation "
                    "and context construction are excluded; phase p50 values use "
                    "independent rank-max samples"
                ),
                "correctness_gate": {
                    "abs_mean": abs_mean,
                    "abs_max": abs_max,
                    "relative": relative,
                    "tolerance_abs_rel": list(BF16_TOLERANCE),
                },
                "wall_phases_s": {
                    "setup": round(setup_end - wall0, 1),
                    "correctness_gate": round(correctness_end - setup_end, 1),
                    "warmup": round(warmup_end - correctness_end, 1),
                    "profile_once": round(profile_end - warmup_end, 1),
                    "measured": round(measured_end - profile_end, 1),
                    "total": round(measured_end - wall0, 1),
                },
                "loadavg_end": _loadavg(),
                "gpu_snapshot_end": safe_gpu_snapshot(),
                "provenance": provenance(),
            },
            "phases": summaries,
            "closure": {
                "phase_p50_sum_ms": round(phase_sum, 4),
                "total_p50_ms": total,
                "total_minus_phase_sum_ms": round(total - phase_sum, 4),
                "total_minus_phase_sum_pct": round((total - phase_sum) / total * 100, 4),
            },
        }
        temporary = OUTPUT + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(record, stream, indent=1)
        os.replace(temporary, OUTPUT)
        print(
            "SUBPHASES|"
            + "|".join(
                f"{phase}={summaries[phase]['p50_ms']:.4f}ms" for phase in PHASES
            )
            + f"|closure_delta={total - phase_sum:.4f}ms"
            + f"|abs_max={abs_max:.5f}|relative={relative:.6f}|out={OUTPUT}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
