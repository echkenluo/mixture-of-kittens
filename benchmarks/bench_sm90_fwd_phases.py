"""Canary-only phase attribution for the SM90 BF16 forward path.

This does not replace ``bench_sm90_fwd`` or its frozen timing contract.  It
uses the same V4 inputs and correctness truth, then measures three independent
rank-max series after a shared warmup:

* schedule-only: build_schedule for the fixed routing input;
* forward-only: forward with one valid cached schedule;
* combined: build_schedule + forward, matching the frozen harness boundary.

The combined result must reproduce the frozen baseline before either isolated
series is used for mechanism attribution.  The two isolated medians are not
treated as algebraically additive unless their sum closes against combined.
"""

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

OUTPUT = os.environ.get("BENCH_PHASE_OUTPUT", "/mok/bench-sm90-fwd-phases.json")


def percentiles(samples):
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[max(0, int(len(ordered) * 0.95) - 1)]


def provenance():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.abspath(__file__), "rb") as stream:
        harness_sha = hashlib.sha256(stream.read()).hexdigest()
    import glob
    shared_objects = sorted(glob.glob(os.path.join(repo, "mok", "_C*.so")))
    if len(shared_objects) != 1:
        raise RuntimeError(f"need exactly one MoK shared object, found {len(shared_objects)}")
    with open(shared_objects[0], "rb") as stream:
        so_sha = hashlib.sha256(stream.read()).hexdigest()
    return {
        "source_commit_env": os.environ.get("MOK_SOURCE_COMMIT", "unset"),
        "harness_sha256": harness_sha,
        "so_path": shared_objects[0],
        "so_sha256": so_sha,
        "diagnostic_only": True,
    }


def main() -> None:
    wall0 = time.time()
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK,
                             NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    (x, topk_experts, router_weights,
     w_shared_gate, w_shared_up, w_shared_down,
     w_routed_gate, w_routed_up, w_routed_down, _d_output) = inputs

    config = functional.MoKConfig(
        fwd_num_comm_sms=COMM_SMS,
        bwd_num_comm_sms=COMM_SMS,
        minibatch_size=MINIBATCH_SIZE,
        macrobatch_size=MACROBATCH_SIZE,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=x.device,
        num_local_tokens=x.shape[0], hidden_size=x.shape[1],
        topk=topk_experts.shape[1])

    def build_schedule():
        return functional.build_schedule(
            workspace, config, topk_experts, num_local_experts=num_local_experts)

    def run_forward(schedule):
        output, _context = functional.forward(
            config, workspace, schedule, x, router_weights,
            w_shared_gate, w_shared_up, w_shared_down,
            w_routed_gate, w_routed_up, w_routed_down)
        return output

    def run_combined():
        return run_forward(build_schedule())

    torch.cuda.synchronize()
    setup_end = time.time()

    (ref_combine_buffer, _rg, _ru, _rh, ref_y_shared) = run_forward_reference_bf16(
        x, topk_experts, w_shared_gate, w_shared_up, w_shared_down,
        w_routed_gate, w_routed_up, w_routed_down)
    ref_out = run_fwd_epilogue_reference(ref_y_shared, ref_combine_buffer,
                                         router_weights)
    del ref_combine_buffer, _rg, _ru, _rh, ref_y_shared
    out = run_combined()
    torch.cuda.synchronize()
    abs_mean, abs_max, relative = get_error_stats(ref_out, out)
    absolute_tolerance, relative_tolerance = BF16_TOLERANCE
    gate_pass = (all(math.isfinite(value) for value in (abs_mean, abs_max, relative))
                 and abs_max <= absolute_tolerance
                 and relative <= relative_tolerance)
    if not gate_pass:
        raise RuntimeError(
            f"correctness gate FAILED on rank {rank}: abs_mean={abs_mean:.6f} "
            f"abs_max={abs_max:.6f} relative={relative:.6f} vs tolerance "
            f"(abs={absolute_tolerance}, rel={relative_tolerance})")
    del ref_out, out
    correctness_end = time.time()

    for _ in range(WARMUP):
        run_combined()
    torch.cuda.synchronize()
    warmup_end = time.time()

    def measure(call):
        events = [(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
                  for _ in range(TIMED_ITERS)]
        barrier = dist.barrier(async_op=True)
        barrier.block_current_stream()
        for start, end in events:
            start.record()
            call()
            end.record()
        torch.cuda.synchronize()
        dist.barrier()
        return rank_max_samples([start.elapsed_time(end) for start, end in events], device)

    schedule_samples = measure(build_schedule)
    cached_schedule = build_schedule()
    torch.cuda.synchronize()
    forward_samples = measure(lambda: run_forward(cached_schedule))
    combined_samples = measure(run_combined)
    measured_end = time.time()

    if rank == 0:
        schedule_p50, schedule_p95 = percentiles(schedule_samples)
        forward_p50, forward_p95 = percentiles(forward_samples)
        combined_p50, combined_p95 = percentiles(combined_samples)
        isolated_sum = schedule_p50 + forward_p50
        record = {
            "schema": "bench-sm90-fwd-phases.v1",
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
                "warmup_iters": WARMUP,
                "timed_iters_per_series": TIMED_ITERS,
                "series_order": ["schedule_only", "forward_only_cached_schedule", "combined"],
                "semantics": "diagnostic only; combined must reproduce frozen harness; "
                             "isolated medians are attributable only when their sum closes",
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
                    "three_measured_series": round(measured_end - warmup_end, 1),
                    "total": round(measured_end - wall0, 1),
                },
                "loadavg_end": _loadavg(),
                "gpu_snapshot_end": gpu_snapshot(),
                "provenance": provenance(),
            },
            "schedule_only": {
                "samples_ms": [round(value, 4) for value in schedule_samples],
                "p50_ms": round(schedule_p50, 4),
                "p95_ms": round(schedule_p95, 4),
            },
            "forward_only_cached_schedule": {
                "samples_ms": [round(value, 4) for value in forward_samples],
                "p50_ms": round(forward_p50, 4),
                "p95_ms": round(forward_p95, 4),
            },
            "combined": {
                "samples_ms": [round(value, 4) for value in combined_samples],
                "p50_ms": round(combined_p50, 4),
                "p95_ms": round(combined_p95, 4),
            },
            "closure": {
                "isolated_p50_sum_ms": round(isolated_sum, 4),
                "combined_minus_isolated_sum_ms": round(combined_p50 - isolated_sum, 4),
                "combined_minus_isolated_sum_pct": round(
                    (combined_p50 - isolated_sum) / combined_p50 * 100, 4),
            },
        }
        temporary = OUTPUT + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(record, stream, indent=1)
        os.replace(temporary, OUTPUT)
        print(
            f"PHASES|comm_sms={COMM_SMS}|schedule_p50={schedule_p50:.4f}ms"
            f"|forward_p50={forward_p50:.4f}ms|combined_p50={combined_p50:.4f}ms"
            f"|closure_delta={combined_p50 - isolated_sum:.4f}ms"
            f"|abs_max={abs_max:.5f}|relative={relative:.6f}|out={OUTPUT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
