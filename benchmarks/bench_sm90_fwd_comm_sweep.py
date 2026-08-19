"""Canary-only comm-SM sweep for the SM90 BF16 forward path.

The process creates one fixed input/reference pair and one cached workspace,
then varies only ``fwd_num_comm_sms``.  Every candidate passes the BF16
correctness gate before timing.  Two timed passes use forward and reverse
orders to reduce order/temperature bias; each series stores raw rank-max
samples.  This diagnostic does not replace the frozen deployment harness.
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


OUTPUT = os.environ.get("BENCH_COMM_SWEEP_OUTPUT", "/mok/bench-sm90-fwd-comm-sweep.json")
COMM_SMS_VALUES = tuple(
    int(value.strip())
    for value in os.environ.get("BENCH_COMM_SMS_LIST", "2,4,8,16").split(",")
    if value.strip()
)


def percentiles(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[max(0, int(len(ordered) * 0.95) - 1)]


def provenance() -> dict[str, object]:
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


def safe_gpu_snapshot() -> dict[str, object]:
    try:
        return gpu_snapshot()
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def validate_sweep_values() -> None:
    if not COMM_SMS_VALUES:
        raise ValueError("BENCH_COMM_SMS_LIST must contain at least one value")
    if len(set(COMM_SMS_VALUES)) != len(COMM_SMS_VALUES):
        raise ValueError("BENCH_COMM_SMS_LIST must not contain duplicates")
    if any(value <= 0 or value % 2 for value in COMM_SMS_VALUES):
        raise ValueError("every comm-SM value must be a positive even integer")


def main() -> None:
    validate_sweep_values()
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

    configs = {
        comm_sms: functional.MoKConfig(
            fwd_num_comm_sms=comm_sms,
            bwd_num_comm_sms=comm_sms,
            minibatch_size=MINIBATCH_SIZE,
            macrobatch_size=MACROBATCH_SIZE,
        )
        for comm_sms in COMM_SMS_VALUES
    }
    workspace = functional.get_workspace(
        configs[COMM_SMS_VALUES[0]],
        dist.group.WORLD,
        device=x.device,
        num_local_tokens=x.shape[0],
        hidden_size=x.shape[1],
        topk=topk_experts.shape[1],
    )

    def run_forward(comm_sms: int):
        config = configs[comm_sms]
        schedule = functional.build_schedule(
            workspace, config, topk_experts, num_local_experts=num_local_experts
        )
        output, _context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
        )
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
    ref_out = run_fwd_epilogue_reference(ref_y_shared, ref_combine_buffer, router_weights)
    del ref_combine_buffer, _rg, _ru, _rh, ref_y_shared

    correctness: dict[int, dict[str, float]] = {}
    absolute_tolerance, relative_tolerance = BF16_TOLERANCE
    for comm_sms in COMM_SMS_VALUES:
        out = run_forward(comm_sms)
        torch.cuda.synchronize()
        abs_mean, abs_max, relative = get_error_stats(ref_out, out)
        gate_pass = (
            all(math.isfinite(value) for value in (abs_mean, abs_max, relative))
            and abs_max <= absolute_tolerance
            and relative <= relative_tolerance
        )
        if not gate_pass:
            raise RuntimeError(
                f"correctness gate FAILED on rank {rank}, comm_sms={comm_sms}: "
                f"abs_mean={abs_mean:.6f} abs_max={abs_max:.6f} "
                f"relative={relative:.6f} vs tolerance "
                f"(abs={absolute_tolerance}, rel={relative_tolerance})"
            )
        correctness[comm_sms] = {
            "abs_mean": abs_mean,
            "abs_max": abs_max,
            "relative": relative,
        }
        del out
    del ref_out
    correctness_end = time.time()

    for comm_sms in COMM_SMS_VALUES:
        for _ in range(WARMUP):
            run_forward(comm_sms)
    torch.cuda.synchronize()
    warmup_end = time.time()

    def measure(comm_sms: int) -> list[float]:
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(TIMED_ITERS)
        ]
        barrier = dist.barrier(async_op=True)
        barrier.block_current_stream()
        for start, end in events:
            start.record()
            run_forward(comm_sms)
            end.record()
        torch.cuda.synchronize()
        dist.barrier()
        return rank_max_samples(
            [start.elapsed_time(end) for start, end in events], device
        )

    series: dict[int, list[list[float]]] = {value: [] for value in COMM_SMS_VALUES}
    pass_orders = (COMM_SMS_VALUES, tuple(reversed(COMM_SMS_VALUES)))
    for order in pass_orders:
        for comm_sms in order:
            series[comm_sms].append(measure(comm_sms))
    measured_end = time.time()

    if rank == 0:
        results: dict[str, object] = {}
        for comm_sms in COMM_SMS_VALUES:
            flattened = [value for values in series[comm_sms] for value in values]
            p50, p95 = percentiles(flattened)
            results[str(comm_sms)] = {
                "correctness_gate": {
                    **correctness[comm_sms],
                    "tolerance_abs_rel": list(BF16_TOLERANCE),
                },
                "series_order_positions": [
                    list(order).index(comm_sms) for order in pass_orders
                ],
                "series_samples_ms": [
                    [round(value, 4) for value in values]
                    for values in series[comm_sms]
                ],
                "series_p50_ms": [
                    round(percentiles(values)[0], 4) for values in series[comm_sms]
                ],
                "aggregate_p50_ms": round(p50, 4),
                "aggregate_p95_ms": round(p95, 4),
            }
        record = {
            "schema": "bench-sm90-fwd-comm-sweep.v1",
            "meta": {
                "shape": {
                    "tokens_per_rank": NUM_LOCAL_TOKENS,
                    "hidden": HIDDEN_DIM,
                    "intermediate": INTERMEDIATE_DIM,
                    "experts": NUM_EXPERTS,
                    "topk": TOPK,
                    "world_size": world_size,
                },
                "comm_sms_values": list(COMM_SMS_VALUES),
                "pass_orders": [list(order) for order in pass_orders],
                "minibatch": MINIBATCH_SIZE,
                "macrobatch": MACROBATCH_SIZE,
                "warmup_iters_per_value": WARMUP,
                "timed_iters_per_series": TIMED_ITERS,
                "semantics": (
                    "diagnostic only; same fixed input, reference, and cached workspace; "
                    "combined build_schedule+forward; two counter-ordered rank-max series"
                ),
                "wall_phases_s": {
                    "setup": round(setup_end - wall0, 1),
                    "correctness_gates": round(correctness_end - setup_end, 1),
                    "warmup": round(warmup_end - correctness_end, 1),
                    "measured": round(measured_end - warmup_end, 1),
                    "total": round(measured_end - wall0, 1),
                },
                "loadavg_end": _loadavg(),
                "gpu_snapshot_end": safe_gpu_snapshot(),
                "provenance": provenance(),
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)
        with open(OUTPUT, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        best = min(
            COMM_SMS_VALUES,
            key=lambda value: results[str(value)]["aggregate_p50_ms"],
        )
        print(
            "COMM_SWEEP|"
            + "|".join(
                f"sm{value}={results[str(value)]['aggregate_p50_ms']:.4f}ms"
                for value in COMM_SMS_VALUES
            )
            + f"|best=sm{best}|out={OUTPUT}",
            flush=True,
        )


if __name__ == "__main__":
    main()
