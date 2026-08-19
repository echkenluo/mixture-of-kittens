#!/usr/bin/env python3
"""Compare held-chain and coarse-M64 stage-queue scheduling.

This is a scheduler-only probe.  Its REDUCE stage is a ready-task surrogate;
cross-rank payload visibility and production numerics remain separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.cpp_extension import load


STAGES = ("dispatch", "w13", "act", "w2", "push", "reduce")
NSTAGE = len(STAGES)
QUEUED_STAGES = NSTAGE - 1
MODE_HELD = 0
MODE_QUEUED = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="1,58,117")
    parser.add_argument(
        "--m-tiles",
        default="auto",
        help="comma-separated values, or auto for boundary cells per grid",
    )
    parser.add_argument("--profiles", default="ZERO,V4_RATIO,SKEW")
    parser.add_argument("--window-multiple", type=int, default=2)
    parser.add_argument("--cycle-quantum", type=int, default=128)
    parser.add_argument("--skew-cycles", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--require-pipeline-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return float(ordered[index])


def build_extension():
    return load(
        name="mok_coarse_stage_pipeline_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        verbose=False,
    )


def zeros(size: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return torch.zeros(size, dtype=dtype, device="cuda")


@dataclass
class Workspace:
    source_head: torch.Tensor
    inflight: torch.Tensor
    terminal_count: torch.Tensor
    queue_state: torch.Tensor
    queue_descriptor: torch.Tensor
    queue_commit: torch.Tensor
    tile_state: torch.Tensor
    leader_visits: torch.Tensor
    paired_visits: torch.Tensor
    owner_cluster: torch.Tensor
    ready_ns: torch.Tensor
    start_ns: torch.Tensor
    end_ns: torch.Tensor
    active_stage: torch.Tensor
    max_active_stage: torch.Tensor
    overlap_mask: torch.Tensor
    worker_descriptor: torch.Tensor
    mismatch: torch.Tensor
    cycle_sink: torch.Tensor

    @classmethod
    def create(cls, clusters: int, m_tiles: int) -> "Workspace":
        stage_items = NSTAGE * m_tiles
        return cls(
            source_head=zeros(1),
            inflight=zeros(1),
            terminal_count=zeros(1),
            queue_state=zeros(QUEUED_STAGES * 3),
            queue_descriptor=zeros(QUEUED_STAGES * m_tiles),
            queue_commit=zeros(QUEUED_STAGES * m_tiles),
            tile_state=zeros(m_tiles),
            leader_visits=zeros(stage_items),
            paired_visits=zeros(stage_items),
            owner_cluster=zeros(stage_items),
            ready_ns=zeros(stage_items, torch.int64),
            start_ns=zeros(stage_items, torch.int64),
            end_ns=zeros(stage_items, torch.int64),
            active_stage=zeros(NSTAGE),
            max_active_stage=zeros(NSTAGE),
            overlap_mask=zeros(1, torch.int64),
            worker_descriptor=zeros(clusters * 2),
            mismatch=zeros(1),
            cycle_sink=zeros(1),
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, field) for field in self.__dataclass_fields__)

    def reset(self) -> None:
        for tensor in self.tensors():
            tensor.zero_()


def profile_config(
    profile: str, cycle_quantum: int, skew_cycles: int
) -> tuple[list[int], int]:
    if profile == "ZERO":
        return [0] * NSTAGE, 0
    if profile not in {"V4_RATIO", "SKEW"}:
        raise ValueError(f"unknown profile: {profile}")
    # Synthetic ratios only.  They preserve the broad split-profile shape but
    # are not production kernel timings.
    ratios = [4, 40, 1, 35, 7, 3]
    cycles = [ratio * cycle_quantum for ratio in ratios]
    return cycles, skew_cycles if profile == "SKEW" else 0


def m_tiles_for_grid(argument: str, clusters: int) -> list[int]:
    if argument != "auto":
        values = csv_ints(argument)
    else:
        values = [
            0,
            1,
            max(1, clusters - 1),
            clusters,
            clusters + 1,
            clusters * 2,
            384,
        ]
    return sorted(set(values))


def interval_metrics(
    owners: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> dict[str, float | int]:
    m_tiles = starts.shape[1]
    if m_tiles == 0:
        return {
            "handoffs": 0,
            "handoff_rate": 0.0,
            "max_stage_depth": 0,
            "overlap_ratio": 0.0,
            "steady_mtiles_per_ms": 0.0,
            "steady_rows_per_s": 0.0,
            "terminal_gap_p50_ns": 0.0,
            "terminal_gap_p95_ns": 0.0,
        }

    handoffs = int((owners[1:] != owners[:-1]).sum().item())
    handoff_denominator = (NSTAGE - 1) * m_tiles
    events: list[tuple[int, int, int]] = []
    for stage in range(NSTAGE):
        for tile in range(m_tiles):
            start = int(starts[stage, tile].item())
            end = int(ends[stage, tile].item())
            events.append((start, 1, stage))
            events.append((end, -1, stage))
    events.sort(key=lambda item: item[0])

    active = [0] * NSTAGE
    previous = events[0][0]
    busy_ns = 0
    overlap_ns = 0
    max_depth = 0
    position = 0
    while position < len(events):
        timestamp = events[position][0]
        depth = sum(count > 0 for count in active)
        span = timestamp - previous
        if depth:
            busy_ns += span
        if depth >= 2:
            overlap_ns += span
        max_depth = max(max_depth, depth)
        while position < len(events) and events[position][0] == timestamp:
            _, delta, stage = events[position]
            active[stage] += delta
            if active[stage] < 0:
                raise RuntimeError("negative active-stage count in trace")
            position += 1
        max_depth = max(max_depth, sum(count > 0 for count in active))
        previous = timestamp

    terminal = sorted(int(value) for value in ends[-1].tolist())
    gaps = [right - left for left, right in zip(terminal, terminal[1:])]
    steady_mtiles_per_ms = 0.0
    if len(terminal) >= 10:
        lo = int((len(terminal) - 1) * 0.10)
        hi = int((len(terminal) - 1) * 0.90)
        steady_ns = terminal[hi] - terminal[lo]
        if steady_ns > 0 and hi > lo:
            steady_mtiles_per_ms = (hi - lo) * 1.0e6 / steady_ns
    return {
        "handoffs": handoffs,
        "handoff_rate": handoffs / handoff_denominator,
        "max_stage_depth": max_depth,
        "overlap_ratio": overlap_ns / busy_ns if busy_ns else 0.0,
        "steady_mtiles_per_ms": steady_mtiles_per_ms,
        "steady_rows_per_s": steady_mtiles_per_ms * 64.0 * 1000.0,
        "terminal_gap_p50_ns": statistics.median(gaps) if gaps else 0.0,
        "terminal_gap_p95_ns": percentile(gaps, 0.95) if gaps else 0.0,
    }


def validate_and_measure(
    workspace: Workspace, mode: int, m_tiles: int
) -> dict[str, float | int]:
    if int(workspace.source_head.item()) != m_tiles:
        raise RuntimeError("source head did not close")
    if int(workspace.inflight.item()) != 0:
        raise RuntimeError("inflight tile count did not close")
    if int(workspace.terminal_count.item()) != m_tiles:
        raise RuntimeError("terminal count did not close")
    if int(workspace.mismatch.item()) != 0:
        raise RuntimeError("scheduler mismatch counter is nonzero")
    if int(workspace.active_stage.abs().sum().item()) != 0:
        raise RuntimeError("active-stage counters did not close")

    queue_state = workspace.queue_state.reshape(QUEUED_STAGES, 3).cpu()
    queue_expected = m_tiles if mode == MODE_QUEUED else 0
    if not bool((queue_state == queue_expected).all().item()):
        raise RuntimeError(
            f"queue state did not close: expected={queue_expected}, "
            f"observed={queue_state.tolist()}"
        )

    if m_tiles == 0:
        return interval_metrics(
            torch.empty((NSTAGE, 0), dtype=torch.int32),
            torch.empty((NSTAGE, 0), dtype=torch.int64),
            torch.empty((NSTAGE, 0), dtype=torch.int64),
        ) | {
            "overlap_pairs": 0,
            "adjacent_overlap_pairs": 0,
            "queue_wait_p50_ns": 0.0,
            "queue_wait_p95_ns": 0.0,
        }

    tile_state = workspace.tile_state.cpu()
    if int(tile_state.min().item()) != NSTAGE or int(tile_state.max().item()) != NSTAGE:
        raise RuntimeError("tile state did not reach the terminal stage")
    leaders = workspace.leader_visits.reshape(NSTAGE, m_tiles).cpu()
    paired = workspace.paired_visits.reshape(NSTAGE, m_tiles).cpu()
    owners = workspace.owner_cluster.reshape(NSTAGE, m_tiles).cpu()
    if int(leaders.min().item()) != 1 or int(leaders.max().item()) != 1:
        raise RuntimeError("leader visits are not exactly once")
    if int(paired.min().item()) != 2 or int(paired.max().item()) != 2:
        raise RuntimeError("paired CTA visits are not exactly twice")
    if int(owners.min().item()) <= 0:
        raise RuntimeError("owner record is incomplete")

    ready = workspace.ready_ns.reshape(NSTAGE, m_tiles).cpu()
    starts = workspace.start_ns.reshape(NSTAGE, m_tiles).cpu()
    ends = workspace.end_ns.reshape(NSTAGE, m_tiles).cpu()
    if bool((ready <= 0).any().item()) or bool((starts <= 0).any().item()):
        raise RuntimeError("ready/start timestamps are incomplete")
    if bool((ends <= 0).any().item()):
        raise RuntimeError("end timestamps are incomplete")
    if bool((starts < ready).any().item()) or bool((ends < starts).any().item()):
        raise RuntimeError("timestamp order is invalid")

    metrics = interval_metrics(owners, starts, ends)
    if mode == MODE_HELD and int(metrics["handoffs"]) != 0:
        raise RuntimeError("held mode changed owner inside an M64 chain")
    waits = (starts - ready).flatten().tolist()
    mask = int(workspace.overlap_mask.item())
    adjacent_pairs = sum(
        bool(mask & (1 << (stage * NSTAGE + stage + 1)))
        for stage in range(NSTAGE - 1)
    )
    return metrics | {
        "overlap_pairs": mask.bit_count() // 2,
        "adjacent_overlap_pairs": adjacent_pairs,
        "queue_wait_p50_ns": statistics.median(waits),
        "queue_wait_p95_ns": percentile(waits, 0.95),
    }


def run_once(
    module,
    workspace: Workspace,
    *,
    mode: int,
    m_tiles: int,
    cycle_values: list[int],
    skew_cycles: int,
    window: int,
) -> tuple[float, dict[str, float | int], str]:
    workspace.reset()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    module.run(
        *workspace.tensors(), cycle_values, skew_cycles, window, mode
    )
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    metrics = validate_and_measure(workspace, mode, m_tiles)
    digest = hashlib.sha256()
    digest.update(workspace.leader_visits.cpu().numpy().tobytes())
    digest.update(workspace.paired_visits.cpu().numpy().tobytes())
    digest.update(workspace.owner_cluster.cpu().numpy().tobytes())
    return elapsed_ms, metrics, digest.hexdigest()[:16]


def run_leg(
    module,
    *,
    mode: int,
    clusters: int,
    m_tiles: int,
    cycle_values: list[int],
    skew_cycles: int,
    window: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    workspace = Workspace.create(clusters, m_tiles)
    for _ in range(warmup):
        run_once(
            module,
            workspace,
            mode=mode,
            m_tiles=m_tiles,
            cycle_values=cycle_values,
            skew_cycles=skew_cycles,
            window=window,
        )

    times: list[float] = []
    metrics_rows: list[dict[str, float | int]] = []
    digest = hashlib.sha256()
    for _ in range(repeats):
        elapsed, metrics, run_digest = run_once(
            module,
            workspace,
            mode=mode,
            m_tiles=m_tiles,
            cycle_values=cycle_values,
            skew_cycles=skew_cycles,
            window=window,
        )
        times.append(elapsed)
        metrics_rows.append(metrics)
        digest.update(run_digest.encode())

    def metric_median(key: str) -> float:
        return float(statistics.median(float(row[key]) for row in metrics_rows))

    return {
        "p50_ms": round(statistics.median(times), 6),
        "p95_ms": round(percentile(times, 0.95), 6),
        "handoffs": int(min(int(row["handoffs"]) for row in metrics_rows)),
        "handoff_rate": round(metric_median("handoff_rate"), 6),
        "max_stage_depth": int(
            min(int(row["max_stage_depth"]) for row in metrics_rows)
        ),
        "overlap_ratio": round(metric_median("overlap_ratio"), 6),
        "overlap_pairs": int(min(int(row["overlap_pairs"]) for row in metrics_rows)),
        "adjacent_overlap_pairs": int(
            min(int(row["adjacent_overlap_pairs"]) for row in metrics_rows)
        ),
        "steady_mtiles_per_ms": round(
            metric_median("steady_mtiles_per_ms"), 6
        ),
        "steady_rows_per_s": round(metric_median("steady_rows_per_s"), 3),
        "terminal_gap_p50_ns": round(
            metric_median("terminal_gap_p50_ns"), 3
        ),
        "terminal_gap_p95_ns": round(
            metric_median("terminal_gap_p95_ns"), 3
        ),
        "queue_wait_p50_ns": round(metric_median("queue_wait_p50_ns"), 3),
        "queue_wait_p95_ns": round(metric_median("queue_wait_p95_ns"), 3),
        "sha16": digest.hexdigest()[:16],
    }


def main() -> int:
    args = parse_args()
    if args.window_multiple <= 0 or args.cycle_quantum < 0:
        raise ValueError("window multiple and cycle quantum must be valid")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup/repeats must be valid")

    torch.cuda.set_device(0)
    module = build_extension()
    occupancy = {
        "held": int(module.max_active_clusters(0, MODE_HELD)),
        "queued": int(module.max_active_clusters(0, MODE_QUEUED)),
    }
    attributes = {
        "held": [int(value) for value in module.kernel_attributes(0, MODE_HELD)],
        "queued": [
            int(value) for value in module.kernel_attributes(0, MODE_QUEUED)
        ],
    }
    clusters_values = csv_ints(args.clusters)
    usable_occupancy = min(occupancy.values())
    if max(clusters_values) > usable_occupancy:
        raise RuntimeError(
            f"requested {max(clusters_values)} clusters exceeds common "
            f"occupancy {usable_occupancy}"
        )
    print(
        "COARSE_STAGE_RESOURCES="
        + json.dumps(
            {"occupancy": occupancy, "attributes": attributes}, sort_keys=True
        ),
        flush=True,
    )

    profiles = [item for item in args.profiles.split(",") if item]
    rows = []
    for clusters in clusters_values:
        for m_tiles in m_tiles_for_grid(args.m_tiles, clusters):
            if m_tiles < 0:
                raise ValueError("M64 tile count must be nonnegative")
            window = max(
                1, min(max(1, m_tiles), clusters * args.window_multiple)
            )
            for profile in profiles:
                cycle_values, skew_cycles = profile_config(
                    profile, args.cycle_quantum, args.skew_cycles
                )
                held_a = run_leg(
                    module,
                    mode=MODE_HELD,
                    clusters=clusters,
                    m_tiles=m_tiles,
                    cycle_values=cycle_values,
                    skew_cycles=skew_cycles,
                    window=window,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                queued = run_leg(
                    module,
                    mode=MODE_QUEUED,
                    clusters=clusters,
                    m_tiles=m_tiles,
                    cycle_values=cycle_values,
                    skew_cycles=skew_cycles,
                    window=window,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                held_b = run_leg(
                    module,
                    mode=MODE_HELD,
                    clusters=clusters,
                    m_tiles=m_tiles,
                    cycle_values=cycle_values,
                    skew_cycles=skew_cycles,
                    window=window,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )

                held_mid = (float(held_a["p50_ms"]) + float(held_b["p50_ms"])) / 2
                aa_drift = (
                    abs(float(held_a["p50_ms"]) - float(held_b["p50_ms"]))
                    / held_mid
                    if held_mid
                    else 0.0
                )
                queued_delta = (
                    (float(queued["p50_ms"]) - held_mid) / held_mid
                    if held_mid
                    else 0.0
                )
                evidence_required = (
                    args.require_pipeline_evidence
                    and profile != "ZERO"
                    and clusters > 1
                    and m_tiles >= 2 * clusters
                )
                pipeline_evidence = (
                    int(queued["max_stage_depth"]) >= 2
                    and int(queued["handoffs"]) > 0
                    and int(queued["adjacent_overlap_pairs"]) > 0
                )
                if evidence_required and not pipeline_evidence:
                    raise RuntimeError(
                        "queued mode failed pipeline evidence gate: "
                        f"clusters={clusters} m_tiles={m_tiles} "
                        f"profile={profile} metrics={queued}"
                    )

                row = {
                    "clusters": clusters,
                    "m_tiles": m_tiles,
                    "global_tasks": m_tiles * NSTAGE,
                    "profile": profile,
                    "cycle_values": cycle_values,
                    "skew_cycles": skew_cycles,
                    "window": window,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "held_a": held_a,
                    "queued": queued,
                    "held_b": held_b,
                    "aa_drift": round(aa_drift, 6),
                    "queued_latency_delta": round(queued_delta, 6),
                    "pipeline_evidence_required": evidence_required,
                    "pipeline_evidence": pipeline_evidence,
                }
                rows.append(row)
                print(
                    "COARSE_STAGE_CELL=" + json.dumps(row, sort_keys=True),
                    flush=True,
                )

    print(
        "COARSE_STAGE_PIPELINE_PROBE="
        + json.dumps(
            {
                "cells": len(rows),
                "modes": ["HELD", "QUEUED", "HELD"],
                "result": "PASS",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
