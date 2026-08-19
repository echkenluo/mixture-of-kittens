#!/usr/bin/env python3
"""Stress the full ready-only terminal scheduler dependency chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="1,8,64,117")
    parser.add_argument("--m-tiles", default="1,16,64")
    parser.add_argument("--act-per-m", default="8,64")
    parser.add_argument("--delay-cycles", default="0,20000")
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def build_extension():
    return load(
        name="mok_terminal_scheduler_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        verbose=False,
    )


def zeros(size: int) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.int32, device="cuda")


def assert_all_one(name: str, tensor: torch.Tensor) -> None:
    lo = int(tensor.min().item())
    hi = int(tensor.max().item())
    if lo != 1 or hi != 1:
        raise RuntimeError(f"{name} visits not exactly once: [{lo},{hi}]")


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(0)
    module = build_extension()
    max_active_clusters = int(module.max_active_clusters(0))
    attributes = [int(value) for value in module.kernel_attributes(0)]
    requested_clusters = csv_ints(args.clusters)
    if max(requested_clusters) > max_active_clusters:
        raise RuntimeError(
            "requested cluster grid is not fully resident: "
            f"requested={max(requested_clusters)} occupancy={max_active_clusters}"
        )
    print(
        "TERMINAL_SCHEDULER_RESOURCES"
        f"|max_active_clusters={max_active_clusters}"
        f"|registers={attributes[0]}|static_smem={attributes[1]}"
        f"|local_bytes={attributes[2]}|max_threads={attributes[3]}"
        f"|binary_version={attributes[4]}|ptx_version={attributes[5]}",
        flush=True,
    )
    w13_per_m = 32
    w2_per_m = 32
    rows = []
    for clusters in requested_clusters:
        worker_descriptor = zeros(clusters * 2)
        for m_tiles in csv_ints(args.m_tiles):
            for act_per_m in csv_ints(args.act_per_m):
                copy_head = zeros(1)
                terminal_count = zeros(1)
                w13_state, act_state, w2_state = zeros(3), zeros(3), zeros(3)
                w13_descriptor = zeros(m_tiles * w13_per_m)
                w13_commit = torch.zeros_like(w13_descriptor)
                act_descriptor = zeros(m_tiles * act_per_m)
                act_commit = torch.zeros_like(act_descriptor)
                w2_descriptor = zeros(m_tiles * w2_per_m)
                w2_commit = torch.zeros_like(w2_descriptor)
                reduce_ready = zeros(m_tiles)
                copy_visits = zeros(m_tiles)
                w13_visits = torch.zeros_like(w13_descriptor)
                act_visits = torch.zeros_like(act_descriptor)
                w2_visits = torch.zeros_like(w2_descriptor)
                reduce_visits = zeros(m_tiles)
                w13_done, act_done, w2_done = (
                    zeros(m_tiles), zeros(m_tiles), zeros(m_tiles)
                )
                mismatch = zeros(1)
                tensors = (
                    copy_head, terminal_count, w13_state, w13_descriptor,
                    w13_commit, act_state, act_descriptor, act_commit,
                    w2_state, w2_descriptor, w2_commit, reduce_ready,
                    copy_visits, w13_visits, act_visits, w2_visits,
                    reduce_visits, w13_done, act_done, w2_done,
                    worker_descriptor, mismatch,
                )
                for delay_cycles in csv_ints(args.delay_cycles):
                    digest = hashlib.sha256()
                    elapsed_ms = []
                    for repeat in range(args.repeats):
                        for tensor in tensors:
                            tensor.zero_()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        module.run(
                            *tensors, w13_per_m, act_per_m, w2_per_m,
                            delay_cycles,
                        )
                        end.record()
                        torch.cuda.synchronize()
                        elapsed_ms.append(start.elapsed_time(end))
                        if int(copy_head.item()) != m_tiles:
                            raise RuntimeError("copy head did not close")
                        if int(terminal_count.item()) != m_tiles:
                            raise RuntimeError("terminal count did not close")
                        if int(mismatch.item()) != 0:
                            raise RuntimeError("scheduler mismatch counter nonzero")
                        for name, state, expected in (
                            ("w13", w13_state, m_tiles * w13_per_m),
                            ("act", act_state, m_tiles * act_per_m),
                            ("w2", w2_state, m_tiles * w2_per_m),
                        ):
                            values = state.cpu().tolist()
                            if values != [expected, expected, expected]:
                                raise RuntimeError(
                                    f"{name} queue did not close: {values}"
                                )
                        for name, visits in (
                            ("copy", copy_visits), ("w13", w13_visits),
                            ("act", act_visits), ("w2", w2_visits),
                            ("reduce", reduce_visits),
                        ):
                            assert_all_one(name, visits)
                        if (
                            int(w13_done.min().item()) != w13_per_m
                            or int(w13_done.max().item()) != w13_per_m
                            or int(act_done.min().item()) != act_per_m
                            or int(act_done.max().item()) != act_per_m
                            or int(w2_done.min().item()) != w2_per_m
                            or int(w2_done.max().item()) != w2_per_m
                        ):
                            raise RuntimeError("per-M completion counter mismatch")
                        digest.update(copy_visits.cpu().numpy().tobytes())
                        digest.update(w13_visits.cpu().numpy().tobytes())
                        digest.update(act_visits.cpu().numpy().tobytes())
                        digest.update(w2_visits.cpu().numpy().tobytes())
                        digest.update(reduce_visits.cpu().numpy().tobytes())
                    row = {
                        "clusters": clusters,
                        "m_tiles": m_tiles,
                        "act_per_m": act_per_m,
                        "delay_cycles": delay_cycles,
                        "repeats": args.repeats,
                        "p50_ms": round(statistics.median(elapsed_ms), 6),
                        "p95_ms": round(
                            sorted(elapsed_ms)[
                                max(0, int(len(elapsed_ms) * 0.95) - 1)
                            ],
                            6,
                        ),
                        "sha16": digest.hexdigest()[:16],
                    }
                    rows.append(row)
                    print(
                        "TERMINAL_SCHEDULER_PASS|" + "|".join(
                            f"{key}={value}" for key, value in row.items()
                        ),
                        flush=True,
                    )
    print("TERMINAL_SCHEDULER_JSON=" + json.dumps(rows, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
