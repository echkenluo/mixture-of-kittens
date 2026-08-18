#!/usr/bin/env python3
"""Measure double-barrier versus single-boundary cluster worker loops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="1,8,32,64")
    parser.add_argument("--tasks", default="1,8,64,256")
    parser.add_argument("--work", default="0,64")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def build_extension():
    return load(
        name="mok_cluster_boundary_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def elapsed_ms(module, descriptor, output, tasks, work, single, warmup, repeats):
    for _ in range(warmup):
        module.run(descriptor, output, tasks, work, single)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        module.run(descriptor, output, tasks, work, single)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(0)
    module = build_extension()
    rows = []
    for clusters in csv_ints(args.clusters):
        descriptor = torch.zeros(clusters, dtype=torch.int32, device="cuda")
        output = torch.empty(clusters * 2 * 128, dtype=torch.int32, device="cuda")
        for tasks in csv_ints(args.tasks):
            for work in csv_ints(args.work):
                module.run(descriptor, output, tasks, work, False)
                double_output = output.clone()
                module.run(descriptor, output, tasks, work, True)
                torch.cuda.synchronize()
                if not torch.equal(double_output, output):
                    raise RuntimeError(
                        f"output mismatch clusters={clusters} tasks={tasks} work={work}"
                    )
                double_ms = elapsed_ms(
                    module, descriptor, output, tasks, work, False,
                    args.warmup, args.repeats,
                )
                single_ms = elapsed_ms(
                    module, descriptor, output, tasks, work, True,
                    args.warmup, args.repeats,
                )
                row = {
                    "clusters": clusters,
                    "tasks": tasks,
                    "work_iterations": work,
                    "double_ms": double_ms,
                    "single_ms": single_ms,
                    "delta_pct": (single_ms / double_ms - 1.0) * 100.0,
                }
                rows.append(row)
                print("CLUSTER_BOUNDARY|" + "|".join(
                    f"{key}={value:.6f}" if isinstance(value, float)
                    else f"{key}={value}"
                    for key, value in row.items()
                ), flush=True)
    print("CLUSTER_BOUNDARY_JSON=" + json.dumps(rows, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
