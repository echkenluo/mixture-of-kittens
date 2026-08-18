#!/usr/bin/env python3
"""Stress and time one-pull-task-per-M64 terminal scheduling."""

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
    parser.add_argument("--clusters", default="1,58,87,117")
    parser.add_argument("--m-tiles", default="0,1,64,384")
    parser.add_argument("--delay-cycles", default="0,20000")
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def build_extension():
    return load(
        name="mok_tile_pipeline_scheduler_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def zeros(size: int) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.int32, device="cuda")


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(0)
    module = build_extension()
    occupancy = int(module.max_active_clusters(0))
    attributes = [int(value) for value in module.kernel_attributes(0)]
    clusters_values = csv_ints(args.clusters)
    if max(clusters_values) > occupancy:
        raise RuntimeError(
            "requested grid is not fully resident: "
            f"requested={max(clusters_values)} occupancy={occupancy}"
        )
    print(
        "TILE_PIPELINE_RESOURCES"
        f"|max_active_clusters={occupancy}|registers={attributes[0]}"
        f"|static_smem={attributes[1]}|local_bytes={attributes[2]}"
        f"|max_threads={attributes[3]}|binary_version={attributes[4]}"
        f"|ptx_version={attributes[5]}",
        flush=True,
    )

    rows = []
    for clusters in clusters_values:
        descriptor = zeros(clusters * 2)
        for m_tiles in csv_ints(args.m_tiles):
            visits = [zeros(m_tiles) for _ in range(5)]
            head, terminal, mismatch = zeros(1), zeros(1), zeros(1)
            tensors = (head, terminal, descriptor, *visits, mismatch)
            for delay_cycles in csv_ints(args.delay_cycles):
                elapsed_ms = []
                digest = hashlib.sha256()
                for _repeat in range(args.repeats):
                    for tensor in tensors:
                        tensor.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    module.run(*tensors, delay_cycles)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_ms.append(start.elapsed_time(end))
                    if int(head.item()) != m_tiles:
                        raise RuntimeError("tile head did not close")
                    if int(terminal.item()) != m_tiles:
                        raise RuntimeError("terminal count did not close")
                    if int(mismatch.item()) != 0:
                        raise RuntimeError("scheduler mismatch counter nonzero")
                    for stage, tensor in zip(
                        ("copy", "w13", "act", "w2", "push"), visits
                    ):
                        if m_tiles and (
                            int(tensor.min().item()) != 2
                            or int(tensor.max().item()) != 2
                        ):
                            raise RuntimeError(
                                f"{stage} paired-CTA visits not exactly two"
                            )
                        digest.update(tensor.cpu().numpy().tobytes())
                ordered = sorted(elapsed_ms)
                row = {
                    "clusters": clusters,
                    "m_tiles": m_tiles,
                    "delay_cycles": delay_cycles,
                    "repeats": args.repeats,
                    "p50_ms": round(statistics.median(ordered), 6),
                    "p95_ms": round(
                        ordered[max(0, int(len(ordered) * 0.95) - 1)], 6
                    ),
                    "sha16": digest.hexdigest()[:16],
                }
                rows.append(row)
                print(
                    "TILE_PIPELINE_PASS|"
                    + "|".join(f"{key}={value}" for key, value in row.items()),
                    flush=True,
                )
    print("TILE_PIPELINE_JSON=" + json.dumps(rows, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
