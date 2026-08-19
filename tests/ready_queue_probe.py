#!/usr/bin/env python3
"""Stress the terminal design's non-wrapping MPMC ready queue protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="1,8,64,117")
    parser.add_argument("--items", default="1024,8192")
    parser.add_argument("--delay-cycles", default="0,20000")
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def build_extension():
    return load(
        name="mok_ready_queue_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(0)
    module = build_extension()
    rows = []
    for clusters in csv_ints(args.clusters):
        worker_descriptor = torch.zeros(
            clusters * 2, dtype=torch.int32, device="cuda"
        )
        for items in csv_ints(args.items):
            state = torch.zeros(3, dtype=torch.int32, device="cuda")
            descriptor = torch.zeros(items, dtype=torch.int32, device="cuda")
            commit = torch.zeros_like(descriptor)
            visits = torch.zeros_like(descriptor)
            mismatch = torch.zeros(1, dtype=torch.int32, device="cuda")
            for delay_cycles in csv_ints(args.delay_cycles):
                digest = hashlib.sha256()
                for repeat in range(args.repeats):
                    state.zero_()
                    descriptor.zero_()
                    commit.zero_()
                    visits.zero_()
                    worker_descriptor.zero_()
                    mismatch.zero_()
                    module.run(
                        state, descriptor, commit, visits,
                        worker_descriptor, mismatch, delay_cycles,
                    )
                    torch.cuda.synchronize()
                    state_cpu = state.cpu()
                    mismatch_value = int(mismatch.item())
                    min_visit = int(visits.min().item())
                    max_visit = int(visits.max().item())
                    if (
                        int(state_cpu[0]) != items
                        or int(state_cpu[1]) != items
                        or int(state_cpu[2]) != items
                        or mismatch_value != 0
                        or min_visit != 1
                        or max_visit != 1
                    ):
                        raise RuntimeError(
                            "ready queue gate failed "
                            f"clusters={clusters} items={items} "
                            f"delay={delay_cycles} repeat={repeat} "
                            f"state={state_cpu.tolist()} mismatch={mismatch_value} "
                            f"visit=[{min_visit},{max_visit}]"
                        )
                    digest.update(state_cpu.numpy().tobytes())
                    digest.update(visits.cpu().numpy().tobytes())
                row = {
                    "clusters": clusters,
                    "items": items,
                    "delay_cycles": delay_cycles,
                    "repeats": args.repeats,
                    "sha16": digest.hexdigest()[:16],
                }
                rows.append(row)
                print(
                    "READY_QUEUE_PASS|" + "|".join(
                        f"{key}={value}" for key, value in row.items()
                    ),
                    flush=True,
                )
    print("READY_QUEUE_JSON=" + json.dumps(rows, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
