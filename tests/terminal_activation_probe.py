#!/usr/bin/env python3
"""Exact and timing probe for the terminal cluster-2 SwiGLU/FP8 mapping."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="64,128,384")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def build_extension():
    return load(
        name="mok_terminal_activation_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def elapsed(module, input_, output, scale, clustered, warmup, repeats):
    for _ in range(warmup):
        module.run(input_, output, scale, clustered, 10.0)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.run(input_, output, scale, clustered, 10.0)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), sorted(samples)[int(0.95 * (len(samples) - 1))]


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(0)
    module = build_extension()
    attributes = [int(value) for value in module.attributes()]
    print(
        "TERMINAL_ACTIVATION_ATTR"
        f"|ref_regs={attributes[0]}|ref_smem={attributes[1]}"
        f"|ref_local={attributes[2]}|cluster_regs={attributes[3]}"
        f"|cluster_smem={attributes[4]}|cluster_local={attributes[5]}"
        f"|occupancy_clusters={attributes[6]}",
        flush=True,
    )
    results = []
    for rows_text in args.rows.split(","):
        rows = int(rows_text)
        if rows <= 0 or rows % 64:
            raise ValueError("all row counts must be positive M64 multiples")
        for seed in range(args.seeds):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            input_ = (
                torch.randn(
                    (rows, 4096),
                    dtype=torch.float32,
                    device="cuda",
                    generator=generator,
                )
                * 7.0
            ).to(torch.bfloat16)
            # Exercise clamp edges and zero-scale protection deterministically.
            input_[0, :8] = torch.tensor(
                [-20.0, -10.0, -0.0, 0.0, 10.0, 20.0, 1e-8, -1e-8],
                dtype=torch.bfloat16,
                device="cuda",
            )
            ref = torch.empty(
                (rows, 2048), dtype=torch.float8_e4m3fn, device="cuda"
            )
            ref_scale = torch.empty((rows, 16), dtype=torch.float32, device="cuda")
            candidate = torch.empty_like(ref)
            candidate_scale = torch.empty_like(ref_scale)
            module.run(input_, ref, ref_scale, False, 10.0)
            module.run(input_, candidate, candidate_scale, True, 10.0)
            torch.cuda.synchronize()
            fp8_exact = torch.equal(ref.view(torch.uint8), candidate.view(torch.uint8))
            scale_exact = torch.equal(ref_scale.view(torch.int32), candidate_scale.view(torch.int32))
            if not fp8_exact or not scale_exact:
                fp8_mismatch = int(
                    (ref.view(torch.uint8) != candidate.view(torch.uint8)).sum().item()
                )
                scale_mismatch = int(
                    (ref_scale.view(torch.int32) != candidate_scale.view(torch.int32)).sum().item()
                )
                raise RuntimeError(
                    f"activation mismatch rows={rows} seed={seed} "
                    f"fp8={fp8_mismatch} scale={scale_mismatch}"
                )
            print(
                f"TERMINAL_ACTIVATION_EXACT|rows={rows}|seed={seed}"
                "|fp8=1|scale=1",
                flush=True,
            )
        reference_p50, reference_p95 = elapsed(
            module, input_, ref, ref_scale, False, args.warmup, args.repeats
        )
        cluster_p50, cluster_p95 = elapsed(
            module, input_, candidate, candidate_scale, True,
            args.warmup, args.repeats,
        )
        row = {
            "rows": rows,
            "reference_p50_ms": reference_p50,
            "reference_p95_ms": reference_p95,
            "cluster_p50_ms": cluster_p50,
            "cluster_p95_ms": cluster_p95,
        }
        results.append(row)
        print(
            "TERMINAL_ACTIVATION_TIMING|"
            + "|".join(
                f"{key}={value:.6f}" if isinstance(value, float)
                else f"{key}={value}"
                for key, value in row.items()
            ),
            flush=True,
        )
    print("TERMINAL_ACTIVATION_JSON=" + json.dumps(results, sort_keys=True), flush=True)
    print("TERMINAL_ACTIVATION_PROBE|result=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
