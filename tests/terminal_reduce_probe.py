#!/usr/bin/env python3
"""Compare terminal cluster-2 weighted reduce with production MoK epilogue."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="2,64,4096")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def build_extension():
    return load(
        name="mok_terminal_reduce_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        verbose=False,
    )


def elapsed(call, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), sorted(samples)[int(0.95 * (len(samples) - 1))]


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(0)
    from mok.ops import routed_epilogue_out

    module = build_extension()
    attributes = [int(value) for value in module.attributes()]
    print(
        "TERMINAL_REDUCE_ATTR"
        f"|ref_regs={attributes[0]}|ref_smem={attributes[1]}"
        f"|ref_local={attributes[2]}|cluster_regs={attributes[3]}"
        f"|cluster_smem={attributes[4]}|cluster_local={attributes[5]}"
        f"|occupancy_clusters={attributes[6]}",
        flush=True,
    )
    results = []
    for tokens_text in args.tokens.split(","):
        tokens = int(tokens_text)
        if tokens <= 0 or tokens % 2:
            raise ValueError("token counts must be positive and even")
        for seed in range(args.seeds):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            combine = (
                torch.randn(
                    (tokens * args.topk, args.hidden),
                    dtype=torch.float32,
                    device="cuda",
                    generator=generator,
                )
                * 3.0
            ).to(torch.bfloat16)
            weights = torch.rand(
                (tokens, args.topk),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            weights /= weights.sum(dim=-1, keepdim=True)
            reference = torch.empty(
                (tokens, args.hidden), dtype=torch.bfloat16, device="cuda"
            )
            production_reference = torch.empty_like(reference)
            candidate = torch.empty_like(reference)
            module.run(combine, weights, reference, False)
            module.run(combine, weights, candidate, True)
            compare_production = tokens >= 256 and tokens % 256 == 0
            if compare_production:
                routed_epilogue_out(combine, weights, production_reference)
            torch.cuda.synchronize()
            exact = torch.equal(
                reference.view(torch.uint16), candidate.view(torch.uint16)
            )
            mismatch = int(
                (reference.view(torch.uint16) != candidate.view(torch.uint16))
                .sum()
                .item()
            )
            relative_l2 = float(
                torch.linalg.vector_norm(
                    reference.float() - candidate.float()
                ).item()
                / max(torch.linalg.vector_norm(reference.float()).item(), 1e-20)
            )
            print(
                f"TERMINAL_REDUCE_NUMERIC|tokens={tokens}|seed={seed}"
                f"|exact={int(exact)}|mismatch={mismatch}"
                f"|relative_l2={relative_l2:.9g}"
                f"|production_reference={int(compare_production)}",
                flush=True,
            )
            if not exact:
                raise RuntimeError(
                    f"reduce mismatch tokens={tokens} seed={seed} "
                    f"count={mismatch} relative_l2={relative_l2}"
                )
            if compare_production and not torch.equal(
                production_reference.view(torch.uint16),
                candidate.view(torch.uint16),
            ):
                production_mismatch = int(
                    (
                        production_reference.view(torch.uint16)
                        != candidate.view(torch.uint16)
                    )
                    .sum()
                    .item()
                )
                raise RuntimeError(
                    f"production reduce mismatch tokens={tokens} seed={seed} "
                    f"count={production_mismatch}"
                )
        reference_call = (
            (lambda: routed_epilogue_out(combine, weights, production_reference))
            if compare_production
            else (lambda: module.run(combine, weights, reference, False))
        )
        reference_p50, reference_p95 = elapsed(
            reference_call,
            args.warmup,
            args.repeats,
        )
        candidate_p50, candidate_p95 = elapsed(
            lambda: module.run(combine, weights, candidate, True),
            args.warmup,
            args.repeats,
        )
        row = {
            "tokens": tokens,
            "reference_p50_ms": reference_p50,
            "reference_p95_ms": reference_p95,
            "candidate_p50_ms": candidate_p50,
            "candidate_p95_ms": candidate_p95,
        }
        results.append(row)
        print(
            "TERMINAL_REDUCE_TIMING|"
            + "|".join(
                f"{key}={value:.6f}" if isinstance(value, float)
                else f"{key}={value}"
                for key, value in row.items()
            ),
            flush=True,
        )
    print("TERMINAL_REDUCE_JSON=" + json.dumps(results, sort_keys=True), flush=True)
    print("TERMINAL_REDUCE_PROBE|result=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
