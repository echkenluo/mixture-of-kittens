#!/usr/bin/env python3
"""Exact single-kernel probe for terminal W13 -> activation -> W2 compute.

The reference path is the repository's existing split cluster-2 contiguous
WGMMA kernel, followed by the established 256-thread activation oracle, then
the same split WGMMA kernel for W2.  The candidate executes every numerical
task from the committed 65-task/M64 device cursor inside one fixed-resident
cluster-worker kernel.  All four exposed boundaries are compared bitwise.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        default="64,128",
        help="comma-separated positive M64 row counts",
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help="show the complete nvcc/ptxas build log",
    )
    return parser.parse_args()


def thunderkittens_include() -> Path:
    configured = os.environ.get("THUNDERKITTENS_ROOT")
    root = Path(configured) if configured else ROOT / "third_party" / "ThunderKittens"
    header = root / "include" / "kittens.cuh"
    if not header.is_file():
        raise FileNotFoundError(
            f"ThunderKittens header missing: {header}; initialize the submodule "
            "or set THUNDERKITTENS_ROOT"
        )
    return root / "include"


def build_extension(verbose: bool):
    # The production MoK SM90 build uses sm_90a and fast math.  Matching both
    # is required for exact FP32 activation scales and for WGMMA availability.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    return load(
        name="mok_terminal_compute_pipeline_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_include_paths=[str(thunderkittens_include())],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-lineinfo",
            "--use_fast_math",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-Wno-psabi",
            "-Xcompiler=-fno-strict-aliasing",
            "-DKITTENS_SM90",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-Xptxas=-v",
            "-Xptxas=--warn-on-spills",
        ],
        verbose=verbose,
    )


def make_fp8(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    source = torch.randn(
        shape, dtype=torch.float32, device="cuda", generator=generator
    ).clamp_(-3.5, 3.5)
    return source.to(torch.float8_e4m3fn)


def make_scale(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return 0.02 + 0.03 * torch.rand(
        shape, dtype=torch.float32, device="cuda", generator=generator
    )


def mismatch_count(reference: torch.Tensor, candidate: torch.Tensor) -> int:
    if reference.dtype == torch.bfloat16:
        return int((reference.view(torch.int16) != candidate.view(torch.int16)).sum())
    if reference.dtype == torch.float8_e4m3fn:
        return int((reference.view(torch.uint8) != candidate.view(torch.uint8)).sum())
    if reference.dtype == torch.float32:
        return int((reference.view(torch.int32) != candidate.view(torch.int32)).sum())
    raise TypeError(f"unsupported exact-comparison dtype: {reference.dtype}")


def require_exact(
    name: str, rows: int, seed: int, reference: torch.Tensor, candidate: torch.Tensor
) -> None:
    mismatches = mismatch_count(reference, candidate)
    if mismatches:
        raise RuntimeError(
            f"{name} mismatch rows={rows} seed={seed} count={mismatches}"
        )


def main() -> int:
    args = parse_args()
    rows_cases = sorted({int(value) for value in args.rows.split(",") if value})
    if not rows_cases or any(rows <= 0 or rows % 64 for rows in rows_cases):
        raise ValueError("--rows must contain positive M64 multiples")
    if 64 not in rows_cases or not any(rows > 64 for rows in rows_cases):
        raise ValueError("the probe requires both single-M64 and multi-M64 cases")
    if args.seeds < 1 or args.experts < 1:
        raise ValueError("--seeds and --experts must be positive")

    torch.cuda.set_device(0)
    module = build_extension(args.verbose_build)
    attrs = [int(value) for value in module.attributes()]
    print(
        "TERMINAL_COMPUTE_ATTR"
        f"|regs={attrs[0]}|static_smem={attrs[1]}|local={attrs[2]}"
        f"|max_dynamic_smem={attrs[3]}|launch_smem={attrs[4]}"
        f"|resident_clusters={attrs[5]}|split_regs={attrs[6]}"
        f"|activation_regs={attrs[7]}",
        flush=True,
    )

    for seed in range(args.seeds):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        w13 = make_fp8((args.experts, 4096, 4096), generator)
        w13_scale = make_scale((args.experts, 32, 32), generator)
        w2 = make_fp8((args.experts, 4096, 2048), generator)
        w2_scale = make_scale((args.experts, 32, 16), generator)

        for rows in rows_cases:
            x = make_fp8((rows, 4096), generator)
            x_scale = make_scale((rows, 32), generator)
            # Each M64 tile uses one expert, matching the contiguous grouped
            # contract.  Cycling experts also verifies task-local selection.
            m_indices = torch.empty(rows, dtype=torch.int32, device="cuda")
            for m_tile in range(rows // 64):
                m_indices[m_tile * 64 : (m_tile + 1) * 64] = (
                    m_tile % args.experts
                )
            num_tokens = torch.tensor([rows], dtype=torch.int32, device="cuda")

            gate_up_ref = torch.empty(
                (rows, 4096), dtype=torch.bfloat16, device="cuda"
            )
            hidden_ref = torch.empty(
                (rows, 2048), dtype=torch.float8_e4m3fn, device="cuda"
            )
            hidden_scale_ref = torch.empty(
                (rows, 16), dtype=torch.float32, device="cuda"
            )
            y_ref = torch.empty(
                (rows, 4096), dtype=torch.bfloat16, device="cuda"
            )
            module.run_split(
                x,
                x_scale,
                w13,
                w13_scale,
                w2,
                w2_scale,
                m_indices,
                gate_up_ref,
                hidden_ref,
                hidden_scale_ref,
                y_ref,
                args.limit,
            )

            gate_up = torch.full_like(gate_up_ref, float("nan"))
            hidden = torch.empty_like(hidden_ref)
            hidden.view(torch.uint8).fill_(0x7F)
            hidden_scale = torch.full_like(hidden_scale_ref, float("nan"))
            y = torch.full_like(y_ref, float("nan"))
            m_tiles = rows // 64
            total_tasks = m_tiles * 65
            cursor = torch.zeros(1, dtype=torch.int32, device="cuda")
            worker_ticket = torch.zeros(
                max(256, attrs[5]), dtype=torch.int32, device="cuda"
            )
            gate_up_ready = torch.zeros(
                (m_tiles, 16), dtype=torch.int32, device="cuda"
            )
            hidden_ready = torch.zeros(m_tiles, dtype=torch.int32, device="cuda")
            y_ready = torch.zeros(m_tiles, dtype=torch.int32, device="cuda")
            task_visits = torch.zeros(
                total_tasks, dtype=torch.int32, device="cuda"
            )
            errors = torch.zeros(1, dtype=torch.int32, device="cuda")

            # One minibatch contains the whole case so rows>64 exercises the
            # decoder's stage-major ordering across multiple M64 tiles.
            module.run_terminal(
                x,
                x_scale,
                w13,
                w13_scale,
                w2,
                w2_scale,
                m_indices,
                num_tokens,
                gate_up,
                hidden,
                hidden_scale,
                y,
                cursor,
                worker_ticket,
                gate_up_ready,
                hidden_ready,
                y_ready,
                task_visits,
                errors,
                rows,
                rows,
                args.limit,
            )
            torch.cuda.synchronize()

            require_exact("gate", rows, seed, gate_up_ref[:, :2048], gate_up[:, :2048])
            require_exact("up", rows, seed, gate_up_ref[:, 2048:], gate_up[:, 2048:])
            require_exact("activation_fp8", rows, seed, hidden_ref, hidden)
            require_exact(
                "activation_scale", rows, seed, hidden_scale_ref, hidden_scale
            )
            require_exact("w2", rows, seed, y_ref, y)

            observed_cursor = int(cursor.item())
            observed_errors = int(errors.item())
            if observed_cursor != total_tasks:
                raise RuntimeError(
                    f"cursor mismatch rows={rows}: {observed_cursor} != {total_tasks}"
                )
            if observed_errors != 0:
                raise RuntimeError(
                    f"device decode errors rows={rows}: {observed_errors}"
                )
            if not bool((task_visits == 1).all()):
                raise RuntimeError(f"task visit mismatch rows={rows}")
            if not bool((gate_up_ready == 2).all()):
                raise RuntimeError(f"gate/up counter mismatch rows={rows}")
            if not bool((hidden_ready == 1).all()):
                raise RuntimeError(f"hidden counter mismatch rows={rows}")
            if not bool((y_ready == 32).all()):
                raise RuntimeError(f"y counter mismatch rows={rows}")

            print(
                "TERMINAL_COMPUTE_EXACT"
                f"|rows={rows}|m64={m_tiles}|seed={seed}"
                "|gate=1|up=1|activation_fp8=1|activation_scale=1|w2=1"
                f"|tasks={total_tasks}|gate_up_counter=2"
                "|hidden_counter=1|y_counter=32",
                flush=True,
            )

    print("TERMINAL_COMPUTE_PIPELINE_PROBE|result=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
