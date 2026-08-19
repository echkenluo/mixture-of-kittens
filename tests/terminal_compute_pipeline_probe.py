#!/usr/bin/env python3
"""Exact single-kernel probe for terminal W13 -> activation -> W2 compute.

The split reference independently launches W13 and W2, but its activation
kernel intentionally shares ``activate_quant_worker`` with the candidate.  It
therefore proves the persistent-kernel integration, not activation arithmetic
independently.  ``--require-sglang-reference`` adds the production SGLang
activation kernel as that independent oracle, matching terminal_activation_probe.

The cursor follows native reverse-macrobatch and per-minibatch stage order.  Its
stage-local M-major order is an intentional port choice and is not claimed to
equal native MoK's expert-segment 2-D swizzle.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        default="64,128",
        help=(
            "additional comma-separated capacity=active M64 cases; required "
            "serial, reverse-tail, capacity-tail, and empty cases always run"
        ),
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument(
        "--require-sglang-reference",
        action="store_true",
        help="require production SGLang activation exactness in this run",
    )
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help="show the complete nvcc/ptxas build log",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Case:
    name: str
    capacity_rows: int
    active_rows: int
    minibatch_rows: int
    macrobatch_rows: int


def build_cases(extra_rows: list[int]) -> list[Case]:
    required = [
        # One worker executes all 33 logical tickets.  Every compute ticket
        # sequentially reuses the same mbarriers/phase bits for two N128 tiles.
        Case("serial33x2", 64, 64, 64, 64),
        Case("multi_m64", 128, 128, 128, 128),
        # Active rows 256..319 (the partial last macrobatch) are decoded first,
        # followed by the two minibatches in macrobatch zero.  The last capacity
        # M64 is inactive and must retain every sentinel/counter zero.
        Case("reverse_partial_capacity", 384, 320, 128, 256),
        Case("empty_rank", 64, 0, 64, 64),
    ]
    required.extend(
        Case(f"flat_{rows}", rows, rows, rows, rows) for rows in extra_rows
    )
    unique: dict[tuple[int, int, int, int], Case] = {}
    for case in required:
        unique.setdefault(
            (
                case.capacity_rows,
                case.active_rows,
                case.minibatch_rows,
                case.macrobatch_rows,
            ),
            case,
        )
    return list(unique.values())


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
        extra_ldflags=["-lcuda"],
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
    name: str,
    case: Case,
    workers: int,
    seed: int,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> None:
    mismatches = mismatch_count(reference, candidate)
    if mismatches:
        raise RuntimeError(
            f"{name} mismatch case={case.name} workers={workers} "
            f"seed={seed} count={mismatches}"
        )


def require_state(
    name: str, case: Case, workers: int, tensor: torch.Tensor, expected: int
) -> None:
    if tensor.numel() and not bool((tensor == expected).all()):
        raise RuntimeError(
            f"{name} mismatch case={case.name} workers={workers} "
            f"expected={expected}"
        )


def main() -> int:
    args = parse_args()
    extra_rows = sorted({int(value) for value in args.rows.split(",") if value})
    if any(rows <= 0 or rows % 64 for rows in extra_rows):
        raise ValueError("--rows must contain positive M64 multiples")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.experts < 2:
        raise ValueError("--experts must be at least 2 for task-local selection")
    cases = build_cases(extra_rows)

    torch.cuda.set_device(0)
    module = build_extension(args.verbose_build)
    sglang_reference = None
    if args.require_sglang_reference:
        from sglang.jit_kernel.dsv4 import (
            silu_and_mul_contig_post_quant,
        )

        sglang_reference = silu_and_mul_contig_post_quant
    attrs = [int(value) for value in module.attributes()]
    if attrs[2] != 0 or attrs[4] != 41984:
        raise RuntimeError(
            "sequential N256 resource budget violated: "
            f"local_bytes={attrs[2]} launch_smem={attrs[4]}"
        )
    worker_cases = sorted({1, attrs[5]})
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

        for case in cases:
            capacity_rows = case.capacity_rows
            active_rows = case.active_rows
            capacity_tiles = capacity_rows // 64
            active_tiles = active_rows // 64
            active_tasks = active_tiles * 33
            capacity_tasks = capacity_tiles * 33

            x = make_fp8((capacity_rows, 4096), generator)
            x_scale = make_scale((capacity_rows, 32), generator)
            # Each M64 tile uses one expert, matching the contiguous grouped
            # contract.  At least two active M64s select distinct experts.
            m_indices = torch.empty(
                capacity_rows, dtype=torch.int32, device="cuda"
            )
            for m_tile in range(capacity_tiles):
                m_indices[m_tile * 64 : (m_tile + 1) * 64] = (
                    m_tile % args.experts
                )
            if active_tiles >= 2:
                active_experts = m_indices[:active_rows:64]
                if int(torch.unique(active_experts).numel()) < 2:
                    raise RuntimeError(
                        f"multi-expert coverage missing for case={case.name}"
                    )
            num_tokens = torch.tensor(
                [active_rows], dtype=torch.int32, device="cuda"
            )

            gate_up_ref = torch.empty(
                (capacity_rows, 4096), dtype=torch.bfloat16, device="cuda"
            )
            hidden_ref = torch.empty(
                (capacity_rows, 2048),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            hidden_scale_ref = torch.empty(
                (capacity_rows, 16), dtype=torch.float32, device="cuda"
            )
            y_ref = torch.empty(
                (capacity_rows, 4096), dtype=torch.bfloat16, device="cuda"
            )
            if active_rows:
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

            sglang_hidden = None
            sglang_scale = None
            if sglang_reference is not None and active_rows:
                sglang_hidden = torch.empty_like(hidden_ref)
                sglang_scale = torch.empty_like(hidden_scale_ref)
                sglang_reference(
                    input=gate_up_ref[:active_rows],
                    output=sglang_hidden[:active_rows],
                    output_scale=sglang_scale[:active_rows],
                    quant_group_size=128,
                    scale_ue8m0=False,
                    transposed=False,
                    swiglu_limit=args.limit,
                    swizzle=False,
                )
                torch.cuda.synchronize()
                require_exact(
                    "sglang_activation_fp8_vs_split",
                    case,
                    0,
                    seed,
                    sglang_hidden[:active_rows],
                    hidden_ref[:active_rows],
                )
                require_exact(
                    "sglang_activation_scale_vs_split",
                    case,
                    0,
                    seed,
                    sglang_scale[:active_rows],
                    hidden_scale_ref[:active_rows],
                )

            for workers in worker_cases:
                gate_up = torch.full_like(gate_up_ref, float("nan"))
                hidden = torch.empty_like(hidden_ref)
                hidden.view(torch.uint8).fill_(0x7F)
                hidden_scale = torch.full_like(hidden_scale_ref, float("nan"))
                y = torch.full_like(y_ref, float("nan"))
                initial_gate_up = gate_up.clone()
                initial_hidden = hidden.clone()
                initial_hidden_scale = hidden_scale.clone()
                initial_y = y.clone()

                cursor = torch.zeros(1, dtype=torch.int32, device="cuda")
                worker_ticket = torch.zeros(
                    max(256, attrs[5]), dtype=torch.int32, device="cuda"
                )
                gate_up_ready = torch.zeros(
                    (capacity_tiles, 16), dtype=torch.int32, device="cuda"
                )
                hidden_ready = torch.zeros(
                    capacity_tiles, dtype=torch.int32, device="cuda"
                )
                y_ready = torch.zeros(
                    capacity_tiles, dtype=torch.int32, device="cuda"
                )
                task_visits = torch.zeros(
                    capacity_tasks, dtype=torch.int32, device="cuda"
                )
                errors = torch.zeros(1, dtype=torch.int32, device="cuda")

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
                    workers,
                    case.minibatch_rows,
                    case.macrobatch_rows,
                    args.limit,
                )
                torch.cuda.synchronize()

                if active_rows:
                    require_exact(
                        "gate",
                        case,
                        workers,
                        seed,
                        gate_up_ref[:active_rows, :2048],
                        gate_up[:active_rows, :2048],
                    )
                    require_exact(
                        "up",
                        case,
                        workers,
                        seed,
                        gate_up_ref[:active_rows, 2048:],
                        gate_up[:active_rows, 2048:],
                    )
                    require_exact(
                        "activation_fp8",
                        case,
                        workers,
                        seed,
                        hidden_ref[:active_rows],
                        hidden[:active_rows],
                    )
                    require_exact(
                        "activation_scale",
                        case,
                        workers,
                        seed,
                        hidden_scale_ref[:active_rows],
                        hidden_scale[:active_rows],
                    )
                    require_exact(
                        "w2",
                        case,
                        workers,
                        seed,
                        y_ref[:active_rows],
                        y[:active_rows],
                    )
                    if sglang_hidden is not None and sglang_scale is not None:
                        require_exact(
                            "sglang_activation_fp8_vs_candidate",
                            case,
                            workers,
                            seed,
                            sglang_hidden[:active_rows],
                            hidden[:active_rows],
                        )
                        require_exact(
                            "sglang_activation_scale_vs_candidate",
                            case,
                            workers,
                            seed,
                            sglang_scale[:active_rows],
                            hidden_scale[:active_rows],
                        )

                # Inactive capacity rows are a contract boundary, not padding
                # that the active-only cursor is permitted to overwrite.
                require_exact(
                    "inactive_gate_up",
                    case,
                    workers,
                    seed,
                    initial_gate_up[active_rows:],
                    gate_up[active_rows:],
                )
                require_exact(
                    "inactive_hidden",
                    case,
                    workers,
                    seed,
                    initial_hidden[active_rows:],
                    hidden[active_rows:],
                )
                require_exact(
                    "inactive_hidden_scale",
                    case,
                    workers,
                    seed,
                    initial_hidden_scale[active_rows:],
                    hidden_scale[active_rows:],
                )
                require_exact(
                    "inactive_y",
                    case,
                    workers,
                    seed,
                    initial_y[active_rows:],
                    y[active_rows:],
                )

                observed_cursor = int(cursor.item())
                observed_errors = int(errors.item())
                if observed_cursor != active_tasks:
                    raise RuntimeError(
                        f"cursor mismatch case={case.name} workers={workers}: "
                        f"{observed_cursor} != {active_tasks}"
                    )
                if observed_errors != 0:
                    raise RuntimeError(
                        f"device decode errors case={case.name} "
                        f"workers={workers}: {observed_errors}"
                    )
                require_state(
                    "active task visits",
                    case,
                    workers,
                    task_visits[:active_tasks],
                    1,
                )
                require_state(
                    "inactive task visits",
                    case,
                    workers,
                    task_visits[active_tasks:],
                    0,
                )
                require_state(
                    "active gate/up counters",
                    case,
                    workers,
                    gate_up_ready[:active_tiles],
                    2,
                )
                require_state(
                    "inactive gate/up counters",
                    case,
                    workers,
                    gate_up_ready[active_tiles:],
                    0,
                )
                require_state(
                    "active hidden counters",
                    case,
                    workers,
                    hidden_ready[:active_tiles],
                    1,
                )
                require_state(
                    "inactive hidden counters",
                    case,
                    workers,
                    hidden_ready[active_tiles:],
                    0,
                )
                require_state(
                    "active y counters",
                    case,
                    workers,
                    y_ready[:active_tiles],
                    32,
                )
                require_state(
                    "inactive y counters",
                    case,
                    workers,
                    y_ready[active_tiles:],
                    0,
                )

                if case.name == "serial33x2" and workers == 1:
                    if active_tasks != 33:
                        raise RuntimeError("serial phase case must contain 33 tickets")
                    print(
                        "TERMINAL_COMPUTE_PHASE_WRAP"
                        "|workers=1|tickets=33|n128_subtasks=2"
                        "|persistent_mbarrier=1"
                        "|w13_to_activation_to_w2=1",
                        flush=True,
                    )

                print(
                    "TERMINAL_COMPUTE_EXACT"
                    f"|case={case.name}|capacity_rows={capacity_rows}"
                    f"|active_rows={active_rows}|seed={seed}|workers={workers}"
                    f"|minibatch_rows={case.minibatch_rows}"
                    f"|macrobatch_rows={case.macrobatch_rows}"
                    "|order=reverse-macro+stage-major+m-major-port"
                    "|native_swizzle_equivalent=0"
                    f"|gate={int(active_rows > 0)}|up={int(active_rows > 0)}"
                    f"|activation_fp8={int(active_rows > 0)}"
                    f"|activation_scale={int(active_rows > 0)}"
                    f"|w2={int(active_rows > 0)}|tasks={active_tasks}"
                    "|inactive_unchanged=1|counters_exact=1"
                    f"|sglang_activation={int(sglang_reference is not None)}",
                    flush=True,
                )

    if sglang_reference is None:
        print(
            "TERMINAL_COMPUTE_ACTIVATION_ORACLE"
            "|shared_helper_integration_exact=1|independent_sglang=0"
            "|formal_command_add=--require-sglang-reference",
            flush=True,
        )
    else:
        print(
            "TERMINAL_COMPUTE_ACTIVATION_ORACLE"
            "|shared_helper_integration_exact=1|independent_sglang=1"
            "|fp8_exact=1|scale_exact=1",
            flush=True,
        )
    print(
        "TERMINAL_COMPUTE_WORKER_SCAN"
        f"|workers={','.join(str(value) for value in worker_cases)}"
        f"|max={attrs[5]}|result=PASS",
        flush=True,
    )
    print(
        "TERMINAL_COMPUTE_PIPELINE_PROBE"
        f"|sglang_oracle={int(sglang_reference is not None)}|result=PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
