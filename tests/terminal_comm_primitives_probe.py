#!/usr/bin/env python3
"""Exact and route-mapping probe for terminal communication helpers."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cu")
HEADER = REPO_ROOT / "csrc" / "sm90_fp8_block_terminal_comm_primitives.cuh"
ENTRY = REPO_ROOT / "csrc" / "sm90_fp8_block_terminal_entry.cuh"
FUNCTIONAL = REPO_ROOT / "mok" / "functional.py"
OPS = REPO_ROOT / "mok" / "ops.py"
PY_CONTRACT = REPO_ROOT / "mok" / "_terminal_tma_contract.py"
K1 = REPO_ROOT / "csrc" / "sm90_fp8_block_dispatch_gemm.cuh"
K2 = REPO_ROOT / "csrc" / "sm90_fp8_block_gemm_combine.cuh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "skip", "require"), default="auto")
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def check_static_contract() -> None:
    header = HEADER.read_text(encoding="utf-8")
    forbidden = (
        r"atomicAdd\s*\(",
        r"asm\s+volatile",
        r"__syncthreads\s*\(",
        r"__syncwarp\s*\(",
        r"__trap\s*\(",
        r"__nanosleep\s*\(",
        r"ticket_counter",
        r"completion_counter",
        r"barrier_target",
        r"tile_ready",
    )
    matches = [pattern for pattern in forbidden if re.search(pattern, header)]
    if matches:
        raise RuntimeError(f"scheduling/publication leaked into helper header: {matches}")
    entry = ENTRY.read_text(encoding="utf-8")
    native_gates = (
        "!x_buffer.is_alias_of(routed_x)",
        "!x_buffer.is_alias_of(routed_x_scale)",
        "!x_scale_buffer.is_alias_of(routed_x)",
        "!x_scale_buffer.is_alias_of(routed_x_scale)",
        "tma_contract::is_raw_bulk_aligned",
        "tma_contract::byte_intervals_overlap",
        "x_bytes_per_rank",
        "x_scale_bytes_per_rank",
    )
    missing_native = [needle for needle in native_gates if needle not in entry]
    if missing_native:
        raise RuntimeError(
            f"native entry lacks raw bulk-TMA storage gates: {missing_native}"
        )
    python_entry = (
        FUNCTIONAL.read_text(encoding="utf-8")
        + OPS.read_text(encoding="utf-8")
    )
    if (
        "validate_terminal_tma_dispatch_layout(" not in python_entry
        or "untyped_storage().nbytes()" not in python_entry
        or "RAW_BULK_ALIGNMENT = 16"
        not in PY_CONTRACT.read_text(encoding="utf-8")
    ):
        raise RuntimeError("Python entry lacks raw bulk-TMA storage gates")
    k1 = K1.read_text(encoding="utf-8")
    k2 = K2.read_text(encoding="utf-8")
    required = (
        ("K1 dispatch helper", "fp8_block_terminal_comm::dispatch_copy_row", k1),
        ("K1 row publication", "red.release.gpu.global.add.u32", k1),
        ("K2 push helper", "fp8_block_terminal_comm::push_routed_row", k2),
        ("K2 system fence", "fence.release.sys", k2),
        ("probe dispatch helper", "comm::dispatch_copy_row", SOURCE.read_text()),
        ("probe push helper", "comm::push_routed_row", SOURCE.read_text()),
    )
    missing = [label for label, needle, text in required if needle not in text]
    if missing:
        raise RuntimeError(f"shared helper wiring missing: {missing}")
    print(
        "TERMINAL_COMM_STATIC"
        "|scheduling=0|publication=caller-owned|k1_wired=1|k2_wired=1|result=PASS"
    )


def cuda_is_available() -> bool:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False
    import subprocess

    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0 and any(
        line.strip().startswith("9.0") for line in probe.stdout.splitlines()
    )


def build_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="mok_terminal_comm_primitives_probe",
        sources=[str(SOURCE)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
        verbose=False,
    )


def expected_experts(tokens_per_expert, rows: int):
    import torch

    ends = torch.cumsum(tokens_per_expert.cpu(), dim=0)
    result = torch.empty(rows, dtype=torch.int32)
    for row in range(rows):
        expert = 0
        while expert < len(ends) - 1 and row >= int(ends[expert]):
            expert += 1
        result[row] = expert
    return result


def run_device(seeds: int) -> None:
    import torch

    torch.cuda.set_device(0)
    module = build_extension()
    peers = 4
    capacity = 128
    active = 64
    local_tokens = 16
    hidden = 4096
    topk = 6
    route_slots = local_tokens * topk
    tokens_per_expert = torch.tensor(
        [0, 16, 16, 32], dtype=torch.int32, device="cuda"
    )
    schedule_peer = torch.arange(capacity, dtype=torch.int32, device="cuda") % peers
    schedule_token = torch.arange(capacity, dtype=torch.int32, device="cuda") // peers
    schedule_peer[3] = -1
    schedule_peer[7] = peers
    schedule_token[11] = -1
    schedule_token[15] = route_slots
    num_tokens = torch.tensor([active], dtype=torch.int32, device="cuda")
    valid = (
        (schedule_peer >= 0)
        & (schedule_peer < peers)
        & (schedule_token >= 0)
        & (schedule_token < route_slots)
    )

    for seed in range(seeds):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        peer_x = torch.randint(
            0,
            256,
            (peers, local_tokens, hidden),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        peer_scale = torch.randn(
            (peers, local_tokens, hidden // 128),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        shared_x = torch.full(
            (capacity, hidden), 0xA5, dtype=torch.uint8, device="cuda"
        )
        reference_x = shared_x.clone()
        shared_scale = torch.full(
            (capacity, hidden // 128), -123.0, dtype=torch.float32, device="cuda"
        )
        reference_scale = shared_scale.clone()
        shared_indices = torch.full((capacity,), -99, dtype=torch.int32, device="cuda")
        reference_indices = shared_indices.clone()
        module.run_dispatch(
            peer_x,
            peer_scale,
            schedule_peer,
            schedule_token,
            num_tokens,
            tokens_per_expert,
            shared_x,
            shared_scale,
            shared_indices,
            reference_x,
            reference_scale,
            reference_indices,
            topk,
        )
        torch.cuda.synchronize()
        if not torch.equal(shared_x, reference_x):
            raise RuntimeError(f"dispatch FP8 byte mismatch for seed {seed}")
        if not torch.equal(shared_scale, reference_scale):
            raise RuntimeError(f"dispatch scale mismatch for seed {seed}")
        if not torch.equal(shared_indices, reference_indices):
            raise RuntimeError(f"dispatch expert mismatch for seed {seed}")
        if not torch.equal(
            shared_indices[:active].cpu(),
            expected_experts(tokens_per_expert, active),
        ):
            raise RuntimeError(f"dispatch expert mapping mismatch for seed {seed}")

        for row in range(active):
            if bool(valid[row]):
                peer = int(schedule_peer[row])
                source = int(schedule_token[row]) // topk
                if not torch.equal(shared_x[row], peer_x[peer, source]):
                    raise RuntimeError(f"dispatch route mismatch at row {row}")
                if not torch.equal(shared_scale[row], peer_scale[peer, source]):
                    raise RuntimeError(f"dispatch scale route mismatch at row {row}")
            elif torch.count_nonzero(shared_x[row]).item() != 0:
                raise RuntimeError(f"invalid dispatch row {row} was not zeroed")

        routed_y = torch.randn(
            (capacity, hidden),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        shared_combine = torch.full(
            (peers, route_slots, hidden),
            17.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference_combine = shared_combine.clone()
        route_receipt = torch.full(
            (capacity, 4), -99, dtype=torch.int32, device="cuda"
        )
        module.run_push(
            routed_y,
            schedule_peer,
            schedule_token,
            num_tokens,
            shared_combine,
            reference_combine,
            route_receipt,
            topk,
        )
        torch.cuda.synchronize()
        if not torch.equal(shared_combine, reference_combine):
            raise RuntimeError(f"push BF16 mismatch for seed {seed}")
        for row in range(active):
            peer = int(schedule_peer[row])
            token = int(schedule_token[row])
            is_valid = int(bool(valid[row]))
            expected = [peer, token, token // topk if is_valid else -1, is_valid]
            if route_receipt[row].cpu().tolist() != expected:
                raise RuntimeError(f"route receipt mismatch at row {row}")
            if is_valid and not torch.equal(
                shared_combine[peer, token], routed_y[row]
            ):
                raise RuntimeError(f"push route mismatch at row {row}")
        if not torch.all(route_receipt[active:] == -99):
            raise RuntimeError("inactive route receipt tail was modified")
        print(
            f"TERMINAL_COMM_DEVICE|seed={seed}|dispatch_exact=1"
            "|push_exact=1|route_mapping=1|result=PASS"
        )


def main() -> int:
    args = parse_args()
    check_static_contract()
    available = cuda_is_available()
    if args.device == "require" and not available:
        raise RuntimeError("SM90 CUDA device was required but is unavailable")
    if args.device != "skip" and available:
        run_device(args.seeds)
    else:
        print("TERMINAL_COMM_DEVICE|result=SKIP|reason=unavailable")
    print("TERMINAL_COMM_PROBE|result=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
