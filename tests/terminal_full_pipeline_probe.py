#!/usr/bin/env python3
"""Single-launch terminal FP8 M1 N-worker pipeline versus split oracle."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cu")
HEADER = ROOT / "csrc" / "sm90_fp8_block_terminal_full.cuh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "skip", "require"), default="auto")
    parser.add_argument("--rows", default="64,128")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--spin-limit", type=int, default=1 << 27)
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args()


def check_source_contract() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    driver = Path(__file__).read_text(encoding="utf-8")
    required = (
        "sm90_fp8_block_terminal_comm_primitives.cuh",
        "sm90_fp8_block_terminal_compute.cuh",
        "sm90_fp8_block_terminal_route_flags.cuh",
        "dispatch_copy_row",
        "push_routed_row_and_publish",
        "try_claim_ready_token",
        "try_reduce_one_ready_token",
        "reduce_claimed_token",
        "decode_logical_cursor",
        "fence.proxy.async.global",
        "worker_failed",
        "progress_timeouts",
        "overlap_witness",
        "OVERLAP_REDUCE_COMM",
    )
    missing = [needle for needle in required if needle not in header]
    if missing:
        raise RuntimeError(f"full terminal wiring missing: {missing}")
    forbidden = (
        "grid.sync",
        "cooperative_groups",
        "ticket_queue",
        "completion_counter",
        "__fmul_rn",
        "__fmaf_rn",
    )
    leaked = [needle for needle in forbidden if needle in header]
    if leaked:
        raise RuntimeError(f"barrier/queue/copied arithmetic leaked into full header: {leaked}")
    if header.count("__global__ void kernel") != 1:
        raise RuntimeError("full header must expose exactly one kernel")
    probe_start = header.find("try_reduce_one_ready_token")
    probe_end = header.find("compute_and_reduce_role", probe_start)
    probe = header[probe_start:probe_end]
    if "while" in probe or "for (" in probe or "__nanosleep" in probe:
        raise RuntimeError("ready-token probe must be one-shot and nonblocking")
    compute = header[probe_end : header.find("__global__ void kernel", probe_end)]
    if compute.count("try_reduce_one_ready_token(g)") < 3:
        raise RuntimeError(
            "compute workers must probe after tasks, cursor stop, and wait windows"
        )
    if "result == route::claim_result::claimed" not in compute:
        raise RuntimeError("winning CTA does not resume the compute loop")
    run_full = source[source.find("void run_full(") : source.find("std::vector<int64_t> attributes")]
    launches = re.findall(r"full::kernel<gemm_problem>\s*\n?\s*<<<", run_full)
    if len(launches) != 1:
        raise RuntimeError(f"candidate path must launch one kernel, found {len(launches)}")
    dynamic_required = (
        "worker_ticket.numel() == compute_clusters",
        "worker_failed.numel() == compute_clusters",
        "max_resident_clusters - full::COMM_CLUSTERS",
        "full::COMM_CLUSTERS + g.compute_clusters",
    )
    missing_dynamic = [item for item in dynamic_required if item not in source]
    if missing_dynamic:
        raise RuntimeError(f"dynamic resident launch contract missing: {missing_dynamic}")
    scan_required = (
        "min(7, max_compute_clusters)",
        "resident_clusters - 1",
        '"progress_timeouts"',
        '"worker_ticket": torch.zeros(\n                        compute_clusters',
    )
    missing_scan = [item for item in scan_required if item not in driver]
    if missing_scan:
        raise RuntimeError(f"M1 N-scan/progress contract missing: {missing_scan}")
    print(
        "TERMINAL_FULL_SOURCE"
        "|milestone=M1|comm_clusters=1|compute_clusters=dynamic"
        "|reduce_probe=bounded_one_shot|not_ready_wait=0"
        "|cluster_dim=2|candidate_launches=1|grid_barrier=0"
        "|split_fallback=0|core_arithmetic_copy=0|result=PASS",
        flush=True,
    )


def sm90_available() -> bool:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if query.returncode != 0 or not any(
        line.strip().startswith("9.0") for line in query.stdout.splitlines()
    ):
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and any(
        torch.cuda.get_device_capability(index) == (9, 0)
        for index in range(torch.cuda.device_count())
    )


def thunderkittens_include() -> Path:
    configured = os.environ.get("THUNDERKITTENS_ROOT")
    root = Path(configured) if configured else ROOT / "third_party" / "ThunderKittens"
    header = root / "include" / "kittens.cuh"
    if not header.is_file():
        raise FileNotFoundError(f"ThunderKittens header missing: {header}")
    return root / "include"


def build_extension(verbose: bool):
    import torch
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    return load(
        name="mok_terminal_full_pipeline_probe",
        sources=[str(SOURCE)],
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


def first_sm90_device() -> int:
    import torch

    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index) == (9, 0):
            return index
    raise RuntimeError("no SM90 device")


def make_fp8(shape: tuple[int, ...], generator):
    import torch

    return torch.randn(
        shape, dtype=torch.float32, device="cuda", generator=generator
    ).clamp_(-3.5, 3.5).to(torch.float8_e4m3fn)


def make_scale(shape: tuple[int, ...], generator):
    import torch

    return 0.02 + 0.03 * torch.rand(
        shape, dtype=torch.float32, device="cuda", generator=generator
    )


def require_exact(name: str, rows: int, seed: int, reference, candidate) -> None:
    import torch

    if reference.dtype == torch.float8_e4m3fn:
        lhs, rhs = reference.view(torch.uint8), candidate.view(torch.uint8)
    elif reference.dtype == torch.bfloat16:
        lhs, rhs = reference.view(torch.int16), candidate.view(torch.int16)
    elif reference.dtype in (torch.float32, torch.int32):
        lhs, rhs = reference.view(torch.int32), candidate.view(torch.int32)
    else:
        raise TypeError(f"unsupported exact dtype: {reference.dtype}")
    mismatches = int((lhs != rhs).sum().item())
    if mismatches:
        raise RuntimeError(
            f"{name} mismatch rows={rows} seed={seed} count={mismatches}"
        )


def make_route_contract(rows: int, local_tokens: int):
    import torch

    schedule_peer = torch.empty(rows, dtype=torch.int32)
    schedule_slot = torch.empty(rows, dtype=torch.int32)
    next_slot = [0, 0, 0, 0]
    invalid_rows = {5, rows - 3}
    valid_locations: list[tuple[int, int]] = []
    for row in range(rows):
        if row in invalid_rows:
            schedule_peer[row] = -1
            schedule_slot[row] = -1
            continue
        peer = (3 * row + 1) % 4
        slot = next_slot[peer]
        next_slot[peer] += 1
        if slot >= local_tokens * 6:
            raise RuntimeError("local token domain is too small")
        schedule_peer[row] = peer
        schedule_slot[row] = slot
        valid_locations.append((peer, slot))

    push_order = torch.empty(rows, dtype=torch.int32)
    for first in range(0, rows, 64):
        push_order[first : first + 64] = torch.arange(
            first + 63, first - 1, -1, dtype=torch.int32
        )
    if bool(torch.equal(push_order, torch.arange(rows, dtype=torch.int32))):
        raise RuntimeError("push order must be non-identity")
    return (
        schedule_peer.cuda(),
        schedule_slot.cuda(),
        push_order.cuda(),
        invalid_rows,
        valid_locations,
    )


def run_device(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(first_sm90_device())
    module = build_extension(args.verbose_build)
    attrs = [int(value) for value in module.attributes()]
    if attrs[5] < 2 or attrs[6] != attrs[5] - 1:
        raise RuntimeError(f"resident compute capacity is invalid: {attrs}")
    resident_clusters = attrs[5]
    max_compute_clusters = attrs[6]
    compute_cases = sorted(
        {
            1,
            min(7, max_compute_clusters),
            resident_clusters - 1,
        }
    )
    if compute_cases[0] < 1 or compute_cases[-1] > max_compute_clusters:
        raise RuntimeError(
            f"N scan exceeds resident capacity: {compute_cases}, attrs={attrs}"
        )
    print(
        "TERMINAL_FULL_ATTR"
        f"|regs={attrs[0]}|static_smem={attrs[1]}|local={attrs[2]}"
        f"|max_dynamic_smem={attrs[3]}|launch_smem={attrs[4]}"
        f"|resident_clusters={resident_clusters}"
        f"|max_compute_clusters={max_compute_clusters}"
        f"|compute_scan={','.join(map(str, compute_cases))}",
        flush=True,
    )

    rows_cases = sorted({int(value) for value in args.rows.split(",") if value})
    if rows_cases != [64, 128]:
        raise ValueError("--rows must cover exactly 64,128")
    if args.seeds < 1 or args.spin_limit < 1:
        raise ValueError("--seeds and --spin-limit must be positive")

    local_tokens = 8
    global_tokens = 4 * local_tokens
    for seed in range(args.seeds):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        # The communication primitive owns raw E4M3 bytes; routed_x receives
        # those bytes into a typed float8 tensor consumed by WGMMA.
        peer_x = make_fp8((4, local_tokens, 4096), generator).view(torch.uint8)
        peer_scale = make_scale((4, local_tokens, 32), generator)
        w13 = make_fp8((1, 4096, 4096), generator)
        w13_scale = make_scale((1, 32, 32), generator)
        w2 = make_fp8((1, 4096, 2048), generator)
        w2_scale = make_scale((1, 32, 16), generator)

        for rows in rows_cases:
            schedule_peer, schedule_slot, push_order, invalid_rows, valid = (
                make_route_contract(rows, local_tokens)
            )
            num_tokens = torch.tensor([rows], dtype=torch.int32, device="cuda")
            tokens_per_expert = torch.tensor(
                [rows], dtype=torch.int32, device="cuda"
            )
            topk_ids = torch.full(
                (4, local_tokens, 6), -1, dtype=torch.int32, device="cuda"
            )
            route_ready_initial = torch.ones_like(topk_ids)
            for ordinal, (peer, slot) in enumerate(valid):
                topk_ids.view(4, -1)[peer, slot] = ordinal
                route_ready_initial.view(4, -1)[peer, slot] = 0
            weights = torch.zeros(
                (4, local_tokens, 6), dtype=torch.float32, device="cuda"
            )
            random_weights = torch.rand(
                weights.shape, dtype=torch.float32,
                device="cuda", generator=generator
            )
            valid_mask = topk_ids >= 0
            weights[valid_mask] = random_weights[valid_mask]
            sums = weights.sum(dim=-1, keepdim=True)
            weights = torch.where(sums > 0, weights / sums.clamp_min(1e-20), weights)

            def stage_buffers():
                routed_x = torch.empty(
                    (rows, 4096), dtype=torch.float8_e4m3fn, device="cuda"
                )
                routed_x.view(torch.uint8).fill_(0x7F)
                return {
                    "x": routed_x,
                    "x_scale": torch.full(
                        (rows, 32), float("nan"), dtype=torch.float32,
                        device="cuda"
                    ),
                    "m_indices": torch.full(
                        (rows,), -1, dtype=torch.int32, device="cuda"
                    ),
                    "gate_up": torch.full(
                        (rows, 4096), float("nan"), dtype=torch.bfloat16,
                        device="cuda"
                    ),
                    "hidden": torch.empty(
                        (rows, 2048), dtype=torch.float8_e4m3fn, device="cuda"
                    ),
                    "hidden_scale": torch.full(
                        (rows, 16), float("nan"), dtype=torch.float32,
                        device="cuda"
                    ),
                    "y": torch.full(
                        (rows, 4096), float("nan"), dtype=torch.bfloat16,
                        device="cuda"
                    ),
                    "combine": torch.full(
                        (4, local_tokens * 6, 4096), float("nan"),
                        dtype=torch.bfloat16, device="cuda"
                    ),
                    "output": torch.full(
                        (4, local_tokens, 4096), float("nan"),
                        dtype=torch.bfloat16, device="cuda"
                    ),
                }

            reference = stage_buffers()
            reference["hidden"].view(torch.uint8).fill_(0x7F)
            module.run_split(
                peer_x, peer_scale, w13, w13_scale, w2, w2_scale,
                schedule_peer, schedule_slot, tokens_per_expert, push_order,
                weights, topk_ids,
                reference["x"], reference["x_scale"], reference["m_indices"],
                reference["gate_up"], reference["hidden"],
                reference["hidden_scale"], reference["y"],
                reference["combine"], reference["output"], 10.0,
            )

            m_tiles = rows // 64
            total_tasks = rows // 64 * 65
            for compute_clusters in compute_cases:
                candidate = stage_buffers()
                candidate["hidden"].view(torch.uint8).fill_(0x7F)
                route_ready = route_ready_initial.clone()
                state = {
                    "gate_up_ready": torch.zeros(
                        (m_tiles, 16), dtype=torch.int32, device="cuda"
                    ),
                    "hidden_ready": torch.zeros(
                        m_tiles, dtype=torch.int32, device="cuda"
                    ),
                    "y_ready": torch.zeros(
                        m_tiles, dtype=torch.int32, device="cuda"
                    ),
                    "x_ready": torch.zeros(
                        m_tiles, dtype=torch.int32, device="cuda"
                    ),
                    "cursor": torch.zeros(1, dtype=torch.int32, device="cuda"),
                    "worker_ticket": torch.zeros(
                        compute_clusters, dtype=torch.int32, device="cuda"
                    ),
                    "worker_failed": torch.zeros(
                        compute_clusters, dtype=torch.int32, device="cuda"
                    ),
                    "next_reduce_probe": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    "reduce_done": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    "comm_closed": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    "comm_failed": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    "task_visits": torch.zeros(
                        total_tasks, dtype=torch.int32, device="cuda"
                    ),
                    "dispatch_visits": torch.zeros(
                        rows, dtype=torch.int32, device="cuda"
                    ),
                    "push_visits": torch.zeros(
                        rows, dtype=torch.int32, device="cuda"
                    ),
                    "epilogue_claim": torch.zeros(
                        (4, local_tokens), dtype=torch.int32, device="cuda"
                    ),
                    "reduce_visits": torch.zeros(
                        global_tokens, dtype=torch.int32, device="cuda"
                    ),
                    "errors": torch.zeros(1, dtype=torch.int32, device="cuda"),
                    "progress_timeouts": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    "dispatch_tiles_done": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    "compute_started": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    "overlap_witness": torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                }
                module.run_full(
                    peer_x, peer_scale, w13, w13_scale, w2, w2_scale,
                    schedule_peer, schedule_slot, num_tokens, tokens_per_expert,
                    push_order, weights, topk_ids,
                    candidate["x"], candidate["x_scale"],
                    candidate["m_indices"], candidate["gate_up"],
                    candidate["hidden"], candidate["hidden_scale"],
                    candidate["y"], candidate["combine"], route_ready,
                    candidate["output"], state["gate_up_ready"],
                    state["hidden_ready"], state["y_ready"], state["x_ready"],
                    state["cursor"], state["worker_ticket"],
                    state["worker_failed"], state["next_reduce_probe"],
                    state["reduce_done"], state["comm_closed"],
                    state["comm_failed"], state["task_visits"],
                    state["dispatch_visits"], state["push_visits"],
                    state["epilogue_claim"], state["reduce_visits"],
                    state["errors"], state["progress_timeouts"],
                    state["dispatch_tiles_done"], state["compute_started"],
                    state["overlap_witness"], compute_clusters, rows, rows,
                    (1 << 20) if rows > 64 else 0, args.spin_limit, 10.0,
                )
                torch.cuda.synchronize()

                for name in (
                    "x", "x_scale", "m_indices", "gate_up", "hidden",
                    "hidden_scale", "y", "combine", "output"
                ):
                    require_exact(
                        name, rows, seed, reference[name], candidate[name]
                    )
                if not bool(
                    (candidate["x"][list(invalid_rows)].view(torch.uint8) == 0)
                    .all()
                ):
                    raise RuntimeError(
                        f"invalid dispatch payload is not zero rows={rows}"
                    )
                if not bool(
                    (candidate["x_scale"][list(invalid_rows)] == 0).all()
                ):
                    raise RuntimeError(
                        f"invalid dispatch scale is not zero rows={rows}"
                    )

                scalar_expected = {
                    "cursor": total_tasks,
                    "reduce_done": global_tokens,
                    "comm_closed": 1,
                    "comm_failed": 0,
                    "errors": 0,
                    "progress_timeouts": 0,
                    "dispatch_tiles_done": m_tiles,
                    "compute_started": 1,
                }
                for name, expected in scalar_expected.items():
                    actual = int(state[name].item())
                    if actual != expected:
                        raise RuntimeError(
                            f"{name} mismatch rows={rows} N={compute_clusters}: "
                            f"{actual} != {expected}"
                        )
                if not bool((state["worker_failed"] == 0).all()):
                    raise RuntimeError(
                        f"worker failure rows={rows} N={compute_clusters}"
                    )
                exact_once = (
                    "task_visits", "dispatch_visits", "push_visits",
                    "epilogue_claim", "reduce_visits"
                )
                for name in exact_once:
                    if not bool((state[name] == 1).all()):
                        raise RuntimeError(
                            f"{name} is not exactly once rows={rows} "
                            f"N={compute_clusters}"
                        )
                counter_expected = {
                    "x_ready": 64,
                    "gate_up_ready": 2,
                    "hidden_ready": 1,
                    "y_ready": 32,
                }
                for name, expected in counter_expected.items():
                    if not bool((state[name] == expected).all()):
                        raise RuntimeError(
                            f"{name} mismatch rows={rows} N={compute_clusters}"
                        )
                if not bool((route_ready == 1).all()):
                    raise RuntimeError(
                        f"route flags did not close rows={rows} N={compute_clusters}"
                    )
                probes = int(state["next_reduce_probe"].item())
                if probes < 2 * total_tasks:
                    raise RuntimeError(
                        f"missing task-boundary probes rows={rows} "
                        f"N={compute_clusters}: {probes}"
                    )
                witness = int(state["overlap_witness"].item())
                required_witness = 0
                if rows > 64:
                    required_witness = 0b011
                    if compute_clusters == 1:
                        required_witness |= 0b100
                if witness & required_witness != required_witness:
                    raise RuntimeError(
                        f"overlap witness rows={rows} N={compute_clusters}: "
                        f"0x{witness:x} lacks 0x{required_witness:x}"
                    )

                print(
                    "TERMINAL_FULL_EXACT"
                    f"|rows={rows}|seed={seed}|compute_clusters={compute_clusters}"
                    f"|resident_clusters={resident_clusters}"
                    f"|invalid_routes={len(invalid_rows)}"
                    "|push_order=reverse_m64|candidate_launches=1"
                    f"|overlap_bits=0x{witness:x}|tasks={total_tasks}"
                    f"|probes={probes}|tokens={global_tokens}|result=PASS",
                    flush=True,
                )


def main() -> int:
    args = parse_args()
    check_source_contract()
    available = sm90_available()
    if args.device == "require" and not available:
        raise RuntimeError("SM90 nvcc/device path requested but unavailable")
    if args.device != "skip" and available:
        run_device(args)
    else:
        print("TERMINAL_FULL_DEVICE|result=SKIP|reason=unavailable", flush=True)
    print("TERMINAL_FULL_PROBE|result=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
