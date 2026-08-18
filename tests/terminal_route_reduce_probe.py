#!/usr/bin/env python3
"""Terminal route publication/claim/reduce contract and SM90 device probe.

The device leg is a single-GPU peer-pointer emulation.  One persistent
producer block publishes BF16 rows in out-of-order route order while two
helper blocks race ready-token claims.  It is intentionally not a scheduler:
the production header owns only publication, one-shot readiness, CAS claim,
and reduction primitives.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cu")
HEADER = REPO_ROOT / "csrc" / "sm90_fp8_block_terminal_route_flags.cuh"
COMM_HEADER = REPO_ROOT / "csrc" / "sm90_fp8_block_terminal_comm_primitives.cuh"
PIPELINE_HEADER = REPO_ROOT / "csrc" / "sm90_fp8_block_pipeline_primitives.cuh"

TOPK = 6
TOKENS = 4
SCHEDULE_ROWS = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "skip", "require"), default="auto")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--race-rounds", type=int, default=1000)
    parser.add_argument("--spin-limit", type=int, default=1 << 24)
    return parser.parse_args()


def function_body(text: str, name: str, next_name: str) -> str:
    start = text.find(name)
    end = text.find(next_name, start + len(name))
    if start < 0 or end < 0:
        raise RuntimeError(f"could not isolate {name}")
    return text[start:end]


def check_source_contract() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    comm = COMM_HEADER.read_text(encoding="utf-8")
    pipeline = PIPELINE_HEADER.read_text(encoding="utf-8")

    required_ptx = (
        "fence.release.sys",
        "st.release.sys.global.u32",
        "ld.acquire.sys.global.u32",
    )
    missing_ptx = [needle for needle in required_ptx if needle not in header]
    if missing_ptx:
        raise RuntimeError(f"missing system-scope PTX contract: {missing_ptx}")

    publish = function_body(
        header, "push_routed_row_and_publish", "all_routes_ready_once"
    )
    publish_order = (
        publish.find("comm::push_routed_row"),
        publish.find("release_fence_system"),
        publish.find("__syncwarp"),
        publish.find("store_release_system"),
    )
    if any(index < 0 for index in publish_order) or list(publish_order) != sorted(
        publish_order
    ):
        raise RuntimeError(
            "route publication must be push -> per-lane sys fence -> "
            "warp convergence -> elected release store"
        )
    if "if (!route.valid)" not in publish or "if (lane == 0)" not in publish:
        raise RuntimeError("invalid-route no-op or elected-lane publication is missing")

    ready = function_body(header, "all_routes_ready_once", "try_claim_ready_token")
    if "route < TOPK" not in ready or "load_acquire_system" not in ready:
        raise RuntimeError("ready probe does not acquire-load all fixed top-6 flags")
    if re.search(r"return\s+false", ready):
        raise RuntimeError("ready probe has an early return before all six acquires")

    claim = function_body(
        header, "try_claim_ready_token", "weighted_reduce_valid_element"
    )
    claim_order = (
        claim.find("all_routes_ready_once"),
        claim.find("claim_token_after_ready"),
    )
    if any(index < 0 for index in claim_order) or claim_order[0] >= claim_order[1]:
        raise RuntimeError("CAS must occur only after the six-flag ready probe")
    if "while" in claim or "__nanosleep" in claim:
        raise RuntimeError("destination claim helper must not spin on an unready token")
    claim_step = function_body(
        header, "claim_token_after_ready", "release_fence_system"
    )
    if "atomicCAS" not in claim_step:
        raise RuntimeError("factored ready-token claim is not an atomic CAS")

    reduce = function_body(
        header, "weighted_reduce_valid_element", "reduce_claimed_token"
    )
    if "topk_ids[route_index] >= 0" not in reduce:
        raise RuntimeError("invalid route skip is missing")
    if reduce.find("topk_ids[route_index] >= 0") >= reduce.find(
        "combine[route_index * hidden + column]"
    ):
        raise RuntimeError("invalid combine storage may be read before validity is known")
    if "pipeline::weighted_reduce_element" not in reduce:
        raise RuntimeError("terminal reducer is not using the shared arithmetic core")
    if "__float2bfloat16_rn(0.0f)" not in reduce:
        raise RuntimeError("all-invalid token does not explicitly write BF16 +0")
    if "Do not canonicalize signed zero" not in reduce:
        raise RuntimeError("valid-route signed-zero preservation contract is missing")

    forbidden_header = (
        "__syncthreads(",
        "multimem.",
        "atomicAdd(",
        "ticket_counter",
        "completion_counter",
        "barrier_target",
        "next_logical_cluster",
        "__nanosleep(",
    )
    leaked = [needle for needle in forbidden_header if needle in header]
    if leaked:
        raise RuntimeError(f"scheduler/full-rank barrier leaked into primitive: {leaked}")

    cross_file_required = (
        ("comm row store", "push_routed_row", comm),
        ("shared weighted reduce", "weighted_reduce_element", pipeline),
        ("probe publication", "push_routed_row_and_publish", source),
        ("probe one-shot claim", "try_claim_ready_token", source),
        ("probe overlap witness", "overlap_count", source),
    )
    missing = [label for label, needle, text in cross_file_required if needle not in text]
    if missing:
        raise RuntimeError(f"route primitive wiring missing: {missing}")

    print(
        "TERMINAL_ROUTE_SOURCE"
        "|writer_fence=sys_release|convergence=warp"
        "|flag_store=sys_release|flag_load=sys_acquire"
        "|topk_acquires=6|claim=cas_after_ready|unready_spin=0"
        "|full_rank_barrier=0|generic_queue=0|result=PASS",
        flush=True,
    )


def sm90_device_available() -> bool:
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


def build_extension(verbose: bool):
    from torch.utils.cpp_extension import load

    return load(
        name="mok_terminal_route_reduce_probe",
        sources=[str(SOURCE)],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--use_fast_math",
            "-Xptxas=-v",
        ],
        verbose=verbose,
    )


def first_sm90_device() -> int:
    import torch

    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index) == (9, 0):
            return index
    raise RuntimeError("no SM90 device found")


def make_inputs(hidden: int, seed: int, negative_zero: bool):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    routed_y = (
        torch.randn(
            (SCHEDULE_ROWS, hidden),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 3.0
    ).to(torch.bfloat16)
    # Token 3 exercises both signed-zero legs.  Valid routes retain the
    # production mul/FMA sign bit; only all-invalid token 2 is forced to +0.
    routed_y[11:].fill_(-0.0 if negative_zero else 0.0)

    schedule_peer = torch.zeros(
        SCHEDULE_ROWS, dtype=torch.int32, device="cuda"
    )
    schedule_slot = torch.tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,  # token 0
            6,
            7,
            8,
            10,
            11,  # token 1, slot 3 is invalid
            18,
            19,
            20,
            21,
            22,
            23,  # token 3; token 2 is all invalid
        ],
        dtype=torch.int32,
        device="cuda",
    )
    num_tokens = torch.tensor(
        [SCHEDULE_ROWS], dtype=torch.int32, device="cuda"
    )

    combine = torch.full(
        (TOKENS * TOPK, hidden),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    route_ready = torch.zeros(
        (TOKENS, TOPK), dtype=torch.int32, device="cuda"
    )
    topk_ids = torch.arange(
        TOKENS * TOPK, dtype=torch.int32, device="cuda"
    ).reshape(TOKENS, TOPK)
    topk_ids[1, 3] = -1
    topk_ids[2].fill_(-1)
    route_ready[1, 3] = 1
    route_ready[2].fill_(1)

    weights = torch.rand(
        (TOKENS, TOPK),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    weights[1, 3] = 0.0
    weights[2].zero_()
    for token in (0, 1, 3):
        weights[token] /= weights[token].sum()

    output = torch.full(
        (TOKENS, hidden),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    reference = torch.empty_like(output)
    counters = {
        "claim": torch.zeros(TOKENS, dtype=torch.int32, device="cuda"),
        "reduce_count": torch.zeros(TOKENS, dtype=torch.int32, device="cuda"),
        "not_ready": torch.zeros(TOKENS, dtype=torch.int32, device="cuda"),
        "early_claim": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "ready_snapshot": torch.zeros(TOKENS, dtype=torch.int32, device="cuda"),
        "total_done": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "arrival_epoch": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "observed_epoch": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "reduced_signal": torch.zeros(TOKENS, dtype=torch.int32, device="cuda"),
        "overlap_count": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "timeout_count": torch.zeros(1, dtype=torch.int32, device="cuda"),
    }
    return (
        routed_y,
        schedule_peer,
        schedule_slot,
        num_tokens,
        combine,
        route_ready,
        topk_ids,
        weights,
        output,
        reference,
        counters,
    )


def run_device(
    hidden: int, seeds: int, race_rounds: int,
    spin_limit: int,
) -> None:
    import torch

    device = first_sm90_device()
    torch.cuda.set_device(device)
    # Keep the build verbose: -Xptxas=-v is part of the acceptance evidence,
    # not optional troubleshooting output.
    module = build_extension(True)

    attrs = [int(value) for value in module.kernel_attributes()]
    print(
        "TERMINAL_ROUTE_PTXAS"
        f"|route_registers={attrs[0]}|route_local_bytes={attrs[1]}"
        f"|race_registers={attrs[2]}|race_local_bytes={attrs[3]}"
        "|spill_evidence=ptxas_verbose_build_log|result=INFO",
        flush=True,
    )

    race_ready = torch.ones(TOPK, dtype=torch.int32, device="cuda")
    race = {
        "claim": torch.zeros(race_rounds, dtype=torch.int32, device="cuda"),
        "arrivals": torch.zeros(race_rounds, dtype=torch.int32, device="cuda"),
        "claimed": torch.zeros(race_rounds, dtype=torch.int32, device="cuda"),
        "already": torch.zeros(race_rounds, dtype=torch.int32, device="cuda"),
        "unexpected": torch.zeros(race_rounds, dtype=torch.int32, device="cuda"),
        "timeout": torch.zeros(1, dtype=torch.int32, device="cuda"),
    }
    module.run_claim_race(
        race_ready,
        race["claim"],
        race["arrivals"],
        race["claimed"],
        race["already"],
        race["unexpected"],
        race["timeout"],
        race_rounds,
        spin_limit,
    )
    torch.cuda.synchronize()
    expected_ones = torch.ones(race_rounds, dtype=torch.int32, device="cuda")
    expected_twos = torch.full(
        (race_rounds,), 2, dtype=torch.int32, device="cuda"
    )
    if not torch.equal(race["claim"], expected_ones):
        raise RuntimeError("claim-race ownership state was not exactly once")
    if not torch.equal(race["arrivals"], expected_twos):
        raise RuntimeError("both claim-race CTAs did not reach every CAS gate")
    if not torch.equal(race["claimed"], expected_ones):
        raise RuntimeError("claim-race did not produce one winner per round")
    if not torch.equal(race["already"], expected_ones):
        raise RuntimeError("claim-race did not produce one CAS loser per round")
    if torch.count_nonzero(race["unexpected"]).item() != 0:
        raise RuntimeError("claim-race returned an unexpected result")
    if int(race["timeout"].item()) != 0:
        raise RuntimeError("claim-race gate timed out")
    print(
        "TERMINAL_ROUTE_CAS_RACE"
        f"|rounds={race_rounds}|contenders=2"
        f"|claimed={race_rounds}|already_claimed={race_rounds}"
        "|result=PASS",
        flush=True,
    )

    for seed, negative_zero in (
        (seed, negative_zero)
        for seed in range(seeds)
        for negative_zero in (False, True)
    ):
        (
            routed_y,
            schedule_peer,
            schedule_slot,
            num_tokens,
            combine,
            route_ready,
            topk_ids,
            weights,
            output,
            reference,
            counters,
        ) = make_inputs(hidden, seed, negative_zero)
        module.run_probe(
            routed_y,
            schedule_peer,
            schedule_slot,
            num_tokens,
            combine,
            route_ready,
            topk_ids,
            weights,
            output,
            reference,
            counters["claim"],
            counters["reduce_count"],
            counters["not_ready"],
            counters["early_claim"],
            counters["ready_snapshot"],
            counters["total_done"],
            counters["arrival_epoch"],
            counters["observed_epoch"],
            counters["reduced_signal"],
            counters["overlap_count"],
            counters["timeout_count"],
            spin_limit,
        )
        torch.cuda.synchronize()

        def values(name: str) -> list[int]:
            return [int(value) for value in counters[name].cpu().tolist()]

        if values("timeout_count") != [0]:
            raise RuntimeError(f"bounded probe timed out for seed {seed}")
        if values("early_claim") != [0]:
            raise RuntimeError(f"token was claimed before all routes for seed {seed}")
        if values("claim") != [1] * TOKENS:
            raise RuntimeError(f"claim state is not exactly-once: {values('claim')}")
        if values("reduce_count") != [1] * TOKENS:
            raise RuntimeError(
                f"reduce execution is not exactly-once: {values('reduce_count')}"
            )
        if values("total_done") != [TOKENS]:
            raise RuntimeError(f"unexpected completion count: {values('total_done')}")
        if values("ready_snapshot") != [(1 << TOPK) - 1] * TOKENS:
            raise RuntimeError(
                f"claim observed incomplete route flags: {values('ready_snapshot')}"
            )
        if values("reduced_signal") != [1] * TOKENS:
            raise RuntimeError("not every claimed reduction published completion")
        if values("overlap_count") != [1]:
            raise RuntimeError("no reduction completed while later push work remained")
        if values("not_ready")[0] < TOPK - 1:
            raise RuntimeError(
                "out-of-order token was not probed once per partial arrival"
            )
        if not torch.all(route_ready == 1):
            raise RuntimeError("valid route publication did not close every flag")

        exact = torch.equal(output.view(torch.uint16), reference.view(torch.uint16))
        mismatch = int(
            (output.view(torch.uint16) != reference.view(torch.uint16)).sum().item()
        )
        if not exact:
            raise RuntimeError(
                f"weighted reduce mismatch seed={seed} count={mismatch}"
            )
        if torch.isnan(output[1]).any():
            raise RuntimeError("invalid stale-NaN route contaminated token 1")
        if not torch.isnan(combine[1 * TOPK + 3]).all():
            raise RuntimeError("invalid token-1 combine row was overwritten")
        if not torch.isnan(combine[2 * TOPK : 3 * TOPK]).all():
            raise RuntimeError("all-invalid token combine rows were touched")
        if torch.count_nonzero(output[2].view(torch.uint16)).item() != 0:
            raise RuntimeError("all-invalid padded token did not produce BF16 +0")
        zero_bits = 0x8000 if negative_zero else 0x0000
        if not torch.all(output[3].view(torch.uint16) == zero_bits):
            raise RuntimeError(
                "valid signed-zero output diverged from production mul/FMA semantics"
            )
        if not torch.all(
            combine[3 * TOPK : 4 * TOPK].view(torch.uint16) == zero_bits
        ):
            raise RuntimeError("valid signed-zero payload bits were not preserved")
        for row, slot in enumerate(schedule_slot.cpu().tolist()):
            if not torch.equal(
                combine[slot].view(torch.uint16),
                routed_y[row].view(torch.uint16),
            ):
                raise RuntimeError(f"published payload mismatch at schedule row {row}")

        print(
            "TERMINAL_ROUTE_DEVICE"
            f"|seed={seed}|hidden={hidden}"
            f"|input_zero_sign={'negative' if negative_zero else 'positive'}"
            "|out_of_order=1"
            f"|partial_not_ready={values('not_ready')[0]}"
            "|stale_nan_not_read=1|signed_zero=production_exact"
            "|all_invalid_poszero=1|exactly_once=1|overlap=1"
            "|numeric=bitwise_exact|result=PASS",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.hidden <= 0 or args.hidden % 8:
        raise ValueError("hidden must be a positive multiple of 8")
    if args.seeds <= 0 or args.spin_limit <= 0:
        raise ValueError("seeds and spin-limit must be positive")
    if args.race_rounds < 1000:
        raise ValueError("race-rounds must be at least 1000")

    check_source_contract()
    available = sm90_device_available()
    if args.device == "require" and not available:
        raise RuntimeError("SM90 CUDA build/device probe was required but unavailable")
    if args.device != "skip" and available:
        run_device(
            args.hidden,
            args.seeds,
            args.race_rounds,
            args.spin_limit,
        )
    else:
        print(
            "TERMINAL_ROUTE_DEVICE|result=SKIP"
            "|reason=sm90_cuda_or_torch_unavailable",
            flush=True,
        )
    print(
        "TERMINAL_ROUTE_MULTI_RANK|result=GAP"
        "|reason=run_tests_peer_route_flag_probe_py_separately",
        flush=True,
    )
    print("TERMINAL_ROUTE_REDUCE_PROBE|result=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
