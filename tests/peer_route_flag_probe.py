#!/usr/bin/env python3
"""Four-rank terminal-helper peer publication/acquire probe.

The default path calls the production terminal header directly with symmetric
payload and route-ready pointers.  It covers all peer offsets, scrambled and
delayed route arrival, reset/reuse, one invalid route, one empty producer rank,
and a missing-flag bounded-failure leg.  The older phase-valued microprobe is
still exported by the extension as ``run`` but is no longer the acceptance
path for the terminal helper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=6)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--terminal-reuse", type=int, default=16)
    parser.add_argument("--delay-cycles", type=int, default=250_000)
    parser.add_argument("--spin-limit", type=int, default=1 << 22)
    return parser.parse_args()


def build_extension():
    source = Path(__file__).with_suffix(".cu")
    return load(
        name="mok_peer_route_flag_probe",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        verbose=True,
    )


def symmetric_tensor(shape: tuple[int, ...], dtype: torch.dtype, group_name: str):
    tensor = symm_mem.empty(*shape, dtype=dtype, device="cuda")
    tensor.zero_()
    return tensor, symm_mem.rendezvous(tensor, group_name)


def run_terminal_case(
    module,
    *,
    rank: int,
    world: int,
    offset: int,
    case: str,
    iterations: int,
    delay_cycles: int,
    spin_limit: int,
    routed_y: torch.Tensor,
    schedule_rank: torch.Tensor,
    schedule_slot: torch.Tensor,
    num_tokens: torch.Tensor,
    combine: torch.Tensor,
    combine_handle,
    route_ready: torch.Tensor,
    route_ready_handle,
    topk_ids: torch.Tensor,
    claim: torch.Tensor,
    mismatch: torch.Tensor,
    claim_count: torch.Tensor,
    timeout: torch.Tensor,
) -> tuple[int, int, int]:
    order = (5, 1, 3, 0, 4, 2)
    destination = (rank + offset) % world
    incoming_source = (rank - offset) % world
    totals = [0, 0, 0]

    for iteration in range(iterations):
        routed_y.zero_()
        combine.fill_(float("nan"))
        route_ready.zero_()
        topk_ids.copy_(torch.arange(6, dtype=torch.int32, device="cuda"))
        schedule_rank.fill_(destination)
        schedule_slot.copy_(
            torch.tensor(order, dtype=torch.int32, device="cuda")
        )
        num_tokens.fill_(6)
        claim.zero_()
        mismatch.zero_()
        claim_count.zero_()
        timeout.zero_()
        expected_source = incoming_source

        if case == "invalid":
            # Every producer has an invalid slot 3, and every destination
            # pre-marks exactly that slot ready while retaining stale NaN.
            schedule_rank[order.index(3)] = -1
            topk_ids[3] = -1
            route_ready[3] = 1
        elif case == "empty":
            if rank == 0:
                num_tokens.zero_()
            if incoming_source == 0:
                topk_ids.fill_(-1)
                route_ready.fill_(1)
                expected_source = -1
        elif case == "missing":
            # Rank zero owns the expected route but deliberately decodes it
            # invalid, so the exact production helper performs a no-op and
            # its destination must take the bounded NOT_READY path.
            if rank == 0:
                schedule_rank[order.index(3)] = -1
        elif case != "skew":
            raise ValueError(f"unknown terminal peer case: {case}")

        torch.cuda.synchronize()
        dist.barrier()
        module.run_terminal(
            routed_y,
            schedule_rank,
            schedule_slot,
            num_tokens,
            combine,
            route_ready,
            topk_ids,
            claim,
            mismatch,
            claim_count,
            timeout,
            [int(pointer) for pointer in combine_handle.buffer_ptrs],
            [int(pointer) for pointer in route_ready_handle.buffer_ptrs],
            rank,
            expected_source,
            iteration,
            0 if case == "skew" else -1,
            delay_cycles if case == "skew" else 0,
            spin_limit,
        )
        torch.cuda.synchronize()

        if case == "invalid" and not torch.isnan(combine[3]).all().item():
            raise RuntimeError("invalid peer route overwrote stale NaN storage")
        if case == "empty" and incoming_source == 0:
            if not torch.isnan(combine).all().item():
                raise RuntimeError("empty peer rank touched destination payload")

        counts = torch.stack(
            (mismatch[0], timeout[0], claim_count[0])
        ).to(torch.int64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        observed = [int(value) for value in counts.cpu().tolist()]
        if case == "missing":
            expected = [0, 1, world - 1]
        else:
            expected = [0, 0, world]
        if observed != expected:
            raise RuntimeError(
                f"terminal peer case={case} iteration={iteration} "
                f"observed={observed} expected={expected}"
            )
        totals = [left + right for left, right in zip(totals, observed)]
        dist.barrier()

    return tuple(totals)


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"probe requires exactly four ranks, got {world}")

    module = build_extension()
    symm_mem.enable_symm_mem_for_group(group.group_name)
    if args.routes != 6:
        raise ValueError("terminal helper is frozen to topk=6")
    if args.terminal_reuse <= 0 or args.delay_cycles < 0:
        raise ValueError("terminal-reuse must be positive and delay nonnegative")

    combine, combine_handle = symmetric_tensor(
        (args.routes, args.columns), torch.bfloat16, group.group_name
    )
    route_ready, route_ready_handle = symmetric_tensor(
        (args.routes,), torch.int32, group.group_name
    )
    routed_y = torch.empty(
        (args.routes, args.columns), dtype=torch.bfloat16, device="cuda"
    )
    schedule_rank = torch.empty(args.routes, dtype=torch.int32, device="cuda")
    schedule_slot = torch.empty(args.routes, dtype=torch.int32, device="cuda")
    num_tokens = torch.empty(1, dtype=torch.int32, device="cuda")
    topk_ids = torch.empty(args.routes, dtype=torch.int32, device="cuda")
    claim = torch.zeros(1, dtype=torch.int32, device="cuda")
    mismatch = torch.zeros(1, dtype=torch.int32, device="cuda")
    claim_count = torch.zeros(1, dtype=torch.int32, device="cuda")
    timeout = torch.zeros(1, dtype=torch.int32, device="cuda")

    positive_iterations = 0
    for offset in range(1, world):
        totals = run_terminal_case(
            module,
            rank=rank,
            world=world,
            offset=offset,
            case="skew",
            iterations=args.terminal_reuse,
            delay_cycles=args.delay_cycles,
            spin_limit=args.spin_limit,
            routed_y=routed_y,
            schedule_rank=schedule_rank,
            schedule_slot=schedule_slot,
            num_tokens=num_tokens,
            combine=combine,
            combine_handle=combine_handle,
            route_ready=route_ready,
            route_ready_handle=route_ready_handle,
            topk_ids=topk_ids,
            claim=claim,
            mismatch=mismatch,
            claim_count=claim_count,
            timeout=timeout,
        )
        if rank == 0:
            print(
                "TERMINAL_PEER_OFFSET"
                f"|offset={offset}|iterations={args.terminal_reuse}"
                f"|mismatch={totals[0]}|timeout={totals[1]}"
                f"|claims={totals[2]}|result=PASS",
                flush=True,
            )
        positive_iterations += args.terminal_reuse

    for case, iterations in (("invalid", 3), ("empty", 3), ("missing", 1)):
        totals = run_terminal_case(
            module,
            rank=rank,
            world=world,
            offset=1,
            case=case,
            iterations=iterations,
            delay_cycles=args.delay_cycles,
            spin_limit=(
                max(1024, args.spin_limit // 16)
                if case == "missing" else args.spin_limit
            ),
            routed_y=routed_y,
            schedule_rank=schedule_rank,
            schedule_slot=schedule_slot,
            num_tokens=num_tokens,
            combine=combine,
            combine_handle=combine_handle,
            route_ready=route_ready,
            route_ready_handle=route_ready_handle,
            topk_ids=topk_ids,
            claim=claim,
            mismatch=mismatch,
            claim_count=claim_count,
            timeout=timeout,
        )
        if rank == 0:
            print(
                "TERMINAL_PEER_CASE"
                f"|case={case}|iterations={iterations}"
                f"|mismatch={totals[0]}|timeout={totals[1]}"
                f"|claims={totals[2]}|result=PASS",
                flush=True,
            )

    if rank == 0:
        print(
            "TERMINAL_PEER_ROUTE_FLAG_PROBE"
            f"|positive_iterations={positive_iterations}"
            f"|positive_payload_bytes_per_rank="
            f"{positive_iterations * args.routes * args.columns * 2}"
            "|peer_offsets=3|route_skew=1|empty_rank=1|invalid_route=1"
            "|missing_flag_bounded=1|reuse_reset_barrier=1|result=PASS",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
