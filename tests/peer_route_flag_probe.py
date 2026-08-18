#!/usr/bin/env python3
"""Four-rank peer payload/release-flag/acquire visibility probe.

Run with one rank per GPU.  The positive leg rotates through every non-local
peer offset for 999 total iterations.  The negative leg omits one route flag
and must terminate through the bounded timeout counter rather than hang.
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
    parser.add_argument("--iterations-per-peer", type=int, default=333)
    parser.add_argument("--spin-limit", type=int, default=1 << 22)
    return parser.parse_args()


def build_extension():
    source = Path(__file__).with_suffix(".cu")
    return load(
        name="mok_peer_route_flag_probe",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def symmetric_tensor(shape: tuple[int, ...], dtype: torch.dtype, group_name: str):
    tensor = symm_mem.empty(*shape, dtype=dtype, device="cuda")
    tensor.zero_()
    return tensor, symm_mem.rendezvous(tensor, group_name)


def reset_phase(*tensors: torch.Tensor) -> None:
    torch.cuda.synchronize()
    for tensor in tensors:
        tensor.zero_()
    dist.barrier()


def reduced_counts(mismatch: torch.Tensor, timeout: torch.Tensor) -> tuple[int, int]:
    counts = torch.stack((mismatch[0], timeout[0])).to(torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return int(counts[0].item()), int(counts[1].item())


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
    payload, payload_handle = symmetric_tensor(
        (args.routes, args.columns), torch.int16, group.group_name
    )
    flags, flags_handle = symmetric_tensor(
        (args.routes,), torch.int32, group.group_name
    )
    ack, ack_handle = symmetric_tensor(
        (args.routes,), torch.int32, group.group_name
    )
    mismatch = torch.zeros(1, dtype=torch.int32, device="cuda")
    timeout = torch.zeros(1, dtype=torch.int32, device="cuda")

    positive_iterations = 0
    for offset in range(1, world):
        reset_phase(payload, flags, ack, mismatch, timeout)
        destination = (rank + offset) % world
        source = (rank - offset) % world
        module.run(
            payload,
            flags,
            ack,
            int(payload_handle.buffer_ptrs[destination]),
            int(flags_handle.buffer_ptrs[destination]),
            int(ack_handle.buffer_ptrs[source]),
            mismatch,
            timeout,
            rank,
            source,
            args.routes,
            args.columns,
            args.iterations_per_peer,
            -1,
            -1,
            args.spin_limit,
        )
        torch.cuda.synchronize()
        positive_mismatch, positive_timeout = reduced_counts(mismatch, timeout)
        if rank == 0:
            print(
                "PEER_FLAG_POSITIVE"
                f"|offset={offset}|iterations={args.iterations_per_peer}"
                f"|mismatch={positive_mismatch}|timeout={positive_timeout}",
                flush=True,
            )
        if positive_mismatch != 0 or positive_timeout != 0:
            return 1
        positive_iterations += args.iterations_per_peer

    reset_phase(payload, flags, ack, mismatch, timeout)
    destination = (rank + 1) % world
    source = (rank - 1) % world
    module.run(
        payload,
        flags,
        ack,
        int(payload_handle.buffer_ptrs[destination]),
        int(flags_handle.buffer_ptrs[destination]),
        int(ack_handle.buffer_ptrs[source]),
        mismatch,
        timeout,
        rank,
        source,
        args.routes,
        args.columns,
        1,
        0,
        3,
        max(1024, args.spin_limit // 16),
    )
    torch.cuda.synchronize()
    negative_mismatch, negative_timeout = reduced_counts(mismatch, timeout)
    ok = negative_mismatch == 0 and negative_timeout > 0
    if rank == 0:
        print(
            "PEER_ROUTE_FLAG_PROBE"
            f"|positive_iterations={positive_iterations}"
            f"|positive_payload_bytes_per_rank="
            f"{positive_iterations * args.routes * args.columns * 2}"
            f"|negative_mismatch={negative_mismatch}"
            f"|negative_timeout={negative_timeout}"
            f"|result={'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
