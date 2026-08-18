#!/usr/bin/env python3
"""Four-rank asymmetric coarse-M64 production and ready-only reduce probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem
from torch.utils.cpp_extension import load


ROUTES = 4
TOKENS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="58,117")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--delay-cycles", default="0,20000")
    parser.add_argument("--poll-limit", type=int, default=1 << 24)
    parser.add_argument("--negative-poll-limit", type=int, default=1 << 16)
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def build_extension():
    return load(
        name="mok_coarse_route_scheduler_probe",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


def symmetric_tensor(shape: tuple[int, ...], group_name: str):
    tensor = symm_mem.empty(*shape, dtype=torch.int32, device="cuda")
    tensor.zero_()
    return tensor, symm_mem.rendezvous(tensor, group_name)


def rotations(values: list[int]) -> list[list[int]]:
    return [values[offset:] + values[:offset] for offset in range(len(values))]


def scenarios() -> list[tuple[str, list[int]]]:
    rows = [("balanced", [64, 64, 64, 64])]
    rows.extend(
        (f"one-empty-rot{offset}", value)
        for offset, value in enumerate(rotations([0, 1, 64, 191]))
    )
    rows.extend(
        (f"three-empty-rot{offset}", value)
        for offset, value in enumerate(rotations([0, 0, 0, 256]))
    )
    return rows


def zeros(size: int) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.int32, device="cuda")


def reset_tensors(tensors: list[torch.Tensor]) -> None:
    torch.cuda.synchronize()
    for tensor in tensors:
        tensor.zero_()
    dist.barrier()


def local_validation(
    *,
    local_tiles: int,
    tile_head: torch.Tensor,
    claims: torch.Tensor,
    reduced_count: torch.Tensor,
    timeout_count: torch.Tensor,
    mismatch_count: torch.Tensor,
    visits: list[torch.Tensor],
    reduce_visits: torch.Tensor,
    flags: torch.Tensor,
) -> list[str]:
    errors = []
    if int(tile_head.item()) != local_tiles:
        errors.append(f"head={int(tile_head.item())}!={local_tiles}")
    if int(reduced_count.item()) != TOKENS:
        errors.append(f"reduced={int(reduced_count.item())}!={TOKENS}")
    if int(timeout_count.item()) != 0:
        errors.append(f"timeout={int(timeout_count.item())}")
    if int(mismatch_count.item()) != 0:
        errors.append(f"mismatch={int(mismatch_count.item())}")
    if int(claims.min().item()) != 1 or int(claims.max().item()) != 1:
        errors.append("claims-not-exactly-once")
    if (
        int(reduce_visits.min().item()) != 1
        or int(reduce_visits.max().item()) != 1
    ):
        errors.append("reduce-not-exactly-once")
    if int(flags.min().item()) != 1 or int(flags.max().item()) != 1:
        errors.append("route-flags-incomplete")
    for stage, tensor in zip(("copy", "w13", "act", "w2", "push"), visits):
        if local_tiles and (
            int(tensor.min().item()) != 2 or int(tensor.max().item()) != 2
        ):
            errors.append(f"{stage}-visits-not-two")
    return errors


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"probe requires four ranks, got {world}")

    module = build_extension()
    occupancy = int(module.max_active_clusters(local_rank))
    attributes = [int(value) for value in module.kernel_attributes(local_rank)]
    cluster_values = csv_ints(args.clusters)
    if max(cluster_values) > occupancy:
        raise RuntimeError(
            f"requested clusters {max(cluster_values)} exceed {occupancy}"
        )
    if rank == 0:
        print(
            "COARSE_ROUTE_RESOURCES"
            f"|occupancy={occupancy}|registers={attributes[0]}"
            f"|static_smem={attributes[1]}|local_bytes={attributes[2]}"
            f"|max_threads={attributes[3]}|binary={attributes[4]}"
            f"|ptx={attributes[5]}",
            flush=True,
        )

    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    payload, payload_handle = symmetric_tensor(
        (TOKENS, ROUTES), dist.group.WORLD.group_name
    )
    flags, flags_handle = symmetric_tensor(
        (TOKENS, ROUTES), dist.group.WORLD.group_name
    )
    peer_payload_ptrs = torch.tensor(
        [int(ptr) for ptr in payload_handle.buffer_ptrs],
        dtype=torch.int64,
        device="cuda",
    )
    peer_flag_ptrs = torch.tensor(
        [int(ptr) for ptr in flags_handle.buffer_ptrs],
        dtype=torch.int64,
        device="cuda",
    )

    result_rows = []
    positive_kernels = 0
    for scenario_name, tiles in scenarios():
        if sum(tiles) * 64 != world * TOKENS * ROUTES:
            raise RuntimeError(f"invalid route cardinality for {scenario_name}")
        tiles_by_rank = torch.tensor(tiles, dtype=torch.int32, device="cuda")
        local_tiles = tiles[rank]
        delay_rank = max(range(world), key=tiles.__getitem__)
        for clusters in cluster_values:
            descriptor = zeros(clusters * 2)
            poll_counts = zeros(clusters)
            visits = [zeros(local_tiles) for _ in range(5)]
            tile_head, scan_cursor = zeros(1), zeros(1)
            claims, reduce_visits = zeros(TOKENS), zeros(TOKENS)
            reduced_count, timeout_count, mismatch_count = (
                zeros(1),
                zeros(1),
                zeros(1),
            )
            resettable = [
                payload,
                flags,
                tile_head,
                scan_cursor,
                claims,
                reduce_visits,
                reduced_count,
                timeout_count,
                mismatch_count,
                descriptor,
                poll_counts,
                *visits,
            ]
            for delay_cycles in csv_ints(args.delay_cycles):
                times = []
                digest = hashlib.sha256()
                for repeat in range(args.repeats):
                    reset_tensors(resettable)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    module.run(
                        payload,
                        flags,
                        peer_payload_ptrs,
                        peer_flag_ptrs,
                        tiles_by_rank,
                        tile_head,
                        scan_cursor,
                        claims,
                        reduced_count,
                        timeout_count,
                        mismatch_count,
                        descriptor,
                        poll_counts,
                        *visits,
                        reduce_visits,
                        rank,
                        delay_rank,
                        delay_cycles,
                        -1,
                        args.poll_limit,
                    )
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = torch.tensor(
                        [start.elapsed_time(end)], dtype=torch.float64,
                        device="cuda",
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    errors = local_validation(
                        local_tiles=local_tiles,
                        tile_head=tile_head,
                        claims=claims,
                        reduced_count=reduced_count,
                        timeout_count=timeout_count,
                        mismatch_count=mismatch_count,
                        visits=visits,
                        reduce_visits=reduce_visits,
                        flags=flags,
                    )
                    failed = torch.tensor(
                        [int(bool(errors))], dtype=torch.int32, device="cuda"
                    )
                    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                    if int(failed.item()):
                        raise RuntimeError(
                            f"{scenario_name} repeat={repeat} rank={rank}: {errors}"
                        )
                    if rank == 0:
                        times.append(float(elapsed.item()))
                        digest.update(reduce_visits.cpu().numpy().tobytes())
                    positive_kernels += 1
                if rank == 0:
                    ordered = sorted(times)
                    row = {
                        "scenario": scenario_name,
                        "tiles": tiles,
                        "clusters": clusters,
                        "delay_cycles": delay_cycles,
                        "repeats": args.repeats,
                        "p50_ms": round(statistics.median(ordered), 6),
                        "p95_ms": round(
                            ordered[max(0, int(len(ordered) * 0.95) - 1)], 6
                        ),
                        "sha16": digest.hexdigest()[:16],
                    }
                    result_rows.append(row)
                    print(
                        "COARSE_ROUTE_PASS|"
                        + "|".join(f"{key}={value}" for key, value in row.items()),
                        flush=True,
                    )

    # A single omitted route must leave exactly its destination incomplete and
    # terminate by bounded polling instead of hanging the other three ranks.
    tiles = [64, 64, 64, 64]
    tiles_by_rank = torch.tensor(tiles, dtype=torch.int32, device="cuda")
    local_tiles = tiles[rank]
    clusters = max(cluster_values)
    descriptor, poll_counts = zeros(clusters * 2), zeros(clusters)
    visits = [zeros(local_tiles) for _ in range(5)]
    tile_head, scan_cursor = zeros(1), zeros(1)
    claims, reduce_visits = zeros(TOKENS), zeros(TOKENS)
    reduced_count, timeout_count, mismatch_count = zeros(1), zeros(1), zeros(1)
    reset_tensors(
        [payload, flags, tile_head, scan_cursor, claims, reduce_visits,
         reduced_count, timeout_count, mismatch_count, descriptor,
         poll_counts, *visits]
    )
    module.run(
        payload,
        flags,
        peer_payload_ptrs,
        peer_flag_ptrs,
        tiles_by_rank,
        tile_head,
        scan_cursor,
        claims,
        reduced_count,
        timeout_count,
        mismatch_count,
        descriptor,
        poll_counts,
        *visits,
        reduce_visits,
        rank,
        0,
        0,
        0,
        args.negative_poll_limit,
    )
    torch.cuda.synchronize()
    local_negative = {
        "rank": rank,
        "head": int(tile_head.item()),
        "reduced": int(reduced_count.item()),
        "timeout": int(timeout_count.item()),
        "mismatch": int(mismatch_count.item()),
        "missing_flags": int((flags == 0).sum().item()),
    }
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_negative)
    negative_ok = all(row["head"] == 64 for row in gathered)
    negative_ok &= gathered[0]["timeout"] > 0
    negative_ok &= gathered[0]["missing_flags"] == 1
    negative_ok &= gathered[0]["reduced"] == TOKENS - 1
    negative_ok &= all(
        row["timeout"] == 0
        and row["missing_flags"] == 0
        and row["reduced"] == TOKENS
        and row["mismatch"] == 0
        for row in gathered[1:]
    )
    if rank == 0:
        print(
            "COARSE_ROUTE_NEGATIVE|records="
            + json.dumps(gathered, sort_keys=True)
            + f"|result={'PASS' if negative_ok else 'FAIL'}",
            flush=True,
        )
        print(
            "COARSE_ROUTE_SCHEDULER_PROBE"
            f"|positive_kernels_all_ranks={positive_kernels * world}"
            f"|cells={len(result_rows)}|negative=1"
            f"|result={'PASS' if negative_ok else 'FAIL'}",
            flush=True,
        )
        print("COARSE_ROUTE_JSON=" + json.dumps(result_rows, sort_keys=True))
    result = torch.tensor([int(negative_ok)], dtype=torch.int32, device="cuda")
    dist.broadcast(result, src=0)
    dist.destroy_process_group()
    return 0 if int(result.item()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
