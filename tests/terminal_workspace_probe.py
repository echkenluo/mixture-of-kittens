#!/usr/bin/env python3
"""Four-rank allocation probe for the terminal FP8 workspace."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from mok.functional import create_fp8_terminal_workspace


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        workspace = create_fp8_terminal_workspace(
            dist.group.WORLD,
            device=torch.device("cuda", local_rank),
            num_local_tokens=64,
            schedule_capacity=384,
            num_local_experts=64,
            compute_clusters=7,
        )
        if workspace.ep_size != 4 or workspace.ep_rank != dist.get_rank():
            raise RuntimeError("terminal workspace rank identity mismatch")
        if workspace.padded_num_local_tokens != 64:
            raise RuntimeError("terminal workspace token padding mismatch")
        if tuple(workspace.combine_buffer.shape) != (384, 4096):
            raise RuntimeError("terminal combine shape mismatch")
        if tuple(workspace.route_ready.shape) != (64, 6):
            raise RuntimeError("terminal route-ready shape mismatch")
        if tuple(workspace.gate_up_tile_ready.shape) != (6, 16):
            raise RuntimeError("terminal W13 readiness shape mismatch")
        if tuple(workspace.schedule_peer_rank.shape) != (384,):
            raise RuntimeError("terminal schedule capacity mismatch")
        if tuple(workspace.schedule_tokens_per_expert.shape) != (64,):
            raise RuntimeError("terminal expert-count schedule mismatch")
        if tuple(workspace.all_gather_top_experts_buffer.shape) != (4, 64, 6):
            raise RuntimeError("terminal route all-gather shape mismatch")
        if workspace.all_gather_top_experts_buffer_multicast_ptr <= 0:
            raise RuntimeError("terminal route all-gather multicast is null")
        if workspace.worker_ticket.numel() != 7:
            raise RuntimeError("terminal worker-ticket shape mismatch")
        if tuple(workspace.comm_owner.shape) != (1,):
            raise RuntimeError("terminal comm-owner shape mismatch")
        if int(workspace.comm_owner.item()) != -1:
            raise RuntimeError("terminal comm owner must start unclaimed")
        if tuple(workspace.comm_worker_ticket.shape) != (1,):
            raise RuntimeError("terminal comm-worker-ticket shape mismatch")
        if int(workspace.comm_worker_ticket.item()) != 0:
            raise RuntimeError("terminal comm worker ticket must start cleared")
        if not workspace.trap_record.is_pinned() or workspace.trap_record.is_cuda:
            raise RuntimeError("terminal trap record is not host-mapped pinned memory")

        pointer_groups = (
            workspace.x_buffer_ptrs,
            workspace.x_scale_buffer_ptrs,
            workspace.combine_buffer_ptrs,
            workspace.route_ready_ptrs,
            workspace.barrier_buffer_ptrs,
        )
        if any(len(pointers) != 4 for pointers in pointer_groups):
            raise RuntimeError("terminal symmetric peer pointer list mismatch")
        if any(pointer <= 0 for pointers in pointer_groups for pointer in pointers):
            raise RuntimeError("terminal symmetric peer pointer is null")

        passed = torch.ones(1, dtype=torch.int32, device="cuda")
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)
        if int(passed.item()) != 4:
            raise RuntimeError("terminal workspace four-rank closure mismatch")
        if dist.get_rank() == 0:
            print(
                "TERMINAL_WORKSPACE_EP4"
                "|hidden=4096|intermediate=2048|topk=6"
                "|tokens=64|capacity=384|experts=64|compute_clusters=7"
                "|symmetric_inputs=1|symmetric_route=1|result=PASS",
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
