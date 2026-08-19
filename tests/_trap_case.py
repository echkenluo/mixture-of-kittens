"""Destructive trap-negative worker (Stage C).  Run standalone (reentrant)
or under torchrun (contract/timeout); a device trap poisons the CUDA
context, so each case lives in its own process tree.  On the expected trap
the process prints the committed MOK_TRAP record (CPU-only read of the
host-mapped pinned memory) and exits 70, mirroring the production
fatal-error boundary."""

import os
import sys

import torch


def _die(record: torch.Tensor) -> None:
    head = int(record[0].item())
    if head == 0 or head == -1:
        # Error was not a committed MoK trap -- fail the gate loudly.
        print("TRAP_GATE_NO_RECORD", flush=True)
        os._exit(64)
    rec = record.tolist()
    print(
        "MOK_TRAP|code=%d|site=%d|slot=%d|expected=%d|observed=%d"
        "|rank=%d|ticket=%d|iters=%d" % tuple(rec),
        flush=True,
    )
    os._exit(70)


def case_reentrant() -> None:
    from mok import ops

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    in_use = torch.zeros(1, dtype=torch.int32, device=device)
    record = torch.zeros(8, dtype=torch.int64).pin_memory()
    # Concurrent contention: the first acquire holds the lease on stream A;
    # a SECOND stream races its own acquire with no release in between --
    # the cross-stream loser must fail closed with the REENTRANT trap.
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream_a):
        ops.workspace_lease_acquire(in_use, record.data_ptr(), 3)
    with torch.cuda.stream(stream_b):
        ops.workspace_lease_acquire(in_use, record.data_ptr(), 3)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        _die(record)
    print("TRAP_GATE_NO_ERROR", flush=True)
    os._exit(65)


def _distributed_case(mode: str) -> None:
    import torch.distributed as dist

    from mok import functional

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    num_local_tokens, hidden_size, topk, experts, n = 512, 256, 1, 2, 256
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=4096,
        schedule_capacity_multiplier=1.0,
    )
    workspace = functional.get_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_local_experts=experts,
    )
    world_size = dist.get_world_size()
    token_indices = torch.arange(num_local_tokens, device=device)
    top_experts = (
        (token_indices % world_size) * experts
    ).view(-1, 1)
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=experts,
        expert_padding=64,
    )
    x = torch.zeros(
        num_local_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=device
    )
    x_scale = torch.ones(
        num_local_tokens, hidden_size // 128, dtype=torch.float32,
        device=device,
    )
    weight = torch.zeros(
        experts, n, hidden_size, dtype=torch.float8_e4m3fn, device=device
    )
    weight_scale = torch.ones(
        experts, n // 128, hidden_size // 128, dtype=torch.float32,
        device=device,
    )
    gate_up = torch.zeros(
        workspace.schedule_capacity, n, dtype=torch.bfloat16, device=device
    )

    knobs = {}
    if mode == "contract":
        # Device-side injection: a non-M64-aligned num_tokens is only
        # observable on the device (graph replay varies it), so the kernel's
        # contract check must fail closed with a CONTRACT trap.
        schedule.num_tokens.fill_(65)
    elif mode == "timeout":
        # Every M64 tile contains rows striped to copy ticket 0, so a long
        # ticket-0 delay stalls every tile_ready spin past the shortened
        # timeout: TIMEOUT trap at the tile_ready site.
        knobs = dict(
            spin_trap_iters=200_000,
            delay_ticket0_cycles=3_000_000_000,
        )
    try:
        functional.dispatch_gemm_fused_fp8_block(
            workspace, schedule, x, x_scale, weight, weight_scale, gate_up,
            copy_clusters=8, **knobs,
        )
        torch.cuda.synchronize()
    except RuntimeError:
        _die(workspace.trap_record)
    print("TRAP_GATE_NO_ERROR", flush=True)
    os._exit(65)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "reentrant":
        case_reentrant()
    else:
        _distributed_case(mode)
