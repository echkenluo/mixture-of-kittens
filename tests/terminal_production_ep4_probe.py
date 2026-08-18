#!/usr/bin/env python3
"""Real EP4 production-entry gate for the terminal FP8 megakernel.

Run with one process per GPU.  The candidate path goes exclusively through
``create_fp8_terminal_workspace`` and ``megakernel_fp8_block``; the existing
single-GPU split probe supplies a bit-exact numerical oracle.
"""

from __future__ import annotations

import inspect
import os
import time

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")

import torch
import torch.distributed as dist

from mok.functional import (
    MoKSchedule,
    create_fp8_terminal_workspace,
    format_trap_record,
    megakernel_fp8_block,
    megakernel_fp8_block_from_topk,
    megakernel_fp8_block_leased,
)
from mok.ops import fp8_block_megakernel_prewarm
from terminal_full_pipeline_probe import build_extension


EP_SIZE = 4
HIDDEN = 4096
INTERMEDIATE = 2048
TOPK = 6
LOCAL_TOKENS = 8
LOCAL_EXPERTS = 64
CAPACITY = 64
COMPUTE_CLUSTERS = int(os.environ.get("MOK_TERMINAL_EP4_COMPUTE_CLUSTERS", "77"))
ITERATIONS = int(os.environ.get("MOK_TERMINAL_EP4_ITERATIONS", "100"))
REPORT_PROGRESS = os.environ.get("MOK_TERMINAL_EP4_REPORT_PROGRESS") == "1"
MINIBATCH_ROWS = 64
MACROBATCH_ROWS = 64
SPIN_LIMIT = int(os.environ.get("MOK_TERMINAL_EP4_SPIN_LIMIT", str(1 << 29)))


def gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a contiguous tensor and restore an explicit rank axis."""
    tensor = tensor.contiguous()
    flat = torch.empty(
        (EP_SIZE * tensor.shape[0], *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(flat, tensor)
    return flat.view(EP_SIZE, *tensor.shape)


def require_exact(
    name: str, expected: torch.Tensor, actual: torch.Tensor
) -> None:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise RuntimeError(
            f"{name} metadata mismatch: expected={expected.shape}/{expected.dtype}, "
            f"actual={actual.shape}/{actual.dtype}"
        )
    if expected.dtype == torch.float8_e4m3fn:
        lhs, rhs = expected.view(torch.uint8), actual.view(torch.uint8)
    elif expected.dtype == torch.bfloat16:
        lhs, rhs = expected.view(torch.int16), actual.view(torch.int16)
    elif expected.dtype in (torch.float32, torch.int32):
        lhs, rhs = expected.view(torch.int32), actual.view(torch.int32)
    else:
        raise TypeError(f"unsupported exact dtype for {name}: {expected.dtype}")
    mismatches = int((lhs != rhs).sum().item())
    if mismatches:
        raise RuntimeError(f"{name} exact mismatch: {mismatches} elements")


def make_local_schedule(rank: int, device: torch.device) -> MoKSchedule:
    """One valid M64 expert segment with 48 remote routes and 16 pads."""
    peer_rank = torch.full(
        (CAPACITY,), -1, dtype=torch.int32, device=device
    )
    peer_token_idx = torch.full_like(peer_rank, -1)
    peer_rank[: LOCAL_TOKENS * TOPK] = (rank + 1) % EP_SIZE
    peer_token_idx[: LOCAL_TOKENS * TOPK] = torch.arange(
        LOCAL_TOKENS * TOPK, dtype=torch.int32, device=device
    )
    num_tokens = torch.tensor([CAPACITY], dtype=torch.int32, device=device)
    tokens_per_expert = torch.zeros(
        LOCAL_EXPERTS, dtype=torch.int32, device=device
    )
    tokens_per_expert[0] = CAPACITY
    return MoKSchedule(
        peer_rank=peer_rank,
        peer_token_idx=peer_token_idx,
        num_tokens=num_tokens,
        tokens_per_expert=tokens_per_expert,
        expert_padding=64,
    )


def make_fp8_full(
    shape: tuple[int, ...], value: float, device: torch.device
) -> torch.Tensor:
    # Constructing directly in FP8 avoids a multi-gigabyte float32 temporary
    # for the fixed E64 production weight shapes.
    result = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    result.fill_(value)
    return result


def allocate_reference(device: torch.device) -> dict[str, torch.Tensor]:
    routed_x = torch.empty(
        (CAPACITY, HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    routed_x.view(torch.uint8).fill_(0x7F)
    down_input = torch.empty(
        (CAPACITY, INTERMEDIATE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    down_input.view(torch.uint8).fill_(0x7F)
    return {
        "routed_x": routed_x,
        "routed_x_scale": torch.full(
            (CAPACITY, HIDDEN // 128),
            float("nan"), dtype=torch.float32, device=device,
        ),
        "m_indices": torch.full(
            (CAPACITY,), -1, dtype=torch.int32, device=device
        ),
        "gate_up": torch.full(
            (CAPACITY, 2 * INTERMEDIATE),
            float("nan"), dtype=torch.bfloat16, device=device,
        ),
        "down_input": down_input,
        "down_input_scale": torch.full(
            (CAPACITY, INTERMEDIATE // 128),
            float("nan"), dtype=torch.float32, device=device,
        ),
        "routed_y": torch.full(
            (CAPACITY, HIDDEN),
            float("nan"), dtype=torch.bfloat16, device=device,
        ),
        # Only one producer owns any given destination in this probe.  Zero
        # initialization makes the other three split outputs harmless while
        # preserving the complete next-rank target as an exact oracle.
        "combine": torch.zeros(
            (EP_SIZE, LOCAL_TOKENS * TOPK, HIDDEN),
            dtype=torch.bfloat16, device=device,
        ),
        "output": torch.full(
            (EP_SIZE, LOCAL_TOKENS, HIDDEN),
            float("nan"), dtype=torch.bfloat16, device=device,
        ),
    }


def check_python_entry_contract() -> None:
    owned_source = inspect.getsource(megakernel_fp8_block)
    owned_order = (
        "_validate_terminal_forward(",
        "workspace_lease_acquire(",
        "megakernel_fp8_block_leased(",
    )
    positions = [owned_source.find(needle) for needle in owned_order]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RuntimeError(f"terminal owned launch order changed: {positions}")
    if "workspace_lease_release" in owned_source:
        raise RuntimeError("terminal owned entry has a forbidden tail release")

    leased_source = inspect.getsource(megakernel_fp8_block_leased)
    leased_order = (
        "workspace.x_buffer.copy_(x)",
        "workspace.x_scale_buffer.copy_(x_scale)",
        "fp8_block_megakernel_prepare_out(",
        "fp8_block_megakernel_out(",
    )
    positions = [leased_source.find(needle) for needle in leased_order]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RuntimeError(f"terminal leased launch order changed: {positions}")
    if "workspace_lease_acquire" in leased_source:
        raise RuntimeError("terminal leased entry must not reacquire")
    if "workspace_lease_release" in leased_source:
        raise RuntimeError("terminal leased entry has a forbidden tail release")
    if leased_source.count("workspace.comm_owner") != 2:
        raise RuntimeError(
            "terminal leased entry must pass comm_owner to prepare and full"
        )

    orchestrator_source = inspect.getsource(megakernel_fp8_block_from_topk)
    orchestrator_order = (
        "_validate_terminal_forward(",
        "_validate_build_schedule_inputs(",
        "workspace_lease_acquire(",
        "_build_schedule_validated(",
        "megakernel_fp8_block_leased(",
    )
    positions = [
        orchestrator_source.find(needle) for needle in orchestrator_order
    ]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RuntimeError(
            f"terminal orchestrator launch order changed: {positions}"
        )


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != EP_SIZE:
        raise RuntimeError(f"terminal production probe requires EP4, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("terminal production probe requires SM90")
        check_python_entry_contract()

        max_compute_clusters = fp8_block_megakernel_prewarm(local_rank)
        if max_compute_clusters < COMPUTE_CLUSTERS:
            raise RuntimeError(
                f"need {COMPUTE_CLUSTERS} compute clusters, got "
                f"{max_compute_clusters}"
            )
        workspace = create_fp8_terminal_workspace(
            dist.group.WORLD,
            device=device,
            num_local_tokens=LOCAL_TOKENS,
            schedule_capacity=CAPACITY,
            num_local_experts=LOCAL_EXPERTS,
            compute_clusters=COMPUTE_CLUSTERS,
        )
        schedule = make_local_schedule(rank, device)

        token_values = (
            torch.arange(LOCAL_TOKENS, dtype=torch.float32, device=device)
            .mul_(0.0078125)
            .add_(0.03125 * (rank + 1))
        )
        x = token_values[:, None].expand(-1, HIDDEN).to(
            torch.float8_e4m3fn
        ).contiguous()
        x_scale = (
            torch.arange(HIDDEN // 128, dtype=torch.float32, device=device)
            .mul_(0.0009765625)
            .add_(0.5 + 0.0625 * rank)
            .expand(LOCAL_TOKENS, -1)
            .contiguous()
        )
        w13 = make_fp8_full(
            (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
            0.0078125 * (rank + 1),
            device,
        )
        w13_scale = torch.full(
            (LOCAL_EXPERTS, 32, 32), 0.5 + 0.0625 * rank,
            dtype=torch.float32, device=device,
        )
        w2 = make_fp8_full(
            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
            0.00390625 * (rank + 1),
            device,
        )
        w2_scale = torch.full(
            (LOCAL_EXPERTS, 32, 16), 0.5 + 0.0625 * rank,
            dtype=torch.float32, device=device,
        )
        base_weights = torch.arange(
            1, TOPK + 1, dtype=torch.float32, device=device
        )
        topk_weights = torch.roll(base_weights, rank).div_(
            base_weights.sum()
        ).expand(LOCAL_TOKENS, -1).contiguous()
        # Rank r's schedule owns local expert zero and returns to r+1.
        # Consequently target rank t names expert zero of rank t-1.
        topk_ids = torch.full(
            (LOCAL_TOKENS, TOPK),
            ((rank - 1) % EP_SIZE) * LOCAL_EXPERTS,
            dtype=torch.int32,
            device=device,
        )

        peer_x = gather(x.view(torch.uint8))
        peer_scale = gather(x_scale)
        peer_weights = gather(topk_weights)
        peer_topk_ids = gather(topk_ids)

        split = build_extension(verbose=False)
        reference = allocate_reference(device)
        push_order = torch.arange(CAPACITY, dtype=torch.int32, device=device)
        split.run_split(
            peer_x,
            peer_scale,
            w13,
            w13_scale,
            w2,
            w2_scale,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.tokens_per_expert,
            push_order,
            peer_weights,
            peer_topk_ids,
            reference["routed_x"],
            reference["routed_x_scale"],
            reference["m_indices"],
            reference["gate_up"],
            reference["down_input"],
            reference["down_input_scale"],
            reference["routed_y"],
            reference["combine"],
            reference["output"],
            10.0,
        )
        torch.cuda.synchronize(device)

        target = (rank + 1) % EP_SIZE
        expected_outputs = gather(reference["output"][target].contiguous())
        expected_combines = gather(reference["combine"][target].contiguous())
        previous = (rank - 1) % EP_SIZE

        output = torch.full(
            (LOCAL_TOKENS, HIDDEN),
            float("nan"), dtype=torch.bfloat16, device=device,
        )
        dist.barrier()
        owners: list[int] = []
        for iteration in range(ITERATIONS):
            if REPORT_PROGRESS and rank == 0:
                print(f"TERMINAL_EP4_ITERATION|index={iteration}|state=start", flush=True)
            # Rank skew on the first iteration proves that elected resident
            # owners wait at the in-kernel EP4 input barrier instead of
            # consuming a peer's prior input bucket.
            if iteration == 0 and rank == EP_SIZE - 1:
                time.sleep(0.05)
            megakernel_fp8_block(
                workspace,
                schedule,
                x,
                x_scale,
                w13,
                w13_scale,
                w2,
                w2_scale,
                topk_weights,
                topk_ids,
                output,
                minibatch_rows=MINIBATCH_ROWS,
                macrobatch_rows=MACROBATCH_ROWS,
                spin_limit=SPIN_LIMIT,
            )
            try:
                torch.cuda.synchronize(device)
            except RuntimeError:
                # A terminal trap poisons the CUDA context.  The host-mapped
                # record is the only safe diagnostic surface after that point.
                trap = format_trap_record(workspace)
                if trap is not None:
                    print(trap, flush=True)
                raise
            owner = int(workspace.comm_owner.item())
            if not 0 <= owner <= COMPUTE_CLUSTERS:
                raise RuntimeError(
                    f"iteration {iteration} elected invalid comm owner {owner}"
                )
            if REPORT_PROGRESS and rank == 0:
                print(
                    f"TERMINAL_EP4_ITERATION|index={iteration}|state=end"
                    f"|comm_owner={owner}",
                    flush=True,
                )
            owners.append(owner)
            require_exact(
                f"output_iteration_{iteration}",
                expected_outputs[previous],
                output,
            )
            iteration_state = {
                "in_use": 0,
                "next_logical_cluster": 65,
                "producer_done": 65,
                "comm_closed": 2,
                "push_done": CAPACITY,
                "reduce_done": LOCAL_TOKENS,
                "terminate": 1,
                "epilogue_done": 1 + COMPUTE_CLUSTERS,
                "barrier_target": EP_SIZE * (iteration + 1),
                "input_expected_scratch": EP_SIZE * (iteration + 1),
            }
            observed_iteration_state = {
                name: int(getattr(workspace, name).item())
                for name in iteration_state
            }
            if observed_iteration_state != iteration_state:
                raise RuntimeError(
                    f"iteration {iteration} closure mismatch: "
                    f"expected={iteration_state}, "
                    f"observed={observed_iteration_state}"
                )

        require_exact("routed_x", reference["routed_x"], workspace.routed_x)
        require_exact(
            "routed_x_scale",
            reference["routed_x_scale"], workspace.routed_x_scale,
        )
        require_exact("m_indices", reference["m_indices"], workspace.m_indices)
        require_exact("gate_up", reference["gate_up"], workspace.gate_up)
        require_exact(
            "down_input", reference["down_input"], workspace.down_input
        )
        require_exact(
            "down_input_scale",
            reference["down_input_scale"],
            workspace.down_input_scale,
        )
        require_exact("routed_y", reference["routed_y"], workspace.routed_y)
        require_exact(
            "combine",
            expected_combines[previous],
            workspace.combine_buffer[: LOCAL_TOKENS * TOPK],
        )
        require_exact("output", expected_outputs[previous], output)
        if not bool(torch.all(workspace.route_ready == 1).item()):
            raise RuntimeError("terminal route-ready closure is incomplete")
        if not bool(
            torch.all(workspace.epilogue_claim[:LOCAL_TOKENS] == 1).item()
        ):
            raise RuntimeError("terminal output tokens were not claimed exactly once")
        if int(workspace.trap_record[0].item()) != 0:
            raise RuntimeError(
                f"terminal trap record is nonzero: {workspace.trap_record.tolist()}"
            )

        passed = torch.ones(1, dtype=torch.int32, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)
        if int(passed.item()) != EP_SIZE:
            raise RuntimeError("terminal production four-rank consensus failed")
        if rank == 0:
            print(
                "TERMINAL_PRODUCTION_EP4"
                "|hidden=4096|intermediate=2048|topk=6|experts=64"
                f"|tokens=8|capacity=64|compute_clusters={COMPUTE_CLUSTERS}"
                f"|iterations={ITERATIONS}"
                "|remote_dispatch=1|remote_combine=1|input_skew=1"
                f"|comm_owners={','.join(map(str, sorted(set(owners))))}"
                "|boundaries=bitwise_exact|output=bitwise_exact"
                "|closure=producer,comm,push,reduce,terminate"
                "|lease_release=kernel|candidate_launches=1|result=PASS",
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
