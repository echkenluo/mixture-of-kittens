#!/usr/bin/env python3
"""EP4 gate for the owned top-k -> terminal megakernel orchestrator.

This is deliberately separate from the manual-schedule production probe.  It
exercises the production-owned schedule buffers and verifies that one lease
covers fused schedule construction, input staging, prepare, and the terminal
compute kernel.  The numerical oracle remains the established split extension.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")

import torch
import torch.distributed as dist

from mok.functional import (
    MoKConfig,
    acquire_workspace_lease,
    create_fp8_terminal_workspace,
    megakernel_fp8_block_from_topk,
    megakernel_fp8_block_leased,
    release_workspace_lease,
)
from terminal_full_pipeline_probe import build_extension


EP_SIZE = 4
HIDDEN = 4096
INTERMEDIATE = 2048
TOPK = 6
LOCAL_TOKENS = 8
LOCAL_EXPERTS = 64
CAPACITY = 256
ACTIVE_ROWS = 64
VALID_ROUTES = LOCAL_TOKENS * TOPK
COMPUTE_CLUSTERS = 7
SPIN_LIMIT = 1 << 29
POISON_BYTE = 0x7F
POISON_INDEX = -777


def gather(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    gathered = torch.empty(
        (EP_SIZE * tensor.shape[0], *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor)
    return gathered.view(EP_SIZE, *tensor.shape)


def make_fp8_full(
    shape: tuple[int, ...], value: float, device: torch.device
) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    result.fill_(value)
    return result


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
        raise RuntimeError(f"{name} bitwise mismatch: {mismatches} elements")


def check_orchestrator_source_contract() -> None:
    orchestrator = inspect.getsource(megakernel_fp8_block_from_topk)
    ordered = (
        "_validate_terminal_forward(",
        "_validate_build_schedule_inputs(",
        "workspace_lease_acquire(",
        "_build_schedule_validated(",
        "megakernel_fp8_block_leased(",
    )
    positions = [orchestrator.find(needle) for needle in ordered]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RuntimeError(f"terminal orchestrator order changed: {positions}")
    for forbidden in (
        "workspace_lease_release",
        "torch.cuda.synchronize",
        ".item(",
        "megakernel_fp8_block(",
        "dispatch_gemm_fused_fp8_block",
    ):
        if forbidden in orchestrator:
            raise RuntimeError(
                f"terminal orchestrator contains forbidden path: {forbidden}"
            )

    leased = inspect.getsource(megakernel_fp8_block_leased)
    leased_order = (
        "workspace.x_buffer.copy_(x)",
        "workspace.x_scale_buffer.copy_(x_scale)",
        "fp8_block_megakernel_prepare_out(",
        "fp8_block_megakernel_out(",
    )
    positions = [leased.find(needle) for needle in leased_order]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RuntimeError(f"terminal leased order changed: {positions}")
    if leased.count("fp8_block_megakernel_out(") != 1:
        raise RuntimeError("leased terminal path must contain one compute kernel")
    if "workspace_lease_acquire" in leased or "workspace_lease_release" in leased:
        raise RuntimeError("leased terminal path must not alter lease ownership")


def make_expected_schedule(
    rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_rank = (rank + 1) % EP_SIZE
    peer_rank = torch.full(
        (CAPACITY,), -1, dtype=torch.int32, device=device
    )
    peer_token_idx = torch.full_like(peer_rank, -1)
    peer_rank[:VALID_ROUTES] = source_rank
    peer_token_idx[:VALID_ROUTES] = torch.arange(
        VALID_ROUTES, dtype=torch.int32, device=device
    )
    num_tokens = torch.tensor([ACTIVE_ROWS], dtype=torch.int32, device=device)
    tokens_per_expert = torch.zeros(
        LOCAL_EXPERTS, dtype=torch.int32, device=device
    )
    tokens_per_expert[0] = ACTIVE_ROWS
    tokens_per_expert_and_peer = torch.zeros(
        LOCAL_EXPERTS * EP_SIZE, dtype=torch.int32, device=device
    )
    tokens_per_expert_and_peer[source_rank] = VALID_ROUTES
    return (
        peer_rank,
        peer_token_idx,
        num_tokens,
        tokens_per_expert,
        tokens_per_expert_and_peer,
    )


def allocate_reference(device: torch.device) -> dict[str, torch.Tensor]:
    routed_x = torch.empty(
        (ACTIVE_ROWS, HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    routed_x.view(torch.uint8).fill_(POISON_BYTE)
    hidden = torch.empty(
        (ACTIVE_ROWS, INTERMEDIATE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    hidden.view(torch.uint8).fill_(POISON_BYTE)
    return {
        "routed_x": routed_x,
        "routed_x_scale": torch.full(
            (ACTIVE_ROWS, HIDDEN // 128),
            float("nan"),
            dtype=torch.float32,
            device=device,
        ),
        "m_indices": torch.full(
            (ACTIVE_ROWS,), POISON_INDEX, dtype=torch.int32, device=device
        ),
        "gate_up": torch.full(
            (ACTIVE_ROWS, 2 * INTERMEDIATE),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
        "down_input": hidden,
        "down_input_scale": torch.full(
            (ACTIVE_ROWS, INTERMEDIATE // 128),
            float("nan"),
            dtype=torch.float32,
            device=device,
        ),
        "routed_y": torch.full(
            (ACTIVE_ROWS, HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
        "combine": torch.zeros(
            (EP_SIZE, VALID_ROUTES, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        ),
        "output": torch.full(
            (EP_SIZE, LOCAL_TOKENS, HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def poison_terminal(workspace) -> None:
    workspace.routed_x.view(torch.uint8).fill_(POISON_BYTE)
    workspace.routed_x_scale.fill_(float("nan"))
    workspace.m_indices.fill_(POISON_INDEX)
    workspace.schedule_peer_rank.fill_(POISON_INDEX)
    workspace.schedule_peer_token_idx.fill_(POISON_INDEX)
    workspace.schedule_num_tokens.fill_(POISON_INDEX)
    workspace.schedule_tokens_per_expert.fill_(POISON_INDEX)
    workspace.schedule_tokens_per_expert_and_peer.fill_(POISON_INDEX)
    workspace.gate_up.fill_(float("nan"))
    workspace.down_input.view(torch.uint8).fill_(POISON_BYTE)
    workspace.down_input_scale.fill_(float("nan"))
    workspace.routed_y.fill_(float("nan"))
    workspace.combine_buffer.fill_(float("nan"))
    workspace.route_ready.fill_(17)
    for state in (
        workspace.x_routed_ready,
        workspace.gate_up_tile_ready,
        workspace.hidden_row_block_ready,
        workspace.y_routed_ready,
        workspace.y_routed_done,
        workspace.epilogue_claim,
        workspace.next_logical_cluster,
        workspace.next_reduce_probe,
        workspace.role_cursor,
        workspace.cluster_role,
        workspace.dispatch_tile_cursor,
        workspace.dispatch_tiles_done,
        workspace.push_tile_cursor,
        workspace.worker_ticket,
        workspace.comm_worker_ticket,
        workspace.producer_done,
        workspace.comm_closed,
        workspace.push_done,
        workspace.reduce_done,
        workspace.terminate,
        workspace.epilogue_done,
        workspace.input_expected_scratch,
    ):
        state.fill_(17)


def check_schedule(
    workspace,
    expected: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    gathered_topk_ids: torch.Tensor,
) -> None:
    (
        expected_peer_rank,
        expected_peer_token_idx,
        expected_num_tokens,
        expected_tokens_per_expert,
        expected_tokens_per_expert_and_peer,
    ) = expected
    require_exact(
        "schedule_peer_rank",
        expected_peer_rank,
        workspace.schedule_peer_rank,
    )
    require_exact(
        "schedule_peer_token_idx_valid",
        expected_peer_token_idx[:VALID_ROUTES],
        workspace.schedule_peer_token_idx[:VALID_ROUTES],
    )
    require_exact(
        "schedule_num_tokens",
        expected_num_tokens,
        workspace.schedule_num_tokens,
    )
    require_exact(
        "schedule_tokens_per_expert",
        expected_tokens_per_expert,
        workspace.schedule_tokens_per_expert,
    )
    require_exact(
        "schedule_tokens_per_expert_and_peer",
        expected_tokens_per_expert_and_peer,
        workspace.schedule_tokens_per_expert_and_peer,
    )
    require_exact(
        "all_gather_topk_ids",
        gathered_topk_ids,
        workspace.all_gather_top_experts_buffer,
    )


def check_inactive_untouched(workspace) -> None:
    inactive = slice(ACTIVE_ROWS, CAPACITY)
    if not bool(
        torch.all(
            workspace.routed_x[inactive].view(torch.uint8) == POISON_BYTE
        ).item()
    ):
        raise RuntimeError("inactive routed_x rows were overwritten")
    if not bool(torch.isnan(workspace.routed_x_scale[inactive]).all().item()):
        raise RuntimeError("inactive routed_x_scale rows were overwritten")
    if not bool(torch.all(workspace.m_indices[inactive] == POISON_INDEX).item()):
        raise RuntimeError("inactive m_indices rows were overwritten")
    if not bool(torch.isnan(workspace.gate_up[inactive]).all().item()):
        raise RuntimeError("inactive gate_up rows were overwritten")
    if not bool(
        torch.all(
            workspace.down_input[inactive].view(torch.uint8) == POISON_BYTE
        ).item()
    ):
        raise RuntimeError("inactive down_input rows were overwritten")
    if not bool(torch.isnan(workspace.down_input_scale[inactive]).all().item()):
        raise RuntimeError("inactive down_input_scale rows were overwritten")
    if not bool(torch.isnan(workspace.routed_y[inactive]).all().item()):
        raise RuntimeError("inactive routed_y rows were overwritten")


def check_closure(workspace) -> None:
    expected = {
        "in_use": 0,
        "role_cursor": 1 + COMPUTE_CLUSTERS,
        "dispatch_tile_cursor": 1,
        "dispatch_tiles_done": 1,
        "push_tile_cursor": 1,
        "next_logical_cluster": 65,
        "comm_worker_ticket": -2,
        "producer_done": 65,
        "comm_closed": 1,
        "push_done": ACTIVE_ROWS,
        "reduce_done": LOCAL_TOKENS,
        "terminate": 1,
        "epilogue_done": 1 + COMPUTE_CLUSTERS,
        # Fused schedule construction closes one EP barrier before the
        # terminal kernel's own input barrier closes the next generation.
        "barrier_target": 2 * EP_SIZE,
        "input_expected_scratch": 2 * EP_SIZE,
    }
    observed = {
        name: int(getattr(workspace, name).item()) for name in expected
    }
    if observed != expected:
        raise RuntimeError(
            f"terminal closure mismatch: expected={expected}, observed={observed}"
        )
    if sorted(workspace.cluster_role.tolist()) != list(
        range(1 + COMPUTE_CLUSTERS)
    ):
        raise RuntimeError("terminal role ordinals are not unique")
    if not bool(torch.all(workspace.route_ready == 1).item()):
        raise RuntimeError("terminal route-ready closure is incomplete")
    if not bool(
        torch.all(workspace.epilogue_claim[:LOCAL_TOKENS] == 1).item()
    ):
        raise RuntimeError("terminal output tokens were not claimed exactly once")
    if not bool(
        torch.all(workspace.epilogue_claim[LOCAL_TOKENS:] == 0).item()
    ):
        raise RuntimeError("padded output tokens were unexpectedly claimed")
    if int(workspace.trap_record[0].item()) != 0:
        raise RuntimeError(
            f"terminal trap record is nonzero: {workspace.trap_record.tolist()}"
        )


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != EP_SIZE:
        raise RuntimeError(
            f"terminal orchestrator probe requires EP4, got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("terminal orchestrator probe requires SM90")
        check_orchestrator_source_contract()

        config = MoKConfig(
            fwd_num_comm_sms=2,
            bwd_num_comm_sms=2,
            minibatch_size=256,
            macrobatch_size=256,
            all_gather_top_experts_chunk_bytes=192,
        )
        workspace = create_fp8_terminal_workspace(
            dist.group.WORLD,
            device=device,
            num_local_tokens=LOCAL_TOKENS,
            schedule_capacity=CAPACITY,
            num_local_experts=LOCAL_EXPERTS,
            compute_clusters=COMPUTE_CLUSTERS,
        )
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
            (LOCAL_EXPERTS, 32, 32),
            0.5 + 0.0625 * rank,
            dtype=torch.float32,
            device=device,
        )
        w2 = make_fp8_full(
            (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
            0.00390625 * (rank + 1),
            device,
        )
        w2_scale = torch.full(
            (LOCAL_EXPERTS, 32, 16),
            0.5 + 0.0625 * rank,
            dtype=torch.float32,
            device=device,
        )
        base_weights = torch.arange(
            1, TOPK + 1, dtype=torch.float32, device=device
        )
        topk_weights = torch.roll(base_weights, rank).div(
            base_weights.sum()
        ).expand(LOCAL_TOKENS, -1).contiguous()
        # Every rank sends all routes to expert zero on the preceding EP rank.
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
        expected_schedule = make_expected_schedule(rank, device)

        split = build_extension(verbose=False)
        reference = allocate_reference(device)
        push_order = torch.arange(
            ACTIVE_ROWS, dtype=torch.int32, device=device
        )
        split.run_split(
            peer_x,
            peer_scale,
            w13,
            w13_scale,
            w2,
            w2_scale,
            expected_schedule[0][:ACTIVE_ROWS],
            expected_schedule[1][:ACTIVE_ROWS],
            expected_schedule[3],
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

        # Model a completed prior owner.  Poisoning is enclosed by its own
        # lease, so acquire remains the first device operation of the
        # production transaction exercised below.
        acquire_workspace_lease(workspace)
        poison_terminal(workspace)
        release_workspace_lease(workspace)
        output = torch.full(
            (LOCAL_TOKENS, HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        dist.barrier()
        megakernel_fp8_block_from_topk(
            workspace,
            config,
            x,
            x_scale,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            output,
            spin_limit=SPIN_LIMIT,
        )
        torch.cuda.synchronize(device)

        check_schedule(workspace, expected_schedule, peer_topk_ids)
        require_exact(
            "routed_x",
            reference["routed_x"],
            workspace.routed_x[:ACTIVE_ROWS],
        )
        require_exact(
            "routed_x_scale",
            reference["routed_x_scale"],
            workspace.routed_x_scale[:ACTIVE_ROWS],
        )
        require_exact(
            "m_indices",
            reference["m_indices"],
            workspace.m_indices[:ACTIVE_ROWS],
        )
        require_exact(
            "gate_up",
            reference["gate_up"],
            workspace.gate_up[:ACTIVE_ROWS],
        )
        require_exact(
            "down_input",
            reference["down_input"],
            workspace.down_input[:ACTIVE_ROWS],
        )
        require_exact(
            "down_input_scale",
            reference["down_input_scale"],
            workspace.down_input_scale[:ACTIVE_ROWS],
        )
        require_exact(
            "routed_y",
            reference["routed_y"],
            workspace.routed_y[:ACTIVE_ROWS],
        )
        target_rank = (rank + 1) % EP_SIZE
        previous_rank = (rank - 1) % EP_SIZE
        expected_outputs = gather(reference["output"][target_rank].contiguous())
        expected_combines = gather(
            reference["combine"][target_rank].contiguous()
        )
        require_exact(
            "combine",
            expected_combines[previous_rank],
            workspace.combine_buffer[:VALID_ROUTES],
        )
        require_exact("output", expected_outputs[previous_rank], output)
        check_inactive_untouched(workspace)
        if not bool(
            torch.isnan(workspace.combine_buffer[VALID_ROUTES:]).all().item()
        ):
            raise RuntimeError("invalid combine slots were overwritten")
        check_closure(workspace)

        passed = torch.ones(1, dtype=torch.int32, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)
        if int(passed.item()) != EP_SIZE:
            raise RuntimeError("terminal orchestrator EP4 consensus failed")
        if rank == 0:
            print(
                "TERMINAL_ORCHESTRATOR_EP4"
                "|tokens=8|capacity=256|experts=64|topk=6"
                "|schedule_rows=64|valid_routes=48|expert_padding=64"
                "|schedule=exact|boundaries=bitwise_exact"
                "|output=bitwise_exact|inactive=untouched"
                "|lease=single_owner|candidate_compute_launches=1"
                "|closure=PASS|result=PASS",
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
