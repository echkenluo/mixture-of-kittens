#!/usr/bin/env python3
"""Fail-closed host/source contract for terminal raw bulk-TMA dispatch."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "csrc" / "sm90_fp8_block_terminal_tma_comm.cuh"
FULL = ROOT / "csrc" / "sm90_fp8_block_terminal_full.cuh"
ENTRY = ROOT / "csrc" / "sm90_fp8_block_terminal_entry.cuh"
DECODER = ROOT / "csrc" / "sm90_fp8_block_megakernel.cuh"
COMPUTE = ROOT / "csrc" / "sm90_fp8_block_terminal_compute.cuh"
COMM = ROOT / "csrc" / "sm90_fp8_block_terminal_comm_primitives.cuh"
ORIGINAL = ROOT / "csrc" / "mok_megakernel.cuh"
FUNCTIONAL = ROOT / "mok" / "functional.py"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise RuntimeError(f"missing {label}: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise RuntimeError(f"forbidden {label}: {needle}")


def require_order(text: str, needles: tuple[str, ...], label: str) -> None:
    positions = [text.find(needle) for needle in needles]
    if any(position < 0 for position in positions):
        raise RuntimeError(f"missing {label} element: {needles}")
    if positions != sorted(positions):
        raise RuntimeError(f"invalid {label} order: {needles}")


def main() -> None:
    header = HEADER.read_text(encoding="utf-8")
    full = FULL.read_text(encoding="utf-8")
    entry = ENTRY.read_text(encoding="utf-8")
    decoder = DECODER.read_text(encoding="utf-8")
    compute = COMPUTE.read_text(encoding="utf-8")
    comm = COMM.read_text(encoding="utf-8")
    original = ORIGINAL.read_text(encoding="utf-8")
    functional = FUNCTIONAL.read_text(encoding="utf-8")

    for invariant in (
        "CTA_ROWS == 4",
        "PIPE_DEPTH == 4",
        "SCALE_COLUMNS == 32",
        "DATA_BYTES_PER_ROW == 4096",
        "SCALE_BYTES_PER_ROW == 128",
        "BYTES_PER_VALID_ROW == 4224",
        "REQUIRED_SMEM_BYTES == 16896",
        "REQUIRED_SMEM_ALIGNMENT == 128",
    ):
        require(header, invariant, "fixed dispatch-TMA shape")

    dispatch = header[
        header.index("void dispatch_ticket(") :
        header.index("#endif  // defined(KITTENS_SM90)")
    ]
    for needle in (
        "comm::decode_route(g, row)",
        "kittens::tma::expect_bytes(",
        "kittens::tma::load_async(",
        "kittens::wait(",
        "kittens::update_phasebit<0>(phasebits, stage)",
        "kittens::tma::store_async(",
        "kittens::tma::store_async_wait();",
        "fence.proxy.async.global",
        "assign_expert(g, expert_row_end, row)",
    ):
        require(dispatch, needle, "dispatch-TMA payload")
    if dispatch.count("kittens::tma::load_async(") != 2:
        raise RuntimeError("dispatch must issue exactly data+scale raw TMA loads")
    if dispatch.count("kittens::tma::store_async(") != 2:
        raise RuntimeError("dispatch must issue exactly data+scale raw TMA stores")
    for needle in (
        "store_async_read_wait",
        "combine_ticket",
        "combine_staging",
        "push_routed_row",
        "route_ready",
        "MOK_TERMINAL_TMA",
        "#else",
    ):
        forbid(dispatch, needle, "dispatch-only/no-fallback scope")
    require_order(
        dispatch,
        (
            "route_valid ? BYTES_PER_VALID_ROW : 0u",
            "if (!route_valid)",
            "__syncthreads();",
            "kittens::wait(",
            "kittens::update_phasebit<0>(phasebits, stage)",
        ),
        "valid/invalid phase symmetry",
    )
    store_path = dispatch[dispatch.index("kittens::tma::store_async(") :]
    require_order(
        store_path,
        (
            "kittens::tma::store_async(",
            "kittens::tma::store_async_wait();",
            "fence.proxy.async.global",
            "assign_expert(g, expert_row_end, row)",
            "__syncthreads();",
        ),
        "issuer drain and generic-proxy publication",
    )

    original_dispatch = original[
        original.index("void dispatch_kernel(") :
        original.index("void combine_kernel(")
    ]
    for needle in (
        "tma::expect_bytes(",
        "tma::load_async(",
        "wait(inputs_arrived",
        "update_phasebit<0>",
        "tma::store_async(",
        "tma::store_async_wait();",
    ):
        require(original_dispatch, needle, "original MoK dispatch provenance")

    communication = full[
        full.index("__device__ void communication_role") :
        full.index("// One CTA probes exactly one")
    ]
    dispatch_branch = communication[
        communication.index("communication_stage::dispatch") :
        communication.index("communication_stage::combine")
    ]
    for needle in (
        '#include "sm90_fp8_block_terminal_tma_comm.cuh"',
        "dispatch_tma_inputs_arrived[dispatch_tma::PIPE_DEPTH]",
        "WARPS_PER_CTA == dispatch_tma::CTA_ROWS",
        "terminal::COMM_ROWS_PER_CTA_TASK == dispatch_tma::CTA_ROWS",
        "uint32_t dispatch_tma_phasebits = 0xFFFF0000u",
        "reinterpret_cast<uint64_t>(&a_smem[0])",
        "dispatch_tma::dispatch_ticket(",
    ):
        require(full, needle, "terminal dispatch wiring")
    if communication.count("dispatch_tma::dispatch_ticket(") != 1:
        raise RuntimeError("dispatch branch must contain one TMA ticket call")
    init_position = communication.index(
        "dispatch_tma_inputs_arrived[threadIdx.x], 0, 1"
    )
    loop_position = communication.index("while (true)")
    if init_position >= loop_position:
        raise RuntimeError("dispatch mbarriers must be initialized once before loop")
    for needle in (
        "comm::dispatch_copy_row(",
        "dispatch_tma::combine",
        "combine_ticket(",
    ):
        forbid(dispatch_branch, needle, "terminal dispatch branch")
    require_order(
        communication,
        (
            "decode_communication_cursor(shape, ticket)",
            "communication_stage::dispatch",
            "dispatch_tma::dispatch_ticket(",
            "add_acq_rel_gpu(\n                    g.x_ready + m, 1u)",
            "communication_stage::combine",
            "owner_help_one_producer(",
        ),
        "dense cursor, payload, x_ready, owner-help DAG",
    )
    for needle in (
        "route::push_routed_row_and_publish(g, row, lane)",
        "compute::add_release_gpu(g.push_tile_cursor, 1u)",
    ):
        require(communication, needle, "unchanged generic combine path")
    for needle in ("dispatch_tma::combine", "combine_ticket("):
        forbid(communication, needle, "TMA combine before its own candidate")

    for needle in (
        "static_cast<unsigned int>(shape.num_tokens / 4)",
        "claim_bounded(\n                    g.dispatch_tile_cursor, total_comm_tickets)",
        "struct producer_control",
        "static_assert(sizeof(producer_control) == 32",
        "owner_help_one_producer(",
        "production_completion_epilogue(g, cta_rank)",
        "__global__ void kernel",
    ):
        require(full, needle, "winning-core preservation")
    if full.count("__global__ void kernel") != 1:
        raise RuntimeError("terminal path must remain a single kernel")
    for needle in ("grid.sync", "cooperative_groups", "split_fallback"):
        forbid(full, needle, "terminal fallback/barrier")
    for needle in (
        "N128_SUBTASKS_PER_N256 = 2",
        "TASKS_PER_M64 == 33",
        "COMM_ROWS_PER_CTA_TASK == 4",
        "COMM_ROWS_PER_TICKET == 8",
    ):
        require(decoder, needle, "N256/dense-communication geometry")
    for needle in (
        "run_sequential_n128_subtask",
        "terminal::N128_SUBTASKS_PER_N256",
    ):
        require(compute, needle, "sequential N256 arithmetic")
    for needle in ("dispatch_copy_row(", "push_routed_row("):
        require(comm, needle, "shared K1/K2 primitive preservation")

    for needle in (
        "DYNAMIC_SMEM == 41984",
        "fp8_block_terminal_tma_comm::REQUIRED_SMEM_BYTES",
        "fp8_block_terminal_tma_comm::REQUIRED_SMEM_ALIGNMENT - 1",
        "terminal dynamic smem cannot hold aligned dispatch TMA staging",
        "routed_x.size(1) == terminal::HIDDEN_SIZE",
        "x_buffer.size(1) == terminal::HIDDEN_SIZE",
        "terminal::HIDDEN_SIZE / 128",
        "terminal::EP_SIZE",
        "terminal::TOP_K",
        "!x_buffer.is_alias_of(routed_x)",
        "!x_scale_buffer.is_alias_of(routed_x_scale)",
    ):
        require(entry, needle, "host launch/shape gate")
    require(
        functional,
        "terminal dispatch source and routed destination storage must be",
        "Python disjoint-storage gate",
    )

    print(
        "TERMINAL_TMA_DISPATCH_STATIC"
        "|scope=dispatch_only|ticket_rows=8|cta_rows=4|pipe_depth=4"
        "|staging_bytes=16896|workspace=wgmma_shared"
        "|store_drain=issuer_wait0|x_ready=device_release"
        "|comm_cursor=dense_dcd|owner_help=preserved"
        "|combine_tma=0|runtime_fallback=0|result=PASS"
    )


if __name__ == "__main__":
    main()
