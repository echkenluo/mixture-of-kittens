#!/usr/bin/env python3
"""Pure-Python negative gates for terminal raw bulk-TMA storage."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "mok" / "_terminal_tma_contract.py"


def load_contract():
    spec = importlib.util.spec_from_file_location(
        "terminal_tma_contract", CONTRACT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load terminal TMA contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_case() -> dict[str, object]:
    # Duplicate peer virtual addresses are valid for a symmetric allocation.
    return {
        "x_ptrs": [0x1000] * 4,
        "x_bytes_per_rank": [0x400] * 4,
        "x_scale_ptrs": [0x2000] * 4,
        "x_scale_bytes_per_rank": [0x100] * 4,
        "required_x_bytes": 0x400,
        "required_x_scale_bytes": 0x100,
        "routed_x_pointer": 0x4000,
        "routed_x_bytes": 0x800,
        "routed_x_scale_pointer": 0x5000,
        "routed_x_scale_bytes": 0x200,
    }


def clone_case(case: dict[str, object]) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in case.items()
    }


def expect_rejected(contract, label: str, mutate) -> None:
    case = clone_case(valid_case())
    mutate(case)
    try:
        contract.validate_terminal_tma_dispatch_layout(**case)
    except ValueError:
        return
    raise RuntimeError(f"contract accepted invalid case: {label}")


def main() -> None:
    contract = load_contract()
    contract.validate_terminal_tma_dispatch_layout(**valid_case())

    # Adjacent half-open intervals are disjoint and must remain valid.
    adjacent = clone_case(valid_case())
    adjacent["routed_x_pointer"] = 0x1400
    contract.validate_terminal_tma_dispatch_layout(**adjacent)

    negative_cases = (
        (
            "misaligned peer x",
            lambda case: case["x_ptrs"].__setitem__(0, 0x1001),
        ),
        (
            "misaligned peer x_scale",
            lambda case: case["x_scale_ptrs"].__setitem__(3, 0x2001),
        ),
        (
            "misaligned routed_x",
            lambda case: case.__setitem__("routed_x_pointer", 0x4001),
        ),
        (
            "misaligned routed_x_scale",
            lambda case: case.__setitem__("routed_x_scale_pointer", 0x5001),
        ),
        (
            "x partial overlap routed_x",
            lambda case: case["x_ptrs"].__setitem__(1, 0x3F00),
        ),
        (
            "x cross-type overlap routed_x_scale",
            lambda case: case["x_ptrs"].__setitem__(1, 0x4F00),
        ),
        (
            "x_scale cross-type overlap routed_x",
            lambda case: case["x_scale_ptrs"].__setitem__(2, 0x4000),
        ),
        (
            "x_scale overlap routed_x_scale",
            lambda case: case["x_scale_ptrs"].__setitem__(2, 0x5000),
        ),
        (
            "source contains destination",
            lambda case: (
                case["x_ptrs"].__setitem__(0, 0x3000),
                case["x_bytes_per_rank"].__setitem__(0, 0x2000),
            ),
        ),
        (
            "undersized peer source",
            lambda case: case["x_bytes_per_rank"].__setitem__(3, 0x3F0),
        ),
        (
            "source interval overflow",
            lambda case: case["x_ptrs"].__setitem__(
                0, contract.MAX_UINTPTR - 0xFF
            ),
        ),
    )
    for label, mutate in negative_cases:
        expect_rejected(contract, label, mutate)

    print(
        "TERMINAL_TMA_CONTRACT_PYTHON"
        "|alignment=16|source_destination_pairs=16"
        "|symmetric_duplicate_va=allowed|negative_gates=11|result=PASS"
    )


if __name__ == "__main__":
    main()
