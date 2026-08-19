"""Dependency-free source contracts for preloaded terminal inputs."""

import ast
from pathlib import Path


FUNCTIONAL = Path(__file__).resolve().parents[1] / "mok/functional.py"


def _function_source(name: str) -> str:
    source = FUNCTIONAL.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                segment = ast.get_source_segment(source, node)
                assert segment is not None
                return segment
    raise AssertionError(f"function {name} not found")


def test_preloaded_terminal_acquires_before_any_workspace_write():
    acquire = _function_source(
        "acquire_megakernel_fp8_block_from_topk_lease"
    )
    assert "_validate_and_acquire_terminal_from_topk(" in acquire
    for forbidden in (".copy_(", ".zero_(", ".fill_("):
        assert forbidden not in acquire
    assert "exception is process-fatal" in acquire


def test_preloaded_terminal_skips_symmetric_input_copies():
    preloaded = _function_source(
        "megakernel_fp8_block_from_topk_preloaded_leased"
    )
    ordered = (
        "_build_schedule_validated(",
        "megakernel_fp8_block_leased(",
        "inputs_preloaded=True",
    )
    positions = [preloaded.find(needle) for needle in ordered]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    assert ".copy_(" not in preloaded


def test_existing_owned_terminal_copy_contract_is_unchanged():
    leased = _function_source("megakernel_fp8_block_leased")
    assert "workspace.x_buffer.copy_(x)" in leased
    assert "workspace.x_scale_buffer.copy_(x_scale)" in leased
    assert "if inputs_preloaded:" in leased
    assert "fp8_block_megakernel_prepare_out(" in leased
    assert "fp8_block_megakernel_out(" in leased


def test_terminal_transaction_failure_formatter_is_cpu_only():
    formatter = _function_source("format_terminal_transaction_failure")
    assert "format_trap_record(workspace)" in formatter
    assert "except BaseException:" in formatter
    assert "MOK_TERMINAL_TRANSACTION_FATAL" in formatter
    for forbidden in (
        "workspace_lease_release(",
        "torch.cuda",
        "synchronize(",
        "reset_workspace",
    ):
        assert forbidden not in formatter
