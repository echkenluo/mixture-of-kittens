"""Run with --confcutdir=tests/host: no distributed/GPU conftest required."""

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "warprole_assertions", Path(__file__).parents[1] / "warprole_assertions.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
assert_bitwise = MODULE.assert_bitwise


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["both", "expected", "actual"])
def test_nonfinite_is_rejected_even_when_bits_match(value, side):
    bad = torch.tensor([[value]], dtype=torch.bfloat16)
    finite = torch.ones_like(bad)
    with pytest.raises(AssertionError, match="non-finite"):
        assert_bitwise("nonfinite", bad if side != "actual" else finite,
                       bad if side != "expected" else finite)


def test_signed_zero_is_distinct():
    with pytest.raises(AssertionError, match="elements differ"):
        assert_bitwise("signed-zero", torch.tensor([[0.]], dtype=torch.bfloat16),
                       torch.tensor([[-0.]], dtype=torch.bfloat16))


def test_equal_finite_outputs_pass():
    tensor = torch.tensor([[0., 1., -2.]], dtype=torch.bfloat16)
    assert_bitwise("finite", tensor, tensor.clone())
