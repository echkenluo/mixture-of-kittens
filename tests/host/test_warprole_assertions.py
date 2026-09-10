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


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float32])
def test_activation_bits_accept_equal_and_reject_signed_zero(dtype):
    check = MODULE.assert_activation_bitwise
    tensor = torch.tensor([[0., 1., -2.]]).to(dtype)
    check("equal", tensor, tensor.clone())
    with pytest.raises(AssertionError, match="bytes differ"):
        check("signed-zero", torch.tensor([[0.]]).to(dtype), torch.tensor([[-0.]]).to(dtype))


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float32])
@pytest.mark.parametrize("side", ["expected", "actual", "both"])
def test_activation_nonfinite_is_rejected(dtype, side):
    bad = torch.tensor([[float("nan")]]).to(dtype)
    finite = torch.ones(1, 1).to(dtype)
    with pytest.raises(AssertionError, match="non-finite"):
        MODULE.assert_activation_bitwise("nonfinite", bad if side != "actual" else finite,
                                         bad if side != "expected" else finite)


def test_activation_metadata_and_unrepresentable_scale_difference():
    check = MODULE.assert_activation_bitwise
    with pytest.raises(AssertionError, match="metadata"):
        check("dtype", torch.ones(1, 1), torch.ones(1, 1).to(torch.float8_e4m3fn))
    # A difference invisible after BF16 conversion must still fail for FP32 scales.
    a = torch.ones(1, 1); b = a + 1e-6
    assert torch.equal(a.bfloat16(), b.bfloat16())
    with pytest.raises(AssertionError, match="bytes differ"):
        check("fp32-scale", a, b)
