"""Numerical assertions shared by distributed and CPU validator tests."""

import torch


def assert_activation_bitwise(label: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Require finite FP8 codes or FP32 scales and identical storage bits."""
    assert expected.shape == actual.shape and expected.dtype == actual.dtype, (
        f"{label}: activation metadata differ"
    )
    assert expected.dtype in (torch.float8_e4m3fn, torch.float32), (
        f"{label}: expected FP8 activation or FP32 scale"
    )
    for name, tensor in (("expected", expected), ("actual", actual)):
        assert bool(torch.isfinite(tensor.float()).all()), f"{label}: {name} is non-finite"
    left, right = expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
    if not torch.equal(left, right):
        mismatch = int((left != right).sum().item())
        raise AssertionError(f"{label}: {mismatch} of {left.numel()} bytes differ")


def assert_bitwise(label: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Require finite BF16 outputs and identical bits, including signed zero."""
    assert expected.shape == actual.shape and expected.dtype == actual.dtype, (
        f"{label}: output metadata differ"
    )
    assert expected.dtype == torch.bfloat16, f"{label}: expected BF16 outputs"
    for name, tensor in (("expected", expected), ("actual", actual)):
        assert bool(torch.isfinite(tensor).all()), f"{label}: {name} is non-finite"
    left, right = expected.view(torch.int16), actual.view(torch.int16)
    if not torch.equal(left, right):
        mismatch = int((left != right).sum().item())
        raise AssertionError(f"{label}: {mismatch} of {left.numel()} elements differ")
