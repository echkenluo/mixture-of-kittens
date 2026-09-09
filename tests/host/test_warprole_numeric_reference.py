"""Analytic checks of the independent oracle, executable without MoK/CUDA."""

import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "numeric_reference", Path(__file__).resolve().parents[1] / "warprole_numeric_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_block_scale_axes_and_addition():
    a = torch.ones(2, 256)
    b = torch.ones(256, 256)
    a_scale = torch.tensor([[1., 2.], [3., 4.]])
    b_scale = torch.tensor([[5., 6.], [7., 8.]])
    result = reference.block_gemm(a, b, a_scale, b_scale)
    expected = torch.tensor([[2176., 2944.], [4992., 6784.]], dtype=torch.bfloat16).repeat_interleave(128, dim=1)
    assert torch.equal(result, expected)


def test_asymmetric_clamp_zero_and_scale_floor():
    gate = torch.tensor([-20., 20., 0.], dtype=torch.bfloat16)[:, None].expand(3, 128)
    up = torch.tensor([-20., 20., 1.], dtype=torch.bfloat16)[:, None].expand(3, 128)
    q, scale = reference.swiglu_quant(torch.cat((gate, up), dim=1))
    expected = torch.tensor([-20., 10., 0.])
    expected = expected / (1 + (-expected).exp()) * torch.tensor([-10., 10., 1.])
    assert torch.allclose(q.float() * scale, expected[:, None], atol=1e-10, rtol=1e-6)
    assert float(scale[2]) > 0 and bool((q[2].float() == 0).all())


def test_metrics_reject_nonfinite_and_detect_error():
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    assert reference.metrics(x, x)["exact_fraction"] == 1
    altered = x.clone()
    altered[0] *= 2
    assert reference.metrics(altered, x)["max_row_relative_l2"] == 1
    altered[0, 0] = float("nan")
    try:
        reference.metrics(altered, x)
    except AssertionError:
        pass
    else:
        raise AssertionError("nonfinite input was accepted")
