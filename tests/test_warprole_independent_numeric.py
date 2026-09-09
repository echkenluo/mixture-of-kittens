"""SM90 stage tests against a CPU oracle independent of both MoK arms.

Two experts in reversed row order, production W13/W2 shapes, all N/K blocks,
eight preregistered sampled rows, zero/large/random activation scales and
inactive-row sentinels. This does not cover distributed routing or full model
quality. Thresholds: exact >= .999 and max row relative L2 <= .001; W13
quantization scales additionally require max relative error <= 1e-5.
"""

import json

import pytest
import torch

from mok import _C
from .warprole_numeric_reference import block_gemm, metrics, swiglu_quant


SAMPLES = (0, 7, 31, 63, 64, 79, 95, 127)


def inputs(k):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    generator = torch.Generator().manual_seed(20260909 + k)
    a = torch.randn(192, k, generator=generator).clamp(-3, 3).to(torch.float8_e4m3fn)
    b = torch.randn(2, 4096, k, generator=generator).clamp(-3, 3).to(torch.float8_e4m3fn)
    a_scale = torch.rand(192, k // 128, generator=generator) * .09 + .01
    b_scale = torch.rand(2, 32, k // 128, generator=generator) * .09 + .01
    # Zero and large values exercise the scale floor and asymmetric clamp.
    a_scale[0].zero_()
    a_scale[7] *= 100
    a_scale[79] *= 100
    indices = torch.tensor([1] * 64 + [0] * 64 + [1] * 64, dtype=torch.int32)
    return a, b, a_scale, b_scale, indices


def oracle_rows(data):
    a, b, a_scale, b_scale, indices = data
    rows = []
    for expert in (1, 0):
        selected = [i for i in SAMPLES if int(indices[i]) == expert]
        rows.append(block_gemm(a.float()[selected], b[expert], a_scale[selected], b_scale[expert]))
    return torch.cat(rows)


def check(actual, expected, tag):
    receipt = metrics(actual, expected)
    print("INDEPENDENT_NUMERIC " + json.dumps({"tag": tag, **receipt}), flush=True)
    assert receipt["exact_fraction"] >= .999, receipt
    assert receipt["max_row_relative_l2"] <= .001, receipt


@pytest.mark.parametrize("variant", ("c1s6", "c2s4"))
@pytest.mark.parametrize("stage,k", (("w13", 4096), ("w2", 2048)))
def test_gemm_cpu_oracle(variant, stage, k):
    torch.set_num_threads(4)
    data = inputs(k)
    expected = oracle_rows(data)
    a, b, a_scale, b_scale, indices = [x.cuda() for x in data]
    active = torch.tensor([128], dtype=torch.int32, device="cuda")
    out = torch.full((192, 4096), 12345.0, dtype=torch.bfloat16, device="cuda")
    getattr(_C, f"fp8_block_warprole_gemm_{variant}_out")(
        a, b, a_scale, b_scale, indices, active, out)
    actual = out.cpu()
    check(actual[list(SAMPLES)], expected, f"{variant}/{stage}")
    assert bool((actual[128:] == 12345.0).all()), "inactive rows overwritten"


@pytest.mark.parametrize("variant", ("c1s6", "c2s4"))
def test_w13_activation_cpu_oracle(variant):
    torch.set_num_threads(4)
    data = inputs(4096)
    expected, expected_scale = swiglu_quant(oracle_rows(data))
    a, b, a_scale, b_scale, indices = [x.cuda() for x in data]
    active = torch.tensor([128], dtype=torch.int32, device="cuda")
    hidden = torch.ones((192, 2048), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.full((192, 16), -1.0, device="cuda")
    getattr(_C, f"fp8_block_warprole_w13_{variant}_out")(
        a, a_scale, b, b_scale, indices, active, hidden, scales, 10.0)
    actual, actual_scale = hidden.cpu(), scales.cpu()
    check(actual.float()[list(SAMPLES)].to(torch.float8_e4m3fn), expected, f"{variant}/activation")
    selected_scale = actual_scale[list(SAMPLES)]
    assert bool(torch.isfinite(selected_scale).all()) and bool((selected_scale > 0).all())
    relative = float(((selected_scale - expected_scale).abs() / expected_scale).max())
    print("INDEPENDENT_SCALE " + json.dumps({"variant": variant, "max_relative": relative}), flush=True)
    assert relative <= 1e-5
    assert bool((actual[128:].float() == 1.0).all())
    assert bool((actual_scale[128:] == -1.0).all())
