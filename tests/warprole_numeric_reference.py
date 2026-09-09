"""CPU arithmetic oracle; does not import MoK, SGLang, or their kernels.

The reference retains the documented FP8 block-scale and BF16 rounding
boundaries. FP64 dot products independently evaluate each K128 partial;
explicit FP32 operations implement scale multiplication and block addition.
This is a quantized-operation reference, not an unquantized model oracle.
"""

import torch


def block_gemm(a, b, a_scale, b_scale):
    """One expert: [M,K] @ [N,K]. All inputs must be CPU tensors."""
    assert all(x.device.type == "cpu" for x in (a, b, a_scale, b_scale))
    m, k = a.shape
    n, bk = b.shape
    assert bk == k and k % 128 == 0 and n % 128 == 0
    assert a_scale.shape == (m, k // 128)
    assert b_scale.shape == (n // 128, k // 128)
    result = None
    for group in range(k // 128):
        columns = slice(group * 128, (group + 1) * 128)
        partial = (a[:, columns].double() @ b[:, columns].double().T).float()
        scale = a_scale[:, group, None].float() * b_scale[:, group].float().repeat_interleave(128)[None, :]
        scaled = partial * scale
        result = scaled if result is None else result + scaled
    return result.to(torch.bfloat16)


def swiglu_quant(gate_up, limit=10.0):
    """BF16 asymmetric clamp, FP32 SiLU*up, E4M3 with K128 scales."""
    assert gate_up.device.type == "cpu" and gate_up.dtype == torch.bfloat16
    gate, up = gate_up.chunk(2, dim=-1)
    bf16_limit = torch.tensor(limit, dtype=torch.bfloat16).item()
    gate = gate.float().clamp(max=bf16_limit)
    up = up.float().clamp(min=-bf16_limit, max=bf16_limit)
    value = gate / (1.0 + torch.exp(-gate)) * up
    groups = value.reshape(value.shape[0], -1, 128)
    scales = groups.abs().amax(dim=-1).clamp(min=1e-10) / 448.0
    quantized = (groups * scales.reciprocal().unsqueeze(-1)).clamp(-448.0, 448.0)
    return quantized.reshape_as(value).to(torch.float8_e4m3fn), scales


def metrics(actual, reference):
    actual, reference = actual.cpu(), reference.cpu()
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    x, y = actual.double(), reference.double()
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    difference = x - y
    row_norm = y.norm(dim=-1)
    row_error = difference.norm(dim=-1)
    relative = torch.where(row_norm > 0, row_error / row_norm, row_error)
    return {
        "elements": actual.numel(),
        "exact_fraction": float((actual.view(torch.uint8) == reference.view(torch.uint8)).reshape(actual.numel(), -1).all(-1).double().mean()),
        "max_row_relative_l2": float(relative.max()),
        "max_abs": float(difference.abs().max()),
    }
