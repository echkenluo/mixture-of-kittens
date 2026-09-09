"""Analytic sparse-expert oracle, independent of schedule and MoK kernels.

Each expert matrix has one nonzero per output channel. This makes a full
distributed forward independently calculable by indexing and scalar products,
while the candidate still executes its ordinary dense FP8 GEMMs.
"""

import torch

from .warprole_numeric_reference import swiglu_quant


HIDDEN, INTERMEDIATE, TOPK = 4096, 2048, 6


def point_parameters(experts, stage):
    """Return nonzero input-column/value for every output row, on CPU."""
    experts = torch.as_tensor(experts, dtype=torch.int64).reshape(-1, 1)
    rows = torch.arange(4096).reshape(1, -1)
    if stage == "w13":
        columns = (17 * rows + 13 * experts) % HIDDEN
        sign = 1 - 2 * ((rows // 128 + experts) % 2)
        values = sign * (1.0 + (experts % 4) * .25)
    elif stage == "w2":
        columns = (7 * rows + 11 * experts) % INTERMEDIATE
        sign = 1 - 2 * ((rows // 128 + experts // 4) % 2)
        values = sign * (1.0 + (experts % 3) * .5)
    else:
        raise ValueError(stage)
    return columns, values.float()


def block_scales(experts, stage):
    experts = torch.as_tensor(experts, dtype=torch.int64).reshape(-1, 1, 1)
    n = torch.arange(32).reshape(1, -1, 1)
    k = torch.arange(32 if stage == "w13" else 16).reshape(1, 1, -1)
    exponent = (-3 + (experts + n + 2 * k) % 3 if stage == "w13"
                else -2 + (2 * experts + n + k) % 3)
    return torch.pow(2.0, exponent).float()


def point_gemm(x, x_scale, experts, stage):
    """One selected global expert per row, preserving FP32/BF16 boundaries."""
    assert x.device.type == x_scale.device.type == "cpu"
    experts = torch.as_tensor(experts, dtype=torch.int64)
    columns, values = point_parameters(experts, stage)
    n_blocks = torch.arange(4096).reshape(1, -1) // 128
    k_blocks = columns // 128
    if stage == "w13":
        powers = -3 + (experts[:, None] + n_blocks + 2 * k_blocks) % 3
    else:
        powers = -2 + (2 * experts[:, None] + n_blocks + k_blocks) % 3
    scales = x_scale.gather(1, k_blocks) * torch.pow(2.0, powers).float()
    partial = x.float().gather(1, columns) * values
    result = partial * scales
    # The dense dot and subsequent K-block additions include positive zero
    # terms. A zero point contribution therefore finishes as +0, unlike an
    # isolated negative scalar multiplied by a zero activation scale.
    result[result == 0] = 0.0
    return result.to(torch.bfloat16)


def routed_forward(x, x_scale, topk_ids, router_weights):
    """Compute this source rank's complete output without cross-rank buffers."""
    assert all(t.device.type == "cpu" for t in (x, x_scale, topk_ids, router_weights))
    tokens = x.shape[0]
    ids = topk_ids.long().reshape(-1)
    sources = torch.arange(tokens).repeat_interleave(TOPK)
    gate_up = point_gemm(x.float()[sources], x_scale[sources], ids.clamp(min=0), "w13")
    hidden, scales = swiglu_quant(gate_up)
    routed = point_gemm(hidden, scales, ids.clamp(min=0), "w2").reshape(tokens, TOPK, HIDDEN)
    total = torch.zeros(tokens, HIDDEN, dtype=torch.float32)
    initialized = torch.zeros(tokens, dtype=torch.bool)
    for slot in range(TOPK):
        valid = topk_ids[:, slot] >= 0
        value = routed[:, slot].float()
        weight = router_weights[:, slot, None]
        # Evaluate product/add in FP64 and round once to approximate the FP32
        # FMA independently; no candidate reduction code is reused here.
        first = value * weight
        following = (value.double() * weight.double() + total.double()).float()
        next_value = torch.where(initialized[:, None], following, first)
        total = torch.where(valid[:, None], next_value, total)
        initialized |= valid
    return total.to(torch.bfloat16)
