"""Full-output EP4/EP8 route semantics against analytic expert matrices.

T64, E256, top6, rank-specific inputs; all experts covered. Both variants run
uniform and padding cases. The reference never reads the candidate schedule,
dispatch buffers, hidden state, or split output. This is synthetic correctness,
not dense random GEMM accuracy or model quality.
"""

import json

import pytest
import torch

from .test_warprole_ep4 import (
    Case, HIDDEN, LOCAL_EXPERTS, TOTAL_EXPERTS, TOPK,
    build_harness, run_warprole,
)
from .warprole_numeric_reference import metrics
from .warprole_route_reference import block_scales, point_parameters, routed_forward


def install_sparse_weights(harness):
    global_experts = torch.arange(LOCAL_EXPERTS) + harness.rank * LOCAL_EXPERTS
    for stage, weight, scale in (
        ("w13", harness.w13, harness.w13_scale),
        ("w2", harness.w2, harness.w2_scale),
    ):
        columns, values = point_parameters(global_experts, stage)
        _, n, k = weight.shape
        offsets = (torch.arange(LOCAL_EXPERTS)[:, None] * n * k
                   + torch.arange(n)[None, :] * k + columns)
        weight.zero_()
        weight.view(torch.uint8).view(-1).index_copy_(
            0, offsets.reshape(-1).to(weight.device),
            values.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1).to(weight.device))
        scale.copy_(block_scales(global_experts, stage))


@pytest.mark.parametrize("variant", ("c1s6", "c2s4"))
@pytest.mark.parametrize("padding", (False, True))
def test_independent_routing(context, variant, padding):
    rank, world, device = context
    assert world in (4, 8) and TOTAL_EXPERTS == 256
    torch.set_num_threads(4)
    harness = build_harness(Case("independent_route_64", 64, 1.5), rank, device)
    install_sparse_weights(harness)
    token = torch.arange(64, device=device)[:, None]
    slot = torch.arange(TOPK, device=device)[None, :]
    ids = ((rank * 37 + token * TOPK + slot) % TOTAL_EXPERTS).to(torch.int32)
    if padding:
        ids[::5, 0] = -1
        ids[::7, 2:5] = -1
        ids[::11] = -1
    harness.topk_ids.copy_(ids)
    # Keep nonzero weights even for padding; the reducer must honor ids.
    weights = (slot.float() + 1).expand(64, -1) / 21
    harness.router_weights.copy_(weights)
    # Vary signs and magnitudes by row without changing the FP8 input bytes.
    harness.x_scale.copy_(torch.full_like(harness.x_scale, 1 / 128))
    harness.x_scale[1].mul_(128)
    harness.x_scale[2].zero_()
    expected = routed_forward(
        harness.x_fp8.cpu(), harness.x_scale.cpu(), ids.cpu(), weights.cpu())
    actual = run_warprole(harness, variant).cpu()
    result = metrics(actual, expected)
    print("INDEPENDENT_ROUTES " + json.dumps({
        "rank": rank, "world": world, "variant": variant, "padding": padding,
        "global_experts_in_source_routes": int(ids[ids >= 0].unique().numel()), **result,
    }), flush=True)
    assert result["exact_fraction"] >= .999, result
    assert result["max_row_relative_l2"] <= .001, result
    assert bool((actual[2] == 0).all())
    if padding:
        assert bool((actual[::11] == 0).all())
