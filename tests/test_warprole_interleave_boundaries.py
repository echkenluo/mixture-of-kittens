"""Exact Q=1/2/3 communication boundaries, including an empty receiving rank.

The historical random small-token fixture can still create four or more
minibatches after expert padding. These routes force the intended geometry.
Run with MOK_WARPROLE_INTERLEAVE=1 after the same-binary default-path control.
"""
import pytest
import torch

from . import test_warprole_ep4 as base


@pytest.mark.parametrize("variant", base.VARIANTS)
@pytest.mark.parametrize("tokens,batches", [(64, 1), (192, 2), (384, 3)])
@pytest.mark.parametrize("empty_sender", [False, True])
def test_interleave_boundaries(context, monkeypatch, variant, tokens, batches, empty_sender):
    rank, world, device = context
    base.require_warprole(device)
    assert world == base.EP_SIZE

    def fixed_routes(case, source_rank, device):
        # Every sender targets the next rank's first six experts: every live
        # dispatch/combine crosses a rank boundary, even for a single minibatch.
        first = ((source_rank + 1) % world) * base.LOCAL_EXPERTS
        ids = (torch.arange(base.TOPK, dtype=torch.int32, device=device) + first)
        ids = ids.expand(case.graph_tokens, base.TOPK).clone()
        weights = torch.full(ids.shape, 1.0 / base.TOPK, dtype=torch.float32, device=device)
        if empty_sender and source_rank == world - 1:
            ids.fill_(-1)
            weights.zero_()
        return ids, weights

    monkeypatch.setattr(base, "make_routing", fixed_routes)
    name = f"interleave_q{batches}_empty{int(empty_sender)}"
    harness = base.build_harness(base.Case(name, tokens, 1.5), rank, device, fresh=True)
    expected = 0 if empty_sender and rank == 0 else tokens * base.TOPK
    assert harness.active_rows == expected
    assert (harness.active_rows + 1023) // 1024 == (0 if expected == 0 else batches)
    base.compare_arms(harness, variant, name)
