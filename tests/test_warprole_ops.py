"""CPU tests for the warprole weight interleave (plan Task 4)."""
import pytest
import torch

from mok import ops


def test_interleave_w13_pairs_gate_and_up_blocks() -> None:
    experts, inter, k = 3, 512, 256
    gen = torch.Generator().manual_seed(1)
    w13 = torch.randn((experts, 2 * inter, k), generator=gen).to(torch.bfloat16)
    scale = torch.rand((experts, 2 * inter // 128, k // 128), generator=gen)
    weight, wscale = ops.interleave_w13(w13, scale)
    assert weight.shape == w13.shape and wscale.shape == scale.shape
    assert weight.is_contiguous() and wscale.is_contiguous()
    for j in range(inter // 128):
        gate = w13[:, j * 128:(j + 1) * 128]
        up = w13[:, inter + j * 128: inter + (j + 1) * 128]
        assert torch.equal(weight[:, (2 * j) * 128:(2 * j + 1) * 128], gate)
        assert torch.equal(weight[:, (2 * j + 1) * 128:(2 * j + 2) * 128], up)
        assert torch.equal(wscale[:, 2 * j], scale[:, j])
        assert torch.equal(wscale[:, 2 * j + 1], scale[:, inter // 128 + j])


def test_interleave_w13_rejects_bad_shapes() -> None:
    w13 = torch.zeros((2, 384, 128), dtype=torch.bfloat16)   # 2I = 384 is not a multiple of 256
    scale = torch.zeros((2, 3, 1))
    with pytest.raises(ValueError):
        ops.interleave_w13(w13, scale)
    w13 = torch.zeros((2, 512, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        ops.interleave_w13(w13, torch.zeros((2, 4, 2)))
