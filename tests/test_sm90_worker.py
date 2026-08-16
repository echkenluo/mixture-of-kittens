import pytest
import torch

from mok import _C


def require_sm90(device: torch.device) -> None:
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("SM90 worker contract requires Hopper compute capability 9.0")
    assert hasattr(_C, "sm90_worker_test"), "SM90 build did not register sm90_worker_test"
    assert hasattr(_C, "sm90_fp8_block_test"), (
        "SM90 build did not register sm90_fp8_block_test"
    )


@pytest.mark.parametrize("is_ab", [True, False], ids=["AB", "ABt"])
@pytest.mark.parametrize("staged", [False, True], ids=["full", "staged"])
@pytest.mark.parametrize("k", [256, 4096])
def test_sm90_worker_numeric(
    context: tuple[int, int, torch.device], is_ab: bool, staged: bool, k: int
) -> None:
    rank, _, device = context
    require_sm90(device)

    # Run allocation, producer ops, the extension launch, and the reference on
    # a non-default stream. A helper that silently launches on stream 0 races
    # these inputs and fails this test rather than receiving accidental credit.
    stream = torch.cuda.Stream(device=device)
    generator = torch.Generator(device=device).manual_seed(
        20260814 + 100 * rank + 10 * int(staged) + int(is_ab)
    )
    with torch.cuda.stream(stream):
        a = torch.randn((128, k), generator=generator, device=device, dtype=torch.bfloat16)
        b_shape = (k, 128) if is_ab else (128, k)
        b = torch.randn(b_shape, generator=generator, device=device, dtype=torch.bfloat16)
        actual = _C.sm90_worker_test(a, b, is_ab, staged)
        reference = (a.float() @ (b.float() if is_ab else b.float().T)).to(torch.bfloat16)
    stream.synchronize()

    assert actual.dtype == torch.bfloat16
    assert actual.device == device
    assert torch.isfinite(actual).all()
    max_rel = (actual.float() - reference.float()).abs().max() / reference.float().abs().max()
    zero_frac = (actual == 0).float().mean()
    assert max_rel.item() < 0.02, f"max_rel={max_rel.item():.6f}"
    assert zero_frac.item() < 0.01, f"zero_frac={zero_frac.item():.6f}"


def test_sm90_worker_rejects_invalid_inputs(
    context: tuple[int, int, torch.device]
) -> None:
    _, _, device = context
    require_sm90(device)
    good = torch.randn((128, 256), device=device, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="CUDA device"):
        _C.sm90_worker_test(good.cpu(), good, True, False)
    with pytest.raises(RuntimeError, match="bfloat16"):
        _C.sm90_worker_test(good.float(), good, True, False)
    with pytest.raises(RuntimeError, match="contiguous"):
        _C.sm90_worker_test(good.T, good, True, False)
    with pytest.raises(RuntimeError, match="AB expects"):
        _C.sm90_worker_test(good, good[:, :128].contiguous(), True, False)


@pytest.mark.parametrize("k", [128, 4096])
@pytest.mark.parametrize("scaled", [False, True], ids=["unit-scale", "block-scale"])
def test_sm90_fp8_block_numeric(
    context: tuple[int, int, torch.device], k: int, scaled: bool
) -> None:
    rank, _, device = context
    require_sm90(device)

    stream = torch.cuda.Stream(device=device)
    generator = torch.Generator(device=device).manual_seed(
        20260816 + 100 * rank + k + int(scaled)
    )
    k_blocks = k // 128
    with torch.cuda.stream(stream):
        # Keep values well inside E4M3 range.  The reference consumes the
        # already-quantized tensors, so this isolates WGMMA and block scaling
        # rather than attributing quantization error to the kernel.
        a = torch.randn((64, k), generator=generator, device=device).clamp(-3, 3)
        b = torch.randn((64, k), generator=generator, device=device).clamp(-3, 3)
        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)
        if scaled:
            a_scale = torch.rand(
                (64, k_blocks), generator=generator, device=device
            ) * 0.09 + 0.01
            b_scale = torch.rand(
                (k_blocks,), generator=generator, device=device
            ) * 0.09 + 0.01
        else:
            a_scale = torch.ones((64, k_blocks), device=device)
            b_scale = torch.ones((k_blocks,), device=device)

        actual = _C.sm90_fp8_block_test(a, b, a_scale, b_scale)
        reference = torch.zeros((64, 64), device=device, dtype=torch.float32)
        for kb in range(k_blocks):
            sl = slice(kb * 128, (kb + 1) * 128)
            partial = a[:, sl].float() @ b[:, sl].float().T
            reference.add_(partial * a_scale[:, kb, None] * b_scale[kb])
        reference = reference.to(torch.bfloat16)
    stream.synchronize()

    assert actual.dtype == torch.bfloat16
    assert actual.device == device
    assert torch.isfinite(actual).all()
    abs_error = (actual.float() - reference.float()).abs()
    max_rel = abs_error.max() / reference.float().abs().max().clamp_min(1e-6)
    assert max_rel.item() < 0.025, f"max_rel={max_rel.item():.6f}"


def test_sm90_fp8_block_rejects_invalid_inputs(
    context: tuple[int, int, torch.device]
) -> None:
    _, _, device = context
    require_sm90(device)
    a = torch.ones((64, 128), device=device).to(torch.float8_e4m3fn)
    b = torch.ones_like(a)
    a_scale = torch.ones((64, 1), device=device)
    b_scale = torch.ones((1,), device=device)

    with pytest.raises(RuntimeError, match="fp8e4m3"):
        _C.sm90_fp8_block_test(a.float(), b, a_scale, b_scale)
    with pytest.raises(RuntimeError, match="expected A"):
        _C.sm90_fp8_block_test(a, b[:, :64].contiguous(), a_scale, b_scale)
    with pytest.raises(RuntimeError, match="K must"):
        _C.sm90_fp8_block_test(
            a[:, :64].contiguous(), b[:, :64].contiguous(), a_scale, b_scale
        )
    with pytest.raises(RuntimeError, match="A_scale must have shape"):
        _C.sm90_fp8_block_test(a, b, a_scale[:, :0], b_scale)
    with pytest.raises(RuntimeError, match="B_scale must have shape"):
        _C.sm90_fp8_block_test(a, b, a_scale, b_scale.view(1, 1))
    with pytest.raises(RuntimeError, match="float32"):
        _C.sm90_fp8_block_test(a, b, a_scale.bfloat16(), b_scale)
