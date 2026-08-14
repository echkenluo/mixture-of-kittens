import pytest
import torch

from mok import _C


def require_sm90(device: torch.device) -> None:
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("SM90 worker contract requires Hopper compute capability 9.0")
    assert hasattr(_C, "sm90_worker_test"), "SM90 build did not register sm90_worker_test"


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
