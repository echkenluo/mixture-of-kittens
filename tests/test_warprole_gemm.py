"""Single-GPU bitwise tests for the warp-role GEMM primitive (plan Task 2).

Every standalone entry must reproduce `fp8_block_grouped_contiguous_dynamic_out`
bit for bit: same WGMMA K128 partials, same A_scale*B_scale promotion order.
Run on one H20:  python3 -m pytest tests/test_warprole_gemm.py -q
"""
import pytest
import torch

from mok import _C

ENTRIES = {
    "c1s6": "fp8_block_warprole_gemm_c1s6_out",
    "c2s4": "fp8_block_warprole_gemm_c2s4_out",
}
EXPERTS = 64
ROW_PATTERN = [320, 336, 352, 368, 400, 416, 432, 448]   # mean 384 rows/expert, 24576 rows
SHAPES = {"w13": (4096, 4096), "w2": (4096, 2048)}         # (N, K)


def _require_sm90() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("SM90 required")
    return device


def make_inputs(device: torch.device, n: int, k: int, seed: int = 20260904):
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = ROW_PATTERN * (EXPERTS // len(ROW_PATTERN))
    total_m = sum(rows)
    assert total_m % 64 == 0
    k_blocks = k // 128
    a = torch.randn((total_m, k), generator=gen, device=device, dtype=torch.bfloat16)
    a = a.clamp(-3, 3).to(torch.float8_e4m3fn)
    b = torch.randn((EXPERTS, n, k), generator=gen, device=device, dtype=torch.bfloat16)
    b = b.clamp(-3, 3).to(torch.float8_e4m3fn)
    a_scale = torch.rand((total_m, k_blocks), generator=gen, device=device) * 0.09 + 0.01
    b_scale = torch.rand((EXPERTS, n // 128, k_blocks), generator=gen, device=device) * 0.09 + 0.01
    m_indices = torch.repeat_interleave(
        torch.arange(EXPERTS, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int64, device=device),
    ).contiguous()
    return a, b, a_scale, b_scale, m_indices, total_m


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_warprole_gemm_bitwise(entry: str, shape: str) -> None:
    device = _require_sm90()
    n, k = SHAPES[shape]
    a, b, a_scale, b_scale, m_indices, total_m = make_inputs(device, n, k)
    num_tokens = torch.tensor([total_m], dtype=torch.int32, device=device)
    ref = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)
    _C.fp8_block_grouped_contiguous_dynamic_out(a, b, a_scale, b_scale, m_indices, num_tokens, ref)
    out = torch.full((total_m, n), 12345.0, dtype=torch.bfloat16, device=device)
    getattr(_C, ENTRIES[entry])(a, b, a_scale, b_scale, m_indices, num_tokens, out)
    torch.cuda.synchronize()
    assert torch.isfinite(ref.float()).all()
    assert torch.equal(out, ref), f"{entry}/{shape}: output differs from the split GEMM"


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_warprole_gemm_respects_num_tokens(entry: str) -> None:
    device = _require_sm90()
    n, k = SHAPES["w2"]
    a, b, a_scale, b_scale, m_indices, total_m = make_inputs(device, n, k)
    active = 64 * 100
    num_tokens = torch.tensor([active], dtype=torch.int32, device=device)
    ref = torch.full((total_m, n), 12345.0, dtype=torch.bfloat16, device=device)
    _C.fp8_block_grouped_contiguous_dynamic_out(a, b, a_scale, b_scale, m_indices, num_tokens, ref)
    out = torch.full((total_m, n), 12345.0, dtype=torch.bfloat16, device=device)
    getattr(_C, ENTRIES[entry])(a, b, a_scale, b_scale, m_indices, num_tokens, out)
    torch.cuda.synchronize()
    assert torch.equal(out[:active], ref[:active])
    assert bool((out[active:] == 12345.0).all()), "rows beyond num_tokens must stay untouched"


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_warprole_gemm_rejects_bad_n(entry: str) -> None:
    device = _require_sm90()
    a, b, a_scale, b_scale, m_indices, total_m = make_inputs(device, 4096, 2048)
    num_tokens = torch.tensor([total_m], dtype=torch.int32, device=device)
    out = torch.empty((total_m, 4096), dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError):
        getattr(_C, ENTRIES[entry])(a, b[:, :192].contiguous(), a_scale, b_scale[:, :1].contiguous(),
                                     m_indices, num_tokens, out[:, :192].contiguous())
