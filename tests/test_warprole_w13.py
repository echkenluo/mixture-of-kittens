"""Single-GPU bitwise test for the fused W13 epilogue (plan Task 4).

Reference = split path: contiguous W13 GEMM (bf16 gate_up) followed by SGLang's
silu_and_mul_contig_post_quant kernel.  The fused entry must reproduce both the
FP8 values and the per-K128 scales bit for bit.
Run on one H20 inside the SGLang v0.5.17 container:
    python3 -m pytest tests/test_warprole_w13.py -q
"""
import pytest
import torch

from mok import _C

ENTRIES = {
    "c1s6": "fp8_block_warprole_w13_c1s6_out",
    "c2s4": "fp8_block_warprole_w13_c2s4_out",
}
EXPERTS = 64
ROW_PATTERN = [320, 336, 352, 368, 400, 416, 432, 448]
HIDDEN, INTER = 4096, 2048
SWIGLU_LIMIT = 10.0


def _require_sm90() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("SM90 required")
    return device


def _sglang_activation():
    try:
        from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant_dynamic
        return silu_and_mul_contig_post_quant_dynamic, "dynamic"
    except ImportError:
        pass
    try:
        # SGLang 0.5.17 (image a8-base-cu130): no token count, so pass row slices.
        from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant
        return silu_and_mul_contig_post_quant, "contig"
    except ImportError:
        pass
    try:
        from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant
        return silu_and_mul_contig_post_quant, "static"
    except ImportError:
        pytest.skip("SGLang silu_and_mul_contig_post_quant kernel is not importable")


def make_inputs(device: torch.device, seed: int = 20260904):
    gen = torch.Generator(device=device).manual_seed(seed)
    logical_rows = ROW_PATTERN * (EXPERTS // len(ROW_PATTERN))
    # A tile selects its expert from its first row.  Preserve non-M64 logical
    # lengths but pad EACH expert segment, as the production schedule does.
    rows = [((n + 63) // 64) * 64 for n in logical_rows]
    total_m = sum(rows)
    a = torch.randn((total_m, HIDDEN), generator=gen, device=device, dtype=torch.bfloat16)
    a = a.clamp(-3, 3).to(torch.float8_e4m3fn)
    start = 0
    for logical, padded in zip(logical_rows, rows):
        a[start + logical:start + padded].zero_()
        start += padded
    w13 = torch.randn((EXPERTS, 2 * INTER, HIDDEN), generator=gen, device=device, dtype=torch.bfloat16)
    w13 = w13.clamp(-3, 3).to(torch.float8_e4m3fn)
    a_scale = torch.rand((total_m, HIDDEN // 128), generator=gen, device=device) * 0.09 + 0.01
    w13_scale = torch.rand((EXPERTS, 2 * INTER // 128, HIDDEN // 128), generator=gen, device=device) * 0.09 + 0.01
    m_indices = torch.repeat_interleave(
        torch.arange(EXPERTS, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int64, device=device),
    ).contiguous()
    return a, w13, a_scale, w13_scale, m_indices, total_m


def reference(device, a, w13, a_scale, w13_scale, m_indices, total_m, num_tokens):
    activation, kind = _sglang_activation()
    gate_up = torch.empty((total_m, 2 * INTER), dtype=torch.bfloat16, device=device)
    _C.fp8_block_grouped_contiguous_dynamic_out(a, w13, a_scale, w13_scale, m_indices, num_tokens, gate_up)
    hidden = torch.zeros((total_m, INTER), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.zeros((total_m, INTER // 128), dtype=torch.float32, device=device)
    if kind == "dynamic":
        activation(input=gate_up, output=hidden, output_scale=scale, active_tokens=num_tokens,
                   quant_group_size=128, scale_ue8m0=False, transposed=False,
                   swiglu_limit=SWIGLU_LIMIT, swizzle=False)
    elif kind == "contig":
        n = int(num_tokens.item())
        activation(gate_up[:n], hidden[:n], scale[:n], 128, False, False, SWIGLU_LIMIT, False)
    else:
        activation(gate_up, hidden, scale, int(num_tokens.item()), 128, False, False,
                   SWIGLU_LIMIT, False)
    return hidden, scale


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_warprole_w13_bitwise(entry: str) -> None:
    device = _require_sm90()
    a, w13, a_scale, w13_scale, m_indices, total_m = make_inputs(device)
    num_tokens = torch.tensor([total_m], dtype=torch.int32, device=device)
    hidden_ref, scale_ref = reference(device, a, w13, a_scale, w13_scale, m_indices, total_m, num_tokens)
    hidden = torch.zeros_like(hidden_ref)
    scale = torch.zeros_like(scale_ref)
    getattr(_C, ENTRIES[entry])(a, a_scale, w13, w13_scale, m_indices, num_tokens, hidden, scale, SWIGLU_LIMIT)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(hidden.float()).all()) and bool(torch.isfinite(scale).all())
    assert bool(torch.isfinite(hidden_ref.float()).all()) and bool(torch.isfinite(scale_ref).all())
    assert torch.equal(scale, scale_ref), f"{entry}: activation scales differ from the split path"
    assert torch.equal(hidden.view(torch.uint8), hidden_ref.view(torch.uint8)), \
        f"{entry}: FP8 activations differ from the split path"


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_warprole_w13_respects_num_tokens(entry: str) -> None:
    device = _require_sm90()
    a, w13, a_scale, w13_scale, m_indices, total_m = make_inputs(device)
    active = 64 * 50
    num_tokens = torch.tensor([active], dtype=torch.int32, device=device)
    hidden_ref, scale_ref = reference(device, a, w13, a_scale, w13_scale, m_indices, total_m, num_tokens)
    hidden = torch.zeros_like(hidden_ref)
    scale = torch.full_like(scale_ref, -1.0)
    getattr(_C, ENTRIES[entry])(a, a_scale, w13, w13_scale, m_indices, num_tokens, hidden, scale, SWIGLU_LIMIT)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(hidden[:active].float()).all())
    assert bool(torch.isfinite(hidden_ref[:active].float()).all())
    assert bool(torch.isfinite(scale[:active]).all()) and bool(torch.isfinite(scale_ref[:active]).all())
    assert torch.equal(scale[:active], scale_ref[:active])
    assert torch.equal(hidden[:active].view(torch.uint8), hidden_ref[:active].view(torch.uint8))
    assert bool((scale[active:] == -1.0).all()), "rows beyond num_tokens must stay untouched"
