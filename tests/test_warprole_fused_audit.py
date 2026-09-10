"""Actual megakernel capture, repeated-call ownership, and epilogue replay.

Run on a free EP4 H20 group in the pinned SGLang image. These are synthetic
integration tests, not the 43-layer live numeric gate or a performance run.
"""
import pytest
import torch
import torch.distributed as dist

from mok import _C, functional, warprole
from .test_warprole_ep4 import (
    CASES, EP_SIZE, LOCAL_EXPERTS, EXPERT_PADDING, SWIGLU_LIMIT,
    build_harness, run_warprole, require_warprole, sglang_activation,
)
from .warprole_assertions import assert_bitwise


@pytest.mark.parametrize('case_name', ['small_64', 'skew_512', 'all_padding_256'])
def test_actual_fused_w13_capture(context, case_name):
    rank, world_size, device = context
    require_warprole(device)
    assert world_size == EP_SIZE
    assert hasattr(_C, 'fp8_block_warprole_c2s4_audit_out')
    h = build_harness(CASES[case_name], rank, device)
    capture = torch.full_like(h.gate_up, float('nan'))
    original_capture = None
    for repeat in range(2):
        if repeat:
            h.x_scale.mul_(1.125)
        ordinary = run_warprole(h, 'c2s4').clone()
        torch.cuda.synchronize(); dist.barrier()
        functional.acquire_workspace_lease(h.workspace)
        schedule = functional.build_schedule(h.workspace, h.config, h.topk_ids,
            num_local_experts=LOCAL_EXPERTS, expert_padding=EXPERT_PADDING)
        n = int(schedule.num_tokens.item())
        borrowed = warprole.warprole_forward_leased(
            h.workspace, h.state, schedule, h.x_fp8, h.x_scale, h.router_weights,
            h.topk_ids, h.w13, h.w13_scale, h.w2, h.w2_scale,
            variant='c2s4', swiglu_limit=SWIGLU_LIMIT, audit_w13=capture)
        output = borrowed.clone()
        hidden = h.state.hidden.clone(); scale = h.state.hidden_scale.clone()
        torch.cuda.synchronize()
        functional.release_workspace_lease(h.workspace)
        dist.barrier()
        assert_bitwise('capture preserves final output', ordinary, output)
        assert bool(torch.isfinite(capture[:n]).all())
        assert bool(torch.isnan(capture[n:]).all()), 'inactive capacity was written'
        # Reusing the destination must capture the current invocation, not
        # leave a plausible but stale tile from the previous launch.
        if repeat == 0:
            original_capture = capture[:n].cpu().clone()
        elif n:
            assert not torch.equal(capture[:n].cpu(), original_capture)
        replay = torch.empty_like(capture)
        _C.fp8_block_warprole_gemm_c2s4_out(h.state.routed_x, h.w13,
            h.state.routed_x_scale, h.w13_scale, h.state.m_indices, schedule.num_tokens, replay)
        activation, kind = sglang_activation()
        assert kind == 'dynamic', 'use the pinned current SGLang activation'
        ref_hidden = torch.empty_like(hidden); ref_scale = torch.empty_like(scale)
        activation(input=capture, output=ref_hidden, output_scale=ref_scale,
            active_tokens=schedule.num_tokens, quant_group_size=128, scale_ue8m0=False,
            transposed=False, swiglu_limit=SWIGLU_LIMIT, swizzle=False)
        torch.cuda.synchronize(); dist.barrier()
        assert_bitwise('actual fused W13 versus primitive', capture[:n], replay[:n])
        assert_bitwise('actual gate/up reproduce hidden', hidden[:n], ref_hidden[:n])
        assert_bitwise('actual gate/up reproduce scales', scale[:n], ref_scale[:n])
