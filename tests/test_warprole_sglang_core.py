"""Run the actual SGLang native core with synthetic expert weights on GPUs.

This covers SGLang input quantization, split/warp-role selection and the borrowed
output/lease contract. It uses real SGLang process groups and kernels, with no
mocked operations. It does not cover model loading, admission, HTTP serving,
shared experts, graph replay, or model quality.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from . import test_warprole_ep4 as base


@pytest.fixture(scope="module")
def sglang_context(context):
    from sglang.srt.distributed import parallel_state

    rank, world_size, device = context
    parallel_state.set_custom_all_reduce(False)
    parallel_state.init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=device.index, backend="nccl",
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        expert_model_parallel_size=world_size,
        backend="nccl",
    )
    yield context
    dist.barrier()
    parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("tokens", (256, 1792))
@pytest.mark.parametrize("variant", base.VARIANTS)
def test_real_sglang_core(sglang_context, tokens, variant):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner import mok_fp8_native as adapter

    rank, _, device = sglang_context
    harness = base.build_harness(base.Case(f"sglang_core_{tokens}", tokens, 1.0),
                                 rank, device, fresh=True,
                                 group=adapter.get_tp_group().device_group)
    layer = SimpleNamespace(
        layer_id=0, num_local_experts=base.LOCAL_EXPERTS,
        w13_weight=harness.w13, w13_weight_scale_inv=harness.w13_scale,
        w2_weight=harness.w2, w2_weight_scale_inv=harness.w2_scale,
        moe_runner_config=SimpleNamespace(swiglu_limit=base.SWIGLU_LIMIT),
    )
    hidden = torch.randn(
        (tokens, base.HIDDEN), device=device, dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(base.SEED + rank),
    )

    def invoke(x, ids, weights, use_warprole):
        with (envs.SGLANG_OPT_MOK_WARPROLE.override(use_warprole),
              envs.SGLANG_OPT_MOK_WARPROLE_VARIANT.override(variant)):
            borrowed = adapter._run_native_core(
                layer, harness.workspace, base.functional, harness.config,
                x, ids, weights,
            )
            # The real core must return while retaining ownership. The outer
            # service boundary clones then releases; exercise that contract.
            assert int(harness.workspace.in_use.item()) == 1
            owned = borrowed.clone()
            base.functional.release_workspace_lease(harness.workspace)
            assert int(harness.workspace.in_use.item()) == 0
            return owned

    ids, weights = harness.topk_ids, harness.router_weights
    expected_first = invoke(hidden, ids, weights, False)
    first = invoke(hidden, ids, weights, True)
    changed_hidden = -hidden * 0.5
    changed_ids = ((ids + base.LOCAL_EXPERTS) % base.TOTAL_EXPERTS).contiguous()
    changed_weights = weights.roll(1, dims=1).contiguous()
    second = invoke(changed_hidden, changed_ids, changed_weights, True)
    expected_second = invoke(changed_hidden, changed_ids, changed_weights, False)
    base.assert_bitwise(f"sglang/{tokens}/{variant}/first", expected_first, first)
    base.assert_bitwise(f"sglang/{tokens}/{variant}/second", expected_second, second)
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first.view(torch.int16), second.view(torch.int16))
    base.assert_trap_clear(harness, f"sglang/{tokens}/{variant}")
    dist.barrier()
