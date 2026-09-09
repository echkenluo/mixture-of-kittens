"""Extended state reuse without a split/reference call between candidate calls.

Run separately from the frozen 18-case EP matrix. Each variant runs 256 calls
on one workspace, cycling through eight precomputed references. This checks
counter reset, changing routes/input, empty input ranks, maximal expert-rank
skew, and output ownership; it is not a duration or performance benchmark.
"""

import pytest
import torch
import torch.distributed as dist

from . import test_warprole_ep4 as base


ITERATIONS = 256
PATTERNS = 8


def make_patterns(harness):
    patterns = []
    for index in range(PATTERNS):
        generator = torch.Generator(device=harness.device).manual_seed(
            base.SEED + 1009 * index + 97 * harness.rank
        )
        x = torch.randn(
            (harness.case.graph_tokens, base.HIDDEN), generator=generator,
            device=harness.device, dtype=torch.bfloat16,
        ) * (0.5 + index / 8)
        fp8, scale = base.quantize_k128(x)
        ids, weights = base.make_routing(harness.case, harness.rank, harness.device)
        ids = ((ids + index * base.LOCAL_EXPERTS) % base.TOTAL_EXPERTS).contiguous()
        weights = weights.roll(index % base.TOPK, dims=1).contiguous()
        if index == 2 and harness.rank == 0:
            # This rank sends no routes but still owns experts receiving work.
            ids.fill_(-1)
            weights.zero_()
        elif index == 3:
            # All ranks send every token to six experts on rank 0.
            ids.copy_(torch.arange(base.TOPK, device=harness.device, dtype=torch.int32))
        elif index == 4:
            ids.fill_(-1)
            weights.zero_()
        elif index == 5:
            ids[1::2].fill_(-1)
            weights[1::2].zero_()
        patterns.append((fp8, scale, ids, weights))
    return patterns


def install(harness, pattern):
    harness.x_fp8, harness.x_scale, harness.topk_ids, harness.router_weights = pattern


@pytest.mark.parametrize("variant", base.VARIANTS)
def test_warprole_extended_reuse(context, variant):
    rank, world_size, device = context
    base.require_warprole(device)
    assert world_size == base.EP_SIZE
    harness = base.build_harness(
        base.Case("reuse_stress_256", 256, 1.0), rank, device, fresh=True,
    )
    patterns = make_patterns(harness)
    expected = []
    for pattern in patterns:
        install(harness, pattern)
        expected.append(base.run_split(harness).clone())
    torch.cuda.synchronize(device)
    dist.barrier()
    # A retained owning output must survive all subsequent workspace mutation.
    first = None
    for iteration in range(ITERATIONS):
        index = iteration % PATTERNS
        install(harness, patterns[index])
        output = base.run_warprole(harness, variant)
        base.assert_bitwise(f"reuse/{variant}/{iteration}", expected[index], output)
        base.assert_trap_clear(harness, f"reuse/{variant}/{iteration}")
        if first is None:
            first = output
        else:
            assert output.data_ptr() != first.data_ptr()
            base.assert_bitwise(f"owned/{variant}/{iteration}", expected[0], first)
        if (iteration + 1) % 32 == 0:
            print(f"REUSE_PROGRESS rank={rank} variant={variant} calls={iteration + 1}", flush=True)
    dist.barrier()
