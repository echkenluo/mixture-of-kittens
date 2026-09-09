import importlib.util
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "warprole_test_shapes", Path(__file__).parents[1] / "warprole_test_shapes.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("ep", [4, 8])
@pytest.mark.parametrize("tokens", [64, 256, 512, 2048, 3904])
@pytest.mark.parametrize("experts", [64, 256])
def test_route_capacity_alignment_and_worst_case_bound(ep, tokens, experts):
    local_experts = experts // ep
    multiplier = MODULE.schedule_multiplier(tokens, 6, ep, local_experts)
    capacity = tokens * 6 * max(2, math.ceil(ep * multiplier))
    assert capacity % 256 == 0  # build_schedule's independent layout contract
    assert capacity >= tokens * 6 * ep + local_experts * 63


def test_small_ep4_256_expert_regression():
    # The original conservative factor 15 gives 5760 rows, not M256 aligned.
    multiplier = MODULE.schedule_multiplier(64, 6, 4, 64, minimum=1.5)
    assert 64 * 6 * math.ceil(4 * multiplier) == 6144
