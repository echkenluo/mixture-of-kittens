"""Four-rank bitwise tests for the warp-role megakernel (plan Task 6, step 5).

The megakernel has to produce, bit for bit, what the split path produces on the
same inputs:

    dispatch_fp8_block
    grouped_gemm_fp8_block_dynamic_out   (W13, BF16 gate_up)
    silu_and_mul_contig_post_quant       (SGLang activation + FP8/K128 quant)
    grouped_gemm_fp8_block_dynamic_out   (W2, BF16 routed rows)
    combine_reduce_fp8_block_routes

That is the sequence the terminal EP4 benchmark times as its ``split`` arm, and
it is reproduced here call for call so a mismatch points at the kernel and not
at a different reference.

The cases cover the shapes the schedule and the task decode behave differently
on: a full 2048-token batch whose routed rows land on exact minibatch
boundaries, a 3888-token batch bucketed up to 3904 so the schedule has a padded
tail, a 64-token batch that is smaller than one minibatch, and a routing skew
that gives rank 0's experts about three times the rows of any other rank.  A
fifth test runs two forwards back to back on one state, which is the only case
that exercises the per-launch counter reset.

Run on four H20s inside the SGLang v0.5.17 container:

    torchrun --standalone --nproc-per-node=4 -m pytest -s tests/test_warprole_ep4.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("MOK_SM90_EXPERIMENTAL", "1")

import pytest
import torch
import torch.distributed as dist

from mok import _C, functional, warprole

EP_SIZE = 4
TOTAL_EXPERTS = 64
LOCAL_EXPERTS = TOTAL_EXPERTS // EP_SIZE
HIDDEN = 4096
INTERMEDIATE = 2048
TOPK = 6
M_TILE = 64
K_GROUP = 128
FP8_MAX = 448.0
SWIGLU_LIMIT = 10.0
# Expert segments are padded to one M64 tile, the granularity the warprole task
# decode works in; a larger padding would only add empty rows.
EXPERT_PADDING = 64
VARIANTS = ("c1s6", "c2s4")
SEED = 20260904


@dataclass(frozen=True, slots=True)
class Case:
    """One measured shape.

    ``capacity_multiplier`` feeds ``MoKConfig.schedule_capacity_multiplier``,
    which the workspace turns into ``max(2, ceil(ep_size * multiplier))`` times
    ``tokens * topk`` rows.  Small batches need a larger multiplier than the
    EP4 default because per-expert padding, not the route count, dominates
    them: 64 tokens produce about 384 routed rows per rank but 16 padded expert
    segments can still cost 16 x 63 extra rows.
    """

    name: str
    effective_tokens: int
    capacity_multiplier: float
    skew: bool = False

    @property
    def graph_tokens(self) -> int:
        return math.ceil(self.effective_tokens / M_TILE) * M_TILE


CASES = {
    case.name: case
    for case in (
        Case("uniform_2048", 2048, 0.5),
        Case("tail_3888", 3888, 0.5),
        Case("small_64", 64, 1.5),
        Case("skew_512", 512, 1.0, skew=True),
    )
}


def require_warprole(device: torch.device) -> None:
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise AssertionError("the warp-role megakernel is SM90 only")
    for name in (
        "fp8_block_warprole_prepare_out",
        "fp8_block_warprole_c1s6_out",
        "fp8_block_warprole_c2s4_out",
    ):
        assert hasattr(_C, name), f"SM90 build did not register {name}"


def sglang_activation():
    """Resolve the split path's activation kernel across container versions.

    Both ranks of the import are tried on every rank, so a container without
    the kernel skips the test on all four ranks at the same point and no
    collective is left half-issued.
    """
    try:
        from sglang.kernels.ops.attention.dsv4 import (
            silu_and_mul_contig_post_quant_dynamic,
        )

        return silu_and_mul_contig_post_quant_dynamic, "dynamic"
    except ImportError:
        pass
    try:
        # SGLang 0.5.17 (image a8-base-cu130): same keyword interface, no row count.
        from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant

        return silu_and_mul_contig_post_quant, "static"
    except ImportError:
        pass
    try:
        from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant

        return silu_and_mul_contig_post_quant, "static"
    except ImportError:
        pytest.skip("SGLang silu_and_mul_contig_post_quant is not importable")


def chunk_bytes_for(graph_tokens: int) -> int:
    """Largest legal all-gather chunk that divides one rank's route buffer.

    ``build_schedule`` refuses a chunk that does not divide
    ``tokens * topk * 4`` bytes, and the 3904-token bucket is not divisible by
    the 2048-byte default.  Candidates stay multiples of 128 so the TMA bulk
    copy inside the all-gather keeps the alignment the default 2048 gives it;
    an M64-aligned token count always makes 1536 a divisor, so the search
    cannot come up empty.
    """
    route_bytes = graph_tokens * TOPK * 4
    for candidate in range(2048, 0, -128):
        if route_bytes % candidate == 0:
            return candidate
    raise AssertionError(f"no legal chunk size for {route_bytes} route bytes")


def quantize_k128(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 activations to FP8 with one FP32 scale per 128 columns.

    Both arms consume the result unchanged, so this only has to be a valid
    FP8/K128 pair; it is not part of what the two arms are compared on.
    """
    rows, columns = activations.shape
    grouped = activations.float().view(rows, columns // K_GROUP, K_GROUP)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    quantized = (grouped / scale).clamp(-FP8_MAX, FP8_MAX)
    return (
        quantized.view(rows, columns).to(torch.float8_e4m3fn).contiguous(),
        scale.view(rows, columns // K_GROUP).contiguous(),
    )


def make_weights(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random FP8 expert weights with [128,128] block scales.

    Same generator shape as tests/test_warprole_w13.py::make_inputs: values
    clamped into FP8 range before the cast, block scales in [0.01, 0.10) so the
    accumulated products stay well inside BF16.  The seed does not depend on
    the rank: every rank holds the weights of its own 16 local experts and the
    routing decides which rows reach them.
    """
    generator = torch.Generator(device=device).manual_seed(SEED)

    def fp8(*shape: int) -> torch.Tensor:
        values = torch.randn(
            shape, generator=generator, device=device, dtype=torch.bfloat16
        )
        return values.clamp(-3, 3).to(torch.float8_e4m3fn).contiguous()

    def block_scale(*shape: int) -> torch.Tensor:
        values = torch.rand(shape, generator=generator, device=device)
        return (values * 0.09 + 0.01).to(torch.float32).contiguous()

    w13 = fp8(LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w13_scale = block_scale(
        LOCAL_EXPERTS, 2 * INTERMEDIATE // K_GROUP, HIDDEN // K_GROUP
    )
    w2 = fp8(LOCAL_EXPERTS, HIDDEN, INTERMEDIATE)
    w2_scale = block_scale(
        LOCAL_EXPERTS, HIDDEN // K_GROUP, INTERMEDIATE // K_GROUP
    )
    return w13, w13_scale, w2, w2_scale


def make_routing(
    case: Case, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick top-6 experts and router weights for this rank's tokens.

    Uniform cases draw six distinct experts out of 64 with equal probability,
    so each rank receives about a quarter of all routes.  The skewed case
    weights rank 0's sixteen experts three to one, which puts half of every
    rank's routes on rank 0 -- three times what any other rank receives.
    Tokens past ``effective_tokens`` are padding: expert -1 keeps them out of
    the schedule, and the reducer masks them on both arms.
    """
    generator = torch.Generator(device=device).manual_seed(SEED + rank)
    probabilities = torch.ones(TOTAL_EXPERTS, device=device)
    if case.skew:
        probabilities[:LOCAL_EXPERTS] = 3.0
    chosen = torch.multinomial(
        probabilities.expand(case.effective_tokens, TOTAL_EXPERTS).contiguous(),
        TOPK,
        replacement=False,
        generator=generator,
    ).to(torch.int32)

    top_experts = torch.full(
        (case.graph_tokens, TOPK), -1, dtype=torch.int32, device=device
    )
    top_experts[: case.effective_tokens] = chosen
    router_weights = torch.zeros(
        (case.graph_tokens, TOPK), dtype=torch.float32, device=device
    )
    logits = torch.rand(
        (case.effective_tokens, TOPK),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    router_weights[: case.effective_tokens] = logits / logits.sum(
        dim=-1, keepdim=True
    )
    return top_experts.contiguous(), router_weights.contiguous()


@dataclass(slots=True)
class Harness:
    """Everything one case needs: workspace, state, inputs, split scratch."""

    case: Case
    rank: int
    device: torch.device
    workspace: functional.MoKFP8RouteWorkspace
    state: warprole.WarpRoleState
    schedule: functional.MoKSchedule
    active_rows: int
    x_fp8: torch.Tensor
    x_scale: torch.Tensor
    router_weights: torch.Tensor
    topk_ids: torch.Tensor
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    gate_up: torch.Tensor
    down_input: torch.Tensor
    down_input_scale: torch.Tensor
    split_routed_y: torch.Tensor
    split_output: torch.Tensor


_WEIGHTS: dict[int, tuple[torch.Tensor, ...]] = {}
_HARNESSES: dict[str, Harness] = {}


def build_harness(case: Case, rank: int, device: torch.device) -> Harness:
    """Create (or return) the harness for one case.

    Every step here is collective in lockstep across the four ranks: the
    workspace rendezvouses symmetric memory, the state rendezvouses its
    ``push_done`` counter, and ``build_schedule`` all-gathers the routes.  The
    harness is cached so the repeat-forward test reuses the same state rather
    than allocating a second one.
    """
    cached = _HARNESSES.get(case.name)
    if cached is not None:
        return cached

    if not _WEIGHTS:
        _WEIGHTS[0] = make_weights(device)
    w13, w13_scale, w2, w2_scale = _WEIGHTS[0]

    config = functional.MoKConfig(
        schedule_capacity_multiplier=case.capacity_multiplier,
        all_gather_top_experts_chunk_bytes=chunk_bytes_for(case.graph_tokens),
    )
    workspace = functional.get_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=case.graph_tokens,
        hidden_size=HIDDEN,
        topk=TOPK,
        num_local_experts=LOCAL_EXPERTS,
    )
    capacity = workspace.schedule_capacity
    state = warprole.get_warprole_state(
        workspace, dist.group.WORLD, device=device, capacity=capacity
    )

    top_experts, router_weights = make_routing(case, rank, device)
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=LOCAL_EXPERTS,
        expert_padding=EXPERT_PADDING,
    )
    active_rows = int(schedule.num_tokens.item())
    assert active_rows % M_TILE == 0, (
        f"{case.name}: the megakernel only accepts M64-aligned schedules, got "
        f"{active_rows} rows"
    )
    assert 0 < active_rows <= capacity, (
        f"{case.name}: {active_rows} routed rows do not fit capacity "
        f"{capacity}; raise capacity_multiplier"
    )

    activations = torch.randn(
        (case.graph_tokens, HIDDEN),
        generator=torch.Generator(device=device).manual_seed(SEED + 97 * rank),
        device=device,
        dtype=torch.bfloat16,
    )
    x_fp8, x_scale = quantize_k128(activations)

    harness = Harness(
        case=case,
        rank=rank,
        device=device,
        workspace=workspace,
        state=state,
        schedule=schedule,
        active_rows=active_rows,
        x_fp8=x_fp8,
        x_scale=x_scale,
        router_weights=router_weights,
        topk_ids=top_experts,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        gate_up=torch.empty(
            (capacity, 2 * INTERMEDIATE), dtype=torch.bfloat16, device=device
        ),
        down_input=torch.empty(
            (capacity, INTERMEDIATE), dtype=torch.float8_e4m3fn, device=device
        ),
        down_input_scale=torch.empty(
            (capacity, INTERMEDIATE // K_GROUP),
            dtype=torch.float32,
            device=device,
        ),
        split_routed_y=torch.empty(
            (capacity, HIDDEN), dtype=torch.bfloat16, device=device
        ),
        split_output=torch.empty(
            (case.graph_tokens, HIDDEN), dtype=torch.bfloat16, device=device
        ),
    )
    torch.cuda.synchronize()
    dist.barrier()
    _HARNESSES[case.name] = harness
    return harness


def run_split(harness: Harness) -> torch.Tensor:
    """The reference arm, call for call the terminal benchmark's split arm."""
    activation, kind = sglang_activation()
    workspace = harness.workspace
    schedule = harness.schedule
    active_rows = harness.active_rows

    functional.acquire_workspace_lease(workspace)
    functional.dispatch_fp8_block(
        workspace,
        schedule,
        harness.x_fp8,
        harness.x_scale,
        trim_to_active_rows=False,
        prepare_combine=True,
    )
    functional.grouped_gemm_fp8_block_dynamic_out(
        workspace.routed_x,
        harness.w13,
        workspace.routed_x_scale,
        harness.w13_scale,
        workspace.m_indices,
        schedule.num_tokens,
        harness.gate_up,
    )
    if kind == "dynamic":
        # The dynamic entry reads the row count off the device, so it takes the
        # capacity-sized buffers and bounds itself.
        activation(
            input=harness.gate_up,
            output=harness.down_input,
            output_scale=harness.down_input_scale,
            active_tokens=schedule.num_tokens,
            quant_group_size=K_GROUP,
            scale_ue8m0=False,
            transposed=False,
            swiglu_limit=SWIGLU_LIMIT,
            swizzle=False,
        )
    else:
        # The static entry has no row-count argument; the benchmark's split arm
        # slices the buffers instead, and this call mirrors it exactly.
        activation(
            input=harness.gate_up[:active_rows],
            output=harness.down_input[:active_rows],
            output_scale=harness.down_input_scale[:active_rows],
            quant_group_size=K_GROUP,
            scale_ue8m0=False,
            transposed=False,
            swiglu_limit=SWIGLU_LIMIT,
            swizzle=False,
        )
    functional.grouped_gemm_fp8_block_dynamic_out(
        harness.down_input,
        harness.w2,
        harness.down_input_scale,
        harness.w2_scale,
        workspace.m_indices,
        schedule.num_tokens,
        harness.split_routed_y,
    )
    reduced = functional.combine_reduce_fp8_block_routes(
        workspace,
        schedule,
        harness.split_routed_y[:active_rows],
        harness.router_weights,
        combine_precleared=True,
    )
    harness.split_output.copy_(reduced)
    functional.release_workspace_lease(workspace)
    return harness.split_output


def run_warprole(harness: Harness, variant: str) -> torch.Tensor:
    return warprole.warprole_forward(
        harness.workspace,
        harness.state,
        harness.schedule,
        harness.x_fp8,
        harness.x_scale,
        harness.router_weights,
        harness.topk_ids,
        harness.w13,
        harness.w13_scale,
        harness.w2,
        harness.w2_scale,
        variant=variant,
        swiglu_limit=SWIGLU_LIMIT,
    )


def assert_bitwise(label: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Compare BF16 outputs on their bit patterns.

    ``torch.equal`` on the int16 views is the strict form: it does not let a
    NaN pass as a mismatch-free comparison and it separates -0 from +0, which
    the reducer's contract distinguishes.
    """
    assert expected.shape == actual.shape and expected.dtype == actual.dtype, (
        f"{label}: metadata differ, {expected.shape}/{expected.dtype} vs "
        f"{actual.shape}/{actual.dtype}"
    )
    left, right = expected.view(torch.int16), actual.view(torch.int16)
    if not torch.equal(left, right):
        mismatch = int((left != right).sum().item())
        row = int((left != right).any(dim=-1).nonzero()[0].item())
        raise AssertionError(
            f"{label}: {mismatch} of {left.numel()} elements differ from the "
            f"split path, first at token {row}"
        )


def assert_trap_clear(harness: Harness, label: str) -> None:
    record = harness.workspace.trap_record
    assert int(record[0].item()) == 0, (
        f"{label}: trap record is {record.tolist()} "
        f"({functional.format_trap_record(harness.workspace)})"
    )


def compare_arms(harness: Harness, variant: str, label: str) -> None:
    """Run both arms on the same inputs and require identical bits."""
    split_output = run_split(harness)
    torch.cuda.synchronize()
    dist.barrier()

    warprole_output = run_warprole(harness, variant).clone()
    torch.cuda.synchronize()
    dist.barrier()

    effective = harness.case.effective_tokens
    assert_bitwise(
        f"{label}/{variant}",
        split_output[:effective],
        warprole_output[:effective],
    )
    if effective != harness.case.graph_tokens:
        # Padded tokens carry expert -1 in every slot.  The split arm reduces
        # them out of a pre-cleared combine buffer and the megakernel masks
        # them; both contracts say BF16 +0.
        tail = warprole_output[effective:]
        assert torch.equal(tail, torch.zeros_like(tail)), (
            f"{label}/{variant}: padded tokens must reduce to zero"
        )
    assert_trap_clear(harness, f"{label}/{variant}")


@pytest.mark.parametrize("case_name", sorted(CASES))
@pytest.mark.parametrize("variant", VARIANTS)
def test_warprole_matches_split(
    context: tuple[int, int, torch.device], case_name: str, variant: str
) -> None:
    rank, world_size, device = context
    require_warprole(device)
    assert world_size == EP_SIZE, "the warprole contract under test is EP4"
    harness = build_harness(CASES[case_name], rank, device)
    compare_arms(harness, variant, case_name)


@pytest.mark.parametrize("variant", VARIANTS)
def test_warprole_first_on_fresh_workspace(
    context: tuple[int, int, torch.device], variant: str
) -> None:
    """The megakernel must be correct when nothing ran on the workspace before.

    SGLang reaches ``warprole_forward`` without ever running the split path
    on that workspace, so nothing has pre-cleared the combine buffer or
    touched the barrier state.  Every other test runs split first, which
    would hide a dependence on that history.
    """
    rank, world_size, device = context
    require_warprole(device)
    assert world_size == EP_SIZE, "the warprole contract under test is EP4"
    harness = build_harness(CASES["uniform_2048"], rank, device)

    warprole_output = run_warprole(harness, variant).clone()
    torch.cuda.synchronize()
    dist.barrier()
    assert_trap_clear(harness, f"fresh/{variant}")

    split_output = run_split(harness)
    torch.cuda.synchronize()
    dist.barrier()

    effective = harness.case.effective_tokens
    assert_bitwise(
        f"fresh/{variant}", split_output[:effective], warprole_output[:effective]
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_warprole_repeated_forward(
    context: tuple[int, int, torch.device], variant: str
) -> None:
    """Two launches on one state must both match the split reference.

    Nothing between the two calls resets the dependency counters except the
    kernel's own prepare entry, so this is what proves ``x_ready``,
    ``hidden_ready``, ``y_ready`` and the symmetric ``push_done`` come back to
    zero after a launch rather than accumulating across launches.
    """
    rank, world_size, device = context
    require_warprole(device)
    assert world_size == EP_SIZE, "the warprole contract under test is EP4"
    harness = build_harness(CASES["uniform_2048"], rank, device)

    split_output = run_split(harness)
    torch.cuda.synchronize()
    dist.barrier()

    first = run_warprole(harness, variant).clone()
    second = run_warprole(harness, variant).clone()
    torch.cuda.synchronize()
    dist.barrier()

    effective = harness.case.effective_tokens
    assert_bitwise(
        f"repeat/{variant}/first", split_output[:effective], first[:effective]
    )
    assert_bitwise(
        f"repeat/{variant}/second", split_output[:effective], second[:effective]
    )
    assert_trap_clear(harness, f"repeat/{variant}")
