import pytest
import torch
import torch.distributed as dist

from mok import _C
from mok import functional, ops


def require_sm90(device: torch.device) -> None:
    if torch.cuda.get_device_capability(device) != (9, 0):
        pytest.skip("SM90 worker contract requires Hopper compute capability 9.0")
    assert hasattr(_C, "sm90_worker_test"), "SM90 build did not register sm90_worker_test"
    assert hasattr(_C, "sm90_fp8_block_test"), (
        "SM90 build did not register sm90_fp8_block_test"
    )
    assert hasattr(_C, "sm90_fp8_block_grouped_test"), (
        "SM90 build did not register sm90_fp8_block_grouped_test"
    )
    assert hasattr(_C, "sm90_fp8_block_grouped_pipelined_test"), (
        "SM90 build did not register sm90_fp8_block_grouped_pipelined_test"
    )
    assert hasattr(_C, "sm90_fp8_block_grouped_out_test"), (
        "SM90 build did not register sm90_fp8_block_grouped_out_test"
    )
    assert hasattr(_C, "sm90_fp8_block_grouped_pipelined_out_test"), (
        "SM90 build did not register sm90_fp8_block_grouped_pipelined_out_test"
    )
    assert hasattr(_C, "fp8_block_grouped_pipelined_out"), (
        "SM90 build did not register fp8_block_grouped_pipelined_out"
    )
    assert hasattr(_C, "fp8_block_grouped_contiguous_out"), (
        "SM90 build did not register fp8_block_grouped_contiguous_out"
    )
    assert hasattr(_C, "fp8_block_routed_dispatch_out"), (
        "SM90 build did not register fp8_block_routed_dispatch_out"
    )
    assert hasattr(_C, "fp8_block_routed_combine_out"), (
        "SM90 build did not register fp8_block_routed_combine_out"
    )
    assert hasattr(_C, "fp8_block_routed_dispatch_copy_out"), (
        "SM90 build did not register fused FP8 dispatch"
    )
    assert hasattr(_C, "fp8_block_routed_combine_reduce_out"), (
        "SM90 build did not register fused FP8 combine/reduce"
    )
    assert hasattr(_C, "routed_epilogue_out"), (
        "SM90 build did not register routed_epilogue_out"
    )


@pytest.mark.parametrize("active_only", [False, True], ids=["capacity", "active"])
@pytest.mark.parametrize("expert_padding", [256, 64], ids=["m256", "m64"])
def test_sm90_fp8_block_routed_dispatch_combine(
    active_only: bool,
    expert_padding: int,
    context: tuple[int, int, torch.device]
) -> None:
    rank, world_size, device = context
    require_sm90(device)
    assert world_size in (4, 8, 16, 32, 64)

    num_local_tokens = 512
    hidden_size = 256
    topk = 1
    num_local_experts = 2
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=4096,
        schedule_capacity_multiplier=1.0,
    )
    workspace = functional.get_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    token_indices = torch.arange(num_local_tokens, device=device)
    destination_ranks = token_indices % world_size
    local_experts = ((token_indices // world_size) % 4 == 0).to(torch.int64)
    top_experts = (
        destination_ranks * num_local_experts + local_experts
    ).view(-1, 1)
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=num_local_experts,
        expert_padding=expert_padding,
    )

    x = torch.empty(
        num_local_tokens,
        hidden_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    x_scale = torch.empty(
        num_local_tokens,
        hidden_size // 128,
        dtype=torch.float32,
        device=device,
    )
    columns = torch.arange(hidden_size, device=device)
    x.copy_(
        ((token_indices[:, None] * hidden_size + columns[None, :]) % 31 - 15)
        .add(rank * 0.25)
        .to(torch.float8_e4m3fn)
    )
    scale_columns = torch.arange(hidden_size // 128, device=device)
    x_scale.copy_(
        rank * 10000
        + token_indices[:, None] * 100
        + scale_columns[None, :]
    )
    capacity = workspace.schedule_capacity
    valid_rows = int(schedule.num_tokens.item())
    workspace.routed_x.fill_(7)
    workspace.routed_x_scale.fill_(-999)
    workspace.m_indices.fill_(-777)
    workspace.combine_buffer.fill_(float("nan"))
    routed_x, routed_x_scale, m_indices = functional.dispatch_fp8_block(
        workspace,
        schedule,
        x,
        x_scale,
        trim_to_active_rows=active_only,
    )

    returned_rows = valid_rows if active_only else capacity
    assert routed_x.shape[0] == returned_rows
    assert routed_x_scale.shape[0] == returned_rows
    assert m_indices.shape[0] == returned_rows
    peer_ranks = schedule.peer_rank[:valid_rows].to(torch.int64)
    peer_tokens = schedule.peer_token_idx[:valid_rows].to(torch.int64)
    valid = peer_ranks >= 0
    expected_x = torch.zeros_like(routed_x[:valid_rows])
    expected_scale = torch.zeros_like(routed_x_scale[:valid_rows])
    expected_x[valid] = (
        (
            peer_tokens[valid, None] * hidden_size + columns[None, :]
        )
        % 31
        - 15
    ).add(peer_ranks[valid, None] * 0.25).to(torch.float8_e4m3fn)
    expected_scale[valid] = (
        peer_ranks[valid, None] * 10000
        + peer_tokens[valid, None] * 100
        + scale_columns[None, :]
    ).to(torch.float32)
    route_counts = torch.bincount(
        local_experts,
        minlength=num_local_experts,
    )
    expected_tokens_per_expert = (
        torch.div(
            route_counts + expert_padding - 1,
            expert_padding,
            rounding_mode="floor",
        )
        * expert_padding
    ).to(torch.int32)
    expected_m_indices = torch.zeros_like(m_indices)
    expected_m_indices[:valid_rows] = torch.repeat_interleave(
        torch.arange(num_local_experts, dtype=torch.int32, device=device),
        schedule.tokens_per_expert,
        output_size=valid_rows,
    )
    dispatch_mismatches = torch.tensor(
        [
            int(
                (
                    routed_x[:valid_rows].view(torch.uint8)
                    != expected_x.view(torch.uint8)
                ).sum()
            ),
            int((routed_x_scale[:valid_rows] != expected_scale).sum()),
            int((workspace.routed_x[valid_rows:] != (7 if active_only else 0)).sum()),
            int(
                (
                    workspace.routed_x_scale[valid_rows:]
                    != (-999 if active_only else 0)
                ).sum()
            ),
            int((m_indices != expected_m_indices).sum()),
            int(
                (
                    schedule.tokens_per_expert
                    != expected_tokens_per_expert
                ).sum()
            ),
            int(valid.sum() != num_local_tokens),
        ],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(dispatch_mismatches, op=dist.ReduceOp.MAX)
    print(
        f"ROUTED_DISPATCH_MISMATCH|rank={rank}|values="
        f"{dispatch_mismatches.cpu().tolist()}",
        flush=True,
    )
    assert not dispatch_mismatches.any().item()

    routed_y = torch.zeros(
        returned_rows, hidden_size, dtype=torch.bfloat16, device=device
    )
    routed_y[:valid_rows] = (
        rank * 32
        + expected_m_indices[:valid_rows].to(torch.int64) * 16
        + (peer_tokens % 16)
    ).to(torch.bfloat16)[:, None]
    combine_buffer = functional.combine_fp8_block(
        workspace,
        schedule,
        routed_y,
    )
    expected_combine = (
        destination_ranks.to(torch.bfloat16) * 32
        + local_experts.to(torch.bfloat16) * 16
        + (token_indices % 16).to(torch.bfloat16)
    )[:, None].expand(-1, hidden_size)
    combine_mismatches = torch.tensor(
        [int((combine_buffer != expected_combine).sum())],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(combine_mismatches, op=dist.ReduceOp.MAX)
    print(
        f"ROUTED_COMBINE_MISMATCH|rank={rank}|values="
        f"{combine_mismatches.cpu().tolist()}",
        flush=True,
    )
    assert not combine_mismatches.any().item()

    workspace.combine_buffer.fill_(float("nan"))
    fused_output = functional.combine_reduce_fp8_block_routes(
        workspace,
        schedule,
        routed_y,
        torch.ones(
            (num_local_tokens, topk), dtype=torch.float32, device=device
        ),
    )
    fused_mismatches = torch.tensor(
        [int((fused_output != expected_combine).sum())],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(fused_mismatches, op=dist.ReduceOp.MAX)
    print(
        f"ROUTED_FUSED_MISMATCH|rank={rank}|values="
        f"{fused_mismatches.cpu().tolist()}",
        flush=True,
    )
    assert not fused_mismatches.any().item()


@pytest.mark.parametrize("num_local_tokens", [256, 512])
def test_sm90_fp8_block_empty_routes(
    context: tuple[int, int, torch.device], num_local_tokens: int
) -> None:
    _, world_size, device = context
    require_sm90(device)
    assert world_size in (4, 8, 16, 32, 64)

    hidden_size, topk = 256, 1
    config = functional.MoKConfig(
        schedule_capacity_multiplier=1.0,
        all_gather_top_experts_chunk_bytes=1024,
    )
    workspace = functional.get_fp8_route_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        torch.full(
            (num_local_tokens, topk),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        num_local_experts=2,
    )
    active_rows = int(schedule.num_tokens.item())
    assert active_rows == 0

    x = torch.zeros(
        (num_local_tokens, hidden_size),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    x_scale = torch.ones(
        (num_local_tokens, hidden_size // 128),
        dtype=torch.float32,
        device=device,
    )
    routed_x, routed_x_scale, m_indices = functional.dispatch_fp8_block(
        workspace,
        schedule,
        x,
        x_scale,
        trim_to_active_rows=True,
    )
    assert routed_x.shape == (0, hidden_size)
    assert routed_x_scale.shape == (0, hidden_size // 128)
    assert m_indices.shape == (0,)

    workspace.combine_buffer.fill_(float("nan"))
    combine_buffer = functional.combine_fp8_block(
        workspace,
        schedule,
        torch.empty((0, hidden_size), dtype=torch.bfloat16, device=device),
    )
    output = functional.reduce_fp8_block_routes(
        workspace,
        torch.zeros(
            (num_local_tokens, topk), dtype=torch.float32, device=device
        ),
    )
    torch.cuda.synchronize(device)
    assert not combine_buffer.any().item()
    assert not output.any().item()

    fused_output = functional.combine_reduce_fp8_block_routes(
        workspace,
        schedule,
        torch.empty((0, hidden_size), dtype=torch.bfloat16, device=device),
        torch.zeros(
            (num_local_tokens, topk), dtype=torch.float32, device=device
        ),
    )
    torch.cuda.synchronize(device)
    assert not fused_output.any().item()


def test_sm90_fp8_block_routed_rejects_invalid_inputs(
    context: tuple[int, int, torch.device]
) -> None:
    _, _, device = context
    require_sm90(device)

    num_local_tokens = 512
    hidden_size = 256
    capacity = 512
    x = torch.ones(
        (num_local_tokens, hidden_size), device=device
    ).to(torch.float8_e4m3fn)
    x_scale = torch.ones(
        (num_local_tokens, hidden_size // 128), device=device
    )
    routed_x = torch.empty(
        (capacity, hidden_size), dtype=torch.float8_e4m3fn, device=device
    )
    routed_x_scale = torch.empty(
        (capacity, hidden_size // 128), device=device
    )
    m_indices = torch.empty(capacity, dtype=torch.int32, device=device)
    schedule_peer_rank = torch.full(
        (capacity,), -1, dtype=torch.int32, device=device
    )
    schedule_peer_token_idx = torch.full_like(schedule_peer_rank, -1)
    num_tokens = torch.zeros(1, dtype=torch.int32, device=device)
    tokens_per_expert = torch.zeros(2, dtype=torch.int32, device=device)
    pointer_list = [1, 1, 1, 1]

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        ops.fp8_block_routed_dispatch_out(
            x.float(), pointer_list, x_scale, pointer_list,
            routed_x, routed_x_scale, m_indices,
            schedule_peer_rank, schedule_peer_token_idx,
            num_tokens, tokens_per_expert, 1,
        )
    with pytest.raises(ValueError, match="x_scale must"):
        ops.fp8_block_routed_dispatch_out(
            x, pointer_list, x_scale.bfloat16(), pointer_list,
            routed_x, routed_x_scale, m_indices,
            schedule_peer_rank, schedule_peer_token_idx,
            num_tokens, tokens_per_expert, 1,
        )
    with pytest.raises(ValueError, match="x_ptrs length"):
        ops.fp8_block_routed_dispatch_out(
            x, [1], x_scale, pointer_list,
            routed_x, routed_x_scale, m_indices,
            schedule_peer_rank, schedule_peer_token_idx,
            num_tokens, tokens_per_expert, 1,
        )
    with pytest.raises(ValueError, match="nonempty CUDA int32 vector"):
        ops.fp8_block_routed_dispatch_out(
            x, pointer_list, x_scale, pointer_list,
            routed_x, routed_x_scale, m_indices,
            schedule_peer_rank, schedule_peer_token_idx,
            num_tokens, tokens_per_expert[:0], 1,
        )

    routed_y = torch.empty(
        (capacity, hidden_size), dtype=torch.bfloat16, device=device
    )
    combine_buffer = torch.empty(
        (num_local_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    with pytest.raises(TypeError, match="bfloat16"):
        ops.fp8_block_routed_combine_out(
            routed_y.float(), combine_buffer, pointer_list,
            schedule_peer_rank, schedule_peer_token_idx, num_tokens, 1,
        )
    with pytest.raises(ValueError, match="combine_buffer_ptrs length"):
        ops.fp8_block_routed_combine_out(
            routed_y, combine_buffer, [1],
            schedule_peer_rank, schedule_peer_token_idx, num_tokens, 1,
        )


def test_sm90_routed_epilogue_numeric(
    context: tuple[int, int, torch.device]
) -> None:
    rank, _, device = context
    require_sm90(device)
    num_tokens, hidden_size, topk = 512, 256, 3
    token = torch.arange(num_tokens, device=device, dtype=torch.float32)
    column = torch.arange(hidden_size, device=device, dtype=torch.float32)
    route = torch.arange(topk, device=device, dtype=torch.float32)
    combine_buffer = (
        token[:, None, None] * 0.03125
        + route[None, :, None] * 0.5
        + (column[None, None, :] % 17) * 0.015625
        + rank * 0.25
    ).to(torch.bfloat16).reshape(num_tokens * topk, hidden_size)
    topk_weights = torch.tensor(
        [0.25, 0.5, 0.125], dtype=torch.float32, device=device
    ).expand(num_tokens, -1).contiguous()
    output = torch.full(
        (num_tokens, hidden_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    ops.routed_epilogue_out(combine_buffer, topk_weights, output)
    reference = (
        combine_buffer.view(num_tokens, topk, hidden_size).float()
        * topk_weights[:, :, None]
    ).sum(dim=1).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, rtol=0, atol=0.03125)


def test_sm90_routed_epilogue_rejects_invalid_inputs(
    context: tuple[int, int, torch.device]
) -> None:
    _, _, device = context
    require_sm90(device)
    combine_buffer = torch.empty(
        (1024, 256), dtype=torch.bfloat16, device=device
    )
    topk_weights = torch.ones((512, 2), dtype=torch.float32, device=device)
    output = torch.empty((512, 256), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="float32"):
        ops.routed_epilogue_out(
            combine_buffer, topk_weights.to(torch.bfloat16), output
        )
    with pytest.raises(ValueError, match=r"\[T\*topk,H\]"):
        ops.routed_epilogue_out(
            combine_buffer[:512].contiguous(), topk_weights, output
        )
    with pytest.raises(ValueError, match="at least 256"):
        ops.routed_epilogue_out(
            combine_buffer[:256].contiguous(),
            topk_weights[:128].contiguous(),
            output[:128].contiguous(),
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


@pytest.mark.parametrize(
    ("experts", "max_m", "n", "k", "valid_rows"),
    [
        (2, 128, 256, 256, (64, 128)),
        (2, 64, 128, 4096, (64, 32)),
    ],
)
@pytest.mark.parametrize(
    "impl_name",
    [
        "sm90_fp8_block_grouped_test",
        "sm90_fp8_block_grouped_pipelined_test",
    ],
    ids=["sync", "cpasync-2stage"],
)
def test_sm90_fp8_block_grouped_numeric(
    context: tuple[int, int, torch.device],
    experts: int,
    max_m: int,
    n: int,
    k: int,
    valid_rows: tuple[int, ...],
    impl_name: str,
) -> None:
    rank, _, device = context
    require_sm90(device)
    generator = torch.Generator(device=device).manual_seed(
        20260817 + 100 * rank + k
    )
    k_blocks = k // 128
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        a = torch.randn(
            (experts, max_m, k), generator=generator, device=device
        ).clamp(-3, 3).to(torch.float8_e4m3fn)
        b = torch.randn(
            (experts, n, k), generator=generator, device=device
        ).clamp(-3, 3).to(torch.float8_e4m3fn)
        a_scale = torch.rand(
            (experts, max_m, k_blocks), generator=generator, device=device
        ) * 0.09 + 0.01
        b_scale = torch.rand(
            (experts, n // 128, k_blocks), generator=generator, device=device
        ) * 0.09 + 0.01
        masked_m = torch.tensor(valid_rows, dtype=torch.int32, device=device)

        actual = getattr(_C, impl_name)(
            a, b, a_scale, b_scale, masked_m
        )
        references = []
        actuals = []
        for expert, rows in enumerate(valid_rows):
            reference = torch.zeros(
                (rows, n), device=device, dtype=torch.float32
            )
            for kb in range(k_blocks):
                sl = slice(kb * 128, (kb + 1) * 128)
                partial = a[expert, :rows, sl].float() @ b[expert, :, sl].float().T
                weight_scale = b_scale[expert, :, kb].repeat_interleave(128)
                reference.add_(
                    partial * a_scale[expert, :rows, kb, None] * weight_scale
                )
            references.append(reference.to(torch.bfloat16))
            actuals.append(actual[expert, :rows])
    stream.synchronize()

    actual_valid = torch.cat(actuals)
    reference_valid = torch.cat(references)
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (experts, max_m, n)
    assert torch.isfinite(actual_valid).all()
    abs_error = (actual_valid.float() - reference_valid.float()).abs()
    max_rel = abs_error.max() / reference_valid.float().abs().max().clamp_min(1e-6)
    assert max_rel.item() < 0.025, f"max_rel={max_rel.item():.6f}"


@pytest.mark.parametrize(
    "impl_name",
    [
        "sm90_fp8_block_grouped_out_test",
        "fp8_block_grouped_pipelined_out",
    ],
    ids=["sync", "cpasync-2stage"],
)
def test_sm90_fp8_block_grouped_preallocated_output(
    context: tuple[int, int, torch.device], impl_name: str
) -> None:
    rank, _, device = context
    require_sm90(device)
    generator = torch.Generator(device=device).manual_seed(20260818 + rank)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        a = torch.randn(
            (2, 64, 256), generator=generator, device=device
        ).clamp(-3, 3).to(torch.float8_e4m3fn)
        b = torch.randn(
            (2, 128, 256), generator=generator, device=device
        ).clamp(-3, 3).to(torch.float8_e4m3fn)
        a_scale = torch.rand(
            (2, 64, 2), generator=generator, device=device
        ) * 0.09 + 0.01
        b_scale = torch.rand(
            (2, 1, 2), generator=generator, device=device
        ) * 0.09 + 0.01
        masked_m = torch.tensor((64, 32), dtype=torch.int32, device=device)
        expected = _C.sm90_fp8_block_grouped_pipelined_test(
            a, b, a_scale, b_scale, masked_m
        )
        output = torch.full(
            (2, 64, 128), float("nan"), dtype=torch.bfloat16, device=device
        )
        actual = getattr(_C, impl_name)(
            a, b, a_scale, b_scale, masked_m, output
        )
    stream.synchronize()

    assert actual.data_ptr() == output.data_ptr()
    for expert, rows in enumerate((64, 32)):
        torch.testing.assert_close(
            actual[expert, :rows], expected[expert, :rows], rtol=0, atol=0
        )


def test_sm90_fp8_block_grouped_rejects_invalid_inputs(
    context: tuple[int, int, torch.device]
) -> None:
    _, _, device = context
    require_sm90(device)
    a = torch.ones((2, 64, 128), device=device).to(torch.float8_e4m3fn)
    b = torch.ones((2, 128, 128), device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones((2, 64, 1), device=device)
    b_scale = torch.ones((2, 1, 1), device=device)
    masked_m = torch.tensor((64, 32), dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError, match="expert/K dimensions must match"):
        _C.sm90_fp8_block_grouped_test(
            a, b[:1], a_scale, b_scale, masked_m
        )
    with pytest.raises(RuntimeError, match="max_m must"):
        _C.sm90_fp8_block_grouped_test(
            a[:, :32].contiguous(),
            b,
            a_scale[:, :32].contiguous(),
            b_scale,
            masked_m,
        )
    with pytest.raises(RuntimeError, match="B_scale must have shape"):
        _C.sm90_fp8_block_grouped_test(
            a, b, a_scale, b_scale[:, :, :0], masked_m
        )
    with pytest.raises(RuntimeError, match="masked_m must be int32"):
        _C.sm90_fp8_block_grouped_test(
            a, b, a_scale, b_scale, masked_m.to(torch.int64)
        )
    output = torch.empty((2, 64, 128), dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="D must have shape"):
        _C.fp8_block_grouped_pipelined_out(
            a, b, a_scale, b_scale, masked_m, output[:, :, :64]
        )
    with pytest.raises(RuntimeError, match="CUDA bfloat16"):
        _C.fp8_block_grouped_pipelined_out(
            a, b, a_scale, b_scale, masked_m, output.float()
        )
    noncontiguous = torch.empty(
        (2, 128, 64), dtype=torch.bfloat16, device=device
    ).transpose(1, 2)
    with pytest.raises(RuntimeError, match="D must be contiguous"):
        _C.fp8_block_grouped_pipelined_out(
            a, b, a_scale, b_scale, masked_m, noncontiguous
        )


def test_sm90_fp8_block_grouped_contiguous_output(
    context: tuple[int, int, torch.device]
) -> None:
    rank, _, device = context
    require_sm90(device)
    generator = torch.Generator(device=device).manual_seed(20260820 + rank)
    stream = torch.cuda.Stream(device=device)
    experts, max_m, n, k = 2, 256, 128, 256
    rows = (128, 256)
    with torch.cuda.stream(stream):
        a_grouped = torch.randn(
            (experts, max_m, k), generator=generator, device=device
        ).clamp(-3, 3).to(torch.float8_e4m3fn)
        b = torch.randn(
            (experts, n, k), generator=generator, device=device
        ).clamp(-3, 3).to(torch.float8_e4m3fn)
        a_scale_grouped = torch.rand(
            (experts, max_m, k // 128), generator=generator, device=device
        ) * 0.09 + 0.01
        b_scale = torch.rand(
            (experts, n // 128, k // 128), generator=generator, device=device
        ) * 0.09 + 0.01
        masked_m = torch.tensor(rows, dtype=torch.int32, device=device)
        reference_grouped = _C.fp8_block_grouped_pipelined_out(
            a_grouped,
            b,
            a_scale_grouped,
            b_scale,
            masked_m,
            torch.empty(
                (experts, max_m, n), dtype=torch.bfloat16, device=device
            ),
        )

        a = torch.cat(
            [a_grouped[expert, :valid] for expert, valid in enumerate(rows)]
        ).contiguous()
        a_scale = torch.cat(
            [
                a_scale_grouped[expert, :valid]
                for expert, valid in enumerate(rows)
            ]
        ).contiguous()
        m_indices = torch.repeat_interleave(
            torch.arange(experts, dtype=torch.int32, device=device),
            torch.tensor(rows, dtype=torch.int64, device=device),
        )
        output = torch.empty(
            (sum(rows), n), dtype=torch.bfloat16, device=device
        )
        actual = functional.grouped_gemm_fp8_block_out(
            a, b, a_scale, b_scale, m_indices, output
        )
    stream.synchronize()

    reference = torch.cat(
        [
            reference_grouped[expert, :valid]
            for expert, valid in enumerate(rows)
        ]
    )
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_sm90_fp8_block_grouped_contiguous_rejects_invalid_inputs(
    context: tuple[int, int, torch.device]
) -> None:
    _, _, device = context
    require_sm90(device)
    a = torch.ones((128, 128), device=device).to(torch.float8_e4m3fn)
    b = torch.ones((2, 128, 128), device=device).to(torch.float8_e4m3fn)
    a_scale = torch.ones((128, 1), device=device)
    b_scale = torch.ones((2, 1, 1), device=device)
    m_indices = torch.zeros((128,), dtype=torch.int32, device=device)
    output = torch.empty((128, 128), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="M must"):
        ops.fp8_block_grouped_contiguous_out(
            a[:32].contiguous(),
            b,
            a_scale[:32].contiguous(),
            b_scale,
            m_indices[:32].contiguous(),
            output[:32].contiguous(),
        )
    with pytest.raises(ValueError, match="m_indices must be.*int32"):
        ops.fp8_block_grouped_contiguous_out(
            a, b, a_scale, b_scale, m_indices.to(torch.int64), output
        )
    with pytest.raises(ValueError, match=r"output must be.*\[M,N\]"):
        ops.fp8_block_grouped_contiguous_out(
            a, b, a_scale, b_scale, m_indices, output[:, :64]
        )
