"""Check analytic point-matrix construction and padding semantics on CPU."""

import torch

from tests.warprole_route_reference import (
    block_scales, point_gemm, point_parameters, routed_forward,
)


def test_point_matrix_against_explicit_selected_rows():
    torch.set_num_threads(4)
    for stage, k in (("w13", 4096), ("w2", 2048)):
        experts = torch.tensor([0, 65, 255])
        x = ((torch.arange(3 * k).reshape(3, k) % 7) - 3).float()
        xs = torch.full((3, k // 128), .125)
        got = point_gemm(x, xs, experts, stage)
        columns, values = point_parameters(experts, stage)
        scales = block_scales(experts, stage)
        for i in range(3):
            for n in (0, 127, 128, 2047, 2048, 4095):
                dense_row = torch.zeros(k)
                dense_row[columns[i, n]] = values[i, n]
                expected = 0.0
                for group in range(k // 128):
                    section = slice(group * 128, (group + 1) * 128)
                    expected += float(x[i, section] @ dense_row[section]) * float(xs[i, group] * scales[i, n // 128, group])
                assert got[i, n] == torch.tensor(expected, dtype=torch.bfloat16)


def test_padding_ignores_nonzero_router_weights():
    torch.set_num_threads(4)
    x = torch.ones(2, 4096).to(torch.float8_e4m3fn)
    xs = torch.ones(2, 32)
    ids = torch.full((2, 6), -1, dtype=torch.int32)
    weights = torch.ones(2, 6)
    assert bool((routed_forward(x, xs, ids, weights) == 0).all())
