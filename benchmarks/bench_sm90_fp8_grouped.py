"""Canary comparison for the SM90 FP8 block-scale grouped GEMM.

This compares one routed-expert GEMM, not a complete MoE layer.  Inputs use
the DeepSeek-V4 Hopper contract: E4M3 values, per-row K128 activation scales,
per-N128/K128 weight scales, and expert-major masked rows.  Scale layout
conversion for DeepGEMM is setup and excluded from timing.

The benchmark records both the original allocating MoK binding and the
preallocated-output binding used for a boundary-matched DeepGEMM comparison.
"""

import hashlib
import json
import os
import pathlib
import statistics
import subprocess

import deep_gemm
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

from mok import _C


EXPERTS = int(os.environ.get("FP8_GROUPED_EXPERTS", 64))
MAX_M = int(os.environ.get("FP8_GROUPED_MAX_M", 512))
N = int(os.environ.get("FP8_GROUPED_N", 4096))
K = int(os.environ.get("FP8_GROUPED_K", 4096))
EXPECTED_M = int(os.environ.get("FP8_GROUPED_EXPECTED_M", 384))
WARMUP = int(os.environ.get("FP8_GROUPED_WARMUP", 10))
ITERS = int(os.environ.get("FP8_GROUPED_ITERS", 100))
OUTPUT = os.environ.get("FP8_GROUPED_OUTPUT", "/mok/bench-sm90-fp8-grouped.json")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance() -> dict:
    repo = pathlib.Path(__file__).resolve().parents[1]
    shared_objects = sorted((repo / "mok").glob("_C*.so"))
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_head": git.stdout.strip() if git.returncode == 0 else "unavailable",
        "source_state": subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines(),
        "harness_sha256": sha256(pathlib.Path(__file__).resolve()),
        "so_path": str(shared_objects[0]) if len(shared_objects) == 1 else None,
        "so_sha256": sha256(shared_objects[0]) if len(shared_objects) == 1 else None,
        "deep_gemm_module": str(pathlib.Path(deep_gemm.__file__).resolve()),
        "deep_gemm_version": getattr(deep_gemm, "__version__", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


def distribution() -> list[int]:
    if EXPECTED_M == 384 and MAX_M >= 448 and EXPERTS % 8 == 0:
        # Mean is exactly 384 and total routes are 24,576 for E=64, matching
        # 4 ranks * 4096 tokens/rank * top-6 / EP4 under uniform routing.
        pattern = [320, 336, 352, 368, 400, 416, 432, 448]
        return pattern * (EXPERTS // len(pattern))
    return [min(EXPECTED_M, MAX_M)] * EXPERTS


def time_calls(fn) -> list[float]:
    keep = None
    for _ in range(WARMUP):
        keep = fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(ITERS)
    ]
    for start, end in events:
        start.record()
        keep = fn()
        end.record()
    torch.cuda.synchronize()
    assert keep is not None
    return [start.elapsed_time(end) for start, end in events]


def summarize(samples: list[float], effective_flops: int) -> dict:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "tflops_p50": effective_flops / (p50 * 1e9),
        "samples_ms": samples,
    }


def main() -> None:
    assert hasattr(_C, "sm90_fp8_block_grouped_test")
    assert hasattr(_C, "fp8_block_grouped_pipelined_out")
    assert K % 128 == 0 and N % 128 == 0 and MAX_M % 64 == 0
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260816)
    k_blocks = K // 128

    a = torch.randn(
        (EXPERTS, MAX_M, K), generator=generator, device=device,
        dtype=torch.bfloat16,
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    b = torch.randn(
        (EXPERTS, N, K), generator=generator, device=device,
        dtype=torch.bfloat16,
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    a_scale = torch.rand(
        (EXPERTS, MAX_M, k_blocks), generator=generator, device=device
    ) * 0.09 + 0.01
    b_scale = torch.rand(
        (EXPERTS, N // 128, k_blocks), generator=generator, device=device
    ) * 0.09 + 0.01
    valid_rows = distribution()
    masked_m = torch.tensor(valid_rows, dtype=torch.int32, device=device)

    # DeepGEMM consumes the same values/scales through an MN-major TMA layout.
    a_scale_dg = get_mn_major_tma_aligned_tensor(a_scale)
    b_scale_dg = get_mn_major_tma_aligned_tensor(b_scale)
    dg_output = torch.empty(
        (EXPERTS, MAX_M, N), dtype=torch.bfloat16, device=device
    )
    mok_pipe_output = torch.empty_like(dg_output)

    def run_mok_sync():
        return _C.sm90_fp8_block_grouped_test(
            a, b, a_scale, b_scale, masked_m
        )

    def run_mok_pipelined():
        return _C.sm90_fp8_block_grouped_pipelined_test(
            a, b, a_scale, b_scale, masked_m
        )

    def run_mok_pipelined_out():
        return _C.fp8_block_grouped_pipelined_out(
            a, b, a_scale, b_scale, masked_m, mok_pipe_output
        )

    def run_deepgemm():
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (a, a_scale_dg),
            (b, b_scale_dg),
            dg_output,
            masked_m,
            EXPECTED_M,
        )
        return dg_output

    # One untimed numerical gate before any performance claim.
    mok_output = run_mok_pipelined_out()
    run_deepgemm()
    torch.cuda.synchronize()
    abs_max = 0.0
    reference_max = 0.0
    for expert, rows in enumerate(valid_rows):
        diff = (
            mok_output[expert, :rows].float()
            - dg_output[expert, :rows].float()
        ).abs()
        abs_max = max(abs_max, float(diff.max()))
        reference_max = max(
            reference_max, float(dg_output[expert, :rows].float().abs().max())
        )
    rel_maxnorm = abs_max / max(reference_max, 1e-6)
    if not (torch.isfinite(torch.tensor(abs_max)) and rel_maxnorm < 0.025):
        raise RuntimeError(
            f"correctness failed: abs_max={abs_max} rel_maxnorm={rel_maxnorm}"
        )

    effective_flops = 2 * sum(valid_rows) * N * K
    mok_sync_a = time_calls(run_mok_sync)
    mok_pipe_a = time_calls(run_mok_pipelined)
    mok_pipe_out_a = time_calls(run_mok_pipelined_out)
    deepgemm_samples = time_calls(run_deepgemm)
    mok_pipe_out_a2 = time_calls(run_mok_pipelined_out)
    mok_pipe_a2 = time_calls(run_mok_pipelined)
    mok_sync_a2 = time_calls(run_mok_sync)

    snapshot = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,clocks.sm,clocks.mem,power.draw,temperature.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip().splitlines()
    record = {
        "schema": "bench-sm90-fp8-grouped.v2",
        "shape": {
            "experts": EXPERTS,
            "max_m": MAX_M,
            "n": N,
            "k": K,
            "expected_m": EXPECTED_M,
            "masked_m": valid_rows,
            "valid_rows_total": sum(valid_rows),
        },
        "correctness": {
            "comparator": "DeepGEMM on identical quantized values/scales",
            "abs_max": abs_max,
            "rel_maxnorm": rel_maxnorm,
        },
        "timing": {
            "warmup": WARMUP,
            "iters": ITERS,
            "order": [
                "mok_sync_a",
                "mok_pipe_a",
                "mok_pipe_out_a",
                "deepgemm",
                "mok_pipe_out_a2",
                "mok_pipe_a2",
                "mok_sync_a2",
            ],
            "mok_boundary": "binding + output allocation + grouped kernel",
            "mok_preallocated_boundary": (
                "binding + grouped kernel into preallocated output"
            ),
            "deepgemm_boundary": "binding + grouped kernel into preallocated output",
            "scale_layout_conversion": "excluded for both implementations",
        },
        "results": {
            "mok_sync_a": summarize(mok_sync_a, effective_flops),
            "mok_pipe_a": summarize(mok_pipe_a, effective_flops),
            "mok_pipe_out_a": summarize(mok_pipe_out_a, effective_flops),
            "deepgemm": summarize(deepgemm_samples, effective_flops),
            "mok_pipe_out_a2": summarize(mok_pipe_out_a2, effective_flops),
            "mok_pipe_a2": summarize(mok_pipe_a2, effective_flops),
            "mok_sync_a2": summarize(mok_sync_a2, effective_flops),
        },
        "deepgemm_num_sms": (
            deep_gemm.get_num_sms() if hasattr(deep_gemm, "get_num_sms") else None
        ),
        "provenance": provenance(),
        "gpu_snapshot": snapshot,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, OUTPUT)
    print(
        "BENCH|fp8_grouped"
        f"|mok_sync_a={record['results']['mok_sync_a']['p50_ms']:.4f}ms"
        f"|mok_pipe_a={record['results']['mok_pipe_a']['p50_ms']:.4f}ms"
        f"|mok_pipe_out_a={record['results']['mok_pipe_out_a']['p50_ms']:.4f}ms"
        f"|deepgemm={record['results']['deepgemm']['p50_ms']:.4f}ms"
        f"|mok_pipe_out_a2={record['results']['mok_pipe_out_a2']['p50_ms']:.4f}ms"
        f"|mok_pipe_a2={record['results']['mok_pipe_a2']['p50_ms']:.4f}ms"
        f"|mok_sync_a2={record['results']['mok_sync_a2']['p50_ms']:.4f}ms"
        f"|rel={rel_maxnorm:.6g}|out={OUTPUT}"
    )


if __name__ == "__main__":
    main()
