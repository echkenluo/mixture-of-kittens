"""Benchmark the production DeepEP-normal contiguous FP8 expert MLP."""

import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess

import deep_gemm
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor
from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant
from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale

from mok import _C


EXPERTS = int(os.environ.get("FP8_CONTIG_MLP_EXPERTS", 64))
HIDDEN = int(os.environ.get("FP8_CONTIG_MLP_HIDDEN", 4096))
INTERMEDIATE = int(os.environ.get("FP8_CONTIG_MLP_INTERMEDIATE", 2048))
PATTERN = [
    int(value)
    for value in os.environ.get("FP8_CONTIG_MLP_PATTERN", "256").split(",")
]
WARMUP = int(os.environ.get("FP8_CONTIG_MLP_WARMUP", 10))
ITERS = int(os.environ.get("FP8_CONTIG_MLP_ITERS", 50))
OUTPUT = os.environ.get(
    "FP8_CONTIG_MLP_OUTPUT", "/mok/bench-sm90-fp8-contiguous-mlp.json"
)
EXPECTED_COMMIT = os.environ.get("FP8_CONTIG_MLP_EXPECTED_COMMIT")
EXPECTED_SO_SHA256 = os.environ.get("FP8_CONTIG_MLP_EXPECTED_SO_SHA256")
EXPECTED_HARNESS_SHA256 = os.environ.get(
    "FP8_CONTIG_MLP_EXPECTED_HARNESS_SHA256"
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance() -> dict:
    repo = pathlib.Path(__file__).resolve().parents[1]
    shared_objects = sorted((repo / "mok").glob("_C*.so"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else "unavailable",
        "source_state": status.stdout.splitlines(),
        "harness_sha256": sha256(pathlib.Path(__file__).resolve()),
        "so_path": str(shared_objects[0]) if len(shared_objects) == 1 else None,
        "so_sha256": sha256(shared_objects[0]) if len(shared_objects) == 1 else None,
        "deep_gemm_module": str(pathlib.Path(deep_gemm.__file__).resolve()),
        "sglang_activation_module": str(
            pathlib.Path(silu_and_mul_contig_post_quant.__code__.co_filename).resolve()
        ),
        "sglang_scale_layout_module": str(
            pathlib.Path(tma_align_input_scale.__code__.co_filename).resolve()
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


def validate_provenance(record: dict) -> None:
    required = {
        "FP8_CONTIG_MLP_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "FP8_CONTIG_MLP_EXPECTED_SO_SHA256": EXPECTED_SO_SHA256,
        "FP8_CONTIG_MLP_EXPECTED_HARNESS_SHA256": EXPECTED_HARNESS_SHA256,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"missing frozen provenance gates: {missing}")
    mismatches = []
    for label, actual, expected in (
        ("git_head", record["git_head"], EXPECTED_COMMIT),
        ("so_sha256", record["so_sha256"], EXPECTED_SO_SHA256),
        ("harness_sha256", record["harness_sha256"], EXPECTED_HARNESS_SHA256),
    ):
        if actual != expected:
            mismatches.append(f"{label}: actual={actual} expected={expected}")
    if record["source_state"]:
        mismatches.append(f"source_state is dirty: {record['source_state']}")
    if mismatches:
        raise RuntimeError("frozen provenance gate failed: " + "; ".join(mismatches))


def aligned_rows() -> list[int]:
    if not PATTERN or EXPERTS % len(PATTERN) != 0:
        raise ValueError(
            "FP8_CONTIG_MLP_PATTERN length must divide FP8_CONTIG_MLP_EXPERTS"
        )
    rows = PATTERN * (EXPERTS // len(PATTERN))
    if any(value < 0 or value % 128 for value in rows):
        raise ValueError(
            "DeepGEMM comparison requires every active expert segment to be "
            "M128 aligned"
        )
    if not any(rows):
        raise ValueError("at least one expert must have a non-zero segment")
    return rows


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


def summarize(samples: list[float], flops: int | None = None) -> dict:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    result = {
        "p50_ms": p50,
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }
    if flops is not None:
        result["tflops_p50"] = flops / (p50 * 1e9)
    return result


def error(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = actual.float() - reference.float()
    square_reference = torch.sum(reference.float().square())
    abs_max = float(diff.abs().max())
    return {
        "abs_max": abs_max,
        "rel_maxnorm": abs_max / max(float(reference.float().abs().max()), 1e-6),
        "relative_l2": math.sqrt(
            float(torch.sum(diff.square())) / max(float(square_reference), 1e-20)
        ),
    }


def main() -> None:
    assert hasattr(_C, "fp8_block_grouped_contiguous_out")
    assert HIDDEN % 128 == 0 and INTERMEDIATE % 128 == 0
    provenance_record = provenance()
    validate_provenance(provenance_record)
    rows = aligned_rows()
    total_m = sum(rows)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260816)
    hidden = torch.randn(
        (total_m, HIDDEN), generator=generator, device=device, dtype=torch.bfloat16
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    w13 = torch.randn(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    w2 = torch.randn(
        (EXPERTS, HIDDEN, INTERMEDIATE),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    hidden_scale = torch.rand(
        (total_m, HIDDEN // 128), generator=generator, device=device
    ) * 0.09 + 0.01
    w13_scale = torch.rand(
        (EXPERTS, (2 * INTERMEDIATE) // 128, HIDDEN // 128),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
    w2_scale = torch.rand(
        (EXPERTS, HIDDEN // 128, INTERMEDIATE // 128),
        generator=generator,
        device=device,
    ) * 0.09 + 0.01
    m_indices = torch.repeat_interleave(
        torch.arange(EXPERTS, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int64, device=device),
    )

    w13_scale_dg = get_mn_major_tma_aligned_tensor(w13_scale)
    w2_scale_dg = get_mn_major_tma_aligned_tensor(w2_scale)
    mok_gateup = torch.empty(
        (total_m, 2 * INTERMEDIATE), dtype=torch.bfloat16, device=device
    )
    dg_gateup = torch.empty_like(mok_gateup)
    mok_activation = torch.empty(
        (total_m, INTERMEDIATE), dtype=torch.float8_e4m3fn, device=device
    )
    dg_activation = torch.empty_like(mok_activation)
    mok_activation_scale = torch.empty(
        (total_m, INTERMEDIATE // 128), dtype=torch.float32, device=device
    )
    dg_activation_scale = torch.empty_like(mok_activation_scale)
    mok_down = torch.empty((total_m, HIDDEN), dtype=torch.bfloat16, device=device)
    dg_down = torch.empty_like(mok_down)

    def activate(gateup, output, output_scale):
        silu_and_mul_contig_post_quant(
            input=gateup,
            output=output,
            output_scale=output_scale,
            quant_group_size=128,
            scale_ue8m0=False,
            transposed=False,
            swiglu_limit=10.0,
            swizzle=False,
        )
        return output

    def run_mok_gateup():
        return _C.fp8_block_grouped_contiguous_out(
            hidden, w13, hidden_scale, w13_scale, m_indices, mok_gateup
        )

    def run_deepgemm_gateup():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (hidden, tma_align_input_scale(hidden_scale)),
            (w13, w13_scale_dg),
            dg_gateup,
            m_indices,
        )
        return dg_gateup

    def run_mok_down():
        return _C.fp8_block_grouped_contiguous_out(
            mok_activation,
            w2,
            mok_activation_scale,
            w2_scale,
            m_indices,
            mok_down,
        )

    def run_deepgemm_down():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (
                dg_activation,
                tma_align_input_scale(dg_activation_scale),
            ),
            (w2, w2_scale_dg),
            dg_down,
            m_indices,
        )
        return dg_down

    def run_mok_mlp():
        activate(run_mok_gateup(), mok_activation, mok_activation_scale)
        return run_mok_down()

    def run_deepgemm_mlp():
        activate(run_deepgemm_gateup(), dg_activation, dg_activation_scale)
        return run_deepgemm_down()

    run_mok_mlp()
    run_deepgemm_mlp()
    torch.cuda.synchronize()
    gateup_error = error(run_mok_gateup(), run_deepgemm_gateup())
    final_error = error(run_mok_mlp(), run_deepgemm_mlp())
    torch.cuda.synchronize()
    if not (
        gateup_error["rel_maxnorm"] < 0.025
        and final_error["rel_maxnorm"] < 0.05
        and final_error["relative_l2"] < 0.05
    ):
        raise RuntimeError(
            f"correctness failed: gateup={gateup_error} final={final_error}"
        )

    gateup_flops = 2 * total_m * (2 * INTERMEDIATE) * HIDDEN
    down_flops = 2 * total_m * HIDDEN * INTERMEDIATE
    total_flops = gateup_flops + down_flops
    mok_a = time_calls(run_mok_mlp)
    deepgemm_samples = time_calls(run_deepgemm_mlp)
    mok_a2 = time_calls(run_mok_mlp)
    components = {
        "mok_gateup": summarize(time_calls(run_mok_gateup), gateup_flops),
        "deepgemm_gateup_with_scale_layout": summarize(
            time_calls(run_deepgemm_gateup), gateup_flops
        ),
        "mok_activation": summarize(
            time_calls(
                lambda: activate(
                    mok_gateup, mok_activation, mok_activation_scale
                )
            )
        ),
        "deepgemm_activation": summarize(
            time_calls(
                lambda: activate(dg_gateup, dg_activation, dg_activation_scale)
            )
        ),
        "mok_down": summarize(time_calls(run_mok_down), down_flops),
        "deepgemm_down_with_scale_layout": summarize(
            time_calls(run_deepgemm_down), down_flops
        ),
    }
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
        "schema": "bench-sm90-fp8-contiguous-mlp.v1",
        "scope": "routed expert compute; excludes dispatch/combine/shared expert",
        "shape": {
            "experts": EXPERTS,
            "m": total_m,
            "rows_per_expert": rows,
            "avg_aligned_m": total_m / EXPERTS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
        },
        "correctness": {"gateup": gateup_error, "final": final_error},
        "timing": {
            "warmup": WARMUP,
            "iters": ITERS,
            "order": ["mok_a", "deepgemm", "mok_a2", "components"],
            "mok_boundary": "two preallocated GEMMs + production activation",
            "deepgemm_boundary": (
                "two preallocated GEMMs + production activation + two "
                "activation-scale TMA layout transforms"
            ),
            "weight_scale_layout_conversion": "setup and excluded",
        },
        "results": {
            "mok_a": summarize(mok_a, total_flops),
            "deepgemm": summarize(deepgemm_samples, total_flops),
            "mok_a2": summarize(mok_a2, total_flops),
            "components": components,
        },
        "provenance": provenance_record,
        "gpu_snapshot": snapshot,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    temporary = OUTPUT + ".tmp"
    with open(temporary, "w") as output_file:
        json.dump(record, output_file, indent=1)
    os.replace(temporary, OUTPUT)
    print(
        "BENCH|fp8_contiguous_mlp"
        f"|mok_a={record['results']['mok_a']['p50_ms']:.4f}ms"
        f"|deepgemm={record['results']['deepgemm']['p50_ms']:.4f}ms"
        f"|mok_a2={record['results']['mok_a2']['p50_ms']:.4f}ms"
        f"|rel={final_error['rel_maxnorm']:.6f}"
        f"|l2={final_error['relative_l2']:.6f}"
    )


if __name__ == "__main__":
    main()
