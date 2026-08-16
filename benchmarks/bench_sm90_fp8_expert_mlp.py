"""Production-semantics FP8 routed-expert MLP canary for SM90.

This benchmark composes the two grouped GEMMs with SGLang's DeepSeek-V4
clamped SwiGLU and K128 post-activation quantizer:

    FP8 hidden -> BF16 gate/up -> clamped SwiGLU -> FP8 -> BF16 down

It intentionally excludes dispatch, combine, routing, and the shared expert.
The MoK path consumes row-major FP32 block scales.  The DeepGEMM path starts
from the same raw activation scales and includes the TMA-layout transforms
performed by SGLang's masked-GEMM runner.  Weight-scale transforms are setup,
matching weights that have already been prepared by model loading.

Both paths write to preallocated GEMM outputs.  Every timing boundary is
recorded, so this is a fair routed-expert compute canary rather than an E2E
winner claim.
"""

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
from sglang.jit_kernel.dsv4 import silu_and_mul_masked_post_quant

from mok import _C


EXPERTS = int(os.environ.get("FP8_MLP_EXPERTS", 64))
MAX_M = int(os.environ.get("FP8_MLP_MAX_M", 512))
HIDDEN = int(os.environ.get("FP8_MLP_HIDDEN", 4096))
INTERMEDIATE = int(os.environ.get("FP8_MLP_INTERMEDIATE", 2048))
EXPECTED_M = int(os.environ.get("FP8_MLP_EXPECTED_M", 384))
MODEL_TOPK = int(os.environ.get("FP8_MLP_MODEL_TOPK", 6))
SWIGLU_LIMIT = float(os.environ.get("FP8_MLP_SWIGLU_LIMIT", 10.0))
WARMUP = int(os.environ.get("FP8_MLP_WARMUP", 5))
ITERS = int(os.environ.get("FP8_MLP_ITERS", 30))
OUTPUT = os.environ.get("FP8_MLP_OUTPUT", "/mok/bench-sm90-fp8-expert-mlp.json")
BINARY_BUILD_COMMIT = os.environ.get("FP8_MLP_BINARY_BUILD_COMMIT")


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
        "binary_build_commit": BINARY_BUILD_COMMIT,
        "deep_gemm_module": str(pathlib.Path(deep_gemm.__file__).resolve()),
        "deep_gemm_version": getattr(deep_gemm, "__version__", None),
        "sglang_activation_module": str(
            pathlib.Path(silu_and_mul_masked_post_quant.__code__.co_filename).resolve()
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


def distribution() -> list[int]:
    if EXPECTED_M == 384 and MAX_M >= 448 and EXPERTS % 8 == 0:
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


def summarize(samples: list[float], effective_flops: int | None = None) -> dict:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    result = {
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }
    if effective_flops is not None:
        result["tflops_p50"] = effective_flops / (p50 * 1e9)
    return result


def valid_error(
    actual: torch.Tensor, reference: torch.Tensor, valid_rows: list[int]
) -> dict:
    abs_max = 0.0
    reference_max = 0.0
    square_error = 0.0
    square_reference = 0.0
    elements = 0
    for expert, rows in enumerate(valid_rows):
        actual_slice = actual[expert, :rows].float()
        reference_slice = reference[expert, :rows].float()
        diff = actual_slice - reference_slice
        abs_max = max(abs_max, float(diff.abs().max()))
        reference_max = max(reference_max, float(reference_slice.abs().max()))
        square_error += float(torch.sum(diff * diff))
        square_reference += float(torch.sum(reference_slice * reference_slice))
        elements += diff.numel()
    return {
        "abs_max": abs_max,
        "rel_maxnorm": abs_max / max(reference_max, 1e-6),
        "relative_l2": math.sqrt(square_error / max(square_reference, 1e-20)),
        "elements": elements,
    }


def main() -> None:
    if not BINARY_BUILD_COMMIT:
        raise RuntimeError(
            "FP8_MLP_BINARY_BUILD_COMMIT is required to bind the loaded SO "
            "to its clean build source"
        )
    assert hasattr(_C, "fp8_block_grouped_pipelined_out")
    assert HIDDEN % 128 == 0
    assert INTERMEDIATE % 128 == 0
    assert (2 * INTERMEDIATE) % 128 == 0
    assert MAX_M % 64 == 0

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260816)
    valid_rows = distribution()
    valid_rows_total = sum(valid_rows)
    masked_m = torch.tensor(valid_rows, dtype=torch.int32, device=device)

    # The production activation wrapper launches T * topk CTAs.  This
    # expert-major synthetic shape needs enough CTAs to cover every valid row;
    # record the derived launch factor separately from the model's routing top-k.
    activation_launch_topk = math.ceil(valid_rows_total / MAX_M)

    hidden = torch.randn(
        (EXPERTS, MAX_M, HIDDEN),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
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
        (EXPERTS, MAX_M, HIDDEN // 128),
        generator=generator,
        device=device,
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

    # Model-loaded weight scales are setup.  Activation-scale transforms stay
    # inside the DeepGEMM full-pipeline boundary, matching SGLang's runner.
    w13_scale_dg = get_mn_major_tma_aligned_tensor(w13_scale)
    w2_scale_dg = get_mn_major_tma_aligned_tensor(w2_scale)

    dg_gateup = torch.empty(
        (EXPERTS, MAX_M, 2 * INTERMEDIATE),
        dtype=torch.bfloat16,
        device=device,
    )
    mok_gateup_output = torch.empty_like(dg_gateup)
    mok_activation = torch.empty(
        (EXPERTS, MAX_M, INTERMEDIATE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    mok_activation_scale = torch.empty(
        (EXPERTS, MAX_M, INTERMEDIATE // 128),
        dtype=torch.float32,
        device=device,
    )
    dg_activation = torch.empty_like(mok_activation)
    dg_activation_scale = torch.empty_like(mok_activation_scale)
    dg_down = torch.empty(
        (EXPERTS, MAX_M, HIDDEN), dtype=torch.bfloat16, device=device
    )
    mok_down_output = torch.empty_like(dg_down)

    def activate(
        gateup: torch.Tensor, output: torch.Tensor, output_scale: torch.Tensor
    ) -> torch.Tensor:
        silu_and_mul_masked_post_quant(
            gateup,
            output,
            output_scale,
            128,
            masked_m,
            scale_ue8m0=False,
            topk=activation_launch_topk,
            transposed=False,
            swiglu_limit=SWIGLU_LIMIT,
            swizzle=False,
        )
        return output

    def run_mok_gateup() -> torch.Tensor:
        return _C.fp8_block_grouped_pipelined_out(
            hidden,
            w13,
            hidden_scale,
            w13_scale,
            masked_m,
            mok_gateup_output,
        )

    def run_deepgemm_gateup_prepared() -> torch.Tensor:
        hidden_scale_dg = get_mn_major_tma_aligned_tensor(hidden_scale)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (hidden, hidden_scale_dg),
            (w13, w13_scale_dg),
            dg_gateup,
            masked_m,
            EXPECTED_M,
        )
        return dg_gateup

    def run_mok_down() -> torch.Tensor:
        return _C.fp8_block_grouped_pipelined_out(
            mok_activation,
            w2,
            mok_activation_scale,
            w2_scale,
            masked_m,
            mok_down_output,
        )

    def run_deepgemm_down_prepared() -> torch.Tensor:
        dg_activation_scale_dg = get_mn_major_tma_aligned_tensor(
            dg_activation_scale
        )
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (dg_activation, dg_activation_scale_dg),
            (w2, w2_scale_dg),
            dg_down,
            masked_m,
            EXPECTED_M,
        )
        return dg_down

    def prepare_hidden_scale() -> torch.Tensor:
        return get_mn_major_tma_aligned_tensor(hidden_scale)

    def prepare_activation_scale() -> torch.Tensor:
        return get_mn_major_tma_aligned_tensor(dg_activation_scale)

    def run_mok_mlp() -> torch.Tensor:
        gateup = run_mok_gateup()
        activate(gateup, mok_activation, mok_activation_scale)
        return run_mok_down()

    def run_deepgemm_mlp() -> torch.Tensor:
        run_deepgemm_gateup_prepared()
        activate(dg_gateup, dg_activation, dg_activation_scale)
        return run_deepgemm_down_prepared()

    # JIT, numerical, and complete-pipeline gate before timing.
    mok_output = run_mok_mlp()
    dg_output = run_deepgemm_mlp()
    torch.cuda.synchronize()
    final_error = valid_error(mok_output, dg_output, valid_rows)
    gateup_error = valid_error(
        run_mok_gateup(), run_deepgemm_gateup_prepared(), valid_rows
    )
    torch.cuda.synchronize()
    if not (
        math.isfinite(final_error["abs_max"])
        and final_error["rel_maxnorm"] < 0.05
        and final_error["relative_l2"] < 0.05
    ):
        raise RuntimeError(f"final correctness failed: {final_error}")
    if not gateup_error["rel_maxnorm"] < 0.025:
        raise RuntimeError(f"gate/up correctness failed: {gateup_error}")

    gateup_flops = 2 * valid_rows_total * (2 * INTERMEDIATE) * HIDDEN
    down_flops = 2 * valid_rows_total * HIDDEN * INTERMEDIATE
    total_flops = gateup_flops + down_flops

    mok_full_a = time_calls(run_mok_mlp)
    deepgemm_full = time_calls(run_deepgemm_mlp)
    mok_full_a2 = time_calls(run_mok_mlp)

    # Component timings retain production-required raw activation scale layout
    # transforms on DeepGEMM.  Activation uses a fixed, already-produced input.
    mok_gateup_seed = run_mok_gateup()
    run_deepgemm_gateup_prepared()
    activate(mok_gateup_seed, mok_activation, mok_activation_scale)
    activate(dg_gateup, dg_activation, dg_activation_scale)
    torch.cuda.synchronize()
    components = {
        "mok_gateup": summarize(time_calls(run_mok_gateup), gateup_flops),
        "deepgemm_gateup_with_input_scale_layout": summarize(
            time_calls(run_deepgemm_gateup_prepared), gateup_flops
        ),
        "production_activation_on_mok_gateup": summarize(
            time_calls(
                lambda: activate(
                    mok_gateup_seed, mok_activation, mok_activation_scale
                )
            )
        ),
        "production_activation_on_deepgemm_gateup": summarize(
            time_calls(
                lambda: activate(
                    dg_gateup, dg_activation, dg_activation_scale
                )
            )
        ),
        "deepgemm_input_scale_layout_only": summarize(
            time_calls(prepare_hidden_scale)
        ),
        "deepgemm_activation_scale_layout_only": summarize(
            time_calls(prepare_activation_scale)
        ),
        "mok_down": summarize(time_calls(run_mok_down), down_flops),
        "deepgemm_down_with_activation_scale_layout": summarize(
            time_calls(run_deepgemm_down_prepared), down_flops
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
        "schema": "bench-sm90-fp8-expert-mlp.v1",
        "scope": (
            "routed expert compute only; excludes dispatch/combine/shared expert"
        ),
        "shape": {
            "experts": EXPERTS,
            "max_m": MAX_M,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "expected_m": EXPECTED_M,
            "masked_m": valid_rows,
            "valid_rows_total": valid_rows_total,
            "model_topk": MODEL_TOPK,
            "activation_launch_topk": activation_launch_topk,
            "swiglu_limit": SWIGLU_LIMIT,
        },
        "correctness": {
            "comparator": (
                "DeepGEMM with identical inputs/scales and shared SGLang V4 "
                "activation"
            ),
            "gateup": gateup_error,
            "final": final_error,
        },
        "timing": {
            "warmup": WARMUP,
            "iters": ITERS,
            "order": [
                "mok_full_a",
                "deepgemm_full",
                "mok_full_a2",
                "components",
            ],
            "mok_boundary": (
                "two preallocated grouped GEMMs + production activation"
            ),
            "deepgemm_boundary": (
                "two preallocated grouped GEMMs + production activation + "
                "input/activation TMA scale transforms"
            ),
            "weight_scale_layout_conversion": "setup/excluded for both implementations",
        },
        "results": {
            "mok_full_a": summarize(mok_full_a, total_flops),
            "deepgemm_full": summarize(deepgemm_full, total_flops),
            "mok_full_a2": summarize(mok_full_a2, total_flops),
            "components": components,
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
        "BENCH|fp8_expert_mlp"
        f"|mok_a={record['results']['mok_full_a']['p50_ms']:.4f}ms"
        f"|deepgemm={record['results']['deepgemm_full']['p50_ms']:.4f}ms"
        f"|mok_a2={record['results']['mok_full_a2']['p50_ms']:.4f}ms"
        f"|rel={final_error['rel_maxnorm']:.6g}"
        f"|l2={final_error['relative_l2']:.6g}|out={OUTPUT}"
    )


if __name__ == "__main__":
    main()
