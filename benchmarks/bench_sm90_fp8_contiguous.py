"""Compare MoK and DeepGEMM on the DeepEP-normal contiguous FP8 contract."""

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


EXPERTS = int(os.environ.get("FP8_CONTIG_EXPERTS", 64))
N = int(os.environ.get("FP8_CONTIG_N", 4096))
K = int(os.environ.get("FP8_CONTIG_K", 4096))
PATTERN = [
    int(value)
    for value in os.environ.get(
        "FP8_CONTIG_PATTERN", "384,384,384,384,512,512,512,512"
    ).split(",")
]
WARMUP = int(os.environ.get("FP8_CONTIG_WARMUP", 10))
ITERS = int(os.environ.get("FP8_CONTIG_ITERS", 100))
OUTPUT = os.environ.get(
    "FP8_CONTIG_OUTPUT", "/mok/bench-sm90-fp8-contiguous.json"
)
EXPECTED_COMMIT = os.environ.get("FP8_CONTIG_EXPECTED_COMMIT")
EXPECTED_SO_SHA256 = os.environ.get("FP8_CONTIG_EXPECTED_SO_SHA256")
EXPECTED_HARNESS_SHA256 = os.environ.get("FP8_CONTIG_EXPECTED_HARNESS_SHA256")


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
        "deep_gemm_version": getattr(deep_gemm, "__version__", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


def aligned_rows() -> list[int]:
    if not PATTERN or EXPERTS % len(PATTERN) != 0:
        raise ValueError("FP8_CONTIG_PATTERN length must divide FP8_CONTIG_EXPERTS")
    rows = PATTERN * (EXPERTS // len(PATTERN))
    if any(value < 0 or value % 128 for value in rows):
        raise ValueError(
            "DeepGEMM comparison requires every active expert segment to be "
            "M128 aligned"
        )
    if not any(rows):
        raise ValueError("at least one expert must have a non-zero segment")
    return rows


def validate_provenance(record: dict) -> None:
    required = {
        "FP8_CONTIG_EXPECTED_COMMIT": EXPECTED_COMMIT,
        "FP8_CONTIG_EXPECTED_SO_SHA256": EXPECTED_SO_SHA256,
        "FP8_CONTIG_EXPECTED_HARNESS_SHA256": EXPECTED_HARNESS_SHA256,
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


def summarize(samples: list[float], flops: int) -> dict:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "tflops_p50": flops / (p50 * 1e9),
        "samples_ms": samples,
    }


def main() -> None:
    assert hasattr(_C, "fp8_block_grouped_contiguous_out")
    assert K % 128 == 0 and N % 128 == 0
    provenance_record = provenance()
    validate_provenance(provenance_record)
    rows = aligned_rows()
    total_m = sum(rows)
    assert total_m % 64 == 0

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260820)
    k_blocks = K // 128
    a = torch.randn(
        (total_m, K), generator=generator, device=device, dtype=torch.bfloat16
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    b = torch.randn(
        (EXPERTS, N, K), generator=generator, device=device, dtype=torch.bfloat16
    ).clamp(-3, 3).to(torch.float8_e4m3fn)
    a_scale = torch.rand(
        (total_m, k_blocks), generator=generator, device=device
    ) * 0.09 + 0.01
    b_scale = torch.rand(
        (EXPERTS, N // 128, k_blocks), generator=generator, device=device
    ) * 0.09 + 0.01
    m_indices = torch.repeat_interleave(
        torch.arange(EXPERTS, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int64, device=device),
    )

    a_scale_dg = get_mn_major_tma_aligned_tensor(a_scale)
    b_scale_dg = get_mn_major_tma_aligned_tensor(b_scale)
    mok_output = torch.empty((total_m, N), dtype=torch.bfloat16, device=device)
    dg_output = torch.empty_like(mok_output)

    def run_mok():
        return _C.fp8_block_grouped_contiguous_out(
            a, b, a_scale, b_scale, m_indices, mok_output
        )

    def run_deepgemm():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a, a_scale_dg), (b, b_scale_dg), dg_output, m_indices
        )
        return dg_output

    run_mok()
    run_deepgemm()
    torch.cuda.synchronize()
    error = (mok_output.float() - dg_output.float()).abs()
    abs_max = float(error.max())
    rel_maxnorm = abs_max / max(float(dg_output.float().abs().max()), 1e-6)
    if not (bool(torch.isfinite(error).all().item()) and rel_maxnorm < 0.025):
        raise RuntimeError(
            f"correctness failed: abs_max={abs_max} rel_maxnorm={rel_maxnorm}"
        )

    flops = 2 * total_m * N * K
    mok_a = time_calls(run_mok)
    deepgemm_samples = time_calls(run_deepgemm)
    mok_a2 = time_calls(run_mok)
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
        "schema": "bench-sm90-fp8-contiguous.v1",
        "shape": {
            "experts": EXPERTS,
            "m": total_m,
            "aligned_rows_per_expert": rows,
            "avg_aligned_m": total_m / EXPERTS,
            "n": N,
            "k": K,
        },
        "correctness": {
            "comparator": "DeepGEMM on identical quantized values/scales",
            "abs_max": abs_max,
            "rel_maxnorm": rel_maxnorm,
        },
        "timing": {
            "warmup": WARMUP,
            "iters": ITERS,
            "order": ["mok_a", "deepgemm", "mok_a2"],
            "boundary": "binding + grouped GEMM into preallocated output",
            "scale_layout_conversion": "excluded for both implementations",
        },
        "results": {
            "mok_a": summarize(mok_a, flops),
            "deepgemm": summarize(deepgemm_samples, flops),
            "mok_a2": summarize(mok_a2, flops),
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
        "BENCH|fp8_contiguous"
        f"|mok_a={record['results']['mok_a']['p50_ms']:.4f}ms"
        f"|deepgemm={record['results']['deepgemm']['p50_ms']:.4f}ms"
        f"|mok_a2={record['results']['mok_a2']['p50_ms']:.4f}ms"
        f"|rel={rel_maxnorm:.6f}"
    )


if __name__ == "__main__":
    main()
