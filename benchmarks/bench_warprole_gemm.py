#!/usr/bin/env python3
"""Step 1 of the warp-role megakernel plan: A/B/A on the GEMM primitive alone.

The reference is the production `fp8_block_grouped_contiguous_dynamic_out`,
which fills an SM with three concurrent CTAs.  The candidates are the
standalone warp-specialized entries, which put one CTA on an SM and hide the
load latency inside that CTA with a deeper ring buffer.  A candidate is
accepted when it is bitwise identical to the reference and its p50 stays within
1.05x of the reference p50 (the 95%-throughput line of the plan).

The input contract is the contiguous one: activations `[M, K]` with a per-row
K128 scale, weights `[E, N, K]` with N128/K128 block scales, and one expert id
per row in `m_indices`.  That is not the masked per-expert layout of
`bench_sm90_fp8_grouped.py`; the row generator here matches
`tests/test_warprole_gemm.py` so the timed inputs are the ones the bitwise
tests already cover.

Run on one H20, avoiding GPU 0 (its SM clock is pinned at 1830 MHz):

    CUDA_VISIBLE_DEVICES=1 python3 -m benchmarks.bench_warprole_gemm
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

# Imported defensively so that `--help` still works on a machine without the
# CUDA build; main() turns a missing import into a loud error before any work.
try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the machine
    torch = None
    TORCH_IMPORT_ERROR: Exception | None = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from mok import _C
except Exception as exc:  # pragma: no cover - depends on the build
    _C = None
    MOK_IMPORT_ERROR: Exception | None = exc
else:
    MOK_IMPORT_ERROR = None


REFERENCE_ENTRY = "fp8_block_grouped_contiguous_dynamic_out"
CANDIDATE_ENTRIES = {
    "c1s6": "fp8_block_warprole_gemm_c1s6_out",
    "c2s4": "fp8_block_warprole_gemm_c2s4_out",
}
EXPERTS = 64
ROW_PATTERN = [320, 336, 352, 368, 400, 416, 432, 448]   # mean 384 rows/expert, 24576 rows
SHAPES = {"w13": (4096, 4096), "w2": (4096, 2048)}       # (N, K)
SEED = 20260904
AA_DRIFT_LIMIT = 0.0025      # |p50(A1) - p50(A2)| / p50(A1)
RATIO_LIMIT = 1.05           # candidate p50 / reference p50
SENTINEL = 12345.0           # fills every output buffer before it is written


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A/B/A benchmark of the warp-role FP8 block grouped GEMM entries "
            "against the production contiguous GEMM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=f"/mok/runtime-logs/warprole-step1-{utc_stamp()}.json",
        help="JSON result path; the parent directory is created if needed",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="untimed calls before each timed block"
    )
    parser.add_argument(
        "--iters", type=int, default=30, help="timed calls per entry and shape"
    )
    parser.add_argument(
        "--shapes",
        default="w13,w2",
        help="comma-separated subset of " + ",".join(SHAPES),
    )
    parser.add_argument(
        "--candidates",
        default="c1s6,c2s4",
        help="comma-separated subset of " + ",".join(CANDIDATE_ENTRIES),
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device to time on")
    args = parser.parse_args()
    args.shapes = _subset(args.shapes, SHAPES, "--shapes")
    args.candidates = _subset(args.candidates, CANDIDATE_ENTRIES, "--candidates")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("--warmup must be >= 0 and --iters must be >= 1")
    return args


def _subset(raw: str, allowed: dict, flag: str) -> tuple[str, ...]:
    names = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = [name for name in names if name not in allowed]
    if not names or unknown:
        raise ValueError(f"{flag} must be a subset of {sorted(allowed)}, got {raw!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{flag} must not repeat a name, got {raw!r}")
    return names


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(device) -> dict:
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
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        clocks = smi.stdout.strip().splitlines() if smi.returncode == 0 else None
    except OSError:
        clocks = None
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else "unavailable",
        "source_state": status.stdout.splitlines(),
        "harness_sha256": sha256(pathlib.Path(__file__).resolve()),
        "so_path": str(shared_objects[0]) if len(shared_objects) == 1 else None,
        "so_sha256": sha256(shared_objects[0]) if len(shared_objects) == 1 else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_clocks_sm_mem": clocks,
        "timestamp_utc": utc_stamp(),
    }


def resolve_entry(name: str):
    entry = getattr(_C, name, None)
    if entry is None:
        raise RuntimeError(
            f"the binding {name!r} is missing from "
            f"{getattr(_C, '__file__', 'the mok extension')}. The warp-role GEMM "
            "entries exist only after csrc/sm90_fp8_block_warprole_gemm.cuh and "
            "the matching m.def lines in csrc/bindings.cu are compiled; rebuild "
            "with benchmarks/warprole/build_sm90.sh and reinstall the extension "
            "before running this harness."
        )
    return entry


def make_inputs(device, n: int, k: int, seed: int = SEED):
    """Same generator as tests/test_warprole_gemm.py: contiguous rows, 64 experts."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = ROW_PATTERN * (EXPERTS // len(ROW_PATTERN))
    total_m = sum(rows)
    assert total_m % 64 == 0
    k_blocks = k // 128
    a = torch.randn((total_m, k), generator=gen, device=device, dtype=torch.bfloat16)
    a = a.clamp(-3, 3).to(torch.float8_e4m3fn)
    b = torch.randn((EXPERTS, n, k), generator=gen, device=device, dtype=torch.bfloat16)
    b = b.clamp(-3, 3).to(torch.float8_e4m3fn)
    a_scale = torch.rand((total_m, k_blocks), generator=gen, device=device) * 0.09 + 0.01
    b_scale = torch.rand((EXPERTS, n // 128, k_blocks), generator=gen, device=device) * 0.09 + 0.01
    m_indices = torch.repeat_interleave(
        torch.arange(EXPERTS, dtype=torch.int32, device=device),
        torch.tensor(rows, dtype=torch.int64, device=device),
    ).contiguous()
    return a, b, a_scale, b_scale, m_indices, total_m


def time_calls(fn, warmup: int, iters: int) -> list[float]:
    """One CUDA-event pair per call, with a synchronize between calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


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


def run_shape(name: str, reference, candidates: dict, args) -> dict:
    device = torch.device(args.device)
    n, k = SHAPES[name]
    a, b, a_scale, b_scale, m_indices, total_m = make_inputs(device, n, k)
    num_tokens = torch.tensor([total_m], dtype=torch.int32, device=device)
    inputs = (a, b, a_scale, b_scale, m_indices, num_tokens)
    effective_flops = 2 * total_m * n * k

    ref_out = torch.full((total_m, n), SENTINEL, dtype=torch.bfloat16, device=device)
    reference(*inputs, ref_out)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(ref_out.float()).all()):
        raise RuntimeError(f"{name}: the reference GEMM produced non-finite values")

    a1 = summarize(
        time_calls(lambda: reference(*inputs, ref_out), args.warmup, args.iters),
        effective_flops,
    )

    rows = {}
    for cname in args.candidates:
        entry = candidates[cname]
        out = torch.full((total_m, n), SENTINEL, dtype=torch.bfloat16, device=device)
        entry(*inputs, out)
        torch.cuda.synchronize()
        bitwise = bool(torch.equal(out, ref_out))
        row = summarize(
            time_calls(lambda fn=entry, o=out: fn(*inputs, o), args.warmup, args.iters),
            effective_flops,
        )
        row["bitwise"] = bitwise
        row["ratio_vs_reference"] = row["p50_ms"] / a1["p50_ms"]
        row["pass"] = row["ratio_vs_reference"] <= RATIO_LIMIT
        rows[cname] = row
        del out

    a2 = summarize(
        time_calls(lambda: reference(*inputs, ref_out), args.warmup, args.iters),
        effective_flops,
    )
    drift = abs(a1["p50_ms"] - a2["p50_ms"]) / a1["p50_ms"]

    del a, b, a_scale, b_scale, m_indices, num_tokens, inputs, ref_out
    torch.cuda.empty_cache()
    return {
        "shape": name,
        "m": total_m,
        "n": n,
        "k": k,
        "experts": EXPERTS,
        "row_pattern": ROW_PATTERN,
        "effective_flops": effective_flops,
        "order": ["ref_a1", *args.candidates, "ref_a2"],
        "ref_a1": a1,
        "ref_a2": a2,
        "aa_drift": drift,
        "aa_drift_limit": AA_DRIFT_LIMIT,
        "candidates": rows,
        "valid": drift <= AA_DRIFT_LIMIT,
    }


def build_verdict(shapes: dict, candidate_names: tuple[str, ...]) -> dict:
    measured = sorted(shapes)
    ratio_pass = [
        name
        for name in candidate_names
        if all(shapes[shape]["candidates"][name]["pass"] for shape in measured)
    ]
    bitwise_equal = [
        name
        for name in candidate_names
        if all(shapes[shape]["candidates"][name]["bitwise"] for shape in measured)
    ]
    return {
        "all_shapes_valid": all(shapes[shape]["valid"] for shape in measured),
        "shapes_valid": {shape: shapes[shape]["valid"] for shape in measured},
        "gate": (
            f"p50 ratio <= {RATIO_LIMIT} against the reference A1 and bitwise "
            f"equality, in every measured shape; A/A drift <= {AA_DRIFT_LIMIT}"
        ),
        "candidates_passing_ratio": ratio_pass,
        "candidates_bitwise_equal": bitwise_equal,
        "candidates_passing": [name for name in ratio_pass if name in bitwise_equal],
    }


def print_table(shapes: dict, verdict: dict, args) -> None:
    header = (
        f"{'shape':<6} {'entry':<10} {'p50 ms':>9} {'p95 ms':>9} "
        f"{'TFLOP/s':>9} {'ratio':>7} {'bitwise':>8} {'pass':>5}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(shapes):
        result = shapes[name]
        for label in ("ref_a1", "ref_a2"):
            row = result[label]
            print(
                f"{name:<6} {label:<10} {row['p50_ms']:>9.4f} {row['p95_ms']:>9.4f} "
                f"{row['tflops_p50']:>9.1f} {'-':>7} {'-':>8} {'-':>5}"
            )
        for cname in args.candidates:
            row = result["candidates"][cname]
            print(
                f"{name:<6} {cname:<10} {row['p50_ms']:>9.4f} {row['p95_ms']:>9.4f} "
                f"{row['tflops_p50']:>9.1f} {row['ratio_vs_reference']:>7.3f} "
                f"{str(row['bitwise']):>8} {str(row['pass']):>5}"
            )
        print(
            f"{name:<6} A/A drift {result['aa_drift'] * 100:.3f}% "
            f"(limit {AA_DRIFT_LIMIT * 100:.2f}%), valid={result['valid']}"
        )
    print(
        "VERDICT|all_shapes_valid="
        f"{verdict['all_shapes_valid']}"
        f"|passing={','.join(verdict['candidates_passing']) or 'none'}"
        f"|ratio_only={','.join(verdict['candidates_passing_ratio']) or 'none'}"
        f"|bitwise={','.join(verdict['candidates_bitwise_equal']) or 'none'}"
        f"|out={args.output}"
    )


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError(f"torch is required to run this benchmark: {TORCH_IMPORT_ERROR}")
    if _C is None:
        raise RuntimeError(
            "the mok extension could not be imported, so no GEMM entry can be "
            f"resolved: {MOK_IMPORT_ERROR}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required; this benchmark times kernels")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(
            f"{args.device} reports capability "
            f"{torch.cuda.get_device_capability(device)}; the warp-role entries are SM90 only"
        )

    reference = resolve_entry(REFERENCE_ENTRY)
    candidates = {name: resolve_entry(CANDIDATE_ENTRIES[name]) for name in args.candidates}

    shapes = {name: run_shape(name, reference, candidates, args) for name in args.shapes}
    verdict = build_verdict(shapes, args.candidates)
    record = {
        "schema": "bench-warprole-gemm.v1",
        "config": {
            "warmup": args.warmup,
            "iters": args.iters,
            "shapes": list(args.shapes),
            "candidates": list(args.candidates),
            "reference_entry": REFERENCE_ENTRY,
            "candidate_entries": {name: CANDIDATE_ENTRIES[name] for name in args.candidates},
            "seed": SEED,
            "timing_boundary": "binding + kernel into a preallocated output",
        },
        "provenance": provenance(device),
        "shapes": shapes,
        "verdict": verdict,
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w") as sink:
        json.dump(record, sink, indent=1)
    os.replace(tmp, output)
    print_table(shapes, verdict, args)


if __name__ == "__main__":
    main()
