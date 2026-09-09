"""Diagnose the frozen CPU-oracle failure; never emits a numeric PASS verdict.

Run as a module with an empty result directory. All five input variants and
three MoK paths are recorded without stopping at the first numeric difference.
Tensor snapshots retain sampled outputs for independent analysis.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mok import _C
from .test_warprole_independent_numeric import SAMPLES, inputs, oracle_rows
from .warprole_numeric_reference import metrics


def torch_gpu_reference(data):
    a, b, a_scale, b_scale, indices = data
    rows = []
    for expert in (1, 0):
        selected = [i for i in SAMPLES if int(indices[i]) == expert]
        x, weight = a.float()[selected].cuda(), b[expert].float().cuda()
        xs, ws = a_scale[selected].cuda(), b_scale[expert].cuda()
        result = None
        for group in range(a.shape[1] // 128):
            columns = slice(group * 128, (group + 1) * 128)
            partial = x[:, columns] @ weight[:, columns].T
            scale = xs[:, group, None] * ws[:, group].repeat_interleave(128)[None, :]
            scaled = partial * scale
            result = scaled if result is None else result + scaled
        rows.append(result.to(torch.bfloat16).cpu())
    return torch.cat(rows)


def main(destination, extended=False):
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    binary = Path(_C.__file__)
    receipt = {
        "extension_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "samples": SAMPLES, "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
        "cases": [], "verdict": "DIAGNOSTIC_ONLY_NO_NUMERIC_GO",
    }
    cases = ["original", "integer_values", "unit_scales", "single_k128", "single_k32"]
    if extended:
        import deep_gemm
        package = Path(deep_gemm.__file__).parent
        receipt["deepgemm"] = {
            "version": deep_gemm.__version__, "package": str(package),
            "python_sha256": hashlib.sha256((package / "__init__.py").read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256((package / "_C.so").read_bytes()).hexdigest(),
            "entry": "fp8_gemm_nt", "recipe": "runtime default, same FP8/FP32 scale tensors",
        }
        cases += ["normal_unit_scales", "normal_single_k32"]
    for case in cases:
        data = list(inputs(4096))
        if case.startswith("normal_"):
            # E4M3 smallest normal is 2**-6. Remove subnormal inputs in both
            # operands to distinguish input handling from accumulation error.
            for i in (0, 1):
                values = data[i].float()
                values[values.abs() < 2 ** -6] = 0
                data[i] = values.to(torch.float8_e4m3fn)
        if case == "integer_values":
            data[0] = data[0].float().round().to(torch.float8_e4m3fn)
            data[1] = data[1].float().round().to(torch.float8_e4m3fn)
        if case in ("unit_scales", "single_k128", "single_k32", "normal_unit_scales", "normal_single_k32"):
            data[2].fill_(1)
            data[3].fill_(1)
        if case in ("single_k128", "single_k32", "normal_single_k32"):
            cutoff = 128 if case == "single_k128" else 32
            x = data[0].float()
            x[:, cutoff:] = 0
            data[0] = x.to(torch.float8_e4m3fn)
        reference = oracle_rows(data)
        gpu_reference = torch_gpu_reference(data)
        a, b, a_scale, b_scale, indices = [x.cuda() for x in data]
        active = torch.tensor([128], dtype=torch.int32, device="cuda")
        case_receipt = {"name": case, "torch_fp32_vs_cpu": metrics(gpu_reference, reference), "arms": {}}
        snapshots = {"cpu_reference": reference, "torch_fp32_reference": gpu_reference}
        for arm, symbol in (
            ("split", "fp8_block_grouped_contiguous_dynamic_out"),
            ("c1s6", "fp8_block_warprole_gemm_c1s6_out"),
            ("c2s4", "fp8_block_warprole_gemm_c2s4_out"),
        ):
            out = torch.full((192, 4096), 12345.0, dtype=torch.bfloat16, device="cuda")
            getattr(_C, symbol)(a, b, a_scale, b_scale, indices, active, out)
            cpu = out.cpu()
            assert bool((cpu[128:] == 12345.0).all())
            actual = cpu[list(SAMPLES)]
            snapshots[arm] = actual
            case_receipt["arms"][arm] = metrics(actual, reference)
        if extended:
            pieces = []
            for start, expert in ((0, 1), (64, 0)):
                out = torch.empty((64, 4096), dtype=torch.bfloat16, device="cuda")
                deep_gemm.fp8_gemm_nt(
                    (a[start:start + 64], a_scale[start:start + 64]),
                    (b[expert], b_scale[expert]), out)
                selected = [i - start for i in SAMPLES if start <= i < start + 64]
                pieces.append(out.cpu()[selected])
            snapshots["deepgemm"] = torch.cat(pieces)
            case_receipt["arms"]["deepgemm"] = metrics(snapshots["deepgemm"], reference)
            case_receipt["deepgemm_vs_mok"] = metrics(snapshots["deepgemm"], snapshots["split"])
        case_receipt["split_vs_c1s6"] = metrics(snapshots["split"], snapshots["c1s6"])
        case_receipt["split_vs_c2s4"] = metrics(snapshots["split"], snapshots["c2s4"])
        tensor_path = destination / f"{case}.pt"
        torch.save(snapshots, tensor_path)
        case_receipt["tensors_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        receipt["cases"].append(case_receipt)
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("PRECISION_PROBE " + json.dumps(case_receipt), flush=True)
    receipt["complete"] = True
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    main(args.destination, args.extended)
