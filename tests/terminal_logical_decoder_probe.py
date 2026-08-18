#!/usr/bin/env python3
"""Build and run the bounded terminal logical-decoder probe."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cu")
HEADER = REPO_ROOT / "csrc" / "sm90_fp8_block_megakernel.cuh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("auto", "skip", "require"),
        default="auto",
        help="compile/run the CUDA helper when available",
    )
    return parser.parse_args()


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    return completed.stdout


def legacy_scaffold_matches(text: str) -> list[str]:
    forbidden = (
        r"tt<",
        r"tensor_allocator",
        r"mm2_[A-Za-z0-9_]*",
        r"mma2_[A-Za-z0-9_]*",
        r"full_tt_fp8e8m0",
        r"clc::",
        r"config::(?:CLUSTER_SIZE|MLP_Mb|NUM_THREADS)",
        r"increase_registers<256>",
    )
    return [pattern for pattern in forbidden if re.search(pattern, text)]


def check_source_contract() -> None:
    text = HEADER.read_text(encoding="utf-8")
    matches = legacy_scaffold_matches(text)
    if matches:
        raise RuntimeError(f"forbidden legacy scaffold in target header: {matches}")
    if "__global__" in text:
        raise RuntimeError("logical-decoder milestone must not define a kernel body")
    communication_required = (
        "COMM_ROWS_PER_CTA_TASK = 4",
        "communication_total_tickets",
        "decode_communication_cursor",
        "decode_communication_cta_task",
        "communication_stage::combine",
        "communication_stage::dispatch",
    )
    missing = [item for item in communication_required if item not in text]
    if missing:
        raise RuntimeError(
            f"native communication decoder contract missing: {missing}"
        )
    print(
        "TERMINAL_LOGICAL_DECODER_STATIC"
        "|legacy_scaffold=0|kernel_body=0|native_comm_decoder=1|result=PASS"
    )


def cuda_is_available() -> bool:
    if shutil.which("nvcc") is None:
        return False
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0 and any(
        line.strip().startswith("9.0") for line in probe.stdout.splitlines()
    )


def main() -> int:
    args = parse_args()
    check_source_contract()
    with tempfile.TemporaryDirectory(prefix="terminal-logical-decoder-") as tmp:
        tmp_path = Path(tmp)
        host_binary = tmp_path / "host-probe"
        preprocessed = tmp_path / "host-probe.ii"
        cxx = os.environ.get("CXX", "c++")
        run(
            [
                cxx,
                "-x",
                "c++",
                "-std=c++17",
                "-E",
                "-P",
                str(SOURCE),
                "-o",
                str(preprocessed),
            ]
        )
        preprocessed_matches = legacy_scaffold_matches(
            preprocessed.read_text(encoding="utf-8")
        )
        if preprocessed_matches:
            raise RuntimeError(
                "forbidden legacy scaffold in preprocessed probe: "
                f"{preprocessed_matches}"
            )
        print(
            "TERMINAL_LOGICAL_DECODER_PREPROCESSED"
            "|legacy_scaffold=0|result=PASS"
        )
        run(
            [
                cxx,
                "-x",
                "c++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(SOURCE),
                "-o",
                str(host_binary),
            ]
        )
        host_output = run([str(host_binary)])
        if "TERMINAL_LOGICAL_DECODER_PROBE|result=PASS" not in host_output:
            raise RuntimeError("host probe did not report PASS")

        have_cuda = cuda_is_available()
        if args.device == "require" and not have_cuda:
            raise RuntimeError("SM90 nvcc/device path requested but unavailable")
        if args.device != "skip" and have_cuda:
            device_binary = tmp_path / "device-probe"
            run(
                [
                    "nvcc",
                    "-std=c++17",
                    "-O2",
                    "-arch=sm_90a",
                    str(SOURCE),
                    "-o",
                    str(device_binary),
                ]
            )
            device_output = run([str(device_binary), "--device"])
            if "TERMINAL_LOGICAL_DECODER_DEVICE" not in device_output:
                raise RuntimeError("device probe did not exercise device helpers")
        else:
            print("TERMINAL_LOGICAL_DECODER_DEVICE|result=SKIP|reason=unavailable")

    print("TERMINAL_LOGICAL_DECODER_PY|result=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
