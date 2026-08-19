#!/usr/bin/env python3
"""Compile and execute the standard-C++ terminal TMA interval contract."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cpp")


def main() -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for the host contract probe")
    with tempfile.TemporaryDirectory(prefix="mok-terminal-tma-contract-") as tmp:
        output = Path(tmp) / "probe"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(ROOT / "csrc"),
                str(SOURCE),
                "-o",
                str(output),
            ],
            check=True,
        )
        subprocess.run([str(output)], check=True)
    print(
        "TERMINAL_TMA_CONTRACT_HOST"
        "|alignment=16|overlap=half_open|overflow=checked|result=PASS"
    )


if __name__ == "__main__":
    main()
