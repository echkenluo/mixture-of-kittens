"""Stage-C destructive trap gates (host driver).

Each negative runs in its own process tree (a device trap poisons the CUDA
context): REENTRANT (single GPU, double lease acquire), CONTRACT (4-rank,
device-injected non-aligned num_tokens -- also the multi-CTA simultaneous
trap race, since every worker's thread 0 races the record CAS), TIMEOUT
(4-rank, shortened spin limit against a delayed ticket-0 producer).

PASS requires, per case: the child printed a fully committed eight-field
MOK_TRAP record with the expected code/site (two-phase publication makes a
partially written record impossible to observe as committed), and the
process tree exited through the fatal boundary (exit 70; torchrun surfaces
it as its own non-zero exit).  Any timeout, missing record, or wrong field
is a non-zero exit of this driver.
"""

import re
import subprocess
import sys

CASES = (
    ("reentrant", [sys.executable, "tests/_trap_case.py", "reentrant"],
     3, 7, 70),
    ("contract",
     [sys.executable, "-m", "torch.distributed.run", "--standalone",
      "--nproc-per-node=4", "tests/_trap_case.py", "contract"],
     2, 6, None),
    ("timeout",
     [sys.executable, "-m", "torch.distributed.run", "--standalone",
      "--nproc-per-node=4", "tests/_trap_case.py", "timeout"],
     1, 3, None),
)

PATTERN = re.compile(
    r"MOK_TRAP\|code=(-?\d+)\|site=(-?\d+)\|slot=(-?\d+)\|expected=(-?\d+)"
    r"\|observed=(-?\d+)\|rank=(-?\d+)\|ticket=(-?\d+)\|iters=(-?\d+)"
)


def main() -> int:
    failures = []
    for name, cmd, want_code, want_site, want_rc in CASES:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            print(f"TRAP_GATE|case={name}|verdict=TIMEOUT_HANG", flush=True)
            failures.append(name)
            continue
        out = proc.stdout + proc.stderr
        matches = PATTERN.findall(out)
        ok_record = any(
            int(m[0]) == want_code and int(m[1]) == want_site
            for m in matches
        )
        ok_rc = (
            proc.returncode == want_rc
            if want_rc is not None
            else proc.returncode != 0
        )
        bad_markers = [
            marker
            for marker in ("TRAP_GATE_NO_RECORD", "TRAP_GATE_NO_ERROR")
            if marker in out
        ]
        verdict = "PASS" if ok_record and ok_rc and not bad_markers else "FAIL"
        print(
            f"TRAP_GATE|case={name}|verdict={verdict}|rc={proc.returncode}"
            f"|records={len(matches)}|expected=code{want_code}/site{want_site}"
            f"|bad_markers={bad_markers}",
            flush=True,
        )
        if matches:
            print(f"TRAP_GATE|case={name}|first_record={matches[0]}",
                  flush=True)
        if verdict != "PASS":
            failures.append(name)
    if failures:
        print(f"SM90_TRAP_GATES_FAIL|{failures}", flush=True)
        return 1
    print("SM90_TRAP_GATES_PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
