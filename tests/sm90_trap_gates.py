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

import os
import re
import subprocess
import sys

# Children run in script mode, where sys.path[0] is tests/ rather than the
# repo root -- put the root on PYTHONPATH so `from mok import ...` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHILD_ENV = dict(
    os.environ,
    PYTHONPATH=_REPO_ROOT
    + os.pathsep
    + os.environ.get("PYTHONPATH", ""),
)

# (name, cmd, expected code, accepted sites, expected rc).  The timeout
# injection delays the ticket-0 producer, and WHICH spin exceeds the limit
# first is scheduling-dependent: the other copy workers wait on the
# input-publish scratch (site 1) or barrier flag (site 2) that ticket 0
# publishes, while GEMM workers wait on tile_ready (site 3) -- any K1
# timeout site is a valid outcome of this injection.
def _fields_reentrant(m):
    # slot/expected/ticket/iters are structurally 0 for a lease conflict;
    # observed = prior lease value (1); rank = the ep_rank the case passes.
    return (
        int(m[2]) == 0 and int(m[3]) == 0 and int(m[4]) == 1
        and int(m[5]) == 3 and int(m[6]) == 0 and int(m[7]) == 0
    )


def _fields_contract(m):
    # slot 0 (contract site), expected = capacity (2048), observed = the
    # injected num_tokens (65); pre-loop check so ticket and iters are 0.
    return (
        int(m[2]) == 0 and int(m[3]) == 2048 and int(m[4]) == 65
        and 0 <= int(m[5]) < 4 and int(m[6]) == 0 and int(m[7]) == 0
    )


def _fields_timeout(m):
    # Per-site constraints for the delayed-ticket-0 injection (capacity 512
    # -> 8 m-tiles, 8 copy tickets, copy_cta_idx < 16):
    #   site 1 (input scratch): waiting for the publish, expected 1 seen 0;
    #   site 2 (barrier flag): published but arrives short of expected;
    #   site 3 (tile_ready): a GEMM ticket (>= 8) short of 64 rows.
    if int(m[7]) != 200_000 or not (0 <= int(m[5]) < 4):
        return False
    site = int(m[1])
    slot, expected, observed, ticket = (
        int(m[2]), int(m[3]), int(m[4]), int(m[6])
    )
    if site == 1:
        return 0 <= slot < 16 and expected == 1 and observed == 0
    if site == 2:
        return 0 <= slot < 16 and expected > 0 and observed < expected
    if site == 3:
        return 0 <= slot < 8 and expected == 64 and observed < 64             and ticket >= 8
    return False


# (name, cmd, code, accepted sites, rc, repeats, per-field validator).  The
# contract case repeats 3x as the multi-CTA CAS-race stress: every worker's
# thread 0 races the two-phase claim on each run.
CASES = (
    ("reentrant", [sys.executable, "tests/_trap_case.py", "reentrant"],
     3, (7,), 70, 1, _fields_reentrant),
    ("contract",
     [sys.executable, "-m", "torch.distributed.run", "--standalone",
      "--nproc-per-node=4", "tests/_trap_case.py", "contract"],
     2, (6,), None, 3, _fields_contract),
    ("timeout",
     [sys.executable, "-m", "torch.distributed.run", "--standalone",
      "--nproc-per-node=4", "tests/_trap_case.py", "timeout"],
     1, (1, 2, 3), None, 1, _fields_timeout),
)

PATTERN = re.compile(
    r"MOK_TRAP\|code=(-?\d+)\|site=(-?\d+)\|slot=(-?\d+)\|expected=(-?\d+)"
    r"\|observed=(-?\d+)\|rank=(-?\d+)\|ticket=(-?\d+)\|iters=(-?\d+)"
)


def main() -> int:
    failures = []
    for name, cmd, want_code, want_sites, want_rc, repeats, checker in CASES:
      for attempt in range(repeats):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env=_CHILD_ENV,
            )
        except subprocess.TimeoutExpired:
            print(f"TRAP_GATE|case={name}|verdict=TIMEOUT_HANG", flush=True)
            failures.append(name)
            continue
        out = proc.stdout + proc.stderr
        matches = PATTERN.findall(out)
        ok_record = any(
            int(m[0]) == want_code and int(m[1]) in want_sites
            and checker(m)
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
            f"|records={len(matches)}"
            f"|expected=code{want_code}/sites{list(want_sites)}"
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
