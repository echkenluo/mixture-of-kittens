#!/bin/bash
# Frozen-commit gate (tracked). Proves that a cell manifest's FROZEN_COMMIT is
# a real commit in this repository, is reachable from HEAD, and that the
# harness it names had EXACTLY the pinned bytes at that commit. Without this,
# "frozen against commit X" is an assertion in a commit message rather than a
# checkable fact - which is how 38bb4e1 shipped manifests pinned to a commit
# that did not contain the code they described.
#   any violation -> exit 16 FROZEN_GATE_FAIL:<why>
#   valid         -> exit 0, prints FROZEN_GATE_PASS:<commit>:<module>
# Usage: check_frozen_commit_sm90.sh <repo_dir> <manifest>
set -uo pipefail
REPO=${1:?repo dir}; MAN=${2:?manifest path}
[ -f "$MAN" ] || { echo "FROZEN_GATE_FAIL:missing manifest $MAN"; exit 16; }
mget() { grep "^$1=" "$MAN" | head -1 | cut -d= -f2- || true; }
FC=$(mget FROZEN_COMMIT)
HMOD=$(mget HARNESS_MODULE)
[ -n "$HMOD" ] || HMOD=benchmarks.bench_sm90_fwd
EXP=$(mget EXPECTED_HARNESS_SHA256)
echo "$FC" | grep -qE '^[0-9a-f]{40}$' || { echo "FROZEN_GATE_FAIL:FROZEN_COMMIT '$FC' not 40-hex"; exit 16; }
echo "$EXP" | grep -qE '^[0-9a-f]{64}$' || { echo "FROZEN_GATE_FAIL:EXPECTED_HARNESS_SHA256 not 64-hex"; exit 16; }
[ "$(git -C "$REPO" cat-file -t "$FC" 2>/dev/null)" = "commit" ] \
  || { echo "FROZEN_GATE_FAIL:FROZEN_COMMIT $FC is not a commit in this repo"; exit 16; }
git -C "$REPO" merge-base --is-ancestor "$FC" HEAD 2>/dev/null \
  || { echo "FROZEN_GATE_FAIL:FROZEN_COMMIT $FC is not an ancestor of HEAD"; exit 16; }
MODPATH=$(echo "$HMOD" | tr '.' '/').py
BLOB=$(git -C "$REPO" show "$FC:$MODPATH" 2>/dev/null | sha256sum | cut -d' ' -f1)
git -C "$REPO" cat-file -e "$FC:$MODPATH" 2>/dev/null \
  || { echo "FROZEN_GATE_FAIL:$MODPATH does not exist at $FC"; exit 16; }
[ "$BLOB" = "$EXP" ] \
  || { echo "FROZEN_GATE_FAIL:blob sha256 $BLOB at $FC != EXPECTED_HARNESS_SHA256 $EXP"; exit 16; }
echo "FROZEN_GATE_PASS:$FC:$HMOD"
