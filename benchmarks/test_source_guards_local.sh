#!/bin/bash
# Source guards (tracked, no GPU/docker/torch).
#
# The mock integration suite runs the real verifier, but nothing local can run
# host_launch_sm90.sh's process-identification branches: they need a live
# container. Those two greps are exactly where 38bb4e1's hardcoded MoK module
# would have broken every DeepEP cell, and no local execution can catch a
# regression there. So they are pinned as SOURCE assertions instead.
#
# Every guard is mutation-checked: the regression is reintroduced into a copy
# and the guard must reject it. The mutation is a literal string replacement
# and is asserted to have actually changed the file - a mutation that silently
# fails to apply would make the check vacuously green, which is how the first
# version of this file passed while its sed expressions were erroring out.
#
# Usage: bash test_source_guards_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "GUARD_$1_PASS"; PASS=$((PASS+1)); else echo "GUARD_$1_FAIL $3"; FAIL=$((FAIL+1)); fi }

# guard <file> <must-contain-ERE> <must-not-contain-ERE|-> ; 0 = satisfied
guard() {
  local F=$1 MUST=$2 MUSTNOT=$3
  grep -qE -- "$MUST" "$F" || return 1
  [ "$MUSTNOT" = "-" ] && return 0
  grep -qE -- "$MUSTNOT" "$F" && return 2
  return 0
}
mutate() { # file from to out ; nonzero if nothing changed
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
src, frm, to, out = sys.argv[1:5]
s = open(src).read()
if frm not in s:
    sys.exit(3)
n = s.replace(frm, to)
open(out, "w").write(n)
sys.exit(0 if n != s else 4)
PY
}
check() { # name file must mustnot mutate_from mutate_to
  local NAME=$1 F=$2 MUST=$3 MUSTNOT=$4 FRM=$5 TO=$6
  local RC MRC TMP MU
  guard "$F" "$MUST" "$MUSTNOT"; RC=$?
  if [ "$RC" -ne 0 ]; then
    report "$NAME" 1 "guard not satisfied by the real file (rc=$RC)"; return
  fi
  TMP=$(mktemp)
  mutate "$F" "$FRM" "$TO" "$TMP"; MU=$?
  if [ "$MU" -ne 0 ]; then
    rm -f "$TMP"
    report "$NAME" 1 "mutation did not apply (rc=$MU) - the check would be vacuous"; return
  fi
  guard "$TMP" "$MUST" "$MUSTNOT"; MRC=$?
  rm -f "$TMP"
  if [ "$MRC" -eq 0 ]; then
    report "$NAME" 1 "guard did not detect the reintroduced regression"
  else
    report "$NAME" 0
  fi
}

LAUNCH=$DIR/host_launch_sm90.sh
VERIFY=$DIR/verify_run_sm90.sh
RECEIPT=$DIR/make_deploy_receipt.sh
RUNNER=$DIR/run_bench_sm90.sh

# 1-2: both process-identification points must select by the manifest-derived
# module. A fixed module there misjudges process shape and PID attribution for
# the comparator - the launcher would see "0 workers" and fail a valid run.
check L1_ps_uses_manifest_module "$LAUNCH" \
  'ps -eo pid,args 2>/dev/null \| grep -F "\$HARNESS_MODULE"' \
  'ps -eo pid,args 2>/dev/null \| grep "benchmarks\.bench_sm90_fwd"' \
  'ps -eo pid,args 2>/dev/null | grep -F "$HARNESS_MODULE"' \
  'ps -eo pid,args 2>/dev/null | grep "benchmarks.bench_sm90_fwd"'
check L2_dockertop_uses_manifest_module "$LAUNCH" \
  'docker top "\$CT" -eo pid,args 2>/dev/null \| grep -F "\$HARNESS_MODULE"' \
  'docker top "\$CT" -eo pid,args 2>/dev/null \| grep "benchmarks\.bench_sm90_fwd"' \
  'docker top "$CT" -eo pid,args 2>/dev/null | grep -F "$HARNESS_MODULE"' \
  'docker top "$CT" -eo pid,args 2>/dev/null | grep "benchmarks.bench_sm90_fwd"'
# 3: the harness path must be built from the module (the only permitted mention
# of the MoK module in the launcher is the schema-1 default assignment)
check L3_harness_path_derived "$LAUNCH" \
  'HARNESS_FILE=\$MOKDIR/mixture-of-kittens/\$\(echo "\$HARNESS_MODULE" \| tr' \
  'HARNESS_FILE=\$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd\.py' \
  'HARNESS_FILE=$MOKDIR/mixture-of-kittens/$(echo "$HARNESS_MODULE" | tr '"'"'.'"'"' '"'"'/'"'"').py' \
  'HARNESS_FILE=$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py'
# 4: verifier drift check must use the IMPL-derived harness, never a literal
check V1_verify_drift_derived "$VERIFY" \
  '  --harness "\$HFILE" ' \
  '  --harness "\$M/mixture-of-kittens/benchmarks/bench_sm90_fwd\.py"' \
  '  --harness "$HFILE" \' \
  '  --harness "$M/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" \'
# 5: receipt generator must pin the harness the manifest selects
check R1_receipt_harness_derived "$RECEIPT" \
  'HSHA=\$\(sha256sum "\$HFILE" \| cut' \
  'HSHA=\$\(sha256sum benchmarks/bench_sm90_fwd\.py' \
  'HSHA=$(sha256sum "$HFILE" | cut' \
  'HSHA=$(sha256sum benchmarks/bench_sm90_fwd.py | cut'
# 6: runner must launch the module the manifest selects
check N1_runner_module_derived "$RUNNER" \
  '\-m "\$HARNESS_MODULE"' \
  '\-m benchmarks\.bench_sm90_fwd ' \
  '-m "$HARNESS_MODULE"' \
  '-m benchmarks.bench_sm90_fwd '

EXPECTED=6
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "GUARD_COUNT_FAIL:ran $TOTAL guards, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "SOURCE_GUARDS pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
