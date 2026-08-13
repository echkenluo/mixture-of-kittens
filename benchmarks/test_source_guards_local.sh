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

# 7-9: three launcher properties that no local test can execute, because the
# hardened launcher pins PATH and the mock docker/nvidia-smi no longer resolve.
# 7: every manifest field must come from the verified per-run snapshot. Reading
# the caller's live path again is what let a flipped manifest supply BENCH_GPUS
# the anchor never covered while all hashes still matched.
check L4_manifest_fields_from_snapshot "$LAUNCH" \
  'mget\(\) \{ grep "\^\$1=" "\$MCOPY" ' \
  'mget\(\) \{ grep "\^\$1=" "\$MANIFEST" ' \
  'mget() { grep "^$1=" "$MCOPY" ' \
  'mget() { grep "^$1=" "$MANIFEST" '
# 8: same for the receipt
check L5_receipt_fields_from_snapshot "$LAUNCH" \
  'rget\(\) \{ grep "\^\$1=" "\$RCOPY" ' \
  'rget\(\) \{ grep "\^\$1=" "\$RECEIPT" ' \
  'rget() { grep "^$1=" "$RCOPY" ' \
  'rget() { grep "^$1=" "$RECEIPT" '
# 9: a failed docker image inspect must not be read as "local-only image"
check L6_inspect_failure_is_not_none "$LAUNCH" \
  '\[ "\$IRC" -eq 0 \] \|\| rfail "IMAGE_TRUST_FAIL:docker image inspect failed' \
  '-' \
  '[ "$IRC" -eq 0 ] || rfail "IMAGE_TRUST_FAIL:docker image inspect failed' \
  '[ "$IRC" -eq 99 ] || rfail "IMAGE_TRUST_FAIL:docker image inspect failed'

# 10-13: launcher properties this repo cannot drive dynamically. The per-run
# paths contain a run id generated inside the launcher, so nothing outside it
# can pre-create that exact path, and a test-only override in production code
# would be worse than the gap. These are SOURCE STRUCTURE GUARDS - they bind the
# exact creation expressions, not the runtime behaviour, and runtime
# verification is PENDING on a real host.
# 10-12: each of the three host evidence files is created by an O_EXCL
# redirection under noclobber, never by a copy that would follow or truncate an
# existing path
check L7_sidecar_created_o_excl "$LAUNCH" \
  '^if ! \{ : > "\$SIDE"; \} 2>/dev/null; then' \
  '^touch "\$SIDE"' \
  'if ! { : > "$SIDE"; } 2>/dev/null; then' \
  'touch "$SIDE"; if false; then'
check L9_receipt_snapshot_created_o_excl "$LAUNCH" \
  '^if ! \{ cat < "\$RECEIPT" > "\$RCOPY"; \} 2>/dev/null; then' \
  '^cp "\$RECEIPT" "\$RCOPY"' \
  'if ! { cat < "$RECEIPT" > "$RCOPY"; } 2>/dev/null; then' \
  'cp "$RECEIPT" "$RCOPY"; if false; then'
check L10_manifest_snapshot_created_o_excl "$LAUNCH" \
  '^if ! \{ cat < "\$MANIFEST" > "\$MCOPY"; \} 2>/dev/null; then' \
  '^cp "\$MANIFEST" "\$MCOPY"' \
  'if ! { cat < "$MANIFEST" > "$MCOPY"; } 2>/dev/null; then' \
  'cp "$MANIFEST" "$MCOPY"; if false; then'
# 13: the chmod gate is TWO lines - the command and the branch that exits 4.
# Binding only the first line would still pass if the branch became "|| true",
# so the check reads the following line and a mutation proves it. Chmod failure
# is not injectable locally (the process owns the files), so runtime remains
# PENDING on a real host.
lineno() { grep -nF "$2" "$1" | head -1 | cut -d: -f1; }
chmod_gate_ok() { # file
  local F=$1 L
  L=$(lineno "$F" 'chmod 444 "$MCOPY" "$RCOPY" \')
  [ -n "$L" ] || return 1
  sed -n "$((L+1))p" "$F" | grep -qF '|| { echo "LAUNCH_VERIFY_FAIL:cannot make the per-run snapshots read-only"; exit 4; }'
}
MUT=$(mktemp)
if ! chmod_gate_ok "$LAUNCH"; then
  report L11_snapshot_chmod_is_a_gate 1 "the real file does not have the two-line chmod gate"
else
  L=$(lineno "$LAUNCH" 'chmod 444 "$MCOPY" "$RCOPY" \')
  awk -v n=$((L+1)) 'NR==n {print "  || true"; next} {print}' "$LAUNCH" > "$MUT"
  if cmp -s "$LAUNCH" "$MUT"; then
    report L11_snapshot_chmod_is_a_gate 1 "mutation did not apply - the check would be vacuous"
  elif chmod_gate_ok "$MUT"; then
    report L11_snapshot_chmod_is_a_gate 1 "guard did not detect a cross-line || true"
  else
    report L11_snapshot_chmod_is_a_gate 0
  fi
fi
rm -f "$MUT"
# 14: the three creations must actually be ENCLOSED by noclobber. L7/L9/L10 bind
# the expressions; without this, deleting or moving `set -o noclobber` would
# leave them green while the redirections stopped being O_EXCL.
noclobber_encloses() { # file
  local F=$1 NC OFF A B C
  NC=$(lineno "$F" 'set -o noclobber')
  OFF=$(lineno "$F" 'set +o noclobber')
  A=$(lineno "$F" 'if ! { : > "$SIDE"; } 2>/dev/null; then')
  B=$(lineno "$F" 'if ! { cat < "$RECEIPT" > "$RCOPY"; } 2>/dev/null; then')
  C=$(lineno "$F" 'if ! { cat < "$MANIFEST" > "$MCOPY"; } 2>/dev/null; then')
  [ -n "$NC" ] && [ -n "$OFF" ] && [ -n "$A" ] && [ -n "$B" ] && [ -n "$C" ] \
    && [ "$NC" -lt "$A" ] && [ "$A" -lt "$B" ] && [ "$B" -lt "$C" ] && [ "$C" -lt "$OFF" ]
}
M1=$(mktemp); M2=$(mktemp)
grep -vF 'set -o noclobber' "$LAUNCH" > "$M1"
awk -v drop="set -o noclobber" '{ if ($0 ~ /^set -o noclobber$/) next; print }
  /^set \+o noclobber$/ { print "set -o noclobber" }' "$LAUNCH" > "$M2"
if ! noclobber_encloses "$LAUNCH"; then
  report L12_noclobber_encloses_the_creations 1 "the real file does not enclose the three creations"
elif cmp -s "$LAUNCH" "$M1" || cmp -s "$LAUNCH" "$M2"; then
  report L12_noclobber_encloses_the_creations 1 "mutation did not apply - the check would be vacuous"
elif noclobber_encloses "$M1" || noclobber_encloses "$M2"; then
  report L12_noclobber_encloses_the_creations 1 "guard survived deleting or moving the noclobber line"
else
  report L12_noclobber_encloses_the_creations 0
fi
rm -f "$M1" "$M2"
# 15-16: the resolved operational knobs must actually reach the runner, and the
# host completion wait must use the resolved value. Source structure only -
# driving these needs a real container; runtime verification PENDING.
check L13_knobs_reach_the_runner "$LAUNCH" \
  'ENVARGS\+=\(-e PREFLIGHT_TRIES="\$PREFLIGHT_TRIES" -e BENCH_TIMEOUT="\$BENCH_TIMEOUT"\)' \
  '\[ -n "\$\{PREFLIGHT_TRIES:-\}" \] && ENVARGS' \
  'ENVARGS+=(-e PREFLIGHT_TRIES="$PREFLIGHT_TRIES" -e BENCH_TIMEOUT="$BENCH_TIMEOUT")' \
  '[ -n "${PREFLIGHT_TRIES:-}" ] && ENVARGS+=(-e PREFLIGHT_TRIES="$PREFLIGHT_TRIES")'
check L14_completion_wait_uses_resolved_value "$LAUNCH" \
  '^WAIT=\$BENCH_WAIT_SECS$' \
  '^WAIT=\$\{BENCH_WAIT_SECS:-900\}$' \
  'WAIT=$BENCH_WAIT_SECS' \
  'WAIT=${BENCH_WAIT_SECS:-900}'
# 17: a gate that rejects after the sidecar is reserved must record why, and the
# reservation must never be replaced by a renamed temp file
check L8_rejections_are_recorded "$LAUNCH" \
  'rfail\(\) \{ side "LAUNCH_REJECTED:' \
  'mv -f "\$HTMP" "\$SIDE"' \
  'rfail() { side "LAUNCH_REJECTED:' \
  'rfail() { : "LAUNCH_REJECTED:'

EXPECTED=17
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "GUARD_COUNT_FAIL:ran $TOTAL guards, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "SOURCE_GUARDS pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
