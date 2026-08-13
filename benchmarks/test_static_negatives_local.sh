#!/bin/bash
# Local deterministic static tests (tracked). Runs on the packaging host with
# NO docker/GPU: exercises every launcher/validator gate reachable before the
# container boundary, with exact rc + exact reason asserts. Builds a local
# fixture (dummy .so + fixture manifest + hand-built fixture receipt) so the
# launcher's host-side gates pass deterministically up to the gate under test.
# The remote suite (test_runner_negatives.sh) covers the docker/GPU cases.
# Usage: bash test_static_negatives_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/.." && pwd)
VAL=$DIR/validate_manifest_sm90.sh
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "STATIC_$1_PASS"; PASS=$((PASS+1)); else echo "STATIC_$1_FAIL"; FAIL=$((FAIL+1)); fi }

TMPD=$(mktemp -d)
MOKF=$TMPD/mok
mkdir -p "$MOKF/mixture-of-kittens/mok" "$MOKF/host-runs" "$MOKF/runs"
ln -s "$DIR" "$MOKF/mixture-of-kittens/benchmarks"
printf 'not a real so\n' > "$MOKF/mixture-of-kittens/mok/_Cfixture.so"
SOSHA=$(sha256sum "$MOKF/mixture-of-kittens/mok/_Cfixture.so" | cut -d' ' -f1)
HSHA=$(sha256sum "$DIR/bench_sm90_fwd.py" | cut -d' ' -f1)
MANF=$TMPD/fixture.manifest
sed -e "s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=$SOSHA/" \
    -e "s/^EXPECTED_HARNESS_SHA256=.*/EXPECTED_HARNESS_SHA256=$HSHA/" \
    "$DIR/manifests/tiny-h20-v1.manifest" > "$MANF"
chmod 444 "$MANF"
MSHA=$(sha256sum "$MANF" | cut -d' ' -f1)
RECF=$TMPD/fixture.receipt
FAKE40=$(printf '0%.0s' $(seq 1 40))
{
  echo "RECEIPT_SCHEMA=1"
  echo "SOURCE_TREE_COMMIT=$FAKE40"
  echo "HARNESS_COMMIT=$FAKE40"
  echo "BINARY_BUILD_COMMIT=$FAKE40"
  echo "MANIFEST_FILE=fixture.manifest"
  echo "MANIFEST_SHA256=$MSHA"
  echo "MANIFEST_GIT_BLOB=$FAKE40"
  echo "HARNESS_SHA256=$HSHA"
  echo "SO_SHA256=$SOSHA"
  echo "IMAGE_ID=sha256:$(printf '1%.0s' $(seq 1 64))"
  echo "IMAGE_REF=fixture-image:latest"
  echo "IMAGE_REPO_DIGESTS=NONE"
} > "$RECF"
chmod 444 "$RECF"

lcase() { # name launcher-args... then WANT_RC WANT_REASON checked via globals
  local NAME=$1 WANT=$2 REASON=$3; shift 3
  set +e
  local OUTL
  OUTL=$(BENCH_TAG="$NAME" bash "$DIR/host_launch_sm90.sh" "$@" 2>&1)
  local RCX=$?
  set -u
  [ "$RCX" -eq "$WANT" ] && echo "$OUTL" | grep -q "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$RCX want=$WANT($REASON)"
}

# validator positive on the fixture (exact-rc-0 sanity of the fixture itself)
set +e
OUTP=$(bash "$VAL" "$MANF" --harness "$DIR/bench_sm90_fwd.py" --so-dir "$MOKF/mixture-of-kittens/mok" 2>&1)
RCP=$?
set -u
[ "$RCP" -eq 0 ] && echo "$OUTP" | grep -q "MANIFEST_VALID:$MSHA"; report 0_validator_positive $?
echo "  positive rc=$RCP want=0(MANIFEST_VALID)"

# launcher gates, in gate order
cp "$RECF" "$TMPD/writable.receipt"; chmod 644 "$TMPD/writable.receipt"
lcase L1_receipt_writable 14 "RECEIPT_TRUST_FAIL:write bits set" no-ct "$MOKF" "$MANF" "$TMPD/writable.receipt"
grep -v '^SO_SHA256=' "$RECF" > "$TMPD/missingkey.receipt"; chmod 444 "$TMPD/missingkey.receipt"
lcase L2_receipt_missing_key 14 "RECEIPT_TRUST_FAIL:key SO_SHA256 count=0" no-ct "$MOKF" "$MANF" "$TMPD/missingkey.receipt"
{ cat "$RECF"; echo "rogue=1"; } > "$TMPD/rogue.receipt"; chmod 444 "$TMPD/rogue.receipt"
lcase L3_receipt_unknown_key 14 "RECEIPT_TRUST_FAIL:unknown key rogue" no-ct "$MOKF" "$MANF" "$TMPD/rogue.receipt"
lcase L4_manifest_missing 12 "MANIFEST_SCHEMA_FAIL:missing" no-ct "$MOKF" "$TMPD/nonexistent.manifest" "$RECF"
sed 's/^tokens_per_rank=.*/tokens_per_rank=513/' "$MANF" > "$TMPD/mut.manifest"; chmod 444 "$TMPD/mut.manifest"
lcase L5_manifest_mutated 14 "MANIFEST_TRUST_FAIL:manifest sha != receipt" no-ct "$MOKF" "$TMPD/mut.manifest" "$RECF"
sed "s/^MANIFEST_SHA256=.*/MANIFEST_SHA256=$(printf 'b%.0s' $(seq 1 64))/" "$RECF" > "$TMPD/badbind.receipt"; chmod 444 "$TMPD/badbind.receipt"
lcase L6_receipt_binding 14 "MANIFEST_TRUST_FAIL:manifest sha != receipt" no-ct "$MOKF" "$MANF" "$TMPD/badbind.receipt"
sed "s/^HARNESS_SHA256=.*/HARNESS_SHA256=$(printf 'f%.0s' $(seq 1 64))/" "$RECF" > "$TMPD/hmis.receipt"; chmod 444 "$TMPD/hmis.receipt"
lcase L7_receipt_harness_mismatch 14 "RECEIPT_TRUST_FAIL:harness sha receipt != manifest" no-ct "$MOKF" "$MANF" "$TMPD/hmis.receipt"
# bad container after all host gates pass: image inspect must fail exactly
lcase L8_bad_container 14 "IMAGE_TRUST_FAIL:container inspect failed" no-such-container-xyz "$MOKF" "$MANF" "$RECF"

# validate-only surface mutations
vcase() { # name mutator want_rc want_reason [validator extra args...]
  local NAME=$1 MUT=$2 WANT=$3 REASON=$4; shift 4
  local MF=$TMPD/$NAME.manifest
  eval "$MUT" > "$MF"
  set +e
  local OUTV
  OUTV=$(bash "$VAL" "$MF" "$@" 2>&1)
  local RCX=$?
  set -u
  [ "$RCX" -eq "$WANT" ] && echo "$OUTV" | grep -q "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$RCX want=$WANT($REASON)"
}
vcase V1_missing_key   'grep -v "^topk=" "$MANF"' 12 "key topk count=0"
vcase V2_unknown_key   'cat "$MANF"; echo "rogue_key=1"' 12 "unknown key rogue_key"
vcase V3_duplicate_key 'cat "$MANF"; echo "topk=1"' 12 "key topk count=2"
vcase V4_bad_hex       'sed "s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=nothex/" "$MANF"' 12 "EXPECTED_SO_SHA256 not 64-hex"
vcase V5_bad_int       'sed "s/^warmup_iters=.*/warmup_iters=0/" "$MANF"' 12 "warmup_iters not positive int"
vcase V6_world_size    'sed "s/^world_size=.*/world_size=8/" "$MANF"' 12 "world_size must be 4"
vcase V7_gpus_dup      'sed "s/^BENCH_GPUS=.*/BENCH_GPUS=0,0,2,3/" "$MANF"' 12 "BENCH_GPUS ids not unique"
vcase V8_gpus_nonnum   'sed "s/^BENCH_GPUS=.*/BENCH_GPUS=0,1,2,x/" "$MANF"' 12 "BENCH_GPUS non-numeric id"
vcase V9_harness_drift "sed \"s/^EXPECTED_HARNESS_SHA256=.*/EXPECTED_HARNESS_SHA256=$(printf 'd%.0s' $(seq 1 64))/\" \"\$MANF\"" 13 "HARNESS_DRIFT_FAIL" \
  --harness "$DIR/bench_sm90_fwd.py"
vcase V10_so_drift "sed \"s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=$(printf 'e%.0s' $(seq 1 64))/\" \"\$MANF\"" 13 "SO_DRIFT_FAIL" \
  --so-dir "$MOKF/mixture-of-kittens/mok"
vcase V11_two_so 'cat "$MANF"' 13 "need exactly one _C\*.so" --so-dir "$TMPD/twoso"
mkdir -p "$TMPD/twoso"; touch "$TMPD/twoso/_Ca.so" "$TMPD/twoso/_Cb.so"
# rerun V11 after fixture exists (the first invocation above ran before the
# dir was populated and also correctly failed; assert the populated form too)
vcase V11b_two_so 'cat "$MANF"' 13 "need exactly one _C\*.so" --so-dir "$TMPD/twoso"

rm -rf "$TMPD"
EXPECTED=21
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "STATIC_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "STATIC_NEGATIVES pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
