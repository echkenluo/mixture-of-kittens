#!/bin/bash
# Negative suite v5 (tracked; HOST side). Framework rules:
#   - before EVERY runner-level case: poll the build lock until FREE
#   - unique SUITE_ID/CASE_ID; each case examines only its own exact artifacts
#   - each case asserts exact rc + exact rejection reason + zero JSON
#     artifacts for its own ids (not rc alone)
#   - schema mutations go through the validate-only surface
#     (validate_manifest_sm90.sh); the production launcher has NO
#     untrusted-manifest bypass, so launcher-level mutation cases must be
#     rejected by the receipt binding gate (rc14)
#   - mutation fixtures live in TMPD, removed only AFTER the last case
# Usage: bash test_runner_negatives.sh <container> <host_mok_dir> <receipt>
set -uo pipefail
CT=${1:?}; MOKDIR=${2:?}; RECEIPT=${3:?}
DIR=$(cd "$(dirname "$0")" && pwd)
MANI=$DIR/manifests/tiny-h20-v1.manifest
VAL=$DIR/validate_manifest_sm90.sh
SUITE=$(date -u +%H%M%S)-$RANDOM
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "NEGATIVE_$1_PASS"; PASS=$((PASS+1)); else echo "NEGATIVE_$1_FAIL"; FAIL=$((FAIL+1)); fi }
wait_lock_free() {
  for i in $(seq 1 36); do
    docker exec "$CT" sh -c 'flock -n /mok/build.lock true' 2>/dev/null && return 0
    sleep 5
  done
  echo "SUITE_SETUP_FAIL:lock never freed"; exit 3
}
no_json() { # tag/case id -> 0 if no json artifact exists for it
  [ "$(ls "$MOKDIR/runs/$1-"*.json 2>/dev/null | wc -l)" -eq 0 ]
}
REALSHA=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
SOARR=("$MOKDIR"/mixture-of-kittens/mok/_C*.so)
{ [ "${#SOARR[@]}" -eq 1 ] && [ -f "${SOARR[0]}" ]; } || { echo "SUITE_SETUP_FAIL:need exactly one .so"; exit 3; }
REALSO=$(sha256sum "${SOARR[0]}" | cut -d' ' -f1)
REALMSHA=$(sha256sum "$MANI" | cut -d' ' -f1)
RSHA_REAL=$(sha256sum "$RECEIPT" | cut -d' ' -f1)
TMPD=$(mktemp -d)
runner_case() { # CID harness_sha so_sha extra_env... -> sets RC and L
  local CID=$1 SHAX=$2 SOX=$3; shift 3
  docker exec "$@" -e BENCH_TAG="$CID" -e RUN_ID="$CID" \
    -e EXPECTED_HARNESS_SHA256="$SHAX" -e EXPECTED_SO_SHA256="$SOX" \
    -e MANIFEST_SHA256="$REALMSHA" -e RECEIPT_SHA256="$RSHA_REAL" \
    "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh >/dev/null 2>&1
  RC=$?
  L=$MOKDIR/runs/$CID-$CID.log
}

# --- 1 wrong harness hash (runner gate): rc=8 + HASH_GATE_FAIL + no json
wait_lock_free
CID=n1-$SUITE; runner_case "$CID" deadbeef "$REALSO"
[ "$RC" -eq 8 ] && grep -q "HASH_GATE_FAIL" "$L" 2>/dev/null && no_json "$CID"; report 1_wrong_hash $?
echo "  n1 rc=$RC want=8(HASH_GATE_FAIL,no json)"

# --- 2 lock busy: holder confirmed held; rc=9 + LOCK_BUSY + no json; release confirmed
wait_lock_free
docker exec -d "$CT" bash -c "exec 9>/mok/build.lock; flock 9; sleep 20"
HELD=0
for i in $(seq 1 10); do
  docker exec "$CT" sh -c 'flock -n /mok/build.lock true' 2>/dev/null || { HELD=1; break; }
  sleep 1
done
[ "$HELD" -eq 1 ] || { echo "SUITE_SETUP_FAIL:n2 holder never took lock"; exit 3; }
CID=n2-$SUITE; runner_case "$CID" deadbeef "$REALSO"
[ "$RC" -eq 9 ] && grep -q "LOCK_BUSY" "$L" 2>/dev/null && no_json "$CID"; report 2_lock_busy $?
echo "  n2 rc=$RC want=9(LOCK_BUSY,no json)"
wait_lock_free

# --- 3 bad container: all host-side gates pass, image inspect fails -> rc14 exact
CID=n3-$SUITE
set +e
OUT=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" no-such-container "$MOKDIR" "$MANI" "$RECEIPT" 2>&1)
RC=$?
set -u
[ "$RC" -eq 14 ] && echo "$OUT" | grep -q "IMAGE_TRUST_FAIL:container inspect failed" && no_json "$CID"; report 3_bad_container $?
echo "  n3 rc=$RC want=14(IMAGE_TRUST_FAIL:container inspect failed)"

# --- 4 occupied target GPU (runner preflight): holder confirmed on NVML; rc=7 exact; cleanup asserted
wait_lock_free
docker exec -d "$CT" bash -c "python3 -c 'import torch,time;x=torch.ones(1024,1024,device=\"cuda:0\");time.sleep(45)'"
SEEN=0
for i in $(seq 1 15); do
  N=$(docker exec "$CT" sh -c "nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'" 2>/dev/null | head -1)
  [[ "${N:-0}" =~ ^[0-9]+$ ]] && [ "${N:-0}" -ge 1 ] && { SEEN=1; break; }
  sleep 3
done
[ "$SEEN" -eq 1 ] || { echo "SUITE_SETUP_FAIL:n4 holder never appeared on GPU"; exit 3; }
CID=n4-$SUITE; runner_case "$CID" "$REALSHA" "$REALSO" -e PREFLIGHT_TRIES=2
[ "$RC" -eq 7 ] && grep -q "PREFLIGHT_FAIL:target GPUs still occupied" "$L" 2>/dev/null && no_json "$CID"; report 4_occupied $?
echo "  n4 rc=$RC want=7(PREFLIGHT_FAIL occupied,no json)"
for i in $(seq 1 20); do
  N=$(docker exec "$CT" sh -c "nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'" 2>/dev/null | head -1)
  [[ "${N:-0}" =~ ^[0-9]+$ ]] && [ "${N:-0}" -eq 0 ] && break
  sleep 3
done
[[ "${N:-1}" =~ ^[0-9]+$ ]] && [ "${N:-1}" -eq 0 ] || { echo "SUITE_SETUP_FAIL:n4 GPU not cleared after holder (N=$N)"; exit 3; }

# --- 5 unwritable host-runs: launcher must reach the sidecar gate (all trust
# gates pass first: receipt is an absolute path, repo reached via symlink);
# rc=4 + exact reason + zero artifacts
CID=n5-$SUITE
RO=$MOKDIR/negro-$SUITE
mkdir -p "$RO/host-runs"
ln -s "$MOKDIR/mixture-of-kittens" "$RO/mixture-of-kittens"
chmod 555 "$RO/host-runs"
set +e
OUT=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" "$CT" "$RO" "$MANI" "$RECEIPT" 2>&1)
RC=$?
set -u
chmod 755 "$RO/host-runs" 2>/dev/null
NART=$(ls "$RO/runs/" 2>/dev/null | wc -l)
rm -rf "$RO" 2>/dev/null
[ "$RC" -eq 4 ] && echo "$OUT" | grep -q "sidecar not writable" && [ "$NART" -eq 0 ]; report 5_sidecar_unwritable $?
echo "  n5 rc=$RC want=4(sidecar not writable, artifacts=$NART)"

# --- 6 wrong SO (runner gate): rc=10 + SO_GATE_FAIL after hash+preflight pass + no json
wait_lock_free
CID=n6-$SUITE; runner_case "$CID" "$REALSHA" deadbeefso
[ "$RC" -eq 10 ] && grep -q "SO_GATE_FAIL" "$L" 2>/dev/null \
  && grep -q "HASH_GATE_PASS" "$L" && grep -q "PREFLIGHT_PASS" "$L" && no_json "$CID"; report 6_wrong_so $?
echo "  n6 rc=$RC want=10(SO_GATE_FAIL after hash+preflight pass,no json)"

# --- 7 missing manifest: rc=12 exact reason
CID=n7-$SUITE
set +e
OUT=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" "$CT" "$MOKDIR" "$TMPD/nonexistent.manifest" "$RECEIPT" 2>&1)
RC=$?
set -u
[ "$RC" -eq 12 ] && echo "$OUT" | grep -q "MANIFEST_SCHEMA_FAIL:missing" && no_json "$CID"; report 7_manifest_missing $?
echo "  n7 rc=$RC want=12(MANIFEST_SCHEMA_FAIL:missing)"

# --- 8 mutated manifest vs real receipt: production launcher must reject at
# the binding gate (rc14) - proves there is no schema-mutation backdoor
CID=n8-$SUITE
sed 's/^tokens_per_rank=.*/tokens_per_rank=513/' "$MANI" > "$TMPD/n8.manifest"
chmod 444 "$TMPD/n8.manifest"
set +e
OUT=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" "$CT" "$MOKDIR" "$TMPD/n8.manifest" "$RECEIPT" 2>&1)
RC=$?
set -u
[ "$RC" -eq 14 ] && echo "$OUT" | grep -q "MANIFEST_TRUST_FAIL:manifest sha != receipt" && no_json "$CID"; report 8_manifest_mutated $?
echo "  n8 rc=$RC want=14(MANIFEST_TRUST_FAIL:manifest sha != receipt)"

# --- 9 mutated receipt binding field: rc14 at binding gate
CID=n9-$SUITE
sed "s/^MANIFEST_SHA256=.*/MANIFEST_SHA256=$(printf 'b%.0s' $(seq 1 64))/" "$RECEIPT" > "$TMPD/n9.receipt"
chmod 444 "$TMPD/n9.receipt"
set +e
OUT=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" "$CT" "$MOKDIR" "$MANI" "$TMPD/n9.receipt" 2>&1)
RC=$?
set -u
[ "$RC" -eq 14 ] && echo "$OUT" | grep -q "MANIFEST_TRUST_FAIL:manifest sha != receipt" && no_json "$CID"; report 9_receipt_binding $?
echo "  n9 rc=$RC want=14(MANIFEST_TRUST_FAIL binding)"

# --- 10 writable receipt: rc14 write-bits gate
CID=n10-$SUITE
cp "$RECEIPT" "$TMPD/n10.receipt"
chmod 644 "$TMPD/n10.receipt"
set +e
OUT=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" "$CT" "$MOKDIR" "$MANI" "$TMPD/n10.receipt" 2>&1)
RC=$?
set -u
[ "$RC" -eq 14 ] && echo "$OUT" | grep -q "RECEIPT_TRUST_FAIL:write bits set" && no_json "$CID"; report 10_receipt_writable $?
echo "  n10 rc=$RC want=14(RECEIPT_TRUST_FAIL:write bits)"

# --- 11 image identity mismatch: receipt with wrong IMAGE_ID vs live container
CID=n11-$SUITE
sed "s|^IMAGE_ID=.*|IMAGE_ID=sha256:$(printf 'c%.0s' $(seq 1 64))|" "$RECEIPT" > "$TMPD/n11.receipt"
chmod 444 "$TMPD/n11.receipt"
set +e
OUT=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" "$CT" "$MOKDIR" "$MANI" "$TMPD/n11.receipt" 2>&1)
RC=$?
set -u
[ "$RC" -eq 14 ] && echo "$OUT" | grep -q "IMAGE_TRUST_FAIL:image id live" && no_json "$CID"; report 11_image_mismatch $?
echo "  n11 rc=$RC want=14(IMAGE_TRUST_FAIL:image id)"

# --- validate-only surface: schema/drift mutations (no launcher, no docker) ---
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
vcase 12_missing_key   'grep -v "^topk=" "$MANI"' 12 "key topk count=0"
vcase 13_unknown_key   'cat "$MANI"; echo "rogue_key=1"' 12 "unknown key rogue_key"
vcase 14_duplicate_key 'cat "$MANI"; echo "topk=1"' 12 "key topk count=2"
vcase 15_bad_hex       'sed "s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=nothex/" "$MANI"' 12 "EXPECTED_SO_SHA256 not 64-hex"
vcase 16_world_size    'sed "s/^world_size=.*/world_size=8/" "$MANI"' 12 "world_size must be 4"
vcase 17_gpus_dup      'sed "s/^BENCH_GPUS=.*/BENCH_GPUS=0,0,2,3/" "$MANI"' 12 "BENCH_GPUS ids not unique"
vcase 18_harness_drift "sed \"s/^EXPECTED_HARNESS_SHA256=.*/EXPECTED_HARNESS_SHA256=$(printf 'd%.0s' $(seq 1 64))/\" \"\$MANI\"" 13 "HARNESS_DRIFT_FAIL" \
  --harness "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py"
vcase 19_so_drift "sed \"s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=$(printf 'e%.0s' $(seq 1 64))/\" \"\$MANI\"" 13 "SO_DRIFT_FAIL" \
  --so-dir "$MOKDIR/mixture-of-kittens/mok"

# cleanup only AFTER the last case that uses TMPD fixtures
rm -rf "$TMPD"

EXPECTED_CASES=19
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED_CASES" ] || { echo "SUITE_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED_CASES"; FAIL=$((FAIL+1)); }
echo "NEGATIVES suite=$SUITE pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
