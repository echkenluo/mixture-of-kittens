#!/bin/bash
# Negative suite v3 (tracked; HOST side). Framework rules:
#   - before EVERY case: poll the build lock until FREE (timeout = setup fail)
#   - unique SUITE_ID/CASE_ID; each case examines only its own exact artifacts
#   - each case records the real RC and the expected rejection reason
# Usage: bash test_runner_negatives.sh <container> <host_mok_dir>
set -uo pipefail
CT=${1:?}; MOKDIR=${2:?}
DIR=$(cd "$(dirname "$0")" && pwd)
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
runner_case() { # CID expected_sha extra_env... -> sets RC and L
  local CID=$1 SHAX=$2; shift 2
  docker exec "$@" -e BENCH_TAG="$CID" -e RUN_ID="$CID" -e EXPECTED_HARNESS_SHA256="$SHAX" \
    "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh >/dev/null 2>&1
  RC=$?
  L=$MOKDIR/runs/$CID-$CID.log
}

# --- N1 wrong hash: want rc=8 + HASH_GATE_FAIL in this case's exact log
wait_lock_free
CID=n1-$SUITE; runner_case "$CID" deadbeef
[ "$RC" -eq 8 ] && grep -q HASH_GATE_FAIL "$L" 2>/dev/null; report 1_wrong_hash $?
echo "  n1 rc=$RC want=8(HASH_GATE_FAIL) log=$(basename "$L")"

# --- N2 lock busy: confirm holder HAS lock first; want rc=9 + LOCK_BUSY; confirm release
wait_lock_free
docker exec -d "$CT" bash -c "exec 9>/mok/build.lock; flock 9; sleep 20"
HELD=0
for i in $(seq 1 10); do
  docker exec "$CT" sh -c 'flock -n /mok/build.lock true' 2>/dev/null || { HELD=1; break; }
  sleep 1
done
[ "$HELD" -eq 1 ] || { echo "SUITE_SETUP_FAIL:n2 holder never took lock"; exit 3; }
CID=n2-$SUITE; runner_case "$CID" deadbeef
[ "$RC" -eq 9 ] && grep -q LOCK_BUSY "$L" 2>/dev/null; report 2_lock_busy $?
echo "  n2 rc=$RC want=9(LOCK_BUSY)"
wait_lock_free

# --- N3 no-start: launcher must return nonzero + LAUNCH_VERIFY_FAIL
CID=n3-$SUITE
set +e
OUT=$(BENCH_TAG="$CID" MOK_FROZEN_COMMIT=negtest bash "$DIR/host_launch_sm90.sh" no-such-container "$MOKDIR" 2>&1)
RC=$?
set -u
[ "$RC" -ne 0 ] && echo "$OUT" | grep -q LAUNCH_VERIFY_FAIL; report 3_no_start $?
echo "  n3 rc=$RC want=nonzero(LAUNCH_VERIFY_FAIL)"

# --- N4 occupied: confirm holder pid visible on target GPU NVML first; want rc=7; confirm cleared
wait_lock_free
docker exec -d "$CT" bash -c "python3 -c 'import torch,time;x=torch.ones(1024,1024,device=\"cuda:0\");time.sleep(45)'"
SEEN=0
for i in $(seq 1 15); do
  N=$(docker exec "$CT" sh -c "nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'" 2>/dev/null | head -1)
  [[ "${N:-0}" =~ ^[0-9]+$ ]] && [ "${N:-0}" -ge 1 ] && { SEEN=1; break; }
  sleep 3
done
[ "$SEEN" -eq 1 ] || { echo "SUITE_SETUP_FAIL:n4 holder never appeared on GPU"; exit 3; }
REALSHA=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
CID=n4-$SUITE; runner_case "$CID" "$REALSHA" -e PREFLIGHT_TRIES=2
[ "$RC" -eq 7 ] && grep -q PREFLIGHT_FAIL "$L" 2>/dev/null; report 4_occupied $?
echo "  n4 rc=$RC want=7(PREFLIGHT_FAIL)"
for i in $(seq 1 20); do
  N=$(docker exec "$CT" sh -c "nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'" 2>/dev/null | head -1)
  [[ "${N:-0}" =~ ^[0-9]+$ ]] && [ "${N:-0}" -eq 0 ] && break
  sleep 3
done
[[ "${N:-1}" =~ ^[0-9]+$ ]] && [ "${N:-1}" -eq 0 ] || { echo "SUITE_SETUP_FAIL:n4 GPU not cleared after holder (N=$N)"; exit 3; }

# --- N5 unwritable host-runs: rc=4 + exact message + NO runner artifacts created
CID=n5-$SUITE
RO=$MOKDIR/negro-$SUITE
mkdir -p "$RO/host-runs"; chmod 555 "$RO/host-runs"
set +e
OUT=$(BENCH_TAG="$CID" MOK_FROZEN_COMMIT=negtest bash "$DIR/host_launch_sm90.sh" "$CT" "$RO" 2>&1)
RC=$?
set -u
chmod 755 "$RO/host-runs" 2>/dev/null
NART=$(ls "$RO/runs/" 2>/dev/null | wc -l)
rm -rf "$RO" 2>/dev/null
[ "$RC" -eq 4 ] && echo "$OUT" | grep -q "sidecar not writable" && [ "$NART" -eq 0 ]; report 5_sidecar_unwritable $?
echo "  n5 rc=$RC want=4(sidecar not writable, no artifacts=$NART)"

echo "NEGATIVES suite=$SUITE pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
