#!/bin/bash
# Negative tests for the benchmark chain (tracked; run on the HOST).
# N1 wrong hash -> HASH_GATE_FAIL, runner exit 8, launcher rejects
# N2 lock busy  -> LOCK_BUSY, runner exit 9, launcher rejects
# N3 no-start   -> launcher LAUNCH_VERIFY_FAIL on missing exact-path log
# N4 occupied   -> PREFLIGHT_FAIL after bounded tries, launcher rejects
# Usage: bash test_runner_negatives.sh <container> <host_mok_dir>
set -uo pipefail
CT=${1:?}; MOKDIR=${2:?}
DIR=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "NEGATIVE_$1_PASS"; PASS=$((PASS+1)); else echo "NEGATIVE_$1_FAIL"; FAIL=$((FAIL+1)); fi }

# N1: wrong hash
OUT=$(BENCH_TAG=negh RUN_ID=neg1-$RANDOM EXPECTED_HARNESS_SHA256=deadbeef \
  bash -c "docker exec -e BENCH_TAG=negh -e RUN_ID=neg1 -e EXPECTED_HARNESS_SHA256=deadbeef $CT bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh; echo RC:\$?")
echo "$OUT" | grep -q "RC:8" && grep -q HASH_GATE_FAIL "$MOKDIR/runs/negh-neg1.log"; report 1_wrong_hash $?

# N2: lock busy (hold the lock, then try to run)
docker exec -d "$CT" bash -c "exec 9>/mok/build.lock; flock 9; sleep 25"
sleep 2
OUT=$(docker exec -e BENCH_TAG=negl -e RUN_ID=neg2 -e EXPECTED_HARNESS_SHA256=deadbeef "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh; echo "RC:$?")
echo "$OUT" | grep -q "RC:9" && grep -q LOCK_BUSY "$MOKDIR/runs/negl-neg2.log"; report 2_lock_busy $?
sleep 25  # let the holder expire

# N3: launcher must fail fast when runner never starts (bad container name)
set +e
OUT=$(BENCH_TAG=negn bash "$DIR/host_launch_sm90.sh" no-such-container "$MOKDIR" 2>&1)
RC3=$?
set -e 2>/dev/null || true
[ "$RC3" -ne 0 ] && echo "$OUT" | grep -q "LAUNCH_VERIFY_FAIL"; report 3_no_start $?

# N4: occupied target GPU -> preflight hard fail (bounded tries)
docker exec -d "$CT" bash -c "python3 -c 'import torch,time;x=torch.ones(1024,1024,device=\"cuda:0\");time.sleep(40)'"
sleep 8
OUT=$(docker exec -e BENCH_TAG=nego -e RUN_ID=neg4 -e EXPECTED_HARNESS_SHA256=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1) -e PREFLIGHT_TRIES=2 "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh; echo "RC:$?")
echo "$OUT" | grep -q "RC:7" && grep -q "PREFLIGHT_FAIL" "$MOKDIR/runs/nego-neg4.log"; report 4_occupied $?
sleep 35  # holder expiry

# N5: unwritable sidecar dir -> launcher hard fail exit 4
RO=$MOKDIR/negro; mkdir -p "$RO/runs" 2>/dev/null; chmod 555 "$RO/runs" 2>/dev/null
set +e
BENCH_TAG=negs bash "$DIR/host_launch_sm90.sh" "$CT" "$RO" >/dev/null 2>&1
RC5=$?
set -e 2>/dev/null || true
chmod 755 "$RO/runs" 2>/dev/null
[ "$RC5" -eq 4 ]; report 5_sidecar_unwritable $?

echo "NEGATIVES: pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
