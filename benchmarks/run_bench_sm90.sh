#!/bin/bash
# Container-side benchmark runner for bench_sm90_fwd (tracked; runs INSIDE
# the benchmark container). Guarantees per run:
#   - unique RUN_ID artifact paths (no path reuse, stale artifacts impossible)
#   - build lock (flock) with holder recorded
#   - harness hash-equality gate: EXPECTED_HARNESS_SHA256 must match the file
#     actually on disk before anything launches
#   - GPU preflight: wait-loop until no foreign compute process holds the GPUs
#   - RUN_ID / RUN_START / RUN_END / real exit code in the log
# Host-side counterpart (host_launch_sm90.sh) verifies launch and captures
# `docker top` so NVML host PIDs can be matched against this run's workers.
set -u
TAG=${BENCH_TAG:?BENCH_TAG required}
EXPECTED=${EXPECTED_HARNESS_SHA256:?EXPECTED_HARNESS_SHA256 required}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p /mok/runs
LOG=/mok/runs/$TAG-$RUN_ID.log
JSON=/mok/runs/$TAG-$RUN_ID.json
exec 9>/mok/build.lock
flock -n 9 || { echo "LOCK_BUSY $(date -u +%T)" > "$LOG"; exit 9; }
cd /mok/mixture-of-kittens
ACTUAL=$(sha256sum benchmarks/bench_sm90_fwd.py | cut -d' ' -f1)
{
  echo "RUN_ID:$RUN_ID"
  echo "LOCK_HELD_BY:$$"
  echo "HARNESS_SHA256:$ACTUAL"
} > "$LOG"
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "HASH_GATE_FAIL expected=$EXPECTED actual=$ACTUAL" >> "$LOG"
  exit 8
fi
echo "HASH_GATE_PASS" >> "$LOG"
for i in $(seq 1 24); do
  FOREIGN=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$FOREIGN" -eq 0 ] && break
  echo "PREFLIGHT_WAIT:$FOREIGN compute procs" >> "$LOG"
  sleep 5
done
echo "RUN_START:$(date -u +%F_%T)" >> "$LOG"
export BENCH_OUTPUT="$JSON"
timeout ${BENCH_TIMEOUT:-600} python3 -m torch.distributed.run --standalone \
  --nproc-per-node=4 -m benchmarks.bench_sm90_fwd >> "$LOG" 2>&1
RC=$?
echo "RUN_REAL_EXIT:$RC" >> "$LOG"
echo "RUN_END:$(date -u +%F_%T)" >> "$LOG"
ln -sf "$LOG" "/mok/runs/$TAG-latest.log"   # convenience only, never evidence
ln -sf "$JSON" "/mok/runs/$TAG-latest.json" # convenience only, never evidence
exit $RC
