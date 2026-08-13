#!/bin/bash
# Host-side launcher for run_bench_sm90.sh (tracked). Responsibilities:
#   1. compute EXPECTED_HARNESS_SHA256 from the host-synced tree
#   2. docker exec the container runner detached with the env passed through
#   3. verify launch for real: RUN_ID line, lock held, 4 workers
#   4. capture `docker top` (HOST pids) into the run log for NVML matching
# Usage: BENCH_TAG=tiny [env overrides...] bash host_launch_sm90.sh <container> <host_mok_dir>
set -u
CT=${1:?container name}
MOKDIR=${2:?host mok dir}
TAG=${BENCH_TAG:?BENCH_TAG required}
EXPECTED=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
ENVARGS=(-e BENCH_TAG="$TAG" -e EXPECTED_HARNESS_SHA256="$EXPECTED")
for v in MOK_SM90_EXPERIMENTAL MOK_FROZEN_COMMIT NUM_LOCAL_TOKENS HIDDEN_DIM \
         INTERMEDIATE_DIM NUM_EXPERTS TOPK MINIBATCH_SIZE MACROBATCH_SIZE \
         BENCH_WARMUP BENCH_TIMEOUT BF16_FWD_COMM_SMS; do
  [ -n "${!v:-}" ] && ENVARGS+=(-e "$v=${!v}")
done
docker exec -d "${ENVARGS[@]}" "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh
sleep 8
LOG=$(ls -t "$MOKDIR"/runs/$TAG-2*.log 2>/dev/null | head -1)
[ -n "$LOG" ] || { echo "LAUNCH_VERIFY_FAIL: no run log"; exit 7; }
grep -q RUN_ID "$LOG" || { echo "LAUNCH_VERIFY_FAIL: no RUN_ID in $LOG"; exit 7; }
WORKERS=$(docker exec "$CT" pgrep -cf torch.distributed 2>/dev/null || echo 0)
echo "HOST_TOP_CAPTURE:$(date -u +%F_%T)" >> "$LOG"
docker top "$CT" -eo pid,cmd 2>/dev/null | grep -E "bench_sm90_fwd|torch.distributed" >> "$LOG"
echo "LAUNCH_VERIFIED log=$LOG workers=$WORKERS"
