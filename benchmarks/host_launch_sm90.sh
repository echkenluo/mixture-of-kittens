#!/bin/bash
# Host-side launcher (tracked). Generates RUN_ID (exact artifact paths known
# a priori), execs the container runner, then ASSERTS with hard failures:
#   - runner log exists at the exact path with RUN_ID + lock holder
#   - exactly 1 torchrun parent and 4 rank workers for THIS module
#   - PID attribution: NVML compute pids on the HOST GPUs must be a subset
#     of this run's docker-top worker pids -> PID_ATTRIBUTION_PASS/FAIL in
#     the run log; FAIL aborts
# Usage: BENCH_TAG=... [env] bash host_launch_sm90.sh <container> <host_mok_dir> [host_gpu_ids]
set -uo pipefail
CT=${1:?container name}
MOKDIR=${2:?host mok dir}
HOST_GPUS=${3:-4,5,6,7}
TAG=${BENCH_TAG:?BENCH_TAG required}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM
LOG=$MOKDIR/runs/$TAG-$RUN_ID.log
EXPECTED=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
ENVARGS=(-e BENCH_TAG="$TAG" -e RUN_ID="$RUN_ID" -e EXPECTED_HARNESS_SHA256="$EXPECTED")
for v in MOK_SM90_EXPERIMENTAL MOK_FROZEN_COMMIT NUM_LOCAL_TOKENS HIDDEN_DIM \
         INTERMEDIATE_DIM NUM_EXPERTS TOPK MINIBATCH_SIZE MACROBATCH_SIZE \
         BENCH_WARMUP BENCH_TIMEOUT BF16_FWD_COMM_SMS BENCH_GPUS PREFLIGHT_TRIES; do
  [ -n "${!v:-}" ] && ENVARGS+=(-e "$v=${!v}")
done
docker exec -d "${ENVARGS[@]}" "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh

fail() { echo "LAUNCH_VERIFY_FAIL:$1"; [ -f "$LOG" ] && echo "LAUNCH_VERIFY_FAIL:$1" >> "$LOG"; exit "$2"; }
count_in_ct() { # robust numeric pgrep count
  local n
  n=$(docker exec "$CT" sh -c "pgrep -cf '$1' 2>/dev/null" 2>/dev/null | head -1)
  [[ "$n" =~ ^[0-9]+$ ]] || n=0
  echo "$n"
}

for i in $(seq 1 12); do sleep 5; [ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" && break; done
[ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" || fail "no run log at exact path $LOG" 7
if grep -qE "HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY" "$LOG"; then
  fail "runner gate rejected: $(grep -E 'HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY' "$LOG" | head -1)" 8
fi
WOK=0
for i in $(seq 1 12); do
  TOTAL=$(count_in_ct "benchmarks.bench_sm90_fwd")
  PARENT=$(count_in_ct "torch.distributed.run")
  WORKERS=$((TOTAL - PARENT))
  if [ "$PARENT" -eq 1 ] && [ "$WORKERS" -eq 4 ]; then WOK=1; break; fi
  sleep 5
done
[ "$WOK" -eq 1 ] || fail "process shape wrong: parent=$PARENT workers=$WORKERS (want 1/4)" 6
docker exec "$CT" sh -c "flock -n /mok/build.lock true" 2>/dev/null && fail "build lock not held" 6
grep -q "RUN_START" "$LOG" || fail "no RUN_START in this run's log" 6

echo "HOST_TOP_CAPTURE:$(date -u +%F_%T)" >> "$LOG"
docker top "$CT" -eo pid,cmd 2>/dev/null | grep -E "benchmarks.bench_sm90_fwd" >> "$LOG" || true
TOP_PIDS=$(docker top "$CT" -eo pid,cmd 2>/dev/null | grep "benchmarks.bench_sm90_fwd" | awk '{print $1}' | sort -n)
ATTR=FAIL
for i in $(seq 1 24); do
  NVML_PIDS=$(nvidia-smi -i "$HOST_GPUS" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -n | uniq)
  NN=$(printf '%s' "$NVML_PIDS" | grep -c '[0-9]' || true)
  if [ "$NN" -ge 4 ]; then
    BAD=0
    for p in $NVML_PIDS; do echo "$TOP_PIDS" | grep -qx "$p" || BAD=1; done
    [ "$BAD" -eq 0 ] && ATTR=PASS
    break
  fi
  sleep 5
done
{
  echo "PID_ATTRIBUTION_$ATTR"
  echo "TOP_WORKER_HOST_PIDS:$(echo $TOP_PIDS | tr ' ' ',')"
  echo "NVML_HOST_PIDS:$(echo $NVML_PIDS | tr ' ' ',')"
} >> "$LOG"
[ "$ATTR" = PASS ] || fail "NVML pids not attributable to this run's workers" 5
echo "LAUNCH_VERIFIED run_id=$RUN_ID log=$LOG parent=1 workers=4 pid_attribution=PASS"
