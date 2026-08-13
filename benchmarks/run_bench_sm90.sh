#!/bin/bash
# Container-side benchmark runner (tracked). Formal-ready guarantees:
#   - RUN_ID is provided by the host launcher: artifact paths are known a
#     priori, no ls-based discovery, stale artifacts cannot be selected
#   - flock build lock with holder recorded
#   - harness hash-equality gate (EXPECTED_HARNESS_SHA256) before anything
#   - GPU preflight scoped to BENCH_GPUS only, with explicit nvidia-smi RC
#     handling and a HARD FAIL when still occupied after the wait budget
#   - RUN_START/RUN_END/real exit recorded; CUDA_VISIBLE_DEVICES frozen and
#     target GPU UUIDs logged
set -uo pipefail
TAG=${BENCH_TAG:?BENCH_TAG required}
RUN_ID=${RUN_ID:?RUN_ID required (host launcher generates it)}
EXPECTED=${EXPECTED_HARNESS_SHA256:?EXPECTED_HARNESS_SHA256 required}
EXPSO=${EXPECTED_SO_HASH:?EXPECTED_SO_HASH required}
BENCH_GPUS=${BENCH_GPUS:-0,1,2,3}
PREFLIGHT_TRIES=${PREFLIGHT_TRIES:-24}
mkdir -p /mok/runs  # container-owned; host writes only to host-runs/
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
  echo "BENCH_GPUS:$BENCH_GPUS"
} > "$LOG"
UUIDS=$(nvidia-smi -i "$BENCH_GPUS" --query-gpu=index,uuid --format=csv,noheader 2>&1)
RC_UUID=$?
if [ "$RC_UUID" -ne 0 ]; then
  echo "PREFLIGHT_FAIL:nvidia-smi-uuid-query rc=$RC_UUID out=$UUIDS" >> "$LOG"; exit 7
fi
echo "TARGET_GPU_UUIDS:${UUIDS//$'\n'/;}" >> "$LOG"
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "HASH_GATE_FAIL expected=$EXPECTED actual=$ACTUAL" >> "$LOG"; exit 8
fi
echo "HASH_GATE_PASS" >> "$LOG"
CLEAR=0
for i in $(seq 1 "$PREFLIGHT_TRIES"); do
  APPS=$(nvidia-smi -i "$BENCH_GPUS" --query-compute-apps=pid --format=csv,noheader 2>&1)
  RC_APPS=$?
  if [ "$RC_APPS" -ne 0 ]; then
    echo "PREFLIGHT_FAIL:nvidia-smi-apps-query rc=$RC_APPS out=$APPS" >> "$LOG"; exit 7
  fi
  FOREIGN=$(printf '%s' "$APPS" | grep -c '[0-9]' || true)
  if [ "$FOREIGN" -eq 0 ]; then CLEAR=1; break; fi
  echo "PREFLIGHT_WAIT:$FOREIGN compute procs on target GPUs" >> "$LOG"
  sleep 5
done
if [ "$CLEAR" -ne 1 ]; then
  echo "PREFLIGHT_FAIL:target GPUs still occupied after $PREFLIGHT_TRIES tries" >> "$LOG"; exit 7
fi
echo "PREFLIGHT_PASS" >> "$LOG"
SOLIST=(mok/_C*.so)
if [ "${#SOLIST[@]}" -ne 1 ] || [ ! -f "${SOLIST[0]}" ]; then
  echo "SO_GATE_FAIL:need exactly one mok/_C*.so, found ${#SOLIST[@]}" >> "$LOG"; exit 10
fi
SOMD5=$(md5sum "${SOLIST[0]}" | cut -d' ' -f1)
echo "SO_PATH:${SOLIST[0]}" >> "$LOG"
echo "SO_MD5:$SOMD5" >> "$LOG"
if [ "$SOMD5" != "$EXPSO" ]; then
  echo "SO_GATE_FAIL expected=$EXPSO actual=$SOMD5" >> "$LOG"; exit 10
fi
echo "SO_GATE_PASS" >> "$LOG"
export CUDA_VISIBLE_DEVICES="$BENCH_GPUS"
echo "RUN_START:$(date -u +%F_%T)" >> "$LOG"
export BENCH_OUTPUT="$JSON"
timeout "${BENCH_TIMEOUT:-600}" python3 -m torch.distributed.run --standalone \
  --nproc-per-node=4 -m benchmarks.bench_sm90_fwd >> "$LOG" 2>&1
RC=$?
echo "RUN_REAL_EXIT:$RC" >> "$LOG"
echo "RUN_END:$(date -u +%F_%T)" >> "$LOG"
exit $RC
