#!/bin/bash
# Host-side launcher (tracked, v3). All host observations go to a HOST-OWNED
# sidecar ($LOG.host); the container-owned run log is never appended from the
# host (root-owned, appends would fail silently). Hard failures throughout.
#   - GPU identity: container UUIDs (from the runner log) are joined against
#     host `nvidia-smi index,uuid` -> auditable container_idx->uuid->host_idx
#     mapping; no hard-coded host GPU ids
#   - process shape: full-cmdline classification distinguishes the timeout
#     wrapper, the single torchrun parent, and exactly four rank workers
#   - PID attribution: each target GPU UUID must carry exactly one compute
#     pid, and that pid must be one of THIS run's worker host pids
# Usage: BENCH_TAG=... [env] bash host_launch_sm90.sh <container> <host_mok_dir>
set -uo pipefail
CT=${1:?container name}
MOKDIR=${2:?host mok dir}
TAG=${BENCH_TAG:?BENCH_TAG required}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM
LOG=$MOKDIR/runs/$TAG-$RUN_ID.log
SIDE=$MOKDIR/runs/$TAG-$RUN_ID.host
mkdir -p "$MOKDIR/runs" 2>/dev/null || true
touch "$SIDE" 2>/dev/null
if [ ! -w "$SIDE" ]; then
  echo "LAUNCH_VERIFY_FAIL:sidecar not writable at $SIDE"; exit 4
fi
echo "SIDECAR_START:$(date -u +%F_%T)" > "$SIDE"
echo "RUN_ID:$RUN_ID" >> "$SIDE"
side() { echo "$1" >> "$SIDE" || { echo "LAUNCH_VERIFY_FAIL:sidecar write failed"; exit 4; }; }
fail() { side "LAUNCH_VERIFY_FAIL:$1"; echo "LAUNCH_VERIFY_FAIL:$1"; exit "$2"; }

EXPECTED=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
ENVARGS=(-e BENCH_TAG="$TAG" -e RUN_ID="$RUN_ID" -e EXPECTED_HARNESS_SHA256="$EXPECTED")
for v in MOK_SM90_EXPERIMENTAL MOK_FROZEN_COMMIT NUM_LOCAL_TOKENS HIDDEN_DIM \
         INTERMEDIATE_DIM NUM_EXPERTS TOPK MINIBATCH_SIZE MACROBATCH_SIZE \
         BENCH_WARMUP BENCH_TIMEOUT BF16_FWD_COMM_SMS BENCH_GPUS PREFLIGHT_TRIES; do
  [ -n "${!v:-}" ] && ENVARGS+=(-e "$v=${!v}")
done
docker exec -d "${ENVARGS[@]}" "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh \
  || fail "docker exec failed" 7

for i in $(seq 1 12); do sleep 5; [ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" && break; done
[ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" || fail "no run log at exact path $LOG" 7
if grep -qE "HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY" "$LOG"; then
  fail "runner gate rejected: $(grep -E 'HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY' "$LOG" | head -1)" 8
fi

# --- GPU identity join (container UUIDs -> host indices) ---
CUUIDS=$(grep '^TARGET_GPU_UUIDS:' "$LOG" | head -1 | cut -d: -f2- | tr ';' '\n' | awk -F', ' '{print $1","$2}')
[ -n "$CUUIDS" ] || fail "no TARGET_GPU_UUIDS in runner log" 6
HOSTMAP=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
[ -n "$HOSTMAP" ] || fail "host nvidia-smi uuid query failed" 6
TUUIDS=""
side "GPU_MAPPING(container_idx,uuid,host_idx):"
while IFS=, read -r CIDX UUID; do
  [ -n "$UUID" ] || continue
  HIDX=$(printf '%s\n' "$HOSTMAP" | grep "$UUID" | cut -d, -f1)
  [ -n "$HIDX" ] || fail "container uuid $UUID not found on host" 6
  side "MAP:$CIDX,$UUID,$HIDX"
  TUUIDS="$TUUIDS $UUID"
done <<< "$CUUIDS"
NT=$(echo $TUUIDS | wc -w)
[ "$NT" -eq 4 ] || fail "expected 4 target GPUs, mapped $NT" 6

# --- process shape (full-cmdline classification) ---
WOK=0
for i in $(seq 1 12); do
  PSOUT=$(docker exec "$CT" ps -eo pid,args 2>/dev/null | grep "benchmarks.bench_sm90_fwd" | grep -v grep || true)
  NTIMEOUT=$(printf '%s\n' "$PSOUT" | awk '$2=="timeout"' | wc -l)
  NPARENT=$(printf '%s\n' "$PSOUT" | grep "torch.distributed.run" | awk '$2!="timeout"' | wc -l)
  NWORK=$(printf '%s\n' "$PSOUT" | grep -v "torch.distributed.run" | grep -v '^\s*$' | wc -l)
  if [ "$NPARENT" -eq 1 ] && [ "$NWORK" -eq 4 ]; then WOK=1; break; fi
  sleep 5
done
side "PROC_SHAPE:timeout=$NTIMEOUT parent=$NPARENT workers=$NWORK"
[ "$WOK" -eq 1 ] || fail "process shape wrong: parent=$NPARENT workers=$NWORK (want 1/4)" 6
docker exec "$CT" sh -c "flock -n /mok/build.lock true" 2>/dev/null && fail "build lock not held" 6
grep -q "RUN_START" "$LOG" || fail "no RUN_START in run log" 6

# --- PID attribution: exactly one of THIS run's workers per target GPU ---
TOPOUT=$(docker top "$CT" -eo pid,args 2>/dev/null | grep "benchmarks.bench_sm90_fwd" | grep -v "torch.distributed.run" | awk '$2!="timeout"' || true)
WPIDS=$(printf '%s\n' "$TOPOUT" | awk '{print $1}' | sort -n | uniq)
NW=$(echo $WPIDS | wc -w)
side "TOP_WORKER_HOST_PIDS:$(echo $WPIDS | tr ' ' ',') (n=$NW)"
[ "$NW" -eq 4 ] || fail "docker top worker count $NW != 4" 5
ATTR=FAIL
for i in $(seq 1 24); do
  PAIRS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  OK=1; SEEN=0
  for U in $TUUIDS; do
    P=$(printf '%s\n' "$PAIRS" | grep "^$U," | cut -d, -f2)
    NP=$(echo $P | wc -w)
    if [ "$NP" -ne 1 ]; then OK=0; break; fi
    echo "$WPIDS" | tr ' ' '\n' | grep -qx "$P" || { OK=0; break; }
    SEEN=$((SEEN+1))
  done
  if [ "$OK" -eq 1 ] && [ "$SEEN" -eq 4 ]; then ATTR=PASS; break; fi
  sleep 5
done
side "PID_ATTRIBUTION_$ATTR"
side "NVML_UUID_PID_PAIRS:$(printf '%s' "$PAIRS" | grep -f <(echo $TUUIDS | tr ' ' '\n') | tr '\n' ';')"
[ "$ATTR" = PASS ] || fail "per-GPU worker attribution failed" 5
echo "LAUNCH_VERIFIED run_id=$RUN_ID log=$LOG sidecar=$SIDE parent=1 workers=4 pid_attribution=PASS"
