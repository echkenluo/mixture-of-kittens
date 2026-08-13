#!/bin/bash
# Host-side launcher v4 (tracked). The committed read-only manifest is the
# SOLE prior for expected hashes and run config - the launcher never derives
# an expected value at runtime. Flow:
#   1. manifest schema gate (version, required keys complete/unique, no
#      unknown keys)                                    -> exit 12 on violation
#   2. host artifact checks vs manifest (harness sha256, exactly-one .so
#      sha256)                                          -> exit 13 on drift
#   3. per-run manifest copy + its sha256 recorded in the host sidecar
#      BEFORE the runner starts (post-hoc tamper detection)
#   4. docker exec runner with expecteds + config FROM the manifest
#   5. launch verification: exact-path log, gates, 1 parent + 4 workers,
#      lock held, docker-top capture, per-GPU PID attribution verdict
# Usage: BENCH_TAG=... bash host_launch_sm90.sh <container> <host_mok_dir> <manifest>
set -uo pipefail
CT=${1:?container name}
MOKDIR=${2:?host mok dir}
MANIFEST=${3:?manifest path (committed, read-only)}
TAG=${BENCH_TAG:?BENCH_TAG required}

REQ_KEYS="MANIFEST_SCHEMA FROZEN_COMMIT EXPECTED_SO_SHA256 EXPECTED_HARNESS_SHA256 BENCH_GPUS TIMING_SEMANTICS tokens_per_rank hidden intermediate experts topk world_size comm_sms minibatch macrobatch warmup_iters timed_iters"
INT_KEYS="tokens_per_rank hidden intermediate experts topk world_size comm_sms minibatch macrobatch warmup_iters timed_iters"
[ -f "$MANIFEST" ] || { echo "MANIFEST_SCHEMA_FAIL:missing $MANIFEST"; exit 12; }
# trust gate: manifest must live in the repo's manifests dir, be git-tracked,
# and carry no write bits (unless NEG_ALLOW_UNTRUSTED=1 for controlled
# negative-suite mutations, which must never become positive priors)
if [ "${NEG_ALLOW_UNTRUSTED:-0}" != "1" ]; then
  MREAL=$(readlink -f "$MANIFEST")
  MDIR=$(readlink -f "$MOKDIR/mixture-of-kittens/benchmarks/manifests")
  case "$MREAL" in "$MDIR"/*) : ;; *) echo "MANIFEST_TRUST_FAIL:not in trusted manifests dir"; exit 14 ;; esac
  git -C "$MOKDIR/mixture-of-kittens" ls-files --error-unmatch "benchmarks/manifests/$(basename "$MREAL")" >/dev/null 2>&1     || { echo "MANIFEST_TRUST_FAIL:not git-tracked"; exit 14; }
  MODE=$(stat -c %a "$MREAL")
  case "$MODE" in *[2367]*) echo "MANIFEST_TRUST_FAIL:write bits set ($MODE)"; exit 14 ;; esac
fi
head -1 "$MANIFEST" | grep -q '^MANIFEST_SCHEMA=1$' || { echo "MANIFEST_SCHEMA_FAIL:bad or missing schema version"; exit 12; }
for K in $REQ_KEYS; do
  N=$(grep -c "^$K=" "$MANIFEST" || true)
  [ "$N" -eq 1 ] || { echo "MANIFEST_SCHEMA_FAIL:key $K count=$N (need exactly 1)"; exit 12; }
done
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  K=${LINE%%=*}
  echo " $REQ_KEYS " | grep -q " $K " || { echo "MANIFEST_SCHEMA_FAIL:unknown key $K"; exit 12; }
done < "$MANIFEST"
mget() { grep "^$1=" "$MANIFEST" | head -1 | cut -d= -f2-; }
for K in $REQ_KEYS; do
  V=$(mget "$K")
  [ -n "$V" ] || { echo "MANIFEST_SCHEMA_FAIL:key $K empty"; exit 12; }
done
for K in EXPECTED_SO_SHA256 EXPECTED_HARNESS_SHA256; do
  V=$(mget "$K")
  echo "$V" | grep -qE '^[0-9a-f]{64}$' || { echo "MANIFEST_SCHEMA_FAIL:$K not 64-hex"; exit 12; }
done
for K in $INT_KEYS; do
  V=$(mget "$K")
  echo "$V" | grep -qE '^[0-9]+$' && [ "$V" -gt 0 ] || { echo "MANIFEST_SCHEMA_FAIL:$K not positive int"; exit 12; }
done
[ "$(mget world_size)" = "4" ] || { echo "MANIFEST_SCHEMA_FAIL:world_size must be 4"; exit 12; }
NGID=$(mget BENCH_GPUS | tr ',' '\n' | grep -cE '^[0-9]+$' || true)
[ "$NGID" -eq 4 ] || { echo "MANIFEST_SCHEMA_FAIL:BENCH_GPUS needs exactly 4 ids"; exit 12; }
EXPECTED=$(mget EXPECTED_HARNESS_SHA256)
EXPSO=$(mget EXPECTED_SO_SHA256)
FROZEN=$(mget FROZEN_COMMIT)
BGPUS=$(mget BENCH_GPUS)

AH=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
[ "$AH" = "$EXPECTED" ] || { echo "HARNESS_DRIFT_FAIL expected=$EXPECTED actual=$AH"; exit 13; }
SOG=("$MOKDIR"/mixture-of-kittens/mok/_C*.so)
{ [ "${#SOG[@]}" -eq 1 ] && [ -f "${SOG[0]}" ]; } || { echo "SO_DRIFT_FAIL:need exactly one host .so, found ${#SOG[@]}"; exit 13; }
ASO=$(sha256sum "${SOG[0]}" | cut -d' ' -f1)
[ "$ASO" = "$EXPSO" ] || { echo "SO_DRIFT_FAIL expected=$EXPSO actual=$ASO"; exit 13; }

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM
LOG=$MOKDIR/runs/$TAG-$RUN_ID.log
SIDE=$MOKDIR/host-runs/$TAG-$RUN_ID.host
MCOPY=$MOKDIR/host-runs/$TAG-$RUN_ID.manifest
mkdir -p "$MOKDIR/host-runs" 2>/dev/null || true
touch "$SIDE" 2>/dev/null
[ -w "$SIDE" ] || { echo "LAUNCH_VERIFY_FAIL:sidecar not writable at $SIDE"; exit 4; }
side() { echo "$1" >> "$SIDE" || { echo "LAUNCH_VERIFY_FAIL:sidecar write failed"; exit 4; }; }
fail() { side "LAUNCH_VERIFY_FAIL:$1"; echo "LAUNCH_VERIFY_FAIL:$1"; exit "$2"; }
IMGID=$(docker inspect --format '{{.Image}}' "$CT" 2>/dev/null)
IMGRD=$(docker inspect --format '{{index .Config.Image}}' "$CT" 2>/dev/null)
[ -n "$IMGID" ] || { echo "LAUNCH_VERIFY_FAIL:image id inspect failed"; exit 4; }
cp "$MANIFEST" "$MCOPY" || { echo "LAUNCH_VERIFY_FAIL:manifest copy failed"; exit 4; }
MSHA=$(sha256sum "$MCOPY" | cut -d' ' -f1)
{ echo "SIDECAR_START:$(date -u +%F_%T)"; echo "RUN_ID:$RUN_ID"
  echo "MANIFEST_FILE:$(basename "$MANIFEST")"; echo "MANIFEST_SHA256:$MSHA"
  echo "IMAGE_ID:$IMGID"; echo "IMAGE_REPO_DIGESTS:$IMGRD"; } > "$SIDE"

ENVARGS=(-e BENCH_TAG="$TAG" -e RUN_ID="$RUN_ID"
         -e EXPECTED_HARNESS_SHA256="$EXPECTED" -e EXPECTED_SO_SHA256="$EXPSO"
         -e MOK_FROZEN_COMMIT="$FROZEN" -e MANIFEST_SHA256="$MSHA" -e MOK_SM90_EXPERIMENTAL=1
         -e BENCH_GPUS="$BGPUS"
         -e NUM_LOCAL_TOKENS="$(mget tokens_per_rank)" -e HIDDEN_DIM="$(mget hidden)"
         -e INTERMEDIATE_DIM="$(mget intermediate)" -e NUM_EXPERTS="$(mget experts)"
         -e TOPK="$(mget topk)" -e MINIBATCH_SIZE="$(mget minibatch)"
         -e MACROBATCH_SIZE="$(mget macrobatch)" -e BENCH_WARMUP="$(mget warmup_iters)"
         -e BF16_FWD_COMM_SMS="$(mget comm_sms)")
[ -n "${PREFLIGHT_TRIES:-}" ] && ENVARGS+=(-e PREFLIGHT_TRIES="$PREFLIGHT_TRIES")
[ -n "${BENCH_TIMEOUT:-}" ] && ENVARGS+=(-e BENCH_TIMEOUT="$BENCH_TIMEOUT")
docker exec -d "${ENVARGS[@]}" "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh \
  || fail "docker exec failed" 7

for i in $(seq 1 12); do sleep 5; [ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" && break; done
[ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" || fail "no run log at exact path $LOG" 7
if grep -qE "HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY|SO_GATE_FAIL" "$LOG"; then
  fail "runner gate rejected: $(grep -E 'HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY|SO_GATE_FAIL' "$LOG" | head -1)" 8
fi
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
side "HOST_LOADAVG_START:$(cat /proc/loadavg)"
side "HOST_VMSTAT_START:$(vmstat 1 2 2>/dev/null | tail -1 | tr -s ' ')"
side "HOST_GPU_CLOCKS_START:$(nvidia-smi --query-gpu=index,clocks.sm,power.draw --format=csv,noheader 2>/dev/null | tr '\n' ';')"
side "HOST_TOP_CAPTURE:$(date -u +%F_%T)"
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
side "HOST_LOADAVG_LAUNCHED:$(cat /proc/loadavg)"
side "HOST_GPU_CLOCKS_LAUNCHED:$(nvidia-smi --query-gpu=index,clocks.sm,power.draw --format=csv,noheader 2>/dev/null | tr '\n' ';')"
echo "LAUNCH_VERIFIED run_id=$RUN_ID log=$LOG sidecar=$SIDE manifest_sha=$MSHA pid_attribution=PASS"
