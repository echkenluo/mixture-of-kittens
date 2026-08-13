#!/bin/bash
# Host-side launcher v6 (tracked). Trust model: the deployment receipt is
# generated on the trusted packaging host (make_deploy_receipt.sh) and its
# hash travels out-of-band as EXPECTED_RECEIPT_SHA256. The FIRST gate is the
# exact comparison of the receipt bytes against that external prior - a
# read-only bit or schema validity establishes nothing by itself, and a
# self-consistent rewrite of manifest+receipt still fails here. The verified
# end never self-signs and performs no git lookups. Gate order:
#   0. EXPECTED_RECEIPT_SHA256 env present + 64-hex                -> exit 14
#   1. sha256(receipt) == EXPECTED_RECEIPT_SHA256 (exact)          -> exit 14
#   2. receipt well-formed + read-only (shared validate_receipt)   -> exit 14
#   3. BENCH_MODE gate: formal is UNCONDITIONALLY refused (no build-record
#      contract exists yet); canary runs are labeled INVALID_FOR_FORMAL
#      in sidecar, log and JSON                                    -> exit 14
#   4. manifest present -> exit 12; content-bound to receipt and
#      read-only                                                   -> exit 14
#   5. shared manifest validation + harness/SO drift               -> exit 12/13
#   6. receipt<->manifest expected-hash agreement                  -> exit 14
#   7. live image id/ref/RepoDigests == receipt                    -> exit 14
#   8. sidecar (atomic header) + per-run manifest AND receipt
#      copies BEFORE the runner                                    -> exit 4
#   9. prelaunch telemetry BEFORE docker exec: mapping, occupancy
#      wait, clocks/load/vmstat snapshot                           -> exit 6/15
#  10. docker exec runner; launch verification (exact log path,
#      runner gates, uuid set equality, 1 parent + 4 workers, lock
#      held, docker-top, per-GPU PID attribution, running clock
#      floor)                                                      -> exit 5-8/15
#  11. completion wait with PERIODIC foreign-process sampling on
#      the target GPUs; RUN_END + exactly one RUN_REAL_EXIT:0 +
#      parseable JSON                                              -> exit 15
#  12. end telemetry gates: zero foreign occupancy, zero midrun
#      foreign hits, load1 delta; sidecar finalized                -> exit 15
# Gated quantities: target-GPU occupancy (prelaunch / periodic midrun / end),
# absolute prelaunch load1, load1 delta, and a single-sample running SM clock
# LIVENESS floor (not a stability gate - see below). Power draw and vmstat are
# RECORD-ONLY disclosures and are never gates.
# Usage: BENCH_TAG=... EXPECTED_RECEIPT_SHA256=... [BENCH_MODE=formal|canary] \
#          bash host_launch_sm90.sh <container> <host_mok_dir> <manifest> <receipt>
set -uo pipefail
CT=${1:?container name}
MOKDIR=${2:?host mok dir}
MANIFEST=${3:?manifest path (committed, read-only)}
RECEIPT=${4:?deployment receipt path (packaging-generated, read-only)}
TAG=${BENCH_TAG:?BENCH_TAG required}
DIR=$(cd "$(dirname "$0")" && pwd)

EXPR_SHA=${EXPECTED_RECEIPT_SHA256:-}
echo "$EXPR_SHA" | grep -qE '^[0-9a-f]{64}$' \
  || { echo "RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 env missing or not 64-hex"; exit 14; }
[ -f "$RECEIPT" ] || { echo "RECEIPT_TRUST_FAIL:missing receipt $RECEIPT"; exit 14; }
RSHA=$(sha256sum "$RECEIPT" | cut -d' ' -f1)
[ "$RSHA" = "$EXPR_SHA" ] || { echo "RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 mismatch (actual $RSHA expected $EXPR_SHA)"; exit 14; }
bash "$DIR/validate_receipt_sm90.sh" "$RECEIPT" --check-mode || exit 14
rget() { grep "^$1=" "$RECEIPT" | head -1 | cut -d= -f2-; }

BMODE=${BENCH_MODE:-canary}
case "$BMODE" in formal|canary) : ;; *) echo "MODE_FAIL:BENCH_MODE must be formal or canary (got $BMODE)"; exit 14 ;; esac
# Formal mode is UNCONDITIONALLY refused. A 40-hex BINARY_BUILD_COMMIT that
# resolves in the repo proves only that some commit exists - it does not prove
# that commit produced this .so, so any commit could stand in for the real
# one. Until a build-record contract exists (record bytes + hash binding the
# SO hash, source commit and toolchain/image, verified end to end), formal
# provenance cannot be established and must not be simulated.
if [ "$BMODE" = "formal" ]; then
  echo "FORMAL_MODE_FAIL:build-record contract not implemented"; exit 14
fi
FORMAL_VALIDITY=INVALID_FOR_FORMAL

[ -f "$MANIFEST" ] || { echo "MANIFEST_SCHEMA_FAIL:missing $MANIFEST"; exit 12; }
MSHA_ACT=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
[ "$MSHA_ACT" = "$(rget MANIFEST_SHA256)" ] || { echo "MANIFEST_TRUST_FAIL:manifest sha != receipt (actual $MSHA_ACT receipt $(rget MANIFEST_SHA256))"; exit 14; }
MMODE=$(stat -c %a "$MANIFEST")
case "$MMODE" in *[2367]*) echo "MANIFEST_TRUST_FAIL:write bits set ($MMODE)"; exit 14 ;; esac

mget() { grep "^$1=" "$MANIFEST" | head -1 | cut -d= -f2-; }
# schema 2 selects the implementation; schema 1 is MoK-only by definition
HARNESS_MODULE=$(mget HARNESS_MODULE)
[ -n "$HARNESS_MODULE" ] || HARNESS_MODULE=benchmarks.bench_sm90_fwd
HARNESS_FILE=$MOKDIR/mixture-of-kittens/$(echo "$HARNESS_MODULE" | tr '.' '/').py
[ -f "$HARNESS_FILE" ] || { echo "MANIFEST_SCHEMA_FAIL:harness file for $HARNESS_MODULE missing"; exit 12; }
bash "$DIR/validate_manifest_sm90.sh" "$MANIFEST" \
  --harness "$HARNESS_FILE" \
  --so-dir "$MOKDIR/mixture-of-kittens/mok"
VRC=$?
[ "$VRC" -eq 0 ] || exit "$VRC"
EXPECTED=$(mget EXPECTED_HARNESS_SHA256); EXPSO=$(mget EXPECTED_SO_SHA256)
FROZEN=$(mget FROZEN_COMMIT); BGPUS=$(mget BENCH_GPUS)
[ "$(rget HARNESS_SHA256)" = "$EXPECTED" ] || { echo "RECEIPT_TRUST_FAIL:harness sha receipt != manifest"; exit 14; }
[ "$(rget SO_SHA256)" = "$EXPSO" ] || { echo "RECEIPT_TRUST_FAIL:so sha receipt != manifest"; exit 14; }

IMGID=$(docker inspect --format '{{.Image}}' "$CT" 2>/dev/null)
[ -n "$IMGID" ] || { echo "IMAGE_TRUST_FAIL:container inspect failed for $CT"; exit 14; }
IMGREF=$(docker inspect --format '{{.Config.Image}}' "$CT" 2>/dev/null)
[ -n "$IMGREF" ] || { echo "IMAGE_TRUST_FAIL:image ref empty"; exit 14; }
IMGRD=$(docker image inspect --format '{{join .RepoDigests ","}}' "$IMGID" 2>/dev/null)
[ -n "$IMGRD" ] || IMGRD=NONE
[ "$IMGID" = "$(rget IMAGE_ID)" ] || { echo "IMAGE_TRUST_FAIL:image id live $IMGID != receipt $(rget IMAGE_ID)"; exit 14; }
[ "$IMGREF" = "$(rget IMAGE_REF)" ] || { echo "IMAGE_TRUST_FAIL:image ref live $IMGREF != receipt $(rget IMAGE_REF)"; exit 14; }
[ "$IMGRD" = "$(rget IMAGE_REPO_DIGESTS)" ] || { echo "IMAGE_TRUST_FAIL:repo digests live $IMGRD != receipt $(rget IMAGE_REPO_DIGESTS)"; exit 14; }

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM
LOG=$MOKDIR/runs/$TAG-$RUN_ID.log
JSONF=$MOKDIR/runs/$TAG-$RUN_ID.json
SIDE=$MOKDIR/host-runs/$TAG-$RUN_ID.host
MCOPY=$MOKDIR/host-runs/$TAG-$RUN_ID.manifest
RCOPY=$MOKDIR/host-runs/$TAG-$RUN_ID.receipt
mkdir -p "$MOKDIR/host-runs" 2>/dev/null || true
touch "$SIDE" 2>/dev/null
[ -w "$SIDE" ] || { echo "LAUNCH_VERIFY_FAIL:sidecar not writable at $SIDE"; exit 4; }
side() { echo "$1" >> "$SIDE" || { echo "LAUNCH_VERIFY_FAIL:sidecar write failed"; exit 4; }; }
fail() { side "LAUNCH_VERIFY_FAIL:$1"; echo "LAUNCH_VERIFY_FAIL:$1"; exit "$2"; }
cp "$MANIFEST" "$MCOPY" || { echo "LAUNCH_VERIFY_FAIL:manifest copy failed"; exit 4; }
cp "$RECEIPT" "$RCOPY" || { echo "LAUNCH_VERIFY_FAIL:receipt copy failed"; exit 4; }
MSHA=$(sha256sum "$MCOPY" | cut -d' ' -f1)
HTMP=$SIDE.hdr.$$
if ! { echo "SIDECAR_START:$(date -u +%F_%T)"; echo "RUN_ID:$RUN_ID"
       echo "BENCH_MODE:$BMODE"; echo "FORMAL_VALIDITY:$FORMAL_VALIDITY"
       echo "MANIFEST_FILE:$(basename "$MANIFEST")"; echo "MANIFEST_SHA256:$MSHA"
       echo "RECEIPT_FILE:$(basename "$RECEIPT")"; echo "RECEIPT_SHA256:$RSHA"
       echo "RECEIPT_COPY:$(basename "$RCOPY")"
       echo "IMAGE_ID:$IMGID"; echo "IMAGE_REF:$IMGREF"; echo "IMAGE_REPO_DIGESTS:$IMGRD"; } > "$HTMP"; then
  echo "LAUNCH_VERIFY_FAIL:sidecar header write failed"; exit 4
fi
mv -f "$HTMP" "$SIDE" || { echo "LAUNCH_VERIFY_FAIL:sidecar header rename failed"; exit 4; }

# prelaunch telemetry: container->host GPU mapping for BENCH_GPUS only,
# occupancy wait, clocks/power/load snapshot - all BEFORE docker exec.
# power.draw and vmstat lines are record-only disclosures, not gates.
CMAP=$(docker exec "$CT" nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
[ -n "$CMAP" ] || fail "container gpu uuid query failed" 6
HOSTMAP=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
[ -n "$HOSTMAP" ] || fail "host nvidia-smi uuid query failed" 6
TUUIDS=""
side "GPU_MAPPING(container_idx,uuid,host_idx):"
for CIDX in ${BGPUS//,/ }; do
  UUID=$(printf '%s\n' "$CMAP" | grep "^$CIDX," | cut -d, -f2)
  [ -n "$UUID" ] || fail "container gpu index $CIDX has no uuid" 6
  HIDX=$(printf '%s\n' "$HOSTMAP" | grep "$UUID" | cut -d, -f1)
  [ -n "$HIDX" ] || fail "container uuid $UUID not found on host" 6
  side "MAP:$CIDX,$UUID,$HIDX"
  TUUIDS="$TUUIDS $UUID"
done
NT=$(echo $TUUIDS | wc -w)
NTU=$(echo $TUUIDS | tr ' ' '\n' | sort -u | grep -c . || true)
{ [ "$NT" -eq 4 ] && [ "$NTU" -eq 4 ]; } || fail "expected 4 unique target GPUs, mapped $NT unique $NTU" 6
gpuq() { nvidia-smi --query-gpu=uuid,clocks.sm,power.draw --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -f <(echo $TUUIDS | tr ' ' '\n') | tr '\n' ';'; }
occq() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -cf <(echo $TUUIDS | tr ' ' '\n') || true; }
side "TELEMETRY_PRELAUNCH_BEGIN:$(date -u +%F_%T)"
OCC=-1
for i in $(seq 1 12); do OCC=$(occq); [ "$OCC" -eq 0 ] && break; sleep 5; done
side "PRELAUNCH_OCCUPANCY:$OCC"
[ "$OCC" -eq 0 ] || fail "TELEMETRY_GATE_FAIL:prelaunch foreign occupancy=$OCC on target GPUs" 15
LOAD_START=$(cat /proc/loadavg)
side "HOST_LOADAVG_PRELAUNCH:$LOAD_START"
L1P=${LOAD_START%% *}
LOAD1_MAX_PRELAUNCH=${LOAD1_MAX_PRELAUNCH:-64}
PLOK=$(python3 -c "print(1 if float('$L1P') <= float('$LOAD1_MAX_PRELAUNCH') else 0)" 2>/dev/null)
side "PRELAUNCH_LOAD1_GATE:load1=$L1P max=$LOAD1_MAX_PRELAUNCH ok=${PLOK:-0}"
[ "${PLOK:-0}" = "1" ] || fail "TELEMETRY_GATE_FAIL:prelaunch load1 $L1P above $LOAD1_MAX_PRELAUNCH" 15
side "HOST_VMSTAT_PRELAUNCH:$(vmstat 1 2 2>/dev/null | tail -1 | tr -s ' ')"
side "HOST_GPU_CLOCKS_PRELAUNCH:$(gpuq)"
side "TELEMETRY_PRELAUNCH_END:$(date -u +%F_%T)"

ENVARGS=(-e BENCH_TAG="$TAG" -e RUN_ID="$RUN_ID"
         -e EXPECTED_HARNESS_SHA256="$EXPECTED" -e EXPECTED_SO_SHA256="$EXPSO"
         -e MOK_FROZEN_COMMIT="$FROZEN" -e MANIFEST_SHA256="$MSHA"
         -e RECEIPT_SHA256="$RSHA" -e BENCH_MODE="$BMODE" -e MOK_SM90_EXPERIMENTAL=1
         -e BENCH_GPUS="$BGPUS"
         -e NUM_LOCAL_TOKENS="$(mget tokens_per_rank)" -e HIDDEN_DIM="$(mget hidden)"
         -e INTERMEDIATE_DIM="$(mget intermediate)" -e NUM_EXPERTS="$(mget experts)"
         -e TOPK="$(mget topk)" -e MINIBATCH_SIZE="$(mget minibatch)"
         -e MACROBATCH_SIZE="$(mget macrobatch)" -e BENCH_WARMUP="$(mget warmup_iters)"
         -e BF16_FWD_COMM_SMS="$(mget comm_sms)" -e HARNESS_MODULE="$HARNESS_MODULE")
# comparator stack pins: passed only when the manifest declares them, so the
# comparator's own gate fails closed rather than defaulting to something
for K in TORCH_VERSION_PIN DEEPEP_PY_TREE_SHA256 DEEPEP_EXT_SHA256 DEEPEP_TORCH_COMPILE; do
  V=$(mget "$K"); [ -n "$V" ] && ENVARGS+=(-e "$K=$V")
done
[ -n "${PREFLIGHT_TRIES:-}" ] && ENVARGS+=(-e PREFLIGHT_TRIES="$PREFLIGHT_TRIES")
[ -n "${BENCH_TIMEOUT:-}" ] && ENVARGS+=(-e BENCH_TIMEOUT="$BENCH_TIMEOUT")
docker exec -d "${ENVARGS[@]}" "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh \
  || fail "docker exec failed" 7

for i in $(seq 1 12); do sleep 5; [ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" && break; done
[ -f "$LOG" ] && grep -q "RUN_ID:$RUN_ID" "$LOG" || fail "no run log at exact path $LOG" 7
if grep -qE "HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY|SO_GATE_FAIL" "$LOG"; then
  fail "runner gate rejected: $(grep -E 'HASH_GATE_FAIL|PREFLIGHT_FAIL|LOCK_BUSY|SO_GATE_FAIL' "$LOG" | head -1)" 8
fi
RUUIDS=$(grep '^TARGET_GPU_UUIDS:' "$LOG" | head -1 | grep -o 'GPU-[0-9a-f-]*' | sort)
[ -n "$RUUIDS" ] || fail "no TARGET_GPU_UUIDS in runner log" 6
[ "$RUUIDS" = "$(echo $TUUIDS | tr ' ' '\n' | sort)" ] || fail "runner uuid set != prelaunch mapping" 6
WOK=0
NTIMEOUT=0; NPARENT=0; NWORK=0
for i in $(seq 1 12); do
  PSOUT=$(docker exec "$CT" ps -eo pid,args 2>/dev/null | grep -F "$HARNESS_MODULE" | grep -v grep || true)
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
side "HOST_TOP_CAPTURE:$(date -u +%F_%T)"
TOPOUT=$(docker top "$CT" -eo pid,args 2>/dev/null | grep -F "$HARNESS_MODULE" | grep -v "torch.distributed.run" | awk '$2!="timeout"' || true)
WPIDS=$(printf '%s\n' "$TOPOUT" | awk '{print $1}' | sort -n | uniq)
NW=$(echo $WPIDS | wc -w)
side "TOP_WORKER_HOST_PIDS:$(echo $WPIDS | tr ' ' ',') (n=$NW)"
[ "$NW" -eq 4 ] || fail "docker top worker count $NW != 4" 5
ATTR=FAIL
PAIRS=""
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
# single-sample liveness floor taken shortly after launch: it may land in
# setup/JIT rather than the timed region, so it is NOT an environment-
# stability gate and must never be described as one
RUNCLK=$(gpuq)
side "HOST_GPU_CLOCKS_RUNNING:$RUNCLK"
CLK_RUN_MIN_MHZ=${CLK_RUN_MIN_MHZ:-500}
LOWCLK=0
for C in $(printf '%s' "$RUNCLK" | tr ';' '\n' | cut -d, -f2 | grep -oE '^[0-9]+'); do
  [ "$C" -lt "$CLK_RUN_MIN_MHZ" ] && LOWCLK=1
done
side "RUNNING_CLOCK_LIVENESS:min=${CLK_RUN_MIN_MHZ}MHz low=$LOWCLK"
[ "$LOWCLK" -eq 0 ] || fail "TELEMETRY_GATE_FAIL:running sm clock liveness below ${CLK_RUN_MIN_MHZ}MHz" 15
echo "LAUNCH_VERIFIED run_id=$RUN_ID log=$LOG sidecar=$SIDE manifest_sha=$MSHA receipt_sha=$RSHA pid_attribution=PASS"

# completion wait with periodic foreign-process sampling: every poll tick,
# any NVML compute pid on a target GPU that is not one of our 4 workers
# counts as a foreign hit (gated to zero)
WAIT=${BENCH_WAIT_SECS:-900}
DONE=0; NSAMP=0; NFOREIGN=0
for i in $(seq 1 $((WAIT/10))); do
  PAIRSM=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  NSAMP=$((NSAMP+1))
  for U in $TUUIDS; do
    for P in $(printf '%s\n' "$PAIRSM" | grep "^$U," | cut -d, -f2); do
      echo "$WPIDS" | tr ' ' '\n' | grep -qx "$P" || NFOREIGN=$((NFOREIGN+1))
    done
  done
  grep -q '^RUN_END:' "$LOG" && { DONE=1; break; }
  sleep 10
done
side "MIDRUN_FOREIGN_SAMPLES:samples=$NSAMP hits=$NFOREIGN"
[ "$DONE" -eq 1 ] || fail "COMPLETION_FAIL:no RUN_END within ${WAIT}s" 15
[ "$NFOREIGN" -eq 0 ] || fail "TELEMETRY_GATE_FAIL:midrun foreign compute pids on target GPUs (hits=$NFOREIGN)" 15
[ "$(grep -c '^RUN_REAL_EXIT:0$' "$LOG")" -eq 1 ] || fail "COMPLETION_FAIL:run_real_exit=$(grep '^RUN_REAL_EXIT:' "$LOG" | head -1 | cut -d: -f2-)" 15
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$JSONF" 2>/dev/null \
  || fail "COMPLETION_FAIL:json missing or unparseable at $JSONF" 15
side "TELEMETRY_END_BEGIN:$(date -u +%F_%T)"
OCC=-1
for i in $(seq 1 6); do OCC=$(occq); [ "$OCC" -eq 0 ] && break; sleep 5; done
side "END_OCCUPANCY:$OCC"
[ "$OCC" -eq 0 ] || fail "TELEMETRY_GATE_FAIL:post-run foreign occupancy=$OCC on target GPUs" 15
LOAD_END=$(cat /proc/loadavg)
side "HOST_LOADAVG_END:$LOAD_END"
side "HOST_VMSTAT_END:$(vmstat 1 2 2>/dev/null | tail -1 | tr -s ' ')"
side "HOST_GPU_CLOCKS_END:$(gpuq)"
L1S=${LOAD_START%% *}; L1E=${LOAD_END%% *}
LOAD1_DELTA_MAX=${LOAD1_DELTA_MAX:-16}
DELTA_OK=$(python3 -c "print(1 if float('$L1E')-float('$L1S') <= float('$LOAD1_DELTA_MAX') else 0)" 2>/dev/null)
side "LOAD1_DELTA_GATE:start=$L1S end=$L1E max=+$LOAD1_DELTA_MAX ok=${DELTA_OK:-0}"
[ "${DELTA_OK:-0}" = "1" ] || fail "TELEMETRY_GATE_FAIL:load1 rose $L1S -> $L1E (> +$LOAD1_DELTA_MAX)" 15
side "TELEMETRY_FINAL_PASS"
side "SIDECAR_END:$(date -u +%F_%T)"
echo "LAUNCH_COMPLETE run_id=$RUN_ID log=$LOG json=$JSONF sidecar=$SIDE manifest_sha=$MSHA receipt_sha=$RSHA mode=$BMODE formal_validity=$FORMAL_VALIDITY run_real_exit=0 telemetry=PASS"
