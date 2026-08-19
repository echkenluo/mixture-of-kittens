#!/bin/bash
# Host-side launcher v6 (tracked). Trust model: the deployment receipt is
# generated on the trusted packaging host (make_deploy_receipt.sh) and its
# hash travels out-of-band as EXPECTED_RECEIPT_SHA256. The FIRST gate is the
# exact comparison of the receipt bytes against that external prior - a
# read-only bit or schema validity establishes nothing by itself, and a
# self-consistent rewrite of manifest+receipt still fails here. The verified
# end never self-signs and performs no git lookups.
#
# REAL ORDER (the previous version of this comment described a flow that no
# longer exists - it claimed an atomic sidecar header written after the image
# gates, while allocation now happens before any content is verified):
#   A. entrypoint: no inherited BASH_ENV/ENV/SHELLOPTS/BASHOPTS, no exported
#      functions, PATH pinned                                      -> exit 4
#   B. BENCH_TAG is one safe path segment; measurement thresholds are not
#      caller-settable; operational knobs resolved to numbers      -> exit 4/15
#   C. BENCH_MODE gate: formal is UNCONDITIONALLY refused          -> exit 14
#   D. EXPECTED_RECEIPT_SHA256 present + 64-hex; both staged files exist and
#      are not writable (staging hygiene, not a trust property)    -> exit 14/12
#   E. PRE-HEADER ALLOCATION: run log/json paths must not exist, then the
#      sidecar and the two snapshots are created with O_EXCL. Anything that
#      fails here removes ONLY what this run created and leaves nothing behind
#      - a pre-existing path is refused by the create, so it never enters the
#      cleanup list                                                -> exit 4
#   F. PROGRESSIVE sidecar: the header is written line by line into the file
#      this run reserved, starting with LAUNCH_STATE:INCOMPLETE. It is NOT an
#      atomic header, and the cleanup is disarmed from this point: every later
#      refusal is recorded as LAUNCH_REJECTED in the sidecar rather than erased
#   G. receipt: snapshot sha == EXPECTED_RECEIPT_SHA256, shared validator,
#      schema 1 only (a record-bound receipt has no record here)   -> exit 14
#   H. manifest: snapshot sha == receipt, basename == receipt MANIFEST_FILE,
#      shared validator + harness/SO drift, receipt<->manifest hashes
#                                                                  -> exit 12/13/14
#   I. live image id/ref/RepoDigests == receipt; a failed inspect is an error,
#      not a local-only image                                      -> exit 14
#   J. prelaunch telemetry BEFORE docker exec: mapping, occupancy wait,
#      clocks/load/vmstat snapshot                                 -> exit 6/15
#   K. docker exec runner; launch verification (exact log path, runner gates,
#      uuid set equality, 1 parent + 4 workers, lock held, docker-top, per-GPU
#      PID attribution, running clock floor)                       -> exit 5-8/15
#   L. completion wait with periodic foreign-process sampling; RUN_END +
#      exactly one RUN_REAL_EXIT:0 + parseable JSON                -> exit 15
#   M. end telemetry gates; LAUNCH_STATE_FINAL:COMPLETE + SIDECAR_END
#
# All fields are read from the snapshots after E, never from the caller's paths.
# Gated quantities: target-GPU occupancy (prelaunch / periodic midrun / end),
# absolute prelaunch load1, load1 delta, and a single-sample running SM clock
# LIVENESS floor (not a stability gate - see below). Power draw and vmstat are
# RECORD-ONLY disclosures and are never gates.
# Usage: BENCH_TAG=... EXPECTED_RECEIPT_SHA256=... [BENCH_MODE=formal|canary] \
#          bash host_launch_sm90.sh <container> <host_mok_dir> <manifest> <receipt>
set -uo pipefail
# ---------------------------------------------------------------- entrypoint
# Misuse protection ONLY, and it must be said plainly: every gate below is a
# shell-out (sha256sum, stat, grep, cp, docker) resolved through PATH, so a
# caller who controls the environment controls the verdict. A hostile
# sha256sum was shown to make a forged receipt pass the anchor gate while the
# sidecar recorded the anchored hash. Fixing PATH and refusing inherited
# BASH_ENV/exported functions raises the bar; it does NOT make the verified end
# trustworthy. Binding this gate code to the receipt is NOT implemented, so the
# deployment-tooling boundary stays UNVERIFIED (see the contract doc).
# Reserved words and absolute paths only - a function cannot shadow either.
[[ -x /usr/bin/env && -x /usr/bin/grep ]] \
  || { echo "LAUNCH_VERIFY_FAIL:/usr/bin/env or /usr/bin/grep missing; the entrypoint cannot be inspected"; exit 4; }
[[ "${BASH_SOURCE[0]}" == "$0" ]] \
  || { echo "LAUNCH_VERIFY_FAIL:this launcher must be executed, not sourced"; exit 4; }
BAD_ENTRY=$(/usr/bin/env | /usr/bin/grep -m1 -oE '^(BASH_FUNC_[^=%(]*|BASH_ENV|ENV|SHELLOPTS|BASHOPTS)=?' || true)
BAD_ENTRY=${BAD_ENTRY%=}
if [[ -n $BAD_ENTRY ]]; then
  case $BAD_ENTRY in
    BASH_FUNC_*) echo "LAUNCH_VERIFY_FAIL:exported shell function ${BAD_ENTRY#BASH_FUNC_} is present; a function shadows PATH lookups" ;;
    *) echo "LAUNCH_VERIFY_FAIL:$BAD_ENTRY is set; this launcher must be started from a clean entrypoint" ;;
  esac
  exit 4
fi
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CT=${1:?container name}
MOKDIR=${2:?host mok dir}
MANIFEST=${3:?manifest path (committed, read-only)}
RECEIPT=${4:?deployment receipt path (packaging-generated, read-only)}
TAG=${BENCH_TAG:?BENCH_TAG required}
DIR=$(cd "$(dirname "$0")" && pwd)
# BENCH_TAG becomes a filename component for the log, the JSON, the sidecar and
# both per-run copies. Unvalidated, `../evil` put all of them outside
# host-runs/. Validate BEFORE anything is created.
printf '%s' "$TAG" | grep -qE '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$' \
  || { echo "LAUNCH_VERIFY_FAIL:BENCH_TAG must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\$ (it becomes a filename); got [$TAG]"; exit 4; }
# Measurement thresholds are part of the contract, not caller preference: with
# these as env overrides the same machine state passed or failed purely by what
# the caller asked for, and the sidecar recorded the caller's number as if it
# were the contract.
# The operational knobs stay callable, but the earlier claim that they "can
# only make a run fail sooner" was WRONG in both directions: a larger
# PREFLIGHT_TRIES turns an occupancy wait that would have timed out into a
# pass, and a larger BENCH_TIMEOUT/BENCH_WAIT_SECS lets a run that would have
# been abandoned finish. What they cannot do is move a numerical validity
# threshold. They are range-checked and written into the sidecar so a reader
# can see what admission policy the run was given.
for V in LOAD1_MAX_PRELAUNCH LOAD1_DELTA_MAX CLK_RUN_MIN_MHZ; do
  [ -z "$(printenv "$V" || true)" ] \
    || { echo "TELEMETRY_GATE_FAIL:$V is a tracked measurement threshold and cannot be set by the caller"; exit 15; }
done
LOAD1_MAX_PRELAUNCH=64; LOAD1_DELTA_MAX=16; CLK_RUN_MIN_MHZ=500
# Resolved HERE to the number that will actually be in force, so the sidecar
# records the effective policy rather than the word "default", and the runner
# is handed the same number instead of applying its own fallback.
# Bounds are chosen from what the knob does: the preflight loop sleeps 5s per
# try, so 120 tries is a 10-minute admission wait; the benchmark timeout and
# the host completion wait are capped at 1h and 2h. The earlier "below
# 1000000" allowed a ~58-day preflight wait.
knob() { # name default min max
  local N=$1 D=$2 LO=$3 HI=$4 V
  V=$(printenv "$N" || true)
  [ -z "$V" ] && { echo "$D"; return 0; }
  printf '%s' "$V" | grep -qE '^[0-9]+$' \
    || { echo "LAUNCH_VERIFY_FAIL:$N must be an integer, got [$V]" >&2; return 1; }
  { [ "$V" -ge "$LO" ] && [ "$V" -le "$HI" ]; } \
    || { echo "LAUNCH_VERIFY_FAIL:$N=$V outside [$LO,$HI]" >&2; return 1; }
  echo "$V"
}
PREFLIGHT_TRIES=$(knob PREFLIGHT_TRIES 24 1 120) || exit 4
BENCH_TIMEOUT=$(knob BENCH_TIMEOUT 600 1 3600) || exit 4
BENCH_WAIT_SECS=$(knob BENCH_WAIT_SECS 900 1 7200) || exit 4

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

EXPR_SHA=${EXPECTED_RECEIPT_SHA256:-}
echo "$EXPR_SHA" | grep -qE '^[0-9a-f]{64}$' \
  || { echo "RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 env missing or not 64-hex"; exit 14; }
[ -f "$RECEIPT" ] || { echo "RECEIPT_TRUST_FAIL:missing receipt $RECEIPT"; exit 14; }
[ -f "$MANIFEST" ] || { echo "MANIFEST_SCHEMA_FAIL:missing $MANIFEST"; exit 12; }
# staging hygiene, not a security property (the snapshot below is): the
# operator is expected to stage read-only artifacts
for F in "$RECEIPT" "$MANIFEST"; do
  case "$(stat -c %a "$F")" in *[2367]*) echo "RECEIPT_TRUST_FAIL:staged artifact $F is writable ($(stat -c %a "$F"))"; exit 14 ;; esac
done

# ------------------------------------------------- snapshot, verify, then use
# The previous version hashed the live files and then kept re-reading them:
# field reads, the validators, ENVARGS and the copy all went back to the
# caller's path. Flipping the manifest between the hash check and the copy made
# the launcher run with BENCH_GPUS the anchor never covered while every hash
# still matched. Each artifact is now read ONCE into a per-run snapshot, the
# snapshot is what gets verified, and every later read comes from the snapshot.
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM
LOG=$MOKDIR/runs/$TAG-$RUN_ID.log
JSONF=$MOKDIR/runs/$TAG-$RUN_ID.json
SIDE=$MOKDIR/host-runs/$TAG-$RUN_ID.host
MCOPY=$MOKDIR/host-runs/$TAG-$RUN_ID.manifest
RCOPY=$MOKDIR/host-runs/$TAG-$RUN_ID.receipt
mkdir -p "$MOKDIR/host-runs" 2>/dev/null || true
# The three HOST evidence files are created with O_EXCL (noclobber redirection),
# so an existing path or a planted symlink is refused rather than followed.
# NOT covered, and stated so rather than implied: the container's run log and
# the harness JSON are written with plain truncation and os.replace inside the
# container. The pre-check below refuses a run whose log/json path already
# exists, which closes the practical collision but is a CHECK, not an atomic
# create - the window between the check and the container's open stays OPEN.
for F in "$LOG" "$JSONF"; do
  [ -e "$F" ] && { echo "LAUNCH_VERIFY_FAIL:run artifact $F already exists"; exit 4; }
done
# Until the sidecar carries state, a half-finished allocation is worse than
# nothing: an empty .host beside one snapshot looks like the start of a
# canonical run. Only files THIS run created are tracked and removed - a
# pre-existing path is never touched, because the O_EXCL create refuses it and
# it therefore never enters the list.
CREATED=()
cleanup_partial() { local F; for F in ${CREATED[@]+"${CREATED[@]}"}; do rm -f "$F"; done; }
trap cleanup_partial EXIT
set -o noclobber
if ! { : > "$SIDE"; } 2>/dev/null; then echo "LAUNCH_VERIFY_FAIL:sidecar $SIDE already exists"; exit 4; fi
CREATED+=("$SIDE")
if ! { cat < "$RECEIPT" > "$RCOPY"; } 2>/dev/null; then echo "LAUNCH_VERIFY_FAIL:receipt snapshot $RCOPY already exists or the receipt is unreadable"; exit 4; fi
CREATED+=("$RCOPY")
if ! { cat < "$MANIFEST" > "$MCOPY"; } 2>/dev/null; then echo "LAUNCH_VERIFY_FAIL:manifest snapshot $MCOPY already exists or the manifest is unreadable"; exit 4; fi
CREATED+=("$MCOPY")
set +o noclobber
# a chmod that silently fails leaves evidence writable while the log says it is
# not; this is a gate, not a nicety
chmod 444 "$MCOPY" "$RCOPY" \
  || { echo "LAUNCH_VERIFY_FAIL:cannot make the per-run snapshots read-only"; exit 4; }
side() { echo "$1" >> "$SIDE" || { echo "LAUNCH_VERIFY_FAIL:sidecar write failed"; exit 4; }; }
fail() { side "LAUNCH_VERIFY_FAIL:$1"; echo "LAUNCH_VERIFY_FAIL:$1"; exit "$2"; }
# every gate from here on is recorded IN the sidecar before exiting: an
# abandoned empty .host next to two snapshots used to look exactly like the
# start of a canonical run
rfail() { side "LAUNCH_REJECTED:$1"; echo "$1"; exit "$2"; }
# the header goes into the file this run reserved - the previous version wrote
# a temp file and mv -f'd it over the reservation, which threw the O_EXCL
# guarantee away at the last step. Values that are not known yet are appended
# by the gates that establish them, each exactly once.
side "SIDECAR_START:$(date -u +%F_%T)"
side "RUN_ID:$RUN_ID"
side "LAUNCH_STATE:INCOMPLETE"
side "BENCH_MODE:$BMODE"
side "FORMAL_VALIDITY:$FORMAL_VALIDITY"
side "MANIFEST_FILE:$(basename "$MANIFEST")"
side "RECEIPT_FILE:$(basename "$RECEIPT")"
side "RECEIPT_COPY:$(basename "$RCOPY")"
side "MEASUREMENT_THRESHOLDS:load1_max=$LOAD1_MAX_PRELAUNCH load1_delta_max=$LOAD1_DELTA_MAX clk_run_min_mhz=$CLK_RUN_MIN_MHZ (tracked, not caller-set)"
side "OPERATIONAL_KNOBS:preflight_tries=$PREFLIGHT_TRIES bench_timeout=$BENCH_TIMEOUT bench_wait_secs=$BENCH_WAIT_SECS (effective values; admission/completion policy, not validity thresholds)"
# the sidecar now names the run and its policy: from here a rejection is
# recorded rather than erased
trap - EXIT
RSHA=$(sha256sum "$RCOPY" | cut -d' ' -f1)
[ "$RSHA" = "$EXPR_SHA" ] \
  || rfail "RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 mismatch (snapshot $RSHA expected $EXPR_SHA)" 14
VOUT=$(bash "$DIR/validate_receipt_sm90.sh" "$RCOPY" --check-mode 2>&1) || rfail "$VOUT" 14
rget() { grep "^$1=" "$RCOPY" | head -1 | cut -d= -f2-; }
side "RECEIPT_SCHEMA:$(rget RECEIPT_SCHEMA)"
side "RECEIPT_SHA256:$RSHA"
# schema 2 asserts a build-record binding this launcher is given no record to
# check, so accepting it would run under provenance nothing here verified
[ "$(rget RECEIPT_SCHEMA)" = "1" ] \
  || rfail "RECEIPT_TRUST_FAIL:receipt schema $(rget RECEIPT_SCHEMA) claims a build-record binding, and no record is supplied to this launcher" 14


MSHA=$(sha256sum "$MCOPY" | cut -d' ' -f1)
side "MANIFEST_SHA256:$MSHA"
[ "$MSHA" = "$(rget MANIFEST_SHA256)" ] \
  || rfail "MANIFEST_TRUST_FAIL:manifest snapshot sha $MSHA != receipt $(rget MANIFEST_SHA256)" 14
# the receipt names the manifest it describes; the generator writes a bare
# filename, so a name that disagrees means the wrong artifact was staged
[ "$(basename "$MANIFEST")" = "$(rget MANIFEST_FILE)" ] \
  || rfail "MANIFEST_TRUST_FAIL:manifest basename $(basename "$MANIFEST") != receipt MANIFEST_FILE $(rget MANIFEST_FILE)" 14

mget() { grep "^$1=" "$MCOPY" | head -1 | cut -d= -f2-; }
# schema 2 selects the implementation; schema 1 is MoK-only by definition
HARNESS_MODULE=$(mget HARNESS_MODULE)
[ -n "$HARNESS_MODULE" ] || HARNESS_MODULE=benchmarks.bench_sm90_fwd
HARNESS_FILE=$MOKDIR/mixture-of-kittens/$(echo "$HARNESS_MODULE" | tr '.' '/').py
[ -f "$HARNESS_FILE" ] || rfail "MANIFEST_SCHEMA_FAIL:harness file for $HARNESS_MODULE missing" 12
bash "$DIR/validate_manifest_sm90.sh" "$MCOPY" \
  --harness "$HARNESS_FILE" \
  --so-dir "$MOKDIR/mixture-of-kittens/mok"
VRC=$?
[ "$VRC" -eq 0 ] || { side "LAUNCH_REJECTED:manifest snapshot failed the shared validator (rc=$VRC)"; exit "$VRC"; }
EXPECTED=$(mget EXPECTED_HARNESS_SHA256); EXPSO=$(mget EXPECTED_SO_SHA256)
FROZEN=$(mget FROZEN_COMMIT); BGPUS=$(mget BENCH_GPUS)
[ "$(rget HARNESS_SHA256)" = "$EXPECTED" ] || rfail "RECEIPT_TRUST_FAIL:harness sha receipt != manifest" 14
[ "$(rget SO_SHA256)" = "$EXPSO" ] || rfail "RECEIPT_TRUST_FAIL:so sha receipt != manifest" 14

IMGID=$(docker inspect --format '{{.Image}}' "$CT" 2>/dev/null)
[ -n "$IMGID" ] || rfail "IMAGE_TRUST_FAIL:container inspect failed for $CT" 14
IMGREF=$(docker inspect --format '{{.Config.Image}}' "$CT" 2>/dev/null)
[ -n "$IMGREF" ] || rfail "IMAGE_TRUST_FAIL:image ref empty" 14
# a failed inspect is NOT a local-only image: treating both as NONE let a
# docker daemon error satisfy a receipt that says NONE, so a tool failure was
# laundered into a provenance statement
IMGRD=$(docker image inspect --format '{{join .RepoDigests ","}}' "$IMGID" 2>/dev/null); IRC=$?
[ "$IRC" -eq 0 ] || rfail "IMAGE_TRUST_FAIL:docker image inspect failed (rc=$IRC) for $IMGID; a tool failure cannot stand in for NONE" 14
[ -n "$IMGRD" ] || IMGRD=NONE
[ "$IMGID" = "$(rget IMAGE_ID)" ] || rfail "IMAGE_TRUST_FAIL:image id live $IMGID != receipt $(rget IMAGE_ID)" 14
[ "$IMGREF" = "$(rget IMAGE_REF)" ] || rfail "IMAGE_TRUST_FAIL:image ref live $IMGREF != receipt $(rget IMAGE_REF)" 14
[ "$IMGRD" = "$(rget IMAGE_REPO_DIGESTS)" ] || rfail "IMAGE_TRUST_FAIL:repo digests live $IMGRD != receipt $(rget IMAGE_REPO_DIGESTS)" 14

side "IMAGE_ID:$IMGID"
side "IMAGE_REF:$IMGREF"
side "IMAGE_REPO_DIGESTS:$IMGRD"

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
# always passed, so the runner cannot apply a fallback the sidecar never saw
ENVARGS+=(-e PREFLIGHT_TRIES="$PREFLIGHT_TRIES" -e BENCH_TIMEOUT="$BENCH_TIMEOUT")
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
WAIT=$BENCH_WAIT_SECS
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
DELTA_OK=$(python3 -c "print(1 if float('$L1E')-float('$L1S') <= float('$LOAD1_DELTA_MAX') else 0)" 2>/dev/null)
side "LOAD1_DELTA_GATE:start=$L1S end=$L1E max=+$LOAD1_DELTA_MAX ok=${DELTA_OK:-0}"
[ "${DELTA_OK:-0}" = "1" ] || fail "TELEMETRY_GATE_FAIL:load1 rose $L1S -> $L1E (> +$LOAD1_DELTA_MAX)" 15
side "TELEMETRY_FINAL_PASS"
side "LAUNCH_STATE_FINAL:COMPLETE"
side "SIDECAR_END:$(date -u +%F_%T)"
echo "LAUNCH_COMPLETE run_id=$RUN_ID log=$LOG json=$JSONF sidecar=$SIDE manifest_sha=$MSHA receipt_sha=$RSHA mode=$BMODE formal_validity=$FORMAL_VALIDITY run_real_exit=0 telemetry=PASS"
