#!/bin/bash
# Completion gate v6 (tracked). Anchored on the OUT-OF-BAND expected receipt
# hash (EXPECTED_RECEIPT_SHA256 env) - the verifier never treats a hash it
# computed from on-target files as a prior. Reconciliation:
#   - receipt sha256: expected(env) == given file == per-run copy == sidecar
#     == runner log == JSON provenance.receipt_sha256_env (six-way EXACT)
#   - manifest sha256: given file == per-run copy == sidecar == runner log
#     == JSON provenance.manifest_sha256_env (five-way EXACT), and given
#     manifest sha == receipt MANIFEST_SHA256
#   - critical log/sidecar lines are required to appear EXACTLY once before
#     their value is used
#   - full receipt + manifest semantic revalidation via the shared
#     validators (incl. harness/SO drift vs the current tree)
#   - BENCH_GPUS reconciled across manifest, runner log, JSON and the 4
#     sidecar MAP lines
#   - telemetry gate line VALUES checked (occupancy 0/0, midrun hits=0,
#     clock gate low=0, load delta ok=1); power/vmstat are record-only
#   - image identity sidecar == receipt, and live == receipt when a
#     container is given
# Prints VERIFY_PASS mode=<mode> or VERIFY_FAIL:<why>.
# Usage: EXPECTED_RECEIPT_SHA256=... \
#          verify_run_sm90.sh <mokdir> <tag> <run_id> <manifest> <receipt> [container]
set -uo pipefail
M=${1:?}; T=${2:?}; R=${3:?}; MAN=${4:?}; REC=${5:?}; CT=${6:-}
DIR=$(cd "$(dirname "$0")" && pwd)
LOG=$M/runs/$T-$R.log; JSON=$M/runs/$T-$R.json
SIDE=$M/host-runs/$T-$R.host; MCOPY=$M/host-runs/$T-$R.manifest
RCOPY=$M/host-runs/$T-$R.receipt
vf() { echo "VERIFY_FAIL:$1"; exit 1; }
EXPR_SHA=${EXPECTED_RECEIPT_SHA256:-}
echo "$EXPR_SHA" | grep -qE '^[0-9a-f]{64}$' || vf "EXPECTED_RECEIPT_SHA256 env missing or not 64-hex"
[ -f "$MAN" ] || vf "manifest $MAN missing"
[ -f "$REC" ] || vf "receipt $REC missing"
[ -f "$MCOPY" ] || vf "per-run manifest copy missing"
[ -f "$RCOPY" ] || vf "per-run receipt copy missing"
[ -f "$LOG" ] || vf "no log $LOG"
[ -f "$JSON" ] || vf "no json $JSON"
[ -f "$SIDE" ] || vf "no sidecar $SIDE"
RSHA=$(sha256sum "$REC" | cut -d' ' -f1)
[ "$RSHA" = "$EXPR_SHA" ] || vf "EXPECTED_RECEIPT_SHA256 mismatch (given receipt)"
RC_COPY=$(sha256sum "$RCOPY" | cut -d' ' -f1)
[ "$RC_COPY" = "$EXPR_SHA" ] || vf "receipt copy tampered (copy $RC_COPY != expected)"
bash "$DIR/validate_receipt_sm90.sh" "$REC" --check-mode >/dev/null || vf "receipt failed shared validator"
bash "$DIR/validate_manifest_sm90.sh" "$MAN" \
  --harness "$M/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" \
  --so-dir "$M/mixture-of-kittens/mok" >/dev/null || vf "manifest failed shared validator (schema or drift)"
bash "$DIR/validate_manifest_sm90.sh" "$MCOPY" >/dev/null || vf "per-run copy failed shared validator"
cnt1() { [ "$(grep -c "$2" "$1" || true)" -eq 1 ] || vf "need exactly one '$2' line in $(basename "$1")"; }
rget() { grep "^$1=" "$REC" | head -1 | cut -d= -f2-; }
mget() { grep "^$1=" "$MAN" | head -1 | cut -d= -f2-; }
MS_GIVEN=$(sha256sum "$MAN" | cut -d' ' -f1)
MS_COPY=$(sha256sum "$MCOPY" | cut -d' ' -f1)
cnt1 "$SIDE" '^MANIFEST_SHA256:'; MS_SIDE=$(grep '^MANIFEST_SHA256:' "$SIDE" | cut -d: -f2)
cnt1 "$LOG" '^MANIFEST_SHA256:';  MS_LOG=$(grep '^MANIFEST_SHA256:' "$LOG" | cut -d: -f2)
[ "$MS_GIVEN" = "$(rget MANIFEST_SHA256)" ] || vf "given manifest sha != receipt"
[ "$MS_GIVEN" = "$MS_COPY" ] || vf "manifest copy tampered (given $MS_GIVEN != copy $MS_COPY)"
[ "$MS_SIDE" = "$MS_COPY" ] || vf "sidecar pre-start manifest hash != copy"
[ "$MS_LOG" = "$MS_COPY" ] || vf "runner log manifest hash != copy"
cnt1 "$SIDE" '^RECEIPT_SHA256:'; RS_SIDE=$(grep '^RECEIPT_SHA256:' "$SIDE" | cut -d: -f2)
cnt1 "$LOG" '^RECEIPT_SHA256:';  RS_LOG=$(grep '^RECEIPT_SHA256:' "$LOG" | cut -d: -f2)
[ "$RS_SIDE" = "$EXPR_SHA" ] || vf "sidecar receipt sha mismatch"
[ "$RS_LOG" = "$EXPR_SHA" ] || vf "runner log receipt sha mismatch"
for K in IMAGE_ID IMAGE_REF IMAGE_REPO_DIGESTS; do
  cnt1 "$SIDE" "^$K:"
  SV=$(grep "^$K:" "$SIDE" | cut -d: -f2-)
  [ "$SV" = "$(rget "$K")" ] || vf "sidecar $K ($SV) != receipt ($(rget "$K"))"
done
if [ -n "$CT" ]; then
  LIVEID=$(docker inspect --format '{{.Image}}' "$CT" 2>/dev/null)
  [ "$LIVEID" = "$(rget IMAGE_ID)" ] || vf "live image id != receipt"
  LIVEREF=$(docker inspect --format '{{.Config.Image}}' "$CT" 2>/dev/null)
  [ "$LIVEREF" = "$(rget IMAGE_REF)" ] || vf "live image ref != receipt"
  LIVERD=$(docker image inspect --format '{{join .RepoDigests ","}}' "$LIVEID" 2>/dev/null)
  [ -n "$LIVERD" ] || LIVERD=NONE
  [ "$LIVERD" = "$(rget IMAGE_REPO_DIGESTS)" ] || vf "live repo digests != receipt"
fi
cnt1 "$SIDE" '^BENCH_MODE:'; SMODE=$(grep '^BENCH_MODE:' "$SIDE" | cut -d: -f2)
cnt1 "$LOG" '^BENCH_MODE:';  LMODE=$(grep '^BENCH_MODE:' "$LOG" | cut -d: -f2)
[ "$SMODE" = "$LMODE" ] || vf "sidecar mode != log mode"
case "$SMODE" in formal|canary) : ;; *) vf "invalid mode $SMODE" ;; esac
cnt1 "$SIDE" '^FORMAL_VALIDITY:'; FV=$(grep '^FORMAL_VALIDITY:' "$SIDE" | cut -d: -f2)
if [ "$SMODE" = "formal" ]; then
  [ "$FV" = "VALID_FOR_FORMAL" ] || vf "formal mode but validity=$FV"
  [ "$(rget BINARY_BUILD_COMMIT)" != "UNKNOWN" ] || vf "formal mode with UNKNOWN build lineage"
  [ "$(rget IMAGE_REPO_DIGESTS)" != "NONE" ] || vf "formal mode with local-only image"
else
  [ "$FV" = "INVALID_FOR_FORMAL" ] || vf "canary mode but validity=$FV"
fi
SHA=$(mget EXPECTED_HARNESS_SHA256); EXPSO=$(mget EXPECTED_SO_SHA256); FC=$(mget FROZEN_COMMIT)
BGP=$(mget BENCH_GPUS)
[ "$(rget HARNESS_SHA256)" = "$SHA" ] || vf "receipt harness sha != manifest"
[ "$(rget SO_SHA256)" = "$EXPSO" ] || vf "receipt so sha != manifest"
cnt1 "$LOG" "^RUN_ID:$R\$"
cnt1 "$LOG" '^HARNESS_SHA256:'; [ "$(grep '^HARNESS_SHA256:' "$LOG" | cut -d: -f2)" = "$SHA" ] || vf "log harness sha != manifest"
cnt1 "$LOG" '^SO_SHA256:'; [ "$(grep '^SO_SHA256:' "$LOG" | cut -d: -f2)" = "$EXPSO" ] || vf "log so sha != manifest"
cnt1 "$LOG" '^BENCH_GPUS:'; [ "$(grep '^BENCH_GPUS:' "$LOG" | cut -d: -f2)" = "$BGP" ] || vf "log BENCH_GPUS != manifest"
[ "$(grep -c '^RUN_REAL_EXIT:0$' "$LOG")" -eq 1 ] || vf "need exactly one RUN_REAL_EXIT:0, got: $(grep RUN_REAL_EXIT "$LOG" | head -1)"
cnt1 "$LOG" '^RUN_END:'
for A in RUN_START HASH_GATE_PASS PREFLIGHT_PASS SO_GATE_PASS; do
  cnt1 "$LOG" "^$A"
done
[ "$(grep -c '^BENCH|' "$LOG")" -eq 1 ] || vf "need exactly one BENCH line"
cnt1 "$SIDE" "^RUN_ID:$R\$"
[ "$(grep -c '^MAP:' "$SIDE")" -eq 4 ] || vf "need exactly 4 MAP lines"
MAPIDX=$(grep '^MAP:' "$SIDE" | cut -d: -f2 | cut -d, -f1 | sort -n | tr '\n' ',' | sed 's/,$//')
BGPSORT=$(echo "$BGP" | tr ',' '\n' | sort -n | tr '\n' ',' | sed 's/,$//')
[ "$MAPIDX" = "$BGPSORT" ] || vf "MAP container idx set ($MAPIDX) != manifest BENCH_GPUS ($BGPSORT)"
cnt1 "$SIDE" '^NVML_UUID_PID_PAIRS:'
NPAIR=$(grep '^NVML_UUID_PID_PAIRS:' "$SIDE" | tr ';' '\n' | grep -c 'GPU-' || true)
[ "$NPAIR" -eq 4 ] || vf "need 4 nvml uuid,pid pairs, got $NPAIR"
cnt1 "$SIDE" '^PID_ATTRIBUTION_PASS$'
for A in TELEMETRY_PRELAUNCH_BEGIN TELEMETRY_PRELAUNCH_END HOST_GPU_CLOCKS_PRELAUNCH \
         HOST_GPU_CLOCKS_RUNNING TELEMETRY_END_BEGIN HOST_GPU_CLOCKS_END \
         TELEMETRY_FINAL_PASS SIDECAR_END; do
  cnt1 "$SIDE" "^$A"
done
cnt1 "$SIDE" '^PRELAUNCH_OCCUPANCY:'; grep -q '^PRELAUNCH_OCCUPANCY:0$' "$SIDE" || vf "prelaunch occupancy not zero"
cnt1 "$SIDE" '^END_OCCUPANCY:'; grep -q '^END_OCCUPANCY:0$' "$SIDE" || vf "end occupancy not zero"
cnt1 "$SIDE" '^MIDRUN_FOREIGN_SAMPLES:'
grep '^MIDRUN_FOREIGN_SAMPLES:' "$SIDE" | grep -q 'hits=0$' || vf "midrun foreign hits not zero"
cnt1 "$SIDE" '^RUNNING_CLOCK_GATE:'
grep '^RUNNING_CLOCK_GATE:' "$SIDE" | grep -q 'low=0$' || vf "running clock gate not clean"
cnt1 "$SIDE" '^LOAD1_DELTA_GATE:'
grep '^LOAD1_DELTA_GATE:' "$SIDE" | grep -q 'ok=1$' || vf "load1 delta gate not ok"
python3 - "$JSON" "$SHA" "$FC" "$R" "$MAN" "$EXPSO" "$MS_GIVEN" "$EXPR_SHA" "$BGP" "$SMODE" <<'PY' || exit 1
import json, math, statistics, sys
d = json.load(open(sys.argv[1]))
m = d["meta"]; p = m["provenance"]
man = dict(l.strip().split("=", 1) for l in open(sys.argv[5]) if "=" in l)
sh = m["shape"]
checks = [
    (d.get("schema") == "bench-sm90-fwd.v1", "schema mismatch"),
    (p["harness_sha256"] == sys.argv[2], "json harness sha != manifest"),
    (p["frozen_commit_env"] == sys.argv[3], "json frozen commit != manifest"),
    (p.get("so_sha256") == sys.argv[6], "json so sha256 != manifest"),
    (p.get("manifest_sha256_env") == sys.argv[7], "json manifest sha != given manifest (exact)"),
    (p.get("receipt_sha256_env") == sys.argv[8], "json receipt sha != expected (exact)"),
    (p.get("bench_gpus_env") == sys.argv[9], "json bench_gpus != manifest"),
    (p.get("bench_mode_env") == sys.argv[10], "json bench_mode != sidecar"),
    (m.get("run_id") == sys.argv[4], "json run_id mismatch"),
    (len(d["samples_ms"]) == m["timed_iters"], "sample count mismatch"),
    (all(math.isfinite(x) and x > 0 for x in d["samples_ms"]), "samples not all finite>0"),
]
o = sorted(d["samples_ms"])
rp50 = round(statistics.median(o), 4)
rp95 = round(o[max(0, int(len(o) * 0.95) - 1)], 4)
checks.append((abs(rp50 - d["p50_ms"]) < 1e-3, f"p50 recompute {rp50} != {d['p50_ms']}"))
checks.append((abs(rp95 - d["p95_ms"]) < 1e-3, f"p95 recompute {rp95} != {d['p95_ms']}"))
cg = m["correctness_gate"]
ta, tr = cg["tolerance_abs_rel"]
checks.append((math.isfinite(cg["abs_max"]) and cg["abs_max"] <= ta, "correctness abs_max above tolerance"))
checks.append((math.isfinite(cg["relative"]) and cg["relative"] <= tr, "correctness relative above tolerance"))
for k, loc in (("tokens_per_rank", sh), ("hidden", sh), ("intermediate", sh),
               ("experts", sh), ("topk", sh), ("world_size", sh),
               ("comm_sms", m), ("minibatch", m), ("macrobatch", m),
               ("warmup_iters", m), ("timed_iters", m)):
    checks.append((int(man[k]) == int(loc[k]), f"manifest {k}={man[k]} != json {loc[k]}"))
for ok, msg in checks:
    if not ok:
        print(f"VERIFY_FAIL:{msg}"); sys.exit(1)
print("VERIFY_JSON_OK")
PY
if [ -n "$CT" ]; then
  REL=0
  for i in $(seq 1 12); do
    docker exec "$CT" sh -c 'flock -n /mok/build.lock true' 2>/dev/null && { REL=1; break; }
    sleep 5
  done
  [ "$REL" -eq 1 ] || vf "lock not released after RUN_END"
fi
echo "VERIFY_PASS mode=$SMODE formal_validity=$FV"
