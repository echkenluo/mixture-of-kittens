#!/bin/bash
# Mock integration tests for verify_run_sm90.sh (no GPU, no docker, no torch).
#
# Why this exists: 38bb4e1 passed every schema-text test and still hardcoded
# the MoK harness in the verifier and the MoK module in the launcher's process
# greps, so a DeepEP cell would have failed on drift and on process shape.
# Schema-text tests cannot catch that. These cases synthesize a COMPLETE
# artifact set (log + JSON + sidecar + per-run manifest/receipt copies) and run
# the real verifier over it, once per implementation.
#
# Usage: bash test_verify_integration_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "INT_$1_PASS"; PASS=$((PASS+1)); else echo "INT_$1_FAIL"; FAIL=$((FAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }
H64() { printf "$1%.0s" $(seq 1 64); }
H40() { printf "$1%.0s" $(seq 1 40); }

TMPD=$(mktemp -d)
M=$TMPD/mok
mkdir -p "$M/mixture-of-kittens/mok" "$M/runs" "$M/host-runs"
ln -s "$DIR" "$M/mixture-of-kittens/benchmarks"
printf 'not a real so\n' > "$M/mixture-of-kittens/mok/_Cfixture.so"
SOSHA=$(sha256sum "$M/mixture-of-kittens/mok/_Cfixture.so" | cut -d' ' -f1)
MOKH=$(sha256sum "$DIR/bench_sm90_fwd.py" | cut -d' ' -f1)
DEEPH=$(sha256sum "$DIR/bench_deepep_fwd.py" | cut -d' ' -f1)
PYPIN=$(H64 7); EXTPIN=$(H64 8)
IMGID="sha256:$(H64 1)"; IMGREF="fixture-image:latest"; IMGRD="registry.local/mok@sha256:$(H64 9)"

mkmanifest() { # out impl
  local OUT=$1 IMPL=$2
  { echo "MANIFEST_SCHEMA=2"; echo "FROZEN_COMMIT=$(H40 0)"; echo "IMPL=$IMPL"
    echo "SHAPE_ID=tiny_h20"
    if [ "$IMPL" = mok_sm90 ]; then
      echo "HARNESS_MODULE=benchmarks.bench_sm90_fwd"
      echo "EXPECTED_HARNESS_SHA256=$MOKH"
      echo "TIMING_SEMANTICS=build_schedule+forward.v2"
    else
      echo "HARNESS_MODULE=benchmarks.bench_deepep_fwd"
      echo "EXPECTED_HARNESS_SHA256=$DEEPH"
      echo "TIMING_SEMANTICS=dispatch+expert+combine.v2"
    fi
    echo "EXPECTED_SO_SHA256=$SOSHA"; echo "BENCH_GPUS=0,1,2,3"
    echo "tokens_per_rank=512"; echo "hidden=256"; echo "intermediate=256"
    echo "experts=4"; echo "topk=1"; echo "world_size=4"; echo "comm_sms=24"
    echo "minibatch=256"; echo "macrobatch=4096"; echo "warmup_iters=20"; echo "timed_iters=100"
    if [ "$IMPL" = deepep_torch ]; then
      echo "TORCH_VERSION_PIN=2.11.0+cu130"
      echo "DEEPEP_PY_TREE_SHA256=$PYPIN"
      echo "DEEPEP_EXT_SHA256=$EXTPIN"
      echo "DEEPEP_TORCH_COMPILE=on"
    fi; } > "$OUT"
  chmod 444 "$OUT"
}
mkreceipt() { # out manifest_sha harness_sha
  local OUT=$1 MS=$2 HS=$3
  { echo "RECEIPT_SCHEMA=1"; echo "SOURCE_TREE_COMMIT=$(H40 0)"; echo "HARNESS_COMMIT=$(H40 0)"
    echo "BINARY_BUILD_COMMIT=UNKNOWN"; echo "MANIFEST_FILE=$(basename "$1")"
    echo "MANIFEST_SHA256=$MS"; echo "MANIFEST_GIT_BLOB=$(H40 0)"
    echo "HARNESS_SHA256=$HS"; echo "SO_SHA256=$SOSHA"
    echo "IMAGE_ID=$IMGID"; echo "IMAGE_REF=$IMGREF"; echo "IMAGE_REPO_DIGESTS=$IMGRD"; } > "$OUT"
  chmod 444 "$OUT"
}

# build a complete artifact set; $1 impl, $2 tag, $3 run id
build_run() {
  local IMPL=$1 TAG=$2 RID=$3
  local MAN=$TMPD/$TAG.manifest REC=$TMPD/$TAG.receipt
  local HS MOD JSCHEMA
  if [ "$IMPL" = mok_sm90 ]; then
    HS=$MOKH; MOD=benchmarks.bench_sm90_fwd; JSCHEMA=bench-sm90-fwd.v1
  else
    HS=$DEEPH; MOD=benchmarks.bench_deepep_fwd; JSCHEMA=bench-deepep-fwd.v1
  fi
  mkmanifest "$MAN" "$IMPL"
  local MSHA; MSHA=$(sha256sum "$MAN" | cut -d' ' -f1)
  mkreceipt "$REC" "$MSHA" "$HS"
  local RSHA; RSHA=$(sha256sum "$REC" | cut -d' ' -f1)
  cp "$MAN" "$M/host-runs/$TAG-$RID.manifest"
  cp "$REC" "$M/host-runs/$TAG-$RID.receipt"
  { echo "RUN_ID:$RID"; echo "LOCK_HELD_BY:1"; echo "HARNESS_SHA256:$HS"
    echo "MANIFEST_SHA256:$MSHA"; echo "RECEIPT_SHA256:$RSHA"; echo "BENCH_MODE:canary"
    echo "HARNESS_MODULE:$MOD"; echo "BENCH_GPUS:0,1,2,3"
    echo "TARGET_GPU_UUIDS:0, GPU-aaa;1, GPU-bbb;2, GPU-ccc;3, GPU-ddd"
    echo "HASH_GATE_PASS"; echo "PREFLIGHT_PASS"; echo "SO_PATH:mok/_Cfixture.so"
    echo "SO_SHA256:$SOSHA"; echo "SO_GATE_PASS"; echo "RUN_START:2026-08-14_00:00:00"
    echo "BENCH|fixture|comm_sms=24|p50=1.0000ms|p95=1.0000ms|abs_max=0.001|relative=0.001|out=x.json"
    echo "RUN_REAL_EXIT:0"; echo "RUN_END:2026-08-14_00:02:00"; } > "$M/runs/$TAG-$RID.log"
  { echo "SIDECAR_START:2026-08-14_00:00:00"; echo "RUN_ID:$RID"
    echo "BENCH_MODE:canary"; echo "FORMAL_VALIDITY:INVALID_FOR_FORMAL"
    echo "MANIFEST_FILE:$(basename "$MAN")"; echo "MANIFEST_SHA256:$MSHA"
    echo "RECEIPT_FILE:$(basename "$REC")"; echo "RECEIPT_SHA256:$RSHA"
    echo "RECEIPT_COPY:$TAG-$RID.receipt"
    echo "IMAGE_ID:$IMGID"; echo "IMAGE_REF:$IMGREF"; echo "IMAGE_REPO_DIGESTS:$IMGRD"
    echo "GPU_MAPPING(container_idx,uuid,host_idx):"
    echo "MAP:0,GPU-aaa,4"; echo "MAP:1,GPU-bbb,5"; echo "MAP:2,GPU-ccc,6"; echo "MAP:3,GPU-ddd,7"
    echo "TELEMETRY_PRELAUNCH_BEGIN:t"; echo "PRELAUNCH_OCCUPANCY:0"
    echo "HOST_LOADAVG_PRELAUNCH:1.0 1.0 1.0 1/1 1"
    echo "PRELAUNCH_LOAD1_GATE:load1=1.0 max=64 ok=1"
    echo "HOST_GPU_CLOCKS_PRELAUNCH:x"; echo "TELEMETRY_PRELAUNCH_END:t"
    echo "PROC_SHAPE:timeout=1 parent=1 workers=4"
    echo "HOST_TOP_CAPTURE:t"; echo "TOP_WORKER_HOST_PIDS:1,2,3,4 (n=4)"
    echo "PID_ATTRIBUTION_PASS"
    echo "NVML_UUID_PID_PAIRS:GPU-aaa,1;GPU-bbb,2;GPU-ccc,3;GPU-ddd,4;"
    echo "HOST_GPU_CLOCKS_RUNNING:x"; echo "RUNNING_CLOCK_LIVENESS:min=500MHz low=0"
    echo "MIDRUN_FOREIGN_SAMPLES:samples=9 hits=0"
    echo "TELEMETRY_END_BEGIN:t"; echo "END_OCCUPANCY:0"
    echo "HOST_LOADAVG_END:1.0 1.0 1.0 1/1 1"; echo "HOST_GPU_CLOCKS_END:x"
    echo "LOAD1_DELTA_GATE:start=1.0 end=1.0 max=+16 ok=1"
    echo "TELEMETRY_FINAL_PASS"; echo "SIDECAR_END:t"; } > "$M/host-runs/$TAG-$RID.host"
  python3 - "$M/runs/$TAG-$RID.json" "$JSCHEMA" "$HS" "$SOSHA" "$MSHA" "$RSHA" "$RID" "$IMPL" \
           "$PYPIN" "$EXTPIN" "$(H40 0)" <<'PY'
import json, sys
out, schema, hs, so, ms, rs, rid, impl, pypin, extpin, frozen = sys.argv[1:12]
meta = {
    "shape": {"tokens_per_rank": 512, "hidden": 256, "intermediate": 256,
              "experts": 4, "topk": 1, "world_size": 4},
    "comm_sms": 24, "minibatch": 256, "macrobatch": 4096,
    "warmup_iters": 20, "timed_iters": 100,
    "correctness_gate": {"abs_mean": 0.001, "abs_max": 0.01, "relative": 0.001,
                         "tolerance_abs_rel": [0.5, 0.01]},
    "provenance": {"harness_sha256": hs, "frozen_commit_env": frozen, "so_sha256": so,
                   "manifest_sha256_env": ms, "receipt_sha256_env": rs,
                   "bench_gpus_env": "0,1,2,3", "bench_mode_env": "canary"},
    "run_id": rid,
}
if impl == "deepep_torch":
    meta["environment_pins"] = {"torch": "2.11.0+cu130", "deepep_py_tree_sha256": pypin,
                                "deepep_ext_sha256": extpin, "sm90_compiled": True}
    meta["torch_compile"] = "on"
    meta["recv_contract"] = {"routes_total": 2048, "recv_rows": 512,
                             "per_expert_counts_match": True}
json.dump({"schema": schema, "meta": meta, "samples_ms": [1.0] * 100,
           "p50_ms": 1.0, "p95_ms": 1.0}, open(out, "w"))
PY
  echo "$RSHA"
}

vcase() { # name tag rid manifest receipt expected_rc reason-ERE
  local NAME=$1 TAG=$2 RID=$3 MAN=$4 REC=$5 WANT=$6 REASON=$7 O R
  set +e
  O=$(EXPECTED_RECEIPT_SHA256=$(sha256sum "$REC" | cut -d' ' -f1) \
      bash "$DIR/verify_run_sm90.sh" "$M" "$TAG" "$RID" "$MAN" "$REC" 2>&1)
  R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}

build_run mok_sm90 mokrun R1 >/dev/null
build_run deepep_torch deeprun R2 >/dev/null
vcase I1_mok_positive mokrun R1 "$TMPD/mokrun.manifest" "$TMPD/mokrun.receipt" 0 \
  '^VERIFY_PASS mode=canary formal_validity=INVALID_FOR_FORMAL$'
vcase I2_deepep_positive deeprun R2 "$TMPD/deeprun.manifest" "$TMPD/deeprun.receipt" 0 \
  '^VERIFY_PASS mode=canary formal_validity=INVALID_FOR_FORMAL$'

mut_json() { # rid python-expr
  python3 - "$M/runs/$1.json" "$2" <<'PY'
import json, sys
p, expr = sys.argv[1], sys.argv[2]
d = json.load(open(p))
exec(expr, {"d": d})
json.dump(d, open(p, "w"))
PY
}
build_run deepep_torch d3 R3 >/dev/null
mut_json "d3-R3" 'd["schema"]="bench-sm90-fwd.v1"'
vcase I3_wrong_json_schema d3 R3 "$TMPD/d3.manifest" "$TMPD/d3.receipt" 1 \
  '^VERIFY_FAIL:schema bench-sm90-fwd\.v1 != expected bench-deepep-fwd\.v1$'

build_run deepep_torch d4 R4 >/dev/null
grep -v '^HARNESS_MODULE:' "$M/runs/d4-R4.log" > "$TMPD/l4" && mv "$TMPD/l4" "$M/runs/d4-R4.log"
vcase I4_missing_module_line d4 R4 "$TMPD/d4.manifest" "$TMPD/d4.receipt" 1 \
  '^VERIFY_FAIL:log has no HARNESS_MODULE line but manifest IMPL=deepep_torch$'

build_run deepep_torch d5 R5 >/dev/null
sed -i 's|^HARNESS_MODULE:.*|HARNESS_MODULE:benchmarks.bench_sm90_fwd|' "$M/runs/d5-R5.log"
vcase I5_wrong_module_line d5 R5 "$TMPD/d5.manifest" "$TMPD/d5.receipt" 1 \
  '^VERIFY_FAIL:log harness module != manifest IMPL$'

build_run deepep_torch d6 R6 >/dev/null
mut_json "d6-R6" 'd["meta"]["environment_pins"]["deepep_ext_sha256"]="0"*64'
vcase I6_ext_pin_mismatch d6 R6 "$TMPD/d6.manifest" "$TMPD/d6.receipt" 1 \
  '^VERIFY_FAIL:json deepep extension sha != manifest$'

build_run deepep_torch d7 R7 >/dev/null
mut_json "d7-R7" 'd["meta"]["environment_pins"]["deepep_py_tree_sha256"]="0"*64'
vcase I7_py_pin_mismatch d7 R7 "$TMPD/d7.manifest" "$TMPD/d7.receipt" 1 \
  '^VERIFY_FAIL:json deepep python-tree sha != manifest$'

build_run deepep_torch d8 R8 >/dev/null
mut_json "d8-R8" 'd["meta"]["torch_compile"]="off"'
vcase I8_compile_mismatch d8 R8 "$TMPD/d8.manifest" "$TMPD/d8.receipt" 1 \
  '^VERIFY_FAIL:json torch_compile off != manifest on$'

build_run deepep_torch d9 R9 >/dev/null
mut_json "d9-R9" 'd["meta"].pop("recv_contract")'
vcase I9_missing_recv_contract d9 R9 "$TMPD/d9.manifest" "$TMPD/d9.receipt" 1 \
  '^VERIFY_FAIL:json does not record a passed recv-contract check$'

build_run deepep_torch d10 R10 >/dev/null
mut_json "d10-R10" 'd["meta"]["recv_contract"]["routes_total"]=1'
vcase I10_routes_shrink d10 R10 "$TMPD/d10.manifest" "$TMPD/d10.receipt" 1 \
  '^VERIFY_FAIL:recv contract routes_total < recv_rows \(route expansion cannot shrink\)$'

# schema-1 (tiny9-style) run must still verify: adding IMPL resolution must not
# disturb the contract that is already validated on GPU9
build_run mok_sm90 s1run R11 >/dev/null
grep -v '^IMPL=\|^SHAPE_ID=\|^HARNESS_MODULE=' "$TMPD/s1run.manifest" \
  | sed 's/^MANIFEST_SCHEMA=2/MANIFEST_SCHEMA=1/' > "$TMPD/s1.manifest"
chmod 444 "$TMPD/s1.manifest"
S1MSHA=$(sha256sum "$TMPD/s1.manifest" | cut -d' ' -f1)
mkreceipt "$TMPD/s1.receipt" "$S1MSHA" "$MOKH"
S1RSHA=$(sha256sum "$TMPD/s1.receipt" | cut -d' ' -f1)
# build_run already wrote read-only copies here; replace them
rm -f "$M/host-runs/s1run-R11.manifest" "$M/host-runs/s1run-R11.receipt"
cp "$TMPD/s1.manifest" "$M/host-runs/s1run-R11.manifest"
cp "$TMPD/s1.receipt" "$M/host-runs/s1run-R11.receipt"
sed -i "s|^MANIFEST_SHA256:.*|MANIFEST_SHA256:$S1MSHA|; s|^RECEIPT_SHA256:.*|RECEIPT_SHA256:$S1RSHA|" \
  "$M/runs/s1run-R11.log" "$M/host-runs/s1run-R11.host"
python3 - "$M/runs/s1run-R11.json" "$S1MSHA" "$S1RSHA" <<'PY'
import json, sys
p, ms, rs = sys.argv[1:4]
d = json.load(open(p))
d["meta"]["provenance"]["manifest_sha256_env"] = ms
d["meta"]["provenance"]["receipt_sha256_env"] = rs
json.dump(d, open(p, "w"))
PY
vcase I11_schema1_regression s1run R11 "$TMPD/s1.manifest" "$TMPD/s1.receipt" 0 \
  '^VERIFY_PASS mode=canary formal_validity=INVALID_FOR_FORMAL$'

rm -rf "$TMPD"
EXPECTED=11
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "INT_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "VERIFY_INTEGRATION pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
