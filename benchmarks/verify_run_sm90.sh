#!/bin/bash
# Completion gate v4 (tracked). Manifest-anchored four-way reconciliation:
# manifest <-> run log <-> JSON <-> sidecar, with post-hoc tamper detection
# (the sidecar's pre-start MANIFEST_SHA256 must match both the per-run copy
# and the given committed manifest). Prints VERIFY_PASS or VERIFY_FAIL:<why>.
# Usage: verify_run_sm90.sh <mokdir> <tag> <run_id> <manifest> [container]
set -uo pipefail
M=${1:?}; T=${2:?}; R=${3:?}; MAN=${4:?}; CT=${5:-}
LOG=$M/runs/$T-$R.log; JSON=$M/runs/$T-$R.json
SIDE=$M/host-runs/$T-$R.host; MCOPY=$M/host-runs/$T-$R.manifest
vf() { echo "VERIFY_FAIL:$1"; exit 1; }
[ -f "$MAN" ] || vf "manifest $MAN missing"
[ -f "$MCOPY" ] || vf "per-run manifest copy missing"
[ -f "$LOG" ] || vf "no log $LOG"
[ -f "$JSON" ] || vf "no json $JSON"
[ -f "$SIDE" ] || vf "no sidecar $SIDE"
MS_GIVEN=$(sha256sum "$MAN" | cut -d' ' -f1)
MS_COPY=$(sha256sum "$MCOPY" | cut -d' ' -f1)
MS_SIDE=$(grep '^MANIFEST_SHA256:' "$SIDE" | head -1 | cut -d: -f2)
[ "$MS_GIVEN" = "$MS_COPY" ] || vf "manifest copy tampered (given $MS_GIVEN != copy $MS_COPY)"
[ "$MS_SIDE" = "$MS_COPY" ] || vf "sidecar pre-start manifest hash != copy"
mget() { grep "^$1=" "$MAN" | head -1 | cut -d= -f2-; }
SHA=$(mget EXPECTED_HARNESS_SHA256); EXPSO=$(mget EXPECTED_SO_SHA256); FC=$(mget FROZEN_COMMIT)
grep -q "^RUN_ID:$R$" "$LOG" || vf "log run_id mismatch"
grep -q "^HARNESS_SHA256:$SHA$" "$LOG" || vf "log harness sha != manifest"
grep -q "^SO_SHA256:$EXPSO$" "$LOG" || vf "log so sha != manifest"
[ "$(grep -c '^RUN_REAL_EXIT:0$' "$LOG")" -eq 1 ] || vf "need exactly one RUN_REAL_EXIT:0, got: $(grep RUN_REAL_EXIT "$LOG" | head -1)"
[ "$(grep -c '^RUN_END:' "$LOG")" -eq 1 ] || vf "need exactly one RUN_END"
for A in RUN_START HASH_GATE_PASS PREFLIGHT_PASS SO_GATE_PASS; do
  grep -q "^$A" "$LOG" || vf "missing anchor $A"
done
[ "$(grep -c '^BENCH|' "$LOG")" -eq 1 ] || vf "need exactly one BENCH line"
grep -q "^RUN_ID:$R$" "$SIDE" || vf "sidecar run_id mismatch"
[ "$(grep -c '^MAP:' "$SIDE")" -eq 4 ] || vf "need exactly 4 MAP lines"
NPAIR=$(grep '^NVML_UUID_PID_PAIRS:' "$SIDE" | head -1 | tr ';' '\n' | grep -c 'GPU-' || true)
[ "$NPAIR" -eq 4 ] || vf "need 4 nvml uuid,pid pairs, got $NPAIR"
grep -q "^PID_ATTRIBUTION_PASS$" "$SIDE" || vf "no PID_ATTRIBUTION_PASS"
python3 - "$JSON" "$SHA" "$FC" "$R" "$MAN" "$EXPSO" <<'PY' || exit 1
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
echo "VERIFY_PASS"
