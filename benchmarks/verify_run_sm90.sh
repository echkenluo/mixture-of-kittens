#!/bin/bash
# Full positive gate for a completed run (tracked). LAUNCH_VERIFIED is only
# the start gate; completion requires ALL of:
#   RUN_REAL_EXIT:0 in the run log; JSON exists at the exact path and parses;
#   JSON provenance harness_sha256 == expected and frozen_commit == expected;
#   JSON run_id == RUN_ID; sidecar exists with same RUN_ID and
#   PID_ATTRIBUTION_PASS. Prints VERIFY_PASS or VERIFY_FAIL:<reason>.
# Usage: verify_run_sm90.sh <mokdir> <tag> <run_id> <expected_sha> <frozen_commit> [container] [manifest]
# manifest: KEY=VALUE lines checked hard against log/JSON (EXPECTED_SO_MD5,
# tokens_per_rank, hidden, intermediate, experts, topk, world_size, comm_sms,
# minibatch, macrobatch, warmup_iters, timed_iters)
set -uo pipefail
M=${1:?}; T=${2:?}; R=${3:?}; SHA=${4:?}; FC=${5:?}; CT=${6:-}; MAN=${7:-}
LOG=$M/runs/$T-$R.log; JSON=$M/runs/$T-$R.json; SIDE=$M/host-runs/$T-$R.host
[ -f "$LOG" ] || { echo "VERIFY_FAIL:no log $LOG"; exit 1; }
grep -q "^RUN_ID:$R$" "$LOG" || { echo "VERIFY_FAIL:log run_id mismatch"; exit 1; }
grep -q "^HARNESS_SHA256:$SHA$" "$LOG" || { echo "VERIFY_FAIL:log harness sha mismatch"; exit 1; }
[ "$(grep -c '^RUN_REAL_EXIT:0$' "$LOG")" -eq 1 ] || { echo "VERIFY_FAIL:need exactly one RUN_REAL_EXIT:0, got $(grep RUN_REAL_EXIT "$LOG" | head -1)"; exit 1; }
[ "$(grep -c '^RUN_END:' "$LOG")" -eq 1 ] || { echo "VERIFY_FAIL:need exactly one RUN_END"; exit 1; }
[ -f "$JSON" ] || { echo "VERIFY_FAIL:no json $JSON"; exit 1; }
[ -f "$SIDE" ] || { echo "VERIFY_FAIL:no sidecar $SIDE"; exit 1; }
grep -q "^RUN_ID:$R$" "$SIDE" || { echo "VERIFY_FAIL:sidecar run_id mismatch"; exit 1; }
[ "$(grep -c '^MAP:' "$SIDE")" -eq 4 ] || { echo "VERIFY_FAIL:need exactly 4 MAP lines"; exit 1; }
NPAIR=$(grep '^NVML_UUID_PID_PAIRS:' "$SIDE" | head -1 | tr ';' '\n' | grep -c 'GPU-' || true)
[ "$NPAIR" -eq 4 ] || { echo "VERIFY_FAIL:need 4 nvml uuid,pid pairs, got $NPAIR"; exit 1; }
grep -q "^PID_ATTRIBUTION_PASS$" "$SIDE" || { echo "VERIFY_FAIL:no PID_ATTRIBUTION_PASS"; exit 1; }
for A in RUN_START HASH_GATE_PASS PREFLIGHT_PASS SO_GATE_PASS; do
  grep -q "^$A" "$LOG" || { echo "VERIFY_FAIL:missing anchor $A"; exit 1; }
done
[ "$(grep -c '^BENCH|' "$LOG")" -eq 1 ] || { echo "VERIFY_FAIL:need exactly one BENCH line"; exit 1; }
if [ -n "$MAN" ]; then
  [ -f "$MAN" ] || { echo "VERIFY_FAIL:manifest $MAN missing"; exit 1; }
  EXPSO=$(grep '^EXPECTED_SO_MD5=' "$MAN" | cut -d= -f2)
  if [ -n "$EXPSO" ]; then
    grep -q "^SO_MD5:$EXPSO$" "$LOG" || { echo "VERIFY_FAIL:log SO_MD5 != manifest"; exit 1; }
  fi
fi
python3 - "$JSON" "$SHA" "$FC" "$R" "$MAN" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
m = d["meta"]; p = m["provenance"]
checks = [
    (p["harness_sha256"] == sys.argv[2], f"harness_sha {p['harness_sha256'][:12]} != expected"),
    (p["frozen_commit_env"] == sys.argv[3], f"frozen_commit {p['frozen_commit_env'][:12]} != expected"),
    (m.get("run_id") == sys.argv[4], f"json run_id {m.get('run_id')} != {sys.argv[4]}"),
    (len(d["samples_ms"]) == m["timed_iters"], "sample count mismatch"),
    (d.get("schema") == "bench-sm90-fwd.v1", "schema mismatch"),
]
import statistics
o = sorted(d["samples_ms"])
rp50 = round(statistics.median(o), 4)
rp95 = round(o[max(0, int(len(o) * 0.95) - 1)], 4)
checks.append((abs(rp50 - d["p50_ms"]) < 1e-3, f"p50 recompute {rp50} != {d['p50_ms']}"))
checks.append((abs(rp95 - d["p95_ms"]) < 1e-3, f"p95 recompute {rp95} != {d['p95_ms']}"))
import math
checks.append((all(math.isfinite(x) and x > 0 for x in d["samples_ms"]), "samples not all finite>0"))
cg = m["correctness_gate"]
ta, tr = cg["tolerance_abs_rel"]
checks.append((math.isfinite(cg["abs_max"]) and cg["abs_max"] <= ta, "correctness abs_max not below tolerance"))
checks.append((math.isfinite(cg["relative"]) and cg["relative"] <= tr, "correctness relative not below tolerance"))
if len(sys.argv) > 5 and sys.argv[5]:
    man = dict(l.strip().split("=", 1) for l in open(sys.argv[5]) if "=" in l)
    sh = m["shape"]
    for k, loc in (("tokens_per_rank", sh), ("hidden", sh), ("intermediate", sh),
                   ("experts", sh), ("topk", sh), ("world_size", sh),
                   ("comm_sms", m), ("minibatch", m), ("macrobatch", m),
                   ("warmup_iters", m), ("timed_iters", m)):
        if k in man:
            checks.append((int(man[k]) == int(loc[k]), f"manifest {k}={man[k]} != json {loc[k]}"))
    if "EXPECTED_SO_MD5" in man:
        checks.append((m["provenance"]["so_md5"] == man["EXPECTED_SO_MD5"],
                       "json so_md5 != manifest"))
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
  [ "$REL" -eq 1 ] || { echo "VERIFY_FAIL:lock not released after RUN_END"; exit 1; }
fi
echo "VERIFY_PASS"
