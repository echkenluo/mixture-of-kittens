#!/bin/bash
# Full positive gate for a completed run (tracked). LAUNCH_VERIFIED is only
# the start gate; completion requires ALL of:
#   RUN_REAL_EXIT:0 in the run log; JSON exists at the exact path and parses;
#   JSON provenance harness_sha256 == expected and frozen_commit == expected;
#   JSON run_id == RUN_ID; sidecar exists with same RUN_ID and
#   PID_ATTRIBUTION_PASS. Prints VERIFY_PASS or VERIFY_FAIL:<reason>.
# Usage: verify_run_sm90.sh <mokdir> <tag> <run_id> <expected_sha> <frozen_commit>
set -uo pipefail
M=${1:?}; T=${2:?}; R=${3:?}; SHA=${4:?}; FC=${5:?}
LOG=$M/runs/$T-$R.log; JSON=$M/runs/$T-$R.json; SIDE=$M/host-runs/$T-$R.host
[ -f "$LOG" ] || { echo "VERIFY_FAIL:no log $LOG"; exit 1; }
grep -q "^RUN_REAL_EXIT:0$" "$LOG" || { echo "VERIFY_FAIL:exit $(grep RUN_REAL_EXIT "$LOG" | head -1)"; exit 1; }
[ -f "$JSON" ] || { echo "VERIFY_FAIL:no json $JSON"; exit 1; }
[ -f "$SIDE" ] || { echo "VERIFY_FAIL:no sidecar $SIDE"; exit 1; }
grep -q "RUN_ID:$R" "$SIDE" || { echo "VERIFY_FAIL:sidecar run_id mismatch"; exit 1; }
grep -q "PID_ATTRIBUTION_PASS" "$SIDE" || { echo "VERIFY_FAIL:no PID_ATTRIBUTION_PASS"; exit 1; }
python3 - "$JSON" "$SHA" "$FC" "$R" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
m = d["meta"]; p = m["provenance"]
checks = [
    (p["harness_sha256"] == sys.argv[2], f"harness_sha {p['harness_sha256'][:12]} != expected"),
    (p["frozen_commit_env"] == sys.argv[3], f"frozen_commit {p['frozen_commit_env'][:12]} != expected"),
    (m.get("run_id") == sys.argv[4], f"json run_id {m.get('run_id')} != {sys.argv[4]}"),
    (len(d["samples_ms"]) == m["timed_iters"], "sample count mismatch"),
]
for ok, msg in checks:
    if not ok:
        print(f"VERIFY_FAIL:{msg}"); sys.exit(1)
print("VERIFY_JSON_OK")
PY
echo "VERIFY_PASS"
