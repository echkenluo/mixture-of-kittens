#!/bin/bash
# Fail-closed P0 sequence (tracked). The negative suite MUST exit 0 before any
# positive tiny run is attempted: a failing rejection suite invalidates the
# chain, and running the positive anyway would produce an unusable result
# (this replaces the a1a4db6-era driver, which wrongly continued after
# NEG_EXIT:1). After a verified positive run it exercises two post-hoc
# negatives on the real artifacts (per-run copy tamper, config drift).
# Usage: p0_sequence_sm90.sh <container> <host_mok_dir> <receipt> <tag>
set -uo pipefail
CT=${1:?}; MOKDIR=${2:?}; RECEIPT=${3:?}; TAG=${4:?}
DIR=$(cd "$(dirname "$0")" && pwd)
MAN=$DIR/manifests/tiny-h20-v1.manifest
echo "P0SEQ_START:$(date -u +%F_%T)"
sha256sum "$DIR/bench_sm90_fwd.py" "$MAN" "$RECEIPT"

bash "$DIR/test_runner_negatives.sh" "$CT" "$MOKDIR" "$RECEIPT"
NRC=$?
echo "NEG_EXIT:$NRC"
if [ "$NRC" -ne 0 ]; then
  echo "P0SEQ_FAIL_CLOSED:negative suite rc=$NRC, positive run FORBIDDEN"
  echo "P0SEQ_END:$(date -u +%F_%T)"
  exit 1
fi

BENCH_TAG=$TAG bash "$DIR/host_launch_sm90.sh" "$CT" "$MOKDIR" "$MAN" "$RECEIPT" > "$MOKDIR/$TAG-launch.out" 2>&1
LRC=$?
cat "$MOKDIR/$TAG-launch.out"
echo "LAUNCH_EXIT:$LRC"
RID=$(grep -ao 'run_id=[0-9TZ-]*' "$MOKDIR/$TAG-launch.out" | tail -1 | cut -d= -f2)
echo "RID:${RID:-none}"
if [ "$LRC" -ne 0 ] || [ -z "$RID" ]; then
  echo "P0SEQ_FAIL_CLOSED:launch rc=$LRC"
  echo "P0SEQ_END:$(date -u +%F_%T)"
  exit 1
fi

bash "$DIR/verify_run_sm90.sh" "$MOKDIR" "$TAG" "$RID" "$MAN" "$RECEIPT" "$CT"
VRC=$?
echo "VERIFY_EXIT:$VRC"
PHFAIL=0
if [ "$VRC" -eq 0 ]; then
  CP=$MOKDIR/host-runs/$TAG-$RID.manifest
  cp "$CP" "$CP.orig"
  chmod u+w "$CP" && echo "TAMPER=1" >> "$CP"
  if bash "$DIR/verify_run_sm90.sh" "$MOKDIR" "$TAG" "$RID" "$MAN" "$RECEIPT" "$CT" >/dev/null 2>&1; then
    echo "POSTHOC_tamper_FAIL"; PHFAIL=1
  else
    echo "POSTHOC_tamper_PASS"
  fi
  mv -f "$CP.orig" "$CP"
  sed 's/^hidden=.*/hidden=999/' "$MAN" > "/tmp/drift-$TAG.manifest"
  if bash "$DIR/verify_run_sm90.sh" "$MOKDIR" "$TAG" "$RID" "/tmp/drift-$TAG.manifest" "$RECEIPT" "$CT" >/dev/null 2>&1; then
    echo "POSTHOC_config_drift_FAIL"; PHFAIL=1
  else
    echo "POSTHOC_config_drift_PASS"
  fi
  rm -f "/tmp/drift-$TAG.manifest"
fi
echo "P0SEQ_END:$(date -u +%F_%T)"
[ "$VRC" -eq 0 ] && [ "$PHFAIL" -eq 0 ]
