#!/bin/bash
# Fail-closed P0 sequence v2 (tracked). The negative suite MUST exit 0 before
# any positive tiny run is attempted: a failing rejection suite invalidates
# the chain, and running the positive anyway produces an unusable result
# (this replaces the a1a4db6-era driver, which wrongly continued after
# NEG_EXIT:1). After a verified positive run it exercises post-hoc negatives
# on SHADOW COPIES of the real artifacts - originals are never mutated - and
# each post-hoc case asserts an exact VERIFY_FAIL reason line, not merely a
# nonzero exit.
# Usage: EXPECTED_RECEIPT_SHA256=... \
#          p0_sequence_sm90.sh <container> <host_mok_dir> <receipt> <tag>
set -uo pipefail
CT=${1:?}; MOKDIR=${2:?}; RECEIPT=${3:?}; TAG=${4:?}
DIR=$(cd "$(dirname "$0")" && pwd)
MAN=$DIR/manifests/tiny-h20-v1.manifest
EXPR_SHA=${EXPECTED_RECEIPT_SHA256:?EXPECTED_RECEIPT_SHA256 required (out-of-band prior from the packaging host)}
export EXPECTED_RECEIPT_SHA256="$EXPR_SHA"
PHPASS=0; PHFAIL=0
phreport() { if [ "$2" -eq 0 ]; then echo "POSTHOC_$1_PASS"; PHPASS=$((PHPASS+1)); else echo "POSTHOC_$1_FAIL"; PHFAIL=$((PHFAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }

echo "P0SEQ_START:$(date -u +%F_%T)"
sha256sum "$DIR/bench_sm90_fwd.py" "$MAN" "$RECEIPT"
echo "EXPECTED_RECEIPT_SHA256:$EXPR_SHA"

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
if [ "$VRC" -ne 0 ]; then
  echo "P0SEQ_FAIL_CLOSED:verify rc=$VRC"
  echo "P0SEQ_END:$(date -u +%F_%T)"
  exit 1
fi

# ---- post-hoc negatives on shadow copies (originals untouched) ----
SHADOW=$(mktemp -d)
ln -s "$MOKDIR/mixture-of-kittens" "$SHADOW/mixture-of-kittens"
mkdir -p "$SHADOW/runs" "$SHADOW/host-runs"
cp "$MOKDIR/runs/$TAG-$RID.log" "$MOKDIR/runs/$TAG-$RID.json" "$SHADOW/runs/"
cp "$MOKDIR/host-runs/$TAG-$RID.host" "$MOKDIR/host-runs/$TAG-$RID.manifest" \
   "$MOKDIR/host-runs/$TAG-$RID.receipt" "$SHADOW/host-runs/"
chmod u+w "$SHADOW/runs/"* "$SHADOW/host-runs/"*
SMAN=$SHADOW/host-runs/$TAG-$RID.manifest
SREC=$SHADOW/host-runs/$TAG-$RID.receipt
SSIDE=$SHADOW/host-runs/$TAG-$RID.host
SLOG=$SHADOW/runs/$TAG-$RID.log
SJSON=$SHADOW/runs/$TAG-$RID.json
# baseline: the shadow itself must still verify (proves later failures are
# caused by the mutation, not by the shadow layout)
set +e
OUT=$(bash "$DIR/verify_run_sm90.sh" "$SHADOW" "$TAG" "$RID" "$MAN" "$RECEIPT" "$CT" 2>&1); RC=$?
set -u
[ "$RC" -eq 0 ] && has1 "$OUT" '^VERIFY_PASS'; phreport 0_shadow_baseline $?
echo "  shadow_baseline rc=$RC want=0(VERIFY_PASS)"

ph() { # name mutate-cmd expected-reason-ERE
  local NAME=$1 MUT=$2 REASON=$3
  cp "$MOKDIR/runs/$TAG-$RID.log" "$SLOG"; cp "$MOKDIR/runs/$TAG-$RID.json" "$SJSON"
  cp "$MOKDIR/host-runs/$TAG-$RID.host" "$SSIDE"
  cp "$MOKDIR/host-runs/$TAG-$RID.manifest" "$SMAN"
  cp "$MOKDIR/host-runs/$TAG-$RID.receipt" "$SREC"
  chmod u+w "$SLOG" "$SJSON" "$SSIDE" "$SMAN" "$SREC"
  eval "$MUT"
  set +e
  local O R
  O=$(bash "$DIR/verify_run_sm90.sh" "$SHADOW" "$TAG" "$RID" "${PHMAN:-$MAN}" "${PHREC:-$RECEIPT}" "$CT" 2>&1); R=$?
  set -u
  [ "$R" -ne 0 ] && has1 "$O" "$REASON"; phreport "$NAME" $?
  echo "  $NAME rc=$R want=nonzero($REASON)"
  unset PHMAN PHREC
}
ph 1_manifest_copy_tamper 'echo "TAMPER=1" >> "$SMAN"' '^VERIFY_FAIL:per-run copy failed shared validator$'
ph 2_receipt_copy_tamper  'echo "TAMPER=1" >> "$SREC"' '^VERIFY_FAIL:receipt copy tampered'
ph 3_sidecar_manifest_sha 'sed -i "s|^MANIFEST_SHA256:.*|MANIFEST_SHA256:$(printf "a%.0s" $(seq 1 64))|" "$SSIDE"' '^VERIFY_FAIL:sidecar pre-start manifest hash != copy$'
ph 4_sidecar_receipt_sha  'sed -i "s|^RECEIPT_SHA256:.*|RECEIPT_SHA256:$(printf "a%.0s" $(seq 1 64))|" "$SSIDE"' '^VERIFY_FAIL:sidecar receipt sha mismatch$'
ph 5_log_manifest_sha     'sed -i "s|^MANIFEST_SHA256:.*|MANIFEST_SHA256:$(printf "b%.0s" $(seq 1 64))|" "$SLOG"' '^VERIFY_FAIL:runner log manifest hash != copy$'
ph 6_log_receipt_sha      'sed -i "s|^RECEIPT_SHA256:.*|RECEIPT_SHA256:$(printf "b%.0s" $(seq 1 64))|" "$SLOG"' '^VERIFY_FAIL:runner log receipt sha mismatch$'
ph 7_json_manifest_sha    'python3 -c "import json,sys;p=sys.argv[1];d=json.load(open(p));d[\"meta\"][\"provenance\"][\"manifest_sha256_env\"]=\"c\"*64;json.dump(d,open(p,\"w\"))" "$SJSON"' '^VERIFY_FAIL:json manifest sha != given manifest \(exact\)$'
ph 8_json_receipt_sha     'python3 -c "import json,sys;p=sys.argv[1];d=json.load(open(p));d[\"meta\"][\"provenance\"][\"receipt_sha256_env\"]=\"c\"*64;json.dump(d,open(p,\"w\"))" "$SJSON"' '^VERIFY_FAIL:json receipt sha != expected \(exact\)$'
ph 9_duplicate_run_end    'echo "RUN_END:duplicate" >> "$SLOG"' "^VERIFY_FAIL:need exactly one '\^RUN_END:' line"
# config drift: unique temp manifest (no tag collision)
DRIFT=$(mktemp)
sed 's/^hidden=.*/hidden=999/' "$MAN" > "$DRIFT"
PHMAN=$DRIFT ph 10_config_drift ':' '^VERIFY_FAIL:given manifest sha != receipt$'
rm -f "$DRIFT"
# collusion: manifest AND receipt rewritten self-consistently must still fail
# at the out-of-band expected-receipt gate
CMAN=$(mktemp); CREC=$(mktemp)
sed 's/^hidden=.*/hidden=999/' "$MAN" > "$CMAN"
CMSHA=$(sha256sum "$CMAN" | cut -d' ' -f1)
sed "s|^MANIFEST_SHA256=.*|MANIFEST_SHA256=$CMSHA|" "$RECEIPT" > "$CREC"
PHMAN=$CMAN PHREC=$CREC ph 11_collusion ':' '^VERIFY_FAIL:EXPECTED_RECEIPT_SHA256 mismatch \(given receipt\)$'
rm -f "$CMAN" "$CREC"
rm -rf "$SHADOW"

echo "POSTHOC pass=$PHPASS fail=$PHFAIL"
echo "P0SEQ_END:$(date -u +%F_%T)"
[ "$PHFAIL" -eq 0 ] && [ "$PHPASS" -eq 12 ]
