#!/bin/bash
# Shared manifest validator (tracked). Single source of truth for manifest
# semantics, used by host_launch_sm90.sh, verify_run_sm90.sh and the negative
# suite's validate-only surface (schema mutations never enter the launcher -
# there is no untrusted-manifest bypass anywhere).
#   schema violation                          -> exit 12 MANIFEST_SCHEMA_FAIL:<why>
#   drift vs --harness / --so-dir             -> exit 13 {HARNESS,SO}_DRIFT_FAIL...
#   valid                                     -> exit 0, prints MANIFEST_VALID:<sha256>
# Usage: validate_manifest_sm90.sh <manifest> [--harness <file>] [--so-dir <dir>]
set -uo pipefail
MAN=${1:?manifest path}
shift
HARNESS=""; SODIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --harness) HARNESS=$2; shift 2 ;;
    --so-dir)  SODIR=$2;   shift 2 ;;
    *) echo "MANIFEST_SCHEMA_FAIL:unknown option $1"; exit 12 ;;
  esac
done
[ -f "$MAN" ] || { echo "MANIFEST_SCHEMA_FAIL:missing $MAN"; exit 12; }
REQ_KEYS="MANIFEST_SCHEMA FROZEN_COMMIT EXPECTED_SO_SHA256 EXPECTED_HARNESS_SHA256 BENCH_GPUS TIMING_SEMANTICS tokens_per_rank hidden intermediate experts topk world_size comm_sms minibatch macrobatch warmup_iters timed_iters"
INT_KEYS="tokens_per_rank hidden intermediate experts topk world_size comm_sms minibatch macrobatch warmup_iters timed_iters"
head -1 "$MAN" | grep -q '^MANIFEST_SCHEMA=1$' || { echo "MANIFEST_SCHEMA_FAIL:bad or missing schema version"; exit 12; }
for K in $REQ_KEYS; do
  N=$(grep -c "^$K=" "$MAN" || true)
  [ "$N" -eq 1 ] || { echo "MANIFEST_SCHEMA_FAIL:key $K count=$N (need exactly 1)"; exit 12; }
done
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  K=${LINE%%=*}
  echo " $REQ_KEYS " | grep -q " $K " || { echo "MANIFEST_SCHEMA_FAIL:unknown key $K"; exit 12; }
done < "$MAN"
mget() { grep "^$1=" "$MAN" | head -1 | cut -d= -f2-; }
for K in $REQ_KEYS; do
  [ -n "$(mget "$K")" ] || { echo "MANIFEST_SCHEMA_FAIL:key $K empty"; exit 12; }
done
for K in EXPECTED_SO_SHA256 EXPECTED_HARNESS_SHA256; do
  mget "$K" | grep -qE '^[0-9a-f]{64}$' || { echo "MANIFEST_SCHEMA_FAIL:$K not 64-hex"; exit 12; }
done
for K in $INT_KEYS; do
  V=$(mget "$K")
  echo "$V" | grep -qE '^[0-9]+$' && [ "$V" -gt 0 ] || { echo "MANIFEST_SCHEMA_FAIL:$K not positive int"; exit 12; }
done
WS=$(mget world_size)
[ "$WS" = "4" ] || { echo "MANIFEST_SCHEMA_FAIL:world_size must be 4"; exit 12; }
GIDS=$(mget BENCH_GPUS | tr ',' '\n')
NGID=$(printf '%s\n' "$GIDS" | grep -cE '^[0-9]+$' || true)
NLINES=$(printf '%s\n' "$GIDS" | grep -c . || true)
[ "$NGID" -eq "$NLINES" ] || { echo "MANIFEST_SCHEMA_FAIL:BENCH_GPUS non-numeric id"; exit 12; }
NUNIQ=$(printf '%s\n' "$GIDS" | sort -u | grep -c . || true)
[ "$NUNIQ" -eq "$NGID" ] || { echo "MANIFEST_SCHEMA_FAIL:BENCH_GPUS ids not unique"; exit 12; }
[ "$NGID" -eq "$WS" ] || { echo "MANIFEST_SCHEMA_FAIL:BENCH_GPUS count $NGID != world_size $WS"; exit 12; }
if [ -n "$HARNESS" ]; then
  AH=$(sha256sum "$HARNESS" 2>/dev/null | cut -d' ' -f1)
  [ "$AH" = "$(mget EXPECTED_HARNESS_SHA256)" ] || { echo "HARNESS_DRIFT_FAIL expected=$(mget EXPECTED_HARNESS_SHA256) actual=${AH:-unreadable}"; exit 13; }
fi
if [ -n "$SODIR" ]; then
  SOG=("$SODIR"/_C*.so)
  { [ "${#SOG[@]}" -eq 1 ] && [ -f "${SOG[0]}" ]; } || { echo "SO_DRIFT_FAIL:need exactly one _C*.so in $SODIR"; exit 13; }
  ASO=$(sha256sum "${SOG[0]}" | cut -d' ' -f1)
  [ "$ASO" = "$(mget EXPECTED_SO_SHA256)" ] || { echo "SO_DRIFT_FAIL expected=$(mget EXPECTED_SO_SHA256) actual=$ASO"; exit 13; }
fi
echo "MANIFEST_VALID:$(sha256sum "$MAN" | cut -d' ' -f1)"
