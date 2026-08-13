#!/bin/bash
# Shared receipt validator (tracked). Single source of truth for receipt
# semantics, used by make_deploy_receipt.sh (self-check of its own output),
# host_launch_sm90.sh and verify_run_sm90.sh. NOTE: passing this validator
# establishes only well-formedness; TRUST is established solely by the
# EXPECTED_RECEIPT_SHA256 comparison in the caller (an attacker who rewrites
# manifest+receipt self-consistently still fails the expected-hash gate).
#   any violation -> exit 14 RECEIPT_TRUST_FAIL:<why>
#   valid         -> exit 0, prints RECEIPT_VALID:<sha256>
# Usage: validate_receipt_sm90.sh <receipt> [--check-mode]
set -uo pipefail
REC=${1:?receipt path}
CHECK_MODE=0
[ "${2:-}" = "--check-mode" ] && CHECK_MODE=1
[ -f "$REC" ] || { echo "RECEIPT_TRUST_FAIL:missing receipt $REC"; exit 14; }
if [ "$CHECK_MODE" -eq 1 ]; then
  RMODE=$(stat -c %a "$REC")
  case "$RMODE" in *[2367]*) echo "RECEIPT_TRUST_FAIL:write bits set ($RMODE)"; exit 14 ;; esac
fi
RREQ="RECEIPT_SCHEMA SOURCE_TREE_COMMIT HARNESS_COMMIT BINARY_BUILD_COMMIT MANIFEST_FILE MANIFEST_SHA256 MANIFEST_GIT_BLOB HARNESS_SHA256 SO_SHA256 IMAGE_ID IMAGE_REF IMAGE_REPO_DIGESTS"
head -1 "$REC" | grep -q '^RECEIPT_SCHEMA=1$' || { echo "RECEIPT_TRUST_FAIL:bad or missing schema version"; exit 14; }
for K in $RREQ; do
  N=$(grep -c "^$K=" "$REC" || true)
  [ "$N" -eq 1 ] || { echo "RECEIPT_TRUST_FAIL:key $K count=$N (need exactly 1)"; exit 14; }
done
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  K=${LINE%%=*}
  echo " $RREQ " | grep -q " $K " || { echo "RECEIPT_TRUST_FAIL:unknown key $K"; exit 14; }
done < "$REC"
rget() { grep "^$1=" "$REC" | head -1 | cut -d= -f2-; }
for K in $RREQ; do
  [ -n "$(rget "$K")" ] || { echo "RECEIPT_TRUST_FAIL:key $K empty"; exit 14; }
done
for K in MANIFEST_SHA256 HARNESS_SHA256 SO_SHA256; do
  rget "$K" | grep -qE '^[0-9a-f]{64}$' || { echo "RECEIPT_TRUST_FAIL:$K not 64-hex"; exit 14; }
done
for K in SOURCE_TREE_COMMIT HARNESS_COMMIT MANIFEST_GIT_BLOB; do
  rget "$K" | grep -qE '^[0-9a-f]{40}$' || { echo "RECEIPT_TRUST_FAIL:$K not 40-hex"; exit 14; }
done
BB=$(rget BINARY_BUILD_COMMIT)
{ echo "$BB" | grep -qE '^[0-9a-f]{40}$' || [ "$BB" = "UNKNOWN" ]; } \
  || { echo "RECEIPT_TRUST_FAIL:BINARY_BUILD_COMMIT not 40-hex or UNKNOWN"; exit 14; }
rget IMAGE_ID | grep -qE '^sha256:[0-9a-f]{64}$' || { echo "RECEIPT_TRUST_FAIL:IMAGE_ID malformed (want sha256:<64-hex>)"; exit 14; }
# IMAGE_REPO_DIGESTS: literal NONE (local-only image) or a comma-separated
# list of canonical repo@sha256:<64-hex> entries - a free-form string must
# not be able to stand in for registry provenance
RD=$(rget IMAGE_REPO_DIGESTS)
if [ "$RD" != "NONE" ]; then
  printf '%s\n' "$RD" | tr ',' '\n' | while IFS= read -r E; do
    echo "$E" | grep -qE '^[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[0-9a-f]{64}$' || exit 1
  done || { echo "RECEIPT_TRUST_FAIL:IMAGE_REPO_DIGESTS malformed (want NONE or repo@sha256:<64-hex> list)"; exit 14; }
fi
echo "RECEIPT_VALID:$(sha256sum "$REC" | cut -d' ' -f1)"
