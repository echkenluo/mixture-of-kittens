#!/bin/bash
# Shared receipt validator (tracked). Single source of truth for receipt
# semantics, used by make_deploy_receipt.sh (self-check of its own output),
# host_launch_sm90.sh and verify_run_sm90.sh. NOTE: passing this validator
# establishes only well-formedness; TRUST is established solely by the
# EXPECTED_RECEIPT_SHA256 comparison in the caller (an attacker who rewrites
# manifest+receipt self-consistently still fails the expected-hash gate).
# MANIFEST_GIT_BLOB is ANCHORED METADATA, not a binding: the verified end has
# no git, so nothing recomputes it. It is covered by the receipt hash and is
# useful for auditing on the packaging host only.
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
# schema 1: canary receipt, no build provenance - the contract the tiny9 chain
#           runs under; semantics frozen, do not extend
# schema 2: carries BUILD_RECORD_SHA256, binding this receipt to a build record
#           produced on the build host. A schema-2 receipt asserts provenance,
#           so BINARY_BUILD_COMMIT=UNKNOWN is a contradiction and is refused.
RSCHEMA=$(head -1 "$REC" | sed -n 's/^RECEIPT_SCHEMA=//p')
case "$RSCHEMA" in
  1) : ;;
  2) RREQ="$RREQ BUILD_RECORD_SHA256" ;;
  *) echo "RECEIPT_TRUST_FAIL:bad or missing schema version"; exit 14 ;;
esac
for K in $RREQ; do
  N=$(grep -c "^$K=" "$REC" || true)
  [ "$N" -eq 1 ] || { echo "RECEIPT_TRUST_FAIL:key $K count=$N (need exactly 1)"; exit 14; }
done
# exact string comparison, not a grep: the key comes from the receipt, so used
# as a regex a smuggled key like SO_SHA25. matches SO_SHA256 in the allow-list
# and passes as known (the same defect the build-record validator had)
known_key() { local N; for N in $RREQ; do [ "$N" = "$1" ] && return 0; done; return 1; }
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  K=${LINE%%=*}
  known_key "$K" || { echo "RECEIPT_TRUST_FAIL:unknown key $K"; exit 14; }
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
# the generator packages ONE tree, so these are the same HEAD by construction;
# allowing them to differ left a field that looks like a binding and is not
[ "$(rget SOURCE_TREE_COMMIT)" = "$(rget HARNESS_COMMIT)" ] \
  || { echo "RECEIPT_TRUST_FAIL:SOURCE_TREE_COMMIT != HARNESS_COMMIT (one packaged tree has one HEAD)"; exit 14; }
# MANIFEST_FILE is a bare filename by generator contract (basename of the
# packaged manifest); a path here would be a second, unanchored locator
case "$(rget MANIFEST_FILE)" in
  */*|..|.) echo "RECEIPT_TRUST_FAIL:MANIFEST_FILE must be a bare filename, got $(rget MANIFEST_FILE)"; exit 14 ;;
esac
BB=$(rget BINARY_BUILD_COMMIT)
{ echo "$BB" | grep -qE '^[0-9a-f]{40}$' || [ "$BB" = "UNKNOWN" ]; } \
  || { echo "RECEIPT_TRUST_FAIL:BINARY_BUILD_COMMIT not 40-hex or UNKNOWN"; exit 14; }
if [ "$RSCHEMA" = "2" ]; then
  rget BUILD_RECORD_SHA256 | grep -qE '^[0-9a-f]{64}$' \
    || { echo "RECEIPT_TRUST_FAIL:BUILD_RECORD_SHA256 not 64-hex"; exit 14; }
  [ "$BB" != "UNKNOWN" ] \
    || { echo "RECEIPT_TRUST_FAIL:schema 2 receipt with BINARY_BUILD_COMMIT=UNKNOWN (a record-bound receipt cannot disclaim its build)"; exit 14; }
fi
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
