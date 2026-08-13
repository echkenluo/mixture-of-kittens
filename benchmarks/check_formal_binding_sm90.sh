#!/bin/bash
# Formal-binding checker (tracked). Verifies the complete chain a formal run
# would need, in one place, so the chain can be reviewed and tested before
# anything is allowed to depend on it.
#
# NOT WIRED INTO THE LAUNCHER YET, ON PURPOSE. host_launch_sm90.sh still
# refuses formal mode unconditionally. Opening formal requires (a) a real new
# build that produces a record on the build host, and (b) an independent
# review of this chain at runtime. Until both exist, this script is contract
# code with local tests only.
#
# The six-way agreement on the binary, and what each link rules out:
#   1 receipt bytes == EXPECTED_RECEIPT_SHA256 (the single out-of-band prior)
#       -> a receipt swapped on the verified end, and the only trust root
#   2 receipt.BUILD_RECORD_SHA256              == record bytes
#       -> a receipt pointing at a different record
#   3 receipt.BINARY_BUILD_COMMIT              == record.SOURCE_COMMIT
#       -> a receipt naming a commit the build never used
#   4 receipt.SO_SHA256                        == record.SO_SHA256
#       -> a receipt describing a different binary than the one built
#   5 manifest.EXPECTED_SO_SHA256              == record.SO_SHA256
#       -> a measurement contract frozen against a different binary
#   6 deployed .so bytes                       == record.SO_SHA256
#       -> the binary on the target not being the one that was built
#
#   any violation -> exit 18 FORMAL_BINDING_FAIL:<why>
#   all agree     -> exit 0, prints FORMAL_BINDING_PASS:<so_sha256>
# Usage: EXPECTED_RECEIPT_SHA256=... [EXPECTED_BUILD_RECORD_SHA256=...] \
#          check_formal_binding_sm90.sh <receipt> <record> <manifest> <so_dir>
set -uo pipefail
RCPT=${1:?receipt}; REC=${2:?build record}; MAN=${3:?manifest}; SODIR=${4:?dir holding the deployed _C*.so}
DIR=$(cd "$(dirname "$0")" && pwd)
fb() { echo "FORMAL_BINDING_FAIL:$1"; exit 18; }
# THE out-of-band prior is the receipt hash - the same anchor the rest of the
# chain already uses. The record is reached THROUGH the anchored receipt, so a
# second independent root cannot be substituted for it. An optional
# EXPECTED_BUILD_RECORD_SHA256 may be supplied as a cross-check, never as the
# root.
EXPR_RCPT=${EXPECTED_RECEIPT_SHA256:-}
echo "$EXPR_RCPT" | grep -qE '^[0-9a-f]{64}$' \
  || fb "EXPECTED_RECEIPT_SHA256 env missing or not 64-hex"
[ -f "$RCPT" ] || fb "receipt $RCPT missing"
[ -f "$REC" ] || fb "build record $REC missing"
[ -f "$MAN" ] || fb "manifest $MAN missing"
RCPTSHA=$(sha256sum "$RCPT" | cut -d' ' -f1)
[ "$RCPTSHA" = "$EXPR_RCPT" ] || fb "receipt sha $RCPTSHA != EXPECTED_RECEIPT_SHA256 $EXPR_RCPT"
bash "$DIR/validate_receipt_sm90.sh" "$RCPT" --check-mode >/dev/null \
  || fb "receipt failed shared validator"
bash "$DIR/validate_build_record_sm90.sh" "$REC" --check-mode >/dev/null \
  || fb "build record failed shared validator"
bash "$DIR/validate_manifest_sm90.sh" "$MAN" >/dev/null \
  || fb "manifest failed shared validator"
RSHA=$(sha256sum "$REC" | cut -d' ' -f1)
if [ -n "${EXPECTED_BUILD_RECORD_SHA256:-}" ]; then
  [ "$RSHA" = "$EXPECTED_BUILD_RECORD_SHA256" ] \
    || fb "record sha $RSHA != optional cross-check EXPECTED_BUILD_RECORD_SHA256 $EXPECTED_BUILD_RECORD_SHA256"
fi
brget() { grep "^$1=" "$REC"  | head -1 | cut -d= -f2- || true; }
# a fixture record describes an arbitrary test command, not a build of this
# source; it can never satisfy formal provenance
[ "$(brget RECORD_MODE)" = "production" ] \
  || fb "build record RECORD_MODE=$(brget RECORD_MODE) is not a production record"
# formal requires registry provenance for the toolchain image; NONE marks a
# local-only image and is canary/fixture territory
[ "$(brget TOOLCHAIN_IMAGE_REPO_DIGESTS_ATTESTED)" != "NONE" ] \
  || fb "toolchain image has no repo digest (NONE); formal requires registry provenance"
rcget() { grep "^$1=" "$RCPT" | head -1 | cut -d= -f2- || true; }
mnget() { grep "^$1=" "$MAN"  | head -1 | cut -d= -f2- || true; }
[ "$(rcget RECEIPT_SCHEMA)" = "2" ] \
  || fb "receipt schema $(rcget RECEIPT_SCHEMA) cannot carry a build-record binding (need 2)"
[ "$(rcget BUILD_RECORD_SHA256)" = "$RSHA" ] \
  || fb "receipt BUILD_RECORD_SHA256 $(rcget BUILD_RECORD_SHA256) != record $RSHA"
[ "$(rcget BINARY_BUILD_COMMIT)" = "$(brget SOURCE_COMMIT)" ] \
  || fb "receipt BINARY_BUILD_COMMIT $(rcget BINARY_BUILD_COMMIT) != record SOURCE_COMMIT $(brget SOURCE_COMMIT)"
[ "$(rcget SO_SHA256)" = "$(brget SO_SHA256)" ] \
  || fb "receipt SO_SHA256 != record SO_SHA256"
[ "$(mnget EXPECTED_SO_SHA256)" = "$(brget SO_SHA256)" ] \
  || fb "manifest EXPECTED_SO_SHA256 != record SO_SHA256"
# an unmatched glob expands to the literal pattern, so count real files -
# reporting "found 1" for an empty directory would be a misleading diagnosis
SOG=()
for F in "$SODIR"/_C*.so; do [ -f "$F" ] && SOG+=("$F"); done
[ "${#SOG[@]}" -eq 1 ] || fb "need exactly one _C*.so in $SODIR, found ${#SOG[@]}"
[ "$(basename "${SOG[0]}")" = "$(brget SO_BASENAME)" ] \
  || fb "deployed basename $(basename "${SOG[0]}") != record SO_BASENAME $(brget SO_BASENAME)"
DBYTES=$(stat -c %s "${SOG[0]}")
[ "$DBYTES" = "$(brget SO_BYTES)" ] \
  || fb "deployed so size $DBYTES != record SO_BYTES $(brget SO_BYTES)"
DSHA=$(sha256sum "${SOG[0]}" | cut -d' ' -f1)
[ "$DSHA" = "$(brget SO_SHA256)" ] \
  || fb "deployed so sha256 $DSHA != record SO_SHA256 $(brget SO_SHA256)"
echo "FORMAL_BINDING_PASS:$DSHA"
