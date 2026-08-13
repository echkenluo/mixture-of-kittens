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
#   1 expected record sha (out-of-band prior) == record bytes
#       -> a record swapped on the verified end
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
# Usage: EXPECTED_BUILD_RECORD_SHA256=... \
#          check_formal_binding_sm90.sh <record> <receipt> <manifest> <so_dir>
set -uo pipefail
REC=${1:?build record}; RCPT=${2:?receipt}; MAN=${3:?manifest}; SODIR=${4:?dir holding the deployed _C*.so}
DIR=$(cd "$(dirname "$0")" && pwd)
fb() { echo "FORMAL_BINDING_FAIL:$1"; exit 18; }
EXPR_REC=${EXPECTED_BUILD_RECORD_SHA256:-}
echo "$EXPR_REC" | grep -qE '^[0-9a-f]{64}$' \
  || fb "EXPECTED_BUILD_RECORD_SHA256 env missing or not 64-hex"
[ -f "$REC" ] || fb "build record $REC missing"
[ -f "$RCPT" ] || fb "receipt $RCPT missing"
[ -f "$MAN" ] || fb "manifest $MAN missing"
bash "$DIR/validate_build_record_sm90.sh" "$REC" --check-mode >/dev/null \
  || fb "build record failed shared validator"
bash "$DIR/validate_receipt_sm90.sh" "$RCPT" --check-mode >/dev/null \
  || fb "receipt failed shared validator"
bash "$DIR/validate_manifest_sm90.sh" "$MAN" >/dev/null \
  || fb "manifest failed shared validator"
RSHA=$(sha256sum "$REC" | cut -d' ' -f1)
[ "$RSHA" = "$EXPR_REC" ] || fb "record sha $RSHA != EXPECTED_BUILD_RECORD_SHA256 $EXPR_REC"
brget() { grep "^$1=" "$REC"  | head -1 | cut -d= -f2- || true; }
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
SOG=("$SODIR"/_C*.so)
{ [ "${#SOG[@]}" -eq 1 ] && [ -f "${SOG[0]}" ]; } \
  || fb "need exactly one _C*.so in $SODIR, found ${#SOG[@]}"
[ "$(basename "${SOG[0]}")" = "$(brget SO_BASENAME)" ] \
  || fb "deployed basename $(basename "${SOG[0]}") != record SO_BASENAME $(brget SO_BASENAME)"
DBYTES=$(stat -c %s "${SOG[0]}")
[ "$DBYTES" = "$(brget SO_BYTES)" ] \
  || fb "deployed so size $DBYTES != record SO_BYTES $(brget SO_BYTES)"
DSHA=$(sha256sum "${SOG[0]}" | cut -d' ' -f1)
[ "$DSHA" = "$(brget SO_SHA256)" ] \
  || fb "deployed so sha256 $DSHA != record SO_SHA256 $(brget SO_SHA256)"
echo "FORMAL_BINDING_PASS:$DSHA"
