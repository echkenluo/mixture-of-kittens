#!/bin/bash
# Formal-binding checker (tracked). Verifies the complete chain a formal run
# would need, in one place, so the chain can be reviewed and tested before
# anything is allowed to depend on it.
#
# NOT WIRED INTO THE LAUNCHER, AND IT CANNOT BE. host_launch_sm90.sh still
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
#   5 manifest bytes == receipt.MANIFEST_SHA256, receipt.HARNESS_SHA256 ==
#     manifest.EXPECTED_HARNESS_SHA256, manifest.EXPECTED_SO_SHA256 ==
#     record.SO_SHA256
#       -> a measurement contract frozen against a different binary, AND a
#          different manifest (other shape/iters/harness) presented in place of
#          the one the anchored receipt names
#   6 deployed .so bytes                       == record.SO_SHA256
#       -> the binary on the target not being the one that was built
#
# THERE IS NO FORMAL PASS PATH. Nothing in this chain verifies the container
# image identity - it is a string the caller handed the wrapper - so a
# fully-consistent chain still does not establish formal provenance. Emitting
# FORMAL_BINDING_PASS would invite a future integrator to wire it up on the
# strength of a check that only proves internal consistency. The success path
# is therefore LOCAL_BINDING_PASS, and formal is refused unconditionally
# unless the caller explicitly asks for the local-consistency check only.
#   any violation             -> exit 18 FORMAL_BINDING_FAIL:<why>
#   formal requested          -> exit 18, always
#   LOCAL_BINDING_ONLY=1 + ok -> exit 0, prints LOCAL_BINDING_PASS:<so_sha256>
#     which asserts agreement between artifacts ONLY, and only as far as the
#     tooling running it can be trusted - see the entrypoint note below
# Usage: EXPECTED_RECEIPT_SHA256=... [EXPECTED_BUILD_RECORD_SHA256=...] \
#          check_formal_binding_sm90.sh <receipt> <record> <manifest> <so_dir>
set -uo pipefail
# Same misuse protection as the launcher and the generator, and the same
# boundary: every comparison below shells out through PATH, so LOCAL_BINDING_
# PASS means "these artifacts agree WHEN THE TOOLING IS TRUSTED". It is not an
# independently trustworthy local chain, and it is not a step toward formal.
[[ -x /usr/bin/env && -x /usr/bin/grep ]] \
  || { echo "FORMAL_BINDING_FAIL:/usr/bin/env or /usr/bin/grep missing"; exit 18; }
[[ "${BASH_SOURCE[0]}" == "$0" ]] \
  || { echo "FORMAL_BINDING_FAIL:this checker must be executed, not sourced"; exit 18; }
BAD_ENTRY=$(/usr/bin/env | /usr/bin/grep -m1 -oE '^(BASH_FUNC_[^=%(]*|BASH_ENV|ENV|SHELLOPTS|BASHOPTS)=?' || true)
BAD_ENTRY=${BAD_ENTRY%=}
if [[ -n $BAD_ENTRY ]]; then
  case $BAD_ENTRY in
    BASH_FUNC_*) echo "FORMAL_BINDING_FAIL:exported shell function ${BAD_ENTRY#BASH_FUNC_} is present; a function shadows PATH lookups" ;;
    *) echo "FORMAL_BINDING_FAIL:$BAD_ENTRY is set; this checker must be started from a clean entrypoint" ;;
  esac
  exit 18
fi
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
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
# Snapshot first, then verify, then use. Hashing a caller path and afterwards
# re-reading that same path for every field lets the file change in between:
# the launcher had exactly that defect and it was demonstrable. Each artifact
# is copied ONCE into a private 0700 directory, and everything downstream -
# hashes, validators, field reads - happens on the snapshot. Read-only bits on
# the caller's copies are therefore not relied on at all.
SNAP=$(mktemp -d) || fb "cannot create a private snapshot directory"
trap 'rm -rf "$SNAP"' EXIT
set -o noclobber
{ cat < "$RCPT" > "$SNAP/receipt"; } 2>/dev/null || fb "cannot snapshot the receipt"
{ cat < "$REC"  > "$SNAP/record";  } 2>/dev/null || fb "cannot snapshot the build record"
{ cat < "$MAN"  > "$SNAP/manifest";} 2>/dev/null || fb "cannot snapshot the manifest"
set +o noclobber
chmod 444 "$SNAP/receipt" "$SNAP/record" "$SNAP/manifest"
MANNAME=$(basename "$MAN")
RCPT=$SNAP/receipt; REC=$SNAP/record; MAN=$SNAP/manifest
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
# NONE marks a local-only image. NOTE: a non-NONE digest here is still only
# what the CALLER declared - nothing in this chain verifies it against a
# trusted orchestrator artifact. This check therefore cannot be the reason
# formal is opened; formal stays refused until a verified attestation exists.
[ "$(brget TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER)" != "NONE" ] \
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
# The manifest handed to this checker must be THE manifest the anchored receipt
# describes. Comparing only EXPECTED_SO_SHA256 (as the previous version did)
# accepts any manifest built against the same binary: a different shape, a
# different iteration count, a different harness. That leaves the measurement
# contract unbound while every other link looks consistent.
MANSHA=$(sha256sum "$MAN" | cut -d' ' -f1)
[ "$MANSHA" = "$(rcget MANIFEST_SHA256)" ] \
  || fb "manifest sha $MANSHA != receipt MANIFEST_SHA256 $(rcget MANIFEST_SHA256); this is not the manifest the receipt describes"
[ "$MANNAME" = "$(rcget MANIFEST_FILE)" ] \
  || fb "manifest basename $MANNAME != receipt MANIFEST_FILE $(rcget MANIFEST_FILE)"
[ "$(rcget HARNESS_SHA256)" = "$(mnget EXPECTED_HARNESS_SHA256)" ] \
  || fb "receipt HARNESS_SHA256 != manifest EXPECTED_HARNESS_SHA256"
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
# the last gate: internal consistency is not attestation
[ "${LOCAL_BINDING_ONLY:-0}" = "1" ] \
  || fb "trusted toolchain image attestation not implemented (the image identity is a caller declaration); set LOCAL_BINDING_ONLY=1 for a consistency-only check"
echo "LOCAL_BINDING_PASS:$DSHA"
