#!/bin/bash
# Shared build-record validator (tracked). Single source of truth for
# BUILD_RECORD_SCHEMA=1 semantics, used by the generator (self-check before
# publishing), by the packaging host when it binds a receipt to a record, and
# by the remote verified end.
#
# What a build record is FOR: it is the only artifact that can tie a deployed
# .so back to the source and toolchain that produced it. A commit id cannot -
# any commit can be named next to any binary. So every identity field here is
# a content hash or an image digest measured at build time from real bytes;
# paths and human-readable version strings are recorded for review but are
# never the identity of anything.
#
#   any violation -> exit 17 BUILD_RECORD_FAIL:<why>
#   valid         -> exit 0, prints BUILD_RECORD_VALID:<sha256>
# Usage: validate_build_record_sm90.sh <record> [--check-mode]
set -uo pipefail
REC=${1:?build record path}
CHECK_MODE=0
[ "${2:-}" = "--check-mode" ] && CHECK_MODE=1
[ -f "$REC" ] || { echo "BUILD_RECORD_FAIL:missing record $REC"; exit 17; }
if [ "$CHECK_MODE" -eq 1 ]; then
  RMODE=$(stat -c %a "$REC")
  case "$RMODE" in *[2367]*) echo "BUILD_RECORD_FAIL:write bits set ($RMODE)"; exit 17 ;; esac
fi
REQ="BUILD_RECORD_SCHEMA BUILD_UTC SOURCE_COMMIT SOURCE_TREE_SHA256 BUILD_INPUTS_SHA256 BUILD_SCRIPT_SHA256 BUILD_COMMAND_SHA256 BUILD_COMMAND_ARGV TOOLCHAIN_IMAGE_ID TOOLCHAIN_IMAGE_REF TOOLCHAIN_IMAGE_REPO_DIGESTS CUDA_VERSION NVCC_VERSION HOST_COMPILER_VERSION PYTHON_VERSION TORCH_VERSION TARGET_ARCH SO_BASENAME SO_BYTES SO_SHA256 BUILD_LOG_SHA256 BUILD_LOG_BYTES"
HEX64="BUILD_INPUTS_SHA256 BUILD_SCRIPT_SHA256 BUILD_COMMAND_SHA256 SOURCE_TREE_SHA256 SO_SHA256 BUILD_LOG_SHA256"
POSINT="SO_BYTES BUILD_LOG_BYTES"
head -1 "$REC" | grep -q '^BUILD_RECORD_SCHEMA=1$' \
  || { echo "BUILD_RECORD_FAIL:bad or missing schema version"; exit 17; }
for K in $REQ; do
  N=$(grep -c "^$K=" "$REC" || true)
  [ "$N" -eq 1 ] || { echo "BUILD_RECORD_FAIL:key $K count=$N (need exactly 1)"; exit 17; }
done
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  K=${LINE%%=*}
  echo " $REQ " | grep -q " $K " || { echo "BUILD_RECORD_FAIL:unknown key $K"; exit 17; }
done < "$REC"
rget() { grep "^$1=" "$REC" | head -1 | cut -d= -f2-; }
for K in $REQ; do
  [ -n "$(rget "$K")" ] || { echo "BUILD_RECORD_FAIL:key $K empty"; exit 17; }
done
for K in $HEX64; do
  rget "$K" | grep -qE '^[0-9a-f]{64}$' || { echo "BUILD_RECORD_FAIL:$K not 64-hex"; exit 17; }
done
for K in $POSINT; do
  V=$(rget "$K")
  { echo "$V" | grep -qE '^[0-9]+$' && [ "$V" -gt 0 ]; } \
    || { echo "BUILD_RECORD_FAIL:$K not a positive integer"; exit 17; }
done
rget SOURCE_COMMIT | grep -qE '^[0-9a-f]{40}$' \
  || { echo "BUILD_RECORD_FAIL:SOURCE_COMMIT not 40-hex"; exit 17; }
rget BUILD_UTC | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' \
  || { echo "BUILD_RECORD_FAIL:BUILD_UTC not YYYY-MM-DDTHH:MM:SSZ"; exit 17; }
rget TOOLCHAIN_IMAGE_ID | grep -qE '^sha256:[0-9a-f]{64}$' \
  || { echo "BUILD_RECORD_FAIL:TOOLCHAIN_IMAGE_ID malformed (want sha256:<64-hex>)"; exit 17; }
RD=$(rget TOOLCHAIN_IMAGE_REPO_DIGESTS)
if [ "$RD" != "NONE" ]; then
  printf '%s\n' "$RD" | tr ',' '\n' | while IFS= read -r E; do
    echo "$E" | grep -qE '^[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[0-9a-f]{64}$' || exit 1
  done || { echo "BUILD_RECORD_FAIL:TOOLCHAIN_IMAGE_REPO_DIGESTS malformed (want NONE or repo@sha256:<64-hex> list)"; exit 17; }
fi
case "$(rget TARGET_ARCH)" in
  SM90|SM100|SM103) : ;;
  *) echo "BUILD_RECORD_FAIL:TARGET_ARCH not in allowed set {SM90,SM100,SM103}"; exit 17 ;;
esac
# SO_BASENAME is a NAME, not a path: a path in an identity field invites
# "same name, different file" confusion and is never evidence of anything
case "$(rget SO_BASENAME)" in
  */*) echo "BUILD_RECORD_FAIL:SO_BASENAME must be a bare filename, not a path"; exit 17 ;;
esac
rget SO_BASENAME | grep -qE '^_C[A-Za-z0-9._-]*\.so$' \
  || { echo "BUILD_RECORD_FAIL:SO_BASENAME does not look like a built extension (_C*.so)"; exit 17; }
echo "BUILD_RECORD_VALID:$(sha256sum "$REC" | cut -d' ' -f1)"
