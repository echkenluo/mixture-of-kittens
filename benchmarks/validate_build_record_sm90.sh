#!/bin/bash
# Shared build-record validator (tracked). BUILD_RECORD_SCHEMA=3.
#
# Schemas 1 and 2 are GONE, not deprecated, and each for a different reason:
# schema 1 came from a generator that never ran a build; schema 2 existed in
# two mutually incompatible shapes (32 keys with an unbound command and
# tooling, then 36 keys) - reusing one version number for formats that cannot
# be interchanged makes archived records ambiguous. Schema 3 is the first that
# binds the command, the tooling AND the build environment.
#
# The schema separates what was MEASURED by the wrapper (source identity,
# input closure, submodules, the argv spec it exec'd, the output it observed,
# the logs it created, versions it probed) from what is ATTESTED by the
# orchestrator (the container image identity, which cannot be measured from
# inside the container). PROVENANCE_CLASS carries that split and must keep
# saying so - a record that quietly drops the attested marker is refused.
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
REQ="BUILD_RECORD_SCHEMA RECORD_MODE PROVENANCE_CLASS BUILD_START_UTC BUILD_END_UTC BUILD_EXIT_CODE SOURCE_COMMIT SOURCE_TREE_GIT_OID BUILD_INPUT_SPEC_NAME BUILD_INPUT_SPEC_SHA256 BUILD_INPUT_FILE_COUNT BUILD_INPUT_LIST_SHA256 BUILD_INPUT_CONTENT_SHA256 SUBMODULE_COUNT SUBMODULE_LIST_SHA256 TOOLING_FILE_COUNT TOOLING_LIST_SHA256 BUILD_COMMAND_SPEC_NAME BUILD_COMMAND_SPEC_SHA256 BUILD_COMMAND_ARGV_JOINED TOOLCHAIN_IMAGE_ID_DECLARED_BY_CALLER TOOLCHAIN_IMAGE_REF_DECLARED_BY_CALLER TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER MEASURED_CUDA_VERSION MEASURED_NVCC_VERSION MEASURED_HOST_COMPILER_VERSION MEASURED_PYTHON_VERSION MEASURED_TORCH_VERSION MEASURED_EXT_SUFFIX MEASURED_HOST_COMPILER_PATH ENV_PASS_NAMES ENV_APPLIED_SHA256 PROBE_LOG_SHA256 PROBE_LOG_BYTES TARGET_ARCH SO_BASENAME SO_BYTES SO_SHA256 BUILD_LOG_SHA256 BUILD_LOG_BYTES"
HEX64="ENV_APPLIED_SHA256 TOOLING_LIST_SHA256 BUILD_INPUT_SPEC_SHA256 BUILD_INPUT_LIST_SHA256 BUILD_INPUT_CONTENT_SHA256 SUBMODULE_LIST_SHA256 BUILD_COMMAND_SPEC_SHA256 PROBE_LOG_SHA256 SO_SHA256 BUILD_LOG_SHA256"
HEX40="SOURCE_COMMIT SOURCE_TREE_GIT_OID"
POSINT="TOOLING_FILE_COUNT BUILD_INPUT_FILE_COUNT PROBE_LOG_BYTES SO_BYTES BUILD_LOG_BYTES"
head -1 "$REC" | grep -q '^BUILD_RECORD_SCHEMA=3$' \
  || { echo "BUILD_RECORD_FAIL:bad or missing schema version (only 3 is valid; 1 never ran a build, 2 had two incompatible shapes and bound neither the environment nor the host compiler)"; exit 17; }
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
for K in $HEX40; do
  rget "$K" | grep -qE '^[0-9a-f]{40}$' || { echo "BUILD_RECORD_FAIL:$K not 40-hex"; exit 17; }
done
for K in $POSINT; do
  V=$(rget "$K")
  { echo "$V" | grep -qE '^[0-9]+$' && [ "$V" -gt 0 ]; } \
    || { echo "BUILD_RECORD_FAIL:$K not a positive integer"; exit 17; }
done
rget SUBMODULE_COUNT | grep -qE '^[0-9]+$' \
  || { echo "BUILD_RECORD_FAIL:SUBMODULE_COUNT not a non-negative integer"; exit 17; }
# fixture records exist so tests can drive an arbitrary command; they are
# structurally separated here and refused by the formal binding checker
case "$(rget RECORD_MODE)" in
  production|fixture) : ;;
  *) echo "BUILD_RECORD_FAIL:RECORD_MODE not in allowed set {production,fixture}"; exit 17 ;;
esac
# a measured field that says it could not be measured is not a measurement;
# an earlier version allowed "unavailable (see probe log)" to satisfy non-empty
for K in MEASURED_CUDA_VERSION MEASURED_NVCC_VERSION MEASURED_HOST_COMPILER_VERSION \
         MEASURED_PYTHON_VERSION MEASURED_TORCH_VERSION MEASURED_EXT_SUFFIX; do
  case "$(rget "$K")" in
    *unavailable*|*unknown*|*"not measured"*)
      echo "BUILD_RECORD_FAIL:$K is a placeholder, not a measurement: $(rget "$K")"; exit 17 ;;
  esac
done

# a record only exists for a build that succeeded; a failed build has nothing
# to describe and must never be publishable
[ "$(rget BUILD_EXIT_CODE)" = "0" ] \
  || { echo "BUILD_RECORD_FAIL:BUILD_EXIT_CODE $(rget BUILD_EXIT_CODE) != 0 (a record may only describe a successful build)"; exit 17; }
for K in BUILD_START_UTC BUILD_END_UTC; do
  rget "$K" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' \
    || { echo "BUILD_RECORD_FAIL:$K not YYYY-MM-DDTHH:MM:SSZ"; exit 17; }
done
# the measured/attested split must remain visible in the record itself
rget PROVENANCE_CLASS | grep -q 'measured:.*tooling' \
  || { echo "BUILD_RECORD_FAIL:PROVENANCE_CLASS must declare the tooling as measured"; exit 17; }
# the image identity is a caller assertion with no verification behind it;
# calling it "attested" overstated what the code does
rget PROVENANCE_CLASS | grep -q 'declared_unverified:toolchain_image' \
  || { echo "BUILD_RECORD_FAIL:PROVENANCE_CLASS must declare declared_unverified:toolchain_image (the image identity is a caller assertion, not a verified attestation)"; exit 17; }
rget PROVENANCE_CLASS | grep -q 'measured:.*env' \
  || { echo "BUILD_RECORD_FAIL:PROVENANCE_CLASS must declare the build environment as measured"; exit 17; }
rget PROVENANCE_CLASS | grep -q '^measured:' \
  || { echo "BUILD_RECORD_FAIL:PROVENANCE_CLASS must start with the measured: list"; exit 17; }
rget TOOLCHAIN_IMAGE_ID_DECLARED_BY_CALLER | grep -qE '^sha256:[0-9a-f]{64}$' \
  || { echo "BUILD_RECORD_FAIL:TOOLCHAIN_IMAGE_ID_DECLARED_BY_CALLER malformed (want sha256:<64-hex>)"; exit 17; }
RD=$(rget TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER)
if [ "$RD" != "NONE" ]; then
  printf '%s\n' "$RD" | tr ',' '\n' | while IFS= read -r E; do
    echo "$E" | grep -qE '^[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[0-9a-f]{64}$' || exit 1
  done || { echo "BUILD_RECORD_FAIL:TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER malformed (want NONE or repo@sha256:<64-hex> list)"; exit 17; }
fi
case "$(rget TARGET_ARCH)" in
  SM90|SM100|SM103) : ;;
  *) echo "BUILD_RECORD_FAIL:TARGET_ARCH not in allowed set {SM90,SM100,SM103}"; exit 17 ;;
esac
# the ext-suffix the interpreter reported must match the artifact's name
[ "$(rget SO_BASENAME)" = "_C$(rget MEASURED_EXT_SUFFIX)" ] \
  || { echo "BUILD_RECORD_FAIL:SO_BASENAME $(rget SO_BASENAME) != _C$(rget MEASURED_EXT_SUFFIX) implied by MEASURED_EXT_SUFFIX"; exit 17; }
case "$(rget SO_BASENAME)" in
  */*) echo "BUILD_RECORD_FAIL:SO_BASENAME must be a bare filename, not a path"; exit 17 ;;
esac
case "$(rget MEASURED_HOST_COMPILER_PATH)" in
  /*) : ;;
  *) echo "BUILD_RECORD_FAIL:MEASURED_HOST_COMPILER_PATH must be an absolute path"; exit 17 ;;
esac
rget SO_BASENAME | grep -qE '^_C[A-Za-z0-9._-]*\.so$' \
  || { echo "BUILD_RECORD_FAIL:SO_BASENAME does not look like a built extension (_C*.so)"; exit 17; }
echo "BUILD_RECORD_VALID:$(sha256sum "$REC" | cut -d' ' -f1)"
