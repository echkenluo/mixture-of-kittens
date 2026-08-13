#!/bin/bash
# Shared build-record validator (tracked). BUILD_RECORD_SCHEMA=4.
#
# Schemas 1, 2 and 3 are all GONE, each for its own reason: 1 came from a
# generator that never ran a build; 2 existed in two mutually incompatible
# shapes; 3 was likewise published twice with different field sets
# (TOOLING_*/ENV_PASS_NAMES, then BUILD_SIDE_TOOLING_*/ENV_MANIFEST_B64 plus
# the toolchain-path fields). Reusing a version number for formats that cannot
# be interchanged makes archived records ambiguous - the same mistake twice,
# so schema 4 is a real bump and 3 is refused explicitly.
#
# The schema separates what was MEASURED by the wrapper (source identity,
# input closure, submodules, the argv spec it exec'd, the output it observed,
# the logs it created, versions and toolchain paths it probed) from what is
# merely DECLARED BY THE CALLER (the container image identity - nothing here
# verifies it) and what is UNVERIFIED EXTERNALLY (the clean entrypoint, which
# a script cannot establish about itself). PROVENANCE_CLASS carries all three
# and a record that drops any of the markers is refused.
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
REQ="BUILD_RECORD_SCHEMA RECORD_MODE PROVENANCE_CLASS BUILD_START_UTC BUILD_END_UTC BUILD_EXIT_CODE SOURCE_COMMIT SOURCE_TREE_GIT_OID BUILD_INPUT_SPEC_NAME BUILD_INPUT_SPEC_SHA256 BUILD_INPUT_FILE_COUNT BUILD_INPUT_LIST_SHA256 BUILD_INPUT_CONTENT_SHA256 SUBMODULE_COUNT SUBMODULE_LIST_SHA256 BUILD_SIDE_TOOLING_FILE_COUNT BUILD_SIDE_TOOLING_LIST_SHA256 BUILD_COMMAND_SPEC_NAME BUILD_COMMAND_SPEC_SHA256 BUILD_COMMAND_ARGV_JOINED TOOLCHAIN_IMAGE_ID_DECLARED_BY_CALLER TOOLCHAIN_IMAGE_REF_DECLARED_BY_CALLER TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER MEASURED_CUDA_VERSION MEASURED_NVCC_VERSION MEASURED_HOST_COMPILER_VERSION MEASURED_PYTHON_VERSION MEASURED_TORCH_VERSION MEASURED_EXT_SUFFIX MEASURED_PY_INCLUDE MEASURED_TORCH_INCLUDE MEASURED_TORCH_LIBDIR MEASURED_HOST_COMPILER_PATH ENV_MANIFEST_B64 ENV_APPLIED_SHA256 PROBE_LOG_SHA256 PROBE_LOG_BYTES TARGET_ARCH SO_BASENAME SO_BYTES SO_SHA256 BUILD_LOG_SHA256 BUILD_LOG_BYTES"
HEX64="ENV_APPLIED_SHA256 BUILD_SIDE_TOOLING_LIST_SHA256 BUILD_INPUT_SPEC_SHA256 BUILD_INPUT_LIST_SHA256 BUILD_INPUT_CONTENT_SHA256 SUBMODULE_LIST_SHA256 BUILD_COMMAND_SPEC_SHA256 PROBE_LOG_SHA256 SO_SHA256 BUILD_LOG_SHA256"
HEX40="SOURCE_COMMIT SOURCE_TREE_GIT_OID"
POSINT="BUILD_SIDE_TOOLING_FILE_COUNT BUILD_INPUT_FILE_COUNT PROBE_LOG_BYTES SO_BYTES BUILD_LOG_BYTES"
head -1 "$REC" | grep -q '^BUILD_RECORD_SCHEMA=4$' \
  || { echo "BUILD_RECORD_FAIL:bad or missing schema version (only 4 is valid; 1 never ran a build, and 2 and 3 were each published in two incompatible field sets)"; exit 17; }
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
         MEASURED_PYTHON_VERSION MEASURED_TORCH_VERSION MEASURED_EXT_SUFFIX \
         MEASURED_PY_INCLUDE MEASURED_TORCH_INCLUDE MEASURED_TORCH_LIBDIR; do
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
rget PROVENANCE_CLASS | grep -q 'measured:.*build_side_tooling' \
  || { echo "BUILD_RECORD_FAIL:PROVENANCE_CLASS must declare build_side_tooling as measured"; exit 17; }
# the image identity is a caller assertion with no verification behind it;
# calling it "attested" overstated what the code does
rget PROVENANCE_CLASS | grep -q 'unverified_external:clean_entrypoint' \
  || { echo "BUILD_RECORD_FAIL:PROVENANCE_CLASS must declare unverified_external:clean_entrypoint (a script cannot prove its own entrypoint was clean)"; exit 17; }
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
# the env manifest must be present verbatim, not just as a hash
printf '%s' "$(rget ENV_MANIFEST_B64)" | base64 -d >/dev/null 2>&1 \
  || { echo "BUILD_RECORD_FAIL:ENV_MANIFEST_B64 is not valid base64"; exit 17; }
# stream the decode into sha256sum: a command substitution would strip the
# trailing newline and make the comparison depend on an invisible detail
ENVSHA=$(printf '%s' "$(rget ENV_MANIFEST_B64)" | base64 -d | sha256sum | cut -d' ' -f1)
[ "$ENVSHA" = "$(rget ENV_APPLIED_SHA256)" ] \
  || { echo "BUILD_RECORD_FAIL:ENV_APPLIED_SHA256 does not match ENV_MANIFEST_B64"; exit 17; }
# strict parse of the manifest itself: unique NAME=value lines, absolute PATH.
# NOTE: this checks the record is internally well-formed; it does NOT verify
# the environment against the command spec - the spec lives in git and this
# validator is handed only a record.
ENVDEC=$(printf '%s' "$(rget ENV_MANIFEST_B64)" | base64 -d)
while IFS= read -r EL; do
  [ -z "$EL" ] && continue
  case "$EL" in
    [A-Z_]*=*) : ;;
    *) echo "BUILD_RECORD_FAIL:env manifest line is not NAME=value: $EL"; exit 17 ;;
  esac
done <<< "$ENVDEC"
ENVDUP=$(printf '%s\n' "$ENVDEC" | grep -o '^[A-Z_]*=' | LC_ALL=C sort | uniq -d | head -1)
[ -z "$ENVDUP" ] || { echo "BUILD_RECORD_FAIL:env manifest defines ${ENVDUP%=} more than once"; exit 17; }
printf '%s\n' "$ENVDEC" | grep -q '^PATH=/' \
  || { echo "BUILD_RECORD_FAIL:the recorded environment has no absolute PATH"; exit 17; }
case "$(rget MEASURED_HOST_COMPILER_PATH)" in
  /*) : ;;
  *) echo "BUILD_RECORD_FAIL:MEASURED_HOST_COMPILER_PATH must be an absolute path"; exit 17 ;;
esac
rget SO_BASENAME | grep -qE '^_C[A-Za-z0-9._-]*\.so$' \
  || { echo "BUILD_RECORD_FAIL:SO_BASENAME does not look like a built extension (_C*.so)"; exit 17; }
echo "BUILD_RECORD_VALID:$(sha256sum "$REC" | cut -d' ' -f1)"
