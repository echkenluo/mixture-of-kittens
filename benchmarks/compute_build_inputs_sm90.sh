#!/bin/bash
# Build-input identity, computed from git OBJECTS at a given commit (tracked).
#
# One implementation, used by both sides: the build wrapper records these
# values and the packaging host recomputes them at the record's SOURCE_COMMIT.
# NOTE ON WHAT THAT BUYS: sharing the implementation gives CONSISTENCY, not
# correctness - a bug here is reproduced identically on both sides and the
# comparison still passes. Correctness evidence comes from the independent
# negative cases in test_build_record_local.sh, not from the agreement.
#
# The spec path is FIXED here. An earlier version accepted it as an argument,
# which was a caller override of the very thing that must not be caller-
# controlled: a narrower spec means a record that covers less than what was
# compiled while still validating.
#
# It also emits the BUILD-SIDE TOOLING identity - the wrapper, this helper,
# the build-record validator and both spec files at that commit - so that
# "which code produced this record" is recomputable. The packaging and
# verification scripts are NOT in that set.
#
#   failure -> exit 19 BUILD_INPUTS_FAIL:<why>
#   success -> exit 0, prints KEY=VALUE lines
# Usage: compute_build_inputs_sm90.sh <repo_dir> <commit>
set -uo pipefail
REPO=${1:?repo dir}; COMMIT=${2:?commit}
SPEC_PATH=benchmarks/build_input_spec.v1
CMD_SPEC_PATH=benchmarks/build_command_spec.v2
BUILD_SIDE_TOOLING_PATHS="benchmarks/build_and_record_sm90.sh benchmarks/compute_build_inputs_sm90.sh benchmarks/validate_build_record_sm90.sh benchmarks/build_input_spec.v1 benchmarks/build_command_spec.v2"
bf() { echo "BUILD_INPUTS_FAIL:$1"; exit 19; }
[ "$(git -C "$REPO" cat-file -t "$COMMIT" 2>/dev/null)" = "commit" ] \
  || bf "commit $COMMIT does not exist in this repository"

# --- input spec: strict parse, hashed as RAW BLOB BYTES ---
git -C "$REPO" cat-file -e "$COMMIT:$SPEC_PATH" 2>/dev/null \
  || bf "$SPEC_PATH does not exist at $COMMIT"
SPEC_SHA=$(git -C "$REPO" cat-file blob "$COMMIT:$SPEC_PATH" | sha256sum | cut -d' ' -f1)
SPEC=$(git -C "$REPO" cat-file blob "$COMMIT:$SPEC_PATH")
printf '%s\n' "$SPEC" | head -1 | grep -q '^BUILD_INPUT_SPEC=1$' \
  || bf "build input spec at $COMMIT has a bad schema line"
strict_parse() { # spec-content allowed-keys single-keys context
  local CONTENT=$1 ALLOWED=$2 SINGLE=$3 CTX=$4 K V
  while IFS= read -r LINE; do
    [ -z "$LINE" ] && continue
    case "$LINE" in \#*) continue ;; esac
    case "$LINE" in *=*) : ;; *) bf "$CTX: line without '=': $LINE" ;; esac
    K=${LINE%%=*}; V=${LINE#*=}
    echo " $ALLOWED " | grep -q " $K " || bf "$CTX: unknown key $K"
    [ -n "$V" ] || bf "$CTX: key $K has an empty value"
  done <<< "$CONTENT"
  for K in $SINGLE; do
    local N
    N=$(printf '%s\n' "$CONTENT" | grep -c "^$K=" || true)
    [ "$N" -eq 1 ] || bf "$CTX: key $K appears $N times (need exactly 1)"
  done
}
strict_parse "$SPEC" "BUILD_INPUT_SPEC NAME PATH SUBMODULE" "BUILD_INPUT_SPEC NAME" "build input spec"
SPEC_NAME=$(printf '%s\n' "$SPEC" | grep '^NAME=' | head -1 | cut -d= -f2-)
mapfile -t PATHS < <(printf '%s\n' "$SPEC" | grep '^PATH=' | cut -d= -f2-)
mapfile -t SUBS < <(printf '%s\n' "$SPEC" | grep '^SUBMODULE=' | cut -d= -f2-)
[ "${#PATHS[@]}" -gt 0 ] || bf "build input spec lists no PATH entries"
safe_rel() { # path context
  case "$1" in
    /*) bf "$2: absolute path not allowed: $1" ;;
    *..*) bf "$2: path may not contain '..': $1" ;;
    "") bf "$2: empty path" ;;
  esac
}
for P in "${PATHS[@]}"; do safe_rel "$P" "build input spec"; done
for S in "${SUBS[@]}"; do safe_rel "$S" "build input spec"; done
DUPP=$(printf '%s\n' "${PATHS[@]}" | LC_ALL=C sort | uniq -d | head -1)
[ -z "$DUPP" ] || bf "build input spec: duplicate PATH $DUPP"
if [ "${#SUBS[@]}" -gt 0 ]; then
  DUPS=$(printf '%s\n' "${SUBS[@]}" | LC_ALL=C sort | uniq -d | head -1)
  [ -z "$DUPS" ] || bf "build input spec: duplicate SUBMODULE $DUPS"
fi

# --- command spec: strict parse, hashed as RAW BLOB BYTES ---
git -C "$REPO" cat-file -e "$COMMIT:$CMD_SPEC_PATH" 2>/dev/null \
  || bf "$CMD_SPEC_PATH does not exist at $COMMIT"
CMD_SHA=$(git -C "$REPO" cat-file blob "$COMMIT:$CMD_SPEC_PATH" | sha256sum | cut -d' ' -f1)
CMD=$(git -C "$REPO" cat-file blob "$COMMIT:$CMD_SPEC_PATH")
printf '%s\n' "$CMD" | head -1 | grep -q '^BUILD_COMMAND_SPEC=2$' \
  || bf "build command spec at $COMMIT is not schema 2 (schema 1 meant ENV_PASS and is not interchangeable)"
strict_parse "$CMD" "BUILD_COMMAND_SPEC NAME OUTPUT HOST_COMPILER ARGV PROBE ENV_SET TOOLCHAIN_ROOT" "BUILD_COMMAND_SPEC NAME OUTPUT HOST_COMPILER" "build command spec"
CMD_NAME=$(printf '%s\n' "$CMD" | grep '^NAME=' | head -1 | cut -d= -f2-)
CMD_OUTPUT=$(printf '%s\n' "$CMD" | grep '^OUTPUT=' | head -1 | cut -d= -f2-)
safe_rel "$CMD_OUTPUT" "build command spec OUTPUT"
NARGV=$(printf '%s\n' "$CMD" | grep -c '^ARGV=' || true)
[ "$NARGV" -gt 0 ] || bf "build command spec lists no ARGV entries"
NPROBE=$(printf '%s\n' "$CMD" | grep -c '^PROBE=' || true)
[ "$NPROBE" -gt 0 ] || bf "build command spec lists no PROBE entries"
NENV=$(printf '%s\n' "$CMD" | grep -c '^ENV_SET=' || true)
[ "$NENV" -gt 0 ] || bf "build command spec declares no ENV_SET environment"
while IFS= read -r E; do
  case "$E" in [A-Z_]*=*) : ;; *) bf "build command spec: ENV_SET entry is not NAME=value: $E" ;; esac
done < <(printf '%s\n' "$CMD" | grep '^ENV_SET=' | cut -d= -f2-)
NROOT=$(printf '%s\n' "$CMD" | grep -c '^TOOLCHAIN_ROOT=' || true)
[ "$NROOT" -gt 0 ] || bf "build command spec declares no TOOLCHAIN_ROOT"
CMD_HOSTCC=$(printf '%s\n' "$CMD" | grep '^HOST_COMPILER=' | head -1 | cut -d= -f2-)
case "$CMD_HOSTCC" in /*) : ;; *) bf "build command spec HOST_COMPILER must be absolute: $CMD_HOSTCC" ;; esac

TREE_OID=$(git -C "$REPO" rev-parse "$COMMIT^{tree}" 2>/dev/null) || bf "cannot resolve tree of $COMMIT"
LIST=$(git -C "$REPO" ls-tree -r "$COMMIT" -- "${PATHS[@]}" 2>/dev/null \
        | awk '$2=="blob" {print $4"\t"$3}' | LC_ALL=C sort)
COUNT=$(printf '%s\n' "$LIST" | grep -c . || true)
[ "$COUNT" -gt 0 ] || bf "build input closure is empty at $COMMIT"
LIST_SHA=$(printf '%s\n' "$LIST" | sha256sum | cut -d' ' -f1)
CONTENT_SHA=$( { printf '%s\n' "$LIST" | while IFS=$'\t' read -r P O; do
      [ -n "$P" ] || continue
      printf '%s\0' "$P"; git -C "$REPO" cat-file blob "$O"; done; } | sha256sum | cut -d' ' -f1)
SUBLIST=""
if [ "${#SUBS[@]}" -gt 0 ]; then
  SUBLIST=$(git -C "$REPO" ls-tree -r "$COMMIT" -- "${SUBS[@]}" 2>/dev/null \
             | awk '$2=="commit" {print $4"\t"$3}' | LC_ALL=C sort)
  for S in "${SUBS[@]}"; do
    printf '%s\n' "$SUBLIST" | grep -q "^$S	" \
      || bf "submodule $S declared by the spec is not a gitlink at $COMMIT"
  done
fi
SUBCOUNT=$(printf '%s\n' "$SUBLIST" | grep -c . || true)
SUBSHA=$(printf '%s\n' "$SUBLIST" | sha256sum | cut -d' ' -f1)

# --- BUILD-SIDE tooling identity: the five files that produce a record. This
# does NOT cover the packaging or verification side (make_deploy_receipt.sh,
# validate_receipt_sm90.sh, validate_manifest_sm90.sh,
# check_formal_binding_sm90.sh) - claiming otherwise would overstate the
# evidence, so the field is named for what it actually binds. ---
TOOLLIST=""
for T in $BUILD_SIDE_TOOLING_PATHS; do
  git -C "$REPO" cat-file -e "$COMMIT:$T" 2>/dev/null || bf "tooling file $T does not exist at $COMMIT"
  O=$(git -C "$REPO" rev-parse "$COMMIT:$T")
  TOOLLIST="$TOOLLIST$T	$O
"
done
TOOLLIST=$(printf '%s' "$TOOLLIST" | LC_ALL=C sort)
TOOLCOUNT=$(printf '%s\n' "$TOOLLIST" | grep -c . || true)
TOOLSHA=$(printf '%s\n' "$TOOLLIST" | sha256sum | cut -d' ' -f1)

echo "BUILD_INPUT_SPEC_NAME=$SPEC_NAME"
echo "BUILD_INPUT_SPEC_SHA256=$SPEC_SHA"
echo "BUILD_COMMAND_SPEC_NAME=$CMD_NAME"
echo "BUILD_COMMAND_SPEC_SHA256=$CMD_SHA"
echo "SOURCE_TREE_GIT_OID=$TREE_OID"
echo "BUILD_INPUT_FILE_COUNT=$COUNT"
echo "BUILD_INPUT_LIST_SHA256=$LIST_SHA"
echo "BUILD_INPUT_CONTENT_SHA256=$CONTENT_SHA"
echo "SUBMODULE_COUNT=$SUBCOUNT"
echo "SUBMODULE_LIST_SHA256=$SUBSHA"
echo "BUILD_SIDE_TOOLING_FILE_COUNT=$TOOLCOUNT"
echo "BUILD_SIDE_TOOLING_LIST_SHA256=$TOOLSHA"
