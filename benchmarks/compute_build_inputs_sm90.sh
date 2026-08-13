#!/bin/bash
# Build-input identity, computed from git OBJECTS at a given commit (tracked).
#
# One implementation, used by both sides on purpose: the build wrapper records
# these values, and the packaging host RECOMPUTES them at the record's
# SOURCE_COMMIT and compares. If each side had its own implementation, a
# mismatch would be ambiguous; with one, a mismatch means the record is wrong.
#
# Reading from objects rather than the worktree also means the spec itself is
# taken from the commit, so a locally edited spec cannot change what a record
# claims to cover.
#
#   failure -> exit 19 BUILD_INPUTS_FAIL:<why>
#   success -> exit 0, prints KEY=VALUE lines
# Usage: compute_build_inputs_sm90.sh <repo_dir> <commit> [spec_repo_path]
set -uo pipefail
REPO=${1:?repo dir}; COMMIT=${2:?commit}; SPEC_PATH=${3:-benchmarks/build_input_spec.v1}
bf() { echo "BUILD_INPUTS_FAIL:$1"; exit 19; }
[ "$(git -C "$REPO" cat-file -t "$COMMIT" 2>/dev/null)" = "commit" ] \
  || bf "commit $COMMIT does not exist in this repository"
SPEC=$(git -C "$REPO" show "$COMMIT:$SPEC_PATH" 2>/dev/null) \
  || bf "$SPEC_PATH does not exist at $COMMIT"
printf '%s\n' "$SPEC" | head -1 | grep -q '^BUILD_INPUT_SPEC=1$' \
  || bf "build input spec at $COMMIT has a bad schema line"
SPEC_SHA=$(printf '%s\n' "$SPEC" | sha256sum | cut -d' ' -f1)
SPEC_NAME=$(printf '%s\n' "$SPEC" | grep '^NAME=' | head -1 | cut -d= -f2-)
[ -n "$SPEC_NAME" ] || bf "build input spec has no NAME"
mapfile -t PATHS < <(printf '%s\n' "$SPEC" | grep '^PATH=' | cut -d= -f2-)
mapfile -t SUBS < <(printf '%s\n' "$SPEC" | grep '^SUBMODULE=' | cut -d= -f2-)
[ "${#PATHS[@]}" -gt 0 ] || bf "build input spec lists no paths"

TREE_OID=$(git -C "$REPO" rev-parse "$COMMIT^{tree}" 2>/dev/null) || bf "cannot resolve tree of $COMMIT"
# regular files in the closure: "path<TAB>blob-oid", fixed order
LIST=$(git -C "$REPO" ls-tree -r "$COMMIT" -- "${PATHS[@]}" 2>/dev/null \
        | awk '$2=="blob" {print $4"\t"$3}' | LC_ALL=C sort)
COUNT=$(printf '%s\n' "$LIST" | grep -c . || true)
[ "$COUNT" -gt 0 ] || bf "build input closure is empty at $COMMIT"
LIST_SHA=$(printf '%s\n' "$LIST" | sha256sum | cut -d' ' -f1)
CONTENT_SHA=$( { printf '%s\n' "$LIST" | while IFS=$'\t' read -r P O; do
      [ -n "$P" ] || continue
      printf '%s\0' "$P"; git -C "$REPO" cat-file blob "$O"; done; } | sha256sum | cut -d' ' -f1)
# submodules in the closure: gitlink entries, "path<TAB>commit-oid"
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

echo "BUILD_INPUT_SPEC_NAME=$SPEC_NAME"
echo "BUILD_INPUT_SPEC_SHA256=$SPEC_SHA"
echo "SOURCE_TREE_GIT_OID=$TREE_OID"
echo "BUILD_INPUT_FILE_COUNT=$COUNT"
echo "BUILD_INPUT_LIST_SHA256=$LIST_SHA"
echo "BUILD_INPUT_CONTENT_SHA256=$CONTENT_SHA"
echo "SUBMODULE_COUNT=$SUBCOUNT"
echo "SUBMODULE_LIST_SHA256=$SUBSHA"
