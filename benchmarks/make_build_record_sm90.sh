#!/bin/bash
# Build-record generator (tracked). Runs ONLY on the machine that performed the
# build, immediately after it, against the REAL artifacts. It measures bytes;
# it never accepts a hash as input, so a record cannot be manufactured from a
# manifest's EXPECTED_SO_SHA256 or from any other after-the-fact source.
#
# Preconditions it enforces (all fail-closed):
#   - the tracked build inputs are clean vs HEAD, INCLUDING untracked files, so
#     the recorded SOURCE_COMMIT actually describes what was compiled
#   - the .so is a regular file that exists now; its bytes and size are read
#     from disk, not supplied
#   - the build log exists; its bytes and size are read from disk
#   - the toolchain image identity is supplied as three separate fields
#     (id / ref / repo digests) and validated, never as a path
# Output is validated BEFORE publishing (same-dir temp, chmod, validate,
# rename under an EXIT trap), so a malformed record never lands and never
# clobbers a good one.
#
# Usage: make_build_record_sm90.sh <repo_dir> <so_path> <build_log> <out_record>
# Required env:
#   TOOLCHAIN_IMAGE_ID TOOLCHAIN_IMAGE_REF TOOLCHAIN_IMAGE_REPO_DIGESTS
#   BUILD_COMMAND_ARGV TARGET_ARCH
#   CUDA_VERSION NVCC_VERSION HOST_COMPILER_VERSION PYTHON_VERSION TORCH_VERSION
# Optional env: BUILD_INPUT_PATHS (space-separated, default "csrc Makefile")
set -euo pipefail
REPO=${1:?repo dir}; SO=${2:?built .so path}; LOG=${3:?build log path}; OUT=${4:?output record path}
: "${TOOLCHAIN_IMAGE_ID:?TOOLCHAIN_IMAGE_ID required}"
: "${TOOLCHAIN_IMAGE_REF:?TOOLCHAIN_IMAGE_REF required}"
: "${TOOLCHAIN_IMAGE_REPO_DIGESTS:?TOOLCHAIN_IMAGE_REPO_DIGESTS required (literal NONE for a local-only image)}"
: "${BUILD_COMMAND_ARGV:?BUILD_COMMAND_ARGV required (the exact command line that produced this .so)}"
: "${TARGET_ARCH:?TARGET_ARCH required (SM90|SM100|SM103)}"
: "${CUDA_VERSION:?CUDA_VERSION required}"
: "${NVCC_VERSION:?NVCC_VERSION required}"
: "${HOST_COMPILER_VERSION:?HOST_COMPILER_VERSION required}"
: "${PYTHON_VERSION:?PYTHON_VERSION required}"
: "${TORCH_VERSION:?TORCH_VERSION required}"
BUILD_INPUT_PATHS=${BUILD_INPUT_PATHS:-csrc Makefile}
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[ -f "$SO" ] || { echo "BUILD_RECORD_FAIL:so $SO is not a regular file"; exit 2; }
[ -f "$LOG" ] || { echo "BUILD_RECORD_FAIL:build log $LOG is not a regular file"; exit 2; }
cd "$REPO"
# shellcheck disable=SC2086
DIRTY=$(git status --porcelain -- $BUILD_INPUT_PATHS)
[ -z "$DIRTY" ] || { echo "BUILD_RECORD_FAIL:build inputs not clean vs HEAD (incl. untracked):"; echo "$DIRTY"; exit 2; }
SRC=$(git rev-parse HEAD)
# SOURCE_TREE_SHA256: the whole tracked tree at HEAD, so a record cannot be
# reused across a tree that differs anywhere
SOURCE_TREE_SHA256=$(git rev-parse "HEAD^{tree}" | sha256sum | cut -d' ' -f1)
# BUILD_INPUTS_SHA256: the exact bytes of the tracked build inputs, in a fixed
# order, each prefixed by its repo-relative path
# shellcheck disable=SC2086
BUILD_INPUTS_SHA256=$( { for F in $(git ls-files -- $BUILD_INPUT_PATHS | LC_ALL=C sort); do
      printf '%s\0' "$F"; cat "$F"; done; } | sha256sum | cut -d' ' -f1)
BUILD_SCRIPT=$(echo "$BUILD_INPUT_PATHS" | tr ' ' '\n' | grep -x 'Makefile' || true)
[ -n "$BUILD_SCRIPT" ] && [ -f "$BUILD_SCRIPT" ] \
  || { echo "BUILD_RECORD_FAIL:build script (Makefile) not found among BUILD_INPUT_PATHS"; exit 2; }
BUILD_SCRIPT_SHA256=$(sha256sum "$BUILD_SCRIPT" | cut -d' ' -f1)
BUILD_COMMAND_SHA256=$(printf '%s' "$BUILD_COMMAND_ARGV" | sha256sum | cut -d' ' -f1)
SO_BASENAME=$(basename "$SO")
SO_BYTES=$(stat -c %s "$SO")
SO_SHA256=$(sha256sum "$SO" | cut -d' ' -f1)
BUILD_LOG_BYTES=$(stat -c %s "$LOG")
BUILD_LOG_SHA256=$(sha256sum "$LOG" | cut -d' ' -f1)
BUILD_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

OUTDIR=$(dirname "$OUT")
TMP=$(mktemp "$OUTDIR/.buildrecord.XXXXXX")
trap 'rm -f "$TMP"' EXIT
{
  echo "BUILD_RECORD_SCHEMA=1"
  echo "BUILD_UTC=$BUILD_UTC"
  echo "SOURCE_COMMIT=$SRC"
  echo "SOURCE_TREE_SHA256=$SOURCE_TREE_SHA256"
  echo "BUILD_INPUTS_SHA256=$BUILD_INPUTS_SHA256"
  echo "BUILD_SCRIPT_SHA256=$BUILD_SCRIPT_SHA256"
  echo "BUILD_COMMAND_SHA256=$BUILD_COMMAND_SHA256"
  echo "BUILD_COMMAND_ARGV=$BUILD_COMMAND_ARGV"
  echo "TOOLCHAIN_IMAGE_ID=$TOOLCHAIN_IMAGE_ID"
  echo "TOOLCHAIN_IMAGE_REF=$TOOLCHAIN_IMAGE_REF"
  echo "TOOLCHAIN_IMAGE_REPO_DIGESTS=$TOOLCHAIN_IMAGE_REPO_DIGESTS"
  echo "CUDA_VERSION=$CUDA_VERSION"
  echo "NVCC_VERSION=$NVCC_VERSION"
  echo "HOST_COMPILER_VERSION=$HOST_COMPILER_VERSION"
  echo "PYTHON_VERSION=$PYTHON_VERSION"
  echo "TORCH_VERSION=$TORCH_VERSION"
  echo "TARGET_ARCH=$TARGET_ARCH"
  echo "SO_BASENAME=$SO_BASENAME"
  echo "SO_BYTES=$SO_BYTES"
  echo "SO_SHA256=$SO_SHA256"
  echo "BUILD_LOG_SHA256=$BUILD_LOG_SHA256"
  echo "BUILD_LOG_BYTES=$BUILD_LOG_BYTES"
} > "$TMP"
chmod 444 "$TMP"
bash "$DIR/validate_build_record_sm90.sh" "$TMP" --check-mode >/dev/null \
  || { echo "BUILD_RECORD_FAIL:generated record failed self-validation (not published)"; exit 2; }
mv -f "$TMP" "$OUT"
trap - EXIT
echo "EXPECTED_BUILD_RECORD_SHA256:$(sha256sum "$OUT" | cut -d' ' -f1)"
