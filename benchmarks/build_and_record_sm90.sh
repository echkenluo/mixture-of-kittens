#!/bin/bash
# Controlled build-and-record wrapper (tracked). Replaces the earlier
# make_build_record_sm90.sh, which was unsafe: it never ran a build. It read a
# caller-supplied .so and log, so anyone could point it at an old binary on a
# clean HEAD, hand it a plausible command string, and get a fully valid
# record. Refusing an SO hash as input closed one hole but established no
# build -> binary causality at all.
#
# What this does instead: it RUNS the build itself and records only what it
# observed.
#   1. the command comes from a spec FILE, one argv element per line, and is
#      exec'd directly as an argv array - no shell, so no quoting ambiguity
#      and no way for the recorded text to differ from what ran (the record
#      pins the spec file's sha256, and the argv it ran is that file)
#   2. the build closure comes from the tracked build_input_spec.v1; there is
#      no env var to narrow it. Dirty, untracked or submodule drift -> refuse
#   3. the output path is REMOVED before the build, so a pre-existing binary
#      cannot be adopted; afterwards the output must exist and be newer than
#      the build start, and it is hashed immediately
#   4. the build log is created exclusively by this execution (it must not
#      already exist) and the exit code must be 0
#   5. toolchain versions are MEASURED by probing the running toolchain, with
#      the raw probe output kept and hashed. The container image identity
#      cannot be measured from inside, so it is recorded as ATTESTED
#
# What it still does NOT prove: that a rebuild reproduces the same bytes, and
# that the attested image is the one the orchestrator claims. See the record's
# PROVENANCE_CLASS field.
#
# Usage: build_and_record_sm90.sh <repo_dir> <out_record>
# Required env:
#   BUILD_COMMAND_SPEC      file with one argv element per line
#   BUILD_OUTPUT_PATH       repo-relative path the build must create
#   BUILD_LOG_PATH          path this run will create exclusively
#   PROBE_LOG_PATH          path this run will create exclusively
#   TARGET_ARCH             SM90|SM100|SM103
#   TOOLCHAIN_IMAGE_ID TOOLCHAIN_IMAGE_REF TOOLCHAIN_IMAGE_REPO_DIGESTS
#                           attested by the orchestrator, never measured here
set -euo pipefail
REPO=${1:?repo dir}; OUT=${2:?output record path}
: "${BUILD_COMMAND_SPEC:?BUILD_COMMAND_SPEC required (argv spec file)}"
: "${BUILD_OUTPUT_PATH:?BUILD_OUTPUT_PATH required (repo-relative)}"
: "${BUILD_LOG_PATH:?BUILD_LOG_PATH required}"
: "${PROBE_LOG_PATH:?PROBE_LOG_PATH required}"
: "${TARGET_ARCH:?TARGET_ARCH required}"
: "${TOOLCHAIN_IMAGE_ID:?TOOLCHAIN_IMAGE_ID required (attested)}"
: "${TOOLCHAIN_IMAGE_REF:?TOOLCHAIN_IMAGE_REF required (attested)}"
: "${TOOLCHAIN_IMAGE_REPO_DIGESTS:?TOOLCHAIN_IMAGE_REPO_DIGESTS required (attested; NONE for local-only)}"
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SPEC_REPO_PATH=${BUILD_INPUT_SPEC_PATH:-benchmarks/build_input_spec.v1}
fail() { echo "BUILD_RECORD_FAIL:$1"; exit 2; }

[ -f "$BUILD_COMMAND_SPEC" ] || fail "BUILD_COMMAND_SPEC $BUILD_COMMAND_SPEC is not a regular file"
[ -e "$BUILD_LOG_PATH" ] && fail "BUILD_LOG_PATH $BUILD_LOG_PATH already exists (the log must be created by this execution)"
[ -e "$PROBE_LOG_PATH" ] && fail "PROBE_LOG_PATH $PROBE_LOG_PATH already exists (the probe log must be created by this execution)"
cd "$REPO"
# The closure is defined by the spec COMMITTED IN THE SOURCE BEING BUILT, not
# by a copy sitting next to this script: what gets compiled is the repo at
# HEAD, so its own committed spec is the authority (and it is what the
# packaging host will recompute against).
SOURCE_COMMIT=$(git rev-parse HEAD)
SPEC_CONTENT=$(git show "$SOURCE_COMMIT:$SPEC_REPO_PATH" 2>/dev/null) \
  || fail "build input spec $SPEC_REPO_PATH does not exist at $SOURCE_COMMIT"
printf '%s\n' "$SPEC_CONTENT" | head -1 | grep -q '^BUILD_INPUT_SPEC=1$' \
  || fail "build input spec has a bad schema line"
SPEC_PATHS=(); SPEC_SUBS=()
mapfile -t SPEC_PATHS < <(printf '%s\n' "$SPEC_CONTENT" | grep '^PATH=' | cut -d= -f2-)
mapfile -t SPEC_SUBS < <(printf '%s\n' "$SPEC_CONTENT" | grep '^SUBMODULE=' | cut -d= -f2-)
[ "${#SPEC_PATHS[@]}" -gt 0 ] || fail "build input spec lists no paths"

# --- input closure must be clean, including untracked and submodule drift ---
DIRTY=$(git status --porcelain -- "${SPEC_PATHS[@]}")
[ -z "$DIRTY" ] || { echo "BUILD_RECORD_FAIL:build inputs not clean vs HEAD (incl. untracked):"; echo "$DIRTY"; exit 2; }
for SM in "${SPEC_SUBS[@]}"; do
  ST=$(git submodule status -- "$SM" 2>/dev/null || true)
  [ -n "$ST" ] || fail "submodule $SM not registered"
  case "$ST" in
    " "*) : ;;
    *) echo "BUILD_RECORD_FAIL:submodule $SM drifted or uninitialized:"; echo "$ST"; exit 2 ;;
  esac
  SMD=$(git -C "$SM" status --porcelain 2>/dev/null || true)
  [ -z "$SMD" ] || { echo "BUILD_RECORD_FAIL:submodule $SM working tree dirty:"; echo "$SMD"; exit 2; }
done

# input identity comes from the SHARED object-based computation, so the
# packaging host can recompute exactly these numbers at this commit
IDENT=$(bash "$DIR/compute_build_inputs_sm90.sh" "$REPO" "$SOURCE_COMMIT" "$SPEC_REPO_PATH") \
  || fail "cannot compute build-input identity at $SOURCE_COMMIT"
iget() { printf '%s\n' "$IDENT" | grep "^$1=" | head -1 | cut -d= -f2-; }
SPEC_NAME=$(iget BUILD_INPUT_SPEC_NAME)
SPEC_SHA=$(iget BUILD_INPUT_SPEC_SHA256)
SOURCE_TREE_GIT_OID=$(iget SOURCE_TREE_GIT_OID)
BUILD_INPUT_FILE_COUNT=$(iget BUILD_INPUT_FILE_COUNT)
BUILD_INPUT_LIST_SHA256=$(iget BUILD_INPUT_LIST_SHA256)
BUILD_INPUT_CONTENT_SHA256=$(iget BUILD_INPUT_CONTENT_SHA256)
SUBMODULE_COUNT=$(iget SUBMODULE_COUNT)
SUBMODULE_LIST_SHA256=$(iget SUBMODULE_LIST_SHA256)
BUILD_COMMAND_SPEC_SHA256=$(sha256sum "$BUILD_COMMAND_SPEC" | cut -d' ' -f1)
mapfile -t ARGV < "$BUILD_COMMAND_SPEC"
[ "${#ARGV[@]}" -gt 0 ] || fail "BUILD_COMMAND_SPEC is empty"
for A in "${ARGV[@]}"; do [ -n "$A" ] || fail "BUILD_COMMAND_SPEC contains an empty argv element"; done
BUILD_COMMAND_ARGV_JOINED=$(printf '%s ' "${ARGV[@]}" | sed 's/ $//')

# --- the output must be produced by THIS execution ---
rm -f "$BUILD_OUTPUT_PATH"
[ -e "$BUILD_OUTPUT_PATH" ] && fail "could not remove pre-existing output $BUILD_OUTPUT_PATH"
BUILD_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date -u +%s)
set +e
( set -o pipefail; "${ARGV[@]}" ) > "$BUILD_LOG_PATH" 2>&1
BUILD_EXIT_CODE=$?
set -e
BUILD_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
[ "$BUILD_EXIT_CODE" -eq 0 ] || fail "build command exited $BUILD_EXIT_CODE (no record is published for a failed build)"
[ -f "$BUILD_OUTPUT_PATH" ] || fail "build produced no output at $BUILD_OUTPUT_PATH"
OUT_MTIME=$(stat -c %Y "$BUILD_OUTPUT_PATH")
[ "$OUT_MTIME" -ge "$START_EPOCH" ] \
  || fail "output $BUILD_OUTPUT_PATH is older than the build start (not produced by this execution)"
SO_BASENAME=$(basename "$BUILD_OUTPUT_PATH")
SO_BYTES=$(stat -c %s "$BUILD_OUTPUT_PATH")
SO_SHA256=$(sha256sum "$BUILD_OUTPUT_PATH" | cut -d' ' -f1)
BUILD_LOG_BYTES=$(stat -c %s "$BUILD_LOG_PATH")
BUILD_LOG_SHA256=$(sha256sum "$BUILD_LOG_PATH" | cut -d' ' -f1)

# --- MEASURED toolchain versions: probed from the toolchain that just ran ---
probe() { echo "### $*"; "$@" 2>&1 || echo "(probe failed rc=$?)"; }
{
  probe nvcc --version
  probe cc --version
  probe python3 --version
  probe python3 -c 'import torch; print(torch.__version__); print(torch.version.cuda)'
} > "$PROBE_LOG_PATH" 2>&1
PROBE_LOG_BYTES=$(stat -c %s "$PROBE_LOG_PATH")
PROBE_LOG_SHA256=$(sha256sum "$PROBE_LOG_PATH" | cut -d' ' -f1)
MEASURED_NVCC_VERSION=$(grep -m1 'release' "$PROBE_LOG_PATH" | tr -s ' ' | sed 's/^ //' || true)
[ -n "$MEASURED_NVCC_VERSION" ] || MEASURED_NVCC_VERSION="unavailable (see probe log)"
MEASURED_HOST_COMPILER_VERSION=$(sed -n '/^### cc --version/{n;p;}' "$PROBE_LOG_PATH" | tr -s ' ' || true)
[ -n "$MEASURED_HOST_COMPILER_VERSION" ] || MEASURED_HOST_COMPILER_VERSION="unavailable (see probe log)"
MEASURED_PYTHON_VERSION=$(sed -n '/^### python3 --version/{n;p;}' "$PROBE_LOG_PATH" | tr -s ' ' || true)
[ -n "$MEASURED_PYTHON_VERSION" ] || MEASURED_PYTHON_VERSION="unavailable (see probe log)"
MEASURED_TORCH_VERSION=$(sed -n "/^### python3 -c import torch/{n;p;}" "$PROBE_LOG_PATH" | tr -s ' ' || true)
[ -n "$MEASURED_TORCH_VERSION" ] || MEASURED_TORCH_VERSION="unavailable (see probe log)"
MEASURED_CUDA_VERSION=$(sed -n "/^### python3 -c import torch/{n;n;p;}" "$PROBE_LOG_PATH" | tr -s ' ' || true)
[ -n "$MEASURED_CUDA_VERSION" ] || MEASURED_CUDA_VERSION="unavailable (see probe log)"

OUTDIR=$(dirname "$OUT")
TMP=$(mktemp "$OUTDIR/.buildrecord.XXXXXX")
trap 'rm -f "$TMP"' EXIT
{
  echo "BUILD_RECORD_SCHEMA=2"
  echo "PROVENANCE_CLASS=measured:source,inputs,submodules,command,output,log,versions;attested:toolchain_image"
  echo "BUILD_START_UTC=$BUILD_START_UTC"
  echo "BUILD_END_UTC=$BUILD_END_UTC"
  echo "BUILD_EXIT_CODE=$BUILD_EXIT_CODE"
  echo "SOURCE_COMMIT=$SOURCE_COMMIT"
  echo "SOURCE_TREE_GIT_OID=$SOURCE_TREE_GIT_OID"
  echo "BUILD_INPUT_SPEC_NAME=$SPEC_NAME"
  echo "BUILD_INPUT_SPEC_SHA256=$SPEC_SHA"
  echo "BUILD_INPUT_FILE_COUNT=$BUILD_INPUT_FILE_COUNT"
  echo "BUILD_INPUT_LIST_SHA256=$BUILD_INPUT_LIST_SHA256"
  echo "BUILD_INPUT_CONTENT_SHA256=$BUILD_INPUT_CONTENT_SHA256"
  echo "SUBMODULE_COUNT=$SUBMODULE_COUNT"
  echo "SUBMODULE_LIST_SHA256=$SUBMODULE_LIST_SHA256"
  echo "BUILD_COMMAND_SPEC_SHA256=$BUILD_COMMAND_SPEC_SHA256"
  echo "BUILD_COMMAND_ARGV_JOINED=$BUILD_COMMAND_ARGV_JOINED"
  echo "TOOLCHAIN_IMAGE_ID_ATTESTED=$TOOLCHAIN_IMAGE_ID"
  echo "TOOLCHAIN_IMAGE_REF_ATTESTED=$TOOLCHAIN_IMAGE_REF"
  echo "TOOLCHAIN_IMAGE_REPO_DIGESTS_ATTESTED=$TOOLCHAIN_IMAGE_REPO_DIGESTS"
  echo "MEASURED_CUDA_VERSION=$MEASURED_CUDA_VERSION"
  echo "MEASURED_NVCC_VERSION=$MEASURED_NVCC_VERSION"
  echo "MEASURED_HOST_COMPILER_VERSION=$MEASURED_HOST_COMPILER_VERSION"
  echo "MEASURED_PYTHON_VERSION=$MEASURED_PYTHON_VERSION"
  echo "MEASURED_TORCH_VERSION=$MEASURED_TORCH_VERSION"
  echo "PROBE_LOG_SHA256=$PROBE_LOG_SHA256"
  echo "PROBE_LOG_BYTES=$PROBE_LOG_BYTES"
  echo "TARGET_ARCH=$TARGET_ARCH"
  echo "SO_BASENAME=$SO_BASENAME"
  echo "SO_BYTES=$SO_BYTES"
  echo "SO_SHA256=$SO_SHA256"
  echo "BUILD_LOG_SHA256=$BUILD_LOG_SHA256"
  echo "BUILD_LOG_BYTES=$BUILD_LOG_BYTES"
} > "$TMP"
chmod 444 "$TMP"
bash "$DIR/validate_build_record_sm90.sh" "$TMP" --check-mode >/dev/null \
  || fail "generated record failed self-validation (not published)"
mv -f "$TMP" "$OUT"
trap - EXIT
echo "BUILD_RECORD_SHA256:$(sha256sum "$OUT" | cut -d' ' -f1)"
