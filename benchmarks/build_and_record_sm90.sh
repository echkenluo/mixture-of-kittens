#!/bin/bash
# Controlled build-and-record wrapper (tracked).
#
# RETRACTIONS from the previous version, both found by adversarial review and
# both correct:
#   - "the recorded command and the command that ran are the same object" was
#     FALSE. Hashing an argv FILE binds the file, not the program it names:
#     `bash /tmp/cmd.sh` leaves cmd.sh free to change after hashing, or to
#     simply copy an old .so into place. Production mode now takes its argv
#     from a TRACKED spec at the commit being built, so there is no external
#     file to swap, and the argv identity is the spec blob.
#   - "no environment variable can narrow the closure" was FALSE: this script
#     itself had added BUILD_INPUT_SPEC_PATH. That override is gone.
# Also fixed: nothing bound the wrapper, the helper or the validators, so a
# modified wrapper run from /tmp against a clean repo produced a valid-looking
# record. The tooling is now self-bound to the commit.
#
# Production mode (RECORD_MODE=production) takes NOTHING about the build from
# the caller: command, argv, output path, closure and probes all come from
# tracked specs at HEAD. The caller supplies only the artifact directory and
# the attested container image identity.
#
# Fixture mode (RECORD_MODE=fixture) exists so tests can drive an arbitrary
# command. It is structurally isolated: the record carries RECORD_MODE=fixture
# and check_formal_binding_sm90.sh refuses such a record outright, so a fixture
# can never stand in for a production build.
#
# Still NOT proven by any of this: that a rebuild reproduces the same bytes,
# that the attested image is the one the orchestrator claims, and - until a
# real build runs - that the closure covers everything nvcc actually reads.
#
# Usage: build_and_record_sm90.sh <repo_dir> <artifact_dir> <out_record>
# Required env: TOOLCHAIN_IMAGE_ID TOOLCHAIN_IMAGE_REF TOOLCHAIN_IMAGE_REPO_DIGESTS
# Fixture-only env: BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC=<file>
set -euo pipefail
REPO=${1:?repo dir}; ARTDIR=${2:?artifact dir}; OUT=${3:?output record path}
: "${TOOLCHAIN_IMAGE_ID:?TOOLCHAIN_IMAGE_ID required (attested)}"
: "${TOOLCHAIN_IMAGE_REF:?TOOLCHAIN_IMAGE_REF required (attested)}"
: "${TOOLCHAIN_IMAGE_REPO_DIGESTS:?TOOLCHAIN_IMAGE_REPO_DIGESTS required (attested; NONE only for local fixtures)}"
FIXTURE=${BUILD_RECORD_FIXTURE_MODE:-0}
fail() { echo "BUILD_RECORD_FAIL:$1"; exit 2; }

REPO=$(cd "$REPO" && pwd -P) || fail "repo dir unusable"
cd "$REPO"
SOURCE_COMMIT=$(git rev-parse HEAD 2>/dev/null) || fail "not a git repository"

# --- tooling self-binding: the code that produces this record must BE the
# code committed at this commit, and this script must be the tracked copy ---
SELF=$(readlink -f "${BASH_SOURCE[0]}")
CANON_SELF=$REPO/benchmarks/build_and_record_sm90.sh
[ "$SELF" = "$(readlink -f "$CANON_SELF" 2>/dev/null)" ] \
  || fail "wrapper is running from $SELF, not the repository's benchmarks/build_and_record_sm90.sh"
for T in benchmarks/build_and_record_sm90.sh benchmarks/compute_build_inputs_sm90.sh \
         benchmarks/validate_build_record_sm90.sh benchmarks/build_input_spec.v1 \
         benchmarks/build_command_spec.v1; do
  [ -f "$T" ] || fail "tooling file $T missing from the worktree"
  git cat-file -e "$SOURCE_COMMIT:$T" 2>/dev/null || fail "tooling file $T does not exist at $SOURCE_COMMIT"
  W=$(sha256sum "$T" | cut -d' ' -f1)
  B=$(git cat-file blob "$SOURCE_COMMIT:$T" | sha256sum | cut -d' ' -f1)
  [ "$W" = "$B" ] || fail "tooling file $T differs from its committed bytes at $SOURCE_COMMIT ($W != $B)"
done

# --- input + command identity, from git objects, spec path fixed in the helper ---
IDENT=$(bash "$REPO/benchmarks/compute_build_inputs_sm90.sh" "$REPO" "$SOURCE_COMMIT") \
  || fail "cannot compute build-input identity at $SOURCE_COMMIT"
iget() { printf '%s\n' "$IDENT" | grep "^$1=" | head -1 | cut -d= -f2-; }
SPEC=$(git cat-file blob "$SOURCE_COMMIT:benchmarks/build_input_spec.v1")
mapfile -t SPEC_PATHS < <(printf '%s\n' "$SPEC" | grep '^PATH=' | cut -d= -f2-)
mapfile -t SPEC_SUBS < <(printf '%s\n' "$SPEC" | grep '^SUBMODULE=' | cut -d= -f2-)

# --- closure must be clean, including untracked and submodule drift ---
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

# --- the command: tracked in production, caller-supplied only for fixtures ---
if [ "$FIXTURE" = "1" ]; then
  RECORD_MODE=fixture
  CMDSPEC_FILE=${BUILD_FIXTURE_COMMAND_SPEC:?BUILD_FIXTURE_COMMAND_SPEC required in fixture mode}
  [ -f "$CMDSPEC_FILE" ] || fail "fixture command spec $CMDSPEC_FILE is not a regular file"
  CMDSPEC=$(cat "$CMDSPEC_FILE")
  BUILD_COMMAND_SPEC_SHA256=$(sha256sum "$CMDSPEC_FILE" | cut -d' ' -f1)
  BUILD_COMMAND_SPEC_NAME=$(printf '%s\n' "$CMDSPEC" | grep '^NAME=' | head -1 | cut -d= -f2-)
  [ -n "$BUILD_COMMAND_SPEC_NAME" ] || fail "fixture command spec has no NAME"
else
  RECORD_MODE=production
  [ -z "${BUILD_FIXTURE_COMMAND_SPEC:-}" ] \
    || fail "BUILD_FIXTURE_COMMAND_SPEC is set outside fixture mode"
  CMDSPEC=$(git cat-file blob "$SOURCE_COMMIT:benchmarks/build_command_spec.v1")
  BUILD_COMMAND_SPEC_SHA256=$(iget BUILD_COMMAND_SPEC_SHA256)
  BUILD_COMMAND_SPEC_NAME=$(iget BUILD_COMMAND_SPEC_NAME)
fi
mapfile -t ARGV < <(printf '%s\n' "$CMDSPEC" | grep '^ARGV=' | cut -d= -f2-)
[ "${#ARGV[@]}" -gt 0 ] || fail "command spec lists no ARGV entries"
OUTPUT_PATH=$(printf '%s\n' "$CMDSPEC" | grep '^OUTPUT=' | head -1 | cut -d= -f2-)
[ -n "$OUTPUT_PATH" ] || fail "command spec has no OUTPUT"
mapfile -t PROBES < <(printf '%s\n' "$CMDSPEC" | grep '^PROBE=' | cut -d= -f2-)
[ "${#PROBES[@]}" -gt 0 ] || fail "command spec lists no PROBE entries"
BUILD_COMMAND_ARGV_JOINED=$(printf '%s ' "${ARGV[@]}" | sed 's/ $//')
# TARGET_ARCH is derived from the command, not asserted alongside it
TARGET_ARCH=""
for A in "${ARGV[@]}"; do case "$A" in ARCH=*) TARGET_ARCH=${A#ARCH=} ;; esac; done
[ -n "$TARGET_ARCH" ] || fail "command spec does not pin ARCH=; TARGET_ARCH cannot be derived"

# --- path safety: output stays inside the repo and is never a symlink ---
case "$OUTPUT_PATH" in
  /*) fail "OUTPUT must be repository-relative, got absolute $OUTPUT_PATH" ;;
  *..*) fail "OUTPUT may not contain '..': $OUTPUT_PATH" ;;
esac
OUTDIR_ABS=$(cd "$(dirname "$OUTPUT_PATH")" 2>/dev/null && pwd -P) \
  || fail "OUTPUT directory $(dirname "$OUTPUT_PATH") does not exist in the repo"
case "$OUTDIR_ABS/" in "$REPO"/*) : ;; *) fail "OUTPUT resolves outside the repository: $OUTDIR_ABS" ;; esac
if [ -L "$OUTPUT_PATH" ]; then fail "OUTPUT $OUTPUT_PATH is a symlink; refusing to follow it"; fi
# --- artifact dir: real directory inside which logs are created exclusively ---
[ -d "$ARTDIR" ] || fail "artifact dir $ARTDIR is not a directory"
if [ -L "$ARTDIR" ]; then fail "artifact dir $ARTDIR is a symlink"; fi
ARTDIR=$(cd "$ARTDIR" && pwd -P)
BUILD_LOG_PATH=$ARTDIR/build.log
PROBE_LOG_PATH=$ARTDIR/probe.log
set -o noclobber
{ : > "$BUILD_LOG_PATH"; } 2>/dev/null || fail "build log $BUILD_LOG_PATH already exists or is not creatable"
{ : > "$PROBE_LOG_PATH"; } 2>/dev/null || fail "probe log $PROBE_LOG_PATH already exists or is not creatable"
set +o noclobber

# --- the output must be produced by THIS execution ---
rm -f "$OUTPUT_PATH"
if [ -e "$OUTPUT_PATH" ]; then fail "could not remove pre-existing output $OUTPUT_PATH"; fi
BUILD_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date -u +%s)
# the log carries the execution's own frame - what ran, at which commit, when
# it started and how it ended - so a log can be audited on its own and is
# never empty even for a silent command
{ echo "### build_and_record_sm90.sh"
  echo "### source_commit $SOURCE_COMMIT"
  echo "### record_mode $RECORD_MODE"
  echo "### argv $BUILD_COMMAND_ARGV_JOINED"
  echo "### start $BUILD_START_UTC"; } > "$BUILD_LOG_PATH"
set +e
"${ARGV[@]}" >> "$BUILD_LOG_PATH" 2>&1
BUILD_EXIT_CODE=$?
set -e
BUILD_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{ echo "### exit $BUILD_EXIT_CODE"; echo "### end $BUILD_END_UTC"; } >> "$BUILD_LOG_PATH"
[ "$BUILD_EXIT_CODE" -eq 0 ] || fail "build command exited $BUILD_EXIT_CODE (no record is published for a failed build)"
if [ -L "$OUTPUT_PATH" ]; then fail "build created $OUTPUT_PATH as a symlink; refusing to hash the target"; fi
[ -f "$OUTPUT_PATH" ] || fail "build produced no output at $OUTPUT_PATH"
OUT_REAL=$(readlink -f "$OUTPUT_PATH")
case "$OUT_REAL" in "$REPO"/*) : ;; *) fail "build output resolves outside the repository: $OUT_REAL" ;; esac
OUT_MTIME=$(stat -c %Y "$OUTPUT_PATH")
[ "$OUT_MTIME" -ge "$START_EPOCH" ] \
  || fail "output $OUTPUT_PATH is older than the build start (not produced by this execution)"
SO_BASENAME=$(basename "$OUTPUT_PATH")
SO_BYTES=$(stat -c %s "$OUTPUT_PATH")
SO_SHA256=$(sha256sum "$OUTPUT_PATH" | cut -d' ' -f1)
BUILD_LOG_BYTES=$(stat -c %s "$BUILD_LOG_PATH")
BUILD_LOG_SHA256=$(sha256sum "$BUILD_LOG_PATH" | cut -d' ' -f1)

# --- probes: the tools this command names, and every probe must succeed ---
declare -A PROBE_OUT=()
for P in "${PROBES[@]}"; do
  IFS='|' read -r -a PARR <<< "$P"
  LABEL=${PARR[0]}; PARGV=("${PARR[@]:1}")
  [ -n "$LABEL" ] && [ "${#PARGV[@]}" -gt 0 ] || fail "malformed PROBE entry: $P"
  echo "### probe $LABEL: ${PARGV[*]}" >> "$PROBE_LOG_PATH"
  set +e
  POUT=$("${PARGV[@]}" 2>&1); PRC=$?
  set -e
  printf '%s\n' "$POUT" >> "$PROBE_LOG_PATH"
  [ "$PRC" -eq 0 ] \
    || fail "probe $LABEL exited $PRC; the toolchain cannot be described and no record is published"
  PROBE_OUT[$LABEL]=$POUT
done
probe_line() { # label lineno
  printf '%s\n' "${PROBE_OUT[$1]:-}" | grep -v '^[[:space:]]*$' | sed -n "$2p" | tr -s ' ' | sed 's/^ //;s/ $//'
}
MEASURED_NVCC_VERSION=$(printf '%s\n' "${PROBE_OUT[nvcc]:-}" | grep -m1 'release' | tr -s ' ' | sed 's/^ //')
[ -n "$MEASURED_NVCC_VERSION" ] || MEASURED_NVCC_VERSION=$(probe_line nvcc 1)
MEASURED_HOST_COMPILER_VERSION=$(probe_line cc 1)
MEASURED_PYTHON_VERSION=$(probe_line python 1)
MEASURED_TORCH_VERSION=$(probe_line torch 1)
MEASURED_CUDA_VERSION=$(probe_line torch 2)
for V in MEASURED_NVCC_VERSION MEASURED_HOST_COMPILER_VERSION MEASURED_PYTHON_VERSION \
         MEASURED_TORCH_VERSION MEASURED_CUDA_VERSION; do
  [ -n "${!V}" ] || fail "$V could not be measured from the probe output; no record is published"
done
PROBE_LOG_BYTES=$(stat -c %s "$PROBE_LOG_PATH")
PROBE_LOG_SHA256=$(sha256sum "$PROBE_LOG_PATH" | cut -d' ' -f1)

TMP=$(mktemp "$(dirname "$OUT")/.buildrecord.XXXXXX")
trap 'rm -f "$TMP"' EXIT
{
  echo "BUILD_RECORD_SCHEMA=2"
  echo "RECORD_MODE=$RECORD_MODE"
  echo "PROVENANCE_CLASS=measured:source,inputs,submodules,tooling,command,output,logs,versions;attested:toolchain_image"
  echo "BUILD_START_UTC=$BUILD_START_UTC"
  echo "BUILD_END_UTC=$BUILD_END_UTC"
  echo "BUILD_EXIT_CODE=$BUILD_EXIT_CODE"
  echo "SOURCE_COMMIT=$SOURCE_COMMIT"
  echo "SOURCE_TREE_GIT_OID=$(iget SOURCE_TREE_GIT_OID)"
  echo "BUILD_INPUT_SPEC_NAME=$(iget BUILD_INPUT_SPEC_NAME)"
  echo "BUILD_INPUT_SPEC_SHA256=$(iget BUILD_INPUT_SPEC_SHA256)"
  echo "BUILD_INPUT_FILE_COUNT=$(iget BUILD_INPUT_FILE_COUNT)"
  echo "BUILD_INPUT_LIST_SHA256=$(iget BUILD_INPUT_LIST_SHA256)"
  echo "BUILD_INPUT_CONTENT_SHA256=$(iget BUILD_INPUT_CONTENT_SHA256)"
  echo "SUBMODULE_COUNT=$(iget SUBMODULE_COUNT)"
  echo "SUBMODULE_LIST_SHA256=$(iget SUBMODULE_LIST_SHA256)"
  echo "TOOLING_FILE_COUNT=$(iget TOOLING_FILE_COUNT)"
  echo "TOOLING_LIST_SHA256=$(iget TOOLING_LIST_SHA256)"
  echo "BUILD_COMMAND_SPEC_NAME=$BUILD_COMMAND_SPEC_NAME"
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
bash "$REPO/benchmarks/validate_build_record_sm90.sh" "$TMP" --check-mode >/dev/null \
  || fail "generated record failed self-validation (not published)"
mv -f "$TMP" "$OUT"
trap - EXIT
echo "BUILD_RECORD_SHA256:$(sha256sum "$OUT" | cut -d' ' -f1)"
echo "RECORD_MODE:$RECORD_MODE"
