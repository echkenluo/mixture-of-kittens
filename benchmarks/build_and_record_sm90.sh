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
#   - "the build environment is closed" was FALSE while ENV_PASS forwarded the
#     caller's PATH/HOME: make, nvcc and python3 were resolved under a
#     caller-controlled PATH, so a fake `make` could copy an old .so and a fake
#     `nvcc`/`python3` could print exactly the versions the record wanted. The
#     control plane (git, sha256sum, readlink, stat) resolved the same way.
#     ENV_SET now fixes every value in the tracked spec and this script
#     normalises its own PATH and clears Git/Make/compiler overrides first.
#   - "the Make variables that decide what gets compiled are pinned" was FALSE
#     too: PYTHON_INCLUDES, PYTORCH_INCLUDES and PYTORCH_LIBDIR are ?= in the
#     Makefile and were still inheritable, and nothing stopped MAKEFILES,
#     MAKEFLAGS, NVCC_CCBIN, CPATH, LIBRARY_PATH or PYTHONPATH from steering
#     the build. The build now runs under `env -i` with a tracked allowlist.
# Also fixed: nothing bound the wrapper, the helper or the validators, so a
# modified wrapper run from /tmp against a clean repo produced a valid-looking
# record. The tooling is now self-bound to the commit.
#
# Production mode (RECORD_MODE=production) takes NOTHING about the build from
# the caller: command, argv, output path, closure and probes all come from
# tracked specs at HEAD. The caller supplies only the artifact directory and
# the container image identity, which it DECLARES - nothing here verifies it,
# which is why those fields are recorded as *_DECLARED_BY_CALLER.
#
# Fixture mode (RECORD_MODE=fixture) exists so tests can drive an arbitrary
# command. It is structurally isolated: the record carries RECORD_MODE=fixture
# and check_formal_binding_sm90.sh refuses such a record outright, so a fixture
# can never stand in for a production build.
#
# Still NOT proven by any of this: that a rebuild reproduces the same bytes,
# that the declared image is the one the build actually ran in, and - until a
# real build runs - that the closure covers everything nvcc actually reads.
#
# Usage: build_and_record_sm90.sh <repo_dir> <artifact_dir>
#   The record is published INSIDE the artifact directory under a canonical
#   name. An earlier version took the path from the caller and mv -f'd onto it,
#   which could write outside the repo and silently overwrite existing evidence.
# Required env: TOOLCHAIN_IMAGE_ID TOOLCHAIN_IMAGE_REF TOOLCHAIN_IMAGE_REPO_DIGESTS
# Fixture-only env: BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC=<file>
set -euo pipefail
# CONTROL-PLANE NORMALISATION - and an honest statement of its limit.
#
# This normalises PATH and clears Git/Make/compiler overrides so that an
# ACCIDENTALLY dirty environment cannot steer git, sha256sum, readlink or
# stat. It is NOT a proof that the control plane was clean: a non-interactive
# bash reads BASH_ENV BEFORE this line executes, and a shell function defined
# there shadows any command on PATH. A script cannot establish its own clean
# entrypoint from inside itself.
#
# The real boundary belongs to a trusted orchestrator invoking
#   /usr/bin/env -i ... /bin/bash --noprofile --norc <this script>
# (or an execve with a fixed environment). That orchestrator does not exist
# yet, so every record carries unverified_external:clean_entrypoint and no
# record can be promoted to a receipt. The checks below are misuse protection,
# not an attestation.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
      GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CONFIG GIT_CONFIG_GLOBAL \
      GIT_CONFIG_SYSTEM GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS GIT_EXEC_PATH \
      GIT_TEMPLATE_DIR GIT_ATTR_NOSYSTEM GIT_CEILING_DIRECTORIES \
      MAKEFILES MAKEFLAGS GNUMAKEFLAGS MFLAGS \
      NVCC_CCBIN CUDAHOSTCXX CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH \
      LIBRARY_PATH LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME \
      PYTHONSTARTUP CC CXX 2>/dev/null || true
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
# fail closed on the entrypoint variables we CAN see, and on exported shell
# functions, which take precedence over PATH lookups
# printenv, not ${!V}: SHELLOPTS and BASHOPTS always exist as shell variables
# in bash, so reading the variable would reject every invocation. What matters
# is whether the value was INHERITED from the environment.
for V in BASH_ENV ENV SHELLOPTS BASHOPTS; do
  if [ -n "$(printenv "$V" || true)" ]; then
    echo "BUILD_RECORD_FAIL:$V is set; this wrapper must be started from a clean entrypoint (env -i ... bash --noprofile --norc)"
    exit 2
  fi
done
while IFS= read -r FN; do
  echo "BUILD_RECORD_FAIL:exported shell function $FN is present; a function shadows PATH lookups and the entrypoint is not clean"
  exit 2
done < <(declare -Fx | awk '{print $3}')
REPO=${1:?repo dir}; ARTDIR=${2:?artifact dir}
: "${TOOLCHAIN_IMAGE_ID:?TOOLCHAIN_IMAGE_ID required (unverified caller declaration)}"
: "${TOOLCHAIN_IMAGE_REF:?TOOLCHAIN_IMAGE_REF required (unverified caller declaration)}"
: "${TOOLCHAIN_IMAGE_REPO_DIGESTS:?TOOLCHAIN_IMAGE_REPO_DIGESTS required (unverified caller declaration; NONE only for local fixtures)}"
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
         benchmarks/build_command_spec.v2; do
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
  CMDSPEC=$(git cat-file blob "$SOURCE_COMMIT:benchmarks/build_command_spec.v2")
  BUILD_COMMAND_SPEC_SHA256=$(iget BUILD_COMMAND_SPEC_SHA256)
  BUILD_COMMAND_SPEC_NAME=$(iget BUILD_COMMAND_SPEC_NAME)
fi
mapfile -t ARGV < <(printf '%s\n' "$CMDSPEC" | grep '^ARGV=' | cut -d= -f2-)
[ "${#ARGV[@]}" -gt 0 ] || fail "command spec lists no ARGV entries"
OUTPUT_PATH=$(printf '%s\n' "$CMDSPEC" | grep '^OUTPUT=' | head -1 | cut -d= -f2-)
[ -n "$OUTPUT_PATH" ] || fail "command spec has no OUTPUT"
mapfile -t PROBES < <(printf '%s\n' "$CMDSPEC" | grep '^PROBE=' | cut -d= -f2-)
[ "${#PROBES[@]}" -gt 0 ] || fail "command spec lists no PROBE entries"
mapfile -t ENV_SET < <(printf '%s\n' "$CMDSPEC" | grep '^ENV_SET=' | cut -d= -f2-)
[ "${#ENV_SET[@]}" -gt 0 ] || fail "command spec declares no ENV_SET environment"
mapfile -t TOOLCHAIN_ROOTS < <(printf '%s\n' "$CMDSPEC" | grep '^TOOLCHAIN_ROOT=' | cut -d= -f2-)
[ "${#TOOLCHAIN_ROOTS[@]}" -gt 0 ] || fail "command spec declares no TOOLCHAIN_ROOT"
HOST_COMPILER=$(printf '%s\n' "$CMDSPEC" | grep '^HOST_COMPILER=' | head -1 | cut -d= -f2-)
[ -n "$HOST_COMPILER" ] || fail "command spec does not pin HOST_COMPILER"
case "$HOST_COMPILER" in /*) : ;; *) fail "HOST_COMPILER must be an absolute path: $HOST_COMPILER" ;; esac
BUILD_COMMAND_ARGV_JOINED=$(printf '%s ' "${ARGV[@]}" | sed 's/ $//')
# the pinned host compiler must actually be routed into nvcc by the argv
CCBIN_OK=0
for A in "${ARGV[@]}"; do
  case "$A" in NVCC=*"-ccbin $HOST_COMPILER"*) CCBIN_OK=1 ;; esac
done
[ "$CCBIN_OK" -eq 1 ] \
  || fail "no ARGV element routes -ccbin $HOST_COMPILER into NVCC; the host compiler would be chosen by search"
# the nvcc the command runs and the nvcc the probe measures must be the same
# absolute file, or the record describes a compiler that did not build this
NVCC_CMD=""
for A in "${ARGV[@]}"; do case "$A" in NVCC=*) NVCC_CMD=${A#NVCC=}; NVCC_CMD=${NVCC_CMD%% *} ;; esac; done
[ -n "$NVCC_CMD" ] || fail "command spec does not pin NVCC"
case "$NVCC_CMD" in /*) : ;; *) fail "NVCC must be an absolute path in the command spec, got $NVCC_CMD" ;; esac
NVCC_PROBE=""
for P in "${PROBES[@]}"; do case "$P" in nvcc\|*) NVCC_PROBE=${P#nvcc|}; NVCC_PROBE=${NVCC_PROBE%%|*} ;; esac; done
[ "$NVCC_PROBE" = "$NVCC_CMD" ] \
  || fail "the nvcc probe measures $NVCC_PROBE but the command runs $NVCC_CMD"
# the environment the build will see: FIXED VALUES from the tracked spec, with
# nothing taken from the caller. HOME points at an empty directory this run
# creates, so tool/user configuration cannot reach the build either.
BUILD_HOME=$ARTDIR/build_home
mkdir -p "$BUILD_HOME" || fail "cannot create the build HOME under the artifact dir"
[ -z "$(ls -A "$BUILD_HOME" 2>/dev/null)" ] || fail "build HOME $BUILD_HOME is not empty"
ENVARGS=()
for E in "${ENV_SET[@]}"; do
  case "$E" in
    [A-Z_]*=*) : ;;
    *) fail "ENV_SET entry is not NAME=value: $E" ;;
  esac
  E=${E//@ARTIFACT_HOME@/$BUILD_HOME}
  ENVARGS+=("$E")
done
# no trailing newline: a command substitution strips one, so the validator
# would otherwise hash a different string than the one encoded here
ENV_APPLIED=$(printf '%s\n' "${ENVARGS[@]}")
ENV_MANIFEST_B64=$(printf '%s' "$ENV_APPLIED" | base64 -w0)
ENV_APPLIED_SHA256=$(printf '%s' "$ENV_APPLIED" | sha256sum | cut -d' ' -f1)
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
  echo "### env_manifest_b64 $ENV_MANIFEST_B64"
  echo "### env_applied_sha256 $ENV_APPLIED_SHA256"
  echo "### start $BUILD_START_UTC"; } > "$BUILD_LOG_PATH"
set +e
# env -i: the build sees ONLY the allowlisted names. Anything that could steer
# make, nvcc, the header/library search or python imports is absent, not merely
# unused - MAKEFILES, MAKEFLAGS, GNUMAKEFLAGS, MFLAGS, NVCC_CCBIN, CPATH,
# CPLUS_INCLUDE_PATH, LIBRARY_PATH, LD_LIBRARY_PATH, PYTHONPATH included.
env -i "${ENVARGS[@]}" "${ARGV[@]}" >> "$BUILD_LOG_PATH" 2>&1
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
  POUT=$(env -i "${ENVARGS[@]}" "${PARGV[@]}" 2>&1); PRC=$?
  set -e
  printf '%s\n' "$POUT" >> "$PROBE_LOG_PATH"
  [ "$PRC" -eq 0 ] \
    || fail "probe $LABEL exited $PRC; the toolchain cannot be described and no record is published"
  PROBE_OUT[$LABEL]=$POUT
done
probe_line() { # label lineno
  printf '%s\n' "${PROBE_OUT[$1]:-}" | grep -v '^[[:space:]]*$' | sed -n "$2p" | tr -s ' ' | sed 's/^ //;s/ $//'
}
# || true: a probe whose output has no "release" line must fall through to the
# generic first-line extractor, not kill the script under set -e
MEASURED_NVCC_VERSION=$(printf '%s\n' "${PROBE_OUT[nvcc]:-}" | { grep -m1 'release' || true; } | tr -s ' ' | sed 's/^ //')
[ -n "$MEASURED_NVCC_VERSION" ] || MEASURED_NVCC_VERSION=$(probe_line nvcc 1)
MEASURED_HOST_COMPILER_VERSION=$(probe_line hostcc 1)
MEASURED_PYTHON_VERSION=$(probe_line python 1)
MEASURED_TORCH_VERSION=$(probe_line torch 1)
MEASURED_CUDA_VERSION=$(probe_line torch 2)
MEASURED_EXT_SUFFIX=$(probe_line ext_suffix 1)
MEASURED_PY_INCLUDE=$(probe_line py_include 1)
MEASURED_TORCH_INCLUDE=$(probe_line torch_include 1)
MEASURED_TORCH_LIBDIR=$(probe_line torch_libdir 1)
# a pinned include/lib path that the real interpreter and torch do not report
# is a fiction: it may not exist, or may point at another installation
argv_value() { for A in "${ARGV[@]}"; do case "$A" in "$1="*) printf '%s' "${A#$1=}"; return ;; esac; done; }
for PAIR in "PYTHON_INCLUDES:$MEASURED_PY_INCLUDE" \
            "PYTORCH_INCLUDES:$MEASURED_TORCH_INCLUDE" \
            "PYTORCH_LIBDIR:$MEASURED_TORCH_LIBDIR"; do
  K=${PAIR%%:*}; MEAS=${PAIR#*:}
  PINNED=$(argv_value "$K")
  [ -n "$PINNED" ] || fail "command spec does not pin $K"
  [ -n "$MEAS" ] || fail "$K could not be derived from the toolchain"
  [ "$PINNED" = "$MEAS" ] \
    || fail "$K pinned as [$PINNED] but the toolchain reports [$MEAS]"
  # and every path in it must exist under an allowed toolchain root
  for TOK in $MEAS; do
    D=${TOK#-I}; D=${D#-L}
    [ -d "$D" ] || fail "$K path $D does not exist"
    # realpath first: a lexical prefix check is defeated by a symlink
    DR=$(readlink -f "$D") || fail "$K path $D cannot be resolved"
    OK=0
    for R in "${TOOLCHAIN_ROOTS[@]}"; do
      RR=$(readlink -f "$R" 2>/dev/null) || continue
      case "$DR/" in "$RR"/*) OK=1 ;; esac
    done
    [ "$OK" -eq 1 ] \
      || fail "$K path $D resolves to $DR, outside the declared toolchain image roots"
  done
done
# ABI check: a python 3.11 image can compile something and still leave a file
# named cp312. The OUTPUT basename must match what this interpreter reports.
[ -n "$MEASURED_EXT_SUFFIX" ] || fail "ext_suffix could not be measured"
EXPECT_BASENAME=_C$MEASURED_EXT_SUFFIX
[ "$(basename "$OUTPUT_PATH")" = "$EXPECT_BASENAME" ] \
  || fail "OUTPUT basename $(basename "$OUTPUT_PATH") != _C\$EXT_SUFFIX ($EXPECT_BASENAME) reported by the interpreter"
for V in MEASURED_NVCC_VERSION MEASURED_HOST_COMPILER_VERSION MEASURED_PYTHON_VERSION \
         MEASURED_TORCH_VERSION MEASURED_CUDA_VERSION MEASURED_EXT_SUFFIX \
         MEASURED_PY_INCLUDE MEASURED_TORCH_INCLUDE MEASURED_TORCH_LIBDIR; do
  [ -n "${!V}" ] || fail "$V could not be measured from the probe output; no record is published"
done
PROBE_LOG_BYTES=$(stat -c %s "$PROBE_LOG_PATH")
PROBE_LOG_SHA256=$(sha256sum "$PROBE_LOG_PATH" | cut -d' ' -f1)

OUT=$ARTDIR/build_record.v4
if [ -e "$OUT" ] || [ -L "$OUT" ]; then fail "record $OUT already exists; evidence is never overwritten"; fi
TMP=$(mktemp "$ARTDIR/.buildrecord.XXXXXX")
trap 'rm -f "$TMP"' EXIT
{
  echo "BUILD_RECORD_SCHEMA=4"
  echo "RECORD_MODE=$RECORD_MODE"
  echo "PROVENANCE_CLASS=measured:source,inputs,submodules,build_side_tooling,command,env,toolchain_paths,output,logs,versions;declared_unverified:toolchain_image;unverified_external:clean_entrypoint"
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
  echo "BUILD_SIDE_TOOLING_FILE_COUNT=$(iget BUILD_SIDE_TOOLING_FILE_COUNT)"
  echo "BUILD_SIDE_TOOLING_LIST_SHA256=$(iget BUILD_SIDE_TOOLING_LIST_SHA256)"
  echo "BUILD_COMMAND_SPEC_NAME=$BUILD_COMMAND_SPEC_NAME"
  echo "BUILD_COMMAND_SPEC_SHA256=$BUILD_COMMAND_SPEC_SHA256"
  echo "BUILD_COMMAND_ARGV_JOINED=$BUILD_COMMAND_ARGV_JOINED"
  echo "TOOLCHAIN_IMAGE_ID_DECLARED_BY_CALLER=$TOOLCHAIN_IMAGE_ID"
  echo "TOOLCHAIN_IMAGE_REF_DECLARED_BY_CALLER=$TOOLCHAIN_IMAGE_REF"
  echo "TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER=$TOOLCHAIN_IMAGE_REPO_DIGESTS"
  echo "MEASURED_CUDA_VERSION=$MEASURED_CUDA_VERSION"
  echo "MEASURED_NVCC_VERSION=$MEASURED_NVCC_VERSION"
  echo "MEASURED_HOST_COMPILER_VERSION=$MEASURED_HOST_COMPILER_VERSION"
  echo "MEASURED_PYTHON_VERSION=$MEASURED_PYTHON_VERSION"
  echo "MEASURED_TORCH_VERSION=$MEASURED_TORCH_VERSION"
  echo "MEASURED_EXT_SUFFIX=$MEASURED_EXT_SUFFIX"
  echo "MEASURED_PY_INCLUDE=$MEASURED_PY_INCLUDE"
  echo "MEASURED_TORCH_INCLUDE=$MEASURED_TORCH_INCLUDE"
  echo "MEASURED_TORCH_LIBDIR=$MEASURED_TORCH_LIBDIR"
  echo "MEASURED_HOST_COMPILER_PATH=$HOST_COMPILER"
  echo "ENV_MANIFEST_B64=$ENV_MANIFEST_B64"
  echo "ENV_APPLIED_SHA256=$ENV_APPLIED_SHA256"
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
VOUT=$(bash "$REPO/benchmarks/validate_build_record_sm90.sh" "$TMP" --check-mode 2>&1) \
  || fail "generated record failed self-validation (not published): $(printf '%s' "$VOUT" | tail -1)"
# publish by hard link inside the same directory: link fails if the target
# exists, so a concurrent or repeated run can never clobber a good record
ln "$TMP" "$OUT" 2>/dev/null || fail "could not publish record to $OUT (already exists?)"
rm -f "$TMP"
trap - EXIT
echo "BUILD_RECORD_PATH:$OUT"
echo "BUILD_RECORD_SHA256:$(sha256sum "$OUT" | cut -d' ' -f1)"
echo "RECORD_MODE:$RECORD_MODE"
