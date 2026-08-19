#!/bin/bash
# Build-record contract tests (tracked; no GPU, no real build, no docker).
#
# Scope, stated before any number: this file tests a CONTRACT IMPLEMENTATION
# UNDER ADVERSARIAL REVIEW. It does not establish build -> binary causality for
# any real binary: no nvcc build has been run, the production command here is a
# tracked fixture script, and the current .so still has no record.
#
# The A-cases are the important ones: each attack is first run against the
# PREVIOUS implementation (git 8e0035f, extracted into a temp dir) to show it
# succeeded there, then against the current one to show it is refused. A test
# that only shows the current code refusing proves nothing about whether the
# refusal is new.
#
# Usage: bash test_build_record_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/.." && pwd)
VALB=$DIR/validate_build_record_sm90.sh
BIND=$DIR/check_formal_binding_sm90.sh
CBI=$DIR/compute_build_inputs_sm90.sh
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "BR_$1_PASS"; PASS=$((PASS+1)); else echo "BR_$1_FAIL"; FAIL=$((FAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }
H64() { printf "$1%.0s" $(seq 1 64); }
H40() { printf "$1%.0s" $(seq 1 40); }
GIT="git -c user.email=t@e.com -c user.name=t -c protocol.file.allow=always"

TMPD=$(mktemp -d)
OLDBM=$TMPD/old
mkdir -p "$OLDBM"
git -C "$REPO" archive 8e0035f benchmarks 2>/dev/null | tar x -C "$OLDBM" \
  || { echo "BR_SETUP_FAIL:cannot extract the previous implementation (8e0035f)"; exit 3; }
OLDWRAP=$OLDBM/benchmarks/build_and_record_sm90.sh
[ -f "$OLDWRAP" ] || { echo "BR_SETUP_FAIL:old wrapper missing from the extracted tree"; exit 3; }

# ---------- fixture repository: a build host with tracked tooling ----------
SUBSRC=$TMPD/subsrc
mkdir -p "$SUBSRC"; printf '// tk v1\n' > "$SUBSRC/tk.h"
( cd "$SUBSRC" && $GIT init -q && $GIT add -A && $GIT commit -qm tk1 ) >/dev/null 2>&1
FIX=$TMPD/buildhost
mkdir -p "$FIX/csrc" "$FIX/mok" "$FIX/benchmarks" "$FIX/tools"
printf '// fixture kernel\n' > "$FIX/csrc/bindings.cu"
printf 'all:\n\techo build\n' > "$FIX/Makefile"
printf '[build-system]\n' > "$FIX/pyproject.toml"
NONCE="nonce-$$-$(date -u +%s)"
# the production build command is a TRACKED script inside the closure
{ echo '#!/bin/bash'; echo 'set -e'; echo 'echo "fixture build running"'
  echo 'echo "SAW PYTHON_INCLUDES=${PYTHON_INCLUDES:-}"'
  echo 'echo "SAW PYTORCH_LIBDIR=${PYTORCH_LIBDIR:-}"'
  echo 'echo "SAW MAKEFILES=${MAKEFILES:-}"'
  echo 'echo "SAW NVCC_CCBIN=${NVCC_CCBIN:-}"'
  echo 'echo "SAW CPATH=${CPATH:-}"'
  echo 'echo "SAW PATH=${PATH:-}"'
  echo 'echo "SAW HOME=${HOME:-}"'
  echo 'echo "SAW TMPDIR=${TMPDIR:-}"'
  echo "printf 'built %s\\n' \"$NONCE\" > mok/_Cfixture.so"; } > "$FIX/tools/fake_build.sh"
chmod +x "$FIX/tools/fake_build.sh"
{ echo "BUILD_INPUT_SPEC=1"; echo "NAME=fixture-inputs-v1"; echo "PATH=Makefile"
  echo "PATH=pyproject.toml"; echo "PATH=csrc"; echo "PATH=tools"
  echo "SUBMODULE=third_party/tk"; } > "$FIX/benchmarks/build_input_spec.v1"
{ echo "BUILD_COMMAND_SPEC=2"; echo "NAME=fixture-command-v1"
  echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
  echo "ENV_SET=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  echo "ENV_SET=HOME=@ARTIFACT_HOME@"; echo "ENV_SET=TMPDIR=@ARTIFACT_TMP@"
  echo "ENV_SET=PYTHONNOUSERSITE=1"
  echo "ENV_SET=PYTHONPATH="; echo "ENV_SET=TEST_EQ=a=b"
  echo "ENV_SET=LANG=C.UTF-8"; echo "TOOLCHAIN_ROOT=/usr"
  echo "ARGV=PYTHON_INCLUDES=-I/usr/include"
  echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"
  echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"
  echo "PROBE=nvcc|/bin/echo|nvcc release 13.0"
  echo "PROBE=hostcc|/bin/echo|(fixture host compiler) 12.3.0"
  echo "PROBE=python|printf|Python 3.12.3"
  echo "PROBE=torch|printf|2.11.0+cu130\\n13.0\\n"
  echo "PROBE=ext_suffix|printf|fixture.so"
  echo "PROBE=py_include|printf|-I/usr/include"
  echo "PROBE=torch_include|printf|-I/usr/include"
  echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$FIX/benchmarks/build_command_spec.v2"
for T in build_and_record_sm90.sh compute_build_inputs_sm90.sh validate_build_record_sm90.sh; do
  cp "$DIR/$T" "$FIX/benchmarks/$T"
done
( cd "$FIX" && $GIT init -q && $GIT submodule add -q "$SUBSRC" third_party/tk \
  && $GIT add -A && $GIT commit -qm build-inputs ) >/dev/null 2>&1
FIXSRC=$($GIT -C "$FIX" rev-parse HEAD)
FIXWRAP=$FIX/benchmarks/build_and_record_sm90.sh

newart() { local D=$TMPD/art.$1; mkdir -p "$D"; echo "$D"; }
# the artifact dir is computed by the CALLER: runwrap_at runs in a command
# substitution, so anything it assigns would be lost with its subshell
runwrap_at() { # artifact-dir
  ( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF="build:cu130" \
      TOOLCHAIN_IMAGE_REPO_DIGESTS="${N_DIGESTS:-registry.local/build@sha256:$(H64 3)}" \
      bash "$FIXWRAP" "$FIX" "$1" 2>&1 )
}

# ---------- A: attacks that worked on 8e0035f and must now be refused ------
# A1 external command script: the old design hashed the argv FILE, leaving the
# program it names free to copy an old binary into place
printf 'STALE PREBUILT BINARY\n' > "$TMPD/old_prebuilt.so"
EVIL=$TMPD/evil.sh
{ echo '#!/bin/bash'; echo 'echo "pretending to build"'
  echo "cp $TMPD/old_prebuilt.so mok/_Cfixture.so"; } > "$EVIL"
chmod +x "$EVIL"
printf 'bash\n%s\n' "$EVIL" > "$TMPD/evil.argv"
printf 'bash\ntools/fake_build.sh\n' > "$TMPD/tracked.argv"
set +e
OLDOUT=$( cd "$FIX" && BUILD_COMMAND_SPEC="$TMPD/evil.argv" BUILD_OUTPUT_PATH=mok/_Cfixture.so \
    BUILD_LOG_PATH="$TMPD/a1.build.log" PROBE_LOG_PATH="$TMPD/a1.probe.log" TARGET_ARCH=SM90 \
    TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
    bash "$OLDWRAP" "$FIX" "$TMPD/a1.old.record" 2>&1 ); OLDRC=$?
set -u
OLD_TOOK_STALE=1
[ "$OLDRC" -eq 0 ] && grep -q "^SO_SHA256=$(sha256sum "$TMPD/old_prebuilt.so" | cut -d' ' -f1)\$" "$TMPD/a1.old.record" 2>/dev/null || OLD_TOOK_STALE=0
report A1a_old_accepted_prebuilt_binary $([ "$OLD_TOOK_STALE" -eq 1 ] && echo 0 || echo 1)
echo "  A1a old wrapper rc=$OLDRC recorded the pre-built binary as if it had built it"
set +e
NEWOUT=$( cd "$FIX" && BUILD_FIXTURE_COMMAND_SPEC="$TMPD/evil.argv" \
    TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
    TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
    bash "$FIXWRAP" "$FIX" "$(newart a1new)" 2>&1 ); NEWRC=$?
set -u
[ "$NEWRC" -eq 2 ] && has1 "$NEWOUT" '^BUILD_RECORD_FAIL:BUILD_FIXTURE_COMMAND_SPEC is set outside fixture mode$' \
  && [ ! -f "$TMPD/art.a1new/build_record.v4" ]; report A1b_new_refuses_external_command $?
echo "  A1b new wrapper rc=$NEWRC (production mode takes no command from the caller)"

# A2 a modified wrapper run from outside the repository
cp "$FIXWRAP" "$TMPD/tampered_wrapper.sh"
set +e
OLDOUT=$( cd "$FIX" && BUILD_COMMAND_SPEC="$TMPD/evil.argv" BUILD_OUTPUT_PATH=mok/_Cfixture.so \
    BUILD_LOG_PATH="$TMPD/a2.build.log" PROBE_LOG_PATH="$TMPD/a2.probe.log" TARGET_ARCH=SM90 \
    TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
    bash "$OLDWRAP" "$FIX" "$TMPD/a2.old.record" 2>&1 ); OLDRC=$?
set -u
[ "$OLDRC" -eq 0 ] && [ -f "$TMPD/a2.old.record" ]; report A2a_old_ran_from_outside_repo $?
echo "  A2a old wrapper rc=$OLDRC ran happily from $OLDBM (outside the repo it recorded)"
set +e
NEWOUT=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
    TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
    bash "$TMPD/tampered_wrapper.sh" "$FIX" "$(newart a2new)" 2>&1 ); NEWRC=$?
set -u
[ "$NEWRC" -eq 2 ] && has1 "$NEWOUT" '^BUILD_RECORD_FAIL:wrapper is running from .*, not the repository.s benchmarks/build_and_record_sm90\.sh$'; report A2b_new_refuses_foreign_wrapper $?
echo "  A2b new wrapper rc=$NEWRC (self-binding rejects a copy run from elsewhere)"

# A3 BUILD_INPUT_SPEC_PATH: a caller override of the closure
{ echo "BUILD_INPUT_SPEC=1"; echo "NAME=narrow"; echo "PATH=csrc"; } > "$FIX/benchmarks/narrow.spec"
( cd "$FIX" && $GIT add -A && { $GIT commit -qm narrow || $GIT diff --quiet HEAD; } ) >/dev/null 2>&1 \
  || { echo "TEST_HARNESS_FAIL:fixture commit 'narrow' failed"; exit 1; }
NARROWSRC=$($GIT -C "$FIX" rev-parse HEAD)
set +e
OLDOUT=$( cd "$FIX" && BUILD_INPUT_SPEC_PATH=benchmarks/narrow.spec BUILD_COMMAND_SPEC="$TMPD/evil.argv" \
    BUILD_OUTPUT_PATH=mok/_Cfixture.so BUILD_LOG_PATH="$TMPD/a3.build.log" \
    PROBE_LOG_PATH="$TMPD/a3.probe.log" TARGET_ARCH=SM90 \
    TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
    bash "$OLDWRAP" "$FIX" "$TMPD/a3.old.record" 2>&1 ); OLDRC=$?
set -u
OLDCOUNT=$(grep '^BUILD_INPUT_FILE_COUNT=' "$TMPD/a3.old.record" 2>/dev/null | cut -d= -f2)
[ "$OLDRC" -eq 0 ] && [ "${OLDCOUNT:-0}" -eq 1 ]
RA3=$?; [ "$RA3" -eq 0 ] || { echo "--- A3a old wrapper output (rc=$OLDRC) ---"; printf '%s\n' "$OLDOUT" | tail -5; }
report A3a_old_honoured_narrowed_closure "$RA3"
echo "  A3a old wrapper rc=$OLDRC recorded a closure of ${OLDCOUNT:-?} file(s) instead of the full spec"
set +e
ART=$(newart a3new)
NEWOUT=$(BUILD_INPUT_SPEC_PATH=benchmarks/narrow.spec runwrap_at "$ART"); NEWRC=$?
REC=$ART/build_record.v4
set -u
NEWCOUNT=$(grep '^BUILD_INPUT_FILE_COUNT=' "$REC" 2>/dev/null | cut -d= -f2)
[ "$NEWRC" -eq 0 ] && [ "${NEWCOUNT:-0}" -eq 4 ]; report A3b_new_ignores_spec_override $?
echo "  A3b new wrapper rc=$NEWRC closure=${NEWCOUNT:-?} files, want 4 (override ignored; spec path is fixed)"

# ---------- N: the current implementation's own gates ----------
set +e
ART=$(newart n1)
O=$(runwrap_at "$ART"); R=$?
REC_OK=$ART/build_record.v4
set -u
RECSHA=$(sha256sum "$REC_OK" 2>/dev/null | cut -d' ' -f1)
[ "$R" -eq 0 ] && has1 "$O" "^BUILD_RECORD_SHA256:$RECSHA\$" && has1 "$O" '^RECORD_MODE:production$' \
  && [ "$(stat -c %a "$REC_OK")" = "444" ]; report N1_production_positive $?
echo "  N1 rc=$R want=0(production record published)"
[ "$R" -eq 0 ] || echo "  N1 actual: $(printf '%s\n' "$O" | tail -1)"
grep -q "$NONCE" "$FIX/mok/_Cfixture.so" 2>/dev/null \
  && [ "$(grep '^SO_SHA256=' "$REC_OK" | cut -d= -f2)" = "$(sha256sum "$FIX/mok/_Cfixture.so" | cut -d' ' -f1)" ] \
  && [ "$(grep '^BUILD_COMMAND_ARGV_JOINED=' "$REC_OK" | cut -d= -f2-)" = "bash tools/fake_build.sh ARCH=SM90 NVCC=/bin/echo -ccbin /bin/echo PYTHON_INCLUDES=-I/usr/include PYTORCH_INCLUDES=-I/usr/include PYTORCH_LIBDIR=-L/usr/lib" ] \
  && [ "$(grep '^TARGET_ARCH=' "$REC_OK" | cut -d= -f2)" = "SM90" ]; report N2_argv_and_output_from_tracked_spec $?
echo "  N2 argv, ARCH and output all come from the tracked command spec"
GOODSO=$TMPD/built.so; cp "$FIX/mok/_Cfixture.so" "$GOODSO"
GOODREC=$TMPD/rec.good; cp "$REC_OK" "$GOODREC"; chmod 444 "$GOODREC"
# Make-style variables cannot be injected from the environment
set +e
ART=$(newart n3)
O=$(SRC=/evil.cu NVCC=/evil-nvcc PYTHON=/evil-python THUNDERKITTENS_ROOT=/evil ARCH=SM100 \
    OUT=/tmp/escaped.so PYTHON_INCLUDES=-I/tmp/evil PYTORCH_LIBDIR=-L/tmp/evil \
    MAKEFILES=/tmp/evil.mk NVCC_CCBIN=/tmp/evil-g++ CPATH=/tmp/evil \
    runwrap_at "$ART"); R=$?
REC=$ART/build_record.v4
set -u
[ "$R" -eq 0 ] \
  && [ "$(grep '^BUILD_COMMAND_ARGV_JOINED=' "$REC" | cut -d= -f2-)" = "bash tools/fake_build.sh ARCH=SM90 NVCC=/bin/echo -ccbin /bin/echo PYTHON_INCLUDES=-I/usr/include PYTORCH_INCLUDES=-I/usr/include PYTORCH_LIBDIR=-L/usr/lib" ] \
  && [ "$(grep '^TARGET_ARCH=' "$REC" | cut -d= -f2)" = "SM90" ] \
  && [ ! -f /tmp/escaped.so ]; report N3_make_vars_not_injectable $?
echo "  N3 rc=$R SRC/NVCC/PYTHON/TK/ARCH/OUT from the environment changed nothing"
# and the build itself saw NONE of the injected compile-steering variables
BL=$ART/build.log
grep -q '^SAW PYTHON_INCLUDES=$' "$BL" && grep -q '^SAW PYTORCH_LIBDIR=$' "$BL" \
  && grep -q '^SAW MAKEFILES=$' "$BL" && grep -q '^SAW NVCC_CCBIN=$' "$BL" \
  && grep -q '^SAW CPATH=$' "$BL"; report N3b_env_closure_holds $?
echo "  N3b the build saw empty PYTHON_INCLUDES/PYTORCH_LIBDIR/MAKEFILES/NVCC_CCBIN/CPATH"
# the same injection reaches the build on the previous implementation
OLDART=$TMPD/oldenv; mkdir -p "$OLDART"
set +e
OLDOUT=$( cd "$FIX" && PYTHON_INCLUDES=-I/tmp/evil PYTORCH_LIBDIR=-L/tmp/evil \
    MAKEFILES=/tmp/evil.mk NVCC_CCBIN=/tmp/evil-g++ CPATH=/tmp/evil \
    BUILD_COMMAND_SPEC="$TMPD/tracked.argv" BUILD_OUTPUT_PATH=mok/_Cfixture.so \
    BUILD_LOG_PATH="$OLDART/build.log" PROBE_LOG_PATH="$OLDART/probe.log" TARGET_ARCH=SM90 \
    TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
    bash "$OLDWRAP" "$FIX" "$TMPD/env.old.record" 2>&1 ); OLDRC=$?
set -u
grep -q '^SAW PYTHON_INCLUDES=-I/tmp/evil$' "$OLDART/build.log" 2>/dev/null \
  && grep -q '^SAW MAKEFILES=/tmp/evil.mk$' "$OLDART/build.log" 2>/dev/null \
  && grep -qF 'SAW NVCC_CCBIN=/tmp/evil-g++' "$OLDART/build.log" 2>/dev/null; report N3c_old_leaked_env_into_build $?
echo "  N3c old wrapper rc=$OLDRC let PYTHON_INCLUDES/MAKEFILES/NVCC_CCBIN reach the build"
# tooling tamper in the worktree
cp "$FIX/benchmarks/compute_build_inputs_sm90.sh" "$TMPD/cbi.bak"
echo "# tampered" >> "$FIX/benchmarks/compute_build_inputs_sm90.sh"
set +e
ART=$(newart n4)
O=$(runwrap_at "$ART"); R=$?
REC=$ART/build_record.v4
set -u
cp "$TMPD/cbi.bak" "$FIX/benchmarks/compute_build_inputs_sm90.sh"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:tooling file benchmarks/compute_build_inputs_sm90\.sh differs from its committed bytes at [0-9a-f]{40} \([0-9a-f]{64} != [0-9a-f]{64}\)$'; report N4_tooling_tamper_refused $?
echo "  N4 rc=$R want=2(helper modified in the worktree)"
# fixture mode is labelled and isolated
FIXCMD=$TMPD/fixture.cmdspec
{ echo "NAME=fixture-arbitrary"; echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
  echo "ENV_SET=PATH=/usr/bin:/bin"; echo "TOOLCHAIN_ROOT=/usr"
  echo "ARGV=PYTHON_INCLUDES=-I/usr/include"; echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"
  echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
  echo "PROBE=nvcc|/bin/echo|nvcc release 13.0"; echo "PROBE=hostcc|/bin/echo|gcc 12.3.0"
  echo "PROBE=python|printf|Python 3.12.3"; echo "PROBE=torch|printf|2.11.0+cu130\\n13.0\\n"
  echo "PROBE=ext_suffix|printf|fixture.so"
  echo "PROBE=py_include|printf|-I/usr/include"; echo "PROBE=torch_include|printf|-I/usr/include"
  echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$FIXCMD"
set +e
ART=$(newart n5); FIXREC=$ART/build_record.v4
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$FIXCMD" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
     bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 0 ] && grep -q '^RECORD_MODE=fixture$' "$FIXREC"; report N5_fixture_mode_labelled $?
echo "  N5 rc=$R want=0(fixture record carries RECORD_MODE=fixture)"
# output symlink
SYMCMD=$TMPD/sym.cmdspec
{ echo "NAME=fixture-symlink"; echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
  echo "ENV_SET=PATH=/usr/bin:/bin"; echo "TOOLCHAIN_ROOT=/usr"
  echo "ARGV=PYTHON_INCLUDES=-I/usr/include"; echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"
  echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"; echo "ENV_PASS=PATH"
  echo "ARGV=bash"; echo "ARGV=-c"; echo "ARGV=ln -s $TMPD/old_prebuilt.so mok/_Cfixture.so"
  echo "ARGV=ARCH=SM90"; echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
  echo "PROBE=nvcc|/bin/echo|nvcc release 13.0"; echo "PROBE=hostcc|/bin/echo|gcc"; echo "PROBE=python|printf|Python 3"
  echo "PROBE=torch|printf|2.11.0\\n13.0\\n"; echo "PROBE=ext_suffix|printf|fixture.so"
  echo "PROBE=py_include|printf|-I/usr/include"; echo "PROBE=torch_include|printf|-I/usr/include"
  echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$SYMCMD"
ART6=$(newart n6); SYMREC=$ART6/build_record.v4
set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$SYMCMD" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$ART6" 2>&1 ); R=$?
set -u
rm -f "$FIX/mok/_Cfixture.so"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build created mok/_Cfixture\.so as a symlink; refusing to hash the target$' \
  && [ ! -f "$SYMREC" ]; report N6_output_symlink_refused $?
echo "  N6 rc=$R want=2(output created as a symlink to an old binary)"
# probe failure
BADPROBE=$TMPD/badprobe.cmdspec
{ echo "NAME=fixture-badprobe"; echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
  echo "ENV_SET=PATH=/usr/bin:/bin"; echo "TOOLCHAIN_ROOT=/usr"
  echo "ARGV=PYTHON_INCLUDES=-I/usr/include"; echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"
  echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"; echo "ENV_PASS=PATH"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "ARGV=NVCC=/bin/false -ccbin /bin/echo"
  echo "PROBE=nvcc|/bin/false"; echo "PROBE=hostcc|/bin/echo|gcc"; echo "PROBE=python|printf|Python 3"
  echo "PROBE=torch|printf|2.11.0\\n13.0\\n"; echo "PROBE=ext_suffix|printf|fixture.so"
  echo "PROBE=py_include|printf|-I/usr/include"; echo "PROBE=torch_include|printf|-I/usr/include"
  echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$BADPROBE"
ART7=$(newart n7); PROBEREC=$ART7/build_record.v4
set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$BADPROBE" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$ART7" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:probe nvcc exited 1; the toolchain cannot be described and no record is published$' \
  && [ ! -f "$PROBEREC" ]; report N7_probe_failure_refused $?
echo "  N7 rc=$R want=2(a probe failed; no record)"
# artifact dir problems
AD=$TMPD/art.reuse; mkdir -p "$AD"; : > "$AD/build.log"
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE bash "$FIXWRAP" "$FIX" "$AD" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:build log $AD/build\.log already exists or is not creatable\$"; report N8_log_must_be_exclusive $?
echo "  N8 rc=$R want=2(build.log already present in the artifact dir)"
ln -s "$TMPD" "$TMPD/art.symlink"
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE bash "$FIXWRAP" "$FIX" "$TMPD/art.symlink" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:artifact dir $TMPD/art\.symlink is a symlink\$" \
  && [ ! -e "$TMPD/build_home" ] && [ ! -e "$TMPD/build_tmp" ]; report N9_artifact_dir_symlink_refused $?
echo "  N9 rc=$R want=2(symlink rejected before any write through it)"
# A symlink at a run-owned child path must be rejected before either sibling is
# created. `mkdir -p build_tmp` would otherwise follow it.
ART=$(newart n21); LINK_TARGET=$TMPD/n21-link-target; mkdir -p "$LINK_TARGET"
ln -s "$LINK_TARGET" "$ART/build_tmp"
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:build TMPDIR path $ART/build_tmp already exists or is a symlink\$" \
  && [ ! -e "$ART/build_home" ] && [ -z "$(ls -A "$LINK_TARGET")" ]; report N21_internal_symlink_not_followed $?
echo "  N21 rc=$R want=2(internal TMPDIR symlink rejected before any sibling write)"
# A missing artifact directory is input error, not permission to create a new
# evidence location implicitly.
MISSING_ART=$TMPD/art.missing
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE bash "$FIXWRAP" "$FIX" "$MISSING_ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:artifact dir $MISSING_ART is not a directory\$" \
  && [ ! -e "$MISSING_ART" ]; report N19_missing_artifact_dir_not_created $?
echo "  N19 rc=$R want=2(missing artifact dir remains absent)"
# The public contract has exactly two arguments. Extra legacy output paths must
# not be silently ignored, or callers can believe evidence was published where
# it was not.
ART=$(newart n20)
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE bash "$FIXWRAP" "$FIX" "$ART" "$TMPD/legacy.record" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:expected exactly 2 arguments: <repo_dir> <artifact_dir>$' \
  && [ -z "$(ls -A "$ART")" ]; report N20_extra_argument_refused $?
echo "  N20 rc=$R want=2(extra legacy record path is not ignored)"
# dirty closure and submodule drift still refused
touch "$FIX/csrc/untracked.cu"
set +e
ART=$(newart n10)
O=$(runwrap_at "$ART"); R=$?
REC=$ART/build_record.v4
set -u
rm -f "$FIX/csrc/untracked.cu"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build inputs not clean vs HEAD \(incl\. untracked\):$'; report N10_dirty_inputs $?
echo "  N10 rc=$R want=2(untracked file in the closure)"
printf '// tk v2\n' >> "$SUBSRC/tk.h"
( cd "$SUBSRC" && $GIT add -A && $GIT commit -qm tk2 ) >/dev/null 2>&1
( cd "$FIX/third_party/tk" && $GIT fetch -q origin && { $GIT checkout -q origin/HEAD || $GIT checkout -q origin/master || $GIT checkout -q origin/main; } ) >/dev/null 2>&1
set +e
ART=$(newart n11)
O=$(runwrap_at "$ART"); R=$?
REC=$ART/build_record.v4
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:submodule third_party/tk drifted or uninitialized:$'; report N11_submodule_drift $?
echo "  N11 rc=$R want=2(submodule off its recorded gitlink)"
( cd "$FIX" && $GIT submodule update -q --checkout --force third_party/tk ) >/dev/null 2>&1

# ---------- P: PATH hijack, the bypass ENV_PASS left open ----------
# A fake `make` earlier in PATH is enough: the old design forwarded the
# CALLER's PATH into the build, so the command resolved to the attacker's
# binary while the record still said production and stayed internally green.
OLD2=$TMPD/old2
mkdir -p "$OLD2"
git -C "$REPO" archive 013d927 benchmarks 2>/dev/null | tar x -C "$OLD2" || true
OLD2WRAP=$OLD2/benchmarks/build_and_record_sm90.sh
FAKEBIN=$TMPD/fakebin
mkdir -p "$FAKEBIN"
{ echo '#!/bin/bash'; echo 'echo "fake make running"'
  echo "cp $TMPD/old_prebuilt.so mok/_Cfixture.so"; } > "$FAKEBIN/make"
chmod +x "$FAKEBIN/make"
if [ -f "$OLD2WRAP" ]; then
  FIXOLD=$TMPD/buildhost_old
  cp -r "$FIX" "$FIXOLD" 2>/dev/null
  rm -rf "$FIXOLD/.git"
  cp "$OLD2/benchmarks/build_and_record_sm90.sh" "$OLD2/benchmarks/compute_build_inputs_sm90.sh" \
     "$OLD2/benchmarks/validate_build_record_sm90.sh" "$FIXOLD/benchmarks/" 2>/dev/null
  { echo "BUILD_COMMAND_SPEC=1"; echo "NAME=old-style"; echo "OUTPUT=mok/_Cfixture.so"
    echo "HOST_COMPILER=/bin/echo"; echo "ENV_PASS=PATH"; echo "ENV_PASS=HOME"
    echo "ARGV=make"; echo "ARGV=ARCH=SM90"; echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
    echo "PROBE=nvcc|/bin/echo|nvcc release 13.0"; echo "PROBE=hostcc|/bin/echo|gcc"
    echo "PROBE=python|printf|Python 3.12.3"; echo "PROBE=torch|printf|2.11.0\\n13.0\\n"
    echo "PROBE=ext_suffix|printf|fixture.so"; } > "$FIXOLD/benchmarks/build_command_spec.v1"
  rm -rf "$FIXOLD/third_party"
  { echo "BUILD_INPUT_SPEC=1"; echo "NAME=old-inputs"; echo "PATH=Makefile"; echo "PATH=csrc"; } \
    > "$FIXOLD/benchmarks/build_input_spec.v1"
  ( cd "$FIXOLD" && $GIT init -q && $GIT add -A && $GIT commit -qm old ) >/dev/null 2>&1
  OLDART2=$TMPD/art.oldpath; mkdir -p "$OLDART2"
  set +e
  OLDOUT=$( cd "$FIXOLD" && PATH="$FAKEBIN:$PATH" TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" \
      TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
      bash "$FIXOLD/benchmarks/build_and_record_sm90.sh" "$FIXOLD" "$OLDART2" 2>&1 ); OLDRC=$?
  set -u
  STALESHA=$(sha256sum "$TMPD/old_prebuilt.so" | cut -d' ' -f1)
  # the old tooling published build_record.v3; this case replays it verbatim
  [ "$OLDRC" -eq 0 ] && grep -q "^SO_SHA256=$STALESHA\$" "$OLDART2/build_record.v3" 2>/dev/null \
    && grep -q '^RECORD_MODE=production$' "$OLDART2/build_record.v3" 2>/dev/null; report P1a_old_path_hijack_succeeded $?
  echo "  P1a old wrapper rc=$OLDRC produced a PRODUCTION record whose SO is the attacker's file"
else
  report P1a_old_path_hijack_succeeded 1
  echo "  P1a could not extract 013d927 for the comparison"
fi
ART=$(newart p1new)
set +e
O=$( cd "$FIX" && PATH="$FAKEBIN:$PATH" TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" \
    TOOLCHAIN_IMAGE_REF="build:cu130" TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
    bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
REC=$ART/build_record.v4
NEWSO=$(grep '^SO_SHA256=' "$REC" 2>/dev/null | cut -d= -f2)
[ "$R" -eq 0 ] && [ "$NEWSO" != "$(sha256sum "$TMPD/old_prebuilt.so" | cut -d' ' -f1)" ] \
  && grep -q '^SAW PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin$' "$ART/build.log"; report P1b_new_path_is_fixed $?
echo "  P1b new wrapper rc=$R built under the spec's fixed PATH, not the caller's"
[ -d "$ART/build_home" ] && [ -z "$(ls -A "$ART/build_home")" ] \
  && grep -q "^SAW HOME=$ART/build_home\$" "$ART/build.log"; report P1c_home_is_empty_and_owned $?
echo "  P1c HOME pointed at an empty directory this run created"
[ -d "$ART/build_tmp" ] && [ -z "$(ls -A "$ART/build_tmp")" ] \
  && grep -q "^SAW TMPDIR=$ART/build_tmp\$" "$ART/build.log" \
  && printf '%s' "$(grep '^ENV_MANIFEST_B64=' "$REC" | cut -d= -f2-)" | base64 -d \
     | grep -qx "TMPDIR=$ART/build_tmp"; report P1d_tmpdir_is_empty_recorded_and_owned $?
echo "  P1d TMPDIR is an empty recorded directory on the artifact filesystem"
# a pinned include path the toolchain does not report is refused
BADPATH=$TMPD/badpath.cmdspec
{ echo "NAME=fixture-badpath"; echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
  echo "ENV_SET=PATH=/usr/bin:/bin"; echo "TOOLCHAIN_ROOT=/usr"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
  echo "ARGV=PYTHON_INCLUDES=-I/usr/include/python-does-not-exist"
  echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"; echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"
  echo "PROBE=nvcc|/bin/echo|nvcc release 13.0"; echo "PROBE=hostcc|/bin/echo|gcc"
  echo "PROBE=python|printf|Python 3"; echo "PROBE=torch|printf|2.11.0\\n13.0\\n"
  echo "PROBE=ext_suffix|printf|fixture.so"; echo "PROBE=py_include|printf|-I/usr/include"
  echo "PROBE=torch_include|printf|-I/usr/include"; echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$BADPATH"
ART=$(newart p2); set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$BADPATH" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:PYTHON_INCLUDES pinned as \[-I/usr/include/python-does-not-exist\] but the toolchain reports \[-I/usr/include\]$' \
  && [ ! -f "$ART/build_record.v4" ]; report P2_pinned_path_must_match_toolchain $?
echo "  P2 rc=$R want=2(pinned include path is not what the toolchain reports)"

# ---------- E: entrypoint boundary (known limit, not a closed gate) --------
# A non-interactive bash reads BASH_ENV BEFORE the script's first line, and a
# function defined there shadows PATH lookups. This is demonstrated against
# b3d8162 - which claimed the control plane was normalised "before anything
# else runs" - and then shown to be REFUSED here. Refusing an observable
# BASH_ENV is misuse protection; it is NOT proof against a malicious
# pre-exec environment, which only a trusted orchestrator can rule out.
OLD3=$TMPD/old3
mkdir -p "$OLD3"
git -C "$REPO" archive b3d8162 benchmarks 2>/dev/null | tar x -C "$OLD3" || true
HIJACK=$TMPD/hijack.bashenv
{ echo 'git() { command git "$@"; }'; echo 'export -f git'
  echo "echo 'BASH_ENV WAS SOURCED' >> $TMPD/bashenv.witness"; } > "$HIJACK"
: > "$TMPD/bashenv.witness"
if [ -f "$OLD3/benchmarks/build_and_record_sm90.sh" ]; then
  set +e
  ( cd "$FIX" && BASH_ENV="$HIJACK" TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" \
      TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
      bash "$OLD3/benchmarks/build_and_record_sm90.sh" "$FIX" "$(newart e1old)" >/dev/null 2>&1 )
  set -u
  [ -s "$TMPD/bashenv.witness" ]; report E1a_old_bash_env_ran_before_preamble $?
  echo "  E1a BASH_ENV was sourced before b3d8162's hardening preamble could run"
else
  report E1a_old_bash_env_ran_before_preamble 1
  echo "  E1a could not extract b3d8162 for the comparison"
fi
ART=$(newart e1new)
set +e
O=$( cd "$FIX" && BASH_ENV="$HIJACK" TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" \
     TOOLCHAIN_IMAGE_REF="build:cu130" TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
     bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:BASH_ENV is set; this wrapper must be started from a clean entrypoint \(env -i \.\.\. bash --noprofile --norc\)$' \
  && [ ! -f "$ART/build_record.v4" ]; report E1b_new_refuses_bash_env $?
echo "  E1b rc=$R want=2(BASH_ENV observable and refused - misuse protection, not proof)"
ART=$(newart e2)
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF="build:cu130" \
     TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
     bash -c 'sha256sum() { echo "deadbeef  fake"; }; export -f sha256sum; exec bash "$0" "$1" "$2"' \
       "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:exported shell function sha256sum is present; a function shadows PATH lookups and the entrypoint is not clean$'; report E2_exported_function_refused $?
echo "  E2 rc=$R want=2(an exported shell function shadows PATH lookups)"
# the record must say the entrypoint is externally unverified
grep -q '^PROVENANCE_CLASS=.*unverified_external:clean_entrypoint' "$GOODREC"; report E3_record_declares_entrypoint_unverified $?
# E7: a52f336's own guards ran `printenv` and `declare -Fx | awk`, both of which
# an exported function shadows. The demonstration is against a52f336: with
# printenv and awk stubbed out, BASH_ENV is sourced and NO guard fires.
OLD4=$TMPD/old4
mkdir -p "$OLD4"
git -C "$REPO" archive a52f336 benchmarks/build_and_record_sm90.sh 2>/dev/null | tar x -C "$OLD4" || true
SHADOW='printenv() { :; }; awk() { :; }; declare() { :; }; export -f printenv awk declare; exec bash "$0" "$1" "$2"'
: > "$TMPD/e7.witness"
{ echo "echo hijacked >> $TMPD/e7.witness"; } > "$TMPD/e7.bashenv"
if [ -f "$OLD4/benchmarks/build_and_record_sm90.sh" ]; then
  set +e
  OLDOUT=$( cd "$FIX" && BASH_ENV="$TMPD/e7.bashenv" TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" \
      TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
      bash -c "$SHADOW" "$OLD4/benchmarks/build_and_record_sm90.sh" "$FIX" "$(newart e7old)" 2>&1 )
  set -u
  [ -s "$TMPD/e7.witness" ] && ! printf '%s\n' "$OLDOUT" | grep -q 'clean entrypoint'
  report E7a_old_guard_shadowed_by_functions $?
  echo "  E7a a52f336: BASH_ENV sourced and neither entrypoint guard fired"
else
  report E7a_old_guard_shadowed_by_functions 1
  echo "  E7a could not extract a52f336 for the comparison"
fi
ART=$(newart e7new)
set +e
O=$( cd "$FIX" && BASH_ENV="$TMPD/e7.bashenv" TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" \
     TOOLCHAIN_IMAGE_REF="build:cu130" TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
     bash -c "$SHADOW" "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
# either finding is a correct refusal; env output order is not specified
[ "$R" -eq 2 ] \
  && has1 "$O" '^BUILD_RECORD_FAIL:(BASH_ENV is set; this wrapper must be started from a clean entrypoint \(env -i \.\.\. bash --noprofile --norc\)|exported shell function (printenv|awk|declare) is present; a function shadows PATH lookups and the entrypoint is not clean)$' \
  && [ ! -f "$ART/build_record.v4" ]; report E7b_new_guard_survives_shadowing $?
echo "  E7b rc=$R want=2(absolute-path detection cannot be shadowed by a function)"
ART=$(newart e7src)
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF="build:cu130" \
     TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
     bash -c 'source "$1" "$2" "$3"' bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -ne 0 ] && has1 "$O" "^BUILD_RECORD_FAIL:this wrapper must be executed, not sourced; a sourcing shell's own functions stay live$" \
  && [ ! -f "$ART/build_record.v4" ]; report E7c_sourcing_refused $?
echo "  E7c rc=$R want!=0(ordinary sourcing caught; a spoofed \$0 is not detectable)"
echo "  E3 the record declares unverified_external:clean_entrypoint"
# a schema-3 record must be refused outright
sed 's/^BUILD_RECORD_SCHEMA=4$/BUILD_RECORD_SCHEMA=3/' "$GOODREC" > "$TMPD/rec.schema3"
chmod 444 "$TMPD/rec.schema3"
set +e
O=$(bash "$VALB" "$TMPD/rec.schema3" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 17 ] && has1 "$O" '^BUILD_RECORD_FAIL:bad or missing schema version \(only 4 is valid; 1 never ran a build, and 2 and 3 were each published in two incompatible field sets\)$'; report E4_schema3_refused $?
echo "  E4 rc=$R want=17(schema 3 was published twice with different field sets)"
# nvcc: the probed binary must be the one the command runs
MISMATCH=$TMPD/nvccmismatch.cmdspec
{ echo "NAME=fixture-nvcc-mismatch"; echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
  echo "ENV_SET=PATH=/usr/bin:/bin"; echo "TOOLCHAIN_ROOT=/usr"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
  echo "ARGV=PYTHON_INCLUDES=-I/usr/include"; echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"
  echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"
  echo "PROBE=nvcc|/bin/true|--version"; echo "PROBE=hostcc|/bin/echo|gcc"
  echo "PROBE=python|printf|Python 3"; echo "PROBE=torch|printf|2.11.0\\n13.0\\n"
  echo "PROBE=ext_suffix|printf|fixture.so"; echo "PROBE=py_include|printf|-I/usr/include"
  echo "PROBE=torch_include|printf|-I/usr/include"; echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$MISMATCH"
ART=$(newart e5); set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$MISMATCH" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:the nvcc probe measures /bin/true but the command runs /bin/echo$'; report E5_nvcc_probe_must_match_command $?
echo "  E5 rc=$R want=2(probing a different nvcc than the one that builds)"
# packaging has no override left, in code as well as in behaviour
mapfile -t GFILES < <(cd "$DIR/.." && git ls-files 'benchmarks/*.sh' | grep -v '^benchmarks/test_')
GN=${#GFILES[@]}
( cd "$DIR/.." && printf '%s\n' "${GFILES[@]}" | grep -qx 'benchmarks/make_deploy_receipt.sh' ) && [ "$GN" -ge 8 ] \
  && ( cd "$DIR/.." && grep -lq 'RECEIPT_FAIL' "${GFILES[@]}" ) \
  && ! ( cd "$DIR/.." && grep -q 'ALLOW_UNVERIFIED_IMAGE_BINDING' "${GFILES[@]}" ); report E6_no_binding_override_in_source $?
echo "  E6 scanned $GN tracked production scripts (positive control hit; override absent)"

# ---------- S: spec strictness (compute helper, from git objects) ----------
scase() { # name spec-body want-reason-ERE
  local NAME=$1 BODY=$2 REASON=$3 O R
  printf '%s\n' "$BODY" > "$FIX/benchmarks/build_input_spec.v1"
  ( cd "$FIX" && $GIT add -A && $GIT commit -qm "$NAME" ) >/dev/null 2>&1
  local C; C=$($GIT -C "$FIX" rev-parse HEAD)
  set +e
  O=$(bash "$CBI" "$FIX" "$C" 2>&1); R=$?
  set -u
  [ "$R" -eq 19 ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=19($REASON)"
}
GOODSPEC=$(cat "$FIX/benchmarks/build_input_spec.v1")
scase S1_unknown_key "$GOODSPEC
ROGUE=1" '^BUILD_INPUTS_FAIL:build input spec: unknown key ROGUE$'
scase S2_duplicate_name "$GOODSPEC
NAME=second" '^BUILD_INPUTS_FAIL:build input spec: key NAME appears 2 times \(need exactly 1\)$'
scase S3_empty_value "BUILD_INPUT_SPEC=1
NAME=x
PATH=" '^BUILD_INPUTS_FAIL:build input spec: key PATH has an empty value$'
scase S4_absolute_path "BUILD_INPUT_SPEC=1
NAME=x
PATH=/etc" '^BUILD_INPUTS_FAIL:build input spec: absolute path not allowed: /etc$'
scase S5_dotdot_path "BUILD_INPUT_SPEC=1
NAME=x
PATH=../outside" \
  '^BUILD_INPUTS_FAIL:build input spec: path may not contain .\.\..: \.\./outside$'
scase S6_no_paths "BUILD_INPUT_SPEC=1
NAME=x" '^BUILD_INPUTS_FAIL:build input spec lists no PATH entries$'
scase S7_duplicate_path "BUILD_INPUT_SPEC=1
NAME=x
PATH=csrc
PATH=csrc" '^BUILD_INPUTS_FAIL:build input spec: duplicate PATH csrc$'
printf '%s\n' "$GOODSPEC" > "$FIX/benchmarks/build_input_spec.v1"
( cd "$FIX" && $GIT add -A && $GIT commit -qm restore ) >/dev/null 2>&1
# the spec hash is the RAW BLOB, so trailing whitespace is not normalised away
RESTORED=$($GIT -C "$FIX" rev-parse HEAD)
BLOBSHA=$($GIT -C "$FIX" cat-file blob "$RESTORED:benchmarks/build_input_spec.v1" | sha256sum | cut -d' ' -f1)
CALCSHA=$(bash "$CBI" "$FIX" "$RESTORED" | grep '^BUILD_INPUT_SPEC_SHA256=' | cut -d= -f2)
[ "$BLOBSHA" = "$CALCSHA" ]; report S8_spec_hash_is_raw_blob $?
echo "  S8 spec hash equals the raw git blob sha256"

# S9/S10: both "unknown key" checks used the key as a GREP PATTERN, so a key
# with a metacharacter matched an allowed name and passed as known. The key is
# content - in a record it is attacker-influenced - so it must be compared as a
# string. Demonstrated against 07170f0 first.
OLD5=$TMPD/old5
mkdir -p "$OLD5"
git -C "$REPO" archive 07170f0 benchmarks/compute_build_inputs_sm90.sh benchmarks/validate_build_record_sm90.sh 2>/dev/null | tar x -C "$OLD5" || true
scase S9b_metachar_key_refused "$GOODSPEC
PAT.=csrc" '^BUILD_INPUTS_FAIL:build input spec: unknown key PAT\.$'
C9=$($GIT -C "$FIX" rev-parse HEAD)
if [ -f "$OLD5/benchmarks/compute_build_inputs_sm90.sh" ]; then
  set +e
  O=$(bash "$OLD5/benchmarks/compute_build_inputs_sm90.sh" "$FIX" "$C9" 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ]; report S9a_old_accepted_metachar_key $?
  echo "  S9a old helper rc=$R (PAT. matched PATH in the allow-list)"
else
  report S9a_old_accepted_metachar_key 1; echo "  S9a could not extract 07170f0"
fi
printf '%s\n' "$GOODSPEC" > "$FIX/benchmarks/build_input_spec.v1"
( cd "$FIX" && $GIT add -A && $GIT commit -qm restore-spec ) >/dev/null 2>&1
cp "$GOODREC" "$TMPD/rec.smuggled"; chmod 644 "$TMPD/rec.smuggled"
echo 'SO_SHA25.=smuggled' >> "$TMPD/rec.smuggled"
chmod 444 "$TMPD/rec.smuggled"   # the validator refuses a writable record before it parses keys
if [ -f "$OLD5/benchmarks/validate_build_record_sm90.sh" ]; then
  set +e
  O=$(bash "$OLD5/benchmarks/validate_build_record_sm90.sh" "$TMPD/rec.smuggled" --check-mode 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ]; report S10a_old_accepted_smuggled_record_key $?
  echo "  S10a old validator rc=$R (SO_SHA25. matched SO_SHA256)"
else
  report S10a_old_accepted_smuggled_record_key 1; echo "  S10a could not extract 07170f0"
fi
set +e
O=$(bash "$VALB" "$TMPD/rec.smuggled" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 17 ] && has1 "$O" '^BUILD_RECORD_FAIL:unknown key SO_SHA25\.$'; report S10b_smuggled_record_key_refused $?
echo "  S10b rc=$R want=17(a record key is content, not a regex)"

# ---------- N12-N15 / S11-S13: measured version and environment names ------
# Every "old accepts" case below runs 96fc397's own file, swapped into the
# fixture repo and committed, because that wrapper self-binds and refuses to
# run from anywhere else.
OLD6=$TMPD/old6
mkdir -p "$OLD6"
git -C "$REPO" archive 96fc397 benchmarks/build_and_record_sm90.sh \
  benchmarks/compute_build_inputs_sm90.sh benchmarks/validate_build_record_sm90.sh 2>/dev/null | tar x -C "$OLD6" || true
# the wrapper self-validates before publishing, so the demonstration has to run
# the OLD STACK - wrapper and validator together - or the new validator blocks
# the publish and hides how far the old wrapper actually got
# a swallowed commit failure would silently leave the wrapper unbound and show
# up as an unexplained case failure, so this one is loud (a no-op commit is
# tolerated only when the tree really does match HEAD)
fixcommit() { ( cd "$FIX" && $GIT add -A && { $GIT commit -qm "$1" || $GIT diff --quiet HEAD; } ) >/dev/null 2>&1 \
  || { echo "TEST_HARNESS_FAIL:fixture commit '$1' failed"; exit 1; }; }
use_stack() { cp "$1/build_and_record_sm90.sh" "$FIX/benchmarks/build_and_record_sm90.sh"
  cp "$1/validate_build_record_sm90.sh" "$FIX/benchmarks/validate_build_record_sm90.sh"
  fixcommit stack-swap; }
NOREL=$TMPD/norelease.cmdspec
{ echo "NAME=fixture-no-release"; echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
  echo "ENV_SET=PATH=/usr/bin:/bin"; echo "TOOLCHAIN_ROOT=/usr"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
  echo "ARGV=PYTHON_INCLUDES=-I/usr/include"; echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"
  echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"
  echo "PROBE=nvcc|/bin/echo|GNU coreutils 9.4"; echo "PROBE=hostcc|/bin/echo|gcc 12.3.0"
  echo "PROBE=python|printf|Python 3.12.3"; echo "PROBE=torch|printf|2.11.0+cu130\n13.0\n"
  echo "PROBE=ext_suffix|printf|fixture.so"; echo "PROBE=py_include|printf|-I/usr/include"
  echo "PROBE=torch_include|printf|-I/usr/include"; echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$NOREL"
# command and probe agree on /bin/echo; it just is not a CUDA compiler
if [ -f "$OLD6/benchmarks/build_and_record_sm90.sh" ]; then
  use_stack "$OLD6/benchmarks"
  ART=$(newart n12old); set +e
  O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$NOREL" \
       TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
       bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
  set -u
  [ "$R" -eq 0 ] && grep -q '^MEASURED_NVCC_VERSION=GNU coreutils 9.4$' "$ART/build_record.v4" 2>/dev/null
  RN12=$?; [ "$RN12" -eq 0 ] || { echo "--- N12a old stack output (rc=$R) ---"; printf '%s\n' "$O" | tail -4; }
  report N12a_old_recorded_any_first_line "$RN12"
  echo "  N12a old wrapper rc=$R recorded MEASURED_NVCC_VERSION=GNU coreutils 9.4"
  use_stack "$DIR"
else
  report N12a_old_recorded_any_first_line 1; echo "  N12a could not extract 96fc397"
fi
ART=$(newart n12new); set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$NOREL" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:the nvcc probe output has no CUDA 'release <major>\.<minor>' field; /bin/echo did not identify itself as a CUDA compiler and no record is published$" \
  && [ ! -f "$ART/build_record.v4" ]; report N12b_no_release_field_refused $?
echo "  N12b rc=$R want=2(an exit-0 tool that is not nvcc cannot supply a version)"
# and independently on the record, which can arrive from elsewhere
cp "$GOODREC" "$TMPD/rec.norelease"; chmod 644 "$TMPD/rec.norelease"
sed -i 's|^MEASURED_NVCC_VERSION=.*|MEASURED_NVCC_VERSION=GNU coreutils 9.4|' "$TMPD/rec.norelease"
chmod 444 "$TMPD/rec.norelease"
if [ -f "$OLD6/benchmarks/validate_build_record_sm90.sh" ]; then
  set +e
  O=$(bash "$OLD6/benchmarks/validate_build_record_sm90.sh" "$TMPD/rec.norelease" --check-mode 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ]; report N13a_old_validated_coreutils_version $?
  echo "  N13a old validator rc=$R (only unavailable/unknown were rejected)"
else
  report N13a_old_validated_coreutils_version 1; echo "  N13a could not extract 96fc397"
fi
set +e
O=$(bash "$VALB" "$TMPD/rec.norelease" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 17 ] && has1 "$O" "^BUILD_RECORD_FAIL:MEASURED_NVCC_VERSION has no CUDA 'release <major>\.<minor>' field: GNU coreutils 9\.4$"
report N13b_record_version_must_be_a_release $?
echo "  N13b rc=$R want=17(the record is checked independently of the wrapper)"
# S11: an illegal environment NAME in the tracked command spec
cp "$FIX/benchmarks/build_command_spec.v2" "$TMPD/cmdspec.good"
{ cat "$TMPD/cmdspec.good"; echo "ENV_SET=A-B=x"; } > "$FIX/benchmarks/build_command_spec.v2"
( cd "$FIX" && $GIT add -A && $GIT commit -qm badenvname ) >/dev/null 2>&1
CBAD=$($GIT -C "$FIX" rev-parse HEAD)
if [ -f "$OLD6/benchmarks/compute_build_inputs_sm90.sh" ]; then
  set +e
  O=$(bash "$OLD6/benchmarks/compute_build_inputs_sm90.sh" "$FIX" "$CBAD" 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ]; report S11a_old_accepted_illegal_env_name $?
  echo "  S11a old helper rc=$R (case [A-Z_]*=* only pinned the first character)"
else
  report S11a_old_accepted_illegal_env_name 1; echo "  S11a could not extract 96fc397"
fi
set +e
O=$(bash "$CBI" "$FIX" "$CBAD" 2>&1); R=$?
set -u
[ "$R" -eq 19 ] && has1 "$O" '^BUILD_INPUTS_FAIL:build command spec: ENV_SET entry has an illegal variable name \[A-B\]; names must match \^\[A-Z_\]\[A-Z0-9_\]\*\$$'
report S11b_illegal_env_name_refused $?
echo "  S11b rc=$R want=19(A-B is not a variable name any shell can reference)"
# S13: a relative toolchain root, same spec stage
{ cat "$TMPD/cmdspec.good"; echo "TOOLCHAIN_ROOT=usr"; } > "$FIX/benchmarks/build_command_spec.v2"
( cd "$FIX" && $GIT add -A && $GIT commit -qm relroot ) >/dev/null 2>&1
CREL=$($GIT -C "$FIX" rev-parse HEAD)
if [ -f "$OLD6/benchmarks/compute_build_inputs_sm90.sh" ]; then
  set +e
  O=$(bash "$OLD6/benchmarks/compute_build_inputs_sm90.sh" "$FIX" "$CREL" 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ]; report S13a_old_accepted_relative_root $?
  echo "  S13a old helper rc=$R (roots were never required to be absolute)"
else
  report S13a_old_accepted_relative_root 1; echo "  S13a could not extract 96fc397"
fi
set +e
O=$(bash "$CBI" "$FIX" "$CREL" 2>&1); R=$?
set -u
[ "$R" -eq 19 ] && has1 "$O" '^BUILD_INPUTS_FAIL:build command spec: TOOLCHAIN_ROOT must be an absolute path: usr$'
report S13_relative_toolchain_root_refused $?
echo "  S13 rc=$R want=19(readlink -f would resolve it against the repository)"
cp "$TMPD/cmdspec.good" "$FIX/benchmarks/build_command_spec.v2"
( cd "$FIX" && $GIT add -A && $GIT commit -qm restore-cmdspec ) >/dev/null 2>&1
# N15: the wrapper refuses it too, not only the helper
RELROOT=$TMPD/relroot.cmdspec
sed 's|^TOOLCHAIN_ROOT=/usr$|TOOLCHAIN_ROOT=usr|' "$NOREL" > "$RELROOT"
sed -i 's|^PROBE=nvcc.*|PROBE=nvcc\|/bin/echo\|nvcc release 13.0|' "$RELROOT"
ART=$(newart n15); set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$RELROOT" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$ART" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:TOOLCHAIN_ROOT must be an absolute path: usr$' \
  && [ ! -f "$ART/build_record.v4" ]; report N15_wrapper_refuses_relative_root $?
echo "  N15 rc=$R want=2(defence in depth: the wrapper checks the roots as well)"
# S12: an illegal environment NAME inside a record, with the hash recomputed -
# this is what proves the manifest is parsed and not merely hashed
cp "$GOODREC" "$TMPD/rec.badenv"; chmod 644 "$TMPD/rec.badenv"
ENVPLAIN=$(grep '^ENV_MANIFEST_B64=' "$GOODREC" | cut -d= -f2- | base64 -d)
ENVPLAIN=$(printf '%s\nA-B=x' "$ENVPLAIN")
sed -i -e "s|^ENV_MANIFEST_B64=.*|ENV_MANIFEST_B64=$(printf '%s' "$ENVPLAIN" | base64 -w0)|" \
       -e "s|^ENV_APPLIED_SHA256=.*|ENV_APPLIED_SHA256=$(printf '%s' "$ENVPLAIN" | sha256sum | cut -d' ' -f1)|" \
       "$TMPD/rec.badenv"
chmod 444 "$TMPD/rec.badenv"
if [ -f "$OLD6/benchmarks/validate_build_record_sm90.sh" ]; then
  set +e
  O=$(bash "$OLD6/benchmarks/validate_build_record_sm90.sh" "$TMPD/rec.badenv" --check-mode 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ]; report S12a_old_validated_illegal_env_name $?
  echo "  S12a old validator rc=$R (hash agreed, so the malformed name passed)"
else
  report S12a_old_validated_illegal_env_name 1; echo "  S12a could not extract 96fc397"
fi
set +e
O=$(bash "$VALB" "$TMPD/rec.badenv" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 17 ] && has1 "$O" '^BUILD_RECORD_FAIL:env manifest has an illegal variable name \[A-B\]; names must match \^\[A-Z_\]\[A-Z0-9_\]\*\$$'
report S12b_record_illegal_env_name_refused $?
echo "  S12b rc=$R want=17(a matching hash is not a well-formed environment)"
# N16/N17: the wrapper's own environment checks, driven directly. The helper
# refuses a malformed tracked spec, but fixture mode hands the wrapper a spec
# the helper never saw, which is exactly the defence-in-depth path.
mkbadenv() { # outfile env-lines...
  local OUT=$1; shift
  { echo "NAME=fixture-badenv"; echo "OUTPUT=mok/_Cfixture.so"; echo "HOST_COMPILER=/bin/echo"
    echo "TOOLCHAIN_ROOT=/usr"
    local L; for L in "$@"; do echo "ENV_SET=$L"; done
    echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
    echo "ARGV=NVCC=/bin/echo -ccbin /bin/echo"
    echo "ARGV=PYTHON_INCLUDES=-I/usr/include"; echo "ARGV=PYTORCH_INCLUDES=-I/usr/include"
    echo "ARGV=PYTORCH_LIBDIR=-L/usr/lib"
    echo "PROBE=nvcc|/bin/echo|nvcc release 13.0"; echo "PROBE=hostcc|/bin/echo|gcc 12.3.0"
    echo "PROBE=python|printf|Python 3.12.3"; echo "PROBE=torch|printf|2.11.0+cu130\n13.0\n"
    echo "PROBE=ext_suffix|printf|fixture.so"; echo "PROBE=py_include|printf|-I/usr/include"
    echo "PROBE=torch_include|printf|-I/usr/include"; echo "PROBE=torch_libdir|printf|-L/usr/lib"; } > "$OUT"
}
runbadenv() { # name spec want-reason-ERE
  local NAME=$1 SPEC=$2 REASON=$3 A O R
  A=$(newart "$NAME"); set +e
  O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$SPEC" \
       TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
       bash "$FIXWRAP" "$FIX" "$A" 2>&1 ); R=$?
  set -u
  [ "$R" -eq 2 ] && has1 "$O" "$REASON" && [ ! -f "$A/build_record.v4" ]; report "$NAME" $?
  echo "  $NAME rc=$R want=2($REASON)"
}
mkbadenv "$TMPD/wrapenv.bad" "PATH=/usr/bin:/bin" "A-B=x"
runbadenv N16_wrapper_refuses_illegal_env_name "$TMPD/wrapenv.bad" \
  '^BUILD_RECORD_FAIL:ENV_SET entry has an illegal variable name \[A-B\]; names must match \^\[A-Z_\]\[A-Z0-9_\]\*\$$'
mkbadenv "$TMPD/wrapenv.dup" "PATH=/usr/bin:/bin" "PATH=/opt/evil/bin"
runbadenv N17_wrapper_refuses_duplicate_env_name "$TMPD/wrapenv.dup" \
  '^BUILD_RECORD_FAIL:ENV_SET defines PATH more than once$'
# S14: duplicate environment name in the TRACKED command spec. 96fc397 had no
# duplicate gate on the spec at all, so this one is a new refusal.
{ cat "$TMPD/cmdspec.good"; echo "ENV_SET=PATH=/opt/evil/bin"; } > "$FIX/benchmarks/build_command_spec.v2"
( cd "$FIX" && $GIT add -A && { $GIT commit -qm dupenv || $GIT diff --quiet HEAD; } ) >/dev/null 2>&1 \
  || { echo "TEST_HARNESS_FAIL:fixture commit 'dupenv' failed"; exit 1; }
CDUP=$($GIT -C "$FIX" rev-parse HEAD)
if [ -f "$OLD6/benchmarks/compute_build_inputs_sm90.sh" ]; then
  set +e
  O=$(bash "$OLD6/benchmarks/compute_build_inputs_sm90.sh" "$FIX" "$CDUP" 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ]; report S14a_old_accepted_duplicate_env_name $?
  echo "  S14a old helper rc=$R (the spec had no duplicate gate)"
else
  report S14a_old_accepted_duplicate_env_name 1; echo "  S14a could not extract 96fc397"
fi
set +e
O=$(bash "$CBI" "$FIX" "$CDUP" 2>&1); R=$?
set -u
[ "$R" -eq 19 ] && has1 "$O" '^BUILD_INPUTS_FAIL:build command spec: ENV_SET defines PATH more than once$'
report S14b_duplicate_env_name_refused $?
echo "  S14b rc=$R want=19(two ENV_SET=PATH= in the tracked spec)"
cp "$TMPD/cmdspec.good" "$FIX/benchmarks/build_command_spec.v2"
( cd "$FIX" && $GIT add -A && { $GIT commit -qm restore-cmdspec2 || $GIT diff --quiet HEAD; } ) >/dev/null 2>&1 \
  || { echo "TEST_HARNESS_FAIL:fixture commit 'restore-cmdspec2' failed"; exit 1; }
# S15: duplicate LEGAL name inside a record, hash recomputed. REGRESSION GUARD,
# not a new fix: 96fc397 already caught this shape, because grep -o '^[A-Z_]*='
# does see a well-formed name. What changed is that the check now counts parsed
# names, so a malformed one cannot dodge it (S12) - this case pins the old
# behaviour so the rewrite did not lose it.
cp "$GOODREC" "$TMPD/rec.dupenv"; chmod 644 "$TMPD/rec.dupenv"
ENVPLAIN=$(grep '^ENV_MANIFEST_B64=' "$GOODREC" | cut -d= -f2- | base64 -d)
ENVPLAIN=$(printf '%s\nPATH=/opt/evil/bin' "$ENVPLAIN")
sed -i -e "s|^ENV_MANIFEST_B64=.*|ENV_MANIFEST_B64=$(printf '%s' "$ENVPLAIN" | base64 -w0)|" \
       -e "s|^ENV_APPLIED_SHA256=.*|ENV_APPLIED_SHA256=$(printf '%s' "$ENVPLAIN" | sha256sum | cut -d' ' -f1)|" \
       "$TMPD/rec.dupenv"
chmod 444 "$TMPD/rec.dupenv"
set +e
ONEW=$(bash "$VALB" "$TMPD/rec.dupenv" --check-mode 2>&1); RNEW=$?
OOLD=$(bash "$OLD6/benchmarks/validate_build_record_sm90.sh" "$TMPD/rec.dupenv" --check-mode 2>&1); ROLD=$?
set -u
[ "$RNEW" -eq 17 ] && has1 "$ONEW" '^BUILD_RECORD_FAIL:env manifest defines PATH more than once$' && [ "$ROLD" -eq 17 ]
report S15_duplicate_legal_name_regression_guard $?
echo "  S15 new rc=$RNEW old rc=$ROLD - both refuse; this pins existing behaviour, it is not a new fix"
# N18: a legal value containing '=' is kept whole
printf '%s' "$(grep '^ENV_MANIFEST_B64=' "$GOODREC" | cut -d= -f2-)" | base64 -d | grep -qx 'TEST_EQ=a=b'
report N18_env_value_with_equals_kept $?
echo "  N18 TEST_EQ=a=b survives the production path with its value intact"
# N14: the legal empty value survives the whole production path
printf '%s' "$(grep '^ENV_MANIFEST_B64=' "$GOODREC" | cut -d= -f2-)" | base64 -d | grep -qx 'PYTHONPATH='
report N14_empty_env_value_accepted $?
echo "  N14 PYTHONPATH= (empty value) is recorded and validates"

# ---------- F: fixture records and NONE digests cannot reach formal --------
MANF=$TMPD/fix.manifest
SO_SHA=$(grep '^SO_SHA256=' "$GOODREC" | cut -d= -f2)
sed -e "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$SO_SHA|" "$DIR/manifests/tiny-h20-v1.manifest" \
  > "$MANF"
RCPT=$TMPD/f.receipt
{ echo "RECEIPT_SCHEMA=2"; echo "SOURCE_TREE_COMMIT=$(H40 0)"; echo "HARNESS_COMMIT=$(H40 0)"
  echo "BINARY_BUILD_COMMIT=$(grep '^SOURCE_COMMIT=' "$GOODREC" | cut -d= -f2)"
  echo "MANIFEST_FILE=fix.manifest"; echo "MANIFEST_SHA256=$(sha256sum "$MANF" | cut -d' ' -f1)"
  echo "MANIFEST_GIT_BLOB=$(H40 0)"
  echo "HARNESS_SHA256=$(grep '^EXPECTED_HARNESS_SHA256=' "$MANF" | cut -d= -f2)"
  echo "SO_SHA256=$SO_SHA"; echo "IMAGE_ID=sha256:$(H64 1)"; echo "IMAGE_REF=x:1"
  echo "IMAGE_REPO_DIGESTS=registry.local/mok@sha256:$(H64 9)"
  echo "BUILD_RECORD_SHA256=$(sha256sum "$FIXREC" | cut -d' ' -f1)"; } > "$RCPT"
chmod 444 "$RCPT"
SODIR=$TMPD/deployed; mkdir -p "$SODIR"; cp "$GOODSO" "$SODIR/_Cfixture.so"
set +e
O=$(EXPECTED_RECEIPT_SHA256=$(sha256sum "$RCPT" | cut -d' ' -f1) \
    bash "$BIND" "$RCPT" "$FIXREC" "$MANF" "$SODIR" 2>&1); R=$?
set -u
[ "$R" -eq 18 ] && has1 "$O" '^FORMAL_BINDING_FAIL:build record RECORD_MODE=fixture is not a production record$'; report F1_fixture_record_refused $?
echo "  F1 rc=$R want=18(fixture record cannot satisfy formal)"
NONEREC=$TMPD/rec.none
sed 's|^TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER=.*|TOOLCHAIN_IMAGE_REPO_DIGESTS_DECLARED_BY_CALLER=NONE|' \
  "$GOODREC" > "$NONEREC"; chmod 444 "$NONEREC"
RCPT2=$TMPD/f2.receipt
sed "s|^BUILD_RECORD_SHA256=.*|BUILD_RECORD_SHA256=$(sha256sum "$NONEREC" | cut -d' ' -f1)|" "$RCPT" > "$RCPT2"
chmod 444 "$RCPT2"
set +e
O=$(EXPECTED_RECEIPT_SHA256=$(sha256sum "$RCPT2" | cut -d' ' -f1) \
    bash "$BIND" "$RCPT2" "$NONEREC" "$MANF" "$SODIR" 2>&1); R=$?
set -u
[ "$R" -eq 18 ] && has1 "$O" '^FORMAL_BINDING_FAIL:toolchain image has no repo digest \(NONE\); formal requires registry provenance$'; report F2_none_digest_refused $?
echo "  F2 rc=$R want=18(NONE repo digest cannot satisfy formal)"
# THE dangerous branch: a production record, a non-NONE digest and a fully
# consistent chain. Everything internal agrees - and it must STILL be refused,
# because nothing verified the image identity. Without this case the suite
# would only ever have exercised fixture/NONE rejections.
PRODREC=$GOODREC
PSO=$(grep '^SO_SHA256=' "$PRODREC" | cut -d= -f2)
PMAN=$TMPD/prod.manifest
sed -e "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$PSO|" "$DIR/manifests/tiny-h20-v1.manifest" > "$PMAN"
PRCPT=$TMPD/prod.receipt
{ echo "RECEIPT_SCHEMA=2"; echo "SOURCE_TREE_COMMIT=$(H40 0)"; echo "HARNESS_COMMIT=$(H40 0)"
  echo "BINARY_BUILD_COMMIT=$(grep '^SOURCE_COMMIT=' "$PRODREC" | cut -d= -f2)"
  echo "MANIFEST_FILE=prod.manifest"; echo "MANIFEST_SHA256=$(sha256sum "$PMAN" | cut -d' ' -f1)"
  echo "MANIFEST_GIT_BLOB=$(H40 0)"
  echo "HARNESS_SHA256=$(grep '^EXPECTED_HARNESS_SHA256=' "$PMAN" | cut -d= -f2)"
  echo "SO_SHA256=$PSO"; echo "IMAGE_ID=sha256:$(H64 1)"; echo "IMAGE_REF=x:1"
  echo "IMAGE_REPO_DIGESTS=registry.local/mok@sha256:$(H64 9)"
  echo "BUILD_RECORD_SHA256=$(sha256sum "$PRODREC" | cut -d' ' -f1)"; } > "$PRCPT"
chmod 444 "$PRCPT"
PSODIR=$TMPD/prod_deployed; mkdir -p "$PSODIR"; cp "$GOODSO" "$PSODIR/_Cfixture.so"
set +e
O=$(EXPECTED_RECEIPT_SHA256=$(sha256sum "$PRCPT" | cut -d' ' -f1) \
    bash "$BIND" "$PRCPT" "$PRODREC" "$PMAN" "$PSODIR" 2>&1); R=$?
set -u
[ "$R" -eq 18 ] && has1 "$O" '^FORMAL_BINDING_FAIL:trusted toolchain image attestation not implemented' \
  && [ "$(printf '%s\n' "$O" | grep -c 'FORMAL_BINDING_PASS')" -eq 0 ]; report F0_full_positive_still_refused $?
echo "  F0 rc=$R want=18(everything agrees; the image identity is still unverified)"
# the consistency-only path is available but named for what it proves
set +e
O=$(LOCAL_BINDING_ONLY=1 EXPECTED_RECEIPT_SHA256=$(sha256sum "$PRCPT" | cut -d' ' -f1) \
    bash "$BIND" "$PRCPT" "$PRODREC" "$PMAN" "$PSODIR" 2>&1); R=$?
set -u
[ "$R" -eq 0 ] && has1 "$O" "^LOCAL_BINDING_PASS:$PSO\$" \
  && [ "$(printf '%s\n' "$O" | grep -c 'FORMAL_BINDING_PASS')" -eq 0 ]; report F0b_local_only_path_is_labelled $?
echo "  F0b rc=$R want=0(LOCAL_BINDING_PASS, never FORMAL_BINDING_PASS)"
# packaging refuses to bind a production record for the same reason
PKG=$TMPD/pkg
mkdir -p "$PKG/benchmarks/manifests"
cp "$DIR/validate_manifest_sm90.sh" "$DIR/validate_receipt_sm90.sh" \
   "$DIR/validate_build_record_sm90.sh" "$DIR/compute_build_inputs_sm90.sh" \
   "$DIR/make_deploy_receipt.sh" "$DIR/bench_sm90_fwd.py" "$PKG/benchmarks/"
PKGH=$(sha256sum "$PKG/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
sed -e "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$PSO|" \
    -e "s|^EXPECTED_HARNESS_SHA256=.*|EXPECTED_HARNESS_SHA256=$PKGH|" \
    "$DIR/manifests/tiny-h20-v1.manifest" > "$PKG/benchmarks/manifests/fix.manifest"
( cd "$PKG" && $GIT init -q && $GIT add -A && $GIT commit -qm pkg ) >/dev/null 2>&1
set +e
O=$( cd "$PKG" && IMAGE_ID="sha256:$(H64 1)" IMAGE_REF="x:1" \
     IMAGE_REPO_DIGESTS="registry.local/mok@sha256:$(H64 9)" BINARY_BUILD_COMMIT=UNKNOWN \
     BUILD_RECORD="$PRODREC" bash benchmarks/make_deploy_receipt.sh "$PKG" \
     benchmarks/manifests/fix.manifest "$TMPD/prod.pkg.receipt" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^RECEIPT_FAIL:publishing a receipt from a build record requires a trusted toolchain image attestation, which is not implemented$'; report F0c_packaging_refuses_production_binding $?
echo "  F0c rc=$R want=2(packaging will not launder a caller declaration)"
set +e
O=$(bash "$VALB" "$FIXREC" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 0 ] && has1 "$O" '^BUILD_RECORD_VALID:[0-9a-f]{64}$'; report F3_fixture_record_still_valid_schema $?
echo "  F3 rc=$R want=0(a fixture record is schema-valid but not formal-capable)"
# placeholder measurements are rejected
PH=$TMPD/rec.placeholder
sed 's|^MEASURED_NVCC_VERSION=.*|MEASURED_NVCC_VERSION=unavailable (see probe log)|' "$GOODREC" > "$PH"
chmod 444 "$PH"
set +e
O=$(bash "$VALB" "$PH" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 17 ] && has1 "$O" '^BUILD_RECORD_FAIL:MEASURED_NVCC_VERSION is a placeholder, not a measurement: unavailable \(see probe log\)$'; report F4_placeholder_measurement_refused $?
echo "  F4 rc=$R want=17(placeholder version string)"
# formal is still closed
MOKF=$TMPD/mokf; mkdir -p "$MOKF/mixture-of-kittens/mok" "$MOKF/host-runs" "$MOKF/runs"
ln -s "$DIR" "$MOKF/mixture-of-kittens/benchmarks"; cp "$GOODSO" "$MOKF/mixture-of-kittens/mok/_Cfixture.so"
set +e
O=$(BENCH_TAG=formalprobe BENCH_MODE=formal EXPECTED_RECEIPT_SHA256=$(sha256sum "$RCPT" | cut -d' ' -f1) \
    bash "$DIR/host_launch_sm90.sh" no-ct "$MOKF" "$MANF" "$RCPT" 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^FORMAL_MODE_FAIL:build-record contract not implemented$'; report F5_formal_still_blocked $?
echo "  F5 rc=$R want=14(launcher still refuses formal)"

[ "${BR_KEEP_TMPD:-0}" = "1" ] && echo "BR_TMPD_KEPT:$TMPD" || rm -rf "$TMPD"
EXPECTED=75
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "BR_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "BUILD_RECORD_TESTS pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
