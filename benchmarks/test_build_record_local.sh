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
  echo "printf 'built %s\\n' \"$NONCE\" > mok/_Cfixture.so"; } > "$FIX/tools/fake_build.sh"
chmod +x "$FIX/tools/fake_build.sh"
{ echo "BUILD_INPUT_SPEC=1"; echo "NAME=fixture-inputs-v1"; echo "PATH=Makefile"
  echo "PATH=pyproject.toml"; echo "PATH=csrc"; echo "PATH=tools"
  echo "SUBMODULE=third_party/tk"; } > "$FIX/benchmarks/build_input_spec.v1"
{ echo "BUILD_COMMAND_SPEC=1"; echo "NAME=fixture-command-v1"
  echo "OUTPUT=mok/_Cfixture.so"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "PROBE=nvcc|printf|release 13.0, V13.0.88"
  echo "PROBE=cc|printf|gcc (fixture) 12.3.0"
  echo "PROBE=python|printf|Python 3.12.3"
  echo "PROBE=torch|printf|2.11.0+cu130\\n13.0\\n"; } > "$FIX/benchmarks/build_command_spec.v1"
for T in build_and_record_sm90.sh compute_build_inputs_sm90.sh validate_build_record_sm90.sh; do
  cp "$DIR/$T" "$FIX/benchmarks/$T"
done
( cd "$FIX" && $GIT init -q && $GIT submodule add -q "$SUBSRC" third_party/tk \
  && $GIT add -A && $GIT commit -qm build-inputs ) >/dev/null 2>&1
FIXSRC=$($GIT -C "$FIX" rev-parse HEAD)
FIXWRAP=$FIX/benchmarks/build_and_record_sm90.sh

newart() { local D=$TMPD/art.$1; mkdir -p "$D"; echo "$D"; }
runnew() { # artifact-tag out-record [extra env from caller]
  ( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF="build:cu130" \
      TOOLCHAIN_IMAGE_REPO_DIGESTS="${N_DIGESTS:-registry.local/build@sha256:$(H64 3)}" \
      bash "$FIXWRAP" "$FIX" "$(newart "$1")" "$2" 2>&1 )
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
    bash "$FIXWRAP" "$FIX" "$(newart a1new)" "$TMPD/a1.new.record" 2>&1 ); NEWRC=$?
set -u
[ "$NEWRC" -eq 2 ] && has1 "$NEWOUT" '^BUILD_RECORD_FAIL:BUILD_FIXTURE_COMMAND_SPEC is set outside fixture mode$' \
  && [ ! -f "$TMPD/a1.new.record" ]; report A1b_new_refuses_external_command $?
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
    bash "$TMPD/tampered_wrapper.sh" "$FIX" "$(newart a2new)" "$TMPD/a2.new.record" 2>&1 ); NEWRC=$?
set -u
[ "$NEWRC" -eq 2 ] && has1 "$NEWOUT" '^BUILD_RECORD_FAIL:wrapper is running from .*, not the repository.s benchmarks/build_and_record_sm90\.sh$'; report A2b_new_refuses_foreign_wrapper $?
echo "  A2b new wrapper rc=$NEWRC (self-binding rejects a copy run from elsewhere)"

# A3 BUILD_INPUT_SPEC_PATH: a caller override of the closure
{ echo "BUILD_INPUT_SPEC=1"; echo "NAME=narrow"; echo "PATH=csrc"; } > "$FIX/benchmarks/narrow.spec"
( cd "$FIX" && $GIT add -A && $GIT commit -qm narrow ) >/dev/null 2>&1
NARROWSRC=$($GIT -C "$FIX" rev-parse HEAD)
set +e
OLDOUT=$( cd "$FIX" && BUILD_INPUT_SPEC_PATH=benchmarks/narrow.spec BUILD_COMMAND_SPEC="$TMPD/evil.argv" \
    BUILD_OUTPUT_PATH=mok/_Cfixture.so BUILD_LOG_PATH="$TMPD/a3.build.log" \
    PROBE_LOG_PATH="$TMPD/a3.probe.log" TARGET_ARCH=SM90 \
    TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
    bash "$OLDWRAP" "$FIX" "$TMPD/a3.old.record" 2>&1 ); OLDRC=$?
set -u
OLDCOUNT=$(grep '^BUILD_INPUT_FILE_COUNT=' "$TMPD/a3.old.record" 2>/dev/null | cut -d= -f2)
[ "$OLDRC" -eq 0 ] && [ "${OLDCOUNT:-0}" -eq 1 ]; report A3a_old_honoured_narrowed_closure $?
echo "  A3a old wrapper rc=$OLDRC recorded a closure of ${OLDCOUNT:-?} file(s) instead of the full spec"
set +e
NEWOUT=$(BUILD_INPUT_SPEC_PATH=benchmarks/narrow.spec runnew a3new "$TMPD/a3.new.record"); NEWRC=$?
set -u
NEWCOUNT=$(grep '^BUILD_INPUT_FILE_COUNT=' "$TMPD/a3.new.record" 2>/dev/null | cut -d= -f2)
[ "$NEWRC" -eq 0 ] && [ "${NEWCOUNT:-0}" -eq 4 ]; report A3b_new_ignores_spec_override $?
echo "  A3b new wrapper rc=$NEWRC closure=${NEWCOUNT:-?} files, want 4 (override ignored; spec path is fixed)"

# ---------- N: the current implementation's own gates ----------
set +e
O=$(runnew n1 "$TMPD/rec.ok"); R=$?
set -u
RECSHA=$(sha256sum "$TMPD/rec.ok" 2>/dev/null | cut -d' ' -f1)
[ "$R" -eq 0 ] && has1 "$O" "^BUILD_RECORD_SHA256:$RECSHA\$" && has1 "$O" '^RECORD_MODE:production$' \
  && [ "$(stat -c %a "$TMPD/rec.ok")" = "444" ]; report N1_production_positive $?
echo "  N1 rc=$R want=0(production record published)"
grep -q "$NONCE" "$FIX/mok/_Cfixture.so" 2>/dev/null \
  && [ "$(grep '^SO_SHA256=' "$TMPD/rec.ok" | cut -d= -f2)" = "$(sha256sum "$FIX/mok/_Cfixture.so" | cut -d' ' -f1)" ] \
  && [ "$(grep '^BUILD_COMMAND_ARGV_JOINED=' "$TMPD/rec.ok" | cut -d= -f2-)" = "bash tools/fake_build.sh ARCH=SM90" ] \
  && [ "$(grep '^TARGET_ARCH=' "$TMPD/rec.ok" | cut -d= -f2)" = "SM90" ]; report N2_argv_and_output_from_tracked_spec $?
echo "  N2 argv, ARCH and output all come from the tracked command spec"
GOODSO=$TMPD/built.so; cp "$FIX/mok/_Cfixture.so" "$GOODSO"
GOODREC=$TMPD/rec.good; cp "$TMPD/rec.ok" "$GOODREC"; chmod 444 "$GOODREC"
# Make-style variables cannot be injected from the environment
set +e
O=$(SRC=/evil.cu NVCC=/evil-nvcc PYTHON=/evil-python THUNDERKITTENS_ROOT=/evil ARCH=SM100 \
    OUT=/tmp/escaped.so runnew n3 "$TMPD/rec.envoverride"); R=$?
set -u
[ "$R" -eq 0 ] \
  && [ "$(grep '^BUILD_COMMAND_ARGV_JOINED=' "$TMPD/rec.envoverride" | cut -d= -f2-)" = "bash tools/fake_build.sh ARCH=SM90" ] \
  && [ "$(grep '^TARGET_ARCH=' "$TMPD/rec.envoverride" | cut -d= -f2)" = "SM90" ] \
  && [ ! -f /tmp/escaped.so ]; report N3_make_vars_not_injectable $?
echo "  N3 rc=$R SRC/NVCC/PYTHON/TK/ARCH/OUT from the environment changed nothing"
# tooling tamper in the worktree
cp "$FIX/benchmarks/compute_build_inputs_sm90.sh" "$TMPD/cbi.bak"
echo "# tampered" >> "$FIX/benchmarks/compute_build_inputs_sm90.sh"
set +e
O=$(runnew n4 "$TMPD/rec.tampered"); R=$?
set -u
cp "$TMPD/cbi.bak" "$FIX/benchmarks/compute_build_inputs_sm90.sh"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:tooling file benchmarks/compute_build_inputs_sm90\.sh differs from its committed bytes at [0-9a-f]{40} \([0-9a-f]{64} != [0-9a-f]{64}\)$'; report N4_tooling_tamper_refused $?
echo "  N4 rc=$R want=2(helper modified in the worktree)"
# fixture mode is labelled and isolated
FIXCMD=$TMPD/fixture.cmdspec
{ echo "NAME=fixture-arbitrary"; echo "OUTPUT=mok/_Cfixture.so"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "PROBE=nvcc|printf|release 13.0, V13.0.88"; echo "PROBE=cc|printf|gcc 12.3.0"
  echo "PROBE=python|printf|Python 3.12.3"; echo "PROBE=torch|printf|2.11.0+cu130\\n13.0\\n"; } > "$FIXCMD"
set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$FIXCMD" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
     bash "$FIXWRAP" "$FIX" "$(newart n5)" "$TMPD/rec.fixture" 2>&1 ); R=$?
set -u
[ "$R" -eq 0 ] && grep -q '^RECORD_MODE=fixture$' "$TMPD/rec.fixture"; report N5_fixture_mode_labelled $?
echo "  N5 rc=$R want=0(fixture record carries RECORD_MODE=fixture)"
# output symlink
SYMCMD=$TMPD/sym.cmdspec
{ echo "NAME=fixture-symlink"; echo "OUTPUT=mok/_Cfixture.so"
  echo "ARGV=bash"; echo "ARGV=-c"; echo "ARGV=ln -s $TMPD/old_prebuilt.so mok/_Cfixture.so"
  echo "ARGV=ARCH=SM90"
  echo "PROBE=nvcc|printf|release 13.0"; echo "PROBE=cc|printf|gcc"; echo "PROBE=python|printf|Python 3"
  echo "PROBE=torch|printf|2.11.0\\n13.0\\n"; } > "$SYMCMD"
set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$SYMCMD" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$(newart n6)" "$TMPD/rec.sym" 2>&1 ); R=$?
set -u
rm -f "$FIX/mok/_Cfixture.so"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build created mok/_Cfixture\.so as a symlink; refusing to hash the target$' \
  && [ ! -f "$TMPD/rec.sym" ]; report N6_output_symlink_refused $?
echo "  N6 rc=$R want=2(output created as a symlink to an old binary)"
# probe failure
BADPROBE=$TMPD/badprobe.cmdspec
{ echo "NAME=fixture-badprobe"; echo "OUTPUT=mok/_Cfixture.so"
  echo "ARGV=bash"; echo "ARGV=tools/fake_build.sh"; echo "ARGV=ARCH=SM90"
  echo "PROBE=nvcc|false"; echo "PROBE=cc|printf|gcc"; echo "PROBE=python|printf|Python 3"
  echo "PROBE=torch|printf|2.11.0\\n13.0\\n"; } > "$BADPROBE"
set +e
O=$( cd "$FIX" && BUILD_RECORD_FIXTURE_MODE=1 BUILD_FIXTURE_COMMAND_SPEC="$BADPROBE" \
     TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     bash "$FIXWRAP" "$FIX" "$(newart n7)" "$TMPD/rec.badprobe" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:probe nvcc exited 1; the toolchain cannot be described and no record is published$' \
  && [ ! -f "$TMPD/rec.badprobe" ]; report N7_probe_failure_refused $?
echo "  N7 rc=$R want=2(a probe failed; no record)"
# artifact dir problems
AD=$TMPD/art.reuse; mkdir -p "$AD"; : > "$AD/build.log"
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE bash "$FIXWRAP" "$FIX" "$AD" "$TMPD/rec.reuse" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:build log $AD/build\.log already exists or is not creatable\$"; report N8_log_must_be_exclusive $?
echo "  N8 rc=$R want=2(build.log already present in the artifact dir)"
ln -s "$TMPD" "$TMPD/art.symlink"
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=b:1 \
     TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE bash "$FIXWRAP" "$FIX" "$TMPD/art.symlink" "$TMPD/rec.artsym" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:artifact dir $TMPD/art\.symlink is a symlink\$"; report N9_artifact_dir_symlink_refused $?
echo "  N9 rc=$R want=2(artifact dir is a symlink)"
# dirty closure and submodule drift still refused
touch "$FIX/csrc/untracked.cu"
set +e
O=$(runnew n10 "$TMPD/rec.dirty"); R=$?
set -u
rm -f "$FIX/csrc/untracked.cu"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build inputs not clean vs HEAD \(incl\. untracked\):$'; report N10_dirty_inputs $?
echo "  N10 rc=$R want=2(untracked file in the closure)"
printf '// tk v2\n' >> "$SUBSRC/tk.h"
( cd "$SUBSRC" && $GIT add -A && $GIT commit -qm tk2 ) >/dev/null 2>&1
( cd "$FIX/third_party/tk" && $GIT fetch -q origin && { $GIT checkout -q origin/HEAD || $GIT checkout -q origin/master || $GIT checkout -q origin/main; } ) >/dev/null 2>&1
set +e
O=$(runnew n11 "$TMPD/rec.drift"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:submodule third_party/tk drifted or uninitialized:$'; report N11_submodule_drift $?
echo "  N11 rc=$R want=2(submodule off its recorded gitlink)"
( cd "$FIX" && $GIT submodule update -q --checkout --force third_party/tk ) >/dev/null 2>&1

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
  echo "BUILD_RECORD_SHA256=$(sha256sum "$TMPD/rec.fixture" | cut -d' ' -f1)"; } > "$RCPT"
chmod 444 "$RCPT"
SODIR=$TMPD/deployed; mkdir -p "$SODIR"; cp "$GOODSO" "$SODIR/_Cfixture.so"
set +e
O=$(EXPECTED_RECEIPT_SHA256=$(sha256sum "$RCPT" | cut -d' ' -f1) \
    bash "$BIND" "$RCPT" "$TMPD/rec.fixture" "$MANF" "$SODIR" 2>&1); R=$?
set -u
[ "$R" -eq 18 ] && has1 "$O" '^FORMAL_BINDING_FAIL:build record RECORD_MODE=fixture is not a production record$'; report F1_fixture_record_refused $?
echo "  F1 rc=$R want=18(fixture record cannot satisfy formal)"
NONEREC=$TMPD/rec.none
sed 's|^TOOLCHAIN_IMAGE_REPO_DIGESTS_ATTESTED=.*|TOOLCHAIN_IMAGE_REPO_DIGESTS_ATTESTED=NONE|' \
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
set +e
O=$(bash "$VALB" "$TMPD/rec.fixture" --check-mode 2>&1); R=$?
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
EXPECTED=30
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "BR_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "BUILD_RECORD_TESTS pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
