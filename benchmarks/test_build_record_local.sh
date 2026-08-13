#!/bin/bash
# Build-record contract tests (tracked; no GPU, no real build, no docker).
#
# What these DO cover: that build_and_record_sm90.sh actually executes the
# command it records, refuses to adopt a pre-existing binary, refuses a failed
# build, refuses a pre-created log, refuses dirty inputs and submodule drift,
# and that the closure cannot be narrowed by configuration; that the packaging
# host recomputes the record's source claims; and that the formal binding is
# anchored on the receipt hash.
#
# What they do NOT cover, and no fixture can: a real nvcc build. The command
# under test is a fixture that writes a small file. Everything here is
# UNVERIFIED_REAL_BUILD.
#
# Usage: bash test_build_record_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
VALB=$DIR/validate_build_record_sm90.sh
VALR=$DIR/validate_receipt_sm90.sh
WRAP=$DIR/build_and_record_sm90.sh
BIND=$DIR/check_formal_binding_sm90.sh
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "BR_$1_PASS"; PASS=$((PASS+1)); else echo "BR_$1_FAIL"; FAIL=$((FAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }
H64() { printf "$1%.0s" $(seq 1 64); }
H40() { printf "$1%.0s" $(seq 1 40); }
GIT="git -c user.email=t@e.com -c user.name=t -c protocol.file.allow=always"

TMPD=$(mktemp -d)
# ---- fixture build host: tracked build inputs, a real submodule, a build
# command that writes a marker into the output so execution is provable ----
SUBSRC=$TMPD/subsrc
mkdir -p "$SUBSRC"
printf '// tk header v1\n' > "$SUBSRC/tk.h"
( cd "$SUBSRC" && $GIT init -q && $GIT add -A && $GIT commit -qm tk1 ) >/dev/null 2>&1
FIX=$TMPD/buildhost
mkdir -p "$FIX/csrc" "$FIX/mok" "$FIX/benchmarks"
printf '// fixture kernel\n' > "$FIX/csrc/bindings.cu"
printf 'all:\n\techo build\n' > "$FIX/Makefile"
printf '[build-system]\n' > "$FIX/pyproject.toml"
{ echo "BUILD_INPUT_SPEC=1"; echo "NAME=fixture-v1"; echo "PATH=Makefile"
  echo "PATH=pyproject.toml"; echo "PATH=csrc"; echo "SUBMODULE=third_party/tk"; } \
  > "$FIX/benchmarks/build_input_spec.v1"
( cd "$FIX" && $GIT init -q \
  && $GIT submodule add -q "$SUBSRC" third_party/tk \
  && $GIT add -A && $GIT commit -qm build-inputs ) >/dev/null 2>&1
FIXSRC=$($GIT -C "$FIX" rev-parse HEAD)
# the build command: writes a nonce-bearing output, so a record that did not
# come from this execution cannot contain the nonce
NONCE="nonce-$$-$(date -u +%s)"
CMD=$TMPD/cmd.sh
{ echo '#!/bin/bash'; echo 'echo "fixture build running"'
  echo "printf 'built %s\\n' \"$NONCE\" > mok/_Cfixture.so"; } > "$CMD"
chmod +x "$CMD"
SPECF=$TMPD/argv.spec
printf '%s\n%s\n' "bash" "$CMD" > "$SPECF"
SPECSHA=$(sha256sum "$SPECF" | cut -d' ' -f1)

runwrap() { # out_record log probe [extra env via caller]
  ( cd "$FIX" && BUILD_COMMAND_SPEC="${W_SPEC:-$SPECF}" BUILD_OUTPUT_PATH="mok/_Cfixture.so" \
      BUILD_LOG_PATH="$2" PROBE_LOG_PATH="$3" TARGET_ARCH="${W_ARCH:-SM90}" \
      TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF="build-image:cu130" \
      TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
      BUILD_INPUT_PATHS="csrc" \
      bash "$WRAP" "$FIX" "$1" 2>&1 )
}

# ---- W: the wrapper must really run the build ----
set +e
O=$(runwrap "$TMPD/rec.ok" "$TMPD/build1.log" "$TMPD/probe1.log"); R=$?
set -u
RECSHA=$(sha256sum "$TMPD/rec.ok" 2>/dev/null | cut -d' ' -f1)
RECMODE=$(stat -c %a "$TMPD/rec.ok" 2>/dev/null)
[ "$R" -eq 0 ] && has1 "$O" "^BUILD_RECORD_SHA256:$RECSHA\$" && [ "$RECMODE" = "444" ]; report W1_wrapper_runs_and_records $?
echo "  W1 rc=$R want=0(record published, mode=$RECMODE)"
grep -q "$NONCE" "$FIX/mok/_Cfixture.so" 2>/dev/null \
  && [ "$(grep '^SO_SHA256=' "$TMPD/rec.ok" | cut -d= -f2)" = "$(sha256sum "$FIX/mok/_Cfixture.so" | cut -d' ' -f1)" ]; report W2_output_is_from_this_execution $?
echo "  W2 output carries this execution's nonce and the record hashes it"
[ "$(grep '^BUILD_COMMAND_SPEC_SHA256=' "$TMPD/rec.ok" | cut -d= -f2)" = "$SPECSHA" ] \
  && [ "$(grep '^BUILD_EXIT_CODE=' "$TMPD/rec.ok" | cut -d= -f2)" = "0" ] \
  && grep -q '^PROVENANCE_CLASS=measured:.*attested:toolchain_image' "$TMPD/rec.ok"; report W3_command_and_provenance_pinned $?
echo "  W3 record pins the argv spec hash it exec'd and labels measured vs attested"
# keep a copy: the cases below deliberately destroy the build output (the
# wrapper removes it before every run), and the deployment fixtures need it
GOODSO=$TMPD/built_Cfixture.so
cp "$FIX/mok/_Cfixture.so" "$GOODSO"
# a stale binary must not be adopted: the command produces nothing this time
NOOP=$TMPD/noop.sh
{ echo '#!/bin/bash'; echo 'echo "did nothing"'; } > "$NOOP"; chmod +x "$NOOP"
printf 'STALE BINARY\n' > "$FIX/mok/_Cfixture.so"
printf '%s\n%s\n' "bash" "$NOOP" > "$TMPD/argv.noop"
set +e
O=$(W_SPEC=$TMPD/argv.noop runwrap "$TMPD/rec.stale" "$TMPD/build2.log" "$TMPD/probe2.log"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build produced no output at mok/_Cfixture\.so$' \
  && [ ! -f "$TMPD/rec.stale" ]; report W4_stale_output_not_adopted $?
echo "  W4 rc=$R want=2(pre-existing binary removed, no output produced, no record)"
# a failed build publishes nothing
BAD=$TMPD/bad.sh
{ echo '#!/bin/bash'; echo 'echo "compile error"'; echo 'exit 3'; } > "$BAD"; chmod +x "$BAD"
printf '%s\n%s\n' "bash" "$BAD" > "$TMPD/argv.bad"
set +e
O=$(W_SPEC=$TMPD/argv.bad runwrap "$TMPD/rec.bad" "$TMPD/build3.log" "$TMPD/probe3.log"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build command exited 3 \(no record is published for a failed build\)$' \
  && [ ! -f "$TMPD/rec.bad" ]; report W5_failed_build_no_record $?
echo "  W5 rc=$R want=2(nonzero build exit, nothing published)"
set +e
O=$(runwrap "$TMPD/rec.relog" "$TMPD/build1.log" "$TMPD/probe4.log"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:BUILD_LOG_PATH $TMPD/build1\.log already exists \(the log must be created by this execution\)\$"; report W6_log_must_be_exclusive $?
echo "  W6 rc=$R want=2(pre-existing build log)"
set +e
O=$(runwrap "$TMPD/rec.reprobe" "$TMPD/build5.log" "$TMPD/probe1.log"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^BUILD_RECORD_FAIL:PROBE_LOG_PATH $TMPD/probe1\.log already exists \(the probe log must be created by this execution\)\$"; report W7_probe_log_must_be_exclusive $?
echo "  W7 rc=$R want=2(pre-existing probe log)"
touch "$FIX/csrc/untracked.cu"
set +e
O=$(runwrap "$TMPD/rec.dirty" "$TMPD/build6.log" "$TMPD/probe6.log"); R=$?
set -u
rm -f "$FIX/csrc/untracked.cu"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build inputs not clean vs HEAD \(incl\. untracked\):$'; report W8_dirty_inputs $?
echo "  W8 rc=$R want=2(untracked file in the closure)"
# submodule drift: move the submodule to a different commit
printf '// tk header v2\n' >> "$SUBSRC/tk.h"
( cd "$SUBSRC" && $GIT add -A && $GIT commit -qm tk2 ) >/dev/null 2>&1
( cd "$FIX/third_party/tk" && $GIT fetch -q origin && $GIT checkout -q origin/HEAD 2>/dev/null || $GIT checkout -q origin/master 2>/dev/null || $GIT checkout -q origin/main ) >/dev/null 2>&1
set +e
O=$(runwrap "$TMPD/rec.drift" "$TMPD/build7.log" "$TMPD/probe7.log"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:submodule third_party/tk drifted or uninitialized:$'; report W9_submodule_drift $?
echo "  W9 rc=$R want=2(submodule moved off the recorded gitlink)"
( cd "$FIX" && $GIT submodule update -q --checkout --force third_party/tk ) >/dev/null 2>&1
# the closure cannot be narrowed by configuration: runwrap always exports
# BUILD_INPUT_PATHS=csrc, which the old design honoured and this one ignores
NFILES=$(grep '^BUILD_INPUT_FILE_COUNT=' "$TMPD/rec.ok" | cut -d= -f2)
[ "$NFILES" -eq 3 ]; report W10_closure_not_narrowable $?
echo "  W10 closure covers $NFILES files (Makefile+pyproject+csrc) despite BUILD_INPUT_PATHS=csrc"

# ---- V: schema-2 validator ----
vcase() { # name mutator want_rc reason-ERE
  local NAME=$1 MUT=$2 WANT=$3 REASON=$4
  local F=$TMPD/$NAME.record O R
  BASE=$TMPD/rec.ok eval "$MUT" > "$F"; chmod 444 "$F"
  set +e
  O=$(bash "$VALB" "$F" --check-mode 2>&1); R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}
set +e
O=$(bash "$VALB" "$TMPD/rec.ok" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 0 ] && has1 "$O" "^BUILD_RECORD_VALID:$RECSHA\$"; report V0_positive $?
echo "  V0 rc=$R want=0(BUILD_RECORD_VALID)"
vcase V1_schema1_rejected 'sed "s|^BUILD_RECORD_SCHEMA=2|BUILD_RECORD_SCHEMA=1|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:bad or missing schema version \(only 2 is valid; schema 1 records came from a generator that never ran a build\)$'
vcase V2_missing_key 'grep -v "^SO_BYTES=" "$BASE"' 17 '^BUILD_RECORD_FAIL:key SO_BYTES count=0 \(need exactly 1\)$'
vcase V3_duplicate_key 'cat "$BASE"; echo "TARGET_ARCH=SM90"' 17 '^BUILD_RECORD_FAIL:key TARGET_ARCH count=2 \(need exactly 1\)$'
vcase V4_unknown_key 'cat "$BASE"; echo "ROGUE=1"' 17 '^BUILD_RECORD_FAIL:unknown key ROGUE$'
vcase V5_empty_value 'sed "s|^MEASURED_NVCC_VERSION=.*|MEASURED_NVCC_VERSION=|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:key MEASURED_NVCC_VERSION empty$'
vcase V6_bad_so_hash 'sed "s|^SO_SHA256=.*|SO_SHA256=nothex|" "$BASE"' 17 '^BUILD_RECORD_FAIL:SO_SHA256 not 64-hex$'
vcase V7_bad_tree_oid 'sed "s|^SOURCE_TREE_GIT_OID=.*|SOURCE_TREE_GIT_OID=abc123|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:SOURCE_TREE_GIT_OID not 40-hex$'
vcase V8_nonzero_exit 'sed "s|^BUILD_EXIT_CODE=.*|BUILD_EXIT_CODE=1|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:BUILD_EXIT_CODE 1 != 0 \(a record may only describe a successful build\)$'
vcase V9_bad_utc 'sed "s|^BUILD_START_UTC=.*|BUILD_START_UTC=yesterday|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:BUILD_START_UTC not YYYY-MM-DDTHH:MM:SSZ$'
vcase V10_drops_attested_marker 'sed "s|^PROVENANCE_CLASS=.*|PROVENANCE_CLASS=measured:everything|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:PROVENANCE_CLASS must declare attested:toolchain_image \(image identity is not measured\)$'
vcase V11_bad_image_id 'sed "s|^TOOLCHAIN_IMAGE_ID_ATTESTED=.*|TOOLCHAIN_IMAGE_ID_ATTESTED=someimage|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:TOOLCHAIN_IMAGE_ID_ATTESTED malformed \(want sha256:<64-hex>\)$'
vcase V12_bad_arch 'sed "s|^TARGET_ARCH=.*|TARGET_ARCH=SM75|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:TARGET_ARCH not in allowed set \{SM90,SM100,SM103\}$'
vcase V13_basename_is_path 'sed "s|^SO_BASENAME=.*|SO_BASENAME=mok/_Cfixture.so|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:SO_BASENAME must be a bare filename, not a path$'
vcase V14_zero_bytes 'sed "s|^SO_BYTES=.*|SO_BYTES=0|" "$BASE"' 17 '^BUILD_RECORD_FAIL:SO_BYTES not a positive integer$'
cp "$TMPD/rec.ok" "$TMPD/rec.writable"; chmod 644 "$TMPD/rec.writable"
set +e
O=$(bash "$VALB" "$TMPD/rec.writable" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 17 ] && has1 "$O" '^BUILD_RECORD_FAIL:write bits set \(644\)$'; report V15_writable $?
echo "  V15 rc=$R want=17(write bits)"

# ---- S: packaging receipt recomputes the record's source claims ----
SO_SHA=$(grep '^SO_SHA256=' "$TMPD/rec.ok" | cut -d= -f2)
PKG=$TMPD/pkg
$GIT clone -q "$FIX" "$PKG" 2>/dev/null
( cd "$PKG" && $GIT submodule update -q --init --recursive ) >/dev/null 2>&1
mkdir -p "$PKG/benchmarks/manifests"
cp "$DIR/validate_manifest_sm90.sh" "$DIR/validate_receipt_sm90.sh" \
   "$DIR/validate_build_record_sm90.sh" "$DIR/compute_build_inputs_sm90.sh" \
   "$DIR/make_deploy_receipt.sh" "$DIR/bench_sm90_fwd.py" "$PKG/benchmarks/"
PKGH=$(sha256sum "$PKG/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
sed -e "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$SO_SHA|" \
    -e "s|^EXPECTED_HARNESS_SHA256=.*|EXPECTED_HARNESS_SHA256=$PKGH|" \
    "$DIR/manifests/tiny-h20-v1.manifest" > "$PKG/benchmarks/manifests/fix.manifest"
( cd "$PKG" && $GIT add -A && $GIT commit -qm pkg ) >/dev/null 2>&1
mkreceipt2() { # out record [binary_build_commit]
  ( cd "$PKG" && IMAGE_ID="sha256:$(H64 1)" IMAGE_REF="fixture-image:latest" \
      IMAGE_REPO_DIGESTS="registry.local/mok@sha256:$(H64 9)" \
      BINARY_BUILD_COMMIT="${3:-UNKNOWN}" BUILD_RECORD="${2:-}" \
      bash benchmarks/make_deploy_receipt.sh "$PKG" benchmarks/manifests/fix.manifest "$1" 2>&1 )
}
set +e
O=$(mkreceipt2 "$TMPD/r2.receipt" "$TMPD/rec.ok"); R=$?
set -u
R2SHA=$(sha256sum "$TMPD/r2.receipt" 2>/dev/null | cut -d' ' -f1)
[ "$R" -eq 0 ] && has1 "$O" "^EXPECTED_RECEIPT_SHA256:$R2SHA\$" \
  && [ "$(grep '^RECEIPT_SCHEMA=' "$TMPD/r2.receipt" | cut -d= -f2)" = "2" ] \
  && [ "$(grep '^BUILD_RECORD_SHA256=' "$TMPD/r2.receipt" | cut -d= -f2)" = "$RECSHA" ] \
  && [ "$(grep '^BINARY_BUILD_COMMIT=' "$TMPD/r2.receipt" | cut -d= -f2)" = "$FIXSRC" ]; report S1_schema2_from_record $?
echo "  S1 rc=$R want=0(schema 2, commit derived, inputs recomputed)"
scase() { # name record-mutator reason-ERE
  local NAME=$1 MUT=$2 REASON=$3 O R
  local F=$TMPD/$NAME.record
  BASE=$TMPD/rec.ok eval "$MUT" > "$F"; chmod 444 "$F"
  set +e
  O=$(mkreceipt2 "$TMPD/$NAME.receipt" "$F"); R=$?
  set -u
  [ "$R" -eq 2 ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=2($REASON)"
}
scase S2_tree_oid_forged 'sed "s|^SOURCE_TREE_GIT_OID=.*|SOURCE_TREE_GIT_OID=$(H40 c)|" "$BASE"' \
  "^RECEIPT_FAIL:record SOURCE_TREE_GIT_OID=$(H40 c) != recomputed [0-9a-f]{40} at $FIXSRC\$"
scase S3_input_list_forged 'sed "s|^BUILD_INPUT_LIST_SHA256=.*|BUILD_INPUT_LIST_SHA256=$(H64 c)|" "$BASE"' \
  "^RECEIPT_FAIL:record BUILD_INPUT_LIST_SHA256=$(H64 c) != recomputed [0-9a-f]{64} at $FIXSRC\$"
scase S4_spec_sha_forged 'sed "s|^BUILD_INPUT_SPEC_SHA256=.*|BUILD_INPUT_SPEC_SHA256=$(H64 d)|" "$BASE"' \
  "^RECEIPT_FAIL:record BUILD_INPUT_SPEC_SHA256=$(H64 d) != recomputed [0-9a-f]{64} at $FIXSRC\$"
scase S5_submodule_forged 'sed "s|^SUBMODULE_LIST_SHA256=.*|SUBMODULE_LIST_SHA256=$(H64 e)|" "$BASE"' \
  "^RECEIPT_FAIL:record SUBMODULE_LIST_SHA256=$(H64 e) != recomputed [0-9a-f]{64} at $FIXSRC\$"
scase S6_file_count_forged 'sed "s|^BUILD_INPUT_FILE_COUNT=.*|BUILD_INPUT_FILE_COUNT=99|" "$BASE"' \
  "^RECEIPT_FAIL:record BUILD_INPUT_FILE_COUNT=99 != recomputed 3 at $FIXSRC\$"
scase S7_manifest_so_disagreement 'sed "s|^SO_SHA256=.*|SO_SHA256=$(H64 f)|" "$BASE"' \
  "^RECEIPT_FAIL:manifest EXPECTED_SO_SHA256 $SO_SHA != build record SO_SHA256 $(H64 f)\$"
set +e
O=$(mkreceipt2 "$TMPD/r2.badcommit" "$TMPD/rec.ok" "$(H40 b)"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^RECEIPT_FAIL:BINARY_BUILD_COMMIT $(H40 b) != build record SOURCE_COMMIT $FIXSRC\$"; report S8_operator_cannot_assert_commit $?
echo "  S8 rc=$R want=2(operator-asserted commit refused)"

# ---- B: formal binding is anchored on the receipt ----
SODIR=$TMPD/deployed; mkdir -p "$SODIR"
cp "$GOODSO" "$SODIR/_Cfixture.so"
MANF=$PKG/benchmarks/manifests/fix.manifest
bcase() { # name expected_receipt receipt record manifest sodir want_rc reason-ERE
  local NAME=$1 EXP=$2 RCPT=$3 REC=$4 MAN=$5 SD=$6 WANT=$7 REASON=$8 O R
  set +e
  if [ "$EXP" = "__UNSET__" ]; then
    O=$(env -u EXPECTED_RECEIPT_SHA256 -u EXPECTED_BUILD_RECORD_SHA256 bash "$BIND" "$RCPT" "$REC" "$MAN" "$SD" 2>&1)
  else
    O=$(env -u EXPECTED_BUILD_RECORD_SHA256 EXPECTED_RECEIPT_SHA256="$EXP" bash "$BIND" "$RCPT" "$REC" "$MAN" "$SD" 2>&1)
  fi
  R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}
bcase B1_full_positive "$R2SHA" "$TMPD/r2.receipt" "$TMPD/rec.ok" "$MANF" "$SODIR" 0 \
  "^FORMAL_BINDING_PASS:$SO_SHA\$"
bcase B2_unanchored_receipt __UNSET__ "$TMPD/r2.receipt" "$TMPD/rec.ok" "$MANF" "$SODIR" 18 \
  '^FORMAL_BINDING_FAIL:EXPECTED_RECEIPT_SHA256 env missing or not 64-hex$'
bcase B3_receipt_sha_mismatch "$(H64 a)" "$TMPD/r2.receipt" "$TMPD/rec.ok" "$MANF" "$SODIR" 18 \
  "^FORMAL_BINDING_FAIL:receipt sha $R2SHA != EXPECTED_RECEIPT_SHA256 $(H64 a)\$"
# a swapped record cannot be laundered by supplying its own hash: the receipt
# is the root, and the receipt still points at the original record
sed "s|^MEASURED_TORCH_VERSION=.*|MEASURED_TORCH_VERSION=9.9.9|" "$TMPD/rec.ok" > "$TMPD/rec.swapped"
chmod 444 "$TMPD/rec.swapped"
SWSHA=$(sha256sum "$TMPD/rec.swapped" | cut -d' ' -f1)
bcase B4_record_swap_not_launderable "$R2SHA" "$TMPD/r2.receipt" "$TMPD/rec.swapped" "$MANF" "$SODIR" 18 \
  "^FORMAL_BINDING_FAIL:receipt BUILD_RECORD_SHA256 $RECSHA != record $SWSHA\$"
set +e
O=$(EXPECTED_RECEIPT_SHA256="$R2SHA" EXPECTED_BUILD_RECORD_SHA256="$SWSHA" \
    bash "$BIND" "$TMPD/r2.receipt" "$TMPD/rec.ok" "$MANF" "$SODIR" 2>&1); R=$?
set -u
[ "$R" -eq 18 ] && has1 "$O" "^FORMAL_BINDING_FAIL:record sha $RECSHA != optional cross-check EXPECTED_BUILD_RECORD_SHA256 $SWSHA\$"; report B5_optional_crosscheck $?
echo "  B5 rc=$R want=18(optional cross-check is enforced when supplied)"
SWAP=$TMPD/deployed_swapped; mkdir -p "$SWAP"
printf 'a DIFFERENT binary\n' > "$SWAP/_Cfixture.so"
bcase B6_deployed_so_swapped "$R2SHA" "$TMPD/r2.receipt" "$TMPD/rec.ok" "$MANF" "$SWAP" 18 \
  '^FORMAL_BINDING_FAIL:deployed so size [0-9]+ != record SO_BYTES [0-9]+$'
TWO=$TMPD/deployed_two; mkdir -p "$TWO"
cp "$GOODSO" "$TWO/_Cfixture.so"; cp "$GOODSO" "$TWO/_Csecond.so"
bcase B7_two_so "$R2SHA" "$TMPD/r2.receipt" "$TMPD/rec.ok" "$MANF" "$TWO" 18 \
  '^FORMAL_BINDING_FAIL:need exactly one _C\*\.so in .*, found 2$'

# ---- F: none of this opened formal ----
MOKF=$TMPD/mokf
mkdir -p "$MOKF/mixture-of-kittens/mok" "$MOKF/host-runs" "$MOKF/runs"
ln -s "$DIR" "$MOKF/mixture-of-kittens/benchmarks"
cp "$GOODSO" "$MOKF/mixture-of-kittens/mok/_Cfixture.so"
set +e
O=$(BENCH_TAG=formalprobe BENCH_MODE=formal EXPECTED_RECEIPT_SHA256="$R2SHA" \
    bash "$DIR/host_launch_sm90.sh" no-ct "$MOKF" "$MANF" "$TMPD/r2.receipt" 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^FORMAL_MODE_FAIL:build-record contract not implemented$'; report F1_formal_still_blocked $?
echo "  F1 rc=$R want=14(formal refused even with a record-bound receipt)"

# BR_KEEP_TMPD=1 preserves fixtures for debugging a failing case
[ "${BR_KEEP_TMPD:-0}" = "1" ] && echo "BR_TMPD_KEPT:$TMPD" || rm -rf "$TMPD"
EXPECTED=42
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "BR_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "BUILD_RECORD_TESTS pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
