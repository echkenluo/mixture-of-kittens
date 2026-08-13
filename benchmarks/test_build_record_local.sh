#!/bin/bash
# Build-record contract tests (tracked, no GPU, no build, no docker).
#
# Scope note that must travel with any result from this file: these tests
# exercise the CONTRACT - schema rules, the generator's refusals, the
# receipt binding and the six-way formal binding - against synthesized
# fixtures. They do not and cannot prove anything about a real build: no
# build has been performed, the current .so still has NO record, and
# host_launch_sm90.sh still refuses formal mode unconditionally (asserted
# below, so implementing the contract cannot silently open formal).
#
# Usage: bash test_build_record_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
VALB=$DIR/validate_build_record_sm90.sh
VALR=$DIR/validate_receipt_sm90.sh
MKB=$DIR/make_build_record_sm90.sh
BIND=$DIR/check_formal_binding_sm90.sh
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "BR_$1_PASS"; PASS=$((PASS+1)); else echo "BR_$1_FAIL"; FAIL=$((FAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }
H64() { printf "$1%.0s" $(seq 1 64); }
H40() { printf "$1%.0s" $(seq 1 40); }

TMPD=$(mktemp -d)
# ---- fixture build host: a repo with tracked build inputs, a real (fake) .so
# and a build log. The generator measures these bytes; nothing is asserted.
FIX=$TMPD/buildhost
mkdir -p "$FIX/csrc" "$FIX/mok"
printf '// fixture kernel\n' > "$FIX/csrc/kernel.cu"
printf 'all:\n\techo build\n' > "$FIX/Makefile"
( cd "$FIX" && git init -q && git add -A \
  && git -c user.email=t@e.com -c user.name=t commit -qm build-inputs ) >/dev/null 2>&1
FIXSRC=$(git -C "$FIX" rev-parse HEAD)
printf 'fixture shared object bytes\n' > "$FIX/mok/_Cfixture.so"
printf 'nvcc: fixture build log\n' > "$FIX/build.log"
SO_SHA=$(sha256sum "$FIX/mok/_Cfixture.so" | cut -d' ' -f1)
SO_BYTES=$(stat -c %s "$FIX/mok/_Cfixture.so")

genrecord() { # out [extra env overrides applied by caller]
  ( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" \
      TOOLCHAIN_IMAGE_REF="build-image:cu130" \
      TOOLCHAIN_IMAGE_REPO_DIGESTS="registry.local/build@sha256:$(H64 3)" \
      BUILD_COMMAND_ARGV="${MK_ARGV:-make ARCH=SM90 -j16}" \
      TARGET_ARCH="${MK_ARCH:-SM90}" CUDA_VERSION="13.0" NVCC_VERSION="13.0.88" \
      HOST_COMPILER_VERSION="gcc 12.3.0" PYTHON_VERSION="3.12.3" TORCH_VERSION="2.11.0+cu130" \
      bash "$MKB" "$FIX" "$FIX/mok/_Cfixture.so" "$FIX/build.log" "$1" 2>&1 )
}

# ---- generator ----
set +e
O=$(genrecord "$TMPD/rec.ok"); R=$?
set -u
RECSHA=$(sha256sum "$TMPD/rec.ok" 2>/dev/null | cut -d' ' -f1)
RECMODE=$(stat -c %a "$TMPD/rec.ok" 2>/dev/null)
[ "$R" -eq 0 ] && has1 "$O" "^EXPECTED_BUILD_RECORD_SHA256:$RECSHA\$" && [ "$RECMODE" = "444" ]; report G1_generate_ok $?
echo "  G1 rc=$R want=0(prints expected sha, mode=$RECMODE)"
# the record must carry the REAL measured bytes, not anything supplied
[ "$(grep '^SO_SHA256=' "$TMPD/rec.ok" | cut -d= -f2)" = "$SO_SHA" ] \
  && [ "$(grep '^SO_BYTES=' "$TMPD/rec.ok" | cut -d= -f2)" = "$SO_BYTES" ] \
  && [ "$(grep '^SOURCE_COMMIT=' "$TMPD/rec.ok" | cut -d= -f2)" = "$FIXSRC" ]; report G2_measures_real_bytes $?
echo "  G2 so_sha/so_bytes/source_commit measured from the build host"
touch "$FIX/csrc/untracked.cu"
set +e
O=$(genrecord "$TMPD/rec.dirty"); R=$?
set -u
rm -f "$FIX/csrc/untracked.cu"
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build inputs not clean vs HEAD \(incl\. untracked\):$'; report G3_dirty_inputs $?
echo "  G3 rc=$R want=2(dirty build inputs)"
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=x TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     BUILD_COMMAND_ARGV=make TARGET_ARCH=SM90 CUDA_VERSION=1 NVCC_VERSION=1 HOST_COMPILER_VERSION=1 \
     PYTHON_VERSION=1 TORCH_VERSION=1 bash "$MKB" "$FIX" "$FIX/mok/nope.so" "$FIX/build.log" "$TMPD/rec.noso" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:so .*nope\.so is not a regular file$'; report G4_missing_so $?
echo "  G4 rc=$R want=2(missing .so)"
set +e
O=$( cd "$FIX" && TOOLCHAIN_IMAGE_ID="sha256:$(H64 2)" TOOLCHAIN_IMAGE_REF=x TOOLCHAIN_IMAGE_REPO_DIGESTS=NONE \
     BUILD_COMMAND_ARGV=make TARGET_ARCH=SM90 CUDA_VERSION=1 NVCC_VERSION=1 HOST_COMPILER_VERSION=1 \
     PYTHON_VERSION=1 TORCH_VERSION=1 bash "$MKB" "$FIX" "$FIX/mok/_Cfixture.so" "$FIX/nolog.txt" "$TMPD/rec.nolog" 2>&1 ); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:build log .*nolog\.txt is not a regular file$'; report G5_missing_log $?
echo "  G5 rc=$R want=2(missing build log)"
# publish safety: a record that fails its own validation must not be published
# and must not clobber the good one (fault injected via TARGET_ARCH)
PRE=$(sha256sum "$TMPD/rec.ok" | cut -d' ' -f1)
set +e
MK_ARCH=SM75 O=$(MK_ARCH=SM75 genrecord "$TMPD/rec.ok"); R=$?
set -u
POST=$(sha256sum "$TMPD/rec.ok" | cut -d' ' -f1)
NLEFT=$(ls "$TMPD"/.buildrecord.* 2>/dev/null | wc -l)
[ "$R" -eq 2 ] && has1 "$O" '^BUILD_RECORD_FAIL:generated record failed self-validation \(not published\)$' \
  && [ "$PRE" = "$POST" ] && [ "$NLEFT" -eq 0 ]; report G6_publish_atomicity $?
echo "  G6 rc=$R want=2(not published; OUT unchanged=$([ "$PRE" = "$POST" ] && echo yes || echo no); temps=$NLEFT)"

# ---- record validator ----
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
vcase V1_missing_key 'grep -v "^SO_BYTES=" "$BASE"' 17 '^BUILD_RECORD_FAIL:key SO_BYTES count=0 \(need exactly 1\)$'
vcase V2_duplicate_key 'cat "$BASE"; echo "TARGET_ARCH=SM90"' 17 '^BUILD_RECORD_FAIL:key TARGET_ARCH count=2 \(need exactly 1\)$'
vcase V3_unknown_key 'cat "$BASE"; echo "ROGUE=1"' 17 '^BUILD_RECORD_FAIL:unknown key ROGUE$'
vcase V4_empty_value 'sed "s|^NVCC_VERSION=.*|NVCC_VERSION=|" "$BASE"' 17 '^BUILD_RECORD_FAIL:key NVCC_VERSION empty$'
vcase V5_bad_so_hash 'sed "s|^SO_SHA256=.*|SO_SHA256=nothex|" "$BASE"' 17 '^BUILD_RECORD_FAIL:SO_SHA256 not 64-hex$'
vcase V6_bad_source_commit 'sed "s|^SOURCE_COMMIT=.*|SOURCE_COMMIT=abc123|" "$BASE"' 17 '^BUILD_RECORD_FAIL:SOURCE_COMMIT not 40-hex$'
vcase V7_bad_utc 'sed "s|^BUILD_UTC=.*|BUILD_UTC=yesterday|" "$BASE"' 17 '^BUILD_RECORD_FAIL:BUILD_UTC not YYYY-MM-DDTHH:MM:SSZ$'
vcase V8_bad_image_id 'sed "s|^TOOLCHAIN_IMAGE_ID=.*|TOOLCHAIN_IMAGE_ID=someimage|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:TOOLCHAIN_IMAGE_ID malformed \(want sha256:<64-hex>\)$'
vcase V9_bad_repo_digests 'sed "s|^TOOLCHAIN_IMAGE_REPO_DIGESTS=.*|TOOLCHAIN_IMAGE_REPO_DIGESTS=whatever|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:TOOLCHAIN_IMAGE_REPO_DIGESTS malformed \(want NONE or repo@sha256:<64-hex> list\)$'
vcase V10_bad_arch 'sed "s|^TARGET_ARCH=.*|TARGET_ARCH=SM75|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:TARGET_ARCH not in allowed set \{SM90,SM100,SM103\}$'
vcase V11_basename_is_path 'sed "s|^SO_BASENAME=.*|SO_BASENAME=mok/_Cfixture.so|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:SO_BASENAME must be a bare filename, not a path$'
vcase V12_basename_not_ext 'sed "s|^SO_BASENAME=.*|SO_BASENAME=libfoo.so|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:SO_BASENAME does not look like a built extension \(_C\*\.so\)$'
vcase V13_zero_bytes 'sed "s|^SO_BYTES=.*|SO_BYTES=0|" "$BASE"' 17 '^BUILD_RECORD_FAIL:SO_BYTES not a positive integer$'
vcase V14_bad_schema 'sed "s|^BUILD_RECORD_SCHEMA=1|BUILD_RECORD_SCHEMA=2|" "$BASE"' 17 \
  '^BUILD_RECORD_FAIL:bad or missing schema version$'
cp "$TMPD/rec.ok" "$TMPD/rec.writable"; chmod 644 "$TMPD/rec.writable"
set +e
O=$(bash "$VALB" "$TMPD/rec.writable" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 17 ] && has1 "$O" '^BUILD_RECORD_FAIL:write bits set \(644\)$'; report V15_writable $?
echo "  V15 rc=$R want=17(write bits)"

# ---- receipt schema 2: binding a receipt to a record ----
# fixture packaging repo: a benchmarks tree with a manifest whose expected SO
# equals the record's SO, so the binding can be exercised end to end
# The packaging repo is a CLONE of the build host's repo: in reality they are
# the same source repository, and the record's SOURCE_COMMIT must resolve on
# the packaging side (a record naming a commit the packaging repo has never
# seen is refused - case S6 below).
PKG=$TMPD/pkg
git clone -q "$FIX" "$PKG" 2>/dev/null
mkdir -p "$PKG/benchmarks/manifests"
cp "$DIR/validate_manifest_sm90.sh" "$DIR/validate_receipt_sm90.sh" \
   "$DIR/validate_build_record_sm90.sh" "$DIR/make_deploy_receipt.sh" \
   "$DIR/bench_sm90_fwd.py" "$PKG/benchmarks/"
PKGH=$(sha256sum "$PKG/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
sed -e "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$SO_SHA|" \
    -e "s|^EXPECTED_HARNESS_SHA256=.*|EXPECTED_HARNESS_SHA256=$PKGH|" \
    "$DIR/manifests/tiny-h20-v1.manifest" > "$PKG/benchmarks/manifests/fix.manifest"
( cd "$PKG" && git add -A \
  && git -c user.email=t@e.com -c user.name=t commit -qm pkg ) >/dev/null 2>&1
mkreceipt2() { # out [build_record] [binary_build_commit]
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
echo "  S1 rc=$R want=0(schema 2, commit derived from the record)"
set +e
O=$(mkreceipt2 "$TMPD/r2.bad" "$TMPD/rec.ok" "$(H40 b)"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^RECEIPT_FAIL:BINARY_BUILD_COMMIT $(H40 b) != build record SOURCE_COMMIT $FIXSRC\$"; report S2_operator_cannot_assert_commit $?
echo "  S2 rc=$R want=2(operator-asserted commit refused)"
sed "s|^SO_SHA256=.*|SO_SHA256=$(H64 e)|" "$TMPD/rec.ok" > "$TMPD/rec.otherso"; chmod 444 "$TMPD/rec.otherso"
set +e
O=$(mkreceipt2 "$TMPD/r2.otherso" "$TMPD/rec.otherso"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" "^RECEIPT_FAIL:manifest EXPECTED_SO_SHA256 $SO_SHA != build record SO_SHA256 $(H64 e)\$"; report S3_manifest_so_disagreement $?
echo "  S3 rc=$R want=2(manifest SO != record SO)"
# a record whose SOURCE_COMMIT the packaging repo has never seen is refused:
# derived-from-record does not mean unchecked
sed "s|^SOURCE_COMMIT=.*|SOURCE_COMMIT=$(H40 e)|" "$TMPD/rec.ok" > "$TMPD/rec.alien"; chmod 444 "$TMPD/rec.alien"
set +e
O=$(mkreceipt2 "$TMPD/r2.alien" "$TMPD/rec.alien"); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^RECEIPT_FAIL:BINARY_BUILD_COMMIT not resolvable in this repo$'; report S6_record_commit_unknown_to_repo $?
echo "  S6 rc=$R want=2(record SOURCE_COMMIT not in the packaging repo)"
{ grep -v '^BUILD_RECORD_SHA256=' "$TMPD/r2.receipt"; } > "$TMPD/r2.norec"; chmod 444 "$TMPD/r2.norec"
set +e
O=$(bash "$VALR" "$TMPD/r2.norec" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^RECEIPT_TRUST_FAIL:key BUILD_RECORD_SHA256 count=0 \(need exactly 1\)$'; report S4_schema2_missing_binding $?
echo "  S4 rc=$R want=14(schema 2 without the binding key)"
sed 's|^BINARY_BUILD_COMMIT=.*|BINARY_BUILD_COMMIT=UNKNOWN|' "$TMPD/r2.receipt" > "$TMPD/r2.unk"; chmod 444 "$TMPD/r2.unk"
set +e
O=$(bash "$VALR" "$TMPD/r2.unk" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^RECEIPT_TRUST_FAIL:schema 2 receipt with BINARY_BUILD_COMMIT=UNKNOWN \(a record-bound receipt cannot disclaim its build\)$'; report S5_schema2_unknown_commit $?
echo "  S5 rc=$R want=14(schema 2 claiming no build)"

# ---- six-way formal binding ----
SODIR=$TMPD/deployed
mkdir -p "$SODIR"
cp "$FIX/mok/_Cfixture.so" "$SODIR/"
MANF=$PKG/benchmarks/manifests/fix.manifest
bcase() { # name expected_rec record receipt manifest sodir want_rc reason-ERE
  local NAME=$1 EXP=$2 REC=$3 RCPT=$4 MAN=$5 SD=$6 WANT=$7 REASON=$8 O R
  set +e
  if [ "$EXP" = "__UNSET__" ]; then
    O=$(env -u EXPECTED_BUILD_RECORD_SHA256 bash "$BIND" "$REC" "$RCPT" "$MAN" "$SD" 2>&1)
  else
    O=$(EXPECTED_BUILD_RECORD_SHA256="$EXP" bash "$BIND" "$REC" "$RCPT" "$MAN" "$SD" 2>&1)
  fi
  R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}
bcase B1_full_positive "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.receipt" "$MANF" "$SODIR" 0 \
  "^FORMAL_BINDING_PASS:$SO_SHA\$"
bcase B2_expected_env_missing __UNSET__ "$TMPD/rec.ok" "$TMPD/r2.receipt" "$MANF" "$SODIR" 18 \
  '^FORMAL_BINDING_FAIL:EXPECTED_BUILD_RECORD_SHA256 env missing or not 64-hex$'
bcase B3_record_sha_mismatch "$(H64 a)" "$TMPD/rec.ok" "$TMPD/r2.receipt" "$MANF" "$SODIR" 18 \
  "^FORMAL_BINDING_FAIL:record sha $RECSHA != EXPECTED_BUILD_RECORD_SHA256 $(H64 a)\$"
bcase B4_schema1_receipt "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.norec" "$MANF" "$SODIR" 18 \
  '^FORMAL_BINDING_FAIL:receipt failed shared validator$'
sed "s|^BUILD_RECORD_SHA256=.*|BUILD_RECORD_SHA256=$(H64 f)|" "$TMPD/r2.receipt" > "$TMPD/r2.wrongrec"; chmod 444 "$TMPD/r2.wrongrec"
bcase B5_receipt_points_elsewhere "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.wrongrec" "$MANF" "$SODIR" 18 \
  "^FORMAL_BINDING_FAIL:receipt BUILD_RECORD_SHA256 $(H64 f) != record $RECSHA\$"
sed "s|^BINARY_BUILD_COMMIT=.*|BINARY_BUILD_COMMIT=$(H40 c)|" "$TMPD/r2.receipt" > "$TMPD/r2.wrongcommit"; chmod 444 "$TMPD/r2.wrongcommit"
bcase B6_commit_disagreement "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.wrongcommit" "$MANF" "$SODIR" 18 \
  "^FORMAL_BINDING_FAIL:receipt BINARY_BUILD_COMMIT $(H40 c) != record SOURCE_COMMIT $FIXSRC\$"
sed "s|^SO_SHA256=.*|SO_SHA256=$(H64 d)|" "$TMPD/r2.receipt" > "$TMPD/r2.wrongso"; chmod 444 "$TMPD/r2.wrongso"
bcase B7_receipt_so_disagreement "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.wrongso" "$MANF" "$SODIR" 18 \
  '^FORMAL_BINDING_FAIL:receipt SO_SHA256 != record SO_SHA256$'
sed "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$(H64 d)|" "$MANF" > "$TMPD/man.wrongso"
bcase B8_manifest_so_disagreement "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.receipt" "$TMPD/man.wrongso" "$SODIR" 18 \
  '^FORMAL_BINDING_FAIL:manifest EXPECTED_SO_SHA256 != record SO_SHA256$'
SWAP=$TMPD/deployed_swapped; mkdir -p "$SWAP"
printf 'a DIFFERENT binary\n' > "$SWAP/_Cfixture.so"
bcase B9_deployed_so_swapped "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.receipt" "$MANF" "$SWAP" 18 \
  '^FORMAL_BINDING_FAIL:deployed so size [0-9]+ != record SO_BYTES [0-9]+$'
RENAMED=$TMPD/deployed_renamed; mkdir -p "$RENAMED"
cp "$FIX/mok/_Cfixture.so" "$RENAMED/_Cother.so"
bcase B10_basename_mismatch "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.receipt" "$MANF" "$RENAMED" 18 \
  '^FORMAL_BINDING_FAIL:deployed basename _Cother\.so != record SO_BASENAME _Cfixture\.so$'
TWO=$TMPD/deployed_two; mkdir -p "$TWO"
cp "$FIX/mok/_Cfixture.so" "$TWO/_Cfixture.so"; cp "$FIX/mok/_Cfixture.so" "$TWO/_Csecond.so"
bcase B11_two_so "$RECSHA" "$TMPD/rec.ok" "$TMPD/r2.receipt" "$MANF" "$TWO" 18 \
  '^FORMAL_BINDING_FAIL:need exactly one _C\*\.so in .*, found 2$'

# ---- implementing the contract must NOT have opened formal ----
MOKF=$TMPD/mokf
mkdir -p "$MOKF/mixture-of-kittens/mok" "$MOKF/host-runs" "$MOKF/runs"
ln -s "$DIR" "$MOKF/mixture-of-kittens/benchmarks"
cp "$FIX/mok/_Cfixture.so" "$MOKF/mixture-of-kittens/mok/"
set +e
O=$(BENCH_TAG=formalprobe BENCH_MODE=formal EXPECTED_RECEIPT_SHA256="$R2SHA" \
    bash "$DIR/host_launch_sm90.sh" no-ct "$MOKF" "$MANF" "$TMPD/r2.receipt" 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^FORMAL_MODE_FAIL:build-record contract not implemented$'; report F1_formal_still_blocked $?
echo "  F1 rc=$R want=14(formal still refused even with a record-bound receipt)"
# and the current production .so genuinely has no record to bind
CURSO=$(grep '^EXPECTED_SO_SHA256=' "$DIR/manifests/tiny-h20-v1.manifest" | cut -d= -f2)
[ -z "$(ls "$DIR"/../build-records/*.record 2>/dev/null)" ] \
  && [ "$(grep -c "^SO_SHA256=$CURSO\$" "$TMPD"/*.record 2>/dev/null | grep -c ':[1-9]')" -eq 0 ]; report F2_current_so_has_no_record $?
echo "  F2 current production .so ($CURSO) has no build record anywhere in the repo"

rm -rf "$TMPD"
EXPECTED=41
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "BR_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "BUILD_RECORD_TESTS pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
