#!/bin/bash
# Local deterministic static tests v2 (tracked). Runs on the packaging host
# with NO docker/GPU. Covers every gate reachable before the container
# boundary, plus both shared validators and the receipt generator, asserting
# exact rc + the expected reason line matching EXACTLY ONCE (anchored regex,
# never a loose substring). Fixtures (dummy .so, fixture manifest, hand-built
# receipt, throwaway git repo for the generator) make each gate deterministic.
# The docker/GPU cases live in test_runner_negatives.sh.
# Usage: bash test_static_negatives_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
VALM=$DIR/validate_manifest_sm90.sh
VALR=$DIR/validate_receipt_sm90.sh
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "STATIC_$1_PASS"; PASS=$((PASS+1)); else echo "STATIC_$1_FAIL"; FAIL=$((FAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }
H64() { printf "$1%.0s" $(seq 1 64); }
H40() { printf "$1%.0s" $(seq 1 40); }

TMPD=$(mktemp -d)
MOKF=$TMPD/mok
mkdir -p "$MOKF/mixture-of-kittens/mok" "$MOKF/host-runs" "$MOKF/runs"
ln -s "$DIR" "$MOKF/mixture-of-kittens/benchmarks"
printf 'not a real so\n' > "$MOKF/mixture-of-kittens/mok/_Cfixture.so"
SOSHA=$(sha256sum "$MOKF/mixture-of-kittens/mok/_Cfixture.so" | cut -d' ' -f1)
HSHA=$(sha256sum "$DIR/bench_sm90_fwd.py" | cut -d' ' -f1)
MANF=$TMPD/fixture.manifest
sed -e "s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=$SOSHA/" \
    -e "s/^EXPECTED_HARNESS_SHA256=.*/EXPECTED_HARNESS_SHA256=$HSHA/" \
    "$DIR/manifests/tiny-h20-v1.manifest" > "$MANF"
chmod 444 "$MANF"
MSHA=$(sha256sum "$MANF" | cut -d' ' -f1)
mkreceipt() { # out manifest_sha [build_commit] [repo_digests]
  local OUT=$1 MS=$2 BC=${3:-$(H40 0)} RD=${4:-registry.local/mok@sha256:$(H64 9)}
  { echo "RECEIPT_SCHEMA=1"
    echo "SOURCE_TREE_COMMIT=$(H40 0)"
    echo "HARNESS_COMMIT=$(H40 0)"
    echo "BINARY_BUILD_COMMIT=$BC"
    echo "MANIFEST_FILE=$(basename "$MANF")"
    echo "MANIFEST_SHA256=$MS"
    echo "MANIFEST_GIT_BLOB=$(H40 0)"
    echo "HARNESS_SHA256=$HSHA"
    echo "SO_SHA256=$SOSHA"
    echo "IMAGE_ID=sha256:$(H64 1)"
    echo "IMAGE_REF=fixture-image:latest"
    echo "IMAGE_REPO_DIGESTS=$RD"; } > "$OUT"
  chmod 444 "$OUT"
}
RECF=$TMPD/fixture.receipt
mkreceipt "$RECF" "$MSHA"
RSHA=$(sha256sum "$RECF" | cut -d' ' -f1)
# Deliberately reproduce the remote runtime environment: a VALID expected hash
# is exported for the whole suite. A merely malformed value would be useless
# here - the gate rejects malformed and absent with the same message, so an
# inherited-value bug would still look like a pass. With a valid value
# exported, the 'prior absent' case can only pass if env -u really works.
export EXPECTED_RECEIPT_SHA256="$RSHA"

# ---- P0: shared validators accept the fixture (proves later failures are
# caused by the mutation under test, not by a broken fixture) ----
set +e
OUT=$(bash "$VALM" "$MANF" --harness "$DIR/bench_sm90_fwd.py" --so-dir "$MOKF/mixture-of-kittens/mok" 2>&1); RC=$?
set -u
[ "$RC" -eq 0 ] && has1 "$OUT" "^MANIFEST_VALID:$MSHA\$"; report P0a_manifest_positive $?
echo "  P0a rc=$RC want=0(MANIFEST_VALID)"
set +e
OUT=$(bash "$VALR" "$RECF" --check-mode 2>&1); RC=$?
set -u
[ "$RC" -eq 0 ] && has1 "$OUT" "^RECEIPT_VALID:$RSHA\$"; report P0b_receipt_positive $?
echo "  P0b rc=$RC want=0(RECEIPT_VALID)"

# ---- launcher gates, in gate order ----
lcase() { # name expected_sha mode want_rc reason-ERE -- launcher args...
  local NAME=$1 EXP=$2 MODE=$3 WANT=$4 REASON=$5; shift 5
  [ "$1" = "--" ] && shift
  local O R
  set +e
  if [ "$EXP" = "__UNSET__" ]; then
    # env -u, never a plain invocation: this suite deliberately runs with a
    # VALID EXPECTED_RECEIPT_SHA256 exported (see below), so a plain child
    # would inherit it, skip the gate and start a real run. Clearing only that
    # one variable is not enough either - anything that reconfigures the
    # launcher must be cleared so the case tests what it claims to test.
    O=$(env -u EXPECTED_RECEIPT_SHA256 -u PREFLIGHT_TRIES -u BENCH_TIMEOUT \
          -u BENCH_WAIT_SECS -u CLK_RUN_MIN_MHZ -u LOAD1_DELTA_MAX -u LOAD1_MAX_PRELAUNCH \
          BENCH_TAG="$NAME" BENCH_MODE="$MODE" bash "$DIR/host_launch_sm90.sh" "$@" 2>&1)
  else
    O=$(BENCH_TAG="$NAME" BENCH_MODE="$MODE" EXPECTED_RECEIPT_SHA256="$EXP" \
        bash "$DIR/host_launch_sm90.sh" "$@" 2>&1)
  fi
  R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}
lcase L1_expected_env_missing __UNSET__ canary 14 \
  '^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 env missing or not 64-hex$' -- no-ct "$MOKF" "$MANF" "$RECF"
lcase L2_expected_not_hex "notahash" canary 14 \
  '^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 env missing or not 64-hex$' -- no-ct "$MOKF" "$MANF" "$RECF"
lcase L3_expected_mismatch "$(H64 a)" canary 14 \
  "^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 mismatch \(actual $RSHA expected $(H64 a)\)\$" -- no-ct "$MOKF" "$MANF" "$RECF"
# collusion: manifest AND receipt rewritten self-consistently, both read-only
# and schema-valid; the out-of-band prior must still reject them
sed 's/^tokens_per_rank=.*/tokens_per_rank=513/' "$MANF" > "$TMPD/collude.manifest"
chmod 444 "$TMPD/collude.manifest"
CMSHA=$(sha256sum "$TMPD/collude.manifest" | cut -d' ' -f1)
mkreceipt "$TMPD/collude.receipt" "$CMSHA"
CRSHA=$(sha256sum "$TMPD/collude.receipt" | cut -d' ' -f1)
lcase L4_collusion "$RSHA" canary 14 \
  "^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 mismatch \(actual $CRSHA expected $RSHA\)\$" \
  -- no-ct "$MOKF" "$TMPD/collude.manifest" "$TMPD/collude.receipt"
cp "$RECF" "$TMPD/writable.receipt"; chmod 644 "$TMPD/writable.receipt"
lcase L5_receipt_writable "$(sha256sum "$TMPD/writable.receipt" | cut -d' ' -f1)" canary 14 \
  '^RECEIPT_TRUST_FAIL:write bits set \(644\)$' -- no-ct "$MOKF" "$MANF" "$TMPD/writable.receipt"
grep -v '^SO_SHA256=' "$RECF" > "$TMPD/misskey.receipt"; chmod 444 "$TMPD/misskey.receipt"
lcase L6_receipt_missing_key "$(sha256sum "$TMPD/misskey.receipt" | cut -d' ' -f1)" canary 14 \
  '^RECEIPT_TRUST_FAIL:key SO_SHA256 count=0 \(need exactly 1\)$' -- no-ct "$MOKF" "$MANF" "$TMPD/misskey.receipt"
{ cat "$RECF"; echo "rogue=1"; } > "$TMPD/rogue.receipt"; chmod 444 "$TMPD/rogue.receipt"
lcase L7_receipt_unknown_key "$(sha256sum "$TMPD/rogue.receipt" | cut -d' ' -f1)" canary 14 \
  '^RECEIPT_TRUST_FAIL:unknown key rogue$' -- no-ct "$MOKF" "$MANF" "$TMPD/rogue.receipt"
lcase L8_bad_mode "$RSHA" weird 14 \
  '^MODE_FAIL:BENCH_MODE must be formal or canary \(got weird\)$' -- no-ct "$MOKF" "$MANF" "$RECF"
# formal is refused for ANY receipt - including one that looks complete
# (40-hex resolvable build commit + canonical repo digests) - because a
# commit id cannot prove it built this .so
mkreceipt "$TMPD/unk.receipt" "$MSHA" UNKNOWN
lcase L9_formal_blocked_unknown "$(sha256sum "$TMPD/unk.receipt" | cut -d' ' -f1)" formal 14 \
  '^FORMAL_MODE_FAIL:build-record contract not implemented$' \
  -- no-ct "$MOKF" "$MANF" "$TMPD/unk.receipt"
lcase L10_formal_blocked_complete "$RSHA" formal 14 \
  '^FORMAL_MODE_FAIL:build-record contract not implemented$' -- no-ct "$MOKF" "$MANF" "$RECF"
lcase L11_manifest_missing "$RSHA" canary 12 \
  "^MANIFEST_SCHEMA_FAIL:missing $TMPD/nonexistent\.manifest\$" -- no-ct "$MOKF" "$TMPD/nonexistent.manifest" "$RECF"
sed 's/^tokens_per_rank=.*/tokens_per_rank=513/' "$MANF" > "$TMPD/mut.manifest"; chmod 444 "$TMPD/mut.manifest"
lcase L12_manifest_mutated "$RSHA" canary 14 \
  "^MANIFEST_TRUST_FAIL:manifest sha != receipt \(actual $(sha256sum "$TMPD/mut.manifest" | cut -d' ' -f1) receipt $MSHA\)\$" \
  -- no-ct "$MOKF" "$TMPD/mut.manifest" "$RECF"
cp "$MANF" "$TMPD/wmanifest.manifest"; chmod 644 "$TMPD/wmanifest.manifest"
mkreceipt "$TMPD/wman.receipt" "$(sha256sum "$TMPD/wmanifest.manifest" | cut -d' ' -f1)"
lcase L13_manifest_writable "$(sha256sum "$TMPD/wman.receipt" | cut -d' ' -f1)" canary 14 \
  '^MANIFEST_TRUST_FAIL:write bits set \(644\)$' -- no-ct "$MOKF" "$TMPD/wmanifest.manifest" "$TMPD/wman.receipt"
sed "s|^HARNESS_SHA256=.*|HARNESS_SHA256=$(H64 f)|" "$RECF" > "$TMPD/hmis.receipt"; chmod 444 "$TMPD/hmis.receipt"
lcase L14_receipt_harness_mismatch "$(sha256sum "$TMPD/hmis.receipt" | cut -d' ' -f1)" canary 14 \
  '^RECEIPT_TRUST_FAIL:harness sha receipt != manifest$' -- no-ct "$MOKF" "$MANF" "$TMPD/hmis.receipt"
# all host-side gates pass in canary mode -> the container gate is what fails
lcase L15_container_gate "$RSHA" canary 14 \
  '^IMAGE_TRUST_FAIL:container inspect failed for no-such-container-xyz$' -- no-such-container-xyz "$MOKF" "$MANF" "$RECF"

# ---- manifest validator surface ----
vcase() { # name mutator want_rc reason-ERE [extra validator args...]
  local NAME=$1 MUT=$2 WANT=$3 REASON=$4; shift 4
  local MF=$TMPD/$NAME.manifest O R
  eval "$MUT" > "$MF"
  set +e
  O=$(bash "$VALM" "$MF" "$@" 2>&1); R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}
vcase V1_missing_key   'grep -v "^topk=" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:key topk count=0 \(need exactly 1\)$'
vcase V2_unknown_key   'cat "$MANF"; echo "rogue_key=1"' 12 '^MANIFEST_SCHEMA_FAIL:unknown key rogue_key$'
vcase V3_duplicate_key 'cat "$MANF"; echo "topk=1"' 12 '^MANIFEST_SCHEMA_FAIL:key topk count=2 \(need exactly 1\)$'
vcase V4_bad_hex       'sed "s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=nothex/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:EXPECTED_SO_SHA256 not 64-hex$'
vcase V5_bad_int       'sed "s/^warmup_iters=.*/warmup_iters=0/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:warmup_iters not positive int$'
vcase V6_world_size    'sed "s/^world_size=.*/world_size=8/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:world_size must be 4$'
vcase V7_gpus_dup      'sed "s/^BENCH_GPUS=.*/BENCH_GPUS=0,0,2,3/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:BENCH_GPUS ids not unique$'
vcase V8_gpus_nonnum   'sed "s/^BENCH_GPUS=.*/BENCH_GPUS=0,1,2,x/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:BENCH_GPUS non-numeric id$'
vcase V9_frozen_short  'sed "s/^FROZEN_COMMIT=.*/FROZEN_COMMIT=6df8bb7/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:FROZEN_COMMIT not 40-hex$'
vcase V10_timing_enum  'sed "s|^TIMING_SEMANTICS=.*|TIMING_SEMANTICS=forward.v1|" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:TIMING_SEMANTICS not in allowed set \{build_schedule\+forward\.v2\}$'
vcase V11_topk_gt_exp  'sed "s/^topk=.*/topk=9/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:topk > experts$'
vcase V12_mini_gt_macro 'sed "s/^minibatch=.*/minibatch=8192/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:minibatch > macrobatch$'
vcase V13_macro_multiple 'sed "s/^macrobatch=.*/macrobatch=4097/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:macrobatch not a multiple of minibatch$'
vcase V14_range        'sed "s/^comm_sms=.*/comm_sms=999/" "$MANF"' 12 '^MANIFEST_SCHEMA_FAIL:comm_sms=999 outside \[1,132\]$'
vcase V15_harness_drift "sed \"s/^EXPECTED_HARNESS_SHA256=.*/EXPECTED_HARNESS_SHA256=$(H64 d)/\" \"\$MANF\"" 13 \
  "^HARNESS_DRIFT_FAIL expected=$(H64 d) actual=$HSHA\$" --harness "$DIR/bench_sm90_fwd.py"
vcase V16_so_drift "sed \"s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=$(H64 e)/\" \"\$MANF\"" 13 \
  "^SO_DRIFT_FAIL expected=$(H64 e) actual=$SOSHA\$" --so-dir "$MOKF/mixture-of-kittens/mok"
mkdir -p "$TMPD/twoso"; touch "$TMPD/twoso/_Ca.so" "$TMPD/twoso/_Cb.so"
vcase V17_two_so 'cat "$MANF"' 13 "^SO_DRIFT_FAIL:need exactly one _C\*\.so in $TMPD/twoso\$" --so-dir "$TMPD/twoso"

# ---- receipt validator surface ----
rcase() { # name mutator want_rc reason-ERE
  local NAME=$1 MUT=$2 WANT=$3 REASON=$4
  local RF=$TMPD/$NAME.receipt O R
  eval "$MUT" > "$RF"; chmod 444 "$RF"
  set +e
  O=$(bash "$VALR" "$RF" --check-mode 2>&1); R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}
rcase R1_bad_build   'sed "s|^BINARY_BUILD_COMMIT=.*|BINARY_BUILD_COMMIT=abc123|" "$RECF"' 14 '^RECEIPT_TRUST_FAIL:BINARY_BUILD_COMMIT not 40-hex or UNKNOWN$'
rcase R2_bad_image   'sed "s|^IMAGE_ID=.*|IMAGE_ID=notasha|" "$RECF"' 14 '^RECEIPT_TRUST_FAIL:IMAGE_ID malformed \(want sha256:<64-hex>\)$'
rcase R3_empty_val   'sed "s|^IMAGE_REF=.*|IMAGE_REF=|" "$RECF"' 14 '^RECEIPT_TRUST_FAIL:key IMAGE_REF empty$'
rcase R4_missing_key 'grep -v "^HARNESS_COMMIT=" "$RECF"' 14 '^RECEIPT_TRUST_FAIL:key HARNESS_COMMIT count=0 \(need exactly 1\)$'
rcase R5_dup_key     'cat "$RECF"; echo "SO_SHA256='"$(H64 0)"'"' 14 '^RECEIPT_TRUST_FAIL:key SO_SHA256 count=2 \(need exactly 1\)$'
rcase R6_short_commit 'sed "s|^SOURCE_TREE_COMMIT=.*|SOURCE_TREE_COMMIT=abc1234|" "$RECF"' 14 '^RECEIPT_TRUST_FAIL:SOURCE_TREE_COMMIT not 40-hex$'
rcase R7_bad_schema  'sed "s|^RECEIPT_SCHEMA=1|RECEIPT_SCHEMA=2|" "$RECF"' 14 '^RECEIPT_TRUST_FAIL:bad or missing schema version$'
rcase R8_bad_digests 'sed "s|^IMAGE_REPO_DIGESTS=.*|IMAGE_REPO_DIGESTS=some-free-form-string|" "$RECF"' 14 \
  '^RECEIPT_TRUST_FAIL:IMAGE_REPO_DIGESTS malformed \(want NONE or repo@sha256:<64-hex> list\)$'

# ---- receipt generator (throwaway git fixture repo; never on a deploy target) ----
FIX=$TMPD/fixrepo
mkdir -p "$FIX/benchmarks/manifests"
cp "$VALM" "$VALR" "$DIR/make_deploy_receipt.sh" "$DIR/bench_sm90_fwd.py" "$FIX/benchmarks/"
cp "$MANF" "$FIX/benchmarks/manifests/fix.manifest"
chmod 644 "$FIX/benchmarks/manifests/fix.manifest"
( cd "$FIX" && git init -q && git add -A \
  && git -c user.email=t@example.com -c user.name=t commit -qm init ) >/dev/null 2>&1
mk() { # build_commit outname -> runs generator, sets MKRC/MKOUT
  set +e
  MKOUT=$( cd "$FIX" && IMAGE_ID="${MK_IMAGE_ID:-sha256:$(H64 1)}" IMAGE_REF="fixture-image:latest" \
    IMAGE_REPO_DIGESTS="registry.local/mok@sha256:$(H64 9)" BINARY_BUILD_COMMIT="$1" \
    bash benchmarks/make_deploy_receipt.sh "$FIX" benchmarks/manifests/fix.manifest "$TMPD/gen-$2.receipt" 2>&1 )
  MKRC=$?
  set -u
}
mk UNKNOWN ok
GENSHA=$(sha256sum "$TMPD/gen-ok.receipt" 2>/dev/null | cut -d' ' -f1)
GENMODE=$(stat -c %a "$TMPD/gen-ok.receipt" 2>/dev/null)
[ "$MKRC" -eq 0 ] && has1 "$MKOUT" "^EXPECTED_RECEIPT_SHA256:$GENSHA\$" && [ "$GENMODE" = "444" ]; report MK1_generate_ok $?
echo "  MK1 rc=$MKRC want=0(EXPECTED_RECEIPT_SHA256 printed, mode=$GENMODE)"
touch "$FIX/benchmarks/untracked-leftover.txt"
mk UNKNOWN dirty
rm -f "$FIX/benchmarks/untracked-leftover.txt"
[ "$MKRC" -eq 2 ] && has1 "$MKOUT" '^RECEIPT_FAIL:benchmarks tree not clean vs HEAD \(incl\. untracked\):$'; report MK2_untracked_dirty $?
echo "  MK2 rc=$MKRC want=2(untracked file rejected)"
mk abc123 badfmt
[ "$MKRC" -eq 2 ] && has1 "$MKOUT" '^RECEIPT_FAIL:BINARY_BUILD_COMMIT must be full 40-hex or UNKNOWN$'; report MK3_build_format $?
echo "  MK3 rc=$MKRC want=2(BINARY_BUILD_COMMIT format)"
mk "$(H40 0)" unresolvable
[ "$MKRC" -eq 2 ] && has1 "$MKOUT" '^RECEIPT_FAIL:BINARY_BUILD_COMMIT not resolvable in this repo$'; report MK4_build_unresolvable $?
echo "  MK4 rc=$MKRC want=2(BINARY_BUILD_COMMIT unresolvable)"
# tracked-file modification: the porcelain gate fires first (the deeper
# worktree-bytes-vs-committed-blob check behind it is defense in depth and is
# not reachable from here - do not claim it as covered)
echo "# drift" >> "$FIX/benchmarks/manifests/fix.manifest"
mk UNKNOWN trackeddrift
[ "$MKRC" -eq 2 ] && has1 "$MKOUT" '^RECEIPT_FAIL:benchmarks tree not clean vs HEAD \(incl\. untracked\):$'; report MK5_tracked_drift $?
echo "  MK5 rc=$MKRC want=2(modified tracked manifest rejected at porcelain gate)"
git -C "$FIX" checkout -- benchmarks/manifests/fix.manifest 2>/dev/null
# publish safety: a receipt that fails its own validation must not be
# published and must not clobber the existing good one, and must leave no
# temp behind (fault injected through a malformed IMAGE_ID)
PRESHA=$(sha256sum "$TMPD/gen-ok.receipt" | cut -d' ' -f1)
MK_IMAGE_ID=notasha
mk UNKNOWN ok
unset MK_IMAGE_ID
POSTSHA=$(sha256sum "$TMPD/gen-ok.receipt" | cut -d' ' -f1)
NLEFT=$(ls "$TMPD"/.receipt.* 2>/dev/null | wc -l)
[ "$MKRC" -eq 2 ] && has1 "$MKOUT" '^RECEIPT_FAIL:generated receipt failed self-validation \(not published\)$' \
  && [ "$PRESHA" = "$POSTSHA" ] && [ "$NLEFT" -eq 0 ]; report MK6_publish_atomicity $?
echo "  MK6 rc=$MKRC want=2(invalid receipt not published; OUT unchanged=$([ "$PRESHA" = "$POSTSHA" ] && echo yes || echo no); temps left=$NLEFT)"

rm -rf "$TMPD"
EXPECTED=48
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "STATIC_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "STATIC_NEGATIVES pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
