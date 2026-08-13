#!/bin/bash
# Negative suite v6 (tracked; HOST side). Framework rules:
#   - before EVERY runner-level case: poll the build lock until FREE
#   - unique SUITE_ID/CASE_ID; each case examines only its own exact artifacts
#   - every case asserts: exact rc + the expected reason line matching
#     EXACTLY ONCE (anchored regex, not a loose substring) + zero JSON
#     artifacts for its own ids
#   - trust is anchored on the out-of-band EXPECTED_RECEIPT_SHA256; cases that
#     target a gate BEHIND the expected-hash gate pass that fixture's own sha
#     as the prior (a genuine-but-unsuitable receipt), so the gate under test
#     is the one that actually fires
#   - schema mutations go through the validate-only surfaces; the production
#     launcher has NO untrusted bypass
#   - mutation fixtures live in TMPD, removed only AFTER the last case
# Usage: EXPECTED_RECEIPT_SHA256=... \
#          bash test_runner_negatives.sh <container> <host_mok_dir> <receipt>
set -uo pipefail
CT=${1:?}; MOKDIR=${2:?}; RECEIPT=${3:?}
DIR=$(cd "$(dirname "$0")" && pwd)
MANI=$DIR/manifests/tiny-h20-v1.manifest
VALM=$DIR/validate_manifest_sm90.sh
VALR=$DIR/validate_receipt_sm90.sh
EXPR_SHA=${EXPECTED_RECEIPT_SHA256:?EXPECTED_RECEIPT_SHA256 required (out-of-band prior)}
SUITE=$(date -u +%H%M%S)-$RANDOM
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "NEGATIVE_$1_PASS"; PASS=$((PASS+1)); else echo "NEGATIVE_$1_FAIL"; FAIL=$((FAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }
line1() { [ "$(grep -cE "$2" "$1" 2>/dev/null || true)" -eq 1 ]; }
wait_lock_free() {
  for i in $(seq 1 36); do
    docker exec "$CT" sh -c 'flock -n /mok/build.lock true' 2>/dev/null && return 0
    sleep 5
  done
  echo "SUITE_SETUP_FAIL:lock never freed"; exit 3
}
no_json() { [ "$(ls "$MOKDIR/runs/$1-"*.json 2>/dev/null | wc -l)" -eq 0 ]; }
REALSHA=$(sha256sum "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py" | cut -d' ' -f1)
SOARR=("$MOKDIR"/mixture-of-kittens/mok/_C*.so)
{ [ "${#SOARR[@]}" -eq 1 ] && [ -f "${SOARR[0]}" ]; } || { echo "SUITE_SETUP_FAIL:need exactly one .so"; exit 3; }
REALSO=$(sha256sum "${SOARR[0]}" | cut -d' ' -f1)
REALMSHA=$(sha256sum "$MANI" | cut -d' ' -f1)
RSHA_REAL=$(sha256sum "$RECEIPT" | cut -d' ' -f1)
[ "$RSHA_REAL" = "$EXPR_SHA" ] || { echo "SUITE_SETUP_FAIL:given receipt does not match EXPECTED_RECEIPT_SHA256"; exit 3; }
TMPD=$(mktemp -d)
H64() { printf "$1%.0s" $(seq 1 64); }

runner_case() { # CID harness_sha so_sha extra_env... -> sets RC and L
  local CID=$1 SHAX=$2 SOX=$3; shift 3
  docker exec "$@" -e BENCH_TAG="$CID" -e RUN_ID="$CID" \
    -e EXPECTED_HARNESS_SHA256="$SHAX" -e EXPECTED_SO_SHA256="$SOX" \
    -e MANIFEST_SHA256="$REALMSHA" -e RECEIPT_SHA256="$RSHA_REAL" -e BENCH_MODE=canary \
    "$CT" bash /mok/mixture-of-kittens/benchmarks/run_bench_sm90.sh >/dev/null 2>&1
  RC=$?
  L=$MOKDIR/runs/$CID-$CID.log
}
lcase() { # name expected_receipt_sha want_rc reason-ERE -- launcher args...
  local NAME=$1 EXP=$2 WANT=$3 REASON=$4; shift 4
  [ "$1" = "--" ] && shift
  local CID=$NAME-$SUITE O R
  set +e
  O=$(BENCH_TAG="$CID" EXPECTED_RECEIPT_SHA256="$EXP" BENCH_MODE="${LMODE:-canary}" \
      bash "$DIR/host_launch_sm90.sh" "$@" 2>&1)
  R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON" && no_json "$CID"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
  unset LMODE
}

# ---- runner-level gates (container, GPU) ----
wait_lock_free
CID=n1-$SUITE; runner_case "$CID" deadbeef "$REALSO"
[ "$RC" -eq 8 ] && line1 "$L" '^HASH_GATE_FAIL expected=deadbeef actual='"$REALSHA"'$' && no_json "$CID"; report 1_wrong_hash $?
echo "  n1 rc=$RC want=8(HASH_GATE_FAIL exact)"

wait_lock_free
docker exec -d "$CT" bash -c "exec 9>/mok/build.lock; flock 9; sleep 20"
HELD=0
for i in $(seq 1 10); do
  docker exec "$CT" sh -c 'flock -n /mok/build.lock true' 2>/dev/null || { HELD=1; break; }
  sleep 1
done
[ "$HELD" -eq 1 ] || { echo "SUITE_SETUP_FAIL:n2 holder never took lock"; exit 3; }
CID=n2-$SUITE; runner_case "$CID" deadbeef "$REALSO"
[ "$RC" -eq 9 ] && line1 "$L" '^LOCK_BUSY ' && no_json "$CID"; report 2_lock_busy $?
echo "  n2 rc=$RC want=9(LOCK_BUSY exact)"
wait_lock_free

wait_lock_free
docker exec -d "$CT" bash -c "python3 -c 'import torch,time;x=torch.ones(1024,1024,device=\"cuda:0\");time.sleep(45)'"
SEEN=0
for i in $(seq 1 15); do
  N=$(docker exec "$CT" sh -c "nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'" 2>/dev/null | head -1)
  [[ "${N:-0}" =~ ^[0-9]+$ ]] && [ "${N:-0}" -ge 1 ] && { SEEN=1; break; }
  sleep 3
done
[ "$SEEN" -eq 1 ] || { echo "SUITE_SETUP_FAIL:n3 holder never appeared on GPU"; exit 3; }
CID=n3-$SUITE; runner_case "$CID" "$REALSHA" "$REALSO" -e PREFLIGHT_TRIES=2
[ "$RC" -eq 7 ] && line1 "$L" '^PREFLIGHT_FAIL:target GPUs still occupied after 2 tries$' && no_json "$CID"; report 3_occupied $?
echo "  n3 rc=$RC want=7(PREFLIGHT_FAIL exact)"
for i in $(seq 1 20); do
  N=$(docker exec "$CT" sh -c "nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'" 2>/dev/null | head -1)
  [[ "${N:-0}" =~ ^[0-9]+$ ]] && [ "${N:-0}" -eq 0 ] && break
  sleep 3
done
[[ "${N:-1}" =~ ^[0-9]+$ ]] && [ "${N:-1}" -eq 0 ] || { echo "SUITE_SETUP_FAIL:n3 GPU not cleared after holder (N=$N)"; exit 3; }

wait_lock_free
CID=n4-$SUITE; runner_case "$CID" "$REALSHA" deadbeefso
[ "$RC" -eq 10 ] && line1 "$L" '^SO_GATE_FAIL expected=deadbeefso actual='"$REALSO"'$' \
  && line1 "$L" '^HASH_GATE_PASS$' && line1 "$L" '^PREFLIGHT_PASS$' && no_json "$CID"; report 4_wrong_so $?
echo "  n4 rc=$RC want=10(SO_GATE_FAIL after hash+preflight pass)"

# ---- expected-receipt gate (the trust anchor) ----
CID=n5-$SUITE
set +e
O=$(BENCH_TAG="$CID" bash "$DIR/host_launch_sm90.sh" "$CT" "$MOKDIR" "$MANI" "$RECEIPT" 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 env missing or not 64-hex$' && no_json "$CID"; report 5_expected_env_missing $?
echo "  n5 rc=$R want=14(EXPECTED_RECEIPT_SHA256 env missing)"

lcase 6_expected_not_hex "notahash" 14 '^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 env missing or not 64-hex$' \
  -- "$CT" "$MOKDIR" "$MANI" "$RECEIPT"
lcase 7_expected_mismatch "$(H64 a)" 14 '^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 mismatch \(actual '"$RSHA_REAL"' expected '"$(H64 a)"'\)$' \
  -- "$CT" "$MOKDIR" "$MANI" "$RECEIPT"

# collusion: manifest AND receipt rewritten self-consistently (receipt's
# MANIFEST_SHA256 updated to match the mutated manifest, read-only, schema
# valid) - must STILL be rejected by the out-of-band expected-hash gate
sed 's/^tokens_per_rank=.*/tokens_per_rank=513/' "$MANI" > "$TMPD/collude.manifest"
CMSHA=$(sha256sum "$TMPD/collude.manifest" | cut -d' ' -f1)
sed "s|^MANIFEST_SHA256=.*|MANIFEST_SHA256=$CMSHA|" "$RECEIPT" > "$TMPD/collude.receipt"
chmod 444 "$TMPD/collude.manifest" "$TMPD/collude.receipt"
CRSHA=$(sha256sum "$TMPD/collude.receipt" | cut -d' ' -f1)
lcase 8_collusion "$EXPR_SHA" 14 '^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 mismatch \(actual '"$CRSHA"' expected '"$EXPR_SHA"'\)$' \
  -- "$CT" "$MOKDIR" "$TMPD/collude.manifest" "$TMPD/collude.receipt"

# ---- receipt well-formedness gates (prior given for each fixture) ----
cp "$RECEIPT" "$TMPD/writable.receipt"; chmod 644 "$TMPD/writable.receipt"
lcase 9_receipt_writable "$(sha256sum "$TMPD/writable.receipt" | cut -d' ' -f1)" 14 \
  '^RECEIPT_TRUST_FAIL:write bits set \(644\)$' -- "$CT" "$MOKDIR" "$MANI" "$TMPD/writable.receipt"
grep -v '^SO_SHA256=' "$RECEIPT" > "$TMPD/misskey.receipt"; chmod 444 "$TMPD/misskey.receipt"
lcase 10_receipt_missing_key "$(sha256sum "$TMPD/misskey.receipt" | cut -d' ' -f1)" 14 \
  '^RECEIPT_TRUST_FAIL:key SO_SHA256 count=0 \(need exactly 1\)$' -- "$CT" "$MOKDIR" "$MANI" "$TMPD/misskey.receipt"
{ cat "$RECEIPT"; echo "rogue=1"; } > "$TMPD/rogue.receipt"; chmod 444 "$TMPD/rogue.receipt"
lcase 11_receipt_unknown_key "$(sha256sum "$TMPD/rogue.receipt" | cut -d' ' -f1)" 14 \
  '^RECEIPT_TRUST_FAIL:unknown key rogue$' -- "$CT" "$MOKDIR" "$MANI" "$TMPD/rogue.receipt"
sed "s|^MANIFEST_SHA256=.*|MANIFEST_SHA256=$(H64 b)|" "$RECEIPT" > "$TMPD/badbind.receipt"; chmod 444 "$TMPD/badbind.receipt"
lcase 12_receipt_binding "$(sha256sum "$TMPD/badbind.receipt" | cut -d' ' -f1)" 14 \
  '^MANIFEST_TRUST_FAIL:manifest sha != receipt \(actual '"$REALMSHA"' receipt '"$(H64 b)"'\)$' \
  -- "$CT" "$MOKDIR" "$MANI" "$TMPD/badbind.receipt"
sed "s|^IMAGE_ID=.*|IMAGE_ID=sha256:$(H64 c)|" "$RECEIPT" > "$TMPD/badimg.receipt"; chmod 444 "$TMPD/badimg.receipt"
lcase 13_image_mismatch "$(sha256sum "$TMPD/badimg.receipt" | cut -d' ' -f1)" 14 \
  '^IMAGE_TRUST_FAIL:image id live ' -- "$CT" "$MOKDIR" "$MANI" "$TMPD/badimg.receipt"

# ---- formal-mode gates: a genuine receipt without build/registry provenance
# must be refused in formal mode (canary is allowed, labeled INVALID_FOR_FORMAL)
sed "s|^BINARY_BUILD_COMMIT=.*|BINARY_BUILD_COMMIT=UNKNOWN|" "$RECEIPT" > "$TMPD/unk.receipt"; chmod 444 "$TMPD/unk.receipt"
LMODE=formal lcase 14_formal_unknown_build "$(sha256sum "$TMPD/unk.receipt" | cut -d' ' -f1)" 14 \
  '^FORMAL_MODE_FAIL:BINARY_BUILD_COMMIT UNKNOWN \(no build record; formal forbidden\)$' \
  -- "$CT" "$MOKDIR" "$MANI" "$TMPD/unk.receipt"
sed "s|^IMAGE_REPO_DIGESTS=.*|IMAGE_REPO_DIGESTS=NONE|; s|^BINARY_BUILD_COMMIT=.*|BINARY_BUILD_COMMIT=$(printf '0%.0s' $(seq 1 40))|" \
  "$RECEIPT" > "$TMPD/nodig.receipt"; chmod 444 "$TMPD/nodig.receipt"
LMODE=formal lcase 15_formal_local_image "$(sha256sum "$TMPD/nodig.receipt" | cut -d' ' -f1)" 14 \
  '^FORMAL_MODE_FAIL:IMAGE_REPO_DIGESTS NONE \(local-only image; formal forbidden\)$' \
  -- "$CT" "$MOKDIR" "$MANI" "$TMPD/nodig.receipt"

# ---- host-side manifest / sidecar gates ----
lcase 16_manifest_missing "$EXPR_SHA" 12 '^MANIFEST_SCHEMA_FAIL:missing ' \
  -- "$CT" "$MOKDIR" "$TMPD/nonexistent.manifest" "$RECEIPT"
CID=17_sidecar_unwritable-$SUITE
RO=$MOKDIR/negro-$SUITE
mkdir -p "$RO/host-runs"
ln -s "$MOKDIR/mixture-of-kittens" "$RO/mixture-of-kittens"
chmod 555 "$RO/host-runs"
set +e
O=$(BENCH_TAG="$CID" EXPECTED_RECEIPT_SHA256="$EXPR_SHA" bash "$DIR/host_launch_sm90.sh" "$CT" "$RO" "$MANI" "$RECEIPT" 2>&1)
R=$?
set -u
chmod 755 "$RO/host-runs" 2>/dev/null
NART=$(ls "$RO/runs/" 2>/dev/null | wc -l)
rm -rf "$RO" 2>/dev/null
[ "$R" -eq 4 ] && has1 "$O" '^LAUNCH_VERIFY_FAIL:sidecar not writable at ' && [ "$NART" -eq 0 ]; report 17_sidecar_unwritable $?
echo "  17 rc=$R want=4(sidecar not writable, artifacts=$NART)"

# ---- validate-only surfaces (no launcher, no docker) ----
vcase() { # name mutator want_rc reason-ERE [validator extra args...]
  local NAME=$1 MUT=$2 WANT=$3 REASON=$4; shift 4
  local MF=$TMPD/$NAME.manifest O R
  eval "$MUT" > "$MF"
  set +e
  O=$(bash "$VALM" "$MF" "$@" 2>&1); R=$?
  set -u
  [ "$R" -eq "$WANT" ] && has1 "$O" "$REASON"; report "$NAME" $?
  echo "  $NAME rc=$R want=$WANT($REASON)"
}
vcase 18_missing_key   'grep -v "^topk=" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:key topk count=0 \(need exactly 1\)$'
vcase 19_unknown_key   'cat "$MANI"; echo "rogue_key=1"' 12 '^MANIFEST_SCHEMA_FAIL:unknown key rogue_key$'
vcase 20_duplicate_key 'cat "$MANI"; echo "topk=1"' 12 '^MANIFEST_SCHEMA_FAIL:key topk count=2 \(need exactly 1\)$'
vcase 21_bad_hex       'sed "s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=nothex/" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:EXPECTED_SO_SHA256 not 64-hex$'
vcase 22_frozen_short  'sed "s/^FROZEN_COMMIT=.*/FROZEN_COMMIT=6df8bb7/" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:FROZEN_COMMIT not 40-hex$'
vcase 23_timing_enum   'sed "s|^TIMING_SEMANTICS=.*|TIMING_SEMANTICS=forward.v1|" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:TIMING_SEMANTICS not in allowed set'
vcase 24_world_size    'sed "s/^world_size=.*/world_size=8/" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:world_size must be 4$'
vcase 25_gpus_dup      'sed "s/^BENCH_GPUS=.*/BENCH_GPUS=0,0,2,3/" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:BENCH_GPUS ids not unique$'
vcase 26_topk_gt_exp   'sed "s/^topk=.*/topk=9/" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:topk > experts$'
vcase 27_macro_multiple 'sed "s/^macrobatch=.*/macrobatch=4097/" "$MANI"' 12 '^MANIFEST_SCHEMA_FAIL:macrobatch not a multiple of minibatch$'
vcase 28_harness_drift "sed \"s/^EXPECTED_HARNESS_SHA256=.*/EXPECTED_HARNESS_SHA256=$(H64 d)/\" \"\$MANI\"" 13 \
  '^HARNESS_DRIFT_FAIL expected='"$(H64 d)"' actual='"$REALSHA"'$' --harness "$MOKDIR/mixture-of-kittens/benchmarks/bench_sm90_fwd.py"
vcase 29_so_drift "sed \"s/^EXPECTED_SO_SHA256=.*/EXPECTED_SO_SHA256=$(H64 e)/\" \"\$MANI\"" 13 \
  '^SO_DRIFT_FAIL expected='"$(H64 e)"' actual='"$REALSO"'$' --so-dir "$MOKDIR/mixture-of-kittens/mok"
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
rcase 30_receipt_bad_build 'sed "s|^BINARY_BUILD_COMMIT=.*|BINARY_BUILD_COMMIT=abc123|" "$RECEIPT"' 14 \
  '^RECEIPT_TRUST_FAIL:BINARY_BUILD_COMMIT not 40-hex or UNKNOWN$'
rcase 31_receipt_bad_image 'sed "s|^IMAGE_ID=.*|IMAGE_ID=notasha|" "$RECEIPT"' 14 \
  '^RECEIPT_TRUST_FAIL:IMAGE_ID malformed \(want sha256:<64-hex>\)$'
rcase 32_receipt_empty_val 'sed "s|^IMAGE_REF=.*|IMAGE_REF=|" "$RECEIPT"' 14 \
  '^RECEIPT_TRUST_FAIL:key IMAGE_REF empty$'

# cleanup only AFTER the last case that uses TMPD fixtures
rm -rf "$TMPD"

EXPECTED_CASES=32
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED_CASES" ] || { echo "SUITE_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED_CASES"; FAIL=$((FAIL+1)); }
echo "NEGATIVES suite=$SUITE pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
