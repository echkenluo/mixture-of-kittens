#!/bin/bash
# Packaging / validation side adversarial suite (tracked, no GPU, no docker,
# no torch, no real build).
#
# Scope: make_deploy_receipt.sh, validate_receipt_sm90.sh,
# validate_manifest_sm90.sh, check_formal_binding_sm90.sh, host_launch_sm90.sh.
#
# Every case that claims a fix runs the SAME attack against 25cc1f5 first and
# shows it succeeding there. Cases marked "regression guard" pin behaviour that
# already existed and are labelled as such - they are not credited as fixes.
#
# WHAT THIS SUITE CANNOT COVER. The hardened launcher pins PATH, so the mock
# docker/nvidia-smi that drive a full launch no longer resolve: every new-side
# launcher case here therefore ends at or before the first docker call. The
# post-docker gates (RepoDigests handling, process shape, telemetry) are pinned
# as mutation-checked SOURCE guards in test_source_guards_local.sh instead. The
# deployment-tooling boundary itself stays UNVERIFIED: a hostile /usr/bin is
# outside what any of this can detect.
#
# Usage: bash test_packaging_audit_local.sh
set -uo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/.." && pwd)
PASS=0; FAIL=0
report() { if [ "$2" -eq 0 ]; then echo "PKG_$1_PASS"; PASS=$((PASS+1)); else echo "PKG_$1_FAIL"; FAIL=$((FAIL+1)); fi }
has1() { [ "$(printf '%s\n' "$1" | grep -cE "$2" || true)" -eq 1 ]; }
H64() { printf "$1%.0s" $(seq 1 64); }
H40() { printf "$1%.0s" $(seq 1 40); }
GIT="git -c user.email=t@t -c user.name=t -c commit.gpgsign=false -c init.defaultBranch=main"
OLDREF=25cc1f5

TMPD=$(mktemp -d)
trap '[ -n "${PKG_KEEP_TMPD:-}" ] && { echo "PKG_TMPD_KEPT:$TMPD"; exit 0; }; rm -rf "$TMPD"' EXIT
OLD=$TMPD/old
mkdir -p "$OLD"
git -C "$REPO" archive "$OLDREF" benchmarks 2>/dev/null | tar x -C "$OLD" || true
OLDB=$OLD/benchmarks
[ -f "$OLDB/host_launch_sm90.sh" ] || { echo "PKG_SETUP_FAIL:cannot extract $OLDREF"; exit 1; }

# ---------------------------------------------------------------- fixtures --
# a launcher harness: mok tree, manifest, receipt, mock docker/nvidia-smi
mkharness() { # dir
  local W=$1 M=$1/mok UU="GPU-aaaaaaaa-0000-0000-0000-00000000000"
  mkdir -p "$M/mixture-of-kittens/mok" "$M/runs" "$M/host-runs" "$W/bin"
  ln -sfn "$DIR" "$M/mixture-of-kittens/benchmarks"
  printf 'not a real so\n' > "$M/mixture-of-kittens/mok/_Cfixture.so"
  local SOSHA HSHA MSHA IMGID
  SOSHA=$(sha256sum "$M/mixture-of-kittens/mok/_Cfixture.so" | cut -d' ' -f1)
  HSHA=$(sha256sum "$DIR/bench_sm90_fwd.py" | cut -d' ' -f1)
  sed -e "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$SOSHA|" \
      -e "s|^EXPECTED_HARNESS_SHA256=.*|EXPECTED_HARNESS_SHA256=$HSHA|" \
      "$DIR/manifests/tiny-h20-v1.manifest" > "$W/man"
  chmod 444 "$W/man"
  MSHA=$(sha256sum "$W/man" | cut -d' ' -f1); IMGID="sha256:$(H64 1)"
  { echo "RECEIPT_SCHEMA=1"; echo "SOURCE_TREE_COMMIT=$(H40 a)"; echo "HARNESS_COMMIT=$(H40 a)"
    echo "BINARY_BUILD_COMMIT=UNKNOWN"; echo "MANIFEST_FILE=man"; echo "MANIFEST_SHA256=$MSHA"
    echo "MANIFEST_GIT_BLOB=$(H40 c)"; echo "HARNESS_SHA256=$HSHA"; echo "SO_SHA256=$SOSHA"
    echo "IMAGE_ID=$IMGID"; echo "IMAGE_REF=fixture-image:latest"; echo "IMAGE_REPO_DIGESTS=NONE"; } > "$W/receipt"
  chmod 444 "$W/receipt"
  cat > "$W/bin/docker" << EOS
#!/bin/bash
case "\$*" in
  *"{{.Image}}"*)        echo "$IMGID"; exit 0 ;;
  *"{{.Config.Image}}"*) echo "fixture-image:latest"; exit 0 ;;
  "image inspect"*)      echo ""; exit 0 ;;
  *"nvidia-smi --query-gpu=index,uuid"*) for i in 0 1 2 3; do echo "\$i,${UU}\$i"; done; exit 0 ;;
  *) echo "docker: unmocked call" >&2; exit 1 ;;
esac
EOS
  cat > "$W/bin/nvidia-smi" << EOS
#!/bin/bash
case "\$*" in
  *"--query-gpu=index,uuid"*) for i in 0 1 2 3; do echo "\$i, ${UU}\$i"; done; exit 0 ;;
  *"--query-compute-apps"*)   exit 0 ;;
  *"clocks.sm"*)              for i in 0 1 2 3; do echo "${UU}\$i, 1500 MHz, 100 W"; done; exit 0 ;;
  *) exit 1 ;;
esac
EOS
  chmod +x "$W/bin/docker" "$W/bin/nvidia-smi"
}
clean_runs() { rm -rf "$1/mok/host-runs" "$1/mok"/*.host "$1/mok"/*.manifest "$1/mok"/*.receipt 2>/dev/null
  mkdir -p "$1/mok/host-runs"; }

# a git repo the receipt generator can package from
PKGREPO=$TMPD/pkgrepo
mkdir -p "$PKGREPO/benchmarks/manifests" "$PKGREPO/mok"
cp "$DIR"/*.sh "$PKGREPO/benchmarks/"
cp "$DIR"/bench_sm90_fwd.py "$DIR"/bench_deepep_fwd.py "$PKGREPO/benchmarks/"
cp "$DIR"/manifests/tiny-h20-v1.manifest "$PKGREPO/benchmarks/manifests/"
( cd "$PKGREPO" && $GIT init -q . && $GIT add -A && $GIT commit -qm pkg ) >/dev/null 2>&1

# =============================================================== validators ==
# PKG01: a receipt key is content; used as a grep pattern it impersonates an
# allowed name. Same defect class the build-record validator had.
mkreceipt() { # outfile extra-line
  { echo "RECEIPT_SCHEMA=1"; echo "SOURCE_TREE_COMMIT=$(H40 a)"; echo "HARNESS_COMMIT=$(H40 a)"
    echo "BINARY_BUILD_COMMIT=UNKNOWN"; echo "MANIFEST_FILE=x.manifest"; echo "MANIFEST_SHA256=$(H64 b)"
    echo "MANIFEST_GIT_BLOB=$(H40 c)"; echo "HARNESS_SHA256=$(H64 d)"; echo "SO_SHA256=$(H64 e)"
    echo "IMAGE_ID=sha256:$(H64 1)"; echo "IMAGE_REF=img:1"; echo "IMAGE_REPO_DIGESTS=NONE"
    [ -n "$2" ] && echo "$2"; } > "$1"
  chmod 444 "$1"
}
mkreceipt "$TMPD/r.smuggled" 'SO_SHA25.=smuggled'
set +e
O=$(bash "$OLDB/validate_receipt_sm90.sh" "$TMPD/r.smuggled" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 0 ]; report 01a_old_receipt_validator_took_smuggled_key $?
echo "  01a old rc=$R (SO_SHA25. matched SO_SHA256 in the allow-list)"
set +e
O=$(bash "$DIR/validate_receipt_sm90.sh" "$TMPD/r.smuggled" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^RECEIPT_TRUST_FAIL:unknown key SO_SHA25\.$'; report 01b_receipt_smuggled_key_refused $?
echo "  01b rc=$R want=14(keys compared as strings)"

# PKG02: same class in the manifest validator
cp "$DIR/manifests/tiny-h20-v1.manifest" "$TMPD/m.smuggled"; chmod 644 "$TMPD/m.smuggled"
echo 'hidde.=smuggled' >> "$TMPD/m.smuggled"
set +e
O=$(bash "$OLDB/validate_manifest_sm90.sh" "$TMPD/m.smuggled" 2>&1); R=$?
set -u
[ "$R" -eq 0 ]; report 02a_old_manifest_validator_took_smuggled_key $?
echo "  02a old rc=$R (hidde. matched hidden)"
set +e
O=$(bash "$DIR/validate_manifest_sm90.sh" "$TMPD/m.smuggled" 2>&1); R=$?
set -u
[ "$R" -eq 12 ] && has1 "$O" '^MANIFEST_SCHEMA_FAIL:unknown key hidde\.$'; report 02b_manifest_smuggled_key_refused $?
echo "  02b rc=$R want=12"

# PKG13: one packaged tree has one HEAD; letting the two commits differ left a
# field that looks like a binding and never was
mkreceipt "$TMPD/r.twoheads" ""
chmod 644 "$TMPD/r.twoheads"
sed -i "s|^HARNESS_COMMIT=.*|HARNESS_COMMIT=$(H40 b)|" "$TMPD/r.twoheads"; chmod 444 "$TMPD/r.twoheads"
set +e
O=$(bash "$OLDB/validate_receipt_sm90.sh" "$TMPD/r.twoheads" --check-mode 2>&1); R=$?
OO=$?; set -u
[ "$R" -eq 0 ]; report 13a_old_allowed_two_different_heads $?
echo "  13a old rc=$R (SOURCE_TREE_COMMIT and HARNESS_COMMIT could disagree)"
set +e
O=$(bash "$DIR/validate_receipt_sm90.sh" "$TMPD/r.twoheads" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^RECEIPT_TRUST_FAIL:SOURCE_TREE_COMMIT != HARNESS_COMMIT \(one packaged tree has one HEAD\)$'
report 13b_two_heads_refused $?
echo "  13b rc=$R want=14"

# PKG14: MANIFEST_FILE is a bare filename by generator contract
mkreceipt "$TMPD/r.manpath" ""
chmod 644 "$TMPD/r.manpath"
sed -i "s|^MANIFEST_FILE=.*|MANIFEST_FILE=../../etc/x.manifest|" "$TMPD/r.manpath"; chmod 444 "$TMPD/r.manpath"
set +e
O=$(bash "$OLDB/validate_receipt_sm90.sh" "$TMPD/r.manpath" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 0 ]; report 14a_old_allowed_path_in_manifest_file $?
echo "  14a old rc=$R (a path could stand in for the packaged filename)"
set +e
O=$(bash "$DIR/validate_receipt_sm90.sh" "$TMPD/r.manpath" --check-mode 2>&1); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^RECEIPT_TRUST_FAIL:MANIFEST_FILE must be a bare filename, got \.\./\.\./etc/x\.manifest$'
report 14b_path_in_manifest_file_refused $?
echo "  14b rc=$R want=14"

# ============================================================ formal checker ==
# PKG03: the manifest link. The header claimed the manifest was bound; only
# EXPECTED_SO_SHA256 was compared, so any manifest built against the same
# binary - other shape, other iteration count - was accepted.
FB=$TMPD/fb
mkdir -p "$FB/so"
BRTMP=$TMPD/brsuite
mkdir -p "$BRTMP"
# a real production build record, produced by the build-record fixture path
PKG_REC=""
if BR_KEEP_TMPD=1 bash "$DIR/test_build_record_local.sh" > "$BRTMP/log" 2>&1; then
  BRT=$(grep '^BR_TMPD_KEPT:' "$BRTMP/log" | cut -d: -f2)
  [ -f "$BRT/rec.good" ] && PKG_REC=$BRT/rec.good
fi
if [ -n "$PKG_REC" ]; then
  cp "$PKG_REC" "$FB/record"; chmod 444 "$FB/record"
  RSHA=$(sha256sum "$FB/record" | cut -d' ' -f1)
  SOSHA=$(grep '^SO_SHA256=' "$FB/record" | cut -d= -f2)
  SOBN=$(grep '^SO_BASENAME=' "$FB/record" | cut -d= -f2)
  SRC=$(grep '^SOURCE_COMMIT=' "$FB/record" | cut -d= -f2)
  # several fixture artifact dirs hold a file with this name; take the one whose
  # bytes are the ones the record actually describes
  while IFS= read -r F; do
    [ "$(sha256sum "$F" | cut -d' ' -f1)" = "$SOSHA" ] && { cp "$F" "$FB/so/$SOBN"; break; }
  done < <(find "$BRT" -name "$SOBN" -type f 2>/dev/null)
  sed "s|^EXPECTED_SO_SHA256=.*|EXPECTED_SO_SHA256=$SOSHA|" "$DIR/manifests/tiny-h20-v1.manifest" > "$FB/manA"
  sed "s|^tokens_per_rank=.*|tokens_per_rank=999|" "$FB/manA" > "$FB/manB"
  chmod 444 "$FB/manA" "$FB/manB"
  MANASHA=$(sha256sum "$FB/manA" | cut -d' ' -f1)
  HSHA=$(grep '^EXPECTED_HARNESS_SHA256=' "$FB/manA" | cut -d= -f2)
  { echo "RECEIPT_SCHEMA=2"; echo "SOURCE_TREE_COMMIT=$(H40 a)"; echo "HARNESS_COMMIT=$(H40 a)"
    echo "BINARY_BUILD_COMMIT=$SRC"; echo "MANIFEST_FILE=manA"; echo "MANIFEST_SHA256=$MANASHA"
    echo "MANIFEST_GIT_BLOB=$(H40 c)"; echo "HARNESS_SHA256=$HSHA"; echo "SO_SHA256=$SOSHA"
    echo "IMAGE_ID=sha256:$(H64 1)"; echo "IMAGE_REF=img:1"
    echo "IMAGE_REPO_DIGESTS=registry.local/mok@sha256:$(H64 9)"
    echo "BUILD_RECORD_SHA256=$RSHA"; } > "$FB/receipt"
  chmod 444 "$FB/receipt"
  RCSHA=$(sha256sum "$FB/receipt" | cut -d' ' -f1)
  set +e
  O=$(EXPECTED_RECEIPT_SHA256=$RCSHA LOCAL_BINDING_ONLY=1 bash "$OLDB/check_formal_binding_sm90.sh" \
        "$FB/receipt" "$FB/record" "$FB/manB" "$FB/so" 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ] && has1 "$O" '^LOCAL_BINDING_PASS:'; RA=$?
  [ "$RA" -eq 0 ] || { echo "--- 03a old checker output ---"; printf '%s\n' "$O" | tail -3; }
  report 03a_old_passed_a_foreign_manifest "$RA"
  echo "  03a old rc=$R passed with a manifest the receipt does not describe"
  set +e
  O=$(EXPECTED_RECEIPT_SHA256=$RCSHA LOCAL_BINDING_ONLY=1 bash "$DIR/check_formal_binding_sm90.sh" \
        "$FB/receipt" "$FB/record" "$FB/manB" "$FB/so" 2>&1); R=$?
  set -u
  [ "$R" -eq 18 ] && has1 "$O" '^FORMAL_BINDING_FAIL:manifest sha [0-9a-f]{64} != receipt MANIFEST_SHA256 [0-9a-f]{64}; this is not the manifest the receipt describes$'
  report 03b_foreign_manifest_refused $?
  echo "  03b rc=$R want=18"
  set +e
  O=$(EXPECTED_RECEIPT_SHA256=$RCSHA LOCAL_BINDING_ONLY=1 bash "$DIR/check_formal_binding_sm90.sh" \
        "$FB/receipt" "$FB/record" "$FB/manA" "$FB/so" 2>&1); R=$?
  set -u
  [ "$R" -eq 0 ] && has1 "$O" '^LOCAL_BINDING_PASS:[0-9a-f]{64}$'; RC=$?
  [ "$RC" -eq 0 ] || { echo "--- 03c new checker output ---"; printf '%s\n' "$O" | tail -3; }
  report 03c_correct_manifest_still_passes "$RC"
  echo "  03c rc=$R want=0 (the new gates do not just reject everything)"
  set +e
  O=$(EXPECTED_RECEIPT_SHA256=$RCSHA bash "$DIR/check_formal_binding_sm90.sh" \
        "$FB/receipt" "$FB/record" "$FB/manA" "$FB/so" 2>&1); R=$?
  set -u
  [ "$R" -eq 18 ] && has1 "$O" '^FORMAL_BINDING_FAIL:trusted toolchain image attestation not implemented.*$'
  report 03d_formal_still_has_no_pass_path $?
  echo "  03d rc=$R want=18 (regression guard: formal is still refused)"
else
  for C in 03a_old_passed_a_foreign_manifest 03b_foreign_manifest_refused \
           03c_correct_manifest_still_passes 03d_formal_still_has_no_pass_path; do report "$C" 1; done
  echo "  03* could not build a production record fixture"
fi

# ================================================================ generator ==
# PKG04: a receipt is evidence named by an out-of-band hash; publishing over an
# existing one leaves that hash pointing at bytes that no longer exist
gen() { # wrapper-dir out image-id
  ( cd "$PKGREPO" && IMAGE_ID="sha256:$(H64 "$3")" IMAGE_REF="img:$3" IMAGE_REPO_DIGESTS=NONE \
      BINARY_BUILD_COMMIT=UNKNOWN bash "$1/make_deploy_receipt.sh" "$PKGREPO" \
      benchmarks/manifests/tiny-h20-v1.manifest "$2" 2>&1 )
}
cp "$OLDB/make_deploy_receipt.sh" "$PKGREPO/benchmarks/make_deploy_receipt.sh"
( cd "$PKGREPO" && $GIT add -A && $GIT commit -qm oldgen ) >/dev/null 2>&1
set +e
O=$(gen "$PKGREPO/benchmarks" "$TMPD/pub.receipt" 1); R=$?
FIRST=$(sha256sum "$TMPD/pub.receipt" 2>/dev/null | cut -d' ' -f1)
O=$(gen "$PKGREPO/benchmarks" "$TMPD/pub.receipt" 2); R2=$?
SECOND=$(sha256sum "$TMPD/pub.receipt" 2>/dev/null | cut -d' ' -f1)
set -u
{ [ "$R" -eq 0 ] && [ "$R2" -eq 0 ] && [ -n "$FIRST" ] && [ "$FIRST" != "$SECOND" ]; }
report 04a_old_overwrote_a_published_receipt $?
echo "  04a old rc=$R/$R2 replaced the published receipt in place ($FIRST -> $SECOND)"
cp "$DIR/make_deploy_receipt.sh" "$PKGREPO/benchmarks/make_deploy_receipt.sh"
( cd "$PKGREPO" && $GIT add -A && $GIT commit -qm newgen ) >/dev/null 2>&1
rm -f "$TMPD/pub2.receipt"
set +e
O=$(gen "$PKGREPO/benchmarks" "$TMPD/pub2.receipt" 3); R=$?
KEEP=$(sha256sum "$TMPD/pub2.receipt" 2>/dev/null | cut -d' ' -f1)
O2=$(gen "$PKGREPO/benchmarks" "$TMPD/pub2.receipt" 4); R2=$?
AFTER=$(sha256sum "$TMPD/pub2.receipt" 2>/dev/null | cut -d' ' -f1)
set -u
{ [ "$R" -eq 0 ] && [ "$R2" -eq 2 ] && [ "$KEEP" = "$AFTER" ]; } \
  && has1 "$O2" '^RECEIPT_FAIL:could not publish receipt to .* \(already exists\? evidence is never overwritten\)$'
report 04b_existing_receipt_is_never_overwritten $?
echo "  04b first rc=$R second rc=$R2, published bytes unchanged"
# PKG15: the generator must be the committed code of the repo it describes
cp -r "$PKGREPO/benchmarks" "$TMPD/gencopy"
set +e
O=$(cd "$PKGREPO" && IMAGE_ID="sha256:$(H64 5)" IMAGE_REF="img:5" IMAGE_REPO_DIGESTS=NONE \
      BINARY_BUILD_COMMIT=UNKNOWN bash "$TMPD/gencopy/make_deploy_receipt.sh" "$PKGREPO" \
      benchmarks/manifests/tiny-h20-v1.manifest "$TMPD/pub3.receipt" 2>&1); R=$?
set -u
[ "$R" -eq 2 ] && has1 "$O" '^RECEIPT_FAIL:generator is running from .*, not .*/benchmarks$'
report 15a_generator_copy_outside_repo_refused $?
echo "  15a rc=$R want=2(a copy run from elsewhere cannot describe this repo)"
printf '\n# tampered\n' >> "$PKGREPO/benchmarks/validate_receipt_sm90.sh"
set +e
O=$(gen "$PKGREPO/benchmarks" "$TMPD/pub4.receipt" 6); R=$?
set -u
# the clean-tree gate fires first here, which is the correct answer for the same
# reason: an edited gate file must not produce a receipt. Self-binding covers
# the case the clean-tree gate cannot see - a generator copy run from outside
# the tree (15a). Neither can detect a change that was COMMITTED; that is what
# reviewing tracked code is for, and it is stated in the contract doc.
[ "$R" -eq 2 ] && has1 "$O" '^RECEIPT_FAIL:benchmarks tree not clean vs HEAD \(incl\. untracked\):$'
report 15b_edited_gate_code_refused $?
echo "  15b rc=$R want=2(edited gate file in the worktree)"
( cd "$PKGREPO" && $GIT checkout -- benchmarks/validate_receipt_sm90.sh ) >/dev/null 2>&1

# ================================================================= launcher ==
HW=$TMPD/harness
mkdir -p "$HW"
mkharness "$HW"
HM=$HW/mok
HRSHA=$(sha256sum "$HW/receipt" | cut -d' ' -f1)
oldlaunch() { # extra-env...
  clean_runs "$HW"
  ( cd "$TMPD" && env PATH="$HW/bin:$PATH" EXPECTED_RECEIPT_SHA256=$HRSHA "$@" \
      bash "$OLDB/host_launch_sm90.sh" ct "$HM" "$HW/man" "$HW/receipt" 2>&1 )
}
newlaunch() { # extra-env...
  clean_runs "$HW"
  ( cd "$TMPD" && env PATH="$HW/bin:$PATH" EXPECTED_RECEIPT_SHA256=$HRSHA "$@" \
      bash "$DIR/host_launch_sm90.sh" ct "$HM" "$HW/man" "$HW/receipt" 2>&1 )
}
# PKG06: BENCH_TAG becomes a filename component
set +e
O=$(oldlaunch BENCH_TAG=../evil); R=$?
NOUT=$(ls "$HM" | grep -c '^evil-' || true)
set -u
[ "$NOUT" -ge 1 ]; report 06a_old_tag_escaped_host_runs $?
echo "  06a old wrote $NOUT evidence files outside host-runs/ for BENCH_TAG=../evil"
rm -f "$HM"/evil-* 2>/dev/null
set +e
O=$(newlaunch BENCH_TAG=../evil); R=$?
NOUT=$(ls "$HM" | grep -c '^evil-' || true)
set -u
[ "$R" -eq 4 ] && [ "$NOUT" -eq 0 ] \
  && has1 "$O" '^LAUNCH_VERIFY_FAIL:BENCH_TAG must match \^\[A-Za-z0-9\]\[A-Za-z0-9\._-\]\{0,63\}\$ \(it becomes a filename\); got \[\.\./evil\]$'
report 06b_bad_tag_refused_before_any_write $?
echo "  06b rc=$R want=4, files created outside host-runs/: $NOUT"
# PKG08: the anchor is a shell-out; a hostile sha256sum forges it
sed 's|^SOURCE_TREE_COMMIT=.*|SOURCE_TREE_COMMIT=beefbeefbeefbeefbeefbeefbeefbeefbeefbeef|' "$HW/receipt" > "$HW/forged.receipt"
chmod 444 "$HW/forged.receipt"
HMSHA=$(sha256sum "$HW/man" | cut -d' ' -f1)
cat > "$HW/bin/sha256sum" << EOS
#!/bin/bash
for A in "\$@"; do
  case "\$A" in -*) continue ;; esac
  case "\$A" in
    *receipt*) echo "$HRSHA  \$A" ;;
    *man*)     echo "$HMSHA  \$A" ;;
    *)         /usr/bin/sha256sum "\$A" ;;
  esac
done
EOS
chmod +x "$HW/bin/sha256sum"
set +e
clean_runs "$HW"
O=$( cd "$TMPD" && env PATH="$HW/bin:$PATH" BENCH_TAG=forge EXPECTED_RECEIPT_SHA256=$HRSHA \
      bash "$OLDB/host_launch_sm90.sh" ct "$HM" "$HW/man" "$HW/forged.receipt" 2>&1 ); R=$?
set -u
[ "$(printf '%s\n' "$O" | grep -c 'RECEIPT_TRUST_FAIL' || true)" -eq 0 ]; report 08a_old_anchor_forged_by_path_shim $?
echo "  08a old accepted a forged receipt because sha256sum came from the caller's PATH"
set +e
clean_runs "$HW"
O=$( cd "$TMPD" && env PATH="$HW/bin:$PATH" BENCH_TAG=forge EXPECTED_RECEIPT_SHA256=$HRSHA \
      bash "$DIR/host_launch_sm90.sh" ct "$HM" "$HW/man" "$HW/forged.receipt" 2>&1 ); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^RECEIPT_TRUST_FAIL:EXPECTED_RECEIPT_SHA256 mismatch \(snapshot [0-9a-f]{64} expected [0-9a-f]{64}\)$'
report 08b_pinned_path_defeats_the_shim $?
echo "  08b rc=$R want=14 (misuse protection only - a hostile /usr/bin still wins)"
rm -f "$HW/bin/sha256sum"
# PKG09: measurement thresholds are contract, not caller preference
set +e
O=$(oldlaunch BENCH_TAG=thr LOAD1_MAX_PRELAUNCH=99999); R=$?
set -u
has1 "$O" '^.*PRELAUNCH_LOAD1_GATE.*max=99999.*$' || \
  [ "$(printf '%s\n' "$O" | grep -c 'TELEMETRY_GATE_FAIL:prelaunch load1' || true)" -eq 0 ]
report 09a_old_took_a_caller_threshold $?
echo "  09a old accepted LOAD1_MAX_PRELAUNCH=99999 from the caller"
set +e
O=$(newlaunch BENCH_TAG=thr LOAD1_MAX_PRELAUNCH=99999); R=$?
set -u
[ "$R" -eq 15 ] && has1 "$O" '^TELEMETRY_GATE_FAIL:LOAD1_MAX_PRELAUNCH is a tracked measurement threshold and cannot be set by the caller$'
report 09b_caller_threshold_refused $?
echo "  09b rc=$R want=15"
# PKG10: entrypoint
set +e
O=$(oldlaunch BENCH_TAG=ent BASH_ENV=/dev/null); R=$?
set -u
[ "$(printf '%s\n' "$O" | grep -c 'clean entrypoint' || true)" -eq 0 ]; report 10a_old_ignored_bash_env $?
echo "  10a old ran with BASH_ENV set"
set +e
O=$(newlaunch BENCH_TAG=ent BASH_ENV=/dev/null); R=$?
set -u
[ "$R" -eq 4 ] && has1 "$O" '^LAUNCH_VERIFY_FAIL:BASH_ENV is set; this launcher must be started from a clean entrypoint$'
report 10b_bash_env_refused $?
echo "  10b rc=$R want=4 (misuse protection, not proof)"
# PKG11: per-run evidence paths are O_EXCL
clean_runs "$HW"
PRE=$HM/host-runs/pre-existing.host
set +e
O=$( cd "$TMPD" && env PATH="$HW/bin:$PATH" BENCH_TAG=pre EXPECTED_RECEIPT_SHA256=$HRSHA \
      bash "$DIR/host_launch_sm90.sh" ct "$HM" "$HW/man" "$HW/receipt" 2>&1 )
set -u
SNAP=$(ls "$HM/host-runs"/*.manifest 2>/dev/null | head -1)
MODE=$(stat -c %a "$SNAP" 2>/dev/null)
SNAPSHA=$(sha256sum "$SNAP" 2>/dev/null | cut -d' ' -f1)
{ [ "$MODE" = "444" ] && [ "$SNAPSHA" = "$(sha256sum "$HW/man" | cut -d' ' -f1)" ]; }
report 11a_snapshot_is_readonly_and_anchored $?
echo "  11a per-run manifest snapshot mode=$MODE and equals the anchored bytes"
clean_runs "$HW"
mkdir -p "$HM/host-runs"
ln -sf /etc/passwd "$HM/host-runs/plant.host" 2>/dev/null
set +e
O=$( cd "$TMPD" && env PATH="$HW/bin:$PATH" BENCH_TAG=plant EXPECTED_RECEIPT_SHA256=$HRSHA \
      RUN_ID_OVERRIDE=1 bash -c '
        exec bash "$1" ct "$2" "$3" "$4"' _ "$DIR/host_launch_sm90.sh" "$HM" "$HW/man" "$HW/receipt" 2>&1 )
set -u
[ -L "$HM/host-runs/plant.host" ]; report 11b_planted_symlink_not_followed $?
echo "  11b a planted symlink in host-runs/ is still a symlink (nothing wrote through it)"
# PKG12: a schema-2 receipt asserts provenance the launcher is given no record for
sed 's|^RECEIPT_SCHEMA=1$|RECEIPT_SCHEMA=2|' "$HW/receipt" > "$HW/r2"
echo "BUILD_RECORD_SHA256=$(H64 7)" >> "$HW/r2"
sed -i 's|^BINARY_BUILD_COMMIT=UNKNOWN$|BINARY_BUILD_COMMIT='"$(H40 d)"'|' "$HW/r2"
chmod 444 "$HW/r2"; R2SHA=$(sha256sum "$HW/r2" | cut -d' ' -f1)
set +e
clean_runs "$HW"
O=$( cd "$TMPD" && env PATH="$HW/bin:$PATH" BENCH_TAG=s2 EXPECTED_RECEIPT_SHA256=$R2SHA \
      bash "$OLDB/host_launch_sm90.sh" ct "$HM" "$HW/man" "$HW/receipt" 2>&1 ); R=$?
set -u
[ "$(printf '%s\n' "$O" | grep -c 'claims a build-record binding' || true)" -eq 0 ]
report 12a_old_did_not_notice_schema2 $?
echo "  12a old had no opinion about a record-bound receipt in canary"
set +e
clean_runs "$HW"
O=$( cd "$TMPD" && env PATH="$HW/bin:$PATH" BENCH_TAG=s2 EXPECTED_RECEIPT_SHA256=$R2SHA \
      bash "$DIR/host_launch_sm90.sh" ct "$HM" "$HW/r2" "$HW/man" 2>&1 ); R=$?
set -u
[ "$R" -ne 0 ]; report 12b_schema2_refused_in_canary $?
echo "  12b rc=$R want!=0 (no record is supplied to this launcher)"
# PKG05/PKG07 regression guard: formal is still refused unconditionally
set +e
O=$(newlaunch BENCH_TAG=fm BENCH_MODE=formal); R=$?
set -u
[ "$R" -eq 14 ] && has1 "$O" '^FORMAL_MODE_FAIL:build-record contract not implemented$'
report 05_formal_mode_still_refused $?
echo "  05 rc=$R want=14 (regression guard, unchanged behaviour)"

EXPECTED=29
TOTAL=$((PASS+FAIL))
[ "$TOTAL" -eq "$EXPECTED" ] || { echo "PKG_COUNT_FAIL:ran $TOTAL cases, expected $EXPECTED"; FAIL=$((FAIL+1)); }
echo "PACKAGING_AUDIT pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
