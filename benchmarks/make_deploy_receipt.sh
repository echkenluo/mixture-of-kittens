#!/bin/bash
# Deployment receipt generator (tracked). Runs ONLY on the trusted packaging
# host, inside the git work tree. The receipt plus its printed
# EXPECTED_RECEIPT_SHA256 are the remote trust anchors: the deployment target
# never self-signs, and the expected hash must travel out-of-band (command
# line / recorded in the tracker), NOT be recomputed on the verified end.
#   - SOURCE_TREE_COMMIT / HARNESS_COMMIT: git HEAD the benchmarks tree was
#     packaged from (benchmarks/ must be clean vs HEAD INCLUDING untracked)
#   - BINARY_BUILD_COMMIT: build lineage of the .so. Requires an actual build
#     record; without one it MUST be the literal UNKNOWN (formal mode then
#     refuses to run). Never backfill a plausible commit as fake lineage.
#     If 40-hex it must resolve to a commit in this repository.
#   - MANIFEST_SHA256 bound to the committed blob; HARNESS_SHA256 measured
#     from the tree and cross-checked against the manifest
#   - SO_SHA256 copied from the committed manifest's EXPECTED_SO_SHA256
#     (the packaging host has no .so binary; the remote gate measures bytes)
#   - IMAGE_ID / IMAGE_REF / IMAGE_REPO_DIGESTS of the target container image
#     (IMAGE_REPO_DIGESTS=NONE marks a local-only image: canary-only)
# Output is written atomically (same-dir temp + rename) and self-validated
# with validate_receipt_sm90.sh before the expected hash is printed.
# Usage: make_deploy_receipt.sh <repo_dir> <manifest_relpath> <out_receipt>
# Required env: IMAGE_ID IMAGE_REF IMAGE_REPO_DIGESTS BINARY_BUILD_COMMIT
set -euo pipefail
REPO=${1:?repo dir}; MREL=${2:?manifest relpath}; OUT=${3:?output receipt path}
: "${IMAGE_ID:?IMAGE_ID required}"
: "${IMAGE_REF:?IMAGE_REF required}"
: "${IMAGE_REPO_DIGESTS:?IMAGE_REPO_DIGESTS required (literal NONE for local-only image)}"
: "${BINARY_BUILD_COMMIT:?BINARY_BUILD_COMMIT required (full 40-hex with build record, else literal UNKNOWN)}"
cd "$REPO"
DIRTY=$(git status --porcelain -- benchmarks)
[ -z "$DIRTY" ] || { echo "RECEIPT_FAIL:benchmarks tree not clean vs HEAD (incl. untracked):"; echo "$DIRTY"; exit 2; }
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$DIR/validate_manifest_sm90.sh" "$MREL" >/dev/null || { echo "RECEIPT_FAIL:manifest invalid"; exit 2; }
SRC=$(git rev-parse HEAD)
MSHA=$(sha256sum "$MREL" | cut -d' ' -f1)
BLOB=$(git rev-parse "HEAD:$MREL")
CSHA=$(git cat-file blob "$BLOB" | sha256sum | cut -d' ' -f1)
[ "$MSHA" = "$CSHA" ] || { echo "RECEIPT_FAIL:manifest worktree bytes != committed blob"; exit 2; }
# the harness this manifest actually selects (schema 1 is MoK-only)
# || true is required: this script runs under set -e, and a schema-1 manifest
# legitimately has no HARNESS_MODULE line, so a bare grep would abort the
# generator for exactly the contract the tiny9 chain uses
HMOD=$(grep '^HARNESS_MODULE=' "$MREL" | head -1 | cut -d= -f2- || true)
[ -n "$HMOD" ] || HMOD=benchmarks.bench_sm90_fwd
HFILE=$(echo "$HMOD" | tr '.' '/').py
[ -f "$HFILE" ] || { echo "RECEIPT_FAIL:harness file $HFILE for $HMOD missing"; exit 2; }
HSHA=$(sha256sum "$HFILE" | cut -d' ' -f1)
grep -q "^EXPECTED_HARNESS_SHA256=$HSHA$" "$MREL" || { echo "RECEIPT_FAIL:tree harness sha != manifest EXPECTED_HARNESS_SHA256"; exit 2; }
SOSHA=$(grep '^EXPECTED_SO_SHA256=' "$MREL" | head -1 | cut -d= -f2)
# Optional build-record binding (schema 2). When a record is supplied, the
# commit is DERIVED from it - the operator cannot assert a different one - and
# the manifest's expected SO must already agree with what was actually built.
# A build record can no longer be turned into a receipt. The receipt's IMAGE_*
# fields carry no label, so binding a record would launder a caller
# declaration into something a later reader takes as verified. There is
# deliberately NO override variable: an escape hatch is the same hole with an
# extra step. Receipts are schema 1 (canary) until a trusted toolchain image
# attestation exists.
RSCHEMA=1; BRSHA=""
if [ -n "${BUILD_RECORD:-}" ]; then
  echo "RECEIPT_FAIL:publishing a receipt from a build record requires a trusted toolchain image attestation, which is not implemented"
  exit 2
fi
# resolvability is checked AFTER the record binding: when a record is present
# the commit is derived from it, and a disagreement must be reported as a
# disagreement rather than as "not resolvable"
if [ "$BINARY_BUILD_COMMIT" != "UNKNOWN" ]; then
  echo "$BINARY_BUILD_COMMIT" | grep -qE '^[0-9a-f]{40}$' \
    || { echo "RECEIPT_FAIL:BINARY_BUILD_COMMIT must be full 40-hex or UNKNOWN"; exit 2; }
  [ "$(git cat-file -t "$BINARY_BUILD_COMMIT" 2>/dev/null)" = "commit" ] \
    || { echo "RECEIPT_FAIL:BINARY_BUILD_COMMIT not resolvable in this repo"; exit 2; }
fi
OUTDIR=$(dirname "$OUT")
TMP=$(mktemp "$OUTDIR/.receipt.XXXXXX")
trap 'rm -f "$TMP"' EXIT
{
  echo "RECEIPT_SCHEMA=$RSCHEMA"
  echo "SOURCE_TREE_COMMIT=$SRC"
  echo "HARNESS_COMMIT=$SRC"
  echo "BINARY_BUILD_COMMIT=$BINARY_BUILD_COMMIT"
  echo "MANIFEST_FILE=$(basename "$MREL")"
  echo "MANIFEST_SHA256=$MSHA"
  echo "MANIFEST_GIT_BLOB=$BLOB"
  echo "HARNESS_SHA256=$HSHA"
  echo "SO_SHA256=$SOSHA"
  echo "IMAGE_ID=$IMAGE_ID"
  echo "IMAGE_REF=$IMAGE_REF"
  echo "IMAGE_REPO_DIGESTS=$IMAGE_REPO_DIGESTS"
} > "$TMP"
chmod 444 "$TMP"
# validate BEFORE publishing: a bad receipt must never reach $OUT, and must
# never clobber an existing good one. The EXIT trap removes the temp on any
# failure path.
bash "$DIR/validate_receipt_sm90.sh" "$TMP" --check-mode >/dev/null \
  || { echo "RECEIPT_FAIL:generated receipt failed self-validation (not published)"; exit 2; }
mv -f "$TMP" "$OUT"
trap - EXIT
echo "EXPECTED_RECEIPT_SHA256:$(sha256sum "$OUT" | cut -d' ' -f1)"
