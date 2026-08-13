#!/bin/bash
# Deployment receipt generator (tracked). Runs ONLY on the trusted packaging
# host, inside the git work tree. The receipt is the remote trust anchor: the
# deployment target never self-signs (no git init / no self-derived expecteds
# on the verified end). It records:
#   - SOURCE_TREE_COMMIT / HARNESS_COMMIT: git HEAD the benchmarks tree was
#     packaged from (worktree must be clean vs HEAD for benchmarks/)
#   - BINARY_BUILD_COMMIT: packaging-side assertion of the .so build lineage.
#     This is provenance by operator record, NOT proven by the manifest's
#     FROZEN_COMMIT (which cannot prove build lineage by itself).
#   - MANIFEST_SHA256 bound to the committed blob (worktree bytes must equal
#     git blob), plus HARNESS_SHA256 measured from the tree
#   - SO_SHA256 copied from the committed manifest's EXPECTED_SO_SHA256
#     (the packaging host has no .so binary; the remote gate measures bytes)
#   - IMAGE_ID / IMAGE_REF / IMAGE_REPO_DIGESTS of the target container image
#     (pass IMAGE_REPO_DIGESTS=NONE only for a local-only image)
# Usage: make_deploy_receipt.sh <repo_dir> <manifest_relpath> <out_receipt>
# Required env: IMAGE_ID IMAGE_REF IMAGE_REPO_DIGESTS BINARY_BUILD_COMMIT
set -euo pipefail
REPO=${1:?repo dir}; MREL=${2:?manifest relpath}; OUT=${3:?output receipt path}
: "${IMAGE_ID:?IMAGE_ID required}"
: "${IMAGE_REF:?IMAGE_REF required}"
: "${IMAGE_REPO_DIGESTS:?IMAGE_REPO_DIGESTS required (literal NONE for local-only image)}"
: "${BINARY_BUILD_COMMIT:?BINARY_BUILD_COMMIT required}"
cd "$REPO"
git diff --quiet HEAD -- benchmarks || { echo "RECEIPT_FAIL:benchmarks tree dirty vs HEAD"; exit 2; }
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$DIR/validate_manifest_sm90.sh" "$MREL" >/dev/null || { echo "RECEIPT_FAIL:manifest invalid"; exit 2; }
SRC=$(git rev-parse HEAD)
MSHA=$(sha256sum "$MREL" | cut -d' ' -f1)
BLOB=$(git rev-parse "HEAD:$MREL")
CSHA=$(git cat-file blob "$BLOB" | sha256sum | cut -d' ' -f1)
[ "$MSHA" = "$CSHA" ] || { echo "RECEIPT_FAIL:manifest worktree bytes != committed blob"; exit 2; }
HSHA=$(sha256sum benchmarks/bench_sm90_fwd.py | cut -d' ' -f1)
grep -q "^EXPECTED_HARNESS_SHA256=$HSHA$" "$MREL" || { echo "RECEIPT_FAIL:tree harness sha != manifest EXPECTED_HARNESS_SHA256"; exit 2; }
SOSHA=$(grep '^EXPECTED_SO_SHA256=' "$MREL" | head -1 | cut -d= -f2)
rm -f "$OUT"
{
  echo "RECEIPT_SCHEMA=1"
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
} > "$OUT"
chmod 444 "$OUT"
echo "RECEIPT_SHA256:$(sha256sum "$OUT" | cut -d' ' -f1)"
