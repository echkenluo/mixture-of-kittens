#!/usr/bin/env bash
# CPU-only SM90 build of the MoK extension inside the cu130 container on node 18.
# Writes the full ptxas log and a cuobjdump resource table under $ROOT/reports.
set -euo pipefail
ROOT=${MOK_WARPROLE_ROOT:-/home/lenovo/luocc/mok-warprole}
# The tree to build; another checkout under $ROOT lets kernel work go on while
# an experiment keeps $ROOT/src mounted at a fixed head.
SRC=${MOK_WARPROLE_SRC:-$ROOT/src}
IMG=${MOK_WARPROLE_IMAGE:-harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130}
# Extra docker run arguments, e.g. "--network host" on GPU9 whose bridge network is broken.
DOCKER_ARGS=${MOK_WARPROLE_DOCKER_ARGS:-}
REPORT_ONLY=${1:-}
mkdir -p "$ROOT/reports"
HEAD=$(git -C "$SRC" rev-parse --short HEAD)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/reports/$HEAD-$STAMP"
if [ "$REPORT_ONLY" != "--report-only" ]; then
  docker run --rm --name mok-warprole-build $DOCKER_ARGS --entrypoint bash \
    -v "$SRC:/w/src" "$IMG" -c \
    "cd /w/src && make ARCH=SM90 THUNDERKITTENS_ROOT=/w/src/third_party/ThunderKittens" \
    > "$OUT.ptxas.txt" 2>&1 || { echo "BUILD FAILED, tail of $OUT.ptxas.txt:"; tail -60 "$OUT.ptxas.txt"; exit 1; }
fi
docker run --rm $DOCKER_ARGS --entrypoint bash -v "$SRC:/w/src" "$IMG" -c \
  "cuobjdump --dump-resource-usage /w/src/mok/_C*.so | grep -E 'Function|REG' | paste - - | \
   sed -E 's/.*Function (.{0,110}).*REG:([0-9]+) STACK:([0-9]+) SHARED:([0-9]+).*/\2 reg \3 stack \4 smem  \1/'" \
  > "$OUT.resources.txt"
echo "--- resource lines of interest ---"
grep -iE 'warprole|contiguous|terminal_full' "$OUT.resources.txt" || echo "(none matched)"
echo "--- spill lines (ptxas) ---"
grep -E 'spill' "$OUT.ptxas.txt" | grep -v ' 0 bytes spill' || echo "no spills reported"
echo "reports: $OUT.ptxas.txt $OUT.resources.txt"
