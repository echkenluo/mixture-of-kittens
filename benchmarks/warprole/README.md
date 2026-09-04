# warprole build helper

`build_sm90.sh` compiles the SM90 extension without a GPU, inside the
`harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130` container (CUDA 13.0, Torch 2.11) on node 18,
and keeps two reports per build under `$ROOT/reports/<head>-<utc-stamp>`:

- `*.ptxas.txt` — the complete `make` output including every `-Xptxas=--verbose` line
  (register counts, spill bytes, shared memory per kernel);
- `*.resources.txt` — `cuobjdump --dump-resource-usage` of the produced `.so`, one line per kernel:
  `<reg> reg <stack> stack <static smem> smem  <mangled name>`.

Usage on node 18 (`$ROOT` defaults to `/home/lenovo/luocc/mok-warprole`, source tree in `$ROOT/src`):

```bash
bash benchmarks/warprole/build_sm90.sh            # build + reports (~3 min)
bash benchmarks/warprole/build_sm90.sh --report-only   # only regenerate the cuobjdump table
```

Baseline numbers on `4e925b2` (2026-09-04): `fp8_block_test::contiguous::kernel` 101 reg / 0 stack,
`fp8_block_terminal_full::kernel` 168 reg / 392 stack, routed `dispatch_kernel` 25 reg,
`combine_kernel` 20 reg. Any new `warprole` kernel must keep its K loops free of local-memory
traffic (check the SASS around the `QGMMA` groups with `cuobjdump -sass`, not only the ptxas totals);
the standalone GEMM/W13/comm kernels report 0 stack, the fused kernels a 56-byte frame (see below).

## ptxas findings (2026-09-04)

**`setmaxnreg` is honoured for allocation.** In the fused `c1s6` kernel (launch cap 168) the consumer
region between `USETMAXREG.TRY_ALLOC 0xe8` and the comm `USETMAXREG.DEALLOC` uses registers up to R229,
so ptxas does allocate the consumer code with the 232-register budget rather than the launch cap.

**The two-CTA-per-SM retreat form does not compile.** `<NC=1, STAGES=3, CTAS_PER_SM=2>`
(`__launch_bounds__(384, 2)`) stops with `C7602 Insufficient registers (80)` on the N128 WGMMA. Given the
point above, the likely reason is the per-CTA pool rather than the compile-time cap: with two CTAs the pool
is 32768 registers per CTA, and producer 40 + comm 64 + consumer 184 per thread times 128 threads is 36864.
Its entries are commented out in `csrc/bindings.cu`; a retreat form has to shrink the consumer budget
(M64 x N64 accumulators) instead of adding a second CTA per SM.

**Accumulator spills in the fused kernel (fixed in `ae3677b`).** With `if (kb == 0) copy else add` inside
the K loop of `consumer_task`, ptxas coalesced the running total with the WGMMA accumulator in one of the
two inlined loops of each fused kernel (W2 in `c1s6`, W13 in `c2s4`) and evacuated it with a register
shuffle before every MMA: 224 to 300 bytes of spill stores per kernel, all inside the loop. Peeling the
first K block (`block(0, total)` then an add-only loop) removed it. The remaining 56-byte frame (52 to 56
bytes of spill stores) is scalar reloads of `blockIdx`, `num_rows` and similar at task-loop heads inside
the 40- and 64-register producer/comm regions plus the frames of the non-inlined trap calls; no `STL`/`LDL`
sits between the `QGMMA` groups of either loop. Accepted.

## Getting the built tree onto GPU9

GPU9 cannot be reached from node 18 by ssh and the JumpServer command channel is limited to ~128 KB per
call, so the built tree travels as an image layer instead: node 18 bakes `src/` (git tree at the branch
head plus the untracked `mok/_C*.so`) on top of the build image and pushes it to harbor:

```bash
# node 18, after build_sm90.sh on the wanted head
cd /home/lenovo/luocc/mok-warprole/src
printf 'FROM harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130\nCOPY . /opt/mok-warprole\n' > ../Dockerfile.tree
docker build -f ../Dockerfile.tree -t harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130-mokwarprole-<head> .
docker push harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130-mokwarprole-<head>
```

On GPU9 pull that tag, copy the tree out to the host so logs and edits persist, and run the harnesses
from the host copy with the same image (the toolchain the `.so` was built with):

```bash
docker pull harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130-mokwarprole-<head>
id=$(docker create harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130-mokwarprole-<head> true)
mkdir -p /home/lenovo/luocc/mok-warprole && docker cp $id:/opt/mok-warprole /home/lenovo/luocc/mok-warprole/src && docker rm $id
docker run --rm --network host --gpus all -e CUDA_VISIBLE_DEVICES=1 \
  -v /home/lenovo/luocc/mok-warprole/src:/mok/src -v /home/lenovo/luocc/mok-warprole/runtime-logs:/mok/runtime-logs \
  --entrypoint bash harbor.lenovo.com/luocc/sglang-dsv4:a8-base-cu130-mokwarprole-<head> \
  -c "cd /mok/src && python3 -m benchmarks.bench_warprole_gemm"
```

`--network host` is required on GPU9 (no docker0). The provenance gate of the production bench needs
the host copy's `git status --porcelain` to be empty, so edit only through commits.
