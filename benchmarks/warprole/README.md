# warprole build helper

## H20 ownership API (complete-port branch)

`warprole_forward` has been replaced with two explicitly different APIs:

- `warprole_forward_from_topk(workspace, state, config, ...)` builds the schedule
  while holding the workspace lease and returns an independent output copy.
- `warprole_forward_leased(workspace, state, schedule, ...)` is a low-level
  operation. The caller acquires before any schedule/input mutation, keeps the
  lease through output materialization, and releases afterwards on that stream.
  Its return aliases `state.output`. Failed calls leave the workspace unusable.

Create state collectively before the transaction. The service adapter uses the
leased entry. The layer microbenchmark uses a prebuilt, immutable schedule and
now includes an output copy on both split and warp-role arms; historical timings
without that copy have a different boundary. Concurrent unsequenced calls using
the same workspace fail closed; event-ordered stream hand-off is the supported
reuse pattern. GPU correctness tests default to 256 global experts:

```bash
MOK_SM90_EXPERIMENTAL=1 torchrun --standalone --nproc-per-node=4 -m pytest -x -s tests/test_warprole_ep4.py
MOK_SM90_EXPERIMENTAL=1 torchrun --standalone --nproc-per-node=8 -m pytest -x -s tests/test_warprole_ep4.py
```

The CPU tests under `tests/host` check Python ownership and finite-output
assertions only. They do not establish CUDA synchronization or numerical parity.

Additional complete-port gates run in separate processes after that matrix:

```bash
torchrun --standalone --nproc-per-node=4 -m pytest -x -s tests/test_warprole_reuse_stress.py
torchrun --standalone --nproc-per-node=4 -m tests.warprole_reentry_probe /results
```

Use eight ranks for EP8. Reuse checks 256 candidate calls per variant against
eight precomputed split references, with no split call during the candidate
sequence. It includes a rank sending no routes, every rank targeting six experts
on rank zero, all-padding, partial padding, changed inputs, and retained output
ownership. It is an extended reuse check, not a long-duration soak test.
The reentry probe deliberately traps each CUDA context and exits without CUDA
cleanup. The parent must require all rank receipts and the exact
`EXPECTED_REENTRANT_TRAP_PASS` verdict; an arbitrary nonzero exit is not a pass.
These gates do not establish independent numerical accuracy or service quality.

`tests/test_warprole_sglang_core.py` separately exercises the actual SGLang
`_run_native_core` on EP4/EP8 with synthetic weights, two variants and T256/1792.
Use the complete-port SGLang checkout in `PYTHONPATH`; the deployment runner
requires its full Git commit and verifies the loaded adapter file on each rank.
The test uses real quantization, process groups and kernels. It checks changed
inputs/routes, independent returned copies and device-visible lease ownership.
It does not exercise model loading, policy admission, shared experts, graphs or
HTTP serving and therefore does not replace the complete service gate.

## Service-shaped internal timing diagnostics

`bench_warprole_sched.py` retains its historical 64-expert/two-case defaults.
For the current TP4/EP4 DSV4 geometry, select 256 experts and the explicit
1024/2048 local-token cases with schedule capacity factor 5:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 benchmarks/bench_warprole_sched.py --total-experts 256 --cases service_uniform_1024,service_uniform_2048 --variants c2s4 --knobs real --probe --out /results/service-shaped-timing.json
```

This uses uniform synthetic routes and synthetic weights, not captured model
routes/weights. The raw per-CTA globaltimer stamps are retained alongside the
summary, so phase relationships can be recomputed without subtracting unrelated
medians. Compare timestamps within a rank; cross-GPU clock alignment is not
established. Probe calls occur after timed A/B/A blocks and are not included in
the reported timing samples. Real mode includes prepare in every fused call.

Real-mode time-based warmup uses a rank-zero broadcast stop decision so all ranks
issue the same number of communicating kernel calls, even with host clock skew.
The broadcasts occur outside measured blocks.

The standalone W13/W2 pair includes the fused W13 activation/quantization but
omits dispatch, combine and final reduction, so its difference from the real fused path is not a matched
full-pipeline regression. The historical 3% step-3 overhead gate applies only to
`--knobs all`; real and ablation modes are diagnostic. No mode establishes model
quality or service performance. This benchmark sets destructive ablation knobs
at import and resets them in `main()` according to `--knobs`; never import it
into a serving process.

Each rank now retains its own padded active-row count and raw per-expert token
counts, so rank timing differences can be checked against actual scheduled work.
The runtime Git query trusts only the explicitly mounted repository for that
command; it does not modify global Git configuration. Query failures raise an
error instead of being reported as an empty clean status.
Each measured block also records host wall-clock bounds, including its warmup
and synchronizations, for matching external clock telemetry. These bounds are
not individual kernel timestamps and do not replace CUDA-event samples.

## Build and resource reports

`MOK_WARPROLE_INTERLEAVE=1` selects the experimental communication schedule:
dispatch batches 0 and 1, then combine batch q before dispatching q+2. This
preserves the compute task stream, readiness counters, final push/reduction
protocol, and full-capacity buffers. It is not ring-buffer reuse. The default
remains the existing dispatch-all then combine path. Probe slot 2 still means
last dispatch completion; in interleaved mode the elapsed interval includes
earlier combine work and waits. GPU correctness and same-binary on/off timing
must pass before any service adoption.

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

**ptxas contracts the consumer's scale-and-accumulate into FFMA (found on the first H20 run,
2026-09-05).** `fp8_block_pipeline::run_tile` compiles its per-K128 `mul_row` + `add` to FMUL + FADD
(the copy-or-add sits in another basic block); the peeled warp-role consumer loop put the same pair in
one basic block and ptxas fused it into 64 FFMA per K block (SASS: 64 FFMA, 0 FADD in each standalone
kernel). The outputs then differed from split in ~0.008% of bf16 elements, almost all by one ulp,
uniformly over rows, columns and tiles. `gemm::accumulate_rn` (`__fadd_rn`, never contracted) restores
FMUL + FADD; check with `cuobjdump -sass` that the standalone kernels show 64 FADD and 0 FFMA before
trusting any bitwise result from a new build.

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
  -c "git config --global --add safe.directory /mok/src && cd /mok/src && python3 -m benchmarks.bench_warprole_gemm"
```

`--network host` is required on GPU9 (no docker0). The `safe.directory` line is not optional: the host
copy is owned by another uid than the container's root, so without it git refuses to read the tree and
the harnesses record `git_head: unavailable` with an empty (clean-looking) `source_state`. The
provenance gate of the production bench needs the host copy's `git status --porcelain` to be empty, so
edit only through commits.

Runs longer than a minute or so must be detached from the JumpServer session (`nohup setsid bash
chain.sh > runtime-logs/chain.log 2>&1 < /dev/null &`, then poll the log): an attached
`docker run` died together with the session on 2026-09-05 and left no output. Use
`git --no-pager` inside the session, the pty otherwise stops at a pager prompt.
