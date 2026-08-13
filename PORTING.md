# H20/SM90 Port Notes (branch h20-sm90-port)

Build loop: local edit -> git diff patch -> apply at GPU9:/home/lenovo/luocc/mok/
mixture-of-kittens -> docker exec mok-build "make ARCH=SM90" -> error taxonomy.
Logs: /mok/build-sm90*.log. Runtime: H20 MULTICAST_SUPPORTED=1 (NVLS comm OK).

## Convergence
- r1 (stock + SM90 makefile branch): 96 errors
- r2 (+ e8m0 smem stubs): 55; r3 (+ full_tt stub): 46; r4 (+ CLC shim): 14
  - e8m0 smem family and all cascades (tensor_to_gl/packing/tma::store): CLEARED

## Remaining fronts (r2 taxonomy)
1. P1-B tensor-memory family (~30): `tt<>` accumulator, `tensor_allocator`,
   `full_tt_fp8e8m0<N>` (tmem scale tiles - only used in USE_ROUTED_MXFP8
   branches -> parse stub with subtile<> method is enough for BF16-only),
   `tensor_load_wait/tensor_before/after_thread_sync`, `detail::tcgen05::commit`,
   part of the 19 "::" errors. Real rewrite sites (BF16 path):
   - mma dispatch loop mok_megakernel.cuh:1412-1427 (`mma2_AB/ABt/AtB` -> defined
     in utils.cuh, check semantics; SM90: warpgroup::mm_/mma_ on rt_fl accumulator)
   - commit at 1433 (tcgen05::commit -> not needed with wgmma; arrive semaphore
     directly after mma_async_wait)
   - epilogue drain 1443-1537 (warpgroup::load_async from d_tt subtile +
     inline tcgen05.ld PTX -> accumulator already in registers; restructure to
     store rt -> smem d tiles per EPI_PIPE stage)
   - allocations at 1644 (fwd) and 2226 (bwd): tm_alloc -> remove; rt_fl lives
     in the consumer warpgroup registers. NOTE register budget: rt_fl<Mb/2=128?,
     Nb=256> per warpgroup = check MLP_Mb/2 x MLP_Nb tile split across 4 warps;
     may need Nb split into EPI stages to fit registers (SM90 wgmma N max 256,
     accum fp32 regs 128x256/128threads*4B -> too big; split N into 2-4 chunks
     and loop K per chunk, or reduce MLP_Nb for SM90 config).
2. P1-C CLC family (~14): clc_handle/clc_drain_handle/clc:: -> software
   persistent scheduler shim (atomic ticket on device global; preserve
   handle/pipe call-site shape). Sites concentrated 1598-1650 + worker loop.
3. "expected a ;" x4: inspect after B/C.

## After compile: P2 gates
- make test (torchrun 4-GPU, tests vs mok/_fake_impls.py) on GPUs 0-3/4-7
- V4 shape single-layer vs DeepEP+DeepGEMM reference numerics

## P1-B design decision (r3 prep)
`mma2_*` are TK SM100 2-CTA cluster MMA (both CTAs feed one tmem accumulator).
SM90 rewrite: KEEP the cluster + multicast loads unchanged; split the
accumulator per-CTA (each CTA owns half of N via cta_rank), run plain
warpgroup::mma_AB/ABt/AtB on rt_fl<.,128> halves, mma_async_wait then arrive
the existing semaphores (replaces tcgen05::commit). Epilogue: each CTA drains
its own register half to its d smem tiles (removes tcgen05.ld PTX + d_tt
subtile loads). This also resolves the register-budget concern (128-wide
accum per warpgroup fits). Scale-tt params (a_sc_tt/b_sc_tt) get parse stubs
only (MXFP8 branches discarded in BF16 build).

## P1-C contract (captured from call sites)
clc::handle POD in smem; clc::schedule(handle, semaphore) async-requests next
work and arrives the semaphore; clc::query(handle) -> {bool success; u32 x}
with x a fresh CTA id (x / CLUSTER_SIZE = cluster_idx; success=false => no
more work, worker exits). Software shim: global atomic ticket over the
launched task-id space; MUST mirror TK's SM100 clc header semantics for the
id space bound + per-launch counter reset (read
third_party/ThunderKittens/include/ops/.../clc* next round; do NOT guess).
Both fwd (1600-1849) and bwd (2177-2530) use identical patterns.

## P1-C v1 shim design (from TK util.cuh:237-289)
CLC = clusterlaunchcontrol.try_cancel: steals UNLAUNCHED grid CTA ids.
SM90 v1: disable stealing - query() always {success=0}; every grid block
launches normally and runs only its initial task (correct; persistence is a
later perf step). schedule(h, sem) must COMPLETE the caller's
expect_bytes(sizeof(handle)) tx-count on ALL cluster CTAs' semaphores or the
wait deadlocks: use mbarrier complete_tx w/ cluster multicast (find TK helper:
grep complete_tx / cluster arrive in include/ops/*/util). Then delete-or-keep
the drain pipeline accordingly (drain stages also expect_bytes handle-sized).

## P1-B v1 structural decision (r5 prep)
mma2 splits A and B per-CTA and joins via tmem across the cluster. SM90 wgmma
cannot read peer-CTA smem operands, so v1: each CTA loads the FULL N of B
(drop B's cluster multicast split; A stays M-half per CTA), computes
M-half x full-N into rt_fl register accumulator (per-warp height (Mb/2)/4/16
tiles x N). Smem cost: B width Nb/2 -> Nb per K-stage; if over budget cut
MLP_LOAD_PIPE_DEPTH 6 -> 3 for SM90 config. Epilogue: drain rt directly to
d smem tiles per EPI stage (replaces tmem load_async + tcgen05.ld).
tensor_load_wait -> warpgroup::mma_async_wait; tensor_*_thread_sync -> no-op;
tcgen05::commit -> plain semaphore arrive after mma_async_wait.

## r8 taxonomy (final work surface, 101 errs after tt scaffold unmasked MMA layer)
- 42x tma::cluster::load_async overload: SM90 TK signature differs from SM100
  call form (smem,gmem,coord,sem,mask,0) - check TK SM90 header, adapt sites
  (mechanical, or wrap in compat with the trailing-arg dropped).
- 14x group::load_async from tt subtiles (epilogue drain): DISAPPEAR with the
  register-accumulator rewrite (drain directly from rt_fl).
- 8x mma2_ABt + 8x mm2_ABt undefined: replace per PORTING P1-B v1 design
  (warpgroup::mm_/mma_ABt on per-CTA rt_fl, B full-N).
- 6x load_mxnv_scale_async2: MXFP8-only branches -> parse stub in compat.
Convergence: 96/55/46/14/10/9/3 -> unmasked 101 (deeper layer, mapped).
