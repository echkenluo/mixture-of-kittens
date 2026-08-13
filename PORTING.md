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

## r9 diagnosis
Forwarder failed: (a) COORD param lacked =coord<ST> default so braced-init
coords do not deduce; (b) 22 calls are group-scope (inside group<> struct,
compat overload unreachable). FIX = call-site transform: strip the trailing
", 0" / ", (uint16_t)(...), 0" LAST arg on tma::cluster::load_async calls
under SM90 (python regex over megakernel, calls span 1-2 lines), and guard
the tcgen05.ld drain block (~1482) + tt.addr uses behind #ifndef KITTENS_SM90
with zeroed d_reg scaffold. Then rebuild r10.

## P2 milestone (2026-08-13): device-op surface GREEN on H20
4-rank torchrun: test_build_schedule, test_all_gather_top_experts (NVLS
multicast live), test_schedule, test_fwd_epilogue, test_bwd_epilogue -- all
PASSED. Only red: megakernel fwd/bwd (hangs at MMA scaffold no-ops, combine
semaphores never satisfied -- the designed behavior). Single remaining front:
wgmma register-accumulator rewrite per the P1-B v1 design, then
test_forward_bf16 green = port correctness goal met.

## P2 red/green map CLOSED (2026-08-13)
GREEN 6/6 device ops (schedule/all_gather-NVLS/scheduling/epilogues).
RED single-cause: megakernel fwd bf16 AND mxfp8 both hang at the MMA
scaffold (combine semaphores starve) - confirmed by timeout runs.
ONLY remaining front: wgmma register rewrite of the GEMM worker
(design frozen above). test_forward_bf16 green == port correctness done.

## r14: wgmma worker unit COMPILER-VALIDATED (0 errors in full TU)
wgmma_acc (rt_fl per-warp accumulator, warpgroup::mm_/mma_ABt,
mma_async_wait, warpgroup::store drain) compiles clean at SM90 inside the
megakernel TU. Next patch (r15): replace consumer-branch mma2 call sites
with wgmma_acc step/drain sequence wired to the existing input-ring
semaphores + full-N B loads; then torchrun regression until
test_forward_bf16 green; then V4-shape numerics vs DeepEP+DeepGEMM and
methodology-v2 measurement.

## r15 anchors (verified)
Producer wg = groupid()==NUM_CONSUMERS (TMA loads, warpid3 leader; the MMA
tcgen05-issue subbranch lives here too -> guard out on SM90). Consumer wg
(groupid<NUM_CONSUMERS, 128 threads) currently waits gemm_outputs_arrived
then drains tmem -> SM90: it instead waits gemm_inputs_arrived[ring] per K
stage, runs wgmma_acc::step_ABt, arrives gemm_inputs_finished[ring], then
drain_to the existing d smem/TMA store path. Two anchor edits: producer
branch ~1332-1435 (guard), consumer else ~1439+ (replace head).

## Codex port review ADOPTED (2026-08-13 20:47 review)
All findings verified true: [P0-1] drain_to never called, epilogue still
zero-fills (my R10 scaffold never replaced); [P0-2] CTA ownership covers
only diagonal quadrants (A AND B both cta_rank-halved; R19 "natural fit"
reasoning was wrong); [P1-3] expect_bytes counts CLUSTER_SIZE x (A+B) but
TMA delivers 1x per local barrier -> the hang; [P1-4] capability gate
overexposes SM90; test harness TEST_EXIT:0 on timeouts = false completions;
only 3/4 beacons landed (4th replace was a no-op).

## SM90 v1 ownership decision (adopted per review order step 1)
CLUSTER_SIZE=1 with quarter tiles (MLP_Mb=128, MLP_Nb=128) for SM90:
single-CTA task ownership, full quadrant coverage via task decomposition,
acc = 2 x rt_fl<16,128> = 128 regs/thread (fits), no cross-CTA mbarrier
semantics at all. Next per review: (2) standalone wgmma_acc numeric test
vs torch.matmul BEFORE megakernel rewiring; (3) expect_bytes 1x + ordering;
(4) drain into 8 epilogue N-stages, no zero-fill; (5) harness real exit
codes + pytest summary required; (6) fail-fast MXFP8/wgrad on SM90.

## MILESTONE: wgmma worker numerics VERIFIED (review step 2 complete)
Standalone unit test vs torch.matmul: max_rel 0.39% at K=64/256/4096
(textbook bf16 accumulate), zero_frac 0. Root cause of the earlier wrong
numbers found and documented: TK wgmma smem descriptors IGNORE st_subtile
offsets (both M-chunks read chunk 0) while the register store path honors
them - fixed via stacked-tile reinterpret for A M-halves. Rule for all
future wiring: never feed st_subtile views to wgmma; use real tile objects.
Next: megakernel rewiring - CLUSTER_SIZE=1 quarter-tile ownership + drain
into the 8 epilogue N-stages, then full BF16 matrix under the honest
harness (real exit codes, pytest summary required).

## SM90 dataflow design (final): h-loop emulation of the 2-CTA cluster
Whole worker datapath is built on '2*tile_coord.x + cta_rank' half-splits.
CLUSTER=1 correct form: one CTA sequentially iterates h in {0,1} replacing
cta_rank -- loads BOTH halves per ring stage (a/b smem slots x2, expect_bytes
= 2x(A+B)), mma per half into acc.acc[h], epilogue stores each half at
{2*tile_coord.x + h}. Staged rollout: step A = half-0 drain plumbing only
(predicted numeric gate failure at exactly 50% row coverage validates the
plumbing); step B = full h-loop in producer loads + expect 2x + both-half
stores.

## Codex R29 urgent review ADOPTED - frozen quadrant table (before any wiring)
Defects confirmed: (1) h-loop as written was diagonal-only again; (2) hoisted
wgmma_acc<a_tile,b_tile,128,256> with quarter-config a_tile=st_bf<64,64> is an
OUT-OF-OBJECT smem read (MCH=2 stacks past the single tile); R29 results void.
Standalone worker test remains valid (owns real 128x128 operands).

### Frozen SM90 ownership table (per task = 128x128 output, CLUSTER=1)
Quadrant (m,n), m,n in {0,1}; K-stage = ring slot r; task coords (x,y):
| q | A smem obj | A TMA coord | B smem obj | B TMA coord | acc | epilogue dest |
|---|---|---|---|---|---|---|
| (0,0) | a_smem[r][0] st_bf<64,64> | {2x+0, k} | b_smem[r][0] st_bf<64,64> | {k, 2y+0}/layout var | acc[0][0] rt_fl<16,64> | stages 0-3 rows via {2x+0} |
| (0,1) | a_smem[r][0] | same | b_smem[r][1] | {k, 2y+1} | acc[0][1] | stages 4-7 rows via {2x+0} |
| (1,0) | a_smem[r][1] | {2x+1, k} | b_smem[r][0] | same as (0,0) | acc[1][0] | stages 0-3 rows via {2x+1} |
| (1,1) | a_smem[r][1] | same | b_smem[r][1] | same as (0,1) | acc[1][1] | stages 4-7 rows via {2x+1} |
Barrier bytes per stage: 2*sizeof(a_tile) + 2*sizeof(b_tile) (two real A
objects + two real B objects; no subtiles anywhere near wgmma).
Registers: 4 x rt_fl<16,64> = 128/thread. Smem: doubles vs quarter config =
equals the original full-config budget. Epilogue: outer m loop over the
existing 8-stage loop; stage i uses acc[m][i/4], cols (i%4)*16..+16, store
coord {2*tile_coord.x + m, EPI*tile_coord.y + i}.

## MILESTONE: quadrant worker VERIFIED (both layouts)
wgmma_quad passes vs torch.matmul: AB and ABt, K=256/4096, max_rel 0.39%,
zero_frac 0. Codex P0-2 (quadrant loss) remedied and numerically certified.
Next: step-B megakernel rewiring (producer loads both halves per stage,
expect 2x, epilogue stitches quadrants), then honest-harness BF16 matrix.

## Codex issue-3 remediation COMPLETE (gates real, proven)
Public-API fail-fast on SM90: mxfp8_quantize raises before launch (verified
by real call), fwd_mxfp8/bwd_mxfp8/bwd_bf16 carry _sm90_reject before any
_C. launch (source-order check). Gate proof: tests/sm90_gate_check.py 5/5.
The MXFP8 packing static_assert gating is now justified: unsafe paths are
unreachable from the public API.

## Quadrant worker numerics VERIFIED (R31)
wgmma_quad passes vs torch.matmul on BOTH layouts: AB & ABt, K=256/4096,
max_rel 0.39%, zero_frac 0. Next: step-B megakernel rewiring (producer loads
both a-halves+b-halves per ring stage, expect_bytes 2x(A+B), consumer
quadrant steps, epilogue drains stitched halves at {2x+h}).

## Gate remediation VERIFIED (codex issue 3 closed properly)
tests/sm90_gates_check.py (run on H20): import+torch.ops.mok registration OK
(earlier infer_schema failure = stale __pycache__; AST confirms decorator
adjacency intact); mxfp8_quantize full-signature dispatcher call raises
NotImplementedError; all three fused ops proven to reach _sm90_reject as the
FIRST statement via sentinel monkeypatch on the undecorated implementations
(TypeError shortcuts explicitly not counted). With these real fail-fast
gates, the SM90 scoping of the MXFP8 packing static_assert is justified:
the exempted path is unreachable from the public API.
Separately: R31 quadrant worker unit test green (AB & ABt, K=256/4096,
max_rel 0.39%, zero_frac 0) - kept as an independent result.

## Codex step-B P0 quartet FIXED (pre-build source invariants pass)
(1) smem order a0/b0/a1/b1/scales via early constexpr, blanket-replace
regression removed; (2) b1 mirrors per layout variant (IS_AB {z,k,2y+1},
else {z,2y+1,k}); (3) wgmma_quad takes explicit IS_AB; (4) expect_bytes
ownership moved into the elected producer before the first TMA load
(consumer only waits). Note: two legacy expect sites remain inside the
`false &&`-disabled SM100 issue branch - dead on SM90 by construction.
