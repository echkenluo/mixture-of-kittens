# H20/SM90 Port Notes (branch h20-sm90-port)

Build loop: local edit -> git diff patch -> apply at GPU9:/home/lenovo/luocc/mok/
mixture-of-kittens -> docker exec mok-build "make ARCH=SM90" -> error taxonomy.
Logs: /mok/build-sm90*.log. Runtime: H20 MULTICAST_SUPPORTED=1 (NVLS comm OK).

## Convergence
- r1 (stock + SM90 makefile branch): 96 errors
- r2 (+ sm90_compat.cuh: e8m0->uint8 smem parse stubs): 55 errors
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
