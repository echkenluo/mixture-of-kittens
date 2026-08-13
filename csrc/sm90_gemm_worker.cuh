#pragma once
// SM90 GEMM worker: wgmma register-accumulator implementation of the MoK
// minibatch GEMM stage (replaces the SM100 tcgen05/tmem path).
// Design (PORTING.md P1-B v1): consumer warpgroup computes AND drains;
// per-CTA M-half x full-N accumulator in rt_fl; producer TMA loop unchanged.
#if defined(KITTENS_SM90)
namespace mok_sm90 {
using namespace kittens;

// One K-pipelined BF16 ABt accumulation over the existing smem ring.
// Caller (consumer warpgroup): waits gemm_inputs_arrived[ring] per stage,
// then calls step(); arrives gemm_inputs_finished[ring] after.
template<typename AST, typename BST, int MB, int NB>
struct wgmma_acc {
    // TK SM90 warpgroup mma is fixed at M=64 per call (A height 4 tiles,
    // D height 1 tile/warp): loop the MB rows in 64-row chunks.
    static constexpr int MCH = MB / 64;
    using a_sub_t = st_bf<64, AST::cols>;
    rt_fl<16, NB / 2> acc[MCH];
    __device__ inline void step(const AST &a, const BST &b, bool first) {
        #pragma unroll
        for (int m = 0; m < MCH; ++m) {
            // NOTE: wgmma smem descriptors ignore st_subtile offsets (verified
            // by the unit test: both M-chunks read chunk 0). Reinterpret the
            // M-halves as independent stacked tiles instead.
            using a_half_t = st_bf<64, AST::cols>;
            auto &a_sub = *reinterpret_cast<a_half_t *>(
                reinterpret_cast<char *>(const_cast<AST *>(&a)) + m * sizeof(a_half_t));
            if constexpr (BST::rows == AST::cols) { // B [K,N] -> AB
                if (first) warpgroup::mm_AB (acc[m], a_sub, b);
                else       warpgroup::mma_AB(acc[m], a_sub, b);
            } else {                                // B [N,K] -> ABt
                if (first) warpgroup::mm_ABt (acc[m], a_sub, b);
                else       warpgroup::mma_ABt(acc[m], a_sub, b);
            }
        }
        warpgroup::mma_async_wait();
    }
    template<typename DST> __device__ inline void drain_to(DST &d_smem) {
        #pragma unroll
        for (int m = 0; m < MCH; ++m)
            { auto d_sub = d_smem.template subtile<64, DST::cols>(int2{m, 0});
              warpgroup::store(d_sub, acc[m]); }
        warpgroup::sync(1);
    }
};
} // namespace mok_sm90
#endif
