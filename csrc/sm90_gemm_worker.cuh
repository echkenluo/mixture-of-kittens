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
    rt_fl<MB / 4 / 16, NB> acc; // per-warp rows in TK tile units
    __device__ inline void zero_() { warp::zero(acc); }
    __device__ inline void step_ABt(const AST &a, const BST &b, bool first) {
        if (first) warpgroup::mm_ABt (acc, a, b);
        else       warpgroup::mma_ABt(acc, a, b);
        warpgroup::mma_async_wait();
    }
    template<typename DST> __device__ inline void drain_to(DST &d_smem) {
        warpgroup::store(d_smem, acc); // bf16 convert via TK store
        warpgroup::sync(1);
    }
};
} // namespace mok_sm90
#endif
