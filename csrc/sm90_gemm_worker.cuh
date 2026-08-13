#pragma once
// SM90 quadrant GEMM worker: task tile (2*AH) x (2*BN) covered by 2x2
// quadrant accumulators over A halves (AH x K) and K-major B halves (K x BN).
// Full coverage per codex review; register budget 4 x rt_fl<AH/4, BN>.
#if defined(KITTENS_SM90)
namespace mok_sm90 {
using namespace kittens;

template<typename AST, typename BST>
struct wgmma_quad {
    static constexpr int AH = AST::rows;
    static constexpr int BN = BST::cols;
    rt_fl<AH / 4, BN> acc[2][2];
    __device__ inline void step(const AST &a0, const AST &a1,
                                const BST &b0, const BST &b1, bool first) {
        const AST *as[2] = {&a0, &a1};
        const BST *bs[2] = {&b0, &b1};
        #pragma unroll
        for (int hm = 0; hm < 2; ++hm) {
            #pragma unroll
            for (int hn = 0; hn < 2; ++hn) {
                if (first) warpgroup::mm_AB (acc[hm][hn], *as[hm], *bs[hn]);
                else       warpgroup::mma_AB(acc[hm][hn], *as[hm], *bs[hn]);
            }
        }
        warpgroup::mma_async_wait();
    }
    // d covers the full task tile (2*AH x 2*BN); store path honors subtiles.
    template<typename DST> __device__ inline void drain_to(DST &d) {
        #pragma unroll
        for (int hm = 0; hm < 2; ++hm)
            #pragma unroll
            for (int hn = 0; hn < 2; ++hn) {
                auto sub = d.template subtile<AH, BN>(int2{hm, hn});
                warpgroup::store(sub, acc[hm][hn]);
            }
        warpgroup::sync(1);
    }
};
} // namespace mok_sm90
#endif
