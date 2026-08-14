#pragma once
// SM90 quadrant GEMM worker: task tile (2*AH) x (2*BN) covered by 2x2
// quadrant accumulators over A halves (AH x K) and K-major B halves (K x BN).
// Full coverage per codex review; register budget 4 x rt_fl<AH/4, BN>.
#if defined(KITTENS_SM90)
namespace mok_sm90 {
using namespace kittens;

template<typename AST, typename BST, bool IS_AB>
struct wgmma_quad {
    // Layout is EXPLICIT (codex finding 2): after the quarter config both AB
    // and ABt B tiles are 64x64, so shape-based dispatch is impossible.
    // WGMMA descriptors use tile.data[0] and do not apply st_subtile offsets.
    // Reject view types at instantiation so the earlier silent wrong-address
    // failure cannot be reintroduced by a new call site.
    static_assert(AST::rows == AST::underlying_rows && AST::cols == AST::underlying_cols,
                  "wgmma_quad A must be a real shared tile, not st_subtile");
    static_assert(BST::rows == BST::underlying_rows && BST::cols == BST::underlying_cols,
                  "wgmma_quad B must be a real shared tile, not st_subtile");
    static_assert(std::is_same_v<typename AST::dtype, bf16>
                  && std::is_same_v<typename BST::dtype, bf16>,
                  "SM90 v1 wgmma_quad is BF16-only");
    static_assert(AST::swizzle && BST::swizzle
                  && AST::swizzle_bytes == 128 && BST::swizzle_bytes == 128,
                  "wgmma_quad requires real 128-byte-swizzled operands");
    static constexpr int AH = AST::rows;
    static constexpr int BN = IS_AB ? BST::cols : BST::rows;
    static_assert(AH == 64 && BN == 64,
                  "SM90 v1 quadrant ownership is frozen at 64x64");
    rt_fl<AH / 4, BN> acc[2][2];
    __device__ inline void step(const AST &a0, const AST &a1,
                                const BST &b0, const BST &b1, bool first) {
        const AST *as[2] = {&a0, &a1};
        const BST *bs[2] = {&b0, &b1};
        #pragma unroll
        for (int hm = 0; hm < 2; ++hm) {
            #pragma unroll
            for (int hn = 0; hn < 2; ++hn) {
                if constexpr (IS_AB) {
                    if (first) warpgroup::mm_AB (acc[hm][hn], *as[hm], *bs[hn]);
                    else       warpgroup::mma_AB(acc[hm][hn], *as[hm], *bs[hn]);
                } else {
                    if (first) warpgroup::mm_ABt (acc[hm][hn], *as[hm], *bs[hn]);
                    else       warpgroup::mma_ABt(acc[hm][hn], *as[hm], *bs[hn]);
                }
            }
        }
        warpgroup::mma_async_wait();
    }
    // Store a single quadrant into a 64x64 staging tile (aliased GEMM input
    // slot; kernel-wide static smem is impossible in the fused kernel).
    template<typename DST> __device__ inline void drain_quadrant_to(DST &d, int hm, int hn) {
        warpgroup::store(d, acc[hm][hn]);
        warpgroup::sync(1);
    }

    // Store ONE M-half (both N-quadrants) into a half-tile staging buffer
    // (AH x 2*BN). 16KiB budget form per codex step-B smem review.
    template<typename DST> __device__ inline void drain_half_to(DST &d, int hm) {
        #pragma unroll
        for (int hn = 0; hn < 2; ++hn) {
            auto sub = d.template subtile<AH, BN>(int2{0, hn});
            warpgroup::store(sub, acc[hm][hn]);
        }
        warpgroup::sync(1);
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

// Keep the WGMMA type itself out of unsupported template instances. Merely
// putting its calls behind `if constexpr` is insufficient if an unconditional
// local declaration has already instantiated wgmma_quad with MXFP8 operands.
template<bool ENABLED, typename AST, typename BST, bool IS_AB>
struct wgmma_quad_slot {};

template<typename AST, typename BST, bool IS_AB>
struct wgmma_quad_slot<true, AST, BST, IS_AB> {
    wgmma_quad<AST, BST, IS_AB> value;
};
} // namespace mok_sm90
#endif
